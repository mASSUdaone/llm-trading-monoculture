from collections import defaultdict, deque
import numpy as np
import pandas as pd

# Global state variables
INITIAL_CASH = 100000.0
current_cash = INITIAL_CASH
current_positions = {}  # ticker -> int (share count)
price_history = defaultdict(lambda: deque(maxlen=60))
volume_history = defaultdict(lambda: deque(maxlen=60))
latest_prices = {}
bar_count = 0

# Strategy Parameters
LOOKBACK = 30
MOM_PERIOD = 20
REBALANCE_FREQ = 5
TOP_K = 8
TARGET_EQUITY_UTILIZATION = 0.92  # Reserve 8% cash cushion for open gap slippage


def init(cash: float) -> None:
  """Called exactly once before the first bar by the platform.

  Initializes account state and data structures.
  """
  global current_cash, current_positions, price_history, volume_history, latest_prices, bar_count

  current_cash = float(cash)
  current_positions = {}
  price_history = defaultdict(lambda: deque(maxlen=60))
  volume_history = defaultdict(lambda: deque(maxlen=60))
  latest_prices = {}
  bar_count = 0


def on_bar(history: pd.DataFrame) -> list[tuple[str, str, int]]:
  """Called once per timestep with the cumulative append-only bar history.

  Returns a list of orders formatted as (ticker, "BUY"|"SELL", quantity).
  """
  global current_cash, current_positions, price_history, volume_history, latest_prices, bar_count

  bar_count += 1

  if history.empty:
    return []

  # Fast slice of the current bar using binary search on monotonic timestamps
  timestamps = history["timestamp"].values
  current_ts = timestamps[-1]
  idx = np.searchsorted(timestamps, current_ts, side="left")
  current_bar = history.iloc[idx:]

  # Update local rolling price and volume histories
  tickers_in_bar = set()
  for row in current_bar.itertuples(index=False):
    t = row.ticker
    c = float(row.close)
    v = float(row.volume)
    price_history[t].append(c)
    volume_history[t].append(v)
    latest_prices[t] = c
    tickers_in_bar.add(t)

  # Warmup period and rebalance schedule check
  if bar_count < LOOKBACK or (bar_count % REBALANCE_FREQ != 0):
    return []

  # Calculate momentum factors and volatility metrics
  scores = {}
  vols = {}

  for t, prices in price_history.items():
    if t not in tickers_in_bar or len(prices) < LOOKBACK:
      continue

    p_arr = np.array(prices)
    c_current = p_arr[-1]
    c_past = p_arr[-MOM_PERIOD]

    # 20-bar return
    ret_20 = (c_current - c_past) / c_past if c_past > 0 else 0.0

    # 20-bar volatility of returns
    returns = np.diff(p_arr[-MOM_PERIOD - 1 :]) / p_arr[-MOM_PERIOD - 1 : -1]
    vol = np.std(returns) if len(returns) > 1 else 0.0
    if vol < 1e-6:
      vol = 1e-6

    # Trend filter: Asset must trade above its SMA
    sma_lookback = np.mean(p_arr[-LOOKBACK:])
    if c_current <= sma_lookback:
      continue

    # Risk-adjusted momentum score
    score = ret_20 / vol
    if score > 0:
      scores[t] = score
      vols[t] = vol

  # Select Top K assets by signal score
  sorted_tickers = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
  selected_tickers = sorted_tickers[:TOP_K]

  # Calculate current estimated mark-to-market portfolio value
  total_equity = current_cash
  for t, shares in current_positions.items():
    if shares > 0 and t in latest_prices:
      total_equity += shares * latest_prices[t]

  orders = []

  # If no assets pass criteria, liquidate active positions safely
  if not selected_tickers:
    for t, shares in list(current_positions.items()):
      if shares > 0 and t in tickers_in_bar:
        sell_qty = int(shares)
        orders.append((t, "SELL", sell_qty))
        current_cash += sell_qty * latest_prices[t]
        current_positions[t] = 0
    return orders

  # Inverse-volatility portfolio weighting
  inv_vols = {t: 1.0 / vols[t] for t in selected_tickers}
  sum_inv_vols = sum(inv_vols.values())
  target_weights = {t: inv_vols[t] / sum_inv_vols for t in selected_tickers}

  target_equity_alloc = total_equity * TARGET_EQUITY_UTILIZATION

  target_shares = {}
  for t in selected_tickers:
    target_val = target_equity_alloc * target_weights[t]
    target_shares[t] = int(target_val / latest_prices[t])

  # PHASE 1: PROCESS SELLS FIRST (Frees up cash balance)
  # Liquidate assets no longer in selected set
  for t, shares in list(current_positions.items()):
    if shares > 0 and t not in selected_tickers and t in tickers_in_bar:
      sell_qty = int(shares)
      orders.append((t, "SELL", sell_qty))
      current_cash += sell_qty * latest_prices[t]
      current_positions[t] = 0

  # Trim positions that exceed target size
  for t in selected_tickers:
    if t in tickers_in_bar:
      curr_sh = current_positions.get(t, 0)
      targ_sh = target_shares.get(t, 0)
      if curr_sh > targ_sh:
        sell_qty = int(curr_sh - targ_sh)
        if sell_qty > 0:
          orders.append((t, "SELL", sell_qty))
          current_cash += sell_qty * latest_prices[t]
          current_positions[t] = curr_sh - sell_qty

  # PHASE 2: PROCESS BUYS WITH AVAILABLE CASH
  for t in selected_tickers:
    if t in tickers_in_bar:
      curr_sh = current_positions.get(t, 0)
      targ_sh = target_shares.get(t, 0)
      if targ_sh > curr_sh:
        desired_buy = int(targ_sh - curr_sh)
        price = latest_prices[t]

        # Enforce strict cash cap per buy
        max_buy = int((current_cash * 0.95) / price) if price > 0 else 0
        actual_buy = min(desired_buy, max_buy)

        if actual_buy > 0:
          orders.append((t, "BUY", actual_buy))
          current_cash -= actual_buy * price
          current_positions[t] = curr_sh + actual_buy

  return orders
