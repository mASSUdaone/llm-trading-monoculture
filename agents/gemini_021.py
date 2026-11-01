"""
Multi-Asset Systematic Momentum Agent
Designed for execution on pandas DataFrame history streams.
"""

import numpy as np
import pandas as pd

# Global state variables
my_cash = 100000.0
my_positions = {}       # ticker -> int shares
pending_orders = []     # list of (ticker, direction, size)
peak_prices = {}        # ticker -> float peak price for trailing stops

MAX_POSITIONS = 5
MAX_WEIGHT_PER_ASSET = 0.25
CASH_BUFFER_PCT = 0.05
LOOKBACK_BARS = 70


def init(cash: float):
    """
    Called exactly once before the first bar.
    Initializes portfolio state and cash balance.
    """
    global my_cash, my_positions, pending_orders, peak_prices
    my_cash = float(cash)
    my_positions = {}
    pending_orders = []
    peak_prices = {}


def _process_pending_fills(curr_bar_dict: dict):
    """
    Simulates execution of orders submitted on the previous bar
    at the current bar's OPEN price.
    """
    global my_cash, my_positions, pending_orders

    if not pending_orders:
        return

    for ticker, direction, size in pending_orders:
        if ticker not in curr_bar_dict:
            continue

        open_price = float(curr_bar_dict[ticker]['open'])
        if open_price <= 0.0 or np.isnan(open_price):
            continue

        if direction == "BUY":
            max_affordable = int(my_cash // open_price)
            fill_qty = min(int(size), max_affordable)
            if fill_qty > 0:
                cost = fill_qty * open_price
                my_cash -= cost
                my_positions[ticker] = my_positions.get(ticker, 0) + fill_qty
        elif direction == "SELL":
            curr_qty = my_positions.get(ticker, 0)
            fill_qty = min(int(size), curr_qty)
            if fill_qty > 0:
                revenue = fill_qty * open_price
                my_cash += revenue
                my_positions[ticker] = curr_qty - fill_qty
                if my_positions[ticker] <= 0:
                    my_positions.pop(ticker, None)

    pending_orders = []


def _compute_metrics(df_ticker: pd.DataFrame):
    """
    Computes technical indicators for a single ticker dataframe.
    """
    closes = df_ticker['close'].to_numpy(dtype=float)
    highs = df_ticker['high'].to_numpy(dtype=float)
    lows = df_ticker['low'].to_numpy(dtype=float)
    n = len(closes)

    if n < 20:
        return None

    curr_close = closes[-1]
    if curr_close <= 0.0 or np.isnan(curr_close):
        return None

    # Trend SMA
    sma_period = min(n, 50)
    sma_50 = np.mean(closes[-sma_period:])

    # Return over 20 bars
    ret_20 = (curr_close / closes[-20]) - 1.0 if n >= 20 else 0.0

    # Return Volatility over 20 bars
    if n >= 21:
        pct_changes = np.diff(closes[-21:]) / closes[-21:-1]
        vol_20 = float(np.std(pct_changes))
    else:
        vol_20 = 0.02

    if np.isnan(vol_20) or vol_20 < 1e-4:
        vol_20 = 0.02

    # Average True Range (14 periods)
    if n >= 15:
        tr1 = highs[-14:] - lows[-14:]
        tr2 = np.abs(highs[-14:] - closes[-15:-1])
        tr3 = np.abs(lows[-14:] - closes[-15:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr_14 = float(np.mean(tr))
    else:
        atr_14 = curr_close * 0.02

    mom_score = ret_20 / vol_20

    return {
        'curr_close': curr_close,
        'sma_50': sma_50,
        'ret_20': ret_20,
        'vol_20': vol_20,
        'atr_14': atr_14,
        'mom_score': mom_score
    }


def on_bar(history: pd.DataFrame) -> list:
    """
    Called once per timestep with full history up to the current bar.
    Returns a list of tuples: (ticker, direction, size).
    """
    global my_cash, my_positions, pending_orders, peak_prices

    if history.empty:
        return []

    # Get latest timestamp data
    last_timestamp = history['timestamp'].iloc[-1]
    tail_slice = history.tail(300)
    curr_bar_df = tail_slice[tail_slice['timestamp'] == last_timestamp]

    curr_bar = {}
    for _, row in curr_bar_df.iterrows():
        curr_bar[row['ticker']] = row

    # 1. Process pending orders from previous bar fills at current open
    _process_pending_fills(curr_bar)

    # 2. Update trailing peak prices and compute current portfolio value
    total_val = my_cash
    for ticker, qty in list(my_positions.items()):
        if ticker in curr_bar:
            c_price = float(curr_bar[ticker]['close'])
            total_val += qty * c_price
            peak_prices[ticker] = max(peak_prices.get(ticker, c_price), c_price)
        else:
            # Ticker missing from current bar: retain previous valuation estimate
            total_val += qty * peak_prices.get(ticker, 0.0)

    # 3. Fast signal computation using tail slice
    num_tickers = max(1, curr_bar_df['ticker'].nunique())
    recent_history = history.tail(num_tickers * LOOKBACK_BARS)

    metrics = {}
    grouped = recent_history.groupby('ticker')
    for ticker, df_group in grouped:
        m = _compute_metrics(df_group)
        if m is not None:
            metrics[ticker] = m

    # Filter candidate assets
    candidates = []
    for ticker, m in metrics.items():
        if m['curr_close'] > m['sma_50'] and m['ret_20'] > 0:
            candidates.append((ticker, m['mom_score']))

    candidates.sort(key=lambda x: x[1], reverse=True)
    top_candidates = [t for t, _ in candidates[:MAX_POSITIONS]]

    orders = []

    # 4. Check exit rules for existing positions
    sell_orders = []
    projected_cash = my_cash

    for ticker, qty in list(my_positions.items()):
        if ticker not in curr_bar:
            continue

        m = metrics.get(ticker)
        c_price = float(curr_bar[ticker]['close'])
        peak = peak_prices.get(ticker, c_price)

        should_exit = False

        if m is not None:
            # Trailing stop hit
            if c_price < (peak - 2.5 * m['atr_14']) or c_price < (0.88 * peak):
                should_exit = True
            # Trend reversal
            elif c_price < m['sma_50']:
                should_exit = True

        # Dropped from top candidate list
        if ticker not in top_candidates and len(candidates) >= MAX_POSITIONS:
            should_exit = True

        if should_exit:
            sell_orders.append((ticker, "SELL", int(qty)))
            projected_cash += qty * c_price
            peak_prices.pop(ticker, None)

    # 5. Allocate capital to top candidates (Inverse Volatility Weighting)
    target_weights = {}
    if top_candidates:
        inv_vols = []
        for t in top_candidates:
            v = metrics[t]['vol_20']
            inv_vols.append(1.0 / v)
        total_inv_vol = sum(inv_vols)

        for t, inv_v in zip(top_candidates, inv_vols):
            w = min(MAX_WEIGHT_PER_ASSET, inv_v / total_inv_vol)
            target_weights[t] = w

    allocatable_cash = total_val * (1.0 - CASH_BUFFER_PCT)
    buy_orders = []

    for t in top_candidates:
        if t not in curr_bar:
            continue

        c_price = metrics[t]['curr_close']
        if c_price <= 0:
            continue

        target_dollars = allocatable_cash * target_weights.get(t, 0.0)
        target_shares = int(target_dollars // c_price)
        current_shares = my_positions.get(t, 0)

        # Rebalance check
        diff_shares = target_shares - current_shares
        if diff_shares > 0:
            cost = diff_shares * c_price
            if projected_cash >= cost:
                buy_orders.append((t, "BUY", int(diff_shares)))
                projected_cash -= cost
        elif diff_shares < 0 and t not in [s[0] for s in sell_orders]:
            sell_shares = abs(diff_shares)
            sell_orders.append((t, "SELL", int(sell_shares)))
            projected_cash += sell_shares * c_price

    # Prioritize SELL orders first to release cash
    orders = sell_orders + buy_orders

    # Store submitted orders to simulate fills on the next bar
    pending_orders = list(orders)

    return orders
