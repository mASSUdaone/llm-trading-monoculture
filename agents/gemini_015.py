from collections import defaultdict, deque
import numpy as np
import pandas as pd

# Global state variables
INITIAL_CASH = 100000.0
cash = INITIAL_CASH
positions = defaultdict(int)  # ticker -> shares
pending_sells = []  # list of (ticker, size)
pending_buys = []  # list of (ticker, size)

# Historical price tracking
HIST_LEN = 60
price_history = defaultdict(lambda: deque(maxlen=HIST_LEN))


def init(starting_cash: float):
    """Called exactly once before the first bar."""
    global cash, positions, pending_sells, pending_buys, price_history
    cash = float(starting_cash)
    positions = defaultdict(int)
    pending_sells = []
    pending_buys = []
    price_history = defaultdict(lambda: deque(maxlen=HIST_LEN))


def _extract_latest_bar(history: pd.DataFrame) -> pd.DataFrame:
    """Fast extraction of rows corresponding to the most recent timestamp."""
    ts_values = history["timestamp"].values
    n = len(ts_values)
    if n == 0:
        return pd.DataFrame()

    last_ts = ts_values[-1]
    idx = n - 1
    while idx >= 0 and ts_values[idx] == last_ts:
        idx -= 1

    return history.iloc[idx + 1 :]


def _update_fills(current_bar: pd.DataFrame):
    """Simulate execution of previous step's orders using current bar's OPEN prices."""
    global cash, positions, pending_sells, pending_buys

    if current_bar.empty:
        return

    # Map ticker to current OPEN price
    open_prices = dict(zip(current_bar["ticker"], current_bar["open"]))

    # 1. Process Sells
    for ticker, qty in pending_sells:
        current_qty = positions.get(ticker, 0)
        fill_qty = min(qty, current_qty)
        if fill_qty > 0:
            positions[ticker] -= fill_qty
            if ticker in open_prices and open_prices[ticker] > 0:
                cash += fill_qty * float(open_prices[ticker])

    # 2. Process Buys
    for ticker, qty in pending_buys:
        if ticker in open_prices and open_prices[ticker] > 0:
            open_price = float(open_prices[ticker])
            max_affordable = int(cash // open_price)
            fill_qty = min(qty, max_affordable)
            if fill_qty > 0:
                positions[ticker] = positions.get(ticker, 0) + fill_qty
                cash -= fill_qty * open_price

    pending_sells.clear()
    pending_buys.clear()


def on_bar(history: pd.DataFrame):
    """Called once per time step with full history up to current bar."""
    global cash, positions, pending_sells, pending_buys, price_history

    if history.empty:
        return []

    # Extract the current timestamp's data
    latest_bar = _extract_latest_bar(history)
    if latest_bar.empty:
        return []

    # Update portfolio state based on execution at current OPEN
    _update_fills(latest_bar)

    # Update price histories with current CLOSE
    current_closes = {}
    for _, row in latest_bar.iterrows():
        t = row["ticker"]
        c = float(row["close"])
        price_history[t].append(c)
        current_closes[t] = c

    # Calculate portfolio equity
    total_equity = cash
    for ticker, shares in positions.items():
        if shares > 0 and ticker in current_closes:
            total_equity += shares * current_closes[ticker]

    # Calculate signals for tickers with sufficient history
    scores = {}
    for ticker, prices in price_history.items():
        if len(prices) >= 25 and ticker in current_closes:
            p = np.array(prices)
            ret_20 = (p[-1] - p[-20]) / p[-20]
            returns = np.diff(p[-20:]) / p[-20:-1]
            vol = np.std(returns) if len(returns) > 1 else 0.02
            vol = max(vol, 0.005)  # floor volatility

            sma_20 = np.mean(p[-20:])

            # Trend and momentum condition
            if p[-1] > sma_20 and ret_20 > 0:
                scores[ticker] = ret_20 / vol

    # Rank top target assets
    TOP_N = 5
    sorted_tickers = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    target_tickers = set(sorted_tickers[:TOP_N])

    # Dynamic asset allocation (18% per position, ~10% cash reserve)
    target_weight = 0.18
    target_positions = {}

    for ticker in target_tickers:
        close_price = current_closes[ticker]
        if close_price > 0:
            target_val = total_equity * target_weight
            target_positions[ticker] = int(target_val // close_price)

    orders = []
    pending_sells = []
    pending_buys = []

    # Step 1: Generate SELL orders for assets to exit or scale down
    for ticker, current_shares in list(positions.items()):
        if current_shares <= 0:
            continue
        desired_shares = target_positions.get(ticker, 0)
        if current_shares > desired_shares:
            sell_qty = current_shares - desired_shares
            orders.append((ticker, "SELL", sell_qty))
            pending_sells.append((ticker, sell_qty))

    # Calculate cash estimated to be available for buys
    estimated_cash = cash
    for ticker, qty in pending_sells:
        if ticker in current_closes:
            estimated_cash += qty * current_closes[ticker]

    # Step 2: Generate BUY orders for target positions
    for ticker, desired_shares in target_positions.items():
        current_shares = positions.get(ticker, 0)
        if desired_shares > current_shares:
            needed_shares = desired_shares - current_shares
            close_price = current_closes[ticker]

            # Buffer estimated open price to prevent over-allocation
            est_buy_price = close_price * 1.02
            max_affordable = int(estimated_cash // est_buy_price)
            buy_qty = min(needed_shares, max_affordable)

            if buy_qty > 0:
                orders.append((ticker, "BUY", buy_qty))
                pending_buys.append((ticker, buy_qty))
                estimated_cash -= buy_qty * est_buy_price

    return orders
