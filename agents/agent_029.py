import numpy as np
from collections import deque

MOMENTUM_WINDOW = 20        # bars of lookback for momentum signal
REBALANCE_PERIOD = 5        # rebalance every N bars
TOP_K = 10                  # number of names to hold
MIN_PRICE = 5.0             # ignore very low-priced names
ALLOCATION_FRACTION = 0.95  # fraction of portfolio value to deploy (rest stays cash buffer)

_state = {}


def init(cash):
    global _state
    _state = {
        'cash': float(cash),
        'positions': {},                       # ticker -> shares held (int)
        'pending_orders': [],                  # orders submitted last bar, to settle this bar
        'bar_count': 0,
        'last_len': 0,                         # number of rows of `history` already processed
        'close_history': deque(maxlen=MOMENTUM_WINDOW + 1),  # rolling snapshots: dict ticker->close
    }


def _fill_pending(open_price):
    """Replicate the platform's execution rules using this bar's open prices."""
    global _state
    cash = _state['cash']
    positions = _state['positions']

    for ticker, direction, size in _state['pending_orders']:
        if size <= 0:
            continue
        price = open_price.get(ticker)
        if price is None or price <= 0 or not np.isfinite(price):
            continue  # no price available -> zero fill

        if direction == 'BUY':
            affordable = int(cash // price)
            filled = min(size, affordable)
            if filled > 0:
                cash -= filled * price
                positions[ticker] = positions.get(ticker, 0) + filled

        elif direction == 'SELL':
            held = positions.get(ticker, 0)
            filled = min(size, held)
            if filled > 0:
                cash += filled * price
                remaining = held - filled
                if remaining > 0:
                    positions[ticker] = remaining
                else:
                    positions.pop(ticker, None)

    _state['cash'] = cash


def _generate_orders(close_price):
    global _state
    hist = _state['close_history']
    if len(hist) < MOMENTUM_WINDOW + 1:
        return []  # not enough history yet to compute momentum

    # tickers with a full, uninterrupted window of closes
    common = None
    for snap in hist:
        keys = set(snap.keys())
        common = keys if common is None else (common & keys)
    if not common:
        return []

    first_snap = hist[0]
    last_snap = hist[-1]

    momentum = {}
    for t in common:
        p0 = first_snap.get(t)
        p1 = last_snap.get(t)
        if p0 is None or p1 is None or p0 <= 0:
            continue
        if p1 < MIN_PRICE:
            continue
        momentum[t] = (p1 / p0) - 1.0

    if not momentum:
        return []

    ranked = sorted(momentum.items(), key=lambda kv: kv[1], reverse=True)
    K = min(TOP_K, len(ranked))
    if K == 0:
        return []
    target_tickers = {t for t, _ in ranked[:K]}

    positions = _state['positions']
    cash = _state['cash']

    port_value = cash
    for t, shares in positions.items():
        price = close_price.get(t, last_snap.get(t))
        if price:
            port_value += shares * price

    if port_value <= 0:
        return []

    target_value_per_ticker = port_value * ALLOCATION_FRACTION / K

    sell_orders = []
    buy_orders = []

    # liquidate names that fell out of the target set
    for t, shares in list(positions.items()):
        if t not in target_tickers and shares > 0:
            sell_orders.append((t, 'SELL', shares))

    # trim oversized target positions
    for t in target_tickers:
        price = close_price.get(t, last_snap.get(t))
        if not price or price <= 0:
            continue
        target_shares = int(target_value_per_ticker // price)
        current_shares = positions.get(t, 0)
        if current_shares > target_shares:
            diff = current_shares - target_shares
            if diff > 0:
                sell_orders.append((t, 'SELL', diff))

    # top up undersized target positions
    for t in target_tickers:
        price = close_price.get(t, last_snap.get(t))
        if not price or price <= 0:
            continue
        target_shares = int(target_value_per_ticker // price)
        current_shares = positions.get(t, 0)
        if target_shares > current_shares:
            diff = target_shares - current_shares
            if diff > 0:
                buy_orders.append((t, 'BUY', diff))

    return sell_orders + buy_orders  # sells first so freed cash can fund buys


def on_bar(history):
    global _state

    new_rows = history.iloc[_state['last_len']:]
    _state['last_len'] = len(history)

    if len(new_rows) == 0:
        return []

    open_price = dict(zip(new_rows['ticker'], new_rows['open']))
    close_price = dict(zip(new_rows['ticker'], new_rows['close']))

    # 1. settle orders placed on the previous bar, using this bar's open
    _fill_pending(open_price)

    # 2. update rolling momentum window with this bar's closes
    _state['close_history'].append(close_price)

    _state['bar_count'] += 1

    # 3. periodically rebalance
    orders = []
    if _state['bar_count'] % REBALANCE_PERIOD == 0:
        orders = _generate_orders(close_price)

    _state['pending_orders'] = orders
    return orders
