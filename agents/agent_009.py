import math
from collections import deque

import pandas as pd

# ----------------------------- parameters -----------------------------
LOOKBACK = 20            # bars used to compute momentum score
REBALANCE_PERIOD = 5     # rebalance every N bars
TOP_K = 10               # max number of names held at once
INVEST_FRACTION = 0.95   # fraction of equity to deploy, rest stays cash
MIN_SCORE = 0.0          # only take tickers with positive risk-adj momentum
MIN_RET_SAMPLES = 5      # min number of return observations required
TAIL_ROWS_DEFAULT = 5000 # generous bound on rows scanned per call

# ----------------------------- module state -----------------------------
_state = {}


def init(cash):
    """Called exactly once, before the first bar."""
    _state.clear()
    _state['cash'] = float(cash)
    _state['positions'] = {}          # ticker -> shares held (int)
    _state['pending_orders'] = []     # orders issued last bar, to be settled now
    _state['price_hist'] = {}         # ticker -> deque of recent closes
    _state['last_price'] = {}         # ticker -> most recent known close
    _state['step'] = 0
    _state['prev_block_size'] = 50    # rough size of one timestamp's block of rows


def _apply_pending(open_map):
    """Settle orders issued on the previous bar using this bar's open prices,
    replicating the platform's partial-fill rules exactly."""
    cash = _state['cash']
    positions = _state['positions']

    for ticker, direction, size in _state['pending_orders']:
        price = open_map.get(ticker)
        if price is None or pd.isna(price) or price <= 0 or size <= 0:
            continue  # no price available -> fills for zero shares

        if direction == 'BUY':
            affordable = int(cash // price)
            qty = min(size, affordable)
            if qty > 0:
                cash -= qty * price
                positions[ticker] = positions.get(ticker, 0) + qty

        elif direction == 'SELL':
            held = positions.get(ticker, 0)
            qty = min(size, held)
            if qty > 0:
                cash += qty * price
                remaining = held - qty
                if remaining <= 0:
                    positions.pop(ticker, None)
                else:
                    positions[ticker] = remaining

    _state['cash'] = cash
    _state['pending_orders'] = []


def on_bar(history):
    """Called once per bar, in chronological order."""

    # --- get just the rows for the current (latest) timestamp, cheaply ---
    tail_n = max(TAIL_ROWS_DEFAULT, _state['prev_block_size'] * 4)
    sub = history.tail(tail_n)
    cur_ts = sub['timestamp'].iloc[-1]
    cur_rows = sub[sub['timestamp'] == cur_ts]
    _state['prev_block_size'] = max(len(cur_rows), 1)

    open_map = dict(zip(cur_rows['ticker'], cur_rows['open']))
    close_map = dict(zip(cur_rows['ticker'], cur_rows['close']))

    # 1) settle orders placed last bar at this bar's open
    _apply_pending(open_map)

    # 2) update rolling price history / last known price
    price_hist = _state['price_hist']
    last_price = _state['last_price']
    for ticker, close in close_map.items():
        if close is None or pd.isna(close) or close <= 0:
            continue
        dq = price_hist.get(ticker)
        if dq is None:
            dq = deque(maxlen=LOOKBACK + 1)
            price_hist[ticker] = dq
        dq.append(float(close))
        last_price[ticker] = float(close)

    _state['step'] += 1
    step = _state['step']

    sell_orders = []
    buy_orders = []

    # 3) periodic rebalance
    if step % REBALANCE_PERIOD == 0 and step >= LOOKBACK + 1:
        scores = {}
        for ticker, dq in price_hist.items():
            if len(dq) < LOOKBACK + 1:
                continue
            arr = list(dq)
            rets = [arr[i + 1] / arr[i] - 1.0 for i in range(len(arr) - 1) if arr[i] > 0]
            if len(rets) < MIN_RET_SAMPLES:
                continue
            mean_r = sum(rets) / len(rets)
            var = sum((r - mean_r) ** 2 for r in rets) / len(rets)
            std = math.sqrt(var)
            if std <= 1e-9:
                continue
            score = mean_r / std
            if score > MIN_SCORE:
                scores[ticker] = score

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:TOP_K]
        selected = set(t for t, _ in ranked)

        positions = _state['positions']
        cash = _state['cash']

        # mark-to-market total equity estimate
        total_value = cash
        for t, sh in positions.items():
            p = last_price.get(t)
            if p:
                total_value += sh * p

        k = len(selected)
        target_value_each = (total_value * INVEST_FRACTION / k) if k > 0 else 0.0

        # liquidate anything no longer selected
        for t in list(positions.keys()):
            if t not in selected:
                sh = positions[t]
                if sh > 0:
                    sell_orders.append((t, 'SELL', sh))

        # move remaining/new selections toward equal-weight target
        for t in selected:
            price = last_price.get(t)
            if not price or price <= 0:
                continue
            target_shares = int(target_value_each // price)
            cur_shares = positions.get(t, 0)
            diff = target_shares - cur_shares
            if diff > 0:
                buy_orders.append((t, 'BUY', diff))
            elif diff < 0:
                sell_orders.append((t, 'SELL', -diff))

    # sells first so freed-up cash (per the platform's sequential fill logic)
    # is available for the buys issued in the same batch
    orders = sell_orders + buy_orders

    _state['pending_orders'] = orders
    return orders
