"""
Cross-sectional risk-adjusted momentum trading agent.

Because the platform gives no fill/account feedback, this module maintains
its own ledger (cash, share positions, average cost basis) and replicates
the platform's fill semantics (orders submitted on bar t fill at bar t+1's
open, partial fills capped by available cash/shares, zero fill if no price
is available) every time a new bar arrives, before deciding new orders.

Strategy:
  - Universe: every ticker seen in the data.
  - Signal: lookback return (skipping the most recent few bars) divided by
    the volatility of returns over the same window -> risk-adjusted momentum.
  - Selection: top-K tickers by score (score must be positive).
  - Sizing: equal dollar allocation across selected names, capped at a max
    weight per name, rest held in cash.
  - Risk control: per-position stop-loss checked every bar, independent of
    the rebalance schedule.
  - Rebalance: recompute targets every `rebalance_period` bars; orders are
    generated every bar as the delta between current and target shares.
"""

import math
from collections import deque

import numpy as np

STATE = {}


def init(cash):
    STATE.clear()
    STATE['cash'] = float(cash)
    STATE['positions'] = {}      # ticker -> shares held (int)
    STATE['avg_cost'] = {}       # ticker -> average cost basis per share
    STATE['pending_orders'] = [] # orders submitted last bar, fill this bar at open
    STATE['targets'] = {}        # ticker -> target share count
    STATE['close_hist'] = {}     # ticker -> deque of recent close prices
    STATE['last_price'] = {}     # ticker -> most recently seen close price

    STATE['prev_len'] = 0        # number of rows of `history` already processed
    STATE['bar_index'] = 0

    # --- strategy parameters ---
    STATE['lookback'] = 60          # bars used for momentum window
    STATE['skip'] = 5               # most recent bars excluded from momentum calc
    STATE['rebalance_period'] = 10  # bars between full rebalances
    STATE['top_k'] = 10             # max number of names held at once
    STATE['invest_frac'] = 0.95     # fraction of equity to keep invested
    STATE['max_weight'] = 0.25      # max fraction of equity in a single name
    STATE['stop_loss'] = 0.15       # fractional drawdown from cost basis to exit
    STATE['min_trade_value'] = 50.0 # skip tiny buy orders below this $ value


def _simulate_fills(open_map):
    """Replicate the platform's fill logic for orders queued last bar."""
    for ticker, direction, size in STATE['pending_orders']:
        price = open_map.get(ticker)
        if price is None or not np.isfinite(price) or price <= 0:
            continue  # no price available this bar -> zero fill

        if direction == "BUY":
            affordable = STATE['cash'] / price
            qty = min(size, math.floor(affordable + 1e-9))
            qty = max(qty, 0)
            if qty > 0:
                cost = qty * price
                STATE['cash'] -= cost
                prev_shares = STATE['positions'].get(ticker, 0)
                prev_cost = STATE['avg_cost'].get(ticker, 0.0)
                new_shares = prev_shares + qty
                new_avg = (prev_cost * prev_shares + cost) / new_shares
                STATE['positions'][ticker] = new_shares
                STATE['avg_cost'][ticker] = new_avg

        elif direction == "SELL":
            held = STATE['positions'].get(ticker, 0)
            qty = min(size, held)
            if qty > 0:
                STATE['cash'] += qty * price
                remaining = held - qty
                STATE['positions'][ticker] = remaining
                if remaining == 0:
                    STATE['positions'].pop(ticker, None)
                    STATE['avg_cost'].pop(ticker, None)


def _compute_equity():
    equity = STATE['cash']
    for t, shares in STATE['positions'].items():
        if shares > 0:
            price = STATE['last_price'].get(t, 0.0)
            equity += shares * price
    return equity


def _check_stop_losses(close_map):
    stop_loss = STATE['stop_loss']
    for t, shares in list(STATE['positions'].items()):
        if shares <= 0:
            continue
        price = close_map.get(t, STATE['last_price'].get(t))
        cost = STATE['avg_cost'].get(t)
        if price is None or cost is None or cost <= 0:
            continue
        if price <= cost * (1.0 - stop_loss):
            STATE['targets'][t] = 0


def _rebalance():
    lookback = STATE['lookback']
    skip = STATE['skip']
    top_k = STATE['top_k']
    needed_len = lookback + skip + 1

    scores = {}
    for t, dq in STATE['close_hist'].items():
        if len(dq) < needed_len:
            continue
        window = np.fromiter(dq, dtype=float)[-needed_len:]
        p_start = window[0]
        p_end = window[-(skip + 1)] if skip > 0 else window[-1]
        if p_start <= 0 or not np.isfinite(p_start) or not np.isfinite(p_end):
            continue
        mom = p_end / p_start - 1.0

        rets = np.diff(window) / window[:-1]
        rets = rets[np.isfinite(rets)]
        if rets.size < 2:
            continue
        vol = rets.std()
        if vol == 0 or not np.isfinite(vol):
            continue

        scores[t] = mom / vol

    new_selected_targets = {}
    if scores:
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        selected = [t for t, s in ranked[:top_k] if s > 0]
        n = len(selected)
        if n > 0:
            equity = _compute_equity()
            per_alloc = equity * STATE['invest_frac'] / n
            per_alloc = min(per_alloc, equity * STATE['max_weight'])
            for t in selected:
                price = STATE['last_price'].get(t)
                if price and price > 0 and np.isfinite(price):
                    new_selected_targets[t] = int(per_alloc // price)

    # Anything previously targeted/held but not re-selected now -> target 0.
    all_t = (set(STATE['targets'].keys())
             | set(STATE['positions'].keys())
             | set(new_selected_targets.keys()))
    new_targets = {}
    for t in all_t:
        new_targets[t] = new_selected_targets.get(t, 0)
    STATE['targets'] = new_targets


def _generate_orders():
    targets = STATE['targets']
    positions = STATE['positions']
    all_tickers = set(targets.keys()) | set(positions.keys())

    sells, buys = [], []
    for t in all_tickers:
        target = targets.get(t, 0)
        current = positions.get(t, 0)
        delta = target - current
        if delta < 0:
            sells.append((t, "SELL", -delta))
        elif delta > 0:
            price = STATE['last_price'].get(t, 0.0)
            if price * delta >= STATE['min_trade_value']:
                buys.append((t, "BUY", delta))

    orders = []
    for t, d, q in sells + buys:
        q = int(q)
        if q > 0:
            orders.append((t, d, q))
    return orders


def on_bar(history):
    n = len(history)
    new_rows = history.iloc[STATE['prev_len']:n]
    STATE['prev_len'] = n

    open_map = {}
    close_map = {}
    lookback = STATE['lookback']
    skip = STATE['skip']
    maxlen = lookback + skip + 2

    for row in new_rows.itertuples(index=False):
        t = row.ticker
        o = row.open
        c = row.close

        if o is not None and np.isfinite(o):
            open_map[t] = float(o)
        if c is not None and np.isfinite(c):
            close_map[t] = float(c)
            dq = STATE['close_hist'].get(t)
            if dq is None:
                dq = deque(maxlen=maxlen)
                STATE['close_hist'][t] = dq
            dq.append(float(c))
            STATE['last_price'][t] = float(c)

    # 1. Settle orders placed on the previous bar using this bar's opens.
    _simulate_fills(open_map)
    STATE['pending_orders'] = []

    STATE['bar_index'] += 1

    # 2. Risk control: force-exit anything past its stop-loss.
    _check_stop_losses(close_map)

    # 3. Periodic rebalance of target weights.
    if STATE['bar_index'] % STATE['rebalance_period'] == 0:
        _rebalance()

    # 4. Translate targets into orders for this bar.
    orders = _generate_orders()
    STATE['pending_orders'] = orders
    return orders
