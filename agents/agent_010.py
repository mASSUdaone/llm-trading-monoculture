import numpy as np
import pandas as pd

# ----------------------------- Configuration -----------------------------
LOOKBACK = 60          # bars for momentum calculation
VOL_WINDOW = 20        # bars for volatility estimate
REBALANCE_EVERY = 10   # rebalance cadence in bars
TOP_N = 10             # number of names to hold
TARGET_GROSS = 0.95    # fraction of equity to keep invested
STOP_LOSS = 0.15       # per-position stop loss (fraction below entry)
MAX_WEIGHT_MULT = 2.0  # cap a single name's weight at MAX_WEIGHT_MULT / TOP_N

_state = {}


def init(cash):
    _state.clear()
    _state['cash'] = float(cash)
    _state['positions'] = {}        # ticker -> shares held (int)
    _state['entry_price'] = {}      # ticker -> avg cost basis
    _state['pending_orders'] = []   # orders submitted last call, to be filled now
    _state['bar_count'] = 0
    _state['last_len'] = 0          # rows in history already processed
    _state['ts_starts'] = []        # row index where each bar's block started
    _state['last_prices'] = {}      # ticker -> most recent known close


def _apply_fills(open_map):
    """Replay the platform's fill rules locally so our ledger stays accurate."""
    for ticker, direction, size in _state['pending_orders']:
        if size <= 0:
            continue
        price = open_map.get(ticker)
        if price is None or not np.isfinite(price) or price <= 0:
            continue  # no price available -> zero fill

        if direction == 'BUY':
            affordable = int(_state['cash'] // price)
            qty = min(size, affordable)
            if qty > 0:
                cost = qty * price
                _state['cash'] -= cost
                prev_shares = _state['positions'].get(ticker, 0)
                prev_basis = _state['entry_price'].get(ticker, price) * prev_shares
                new_shares = prev_shares + qty
                _state['positions'][ticker] = new_shares
                _state['entry_price'][ticker] = (prev_basis + cost) / new_shares

        elif direction == 'SELL':
            held = _state['positions'].get(ticker, 0)
            qty = min(size, held)
            if qty > 0:
                _state['cash'] += qty * price
                remaining = held - qty
                if remaining <= 0:
                    _state['positions'].pop(ticker, None)
                    _state['entry_price'].pop(ticker, None)
                else:
                    _state['positions'][ticker] = remaining

    _state['pending_orders'] = []


def _compute_rebalance(history, close_map):
    ts_starts = _state['ts_starts']
    if len(ts_starts) < LOOKBACK + 1:
        return []

    window_start = ts_starts[-(LOOKBACK + 1)]
    hist_window = history.iloc[window_start:]

    try:
        pivot = hist_window.pivot_table(index='timestamp', columns='ticker',
                                         values='close', aggfunc='last')
    except Exception:
        return []

    pivot = pivot.sort_index()
    if len(pivot) < 2:
        return []

    first_row = pivot.iloc[0]
    last_row = pivot.iloc[-1]
    valid = first_row.notna() & last_row.notna() & (first_row > 0)
    momentum = (last_row[valid] / first_row[valid] - 1.0)

    candidates = momentum[momentum > 0]
    if candidates.empty:
        # no attractive names -> liquidate everything
        return [(t, 'SELL', s) for t, s in _state['positions'].items() if s > 0]

    ranked = candidates.sort_values(ascending=False)
    selected = list(ranked.index[:TOP_N])

    vol_pivot = pivot[selected].tail(VOL_WINDOW + 1)
    rets = vol_pivot.pct_change().dropna(how='all')
    vol = rets.std()
    vol = vol.replace(0, np.nan).fillna(vol.max() if vol.notna().any() else 1.0)
    vol = vol.fillna(1.0)
    vol = vol.clip(lower=1e-6)

    inv_vol = 1.0 / vol
    total_inv_vol = inv_vol.sum()
    if total_inv_vol <= 0 or not np.isfinite(total_inv_vol):
        w = pd.Series(1.0 / len(selected), index=selected)
    else:
        w = inv_vol / total_inv_vol

    max_weight = min(1.0, MAX_WEIGHT_MULT / max(len(selected), 1))
    w = w.clip(upper=max_weight)
    w = w / w.sum()
    weights = w * TARGET_GROSS

    # portfolio equity mark-to-market using latest known closes
    equity = _state['cash']
    for ticker, shares in _state['positions'].items():
        price = close_map.get(ticker, _state['last_prices'].get(ticker))
        if price:
            equity += shares * price
    if equity <= 0:
        return []

    target_shares = {}
    for ticker in selected:
        price = close_map.get(ticker)
        if not price or price <= 0 or not np.isfinite(price):
            continue
        target_value = float(weights.get(ticker, 0.0)) * equity
        target_shares[ticker] = int(target_value // price)

    current_positions = dict(_state['positions'])
    orders = []

    # sells first (names dropped or overweight) to free up cash
    for ticker, shares in current_positions.items():
        tgt = target_shares.get(ticker, 0)
        if tgt < shares:
            diff = shares - tgt
            if diff > 0:
                orders.append((ticker, 'SELL', int(diff)))

    # then buys
    for ticker, tgt in target_shares.items():
        cur = current_positions.get(ticker, 0)
        if tgt > cur:
            diff = tgt - cur
            if diff > 0:
                orders.append((ticker, 'BUY', int(diff)))

    return orders


def on_bar(history):
    try:
        _state['bar_count'] += 1

        old_len = _state['last_len']
        new_rows = history.iloc[old_len:]
        _state['ts_starts'].append(old_len)
        _state['last_len'] = len(history)

        if new_rows.empty:
            return []

        open_map = dict(zip(new_rows['ticker'], new_rows['open']))
        close_map = dict(zip(new_rows['ticker'], new_rows['close']))
        _state['last_prices'].update(close_map)

        # 1) settle orders placed on the previous call using this bar's opens
        _apply_fills(open_map)

        orders_to_submit = []

        # 2) stop-loss scan every bar
        for ticker, shares in list(_state['positions'].items()):
            if shares <= 0:
                continue
            entry = _state['entry_price'].get(ticker)
            price = close_map.get(ticker, _state['last_prices'].get(ticker))
            if entry and price and price <= entry * (1.0 - STOP_LOSS):
                orders_to_submit.append((ticker, 'SELL', int(shares)))

        # 3) periodic rebalance
        if (_state['bar_count'] % REBALANCE_EVERY == 0
                and len(_state['ts_starts']) >= LOOKBACK + 1):
            # avoid double-selling a name already stopped out this bar
            already_selling = {t for t, d, s in orders_to_submit if d == 'SELL'}
            rebal_orders = _compute_rebalance(history, close_map)
            for t, d, s in rebal_orders:
                if d == 'SELL' and t in already_selling:
                    continue
                orders_to_submit.append((t, d, s))

        orders_to_submit = [(t, d, int(s)) for t, d, s in orders_to_submit if s > 0]
        _state['pending_orders'] = orders_to_submit
        return orders_to_submit

    except Exception:
        # never let an unexpected data issue kill the whole run
        _state['pending_orders'] = []
        return []
