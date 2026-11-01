import numpy as np
import pandas as pd
from collections import deque

# ---------------------------------------------------------------------------
# Strategy parameters
# ---------------------------------------------------------------------------
LOOKBACK           = 20     # bars used to compute momentum / volatility
REBALANCE_PERIOD   = 10     # rebalance every N bars
TOP_K              = 10     # number of names held at once
MAX_WEIGHT         = 0.25   # cap on any single position's target weight
INVEST_FRACTION    = 0.95   # fraction of equity we target investing (rest = cash buffer)
MIN_VOL            = 1e-6   # floor to avoid div-by-zero in inverse-vol weights

_state = None  # module-level state, set up in init()


def init(cash):
    global _state
    _state = {
        'cash': float(cash),
        'positions': {},        # ticker -> shares held (int), our own ledger
        'pending_orders': [],   # orders we submitted last bar (to be settled now)
        'price_hist': {},       # ticker -> deque of recent closes (len <= LOOKBACK+1)
        'bar_count': 0,
    }


def on_bar(history):
    global _state
    s = _state

    cur_ts = history['timestamp'].iat[-1]
    cur_bar = history[history['timestamp'] == cur_ts]

    # ---- 1. Settle orders placed on the previous call, using this bar's open ----
    _process_fills(cur_bar)

    # ---- 2. Update rolling close-price history for every ticker present today ----
    for t, c in zip(cur_bar['ticker'].values, cur_bar['close'].values):
        if pd.isna(c):
            continue
        dq = s['price_hist'].get(t)
        if dq is None:
            dq = deque(maxlen=LOOKBACK + 1)
            s['price_hist'][t] = dq
        dq.append(float(c))

    s['bar_count'] += 1

    # ---- 3. Periodically rebalance ----
    orders = []
    if s['bar_count'] % REBALANCE_PERIOD == 0 and s['bar_count'] >= LOOKBACK + 1:
        orders = _compute_rebalance_orders(cur_bar)

    # Remember what we sent so we can settle it next call.
    s['pending_orders'] = orders
    return orders


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _process_fills(cur_bar):
    """Replay the platform's deterministic fill logic against our own ledger."""
    s = _state
    if not s['pending_orders']:
        return

    price_map = dict(zip(cur_bar['ticker'].values, cur_bar['open'].values))

    for ticker, direction, size in s['pending_orders']:
        price = price_map.get(ticker)
        if price is None or pd.isna(price) or price <= 0 or size <= 0:
            continue  # no price available -> zero fill

        if direction == 'BUY':
            affordable = int(s['cash'] // price)
            qty = min(size, affordable)
            if qty > 0:
                s['cash'] -= qty * price
                s['positions'][ticker] = s['positions'].get(ticker, 0) + qty

        elif direction == 'SELL':
            held = s['positions'].get(ticker, 0)
            qty = min(size, held)
            if qty > 0:
                s['cash'] += qty * price
                remaining = held - qty
                if remaining > 0:
                    s['positions'][ticker] = remaining
                else:
                    s['positions'].pop(ticker, None)


def _compute_rebalance_orders(cur_bar):
    s = _state
    price_map = dict(zip(cur_bar['ticker'].values, cur_bar['close'].values))

    momentum = {}
    vol = {}
    for t, dq in s['price_hist'].items():
        if len(dq) < LOOKBACK + 1:
            continue
        p_now = price_map.get(t)
        if p_now is None or pd.isna(p_now) or p_now <= 0:
            continue  # need a tradable current price
        arr = np.array(dq, dtype=float)
        if arr[0] <= 0:
            continue
        mom = arr[-1] / arr[0] - 1.0
        rets = np.diff(arr) / arr[:-1]
        v = float(np.std(rets))
        momentum[t] = mom
        vol[t] = v

    # Rank by momentum, keep only positive-momentum names (long-only strategy).
    candidates = [t for t in momentum if momentum[t] > 0]
    candidates.sort(key=lambda t: momentum[t], reverse=True)
    selected = candidates[:TOP_K]

    # Inverse-volatility weights, capped and renormalized, scaled by INVEST_FRACTION.
    weights = {}
    if selected:
        raw = {t: 1.0 / max(vol[t], MIN_VOL) for t in selected}
        total_raw = sum(raw.values())
        norm = {t: raw[t] / total_raw for t in selected}
        capped = {t: min(w, MAX_WEIGHT) for t, w in norm.items()}
        total_capped = sum(capped.values())
        if total_capped > 0:
            weights = {t: (w / total_capped) * INVEST_FRACTION for t, w in capped.items()}

    # Portfolio value marked at current close prices (best available estimate).
    port_value = s['cash']
    for t, qty in s['positions'].items():
        p = price_map.get(t)
        if p is not None and not pd.isna(p):
            port_value += qty * p

    # Target share counts for selected names.
    target_positions = {}
    for t in selected:
        p = price_map[t]
        target_dollar = port_value * weights.get(t, 0.0)
        target_positions[t] = int(target_dollar // p) if p > 0 else 0

    sell_orders = []
    buy_orders = []

    # Sell down/out of anything held that isn't in the new target set (or is oversized).
    for t, held in list(s['positions'].items()):
        target = target_positions.get(t, 0)
        if target < held:
            diff = held - target
            if diff > 0:
                sell_orders.append((t, 'SELL', int(diff)))

    # Buy up to target for selected names.
    for t, target in target_positions.items():
        held = s['positions'].get(t, 0)
        if target > held:
            diff = target - held
            if diff > 0:
                buy_orders.append((t, 'BUY', int(diff)))

    # Sells first (free up cash), then buys, matching how we settle fills internally.
    return sell_orders + buy_orders
