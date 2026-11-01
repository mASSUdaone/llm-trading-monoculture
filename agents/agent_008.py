import numpy as np
from collections import deque, defaultdict

# ----------------------------- parameters -----------------------------
LOOKBACK          = 20      # bars for momentum calculation
VOL_LOOKBACK      = 20      # bars for volatility calculation
MAX_LOOKBACK      = max(LOOKBACK, VOL_LOOKBACK) + 1
REBALANCE_INTERVAL = 5       # rebalance every N bars
TOP_K             = 10      # max number of concurrent positions
MAX_WEIGHT        = 0.20    # cap per-position weight
CASH_BUFFER       = 0.05    # fraction of portfolio always kept as cash
MIN_TRADE_VALUE   = 50.0    # ignore tiny rebalancing trades

_state = {}


def init(cash):
    global _state
    _state = {
        "cash": float(cash),
        "positions": defaultdict(int),          # ticker -> shares held
        "closes": defaultdict(lambda: deque(maxlen=MAX_LOOKBACK)),  # ticker -> recent closes
        "pending_orders": [],                   # orders submitted last bar, to fill this bar
        "bar_count": 0,
        "prev_len": 0,                          # rows of history already consumed
    }


def _process_fills(new_rows):
    """Simulate execution of last bar's orders at this bar's open prices,
    replicating the platform's partial-fill rules exactly."""
    if not _state["pending_orders"]:
        return
    open_prices = dict(zip(new_rows["ticker"], new_rows["open"]))
    for ticker, direction, size in _state["pending_orders"]:
        price = open_prices.get(ticker)
        if price is None or not np.isfinite(price) or price <= 0:
            continue  # no price available -> fills for zero shares
        if direction == "BUY":
            affordable = int(_state["cash"] // price)
            fill = min(size, affordable)
            if fill > 0:
                _state["cash"] -= fill * price
                _state["positions"][ticker] += fill
        elif direction == "SELL":
            held = _state["positions"].get(ticker, 0)
            fill = min(size, held)
            if fill > 0:
                _state["cash"] += fill * price
                _state["positions"][ticker] -= fill
                if _state["positions"][ticker] <= 0:
                    _state["positions"].pop(ticker, None)
    _state["pending_orders"] = []


def _update_price_cache(new_rows):
    for ticker, close in zip(new_rows["ticker"], new_rows["close"]):
        if close is not None and np.isfinite(close) and close > 0:
            _state["closes"][ticker].append(float(close))


def _momentum_and_vol(vals):
    if len(vals) < LOOKBACK + 1:
        return None, None
    arr = np.asarray(vals, dtype=float)
    mom = arr[-1] / arr[-(LOOKBACK + 1)] - 1.0
    rets = arr[1:] / arr[:-1] - 1.0
    if len(rets) < 5:
        return mom, None
    vol = float(np.std(rets[-VOL_LOOKBACK:]))
    return mom, vol


def _cap_weights(weights, max_w):
    weights = dict(weights)
    for _ in range(20):
        over = {t: w for t, w in weights.items() if w > max_w + 1e-12}
        if not over:
            break
        excess = sum(w - max_w for w in over.values())
        for t in over:
            weights[t] = max_w
        under = {t: w for t, w in weights.items() if w < max_w - 1e-12}
        under_total = sum(under.values())
        if under_total <= 0:
            break
        for t in under:
            weights[t] += excess * (weights[t] / under_total)
    return weights


def _compute_target_shares(current_prices):
    candidates = {}
    vols = {}
    for ticker, vals in _state["closes"].items():
        if ticker not in current_prices:
            continue
        mom, vol = _momentum_and_vol(vals)
        if mom is None or vol is None or vol <= 0:
            continue
        if mom > 0:
            candidates[ticker] = mom
            vols[ticker] = vol

    if not candidates:
        return {}

    ranked = sorted(candidates.items(), key=lambda kv: -kv[1])[:TOP_K]
    selected = [t for t, _ in ranked]

    inv_vol = {t: 1.0 / vols[t] for t in selected}
    total_inv = sum(inv_vol.values())
    weights = {t: v / total_inv for t, v in inv_vol.items()}
    weights = _cap_weights(weights, MAX_WEIGHT)

    portfolio_value = _state["cash"] + sum(
        shares * current_prices.get(t, 0.0) for t, shares in _state["positions"].items()
    )
    investable = portfolio_value * (1.0 - CASH_BUFFER)

    target_shares = {}
    for t, w in weights.items():
        price = current_prices.get(t)
        if price is None or price <= 0:
            continue
        target_shares[t] = int((investable * w) // price)
    return target_shares


def _build_orders(target_shares, current_prices):
    sell_orders = []
    buy_orders = []

    for t, shares in list(_state["positions"].items()):
        if t not in target_shares and shares > 0:
            sell_orders.append((t, "SELL", shares))

    for t, target in target_shares.items():
        current = _state["positions"].get(t, 0)
        diff = target - current
        price = current_prices.get(t, 0.0)
        if diff > 0 and diff * price >= MIN_TRADE_VALUE:
            buy_orders.append((t, "BUY", diff))
        elif diff < 0 and (-diff) * price >= MIN_TRADE_VALUE:
            sell_orders.append((t, "SELL", -diff))

    return sell_orders + buy_orders


def on_bar(history):
    global _state

    new_rows = history.iloc[_state["prev_len"]:]
    _state["prev_len"] = len(history)
    if len(new_rows) == 0:
        return []

    # 1. settle orders placed last bar using this bar's open prices
    _process_fills(new_rows)

    # 2. update our rolling price cache with this bar's closes
    _update_price_cache(new_rows)

    _state["bar_count"] += 1

    orders = []
    if _state["bar_count"] % REBALANCE_INTERVAL == 0:
        current_prices = {t: dq[-1] for t, dq in _state["closes"].items() if len(dq) > 0}
        target_shares = _compute_target_shares(current_prices)
        orders = _build_orders(target_shares, current_prices)

    # store for execution simulation & real execution next bar
    _state["pending_orders"] = orders
    return orders
