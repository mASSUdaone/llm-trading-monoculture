"""
Cross-sectional momentum / inverse-vol trading agent with internal book-keeping.

Since the platform never reports fills, cash, or positions back to us, this
module keeps its own ledger and replicates the exchange's fill semantics
(largest affordable/available quantity fills, no shorting/borrowing) so that
sizing decisions stay consistent with reality over time.
"""

import pandas as pd
import numpy as np
from collections import deque

# ----------------------------- parameters -----------------------------
LOOKBACK = 60          # momentum lookback window (bars)
SKIP = 5               # skip most-recent bars (avoid short-term reversal)
VOL_LOOKBACK = 20      # bars used to estimate volatility for weighting
REBAL_PERIOD = 5       # rebalance cadence (bars)
TOP_K = 8              # number of names to hold
TARGET_GROSS = 0.90    # fraction of equity to deploy (keep cash buffer)
PER_NAME_CAP = 0.20    # max weight per single name
STOP_LOSS = 0.15       # trailing stop: sell if price falls this much from peak
MAX_HIST_LEN = 300     # cap on stored per-ticker close history (memory)
MIN_HISTORY = LOOKBACK + SKIP + 1

STATE = {}


def init(cash):
    global STATE
    STATE = {
        "cash": float(cash),
        "positions": {},      # ticker -> shares held (int)
        "closes": {},         # ticker -> deque of recent closes
        "peak": {},           # ticker -> peak price since entry (for stop)
        "pending": [],        # orders submitted last call, to settle now
        "last_len": 0,        # rows of history already processed
        "bar_count": 0,       # number of bars processed
    }


def _settle_pending(opens):
    """Apply pending orders using this bar's open prices, mirroring the
    platform's fill rules exactly, to keep our internal ledger correct."""
    if not STATE["pending"]:
        return
    for (t, d, sz) in STATE["pending"]:
        if sz <= 0:
            continue
        price = opens.get(t)
        if price is None or pd.isna(price) or price <= 0:
            continue
        if d == "BUY":
            affordable = int(STATE["cash"] // price)
            fill = min(sz, affordable)
            if fill > 0:
                STATE["cash"] -= fill * price
                STATE["positions"][t] = STATE["positions"].get(t, 0) + fill
                cur_peak = STATE["peak"].get(t, price)
                STATE["peak"][t] = max(cur_peak, price)
        elif d == "SELL":
            have = STATE["positions"].get(t, 0)
            fill = min(sz, have)
            if fill > 0:
                remaining = have - fill
                STATE["positions"][t] = remaining
                STATE["cash"] += fill * price
                if remaining == 0:
                    STATE["positions"].pop(t, None)
                    STATE["peak"].pop(t, None)
    STATE["pending"] = []


def _update_histories(closes_now):
    for t, c in closes_now.items():
        if c is None or pd.isna(c):
            continue
        dq = STATE["closes"].setdefault(t, deque(maxlen=MAX_HIST_LEN))
        dq.append(float(c))


def _update_peaks(reference_prices):
    for t, shares in list(STATE["positions"].items()):
        if shares <= 0:
            continue
        p = reference_prices.get(t)
        if p is None or pd.isna(p):
            continue
        cur_peak = STATE["peak"].get(t, p)
        STATE["peak"][t] = max(cur_peak, p)


def _momentum_and_vol(closes_deque):
    arr = closes_deque
    n = len(arr)
    if n < MIN_HISTORY:
        return None
    recent = arr[-1 - SKIP]
    base = arr[-1 - SKIP - LOOKBACK]
    if base is None or base <= 0:
        return None
    mom = recent / base - 1.0

    window = list(arr)[-(VOL_LOOKBACK + 1):]
    if len(window) < 3:
        return None
    w = np.array(window, dtype=float)
    rets = np.diff(w) / w[:-1]
    sigma = float(np.std(rets))
    if not np.isfinite(sigma) or sigma <= 1e-6:
        return None
    return mom, sigma


def _generate_rebalance_orders(current_prices):
    equity = STATE["cash"]
    for t, sh in STATE["positions"].items():
        equity += sh * current_prices.get(t, 0.0)

    if equity <= 0:
        return []

    candidates = {}
    for t, dq in STATE["closes"].items():
        if t not in current_prices:
            continue
        res = _momentum_and_vol(dq)
        if res is None:
            continue
        mom, sigma = res
        if mom > 0:
            candidates[t] = (mom, sigma)

    ranked = sorted(candidates.items(), key=lambda kv: -kv[1][0])[:TOP_K]

    weights = {}
    if ranked:
        inv_vol = {t: 1.0 / s for t, (m, s) in ranked}
        total_inv = sum(inv_vol.values())
        raw = {t: v / total_inv for t, v in inv_vol.items()}
        capped = {t: min(v, PER_NAME_CAP) for t, v in raw.items()}
        tw = sum(capped.values())
        if tw > 0:
            for t, v in capped.items():
                weights[t] = v / tw * TARGET_GROSS

    all_relevant = set(weights.keys()) | set(STATE["positions"].keys())

    sells, buys = [], []
    for t in all_relevant:
        price = current_prices.get(t)
        if price is None or pd.isna(price) or price <= 0:
            continue
        cur_shares = STATE["positions"].get(t, 0)
        target_dollars = weights.get(t, 0.0) * equity
        target_shares = int(target_dollars // price)
        diff = target_shares - cur_shares
        if diff > 0:
            buys.append((t, "BUY", diff))
        elif diff < 0:
            sells.append((t, "SELL", -diff))

    return sells + buys  # sell first to free up cash before buying


def _stop_loss_orders(current_prices):
    orders = []
    for t, shares in list(STATE["positions"].items()):
        if shares <= 0:
            continue
        price = current_prices.get(t)
        if price is None or pd.isna(price):
            continue
        peak = STATE["peak"].get(t, price)
        if peak > 0 and price <= peak * (1.0 - STOP_LOSS):
            orders.append((t, "SELL", shares))
    return orders


def on_bar(history):
    global STATE
    try:
        n = len(history)
        if n <= STATE.get("last_len", 0):
            return []

        new_rows = history.iloc[STATE["last_len"]:]
        STATE["last_len"] = n

        grouped = new_rows.groupby("timestamp", sort=True)
        timestamps = list(grouped.groups.keys())
        if not timestamps:
            return []
        last_ts = timestamps[-1]

        for ts, grp in grouped:
            opens = dict(zip(grp["ticker"], grp["open"]))
            closes_now = dict(zip(grp["ticker"], grp["close"]))
            highs_now = dict(zip(grp["ticker"], grp["high"]))

            _settle_pending(opens)
            _update_histories(closes_now)

            ref = {t: highs_now.get(t, closes_now.get(t)) for t in
                   set(list(highs_now.keys()) + list(closes_now.keys()))}
            _update_peaks(ref)

            STATE["bar_count"] += 1

        last_grp = grouped.get_group(last_ts)
        current_prices = {
            t: float(p) for t, p in zip(last_grp["ticker"], last_grp["close"])
            if not pd.isna(p)
        }

        new_orders = []

        stop_orders = _stop_loss_orders(current_prices)
        stop_tickers = {t for t, _, _ in stop_orders}
        new_orders.extend(stop_orders)

        if STATE["bar_count"] % REBAL_PERIOD == 0:
            rebal_orders = _generate_rebalance_orders(current_prices)
            rebal_orders = [o for o in rebal_orders if o[0] not in stop_tickers]
            new_orders.extend(rebal_orders)

        new_orders = [o for o in new_orders if o[2] > 0]

        STATE["pending"] = new_orders
        return new_orders

    except Exception:
        # Never let an unexpected data issue crash the whole run.
        return []
