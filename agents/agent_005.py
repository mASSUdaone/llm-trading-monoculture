import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Strategy parameters
# ---------------------------------------------------------------------------
LOOKBACK          = 60      # bars used for momentum / vol estimation
MIN_COVERAGE      = 0.9     # ticker must have data in >=90% of the window
N_TOP             = 8       # number of names held at once
MIN_MOMENTUM      = 0.0     # only long names with positive trailing return
VOL_FLOOR         = 1e-4    # avoid div-by-zero for very low-vol names
MAX_WEIGHT        = 0.35    # cap on any single name's target weight
INVESTED_FRACTION = 0.95    # leave some cash buffer
REBALANCE_EVERY   = 5       # bars between full rebalances
STOP_LOSS         = 0.15    # sell if price is 15% below our avg cost
TRAIL_STOP        = 0.20    # sell if price is 20% below post-entry peak

STATE = None  # module-level state, set in init()


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------
def init(cash):
    global STATE
    STATE = {
        "cash": float(cash),
        "positions": {},        # ticker -> int shares held
        "avg_cost": {},         # ticker -> average cost basis per share
        "high_water": {},       # ticker -> peak close observed since entry
        "pending_orders": [],   # orders we returned last bar, to fill now
        "bar_count": 0,
        "unique_timestamps": [],
    }


def on_bar(history):
    global STATE
    try:
        return _on_bar_impl(history)
    except Exception:
        # Fail safe: never crash the run.
        return []


# ---------------------------------------------------------------------------
# Core per-bar logic
# ---------------------------------------------------------------------------
def _on_bar_impl(history):
    ts_col = history["timestamp"].values
    current_time = ts_col[-1]

    STATE["bar_count"] += 1
    STATE["unique_timestamps"].append(current_time)

    # Slice of rows belonging to the current bar (all tickers today).
    start_idx_current = int(np.searchsorted(ts_col, current_time, side="left"))
    current_bar_df = history.iloc[start_idx_current:]

    open_map = _price_map(current_bar_df, "open")
    close_map = _price_map(current_bar_df, "close")

    # 1) Execute orders we placed on the previous bar, at THIS bar's open.
    if STATE["pending_orders"]:
        _execute_orders(STATE, STATE["pending_orders"], open_map)
    STATE["pending_orders"] = []

    # 2) Risk management: stop-loss / trailing-stop checks (every bar).
    forced_sells = []
    for ticker, shares in list(STATE["positions"].items()):
        if shares <= 0:
            continue
        price = close_map.get(ticker)
        if price is None:
            continue
        hw = STATE["high_water"].get(ticker, price)
        if price > hw:
            hw = price
        STATE["high_water"][ticker] = hw

        avg_cost = STATE["avg_cost"].get(ticker, price)
        loss_pct = (price - avg_cost) / avg_cost if avg_cost > 0 else 0.0
        dd_from_high = (price - hw) / hw if hw > 0 else 0.0

        if loss_pct <= -STOP_LOSS or dd_from_high <= -TRAIL_STOP:
            forced_sells.append((ticker, "SELL", int(shares)))

    excluded = {t for t, _, _ in forced_sells}

    # 3) Periodic momentum rebalance.
    rebalance_orders = []
    if STATE["bar_count"] % REBALANCE_EVERY == 0:
        rebalance_orders = _compute_rebalance_orders(
            STATE, history, ts_col, close_map, excluded
        )

    new_orders = forced_sells + rebalance_orders
    new_orders = [o for o in new_orders if o[2] > 0]

    STATE["pending_orders"] = new_orders
    return new_orders


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _price_map(df, col):
    sub = df[["ticker", col]].dropna()
    return dict(zip(sub["ticker"], sub[col]))


def _execute_orders(state, orders, price_map):
    """Mirrors the platform's fill rules exactly, so our ledger stays correct."""
    for ticker, direction, size in orders:
        price = price_map.get(ticker)
        if price is None or price <= 0 or size <= 0:
            continue

        if direction == "BUY":
            max_affordable = int(state["cash"] // price)
            qty = min(int(size), max_affordable)
            if qty <= 0:
                continue
            cost = qty * price
            state["cash"] -= cost
            prev_shares = state["positions"].get(ticker, 0)
            prev_cost = state["avg_cost"].get(ticker, price)
            new_shares = prev_shares + qty
            state["avg_cost"][ticker] = (
                (prev_cost * prev_shares + cost) / new_shares if new_shares > 0 else price
            )
            state["positions"][ticker] = new_shares
            state["high_water"][ticker] = max(state["high_water"].get(ticker, price), price)

        elif direction == "SELL":
            held = state["positions"].get(ticker, 0)
            qty = min(int(size), held)
            if qty <= 0:
                continue
            state["cash"] += qty * price
            remaining = held - qty
            if remaining <= 0:
                state["positions"].pop(ticker, None)
                state["avg_cost"].pop(ticker, None)
                state["high_water"].pop(ticker, None)
            else:
                state["positions"][ticker] = remaining


def _compute_signals(window_df, min_count):
    """Momentum (window return) and volatility (std of returns) per ticker."""
    w = window_df.dropna(subset=["close"])
    counts = w.groupby("ticker")["close"].count()
    valid = counts[counts >= min_count].index
    if len(valid) == 0:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    sub = w[w["ticker"].isin(valid)].sort_values(["ticker", "timestamp"])
    g = sub.groupby("ticker")["close"]
    first_close = g.first()
    last_close = g.last()
    momentum = last_close / first_close - 1.0

    returns = sub.groupby("ticker")["close"].pct_change()
    vol = returns.groupby(sub["ticker"]).std()

    return momentum, vol


def _compute_rebalance_orders(state, history, ts_col, close_map, excluded):
    unique_ts = state["unique_timestamps"]
    if len(unique_ts) < LOOKBACK:
        return []

    cutoff_time = unique_ts[-LOOKBACK]
    start_idx = int(np.searchsorted(ts_col, cutoff_time, side="left"))
    window_df = history.iloc[start_idx:]

    min_count = int(LOOKBACK * MIN_COVERAGE)
    momentum, vol = _compute_signals(window_df, min_count)
    if momentum.empty:
        return []

    momentum = momentum[momentum > MIN_MOMENTUM]
    momentum = momentum.drop(labels=[t for t in excluded if t in momentum.index],
                              errors="ignore")

    portfolio_value = state["cash"] + sum(
        shares * close_map.get(t, state["avg_cost"].get(t, 0.0))
        for t, shares in state["positions"].items()
    )

    orders = []

    if momentum.empty:
        # Nothing qualifies as a long -> go to cash.
        for t, shares in state["positions"].items():
            if shares > 0 and t not in excluded:
                orders.append((t, "SELL", int(shares)))
        return orders

    top = momentum.sort_values(ascending=False).head(N_TOP)
    vol_top = vol.reindex(top.index).clip(lower=VOL_FLOOR).fillna(VOL_FLOOR)
    inv_vol = 1.0 / vol_top
    weights = inv_vol / inv_vol.sum()
    weights = weights.clip(upper=MAX_WEIGHT)
    weights = weights / weights.sum()

    target_value = weights * portfolio_value * INVESTED_FRACTION

    # Exit positions that fell out of the target set.
    for t, shares in list(state["positions"].items()):
        if shares <= 0 or t in excluded:
            continue
        if t not in target_value.index:
            orders.append((t, "SELL", int(shares)))

    # Move existing/target positions toward their target dollar value.
    for t, tv in target_value.items():
        price = close_map.get(t)
        if price is None or price <= 0:
            continue
        cur_shares = state["positions"].get(t, 0)
        cur_value = cur_shares * price
        diff = tv - cur_value

        if diff > 0:
            qty = int(diff // price)
            if qty > 0:
                orders.append((t, "BUY", qty))
        elif diff < 0:
            qty = min(int((-diff) // price), cur_shares)
            if qty > 0:
                orders.append((t, "SELL", qty))

    return orders
