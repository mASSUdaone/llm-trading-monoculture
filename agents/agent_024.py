import numpy as np
import pandas as pd

state = {}


def init(cash):
    global state
    state = {
        "cash": float(cash),
        "positions": {},          # ticker -> shares held (int)
        "pending_orders": [],     # orders to be filled at *this* bar's open
        "bar_count": 0,

        # --- strategy knobs ---
        "lookback": 63,           # bars of momentum lookback
        "rebalance_every": 21,    # bars between rebalances
        "top_k": 10,              # number of names to hold
        "max_weight": 0.15,       # cap per-position weight
        "cash_buffer": 0.03,      # fraction of equity kept in cash
        "min_price": 5.0,         # avoid penny stocks
        "min_dollar_vol": 250_000.0,  # avoid illiquid names
    }


def fill_orders(price_map):
    """Simulate the platform's own fill mechanics to keep our ledger in sync."""
    global state
    for ticker, direction, size in state["pending_orders"]:
        if size <= 0:
            continue
        price = price_map.get(ticker)
        if price is None or not np.isfinite(price) or price <= 0:
            continue

        if direction == "BUY":
            affordable = int(state["cash"] // price)
            qty = min(size, affordable)
            if qty > 0:
                state["cash"] -= qty * price
                state["positions"][ticker] = state["positions"].get(ticker, 0) + qty

        elif direction == "SELL":
            held = state["positions"].get(ticker, 0)
            qty = min(size, held)
            if qty > 0:
                state["cash"] += qty * price
                remaining = held - qty
                if remaining > 0:
                    state["positions"][ticker] = remaining
                else:
                    state["positions"].pop(ticker, None)


def liquidate_all():
    orders = []
    for t, sh in state["positions"].items():
        if sh > 0:
            orders.append((t, "SELL", sh))
    return orders


def generate_orders(history):
    cfg = state
    lookback = cfg["lookback"]
    top_k = cfg["top_k"]
    max_weight = cfg["max_weight"]

    all_ts = history["timestamp"].unique()
    if len(all_ts) < lookback + 1:
        return []

    window_ts = all_ts[-(lookback + 1):]
    window = history[history["timestamp"].isin(window_ts)]

    close_pivot = window.pivot_table(index="timestamp", columns="ticker", values="close")
    close_pivot = close_pivot.sort_index()
    vol_pivot = window.pivot_table(index="timestamp", columns="ticker", values="volume")
    vol_pivot = vol_pivot.reindex(index=close_pivot.index, columns=close_pivot.columns)

    filled = close_pivot.ffill()

    if filled.shape[0] < lookback + 1:
        return []

    start_valid = filled.iloc[0].notna()
    end_valid = filled.iloc[-1].notna()
    price_ok = filled.iloc[-1].fillna(0) >= cfg["min_price"]

    dollar_vol = (filled * vol_pivot).mean()
    liquid_ok = dollar_vol.fillna(0) >= cfg["min_dollar_vol"]

    mask = start_valid & end_valid & price_ok & liquid_ok
    tickers = filled.columns[mask]

    if len(tickers) == 0:
        return liquidate_all()

    sub = filled[tickers]
    returns = (sub.iloc[-1] / sub.iloc[0] - 1.0).dropna()
    daily_ret = sub.pct_change()
    vol = daily_ret.std()

    ranked = returns.sort_values(ascending=False)
    selected = ranked[ranked > 0].head(top_k)

    if len(selected) == 0:
        return liquidate_all()

    current_prices = sub.iloc[-1]

    sel_vol = vol.reindex(selected.index).replace(0, np.nan)
    if sel_vol.notna().sum() == 0:
        inv_vol = pd.Series(1.0, index=selected.index)
    else:
        med = sel_vol.median()
        sel_vol = sel_vol.fillna(med if np.isfinite(med) and med > 0 else 1.0)
        inv_vol = 1.0 / sel_vol

    inv_vol = inv_vol.replace([np.inf, -np.inf], np.nan).fillna(inv_vol.median())
    weights = inv_vol / inv_vol.sum()
    weights = weights.clip(upper=max_weight)
    total_w = weights.sum()
    if total_w <= 0:
        return liquidate_all()
    weights = weights / total_w

    equity = cfg["cash"]
    for t, sh in cfg["positions"].items():
        p = current_prices.get(t)
        if p is not None and np.isfinite(p):
            equity += sh * p

    investable = max(equity * (1 - cfg["cash_buffer"]), 0.0)

    target_shares = {}
    for t, w in weights.items():
        p = current_prices.get(t)
        if p is None or not np.isfinite(p) or p <= 0:
            continue
        target_shares[t] = int((investable * w) // p)

    orders = []
    held_tickers = set(cfg["positions"].keys())

    # sells first (reduce or exit names not in the new target set)
    for t in held_tickers:
        cur_shares = cfg["positions"].get(t, 0)
        tgt = target_shares.get(t, 0)
        if cur_shares > tgt:
            orders.append((t, "SELL", cur_shares - tgt))

    # then buys, up to newly available cash
    for t, tgt in target_shares.items():
        cur_shares = cfg["positions"].get(t, 0)
        if tgt > cur_shares:
            orders.append((t, "BUY", tgt - cur_shares))

    return orders


def on_bar(history):
    global state

    # 1) settle orders placed on the previous call, using this bar's open
    try:
        ts = history["timestamp"].iloc[-1]
        cur = history[history["timestamp"] == ts]
        price_map = dict(zip(cur["ticker"], cur["open"]))
        fill_orders(price_map)
    except Exception:
        pass

    state["bar_count"] = state.get("bar_count", 0) + 1

    # 2) decide new orders (only on rebalance bars)
    orders = []
    try:
        if state["bar_count"] % state["rebalance_every"] == 0:
            orders = generate_orders(history)
    except Exception:
        orders = []

    state["pending_orders"] = orders
    return orders
