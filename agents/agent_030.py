import numpy as np
import pandas as pd

# ----------------------------- Config -----------------------------
REBALANCE_PERIOD   = 5      # rebalance every N bars
LOOKBACK_MOM        = 60    # momentum lookback (bars)
LOOKBACK_TREND      = 100   # trend filter SMA lookback (bars)
LOOKBACK_VOL        = 20    # volatility lookback (bars)
TOP_K               = 10    # number of names to hold
MAX_WEIGHT          = 0.15  # cap per-name weight
CASH_BUFFER         = 0.05  # fraction of equity kept uninvested
MIN_TRADE_VALUE     = 50.0  # skip trades smaller than this $ amount

# ----------------------------- State -----------------------------
_state = {
    "cash": 0.0,
    "positions": {},        # ticker -> int shares
    "pending_orders": [],   # orders submitted last bar, to fill at this bar's open
    "bar_count": 0,
}


def init(cash):
    global _state
    _state = {
        "cash": float(cash),
        "positions": {},
        "pending_orders": [],
        "bar_count": 0,
    }


def _apply_fills(current_bar_df):
    """Simulate execution of pending orders at this bar's open, mirroring
    the platform's fill rules exactly, to keep our internal ledger correct."""
    if not _state["pending_orders"]:
        return

    price_map = dict(zip(current_bar_df["ticker"], current_bar_df["open"]))

    for ticker, direction, size in _state["pending_orders"]:
        price = price_map.get(ticker)
        if price is None or pd.isna(price) or price <= 0:
            continue  # no price available -> zero fill

        if direction == "BUY":
            affordable = int(_state["cash"] // price)
            qty = min(size, affordable)
            if qty > 0:
                _state["cash"] -= qty * price
                _state["positions"][ticker] = _state["positions"].get(ticker, 0) + qty

        elif direction == "SELL":
            held = _state["positions"].get(ticker, 0)
            qty = min(size, held)
            if qty > 0:
                _state["cash"] += qty * price
                remaining = held - qty
                if remaining > 0:
                    _state["positions"][ticker] = remaining
                else:
                    _state["positions"].pop(ticker, None)

    _state["pending_orders"] = []


def _compute_rebalance_orders(history):
    try:
        timestamps = pd.Index(history["timestamp"].unique()).sort_values()
        lookback = max(LOOKBACK_MOM, LOOKBACK_TREND, LOOKBACK_VOL) + 5
        recent_ts = timestamps[-lookback:] if len(timestamps) >= lookback else timestamps
        recent = history[history["timestamp"].isin(recent_ts)]

        pivot_close = recent.pivot_table(index="timestamp", columns="ticker", values="close")
        pivot_close = pivot_close.sort_index()

        if pivot_close.shape[0] < LOOKBACK_TREND:
            return []

        last_prices = pivot_close.iloc[-1]
        sma_trend = pivot_close.tail(LOOKBACK_TREND).mean()
        mom_ret = pivot_close.iloc[-1] / pivot_close.iloc[-LOOKBACK_MOM] - 1.0

        rets = pivot_close.pct_change()
        vol = rets.tail(LOOKBACK_VOL).std()

        eligible = (
            last_prices.notna()
            & sma_trend.notna()
            & (last_prices > sma_trend)
            & vol.notna()
            & (vol > 1e-6)
            & mom_ret.notna()
        )

        candidates = mom_ret[eligible].sort_values(ascending=False)
        top = candidates.head(TOP_K)

        equity = _state["cash"]
        for t, sh in _state["positions"].items():
            p = last_prices.get(t, np.nan)
            if not pd.isna(p):
                equity += sh * p

        target_shares = {}
        if len(top) > 0:
            inv_vol = 1.0 / vol[top.index]
            weights = inv_vol / inv_vol.sum()
            weights = weights.clip(upper=MAX_WEIGHT)
            wsum = weights.sum()
            if wsum > 0:
                weights = weights / wsum
            weights = weights * (1.0 - CASH_BUFFER)

            for t in top.index:
                price = last_prices[t]
                if price > 0:
                    dollar = weights[t] * equity
                    target_shares[t] = int(dollar // price)

        current_positions = dict(_state["positions"])
        all_tickers = set(current_positions.keys()) | set(target_shares.keys())

        sell_orders = []
        buy_orders = []

        for t in all_tickers:
            cur = current_positions.get(t, 0)
            tgt = target_shares.get(t, 0)
            diff = tgt - cur
            if diff == 0:
                continue

            price = last_prices.get(t, np.nan)
            value = abs(diff) * price if not pd.isna(price) else 0.0

            if diff < 0:
                if value >= MIN_TRADE_VALUE or tgt == 0:
                    sell_orders.append((t, "SELL", int(abs(diff))))
            else:
                if value >= MIN_TRADE_VALUE:
                    buy_orders.append((t, "BUY", int(diff)))

        return sell_orders + buy_orders

    except Exception:
        # Never let a computation error crash the whole run.
        return []


def on_bar(history):
    global _state
    _state["bar_count"] += 1

    current_ts = history["timestamp"].iloc[-1]
    current_bar = history[history["timestamp"] == current_ts]

    # 1) Settle orders submitted on the previous call against this bar's open.
    _apply_fills(current_bar)

    # 2) Decide whether to rebalance on this bar.
    new_orders = []
    if (
        _state["bar_count"] % REBALANCE_PERIOD == 0
        and _state["bar_count"] >= LOOKBACK_TREND
    ):
        new_orders = _compute_rebalance_orders(history)

    # 3) Remember these orders so we can settle them next bar.
    _state["pending_orders"] = list(new_orders)

    return new_orders
