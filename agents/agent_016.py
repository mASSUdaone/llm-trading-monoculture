import math
from collections import deque

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Strategy parameters
# ---------------------------------------------------------------------
LOOKBACK = 20          # bars for momentum calculation
VOL_LOOKBACK = 20      # bars for volatility calculation
REBALANCE_FREQ = 5     # rebalance every N bars
TOP_K = 8              # max number of names held at once
CASH_BUFFER = 0.05     # fraction of portfolio kept in cash
MAX_WEIGHT = 0.25       # max weight per single name
VOL_FLOOR = 1e-4        # avoid div-by-zero / crazy inv-vol weights
HIST_MAXLEN = LOOKBACK + VOL_LOOKBACK + 5

_state = {}


def init(cash):
    global _state
    _state = {
        "cash": float(cash),
        "positions": {},          # ticker -> shares held (int)
        "pending_orders": [],     # orders submitted last call, to settle now
        "close_hist": {},         # ticker -> deque of recent closes
        "last_price": {},         # ticker -> most recently seen close
        "prev_len": 0,            # rows of history already processed
        "bar_count": 0,
    }


def _settle_pending_orders(current_bar_df):
    """Replay the platform's fill rule for orders queued on the previous call,
    using this bar's open prices. Keeps our cash/position ledger in sync."""
    cash = _state["cash"]
    positions = _state["positions"]

    for ticker, direction, size in _state["pending_orders"]:
        if size <= 0:
            continue
        if ticker not in current_bar_df.index:
            continue  # no price available this bar -> fills for zero shares
        price = current_bar_df.loc[ticker, "open"]
        if not np.isfinite(price) or price <= 0:
            continue

        if direction == "BUY":
            max_affordable = int(math.floor(cash / price))
            fill = min(size, max_affordable)
            if fill > 0:
                cash -= fill * price
                positions[ticker] = positions.get(ticker, 0) + fill
        elif direction == "SELL":
            held = positions.get(ticker, 0)
            fill = min(size, held)
            if fill > 0:
                cash += fill * price
                positions[ticker] = held - fill
                if positions[ticker] == 0:
                    del positions[ticker]

    _state["cash"] = cash
    _state["pending_orders"] = []


def _update_histories(current_bar_df):
    close_hist = _state["close_hist"]
    last_price = _state["last_price"]
    for ticker, row in current_bar_df.iterrows():
        c = row["close"]
        if not np.isfinite(c):
            continue
        dq = close_hist.get(ticker)
        if dq is None:
            dq = deque(maxlen=HIST_MAXLEN)
            close_hist[ticker] = dq
        dq.append(c)
        last_price[ticker] = c


def _compute_targets():
    """Return dict ticker -> target weight (fraction of portfolio value)."""
    close_hist = _state["close_hist"]
    candidates = []  # (ticker, momentum, vol)

    for ticker, dq in close_hist.items():
        n = len(dq)
        if n < LOOKBACK + 1 or n < VOL_LOOKBACK + 1:
            continue
        arr = np.asarray(dq, dtype=float)
        momentum = arr[-1] / arr[-LOOKBACK - 1] - 1.0
        if not np.isfinite(momentum) or momentum <= 0:
            continue
        recent = arr[-(VOL_LOOKBACK + 1):]
        rets = recent[1:] / recent[:-1] - 1.0
        vol = np.std(rets)
        if not np.isfinite(vol):
            continue
        vol = max(vol, VOL_FLOOR)
        candidates.append((ticker, momentum, vol))

    if not candidates:
        return {}

    candidates.sort(key=lambda x: x[1], reverse=True)
    selected = candidates[:TOP_K]

    inv_vol = {t: 1.0 / v for (t, m, v) in selected}
    total_inv_vol = sum(inv_vol.values())
    invest_frac = 1.0 - CASH_BUFFER

    weights = {}
    for t in inv_vol:
        w = inv_vol[t] / total_inv_vol * invest_frac
        w = min(w, MAX_WEIGHT)
        weights[t] = w

    return weights


def _portfolio_value():
    value = _state["cash"]
    for t, shares in _state["positions"].items():
        price = _state["last_price"].get(t)
        if price is not None:
            value += shares * price
    return value


def _build_orders(weights):
    positions = _state["positions"]
    last_price = _state["last_price"]
    port_value = _portfolio_value()

    target_shares = {}
    for t, w in weights.items():
        price = last_price.get(t)
        if price is None or price <= 0:
            continue
        target_shares[t] = int(math.floor(w * port_value / price))

    all_tickers = set(target_shares.keys()) | set(positions.keys())

    sells = []
    buys = []
    for t in all_tickers:
        cur = positions.get(t, 0)
        tgt = target_shares.get(t, 0)
        delta = tgt - cur
        if delta < 0:
            size = -delta
            if size > 0:
                sells.append((t, "SELL", size))
        elif delta > 0:
            if t in last_price:  # only buy if we have a price reference
                size = delta
                if size > 0:
                    buys.append((t, "BUY", size))

    return sells + buys


def on_bar(history):
    # --- pull only the newly appended rows for this timestep ---
    prev_len = _state["prev_len"]
    new_rows = history.iloc[prev_len:]
    _state["prev_len"] = len(history)

    if new_rows.empty:
        return []

    current_bar_df = new_rows.drop_duplicates(subset="ticker", keep="last").set_index("ticker")

    # 1. Settle orders queued on the previous call using this bar's opens.
    _settle_pending_orders(current_bar_df)

    # 2. Update rolling price history / last known prices.
    _update_histories(current_bar_df)

    _state["bar_count"] += 1

    # 3. Decide whether to rebalance this bar.
    if _state["bar_count"] < REBALANCE_FREQ or _state["bar_count"] % REBALANCE_FREQ != 0:
        return []

    weights = _compute_targets()
    orders = _build_orders(weights)

    # Queue these orders so we can settle them (and thus track fills) next call.
    _state["pending_orders"] = list(orders)

    return orders
