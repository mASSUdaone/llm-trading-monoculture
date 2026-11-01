"""
Momentum / inverse-volatility rotation strategy.

Design summary
---------------
- Every REBALANCE_EVERY bars, rank all tickers with a full price history
  over LOOKBACK bars by trailing return (momentum).
- Keep only tickers with positive momentum; take the top K.
- Weight them inversely to their recent volatility (risk parity style),
  capped at MAX_WEIGHT per name, and only deploy INVEST_FRACTION of
  total equity, leaving the rest in cash as a buffer/safety margin.
- Translate target weights into target share counts using latest close,
  then emit BUY/SELL orders for the delta versus what we believe we
  currently hold.
- Because the platform gives no fill confirmations, we keep an internal
  shadow ledger (cash + positions). Orders we submit on bar t execute at
  bar t+1's open. When bar t+1 arrives (as the newest row in `history`),
  we replay the platform's own fill rule (largest affordable / available
  quantity fills, using the *actual* observed open price) against our
  shadow ledger before deciding anything new. Because we never order
  more than our own ledger says we can afford/hold, this replay should
  stay exactly in sync with the platform's real state.
"""

import numpy as np
import pandas as pd

REBALANCE_EVERY = 5
LOOKBACK = 20
TOP_K = 5
MAX_WEIGHT = 0.35
INVEST_FRACTION = 0.95

_state = {}


def init(cash):
    _state.clear()
    _state["cash"] = float(cash)
    _state["positions"] = {}       # ticker -> shares held (int)
    _state["pending"] = []         # orders submitted last bar, awaiting fill-sim
    _state["last_price"] = {}      # ticker -> most recent known close
    _state["step"] = -1
    _state["last_rebalance_step"] = -10**9


def _resolve_pending_fills(current_bar):
    """Replay the platform's fill rule against our shadow ledger using
    this bar's open prices (these are the actual execution prices for
    orders submitted on the previous bar)."""
    pending = _state["pending"]
    if not pending:
        return
    open_map = dict(zip(current_bar["ticker"], current_bar["open"]))
    positions = _state["positions"]

    for ticker, direction, size in pending:
        if size <= 0:
            continue
        price = open_map.get(ticker)
        if price is None or not np.isfinite(price) or price <= 0:
            continue  # no price available -> zero fill

        if direction == "BUY":
            affordable = int(_state["cash"] // price)
            fill = min(size, affordable)
            if fill > 0:
                _state["cash"] -= fill * price
                positions[ticker] = positions.get(ticker, 0) + fill
        elif direction == "SELL":
            held = positions.get(ticker, 0)
            fill = min(size, held)
            if fill > 0:
                positions[ticker] = held - fill
                if positions[ticker] <= 0:
                    positions.pop(ticker, None)
                _state["cash"] += fill * price

    _state["pending"] = []


def _update_last_prices(current_bar):
    for t, c in zip(current_bar["ticker"], current_bar["close"]):
        if pd.notna(c):
            _state["last_price"][t] = float(c)


def _portfolio_value():
    val = _state["cash"]
    lp = _state["last_price"]
    for t, sh in _state["positions"].items():
        val += sh * lp.get(t, 0.0)
    return val


def _compute_rebalance_orders(history, current_ts, current_bar):
    all_ts = history["timestamp"].drop_duplicates().sort_values()
    if len(all_ts) < LOOKBACK + 1:
        return []
    recent_ts = all_ts.iloc[-(LOOKBACK + 1):]
    recent = history[history["timestamp"].isin(recent_ts)]

    pivot = recent.pivot_table(index="timestamp", columns="ticker", values="close")
    pivot = pivot.sort_index()

    # Only trade names with a complete price history over the lookback window
    valid = pivot.dropna(axis=1, how="any")
    if valid.shape[1] == 0 or valid.shape[0] < LOOKBACK + 1:
        return _liquidate_all()

    momentum = valid.iloc[-1] / valid.iloc[0] - 1.0
    rets = valid.pct_change().dropna(how="all")
    vol = rets.std(ddof=0)

    candidates = momentum[momentum > 0].sort_values(ascending=False)
    selected = list(candidates.index[:TOP_K])

    price_map = dict(zip(current_bar["ticker"], current_bar["close"]))

    if not selected:
        return _liquidate_all()

    inv_vol = {}
    for t in selected:
        v = vol.get(t, np.nan)
        if not np.isfinite(v) or v <= 0:
            v = rets[selected].std(ddof=0).replace(0, np.nan).mean()
            if not np.isfinite(v) or v <= 0:
                v = 1.0
        inv_vol[t] = 1.0 / v

    total_iv = sum(inv_vol.values())
    weights = {t: iv / total_iv for t, iv in inv_vol.items()}

    # Cap and renormalize
    for _ in range(3):
        capped = {t: min(w, MAX_WEIGHT) for t, w in weights.items()}
        s = sum(capped.values())
        if s <= 0:
            break
        weights = {t: w / s for t, w in capped.items()}
        if all(w <= MAX_WEIGHT + 1e-9 for w in weights.values()):
            break

    port_value = _portfolio_value()
    investable = max(port_value, 0.0) * INVEST_FRACTION

    target_shares = {}
    for t in selected:
        price = price_map.get(t)
        if price is None or not np.isfinite(price) or price <= 0:
            continue
        dollars = investable * weights.get(t, 0.0)
        target_shares[t] = int(dollars // price)

    orders = []
    positions = _state["positions"]

    # Liquidate names no longer selected
    for t, held in list(positions.items()):
        if t not in target_shares and held > 0:
            orders.append((t, "SELL", int(held)))

    # Trim positions that are above target
    for t, tgt in target_shares.items():
        held = positions.get(t, 0)
        if tgt < held:
            orders.append((t, "SELL", int(held - tgt)))

    # Grow positions that are below target
    for t, tgt in target_shares.items():
        held = positions.get(t, 0)
        if tgt > held:
            orders.append((t, "BUY", int(tgt - held)))

    orders = [o for o in orders if o[2] > 0]
    return orders


def _liquidate_all():
    orders = []
    for t, held in list(_state["positions"].items()):
        if held > 0:
            orders.append((t, "SELL", int(held)))
    return orders


def on_bar(history):
    _state["step"] += 1
    step = _state["step"]

    current_ts = history["timestamp"].iloc[-1]
    current_bar = history[history["timestamp"] == current_ts]

    # 1. Reconcile fills from orders placed on the previous bar
    _resolve_pending_fills(current_bar)

    # 2. Update our price cache for valuation purposes
    _update_last_prices(current_bar)

    orders = []
    try:
        if step >= LOOKBACK and (step - _state["last_rebalance_step"] >= REBALANCE_EVERY):
            orders = _compute_rebalance_orders(history, current_ts, current_bar)
            _state["last_rebalance_step"] = step
    except Exception:
        orders = []

    # Remember what we submitted so we can reconcile fills next bar
    _state["pending"] = list(orders)

    return orders
