import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Strategy configuration
# ----------------------------------------------------------------------
LOOKBACK_LONG = 60        # bars used for the long momentum window
LOOKBACK_SHORT = 10       # bars subtracted out to avoid short-term reversal
REBALANCE_EVERY = 5       # full re-optimization frequency (bars)
TOP_N = 10                # max number of names held at once
MAX_WEIGHT = 0.20         # cap on any single position's weight
CASH_BUFFER = 0.05        # fraction of equity always kept uninvested
MIN_TRADE_VALUE = 50.0    # ignore rebalancing trades smaller than this ($)
STOP_LOSS_PCT = 0.15      # per-position stop loss
DRAWDOWN_HALT_PCT = 0.25  # portfolio-level circuit breaker trigger
DRAWDOWN_RESUME_PCT = 0.12  # hysteresis: resume trading once dd shrinks below this

_STATE = {}  # module-level state populated by init()


def init(cash):
    global _STATE
    _STATE = {
        "cash": float(cash),
        "positions": {},      # ticker -> shares held (int)
        "avg_cost": {},       # ticker -> average cost basis per share
        "pending_orders": [], # orders returned last call, to be settled now
        "bar_index": 0,
        "peak_equity": float(cash),
        "risk_off": False,
    }


# ----------------------------------------------------------------------
# Internal bookkeeping helpers
# ----------------------------------------------------------------------
def _settle_orders(price_map):
    """Apply last bar's orders using this bar's open prices, following the
    platform's exact fill rules, updating our internal ledger."""
    s = _STATE
    for ticker, direction, size in s["pending_orders"]:
        if size <= 0:
            continue
        price = price_map.get(ticker)
        if price is None or not np.isfinite(price) or price <= 0:
            continue  # no price available -> fills for zero shares

        if direction == "BUY":
            affordable = int(s["cash"] // price)
            fill = min(size, affordable)
            if fill > 0:
                cost = fill * price
                prev_shares = s["positions"].get(ticker, 0)
                prev_cost = s["avg_cost"].get(ticker, 0.0)
                new_shares = prev_shares + fill
                s["avg_cost"][ticker] = (
                    (prev_cost * prev_shares + cost) / new_shares
                    if new_shares > 0 else 0.0
                )
                s["positions"][ticker] = new_shares
                s["cash"] -= cost

        elif direction == "SELL":
            held = s["positions"].get(ticker, 0)
            fill = min(size, held)
            if fill > 0:
                remaining = held - fill
                s["cash"] += fill * price
                if remaining > 0:
                    s["positions"][ticker] = remaining
                else:
                    s["positions"].pop(ticker, None)
                    s["avg_cost"].pop(ticker, None)


def _portfolio_value(close_map):
    s = _STATE
    value = s["cash"]
    for t, shares in s["positions"].items():
        px = close_map.get(t)
        if px is not None and np.isfinite(px):
            value += shares * px
    return value


def _momentum_universe(history):
    """Return (score, vol) Series indexed by ticker, using only tickers
    with a complete price history over the long lookback window."""
    ts_all = history["timestamp"].unique()  # already chronological
    if len(ts_all) < LOOKBACK_LONG + 1:
        return None, None

    threshold = ts_all[-(LOOKBACK_LONG + 1)]
    window = history[history["timestamp"] >= threshold]

    pivot = window.pivot_table(
        index="timestamp", columns="ticker", values="close", aggfunc="last"
    ).sort_index()

    valid = pivot.dropna(axis=1, how="any")
    if valid.shape[1] == 0 or valid.shape[0] < LOOKBACK_LONG + 1:
        return None, None

    long_ret = valid.iloc[-1] / valid.iloc[0] - 1.0
    short_ret = valid.iloc[-1] / valid.iloc[-(LOOKBACK_SHORT + 1)] - 1.0
    score = long_ret - short_ret

    rets = valid.pct_change().dropna(how="all")
    vol = rets.std().replace(0, np.nan).fillna(1e-4)

    return score, vol


def _rebalance_orders(history, close_map, portfolio_value, skip_tickers):
    s = _STATE
    score, vol = _momentum_universe(history)
    orders = []

    if score is None:
        target_shares = {}  # not enough data -> stay flat / hold cash
    else:
        candidates = score[~score.index.isin(skip_tickers)]
        candidates = candidates[candidates > 0].sort_values(ascending=False)
        chosen = candidates.head(TOP_N)

        target_shares = {}
        if len(chosen) > 0:
            inv_vol = 1.0 / vol.reindex(chosen.index)
            weights = inv_vol / inv_vol.sum()
            weights = weights.clip(upper=MAX_WEIGHT)
            if weights.sum() > 0:
                weights = weights / weights.sum() * (1.0 - CASH_BUFFER)

            for ticker, w in weights.items():
                px = close_map.get(ticker)
                if px is None or not np.isfinite(px) or px <= 0:
                    continue
                dollars = w * portfolio_value
                target_shares[ticker] = int(dollars // px)

    # union of currently held tickers and newly targeted tickers
    all_tickers = set(s["positions"].keys()) | set(target_shares.keys())
    sells, buys = [], []
    for ticker in all_tickers:
        if ticker in skip_tickers:
            continue  # already being handled by stop loss / circuit breaker
        cur_shares = s["positions"].get(ticker, 0)
        tgt_shares = target_shares.get(ticker, 0)
        delta = tgt_shares - cur_shares
        if delta == 0:
            continue
        px = close_map.get(ticker, 0.0)
        if abs(delta) * px < MIN_TRADE_VALUE:
            continue
        if delta > 0:
            buys.append((ticker, "BUY", int(delta)))
        else:
            sells.append((ticker, "SELL", int(-delta)))

    orders.extend(sells)
    orders.extend(buys)
    return orders


def _stop_loss_orders(close_map):
    s = _STATE
    orders = []
    hit = set()
    for ticker, shares in list(s["positions"].items()):
        px = close_map.get(ticker)
        cost = s["avg_cost"].get(ticker)
        if px is None or cost is None or cost <= 0:
            continue
        if px <= cost * (1.0 - STOP_LOSS_PCT) and shares > 0:
            orders.append((ticker, "SELL", int(shares)))
            hit.add(ticker)
    return orders, hit


def _circuit_breaker_orders(close_map, portfolio_value):
    s = _STATE
    s["peak_equity"] = max(s["peak_equity"], portfolio_value)
    drawdown = 1.0 - (portfolio_value / s["peak_equity"]) if s["peak_equity"] > 0 else 0.0

    orders = []
    hit = set()
    if not s["risk_off"] and drawdown >= DRAWDOWN_HALT_PCT:
        s["risk_off"] = True

    if s["risk_off"]:
        if drawdown <= DRAWDOWN_RESUME_PCT:
            s["risk_off"] = False
        else:
            for ticker, shares in list(s["positions"].items()):
                if shares > 0:
                    orders.append((ticker, "SELL", int(shares)))
                    hit.add(ticker)
    return orders, hit


# ----------------------------------------------------------------------
# Platform entry point
# ----------------------------------------------------------------------
def on_bar(history):
    s = _STATE

    last_ts = history["timestamp"].iloc[-1]
    cur = history[history["timestamp"] == last_ts]
    open_map = dict(zip(cur["ticker"], cur["open"]))
    close_map = dict(zip(cur["ticker"], cur["close"]))

    # 1. Settle orders placed last call at this bar's open prices.
    _settle_orders(open_map)

    s["bar_index"] += 1

    # 2. Valuation and risk checks (every bar).
    portfolio_value = _portfolio_value(close_map)

    orders = []
    touched = set()

    cb_orders, cb_hit = _circuit_breaker_orders(close_map, portfolio_value)
    orders.extend(cb_orders)
    touched |= cb_hit

    if not s["risk_off"]:
        sl_orders, sl_hit = _stop_loss_orders(close_map)
        sl_orders = [o for o in sl_orders if o[0] not in touched]
        orders.extend(sl_orders)
        touched |= sl_hit

    # 3. Scheduled full rebalance (skipped while in risk-off mode).
    if not s["risk_off"] and s["bar_index"] % REBALANCE_EVERY == 0:
        reb_orders = _rebalance_orders(history, close_map, portfolio_value, touched)
        orders.extend(reb_orders)

    # 4. Clean up: positive integer sizes only.
    final_orders = [(t, d, int(q)) for (t, d, q) in orders if q and int(q) > 0]

    s["pending_orders"] = final_orders
    return final_orders
