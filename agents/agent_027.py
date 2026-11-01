import math
from collections import deque

import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Strategy parameters
# ------------------------------------------------------------------
LOOKBACK_TREND   = 200      # bars for the trend-filter SMA
LOOKBACK_MOM     = 126      # bars for the momentum lookback
DEQUE_MAXLEN     = LOOKBACK_TREND + 10
REBALANCE_PERIOD = 21       # rebalance roughly monthly
TOP_K            = 10       # max number of names held at once
CASH_BUFFER      = 0.05     # fraction of equity always left uninvested
MAX_WEIGHT       = 0.15     # cap on any single name's weight
STOP_LOSS_PCT    = 0.20     # trailing stop from the peak price
MIN_TRADE_FRAC   = 0.005    # ignore rebalance trades smaller than this fraction of equity
MIN_TRADE_ABS    = 50.0     # ... or this many dollars, whichever is larger

_state = {}


def init(cash):
    _state.clear()
    _state.update({
        "cash": float(cash),
        "positions": {},          # ticker -> shares held
        "pending_orders": [],     # orders submitted last call, to settle now
        "price_hist": {},         # ticker -> deque of recent closes
        "last_price": {},         # ticker -> most recently seen close
        "peak_price": {},         # ticker -> peak close since entry (for stop loss)
        "day_count": 0,
    })


def _get_current_bar(history):
    ts = history["timestamp"].values
    last_ts = ts[-1]
    start_idx = np.searchsorted(ts, last_ts, side="left")
    return history.iloc[start_idx:]


def _settle_pending_orders(current_bar):
    if not _state["pending_orders"]:
        return
    open_px = dict(zip(current_bar["ticker"], current_bar["open"]))
    cash = _state["cash"]
    positions = _state["positions"]

    for ticker, direction, size in _state["pending_orders"]:
        price = open_px.get(ticker)
        if price is None or price <= 0 or size <= 0:
            continue
        if direction == "BUY":
            max_affordable = math.floor(cash / price)
            fill_qty = min(size, max_affordable)
            if fill_qty > 0:
                cash -= fill_qty * price
                positions[ticker] = positions.get(ticker, 0) + fill_qty
        elif direction == "SELL":
            held = positions.get(ticker, 0)
            fill_qty = min(size, held)
            if fill_qty > 0:
                cash += fill_qty * price
                remaining = held - fill_qty
                if remaining > 0:
                    positions[ticker] = remaining
                else:
                    positions.pop(ticker, None)

    _state["cash"] = cash
    _state["pending_orders"] = []


def _update_histories(current_bar):
    price_hist = _state["price_hist"]
    last_price = _state["last_price"]
    for ticker, close in zip(current_bar["ticker"], current_bar["close"]):
        if pd.isna(close):
            continue
        dq = price_hist.get(ticker)
        if dq is None:
            dq = deque(maxlen=DEQUE_MAXLEN)
            price_hist[ticker] = dq
        dq.append(float(close))
        last_price[ticker] = float(close)


def _current_price_map(current_bar):
    return {t: float(c) for t, c in zip(current_bar["ticker"], current_bar["close"]) if not pd.isna(c)}


def _check_stop_losses(current_prices):
    orders = []
    hit = set()
    positions = _state["positions"]
    peak_price = _state["peak_price"]
    last_price = _state["last_price"]

    for ticker in list(positions.keys()):
        shares = positions[ticker]
        if shares <= 0:
            continue
        price = current_prices.get(ticker, last_price.get(ticker))
        if price is None:
            continue
        peak = max(peak_price.get(ticker, price), price)
        peak_price[ticker] = peak
        if price <= peak * (1.0 - STOP_LOSS_PCT):
            orders.append((ticker, "SELL", shares))
            hit.add(ticker)
    return orders, hit


def _select_targets(exclude):
    price_hist = _state["price_hist"]
    candidates = []
    for ticker, dq in price_hist.items():
        if ticker in exclude:
            continue
        if len(dq) < LOOKBACK_TREND:
            continue
        arr = np.asarray(dq)
        last = arr[-1]
        sma200 = arr[-LOOKBACK_TREND:].mean()
        if last <= sma200:
            continue
        if len(arr) < LOOKBACK_MOM + 1:
            continue
        past = arr[-(LOOKBACK_MOM + 1)]
        if past <= 0:
            continue
        mom = last / past - 1.0
        candidates.append((ticker, mom))

    candidates.sort(key=lambda x: x[1], reverse=True)
    selected = [t for t, _ in candidates[:TOP_K]]

    n = len(selected)
    weights = {}
    if n > 0:
        w = min((1.0 - CASH_BUFFER) / n, MAX_WEIGHT)
        for t in selected:
            weights[t] = w
    return weights


def _build_rebalance_orders(weights, current_prices, exclude):
    positions = _state["positions"]
    last_price = _state["last_price"]

    equity = _state["cash"]
    for t, sh in positions.items():
        p = current_prices.get(t, last_price.get(t))
        if p is not None:
            equity += sh * p

    threshold = max(equity * MIN_TRADE_FRAC, MIN_TRADE_ABS)

    tickers = set(positions.keys()) | set(weights.keys())
    sells, buys = [], []

    for ticker in tickers:
        if ticker in exclude:
            continue
        price = current_prices.get(ticker, last_price.get(ticker))
        if price is None or price <= 0:
            continue
        current_shares = positions.get(ticker, 0)
        current_value = current_shares * price
        target_value = equity * weights.get(ticker, 0.0)
        diff = target_value - current_value

        if diff > threshold:
            qty = int(diff // price)
            if qty > 0:
                buys.append((ticker, "BUY", qty))
        elif diff < -threshold:
            qty = int((-diff) // price)
            qty = min(qty, current_shares)
            if qty > 0:
                sells.append((ticker, "SELL", qty))

    return sells, buys


def on_bar(history):
    if history.empty:
        return []

    current_bar = _get_current_bar(history)

    # 1. settle orders placed on the previous call using this bar's opens
    _settle_pending_orders(current_bar)

    # 2. update running per-ticker price windows
    _update_histories(current_bar)

    _state["day_count"] += 1
    current_prices = _current_price_map(current_bar)

    # 3. stop-loss checks (every bar)
    stop_orders, stopped_tickers = _check_stop_losses(current_prices)

    orders = list(stop_orders)

    # 4. periodic rebalance
    day = _state["day_count"]
    if day >= LOOKBACK_TREND and day % REBALANCE_PERIOD == 0:
        weights = _select_targets(exclude=stopped_tickers)
        sells, buys = _build_rebalance_orders(weights, current_prices, exclude=stopped_tickers)
        orders.extend(sells)
        orders.extend(buys)

    # 5. remember what we submitted so we can settle it next call
    _state["pending_orders"] = list(orders)

    return orders
