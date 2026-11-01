import math
from collections import deque

# ----------------------------------------------------------------------
# Module-level state (the platform imports this module once and calls
# init() then on_bar() repeatedly; no execution happens at import time).
# ----------------------------------------------------------------------
_STATE = None

# ----- strategy constants -----
LOOKBACK = 20            # bars used to compute momentum
REBALANCE_PERIOD = 5     # rebalance the portfolio every N bars
TOP_N = 8                # number of names to hold at once
CASH_BUFFER = 0.05       # fraction of equity always kept in cash
STOP_LOSS = 0.12         # exit a position if it falls 12% from avg entry
MIN_PRICE = 1.0          # ignore penny/garbage tickers


def init(cash):
    global _STATE
    _STATE = {
        "cash": float(cash),
        "positions": {},        # ticker -> shares held (int)
        "entry_price": {},      # ticker -> volume-weighted avg entry price
        "pending_orders": [],   # orders submitted last call, to fill this bar's open
        "prev_len": 0,          # how many rows of history we've already consumed
        "price_hist": {},       # ticker -> deque of recent closes (len LOOKBACK+1)
        "last_price": {},       # ticker -> last known close (for equity marking)
        "bar_count": 0,
        "last_rebalance_bar": -REBALANCE_PERIOD,  # force rebalance once warmed up
    }


def _simulate_fill(order, prices):
    """Replay the platform's exact fill rule against our internal ledger."""
    ticker, direction, size = order
    price = prices.get(ticker)
    if price is None or price <= 0 or size <= 0:
        return
    st = _STATE
    if direction == "BUY":
        affordable = int(st["cash"] // price)
        fill = min(size, affordable)
        if fill > 0:
            st["cash"] -= fill * price
            old_shares = st["positions"].get(ticker, 0)
            old_entry = st["entry_price"].get(ticker, price)
            new_shares = old_shares + fill
            st["entry_price"][ticker] = (
                (old_entry * old_shares + price * fill) / new_shares
            )
            st["positions"][ticker] = new_shares
    elif direction == "SELL":
        held = st["positions"].get(ticker, 0)
        fill = min(size, held)
        if fill > 0:
            st["cash"] += fill * price
            remaining = held - fill
            st["positions"][ticker] = remaining
            if remaining <= 0:
                st["positions"].pop(ticker, None)
                st["entry_price"].pop(ticker, None)


def on_bar(history):
    st = _STATE

    # ---- 1. isolate rows new since the last call (cheap, O(new rows)) ----
    new_rows = history.iloc[st["prev_len"]:]
    st["prev_len"] = len(history)
    if new_rows.empty:
        return []

    current_open = dict(zip(new_rows["ticker"], new_rows["open"]))
    current_close = dict(zip(new_rows["ticker"], new_rows["close"]))
    current_vol = dict(zip(new_rows["ticker"], new_rows["volume"]))

    # ---- 2. settle orders we submitted last bar, using THIS bar's open ----
    for order in st["pending_orders"]:
        _simulate_fill(order, current_open)
    st["pending_orders"] = []

    # ---- 3. update rolling price history / last known prices ----
    for ticker, close in current_close.items():
        if close is None or close != close:  # NaN guard
            continue
        st["last_price"][ticker] = close
        dq = st["price_hist"].get(ticker)
        if dq is None:
            dq = deque(maxlen=LOOKBACK + 1)
            st["price_hist"][ticker] = dq
        dq.append(close)

    st["bar_count"] += 1
    orders = []

    # ---- 4. stop-loss check on every bar (risk control) ----
    to_force_sell = set()
    for ticker, shares in list(st["positions"].items()):
        if shares <= 0:
            continue
        price = current_close.get(ticker, st["last_price"].get(ticker))
        entry = st["entry_price"].get(ticker)
        if price is None or entry is None or entry <= 0:
            continue
        if price <= entry * (1.0 - STOP_LOSS):
            to_force_sell.add(ticker)
            orders.append((ticker, "SELL", int(shares)))

    # ---- 5. periodic rebalance based on cross-sectional momentum ----
    do_rebalance = (st["bar_count"] - st["last_rebalance_bar"]) >= REBALANCE_PERIOD
    if do_rebalance:
        candidates = []
        for ticker, dq in st["price_hist"].items():
            if ticker in to_force_sell:
                continue
            if len(dq) < LOOKBACK + 1:
                continue
            first, last = dq[0], dq[-1]
            if first is None or first <= 0 or last is None or last < MIN_PRICE:
                continue
            avg_vol = current_vol.get(ticker, None)
            mom = (last / first) - 1.0
            if mom > 0:
                candidates.append((mom, ticker, last))

        candidates.sort(key=lambda x: x[0], reverse=True)
        selected = candidates[:TOP_N]
        selected_tickers = {t for _, t, _ in selected}

        # mark-to-market equity using best-known prices
        equity = st["cash"]
        for ticker, shares in st["positions"].items():
            price = current_close.get(ticker, st["last_price"].get(ticker, 0.0))
            equity += shares * price

        investable = equity * (1.0 - CASH_BUFFER)
        weight = investable / len(selected) if selected else 0.0

        target_shares = {}
        for mom, ticker, price in selected:
            if price and price > 0:
                target_shares[ticker] = int(math.floor(weight / price))

        # any currently held ticker not selected (and not already force-sold)
        # should be trimmed to zero as part of the rebalance
        all_relevant = set(st["positions"].keys()) | selected_tickers
        for ticker in all_relevant:
            if ticker in to_force_sell:
                continue  # already selling it above
            held = st["positions"].get(ticker, 0)
            target = target_shares.get(ticker, 0)
            delta = target - held
            if delta > 0:
                orders.append((ticker, "BUY", int(delta)))
            elif delta < 0:
                orders.append((ticker, "SELL", int(-delta)))

        st["last_rebalance_bar"] = st["bar_count"]

    # ---- 6. sells before buys (frees cash first, matches our own ledger sim) ----
    sells = [o for o in orders if o[1] == "SELL"]
    buys = [o for o in orders if o[1] == "BUY"]
    final_orders = sells + buys

    # ---- 7. remember what we submitted so we can settle it next call ----
    st["pending_orders"] = list(final_orders)

    return final_orders
