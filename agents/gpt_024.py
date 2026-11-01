import math
from collections import deque

import numpy as np


# All state is held in memory because the platform does not provide
# account statements or fill reports.
_STATE = None


def init(cash):
    """
    Called once by the platform before the first bar.
    """
    global _STATE

    _STATE = {
        "cash": float(cash),
        "positions": {},          # ticker -> shares
        "entry_price": {},        # ticker -> average purchase price
        "pending": [],            # orders submitted on the previous bar
        "closes": {},             # ticker -> rolling close deque
        "volumes": {},            # ticker -> rolling volume deque
        "last_price": {},         # ticker -> latest observed close
        "bar_number": 0,
        "cooldown_until": {},     # avoid immediately re-entering stopped names
    }


def _valid_price(x):
    try:
        return math.isfinite(float(x)) and float(x) > 0.0
    except (TypeError, ValueError):
        return False


def _valid_number(x):
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _apply_pending_orders(current_opens):
    """
    Reconstruct fills from orders sent on the previous bar.

    Orders execute at this bar's open.  The returned order list is processed
    in order, matching the intended platform behavior.  Sell orders are
    emitted before buy orders, so sale proceeds may fund the replacement buy.
    """
    state = _STATE
    pending = state["pending"]
    state["pending"] = []

    for ticker, direction, requested_size in pending:
        requested_size = int(requested_size)
        if requested_size <= 0:
            continue

        price = current_opens.get(ticker)
        if price is None or not _valid_price(price):
            # A missing execution-bar price means the order fills for zero.
            continue

        price = float(price)

        if direction == "SELL":
            held = int(state["positions"].get(ticker, 0))
            filled = min(held, requested_size)

            if filled <= 0:
                continue

            state["cash"] += filled * price
            remaining = held - filled

            if remaining > 0:
                state["positions"][ticker] = remaining
            else:
                state["positions"].pop(ticker, None)
                state["entry_price"].pop(ticker, None)

        elif direction == "BUY":
            affordable = int(state["cash"] // price)
            filled = min(affordable, requested_size)

            if filled <= 0:
                continue

            old_shares = int(state["positions"].get(ticker, 0))
            old_basis = float(state["entry_price"].get(ticker, price))

            state["cash"] -= filled * price
            new_shares = old_shares + filled

            state["positions"][ticker] = new_shares
            state["entry_price"][ticker] = (
                (old_shares * old_basis + filled * price) / new_shares
            )


def _update_market_data(current_rows):
    """
    Add the current closes and volumes to rolling per-ticker histories.
    Also returns open prices for processing prior pending orders.
    """
    state = _STATE
    current_opens = {}

    for row in current_rows.itertuples(index=False):
        ticker = str(row.ticker)

        if _valid_price(row.open):
            current_opens[ticker] = float(row.open)

        if not _valid_price(row.close):
            continue

        close = float(row.close)
        volume = float(row.volume) if _valid_number(row.volume) else 0.0
        volume = max(volume, 0.0)

        if ticker not in state["closes"]:
            # 70 observations covers the 60-bar momentum signal plus
            # the 50-bar trend filter.
            state["closes"][ticker] = deque(maxlen=70)
            state["volumes"][ticker] = deque(maxlen=70)

        state["closes"][ticker].append(close)
        state["volumes"][ticker].append(volume)
        state["last_price"][ticker] = close

    return current_opens


def _score_candidates():
    """
    Cross-sectional momentum selection.

    A candidate must:
      * have at least 60 bars of history,
      * be above its 50-bar moving average,
      * have positive 60-bar momentum,
      * not be in a sharp recent breakdown,
      * trade with non-zero recent dollar volume.

    The score rewards 60-bar and 20-bar momentum while penalizing volatility.
    """
    state = _STATE
    candidates = []

    for ticker, close_history in state["closes"].items():
        if state["bar_number"] < state["cooldown_until"].get(ticker, 0):
            continue

        if len(close_history) < 61:
            continue

        prices = np.asarray(close_history, dtype=float)
        current = float(prices[-1])

        if not np.isfinite(current) or current <= 0:
            continue

        ma50 = float(np.mean(prices[-50:]))
        p20 = float(prices[-21])
        p60 = float(prices[-61])

        if ma50 <= 0 or p20 <= 0 or p60 <= 0:
            continue

        ret20 = current / p20 - 1.0
        ret60 = current / p60 - 1.0

        # Daily log-return volatility over approximately one month.
        recent = prices[-21:]
        log_returns = np.diff(np.log(recent))
        volatility = float(np.std(log_returns, ddof=0))

        volumes = state["volumes"].get(ticker)
        avg_volume = float(np.mean(np.asarray(volumes, dtype=float)[-20:]))
        dollar_volume = avg_volume * current

        # Avoid names that cannot plausibly be traded.  The threshold is
        # deliberately mild so this works on both real and synthetic universes.
        if dollar_volume <= 0:
            continue

        # Trend and momentum filters.
        if current <= ma50:
            continue
        if ret60 <= 0.0:
            continue
        if ret20 < -0.08:
            continue

        # Momentum-heavy score with a modest risk penalty.
        score = 0.65 * ret60 + 0.35 * ret20 - 0.25 * volatility

        # Small liquidity preference, capped so it cannot overwhelm momentum.
        liquidity_bonus = min(0.03, max(0.0, math.log10(dollar_volume + 1.0) * 0.003))
        score += liquidity_bonus

        candidates.append((score, ticker))

    if not candidates:
        return None, None

    candidates.sort(reverse=True)
    return candidates[0][1], candidates[0][0]


def _should_stop_out(ticker):
    """
    Protective exit logic for an existing position.

    Exit if:
      * price falls materially below the 50-bar trend,
      * price loses 10% from the average entry price, or
      * short-term momentum becomes severely negative.
    """
    state = _STATE
    history = state["closes"].get(ticker)

    if history is None or len(history) < 21:
        return False

    prices = np.asarray(history, dtype=float)
    current = float(prices[-1])

    if current <= 0:
        return False

    entry = state["entry_price"].get(ticker)
    if entry is not None and current < 0.90 * float(entry):
        return True

    if len(prices) >= 50:
        ma50 = float(np.mean(prices[-50:]))
        if current < 0.98 * ma50:
            return True

    p20 = float(prices[-21])
    if p20 > 0 and current / p20 - 1.0 < -0.12:
        return True

    return False


def _estimated_equity():
    """
    Estimate account equity using latest observed closes.  This is only used
    to size future orders; actual cash and positions are reconstructed from
    next-open executions in _apply_pending_orders.
    """
    state = _STATE
    equity = float(state["cash"])

    for ticker, shares in state["positions"].items():
        price = state["last_price"].get(ticker)
        if price is not None and _valid_price(price):
            equity += int(shares) * float(price)

    return equity


def on_bar(history):
    """
    Main trading callback.

    Strategy:
      * Review the market every five bars.
      * Hold at most one high-momentum, above-trend ticker.
      * Invest approximately 90% of estimated account equity.
      * Move to a stronger name during scheduled rebalance windows.
      * Exit immediately on trend failure, major loss, or severe breakdown.
      * Keep a short cooldown after a stop-out to reduce whipsaw re-entry.
    """
    global _STATE

    if _STATE is None:
        # Defensive fallback; platform specification says init is always called.
        init(0.0)

    if history is None or len(history) == 0:
        return []

    state = _STATE

    # Rows for the newest timestamp are contiguous because history is sorted
    # by timestamp, then ticker.
    latest_timestamp = history["timestamp"].iloc[-1]
    start = history["timestamp"].searchsorted(latest_timestamp, side="left")
    current_rows = history.iloc[int(start):]

    # First reconcile fills from the previous callback at this timestamp's open.
    current_opens = {}
    for row in current_rows.itertuples(index=False):
        if _valid_price(row.open):
            current_opens[str(row.ticker)] = float(row.open)
    _apply_pending_orders(current_opens)

    # Then consume this bar's close for signal generation.
    _update_market_data(current_rows)
    state["bar_number"] += 1

    orders = []

    # Protective stops run every bar, rather than waiting for a rebalance date.
    stopped_any = False
    for ticker, shares in list(state["positions"].items()):
        if shares > 0 and _should_stop_out(ticker):
            orders.append((ticker, "SELL", int(shares)))
            state["cooldown_until"][ticker] = state["bar_number"] + 10
            stopped_any = True

    # Do not open a replacement position on the same close as a stop signal.
    # This reduces turnover during broad market reversals.
    if stopped_any:
        state["pending"] = orders
        return orders

    # Rebalance only every five bars after enough history has accumulated.
    if state["bar_number"] < 61 or state["bar_number"] % 5 != 0:
        state["pending"] = orders
        return orders

    target_ticker, _target_score = _score_candidates()

    # Sell anything that is no longer the selected target.
    for ticker, shares in list(state["positions"].items()):
        if shares <= 0:
            continue
        if ticker != target_ticker:
            orders.append((ticker, "SELL", int(shares)))

    # No qualifying trend: remain in cash after liquidating existing holdings.
    if target_ticker is None:
        state["pending"] = orders
        return orders

    target_price = state["last_price"].get(target_ticker)
    if target_price is None or not _valid_price(target_price):
        state["pending"] = orders
        return orders

    # Use 90% of estimated equity.  The spare cash reduces failed fills when
    # the next open gaps upward.
    estimated_equity = _estimated_equity()
    target_dollars = 0.90 * estimated_equity
    desired_shares = int(target_dollars // float(target_price))
    current_shares = int(state["positions"].get(target_ticker, 0))

    if desired_shares > current_shares:
        # Sell orders were appended first, allowing their proceeds to fund
        # this purchase when the platform processes next-bar-open orders.
        orders.append((target_ticker, "BUY", desired_shares - current_shares))
    elif desired_shares < current_shares:
        orders.append((target_ticker, "SELL", current_shares - desired_shares))

    state["pending"] = orders
    return orders
