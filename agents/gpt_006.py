import math
from collections import deque

import numpy as np
import pandas as pd


# Strategy settings
LOOKBACK_MOM_1 = 63
LOOKBACK_MOM_2 = 126
LOOKBACK_SMA = 100
LOOKBACK_VOL = 21
MAX_HOLDINGS = 8
REBALANCE_EVERY = 5
EQUITY_ALLOCATION = 0.90
CASH_BUFFER = 0.08
COOLDOWN_BARS = 10


# Account state reconstructed locally from submitted orders and next-bar opens.
_cash = 0.0
_positions = {}
_pending_orders = []

# Market-data state.
_processed_rows = 0
_step = 0
_closes = {}
_last_close = {}
_cooldown_until = {}


def init(cash):
    """
    Called once by the platform before the first bar.
    """
    global _cash, _positions, _pending_orders
    global _processed_rows, _step, _closes, _last_close, _cooldown_until

    _cash = float(cash)
    _positions = {}
    _pending_orders = []

    _processed_rows = 0
    _step = 0
    _closes = {}
    _last_close = {}
    _cooldown_until = {}


def _finite_positive(value):
    try:
        value = float(value)
        return math.isfinite(value) and value > 0.0
    except (TypeError, ValueError):
        return False


def _apply_pending_fills(open_prices):
    """
    Reconstruct fills from the orders submitted on the previous bar.

    Orders are emitted with sells before buys. We simulate fills in that same
    order. Buy orders are allowed to fill only with cash actually available at
    the next open.
    """
    global _cash, _positions, _pending_orders

    if not _pending_orders:
        return

    for ticker, direction, requested_size in _pending_orders:
        price = open_prices.get(ticker)

        if not _finite_positive(price):
            continue

        price = float(price)
        requested_size = int(requested_size)

        if direction == "SELL":
            held = int(_positions.get(ticker, 0))
            filled = min(held, requested_size)

            if filled > 0:
                _positions[ticker] = held - filled
                _cash += filled * price

                if _positions[ticker] <= 0:
                    _positions.pop(ticker, None)

        elif direction == "BUY":
            affordable = int(math.floor(_cash / price))
            filled = min(requested_size, max(0, affordable))

            if filled > 0:
                _positions[ticker] = int(_positions.get(ticker, 0)) + filled
                _cash -= filled * price

    _pending_orders = []


def _history_array(ticker):
    values = _closes.get(ticker)
    if values is None:
        return None
    return np.asarray(values, dtype=float)


def _signal_for_ticker(ticker):
    """
    Return a momentum / trend score for a ticker, or None if insufficient data.
    Higher score is better.
    """
    prices = _history_array(ticker)

    if prices is None or len(prices) < LOOKBACK_MOM_2 + 1:
        return None

    current = float(prices[-1])
    if not _finite_positive(current):
        return None

    sma_100 = float(np.mean(prices[-LOOKBACK_SMA:]))
    if not _finite_positive(sma_100):
        return None

    price_63 = float(prices[-(LOOKBACK_MOM_1 + 1)])
    price_126 = float(prices[-(LOOKBACK_MOM_2 + 1)])

    if not _finite_positive(price_63) or not _finite_positive(price_126):
        return None

    ret_63 = current / price_63 - 1.0
    ret_126 = current / price_126 - 1.0

    recent = prices[-(LOOKBACK_VOL + 1):]
    if np.any(recent <= 0.0):
        return None

    log_returns = np.diff(np.log(recent))
    volatility = float(np.std(log_returns, ddof=1)) if len(log_returns) > 1 else 0.0

    # Long-only trend filter. The score rewards medium-term and longer-term
    # momentum while penalizing unstable names.
    if current <= sma_100 or ret_63 <= 0.0:
        return {
            "eligible": False,
            "score": -np.inf,
            "above_sma": current > sma_100,
        }

    momentum = 0.60 * ret_63 + 0.40 * ret_126
    risk_scale = max(volatility * math.sqrt(LOOKBACK_MOM_1), 0.05)
    score = momentum / risk_scale

    return {
        "eligible": True,
        "score": score,
        "above_sma": True,
    }


def _should_stop_out(ticker):
    """
    Protective exit rule for held positions.

    A position exits when it materially breaks its medium-term trend or loses
    a large amount from its recent high.
    """
    prices = _history_array(ticker)

    if prices is None or len(prices) < 55:
        return False

    current = float(prices[-1])
    sma_50 = float(np.mean(prices[-50:]))
    recent_high = float(np.max(prices[-min(len(prices), 126):]))

    if not _finite_positive(current):
        return False

    below_trend = current < sma_50 * 0.97
    trailing_stop = current < recent_high * 0.82

    return below_trend or trailing_stop


def _estimate_equity():
    """
    Mark the locally reconstructed account to the latest available closes.
    """
    equity = float(_cash)

    for ticker, shares in _positions.items():
        if shares <= 0:
            continue

        price = _last_close.get(ticker)
        if _finite_positive(price):
            equity += int(shares) * float(price)

    return max(0.0, equity)


def _select_targets(current_tickers):
    """
    Select the strongest diversified long-only basket.

    When broad market breadth is weak, the strategy holds cash instead of
    forcing long exposure.
    """
    scored = []
    valid_count = 0
    above_trend_count = 0

    for ticker in current_tickers:
        if _step < _cooldown_until.get(ticker, -1):
            continue

        signal = _signal_for_ticker(ticker)
        if signal is None:
            continue

        valid_count += 1
        if signal["above_sma"]:
            above_trend_count += 1

        if signal["eligible"] and math.isfinite(signal["score"]):
            scored.append((float(signal["score"]), ticker))

    # Broad market risk filter. It is only used when the universe is large
    # enough that breadth is meaningful.
    if valid_count >= 10:
        breadth = above_trend_count / valid_count
        if breadth < 0.35:
            return []

    scored.sort(reverse=True)
    return [ticker for _, ticker in scored[:MAX_HOLDINGS]]


def on_bar(history):
    """
    Called once per timestamp. Returns a list of:
        (ticker, "BUY" or "SELL", positive_integer_share_count)
    """
    global _processed_rows, _step, _pending_orders

    if history is None or len(history) == 0:
        return []

    # history is append-only, so only process rows newly added since the last
    # call rather than repeatedly scanning the full DataFrame.
    new_rows = history.iloc[_processed_rows:]
    _processed_rows = len(history)

    if len(new_rows) == 0:
        return []

    _step += 1

    current_tickers = set()
    open_prices = {}

    # At this point new_rows is the current timestamp's cross-section.
    for ticker, open_price, close_price in zip(
        new_rows["ticker"].to_numpy(),
        new_rows["open"].to_numpy(),
        new_rows["close"].to_numpy(),
    ):
        current_tickers.add(ticker)

        if _finite_positive(open_price):
            open_prices[ticker] = float(open_price)

        if _finite_positive(close_price):
            close_price = float(close_price)

            if ticker not in _closes:
                # 260 observations supports all indicators while bounding
                # memory for large universes.
                _closes[ticker] = deque(maxlen=260)

            _closes[ticker].append(close_price)
            _last_close[ticker] = close_price

    # Previous-bar orders execute at this bar's open.
    _apply_pending_fills(open_prices)

    orders = []
    stopped_today = set()

    # Protective stops are evaluated every bar, not just at rebalance dates.
    for ticker, shares in list(_positions.items()):
        if shares <= 0 or ticker not in current_tickers:
            continue

        if _should_stop_out(ticker):
            orders.append((ticker, "SELL", int(shares)))
            stopped_today.add(ticker)
            _cooldown_until[ticker] = _step + COOLDOWN_BARS

    # Rebalance periodically. The first bar is also a rebalance attempt, but
    # indicators will naturally wait until enough data has accumulated.
    do_rebalance = (_step == 1) or (_step % REBALANCE_EVERY == 0)

    if not do_rebalance:
        _pending_orders = orders
        return orders

    targets = _select_targets(current_tickers)
    target_set = set(targets)

    # A stopped ticker should not be repurchased on the same decision bar.
    target_set.difference_update(stopped_today)
    targets = [ticker for ticker in targets if ticker in target_set]

    equity = _estimate_equity()

    if targets:
        target_dollars = EQUITY_ALLOCATION * equity / len(targets)
    else:
        target_dollars = 0.0

    desired_shares = {}
    for ticker in targets:
        price = _last_close.get(ticker)
        if _finite_positive(price):
            desired_shares[ticker] = int(math.floor(target_dollars / float(price)))

    already_selling = {ticker for ticker, direction, _ in orders if direction == "SELL"}

    # First submit reductions and removals. Sells are intentionally listed
    # before buys so a platform that processes order lists sequentially makes
    # released cash available first.
    for ticker, held in list(_positions.items()):
        if held <= 0 or ticker not in current_tickers or ticker in already_selling:
            continue

        desired = int(desired_shares.get(ticker, 0))
        if held > desired:
            orders.append((ticker, "SELL", int(held - desired)))
            already_selling.add(ticker)

    # Buy only with cash already available in the local account model. This is
    # conservative: capital from rotation sales is used on later rebalance
    # cycles if it was not yet known to be available when orders were formed.
    reserve = CASH_BUFFER * equity
    spendable_cash = max(0.0, _cash - reserve)

    candidates_to_buy = []
    requested_cost = 0.0

    for ticker in targets:
        if ticker in stopped_today:
            continue

        held = int(_positions.get(ticker, 0))
        desired = int(desired_shares.get(ticker, 0))
        quantity = max(0, desired - held)

        price = _last_close.get(ticker)
        if quantity > 0 and _finite_positive(price):
            cost = quantity * float(price)
            candidates_to_buy.append((ticker, quantity, float(price)))
            requested_cost += cost

    if requested_cost > 0.0 and spendable_cash > 0.0:
        # Leave a small extra margin for next-open gaps.
        scale = min(1.0, (spendable_cash * 0.995) / requested_cost)

        for ticker, quantity, _price in candidates_to_buy:
            scaled_quantity = int(math.floor(quantity * scale))
            if scaled_quantity > 0:
                orders.append((ticker, "BUY", scaled_quantity))

    _pending_orders = orders
    return orders
