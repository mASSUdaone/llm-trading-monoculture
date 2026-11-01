import math
from collections import defaultdict, deque

import numpy as np
import pandas as pd


# Strategy parameters
LOOKBACK = 70
REBALANCE_EVERY = 5
MAX_POSITIONS = 8
TARGET_EXPOSURE = 0.90
STOP_LOSS = 0.14
CASH_BUFFER = 0.97


# Module-level state. init() resets all of it before each run.
_cash = 0.0
_positions = {}
_pending_orders = []
_price_history = {}
_last_price = {}
_peak_price = {}
_bar_number = 0


def init(cash):
    """
    Initialize the agent state.

    cash is the initial starting balance supplied by the platform.
    """
    global _cash, _positions, _pending_orders
    global _price_history, _last_price, _peak_price, _bar_number

    _cash = float(cash)
    _positions = {}
    _pending_orders = []

    # Per-ticker close history, kept bounded so memory stays controlled.
    _price_history = defaultdict(lambda: deque(maxlen=LOOKBACK))
    _last_price = {}
    _peak_price = {}
    _bar_number = 0


def _valid_price(value):
    try:
        value = float(value)
        return math.isfinite(value) and value > 0.0
    except (TypeError, ValueError):
        return False


def _current_rows(history):
    """
    history is sorted by timestamp then ticker. Find only the rows belonging
    to the latest timestamp without scanning the entire DataFrame.
    """
    if history.empty:
        return history

    latest_timestamp = history["timestamp"].iloc[-1]
    start = int(history["timestamp"].searchsorted(latest_timestamp, side="left"))
    return history.iloc[start:]


def _apply_pending_orders(current_rows):
    """
    Reconstruct fills for orders submitted on the prior bar.

    Since no fill report is provided by the platform, this internally mirrors
    the documented execution model as closely as possible: orders execute at
    the next bar's open, sells first if they were returned first, and partial
    fills are allowed based on tracked cash and positions.
    """
    global _cash, _positions, _pending_orders

    if not _pending_orders:
        return

    open_prices = {}
    for row in current_rows.itertuples(index=False):
        ticker = row.ticker
        price = row.open
        if _valid_price(price):
            open_prices[ticker] = float(price)

    for ticker, direction, requested_size in _pending_orders:
        price = open_prices.get(ticker)
        if price is None:
            continue

        requested_size = int(requested_size)
        if requested_size <= 0:
            continue

        if direction == "SELL":
            held = int(_positions.get(ticker, 0))
            filled = min(held, requested_size)
            if filled > 0:
                _positions[ticker] = held - filled
                _cash += filled * price
                if _positions[ticker] <= 0:
                    _positions.pop(ticker, None)
                    _peak_price.pop(ticker, None)

        elif direction == "BUY":
            affordable = int(_cash // price)
            filled = min(affordable, requested_size)
            if filled > 0:
                _positions[ticker] = int(_positions.get(ticker, 0)) + filled
                _cash -= filled * price

    _pending_orders = []


def _update_market_data(current_rows):
    """
    Add the current closing prices to the rolling per-ticker histories.
    """
    current_close = {}

    for row in current_rows.itertuples(index=False):
        ticker = row.ticker
        close = row.close

        if not _valid_price(close):
            continue

        close = float(close)
        current_close[ticker] = close
        _last_price[ticker] = close
        _price_history[ticker].append(close)

    # Maintain a high-water mark for trailing-stop protection.
    for ticker, shares in list(_positions.items()):
        if shares <= 0:
            continue

        price = current_close.get(ticker)
        if price is None:
            continue

        previous_peak = _peak_price.get(ticker, price)
        _peak_price[ticker] = max(previous_peak, price)

    return current_close


def _portfolio_value():
    """
    Estimate marked-to-market portfolio value using the most recent available
    closing price for each held ticker.
    """
    value = _cash

    for ticker, shares in _positions.items():
        if shares <= 0:
            continue

        price = _last_price.get(ticker)
        if _valid_price(price):
            value += shares * price

    return value


def _score_candidates(current_close, blocked_tickers):
    """
    Cross-sectional momentum and trend selection.

    A ticker must:
      * Be present on the current bar.
      * Have at least 61 observed closes.
      * Trade above its 40-bar moving average.
      * Have positive 20-bar and 60-bar momentum.

    Ranking prefers strong medium-term momentum, strong shorter-term
    momentum, and lower recent volatility.
    """
    candidates = []

    for ticker, price in current_close.items():
        if ticker in blocked_tickers:
            continue

        closes = _price_history.get(ticker)
        if closes is None or len(closes) < 61:
            continue

        prices = np.asarray(closes, dtype=float)
        if np.any(~np.isfinite(prices)) or np.any(prices <= 0.0):
            continue

        current = prices[-1]
        ma40 = float(np.mean(prices[-40:]))

        momentum_20 = current / prices[-21] - 1.0
        momentum_60 = current / prices[-61] - 1.0

        # Log-return volatility over the latest 20 observations.
        log_returns = np.diff(np.log(prices[-21:]))
        volatility_20 = float(np.std(log_returns, ddof=1)) if len(log_returns) > 1 else 0.0

        if current <= ma40:
            continue
        if momentum_20 <= 0.0 or momentum_60 <= 0.0:
            continue
        if not math.isfinite(volatility_20):
            continue

        # Trend / momentum score with a modest volatility penalty.
        score = (
            0.55 * momentum_20
            + 0.45 * momentum_60
            - 0.75 * volatility_20
        )

        candidates.append((score, ticker))

    candidates.sort(reverse=True)
    return candidates[:MAX_POSITIONS]


def _stop_loss_tickers(current_close):
    """
    Identify holdings that have fallen materially from their post-entry peak.
    """
    stopped = set()

    for ticker, shares in _positions.items():
        if shares <= 0:
            continue

        current = current_close.get(ticker)
        peak = _peak_price.get(ticker)

        if current is None or peak is None or peak <= 0.0:
            continue

        if current <= peak * (1.0 - STOP_LOSS):
            stopped.add(ticker)

    return stopped


def _rebalance(current_close, forced_exits):
    """
    Build a sell-first rebalance.

    Holdings are allocated approximately equally among the strongest qualified
    momentum names. A cash reserve is retained, and requested buy quantities
    are conservatively capped so that expected proceeds from sells are not
    overcommitted.
    """
    portfolio_value = _portfolio_value()
    if portfolio_value <= 0.0:
        return []

    ranked = _score_candidates(current_close, forced_exits)
    selected = [ticker for _, ticker in ranked]

    # Lower exposure when only a small number of assets pass the trend filter.
    if len(selected) >= 3:
        exposure = TARGET_EXPOSURE
    elif len(selected) > 0:
        exposure = 0.60
    else:
        exposure = 0.0

    desired_shares = {}

    if selected and exposure > 0.0:
        target_value_per_position = portfolio_value * exposure / len(selected)

        for ticker in selected:
            price = current_close.get(ticker)
            if _valid_price(price):
                desired_shares[ticker] = int(target_value_per_position // price)

    # Any ticker not selected is fully exited. Forced exits override selection.
    all_relevant = set(_positions) | set(desired_shares) | set(forced_exits)

    sell_orders = []
    expected_sale_proceeds = 0.0

    for ticker in all_relevant:
        held = int(_positions.get(ticker, 0))
        target = int(desired_shares.get(ticker, 0))

        if ticker in forced_exits:
            target = 0

        if held > target:
            quantity = held - target
            sell_orders.append((ticker, "SELL", quantity))

            price = current_close.get(ticker, _last_price.get(ticker))
            if _valid_price(price):
                expected_sale_proceeds += quantity * float(price)

    # Determine desired purchases after planned reductions.
    desired_buys = []
    total_buy_cost = 0.0

    for ticker, target in desired_shares.items():
        held = int(_positions.get(ticker, 0))
        if target <= held:
            continue

        price = current_close.get(ticker)
        if not _valid_price(price):
            continue

        quantity = target - held
        desired_buys.append((ticker, quantity, float(price)))
        total_buy_cost += quantity * float(price)

    # Sells are returned before buys. Reserve a buffer because execution occurs
    # at the next open rather than at the current close used for planning.
    estimated_available_cash = max(
        0.0,
        (_cash + expected_sale_proceeds) * CASH_BUFFER,
    )

    buy_orders = []

    if total_buy_cost > 0.0 and estimated_available_cash > 0.0:
        scale = min(1.0, estimated_available_cash / total_buy_cost)

        for ticker, quantity, price in desired_buys:
            scaled_quantity = int(math.floor(quantity * scale))
            if scaled_quantity > 0:
                buy_orders.append((ticker, "BUY", scaled_quantity))

    return sell_orders + buy_orders


def on_bar(history):
    """
    Called once per timestamp.

    The agent:
      1. Reconstructs fills for last bar's submitted orders.
      2. Updates rolling market data.
      3. Checks trailing stops every bar.
      4. Rebalances every REBALANCE_EVERY bars into the strongest momentum
         names, using sell-first orders and conservative cash sizing.
    """
    global _pending_orders, _bar_number

    if history is None or history.empty:
        return []

    current_rows = _current_rows(history)

    # First reconcile orders submitted on the preceding timestamp.
    _apply_pending_orders(current_rows)

    # Then process this bar's closing information.
    current_close = _update_market_data(current_rows)
    _bar_number += 1

    if not current_close:
        return []

    forced_exits = _stop_loss_tickers(current_close)

    # Rebalance periodically. Stops are also honored on non-rebalance bars.
    if _bar_number % REBALANCE_EVERY == 0:
        orders = _rebalance(current_close, forced_exits)
    else:
        orders = []
        for ticker in sorted(forced_exits):
            shares = int(_positions.get(ticker, 0))
            if shares > 0:
                orders.append((ticker, "SELL", shares))

    # Store exactly what was returned so next bar can reconstruct fills.
    _pending_orders = list(orders)
    return orders
