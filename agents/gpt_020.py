import math
from collections import defaultdict, deque

import numpy as np


# Portfolio/account state maintained locally because the platform does not
# provide fill reports or account statements.
_cash = 0.0
_positions = {}
_pending_orders = []
_processed_rows = 0
_bar_count = 0

# Per-ticker rolling market data.
_closes = defaultdict(lambda: deque(maxlen=70))
_volumes = defaultdict(lambda: deque(maxlen=70))
_last_close = {}
_peak_close = {}

# Strategy settings.
_LOOKBACK_FAST = 20
_LOOKBACK_SLOW = 60
_TREND_LOOKBACK = 50
_REBALANCE_EVERY = 5
_MAX_POSITIONS = 8
_TARGET_INVESTED_FRACTION = 0.80
_STOP_LOSS_FROM_PEAK = 0.12


def init(cash):
    """Initialize the agent state. Called exactly once by the platform."""
    global _cash, _positions, _pending_orders, _processed_rows, _bar_count
    global _closes, _volumes, _last_close, _peak_close

    _cash = float(cash)
    _positions = {}
    _pending_orders = []
    _processed_rows = 0
    _bar_count = 0

    _closes = defaultdict(lambda: deque(maxlen=70))
    _volumes = defaultdict(lambda: deque(maxlen=70))
    _last_close = {}
    _peak_close = {}


def _valid_price(value):
    try:
        value = float(value)
        return math.isfinite(value) and value > 0.0
    except (TypeError, ValueError):
        return False


def _safe_float(value, default=0.0):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def _apply_pending_orders(open_prices):
    """
    Reconstruct fills for orders submitted on the previous bar.

    Orders are modeled in their submitted order. The strategy always places
    sells before buys, matching the intended use of sale proceeds for buys.
    """
    global _cash, _positions, _pending_orders

    for ticker, direction, requested_size in _pending_orders:
        price = open_prices.get(ticker)

        if not _valid_price(price):
            continue

        price = float(price)
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
                    _peak_close.pop(ticker, None)

        elif direction == "BUY":
            affordable = int(_cash // price)
            filled = min(requested_size, affordable)

            if filled > 0:
                _positions[ticker] = int(_positions.get(ticker, 0)) + filled
                _cash -= filled * price

    _pending_orders = []


def _portfolio_value(current_prices):
    """Mark the known portfolio using current closes or the latest close."""
    total = _cash

    for ticker, shares in _positions.items():
        if shares <= 0:
            continue

        price = current_prices.get(ticker)
        if not _valid_price(price):
            price = _last_close.get(ticker)

        if _valid_price(price):
            total += shares * float(price)

    return total


def _score_ticker(ticker, current_close):
    """
    Return a risk-adjusted trend score, or None when there is insufficient data.

    The score favors:
    - positive medium-term and longer-term momentum,
    - prices above their 50-bar moving average,
    - lower realized volatility for a given momentum level.
    """
    prices = _closes[ticker]
    vols = _volumes[ticker]

    if len(prices) < _LOOKBACK_SLOW + 1:
        return None

    arr = np.asarray(prices, dtype=float)
    vol_arr = np.asarray(vols, dtype=float)

    if (
        not np.all(np.isfinite(arr))
        or np.any(arr <= 0.0)
        or not _valid_price(current_close)
    ):
        return None

    fast_return = arr[-1] / arr[-(_LOOKBACK_FAST + 1)] - 1.0
    slow_return = arr[-1] / arr[-(_LOOKBACK_SLOW + 1)] - 1.0
    moving_average = float(np.mean(arr[-_TREND_LOOKBACK:]))

    if not math.isfinite(moving_average) or moving_average <= 0.0:
        return None

    # Require a positive long-term trend.
    if arr[-1] <= moving_average:
        return None

    # Require non-negative medium-term momentum; this avoids buying names
    # whose long-term return is positive but are currently deteriorating.
    if fast_return <= 0.0:
        return None

    log_returns = np.diff(np.log(arr[-(_LOOKBACK_FAST + 1):]))
    realized_vol = float(np.std(log_returns, ddof=1))

    if not math.isfinite(realized_vol):
        return None

    realized_vol = max(realized_vol, 0.005)

    # Avoid names with effectively unusable reported activity.
    recent_dollar_volume = float(np.mean(arr[-20:] * np.maximum(vol_arr[-20:], 0.0)))
    if not math.isfinite(recent_dollar_volume) or recent_dollar_volume <= 1000.0:
        return None

    # Momentum weighted toward recent behavior, adjusted for volatility.
    raw_momentum = 0.65 * fast_return + 0.35 * slow_return
    score = raw_momentum / realized_vol

    if not math.isfinite(score):
        return None

    return score


def _build_selection(current_prices):
    """Choose the strongest currently tradable tickers."""
    candidates = []

    for ticker, close in current_prices.items():
        score = _score_ticker(ticker, close)
        if score is not None:
            candidates.append((score, str(ticker), ticker))

    candidates.sort(key=lambda item: (-item[0], item[1]))

    # Do not make a portfolio decision from an extremely narrow universe.
    if len(candidates) < 3:
        return None

    return [ticker for _, _, ticker in candidates[:_MAX_POSITIONS]]


def on_bar(history):
    """
    Process the newest timestamp and return orders for next-bar open execution.

    Strategy:
    - Hold up to eight stocks with strong 20/60-bar risk-adjusted momentum.
    - Require price above its 50-bar average.
    - Rebalance every five bars.
    - Keep roughly 80% of equity invested, leaving a cash buffer.
    - Exit holdings that fall 12% below their post-entry high-water mark.
    """
    global _processed_rows, _bar_count, _pending_orders, _peak_close

    # Since history is append-only and on_bar is called once per timestamp,
    # rows after _processed_rows are exactly the newly available bar.
    if len(history) < _processed_rows:
        # Defensive fallback for an unexpected history reset.
        _processed_rows = 0

    current = history.iloc[_processed_rows:]
    _processed_rows = len(history)

    if current.empty:
        _pending_orders = []
        return []

    _bar_count += 1

    open_prices = {}
    close_prices = {}

    # Read this timestamp's bars, update rolling market state, and collect
    # prices required both for fill reconstruction and decision making.
    for row in current[["ticker", "open", "close", "volume"]].itertuples(index=False):
        ticker, open_price, close_price, volume = row

        if _valid_price(open_price):
            open_prices[ticker] = float(open_price)

        if _valid_price(close_price):
            close_value = float(close_price)
            close_prices[ticker] = close_value
            _last_close[ticker] = close_value
            _closes[ticker].append(close_value)
            _volumes[ticker].append(max(0.0, _safe_float(volume, 0.0)))

    # Previous orders execute at this bar's open before today's close-based
    # signal calculation.
    _apply_pending_orders(open_prices)

    # Update high-water marks used by the trailing stop.
    for ticker, shares in list(_positions.items()):
        if shares <= 0:
            _positions.pop(ticker, None)
            _peak_close.pop(ticker, None)
            continue

        close = close_prices.get(ticker)
        if _valid_price(close):
            previous_peak = _peak_close.get(ticker, float(close))
            _peak_close[ticker] = max(previous_peak, float(close))
        elif ticker not in _peak_close:
            fallback = _last_close.get(ticker)
            if _valid_price(fallback):
                _peak_close[ticker] = float(fallback)

    # Stop-loss exits can happen on any bar, not only rebalance bars.
    forced_exits = set()
    for ticker, shares in _positions.items():
        if shares <= 0:
            continue

        close = close_prices.get(ticker)
        peak = _peak_close.get(ticker)

        if _valid_price(close) and _valid_price(peak):
            if float(close) <= float(peak) * (1.0 - _STOP_LOSS_FROM_PEAK):
                forced_exits.add(ticker)

    rebalance_now = (_bar_count % _REBALANCE_EVERY == 0)
    desired_shares = None

    if rebalance_now:
        selection = _build_selection(close_prices)

        if selection is not None:
            equity = _portfolio_value(close_prices)
            per_position_dollars = (
                equity * _TARGET_INVESTED_FRACTION / max(1, len(selection))
            )

            desired_shares = {}
            for ticker in selection:
                price = close_prices.get(ticker)
                if _valid_price(price):
                    desired_shares[ticker] = max(
                        0, int(per_position_dollars // float(price))
                    )

    # If there is no sufficiently broad eligible universe yet, only act on
    # risk exits. This prevents early-history churn.
    if desired_shares is None:
        if not forced_exits:
            _pending_orders = []
            return []

        orders = []
        for ticker in sorted(forced_exits, key=str):
            shares = int(_positions.get(ticker, 0))
            if shares > 0:
                orders.append((ticker, "SELL", shares))

        _pending_orders = orders[:]
        return orders

    # Stops override any rebalance target.
    for ticker in forced_exits:
        desired_shares.pop(ticker, None)

    orders = []
    estimated_sale_proceeds = 0.0

    # Sells are intentionally placed first so platform execution can use those
    # proceeds before handling buys.
    for ticker in sorted(list(_positions.keys()), key=str):
        held = int(_positions.get(ticker, 0))
        target = int(desired_shares.get(ticker, 0))

        if ticker in forced_exits:
            target = 0

        if held > target:
            quantity = held - target
            orders.append((ticker, "SELL", quantity))

            mark = close_prices.get(ticker, _last_close.get(ticker))
            if _valid_price(mark):
                # Conservative estimate to avoid relying fully on sale proceeds.
                estimated_sale_proceeds += quantity * float(mark) * 0.97

    # Keep another buffer against overnight gaps. Buy quantities are capped
    # across the complete order list, rather than allowing each position to
    # independently assume all cash is available.
    estimated_buy_budget = max(0.0, (_cash + estimated_sale_proceeds) * 0.95)

    for ticker in sorted(desired_shares.keys(), key=str):
        if ticker in forced_exits:
            continue

        target = int(desired_shares[ticker])
        held = int(_positions.get(ticker, 0))
        needed = target - held

        if needed <= 0:
            continue

        price = close_prices.get(ticker)
        if not _valid_price(price):
            continue

        affordable_in_plan = int(estimated_buy_budget // float(price))
        quantity = min(needed, affordable_in_plan)

        if quantity > 0:
            orders.append((ticker, "BUY", quantity))
            estimated_buy_budget -= quantity * float(price)

    _pending_orders = orders[:]
    return orders
