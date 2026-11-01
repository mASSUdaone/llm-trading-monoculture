import math
from collections import deque

import numpy as np


# Portfolio/account state
_cash = 0.0
_positions = {}
_peaks = {}
_pending_orders = []

# Market-data / indicator state
_state = {}
_last_close = {}
_seen_rows = 0
_bar_number = 0


def init(cash):
    """Initialize the strategy once before the first bar."""
    global _cash, _positions, _peaks, _pending_orders
    global _state, _last_close, _seen_rows, _bar_number

    _cash = float(cash)
    _positions = {}
    _peaks = {}
    _pending_orders = []

    _state = {}
    _last_close = {}
    _seen_rows = 0
    _bar_number = 0


def _finite_positive(value):
    try:
        x = float(value)
        return math.isfinite(x) and x > 0.0
    except (TypeError, ValueError):
        return False


def _price_map(rows, field):
    result = {}
    for row in rows.itertuples(index=False):
        value = getattr(row, field)
        if _finite_positive(value):
            result[row.ticker] = float(value)
    return result


def _apply_pending_orders(open_prices):
    """
    Reconstruct fills ourselves. Orders are submitted in sell-then-buy order,
    which is also the order used for this internal accounting.
    """
    global _cash, _pending_orders

    for ticker, direction, requested_size in _pending_orders:
        price = open_prices.get(ticker)
        if price is None or price <= 0.0:
            continue

        requested_size = int(requested_size)
        if requested_size <= 0:
            continue

        if direction == "SELL":
            owned = int(_positions.get(ticker, 0))
            filled = min(owned, requested_size)

            if filled > 0:
                _positions[ticker] = owned - filled
                _cash += filled * price

                if _positions[ticker] <= 0:
                    _positions.pop(ticker, None)
                    _peaks.pop(ticker, None)

        elif direction == "BUY":
            affordable = int((_cash + 1e-9) // price)
            filled = min(requested_size, affordable)

            if filled > 0:
                _cash -= filled * price
                _positions[ticker] = int(_positions.get(ticker, 0)) + filled
                _peaks[ticker] = max(float(_peaks.get(ticker, price)), price)

    _pending_orders = []


def _update_indicators(rows):
    """
    Maintain indicators incrementally rather than repeatedly recalculating
    all history.

    Indicators:
      - 20-bar fast EMA
      - 80-bar slow EMA
      - EWMA return volatility
      - 60-observation momentum
      - EWMA dollar-volume liquidity
    """
    close_prices = {}

    for row in rows.itertuples(index=False):
        ticker = row.ticker

        if not _finite_positive(row.close):
            continue

        close = float(row.close)
        volume = float(row.volume) if _finite_positive(row.volume) else 0.0
        close_prices[ticker] = close
        _last_close[ticker] = close

        s = _state.get(ticker)
        if s is None:
            s = {
                "count": 0,
                "prev_close": None,
                "ema_fast": close,
                "ema_slow": close,
                "variance": 0.0,
                "dollar_volume": close * volume,
                "closes": deque(maxlen=65),
                "momentum": None,
            }
            _state[ticker] = s

        previous = s["prev_close"]

        # Momentum is measured before adding today's close, so this compares
        # today's price with the price 60 observed bars earlier.
        if len(s["closes"]) >= 60:
            old_close = s["closes"][-60]
            if old_close > 0.0:
                s["momentum"] = close / old_close - 1.0

        s["closes"].append(close)
        s["count"] += 1

        fast_alpha = 2.0 / 21.0
        slow_alpha = 2.0 / 81.0
        vol_alpha = 0.06
        liquidity_alpha = 0.05

        s["ema_fast"] = fast_alpha * close + (1.0 - fast_alpha) * s["ema_fast"]
        s["ema_slow"] = slow_alpha * close + (1.0 - slow_alpha) * s["ema_slow"]

        if previous is not None and previous > 0.0:
            ret = close / previous - 1.0
            s["variance"] = (
                vol_alpha * ret * ret + (1.0 - vol_alpha) * s["variance"]
            )

        dollar_volume = close * volume
        s["dollar_volume"] = (
            liquidity_alpha * dollar_volume
            + (1.0 - liquidity_alpha) * s["dollar_volume"]
        )

        s["prev_close"] = close

    return close_prices


def _portfolio_equity():
    """Mark positions using the latest available adjusted close."""
    equity = _cash

    for ticker, shares in _positions.items():
        price = _last_close.get(ticker)
        if price is not None and price > 0.0:
            equity += int(shares) * price

    return max(0.0, equity)


def _select_targets():
    """
    Cross-sectional trend following.

    A stock qualifies when:
      - It has enough history for all indicators.
      - Price is above its slow trend.
      - Fast EMA is above slow EMA.
      - 60-bar momentum is positive.

    Candidates are ranked by volatility-adjusted momentum. The strategy holds
    at most eight of the most liquid qualifying names.
    """
    ready = []

    for ticker, s in _state.items():
        if s["count"] < 65:
            continue

        close = _last_close.get(ticker)
        momentum = s["momentum"]

        if close is None or momentum is None:
            continue
        if s["dollar_volume"] <= 0.0:
            continue

        ready.append((ticker, s))

    if not ready:
        return []

    # First restrict the universe to liquid names. This is relative rather
    # than relying on a fixed dollar-volume threshold, making it work across
    # datasets with different scale.
    ready.sort(key=lambda x: x[1]["dollar_volume"], reverse=True)
    liquid_count = min(120, len(ready))
    liquid = ready[:liquid_count]

    candidates = []

    for ticker, s in liquid:
        close = _last_close[ticker]
        fast = s["ema_fast"]
        slow = s["ema_slow"]
        momentum = s["momentum"]
        volatility = math.sqrt(max(s["variance"], 0.0))

        if close <= slow:
            continue
        if fast <= slow:
            continue
        if momentum <= 0.0:
            continue

        # A small volatility floor prevents near-zero historical volatility
        # from producing an unbounded score.
        score = momentum / max(volatility, 0.01)
        candidates.append((score, ticker))

    candidates.sort(reverse=True)
    return [ticker for _, ticker in candidates[:8]]


def _emergency_exit_orders(close_prices):
    """
    A wide protective stop is checked every bar. Normal portfolio changes are
    only made every five bars, reducing turnover; this exit handles unusually
    sharp reversals between scheduled rebalance dates.
    """
    orders = []

    for ticker, shares in list(_positions.items()):
        if shares <= 0:
            continue

        close = close_prices.get(ticker)
        if close is None:
            continue

        peak = max(float(_peaks.get(ticker, close)), close)
        _peaks[ticker] = peak

        s = _state.get(ticker)
        slow = s["ema_slow"] if s is not None else None

        # Exit on a 16% trailing decline, or a decisive break below trend.
        trailing_stop = close < peak * 0.84
        trend_failure = slow is not None and close < slow * 0.92

        if trailing_stop or trend_failure:
            orders.append((ticker, "SELL", int(shares)))

    return orders


def _rebalance_orders(target_tickers):
    """
    Construct a target-weight portfolio.

    Exposure is capped near 90% of marked equity, split equally between up to
    eight selected tickers. A cash buffer helps absorb overnight gaps because
    submitted orders fill at the next open.
    """
    equity = _portfolio_equity()
    if equity <= 0.0:
        return []

    target_tickers = list(target_tickers)
    if not target_tickers:
        return [
            (ticker, "SELL", int(shares))
            for ticker, shares in _positions.items()
            if shares > 0
        ]

    per_position_value = (equity * 0.90) / len(target_tickers)

    desired_shares = {}
    for ticker in target_tickers:
        price = _last_close.get(ticker)
        if price is not None and price > 0.0:
            desired_shares[ticker] = max(0, int(per_position_value // price))

    sells = []
    buys = []
    expected_sale_proceeds = 0.0

    all_tickers = set(_positions) | set(desired_shares)

    for ticker in all_tickers:
        current = int(_positions.get(ticker, 0))
        desired = int(desired_shares.get(ticker, 0))
        mark = _last_close.get(ticker, 0.0)

        if current > desired:
            quantity = current - desired
            sells.append((ticker, "SELL", quantity))
            if mark > 0.0:
                expected_sale_proceeds += quantity * mark

        elif desired > current:
            buys.append((ticker, "BUY", desired - current))

    # Do not assume every expected sale will fill perfectly. Scale purchases
    # to a conservative estimate of available post-sale cash.
    estimated_cash = _cash + 0.95 * expected_sale_proceeds
    planned_buy_cost = sum(
        qty * _last_close.get(ticker, 0.0)
        for ticker, _, qty in buys
    )

    if planned_buy_cost > 0.0:
        scale = min(1.0, max(0.0, 0.95 * estimated_cash / planned_buy_cost))
    else:
        scale = 1.0

    scaled_buys = []
    for ticker, direction, quantity in buys:
        scaled_quantity = int(quantity * scale)
        if scaled_quantity > 0:
            scaled_buys.append((ticker, direction, scaled_quantity))

    # Sales first so platforms that process returned orders in list order can
    # use sale proceeds for replacement purchases.
    return sells + scaled_buys


def on_bar(history):
    global _seen_rows, _bar_number, _pending_orders

    if history is None or len(history) == 0:
        return []

    # history is append-only and on_bar is called once per timestamp.
    # New rows are therefore the complete current cross-section.
    if _seen_rows >= len(history):
        return []

    current_rows = history.iloc[_seen_rows:]
    _seen_rows = len(history)
    _bar_number += 1

    open_prices = _price_map(current_rows, "open")

    # Orders returned at the previous timestamp execute now.
    _apply_pending_orders(open_prices)

    close_prices = _update_indicators(current_rows)

    # Update trailing peaks for all currently held securities.
    for ticker in list(_positions):
        close = close_prices.get(ticker)
        if close is not None:
            _peaks[ticker] = max(float(_peaks.get(ticker, close)), close)

    # Risk exits take priority over scheduled rebalancing.
    emergency_orders = _emergency_exit_orders(close_prices)
    if emergency_orders:
        _pending_orders = emergency_orders
        return emergency_orders

    # Rebalance every five timestamps after sufficient indicator history exists.
    if _bar_number % 5 != 0:
        return []

    targets = _select_targets()
    orders = _rebalance_orders(targets)

    _pending_orders = orders
    return orders
