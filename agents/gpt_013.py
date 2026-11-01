"""
Cross-sectional momentum / trend-following trading agent.

Strategy summary:
- Tracks each ticker incrementally without repeatedly recomputing indicators
  over the full history.
- Every 5 bars, ranks sufficiently liquid tickers by:
    * 20-bar momentum
    * 60-bar momentum
    * fast/slow EMA trend
    * volatility penalty
- Holds up to 8 names with the strongest positive trends.
- Targets roughly 90% invested, equally weighted among selected names.
- Sells names no longer selected or trims oversized positions.
- Uses conservative cash budgeting for buys because orders execute next open.
- Maintains an internal account ledger by simulating previously submitted
  orders at the following bar's open.

No files, network calls, or work at import time.
"""

from collections import deque
import math
import numpy as np


# Module-level state, initialized by init().
_cash = 0.0
_positions = {}
_pending_orders = []
_ticker_state = {}
_last_timestamp = None
_bar_count = 0


def init(cash):
    """Initialize strategy state. Called exactly once by the platform."""
    global _cash, _positions, _pending_orders
    global _ticker_state, _last_timestamp, _bar_count

    _cash = float(cash)
    _positions = {}
    _pending_orders = []
    _ticker_state = {}
    _last_timestamp = None
    _bar_count = 0


def _finite_positive(value):
    try:
        return np.isfinite(value) and value > 0.0
    except (TypeError, ValueError):
        return False


def _process_pending_orders(current_rows):
    """
    Simulate fills of orders returned on the previous bar.

    Orders execute on this bar's open. The strategy sends sells before buys,
    and simulates them in the same sequence.
    """
    global _cash, _positions, _pending_orders

    if not _pending_orders:
        return

    opens = {}

    tickers = current_rows["ticker"].to_numpy()
    open_prices = current_rows["open"].to_numpy()

    for ticker, price in zip(tickers, open_prices):
        if _finite_positive(price):
            opens[ticker] = float(price)

    for ticker, direction, size in _pending_orders:
        price = opens.get(ticker)

        # Missing ticker or invalid opening price means no fill.
        if price is None or size <= 0:
            continue

        if direction == "SELL":
            held = int(_positions.get(ticker, 0))
            filled = min(held, int(size))

            if filled > 0:
                _positions[ticker] = held - filled
                _cash += filled * price

                if _positions[ticker] <= 0:
                    _positions.pop(ticker, None)

        elif direction == "BUY":
            affordable = int(_cash // price)
            filled = min(int(size), affordable)

            if filled > 0:
                _positions[ticker] = int(_positions.get(ticker, 0)) + filled
                _cash -= filled * price

    # Orders do not persist after their attempted execution bar.
    _pending_orders = []


def _update_indicators(current_rows):
    """
    Update ticker indicator state from the latest completed bar.

    Each ticker stores only a small bounded close-price history plus EMA,
    volatility, and dollar-volume estimates.
    """
    closes = current_rows["close"].to_numpy()
    volumes = current_rows["volume"].to_numpy()
    tickers = current_rows["ticker"].to_numpy()

    for ticker, close, volume in zip(tickers, closes, volumes):
        if not _finite_positive(close):
            continue

        close = float(close)
        volume = float(volume) if _finite_positive(volume) else 0.0

        state = _ticker_state.get(ticker)

        if state is None:
            state = {
                "closes": deque(maxlen=130),
                "ema_fast": close,
                "ema_slow": close,
                "vol_var": 0.0,
                "dollar_volume_ema": close * volume,
                "last_close": close,
                "bars": 0,
            }
            _ticker_state[ticker] = state
        else:
            previous_close = state["last_close"]

            if _finite_positive(previous_close):
                log_return = math.log(close / previous_close)

                # EWMA volatility estimate. Lambda close to 0.94 is common
                # for daily return volatility models.
                state["vol_var"] = (
                    0.94 * state["vol_var"] + 0.06 * log_return * log_return
                )

            # Fast and slow trend estimates.
            state["ema_fast"] = 0.90 * state["ema_fast"] + 0.10 * close
            state["ema_slow"] = 0.975 * state["ema_slow"] + 0.025 * close

            # Approximate rolling dollar-volume liquidity.
            dollar_volume = close * volume
            state["dollar_volume_ema"] = (
                0.90 * state["dollar_volume_ema"] + 0.10 * dollar_volume
            )

            state["last_close"] = close

        state["closes"].append(close)
        state["bars"] += 1


def _estimate_equity():
    """Estimate account equity using latest known closing prices."""
    equity = float(_cash)

    for ticker, shares in _positions.items():
        state = _ticker_state.get(ticker)
        if state is not None:
            price = state.get("last_close", 0.0)
            if _finite_positive(price):
                equity += int(shares) * float(price)

    return max(equity, 0.0)


def _select_candidates(current_rows):
    """
    Return selected tickers in descending score order.

    Uses only tickers available on the current bar, so proposed trades are
    more likely to have an execution price on the next bar.
    """
    current_tickers = current_rows["ticker"].to_numpy()

    candidates = []

    for ticker in current_tickers:
        state = _ticker_state.get(ticker)
        if state is None:
            continue

        closes = state["closes"]

        # Need enough observations for a meaningful 60-bar momentum signal.
        if len(closes) < 65:
            continue

        close = state["last_close"]
        price_20_bars_ago = closes[-21]
        price_60_bars_ago = closes[-61]

        if (
            not _finite_positive(close)
            or not _finite_positive(price_20_bars_ago)
            or not _finite_positive(price_60_bars_ago)
        ):
            continue

        ret_20 = close / price_20_bars_ago - 1.0
        ret_60 = close / price_60_bars_ago - 1.0

        ema_fast = state["ema_fast"]
        ema_slow = state["ema_slow"]

        if not _finite_positive(ema_fast) or not _finite_positive(ema_slow):
            continue

        trend = ema_fast / ema_slow - 1.0
        volatility = math.sqrt(max(state["vol_var"], 0.0))
        dollar_volume = state["dollar_volume_ema"]

        candidates.append(
            {
                "ticker": ticker,
                "ret_20": ret_20,
                "ret_60": ret_60,
                "trend": trend,
                "volatility": volatility,
                "dollar_volume": dollar_volume,
                "price": close,
            }
        )

    if not candidates:
        return []

    liquidities = np.array(
        [item["dollar_volume"] for item in candidates],
        dtype=float,
    )
    median_liquidity = float(np.nanmedian(liquidities))

    # Dynamic liquidity screen. The absolute floor avoids trading extremely
    # illiquid assets if the full universe itself is illiquid.
    liquidity_floor = max(100000.0, median_liquidity * 0.02)

    scored = []

    for item in candidates:
        if item["dollar_volume"] < liquidity_floor:
            continue

        # Avoid strongly negative-trend names. This allows the strategy to
        # move partly or completely to cash in broadly weak markets.
        if item["ret_60"] < -0.12:
            continue

        if item["trend"] < -0.03:
            continue

        score = (
            0.55 * item["ret_20"]
            + 0.35 * item["ret_60"]
            + 0.10 * item["trend"]
            - 0.35 * item["volatility"]
        )

        item["score"] = score
        scored.append(item)

    scored.sort(key=lambda item: item["score"], reverse=True)

    # Prefer positive short-term momentum. If markets are weak, retain fewer
    # names instead of forcing exposure.
    selected = []

    for item in scored:
        if item["ret_20"] < -0.02:
            continue

        selected.append(item)

        if len(selected) >= 8:
            break

    return selected


def _build_orders(current_rows):
    """
    Create the next-bar order list.

    Sells are emitted before buys. Buy sizing uses only currently tracked cash,
    not expected proceeds from newly submitted sells. This is intentionally
    conservative because no fill reports are provided.
    """
    selected = _select_candidates(current_rows)
    selected_by_ticker = {item["ticker"]: item for item in selected}
    selected_tickers = set(selected_by_ticker)

    equity = _estimate_equity()

    if equity <= 0.0:
        return []

    orders = []
    estimated_sale_value = 0.0

    # If no acceptable trend candidates exist, move to cash where possible.
    if not selected:
        for ticker, shares in list(_positions.items()):
            state = _ticker_state.get(ticker)

            if state is None:
                continue

            price = state.get("last_close", 0.0)

            if int(shares) > 0 and _finite_positive(price):
                orders.append((ticker, "SELL", int(shares)))

        return orders

    # Maintain about 90% target exposure, equally split among selected names.
    target_fraction = 0.90 / len(selected)

    # Sell names no longer selected and trim names materially above target.
    for ticker, shares in list(_positions.items()):
        shares = int(shares)

        if shares <= 0:
            continue

        state = _ticker_state.get(ticker)

        if state is None:
            continue

        price = state.get("last_close", 0.0)

        if not _finite_positive(price):
            continue

        if ticker not in selected_tickers:
            orders.append((ticker, "SELL", shares))
            estimated_sale_value += shares * price
            continue

        desired_value = equity * target_fraction
        desired_shares = int(desired_value // price)

        # Rebalance only when the position is meaningfully oversized to reduce
        # churn from small day-to-day price movements.
        if shares > int(desired_shares * 1.08) + 1:
            sell_size = shares - desired_shares
            orders.append((ticker, "SELL", sell_size))
            estimated_sale_value += sell_size * price

    # Buy budget is deliberately based only on known cash. A 10% reserve
    # provides some protection against next-open gaps.
    buy_budget = max(0.0, _cash * 0.90)

    # Score order determines which names get priority if cash is limited.
    for item in selected:
        ticker = item["ticker"]
        price = item["price"]
        held = int(_positions.get(ticker, 0))

        if not _finite_positive(price):
            continue

        desired_value = equity * target_fraction
        desired_shares = int(desired_value // price)

        # Only add when a position is materially underweight.
        if held >= int(desired_shares * 0.85):
            continue

        requested = desired_shares - held

        if requested <= 0:
            continue

        # Use a further gap buffer while sizing shares.
        affordable = int(buy_budget // (price * 1.03))
        buy_size = min(requested, affordable)

        if buy_size <= 0:
            continue

        orders.append((ticker, "BUY", int(buy_size)))
        buy_budget -= buy_size * price

        if buy_budget <= 0.0:
            break

    return orders


def on_bar(history):
    """
    Called once for every timestamp.

    Returns a list of:
        (ticker, "BUY" or "SELL", positive_integer_share_count)
    """
    global _pending_orders, _last_timestamp, _bar_count

    if history is None or len(history) == 0:
        return []

    timestamps = history["timestamp"].to_numpy()
    current_timestamp = timestamps[-1]

    # Defensive guard in case the caller invokes the function more than once
    # with the same history timestamp.
    if _last_timestamp is not None and current_timestamp == _last_timestamp:
        return []

    # History is sorted by timestamp, so searchsorted isolates just the rows
    # belonging to the latest timestamp without scanning all past history.
    start = int(np.searchsorted(timestamps, current_timestamp, side="left"))
    current_rows = history.iloc[start:]

    # First process fills for orders submitted on the prior bar.
    _process_pending_orders(current_rows)

    # Then incorporate the newly completed close data into indicators.
    _update_indicators(current_rows)

    _last_timestamp = current_timestamp
    _bar_count += 1

    # Wait for enough indicator history and rebalance periodically.
    if _bar_count < 65:
        return []

    if _bar_count % 5 != 0:
        return []

    orders = _build_orders(current_rows)

    # Store exactly what is returned so next bar's internal account simulation
    # matches the submitted order sequence.
    _pending_orders = list(orders)

    return orders
