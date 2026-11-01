import numpy as np
from collections import deque

# Module-level state. init() resets all of this before a run.
_cash = 0.0
_positions = {}
_pending_orders = []
_symbol_state = {}
_last_close = {}
_last_timestamp = None
_bar_number = 0
_initialized = False

MAX_POSITIONS = 8
TARGET_GROSS_EXPOSURE = 0.82
REBALANCE_EVERY_BARS = 5
MIN_HISTORY_BARS = 65


def init(cash):
    """Initialize the agent's internal account and signal state."""
    global _cash, _positions, _pending_orders, _symbol_state
    global _last_close, _last_timestamp, _bar_number, _initialized

    _cash = float(cash)
    _positions = {}
    _pending_orders = []
    _symbol_state = {}
    _last_close = {}
    _last_timestamp = None
    _bar_number = 0
    _initialized = True


def _valid_number(value):
    try:
        return np.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _update_symbol(ticker, close, volume):
    """Update rolling indicators for one symbol and return its state."""
    state = _symbol_state.get(ticker)

    if state is None:
        state = {
            "closes": deque(maxlen=126),
            "returns": deque(maxlen=21),
            "count": 0,
            "last_close": None,
            "ema_fast": None,   # approximately 15 bars
            "ema_slow": None,   # approximately 50 bars
            "dollar_volume": None,
        }
        _symbol_state[ticker] = state

    close = float(close)
    volume = float(volume) if _valid_number(volume) else 0.0

    previous = state["last_close"]
    if previous is not None and previous > 0:
        ret = close / previous - 1.0
        if np.isfinite(ret):
            state["returns"].append(ret)

    # Exponential moving averages.
    fast_alpha = 2.0 / 16.0
    slow_alpha = 2.0 / 51.0

    if state["ema_fast"] is None:
        state["ema_fast"] = close
        state["ema_slow"] = close
    else:
        state["ema_fast"] = fast_alpha * close + (1.0 - fast_alpha) * state["ema_fast"]
        state["ema_slow"] = slow_alpha * close + (1.0 - slow_alpha) * state["ema_slow"]

    dollar_volume = max(0.0, close * volume)
    if state["dollar_volume"] is None:
        state["dollar_volume"] = dollar_volume
    else:
        # A roughly 20-bar liquidity estimate.
        state["dollar_volume"] = 0.10 * dollar_volume + 0.90 * state["dollar_volume"]

    state["closes"].append(close)
    state["last_close"] = close
    state["count"] += 1
    _last_close[ticker] = close

    return state


def _execute_pending_orders(current_open_prices):
    """
    Reconstruct fills from the prior bar's submitted orders.

    Orders are submitted in sell-then-buy order. This function intentionally
    processes them in that same order so internal cash and positions track the
    platform's expected behavior as closely as possible.
    """
    global _cash, _positions, _pending_orders

    if not _pending_orders:
        return

    for ticker, direction, size in _pending_orders:
        price = current_open_prices.get(ticker)

        if price is None or not _valid_number(price):
            continue

        price = float(price)
        if price <= 0:
            continue

        requested = int(size)
        if requested <= 0:
            continue

        if direction == "SELL":
            held = int(_positions.get(ticker, 0))
            filled = min(held, requested)

            if filled > 0:
                _positions[ticker] = held - filled
                _cash += filled * price
                if _positions[ticker] <= 0:
                    _positions.pop(ticker, None)

        elif direction == "BUY":
            affordable = int(np.floor((_cash + 1e-9) / price))
            filled = min(requested, max(0, affordable))

            if filled > 0:
                _positions[ticker] = int(_positions.get(ticker, 0)) + filled
                _cash -= filled * price

    # Orders not filled at this execution opportunity are discarded.
    _pending_orders = []


def _portfolio_value():
    """Mark the internal portfolio using the latest available close per ticker."""
    value = _cash

    for ticker, shares in _positions.items():
        price = _last_close.get(ticker)
        if price is not None and _valid_number(price) and float(price) > 0:
            value += int(shares) * float(price)

    return max(0.0, value)


def _signal_for_symbol(ticker, close):
    """
    Return (score, liquidity) for a tradable momentum candidate, or None.

    The signal favors:
    - positive medium-term momentum,
    - positive shorter-term momentum,
    - trading above a medium-term trend,
    - lower realized volatility,
    - reasonably liquid names relative to the current universe.
    """
    state = _symbol_state.get(ticker)
    if state is None or state["count"] < MIN_HISTORY_BARS:
        return None

    closes = state["closes"]
    if len(closes) < 61 or len(state["returns"]) < 15:
        return None

    close_20_bars_ago = closes[-21]
    close_60_bars_ago = closes[-61]

    if close_20_bars_ago <= 0 or close_60_bars_ago <= 0:
        return None

    mom_20 = close / close_20_bars_ago - 1.0
    mom_60 = close / close_60_bars_ago - 1.0

    returns = np.asarray(state["returns"], dtype=float)
    volatility = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    volatility = max(volatility, 0.01)

    slow = state["ema_slow"]
    fast = state["ema_fast"]

    if slow is None or fast is None or slow <= 0:
        return None

    # Long-only trend filter. Requiring positive medium-term momentum avoids
    # holding persistent downtrends merely because they have low volatility.
    if close <= slow:
        return None
    if fast < slow:
        return None
    if mom_60 <= 0.0:
        return None
    if mom_20 <= -0.08:
        return None

    trend_strength = close / slow - 1.0

    # Momentum adjusted for recent volatility. Trend strength receives a small
    # additional weight to distinguish names with similar raw momentum.
    score = (
        0.70 * mom_60 / volatility
        + 0.25 * mom_20 / volatility
        + 0.05 * trend_strength / volatility
    )

    liquidity = float(state["dollar_volume"] or 0.0)
    return score, liquidity


def on_bar(history):
    """
    Generate next-open orders.

    Strategy:
    - Maintain a cross-sectional, long-only momentum portfolio.
    - Rebalance every five market bars into up to eight trend-confirmed names.
    - Keep roughly 82% of marked equity invested, equally weighted.
    - Exit existing positions early when they break their slow trend.
    - Submit sells before buys to make sale proceeds available first.
    """
    global _last_timestamp, _bar_number, _pending_orders

    if not _initialized:
        raise RuntimeError("init(cash) must be called before on_bar(history).")

    if history is None or len(history) == 0:
        return []

    current_timestamp = history["timestamp"].iloc[-1]

    # Defensive behavior if the platform were to call the same timestamp twice.
    if _last_timestamp is not None and current_timestamp == _last_timestamp:
        return []

    # Rows are sorted by timestamp and ticker. Find only the newly current bar
    # without scanning the entire table with a boolean mask.
    timestamps = history["timestamp"].to_numpy()
    start = int(np.searchsorted(timestamps, current_timestamp, side="left"))
    current_rows = history.iloc[start:]

    # First reconcile orders submitted on the previous timestamp. Their fills
    # occur at this bar's open.
    open_prices = {}
    for row in current_rows.itertuples(index=False):
        ticker = row.ticker
        if _valid_number(row.open) and float(row.open) > 0:
            open_prices[ticker] = float(row.open)

    _execute_pending_orders(open_prices)

    _bar_number += 1
    _last_timestamp = current_timestamp

    current_closes = {}
    candidates = []

    # Update indicators using information available at the current close.
    for row in current_rows.itertuples(index=False):
        ticker = row.ticker

        if not _valid_number(row.close):
            continue

        close = float(row.close)
        if close <= 0:
            continue

        volume = row.volume if _valid_number(row.volume) else 0.0
        _update_symbol(ticker, close, volume)
        current_closes[ticker] = close

        signal = _signal_for_symbol(ticker, close)
        if signal is not None:
            score, liquidity = signal
            candidates.append((score, liquidity, ticker))

    orders = []

    # Daily risk management: sell held symbols that are present today and have
    # fallen materially below their slow trend. These exits do not wait for the
    # scheduled rebalance.
    forced_exits = set()
    for ticker, shares in list(_positions.items()):
        if shares <= 0 or ticker not in current_closes:
            continue

        state = _symbol_state.get(ticker)
        if state is None or state["ema_slow"] is None:
            continue

        close = current_closes[ticker]
        slow = float(state["ema_slow"])

        if slow > 0 and close < 0.955 * slow:
            forced_exits.add(ticker)

    rebalance_now = (
        _bar_number >= MIN_HISTORY_BARS
        and _bar_number % REBALANCE_EVERY_BARS == 0
    )

    if not rebalance_now:
        for ticker in sorted(forced_exits):
            shares = int(_positions.get(ticker, 0))
            if shares > 0:
                orders.append((ticker, "SELL", shares))

        _pending_orders = list(orders)
        return orders

    # Keep only the more liquid portion of today's eligible universe. The
    # percentile threshold adapts to the instruments supplied by the platform.
    if candidates:
        liquid_values = np.asarray(
            [liquidity for _, liquidity, _ in candidates if liquidity > 0],
            dtype=float,
        )

        if len(liquid_values) >= 5:
            liquidity_floor = float(np.percentile(liquid_values, 30))
        else:
            liquidity_floor = 0.0

        candidates = [
            item for item in candidates
            if item[1] >= liquidity_floor
        ]

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = [ticker for _, _, ticker in candidates[:MAX_POSITIONS]]
    selected_set = set(selected)

    # A forced exit overrides selection for this rebalance.
    selected_set.difference_update(forced_exits)
    selected = [ticker for ticker in selected if ticker in selected_set]

    equity = _portfolio_value()
    target_per_position = (
        TARGET_GROSS_EXPOSURE * equity / len(selected)
        if selected else 0.0
    )

    target_shares = {}
    for ticker in selected:
        close = current_closes.get(ticker)
        if close is None or close <= 0:
            continue
        target_shares[ticker] = max(0, int(np.floor(target_per_position / close)))

    # Submit all sells first. Positions absent from the current bar cannot be
    # sold because they have no current execution price.
    sell_orders = []
    for ticker, held in list(_positions.items()):
        held = int(held)
        if held <= 0 or ticker not in current_closes:
            continue

        desired = int(target_shares.get(ticker, 0))
        sell_size = max(0, held - desired)

        if ticker in forced_exits:
            sell_size = held

        if sell_size > 0:
            sell_orders.append((ticker, "SELL", sell_size))

    # Then purchase underweight selected names. The 18% cash reserve and the
    # sell-first ordering reduce the chance that ordinary overnight gaps cause
    # partial fills, while the platform's affordability rule remains respected.
    buy_orders = []
    for ticker in selected:
        desired = int(target_shares.get(ticker, 0))
        held = int(_positions.get(ticker, 0))
        buy_size = max(0, desired - held)

        if buy_size > 0:
            buy_orders.append((ticker, "BUY", buy_size))

    orders = sell_orders + buy_orders
    _pending_orders = list(orders)

    return orders
