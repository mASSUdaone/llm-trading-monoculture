import math
from collections import deque

import numpy as np
import pandas as pd


# Strategy state. Everything is reset by init().
_cash = 0.0
_positions = {}
_pending_orders = []
_last_close = {}
_series = {}
_bar_number = 0
_initialized_history = False
_has_traded = False


def init(cash):
    """
    Called once by the platform before the first bar.
    """
    global _cash, _positions, _pending_orders, _last_close
    global _series, _bar_number, _initialized_history, _has_traded

    _cash = float(cash)
    _positions = {}
    _pending_orders = []
    _last_close = {}
    _series = {}
    _bar_number = 0
    _initialized_history = False
    _has_traded = False


def _valid_price(x):
    try:
        return x is not None and np.isfinite(float(x)) and float(x) > 0.0
    except (TypeError, ValueError):
        return False


def _get_ticker_state(ticker):
    state = _series.get(ticker)
    if state is None:
        state = {
            # Enough data for 60-day momentum, 50-day trend, and volatility.
            "close": deque(maxlen=130),
            "volume": deque(maxlen=130),
        }
        _series[ticker] = state
    return state


def _update_indicators_from_rows(rows):
    """
    Add one timestamp's closing data to each ticker's rolling history.
    """
    global _bar_number

    for ticker, close, volume in rows[["ticker", "close", "volume"]].itertuples(
        index=False, name=None
    ):
        if not _valid_price(close):
            continue

        close = float(close)
        try:
            volume = float(volume)
            if not np.isfinite(volume) or volume < 0:
                volume = 0.0
        except (TypeError, ValueError):
            volume = 0.0

        state = _get_ticker_state(ticker)
        state["close"].append(close)
        state["volume"].append(volume)
        _last_close[ticker] = close

    _bar_number += 1


def _apply_pending_orders(current_rows):
    """
    The platform fills orders submitted on the preceding bar at this bar's open.
    We maintain a conservative internal ledger because the platform does not
    provide fill reports.
    """
    global _cash, _pending_orders

    if not _pending_orders:
        return

    open_prices = {}
    for ticker, open_price in current_rows[["ticker", "open"]].itertuples(
        index=False, name=None
    ):
        if _valid_price(open_price):
            open_prices[ticker] = float(open_price)

    # Orders are intentionally returned in SELL-then-BUY order, and accounting
    # here follows that same order.
    for ticker, direction, requested_size in _pending_orders:
        price = open_prices.get(ticker)
        if price is None:
            # The specification says such an order fills zero shares.
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

        elif direction == "BUY":
            affordable = int(math.floor((_cash + 1e-9) / price))
            filled = min(requested_size, max(0, affordable))
            if filled > 0:
                _positions[ticker] = int(_positions.get(ticker, 0)) + filled
                _cash -= filled * price

        _last_close[ticker] = price

    _pending_orders = []


def _score_ticker(ticker):
    """
    Cross-sectional score combining:
      - 20-day momentum
      - 60-day momentum
      - 50-day trend filter
      - inverse recent volatility
      - basic dollar-volume liquidity filter

    Returns None when a ticker is not eligible.
    """
    state = _series.get(ticker)
    if state is None:
        return None

    closes = state["close"]
    volumes = state["volume"]

    if len(closes) < 65 or len(volumes) < 20:
        return None

    c = np.asarray(closes, dtype=float)
    v = np.asarray(volumes, dtype=float)

    current = c[-1]
    if current < 2.0:
        return None

    sma_50 = float(np.mean(c[-50:]))
    if current <= sma_50:
        return None

    momentum_20 = current / c[-21] - 1.0
    momentum_60 = current / c[-61] - 1.0

    # Long-only strategy: require both medium and short horizon momentum.
    if momentum_20 <= 0.0 or momentum_60 <= -0.03:
        return None

    returns = c[-21:] / c[-22:-1] - 1.0
    volatility = float(np.std(returns, ddof=1))

    if not np.isfinite(volatility) or volatility <= 0.0:
        return None

    median_dollar_volume = float(np.median(v[-20:] * c[-20:]))
    if median_dollar_volume < 200000.0:
        return None

    # A volatility floor prevents extremely quiet names from dominating.
    adjusted_volatility = max(volatility, 0.012)

    # Reward both recent leadership and sustained medium-term trend.
    raw_return = 0.70 * momentum_20 + 0.30 * momentum_60
    score = raw_return / adjusted_volatility

    if not np.isfinite(score):
        return None

    return float(score)


def _make_rebalance_orders(current_rows):
    """
    Select the strongest eligible names and target an approximately equal-weight
    long-only portfolio. A 10% portfolio cash reserve is retained.

    Sales are placed first. Purchases only use cash already known to exist,
    rather than assuming sale proceeds will arrive at a favorable next open.
    """
    current_prices = {}
    current_tickers = set()

    for ticker, close in current_rows[["ticker", "close"]].itertuples(
        index=False, name=None
    ):
        current_tickers.add(ticker)
        if _valid_price(close):
            current_prices[ticker] = float(close)

    scored = []
    for ticker in current_tickers:
        score = _score_ticker(ticker)
        if score is not None:
            scored.append((score, ticker))

    scored.sort(reverse=True, key=lambda x: x[0])

    # Five names is diversified enough for a cross-sectional momentum system,
    # while still concentrating capital in the strongest trends.
    selected = scored[:5]
    selected_tickers = {ticker for _, ticker in selected}

    # Mark positions using the latest known close for tickers absent today.
    equity = float(_cash)
    for ticker, shares in _positions.items():
        mark = _last_close.get(ticker)
        if _valid_price(mark):
            equity += int(shares) * float(mark)

    if equity <= 0:
        return []

    if selected:
        target_value_per_name = 0.90 * equity / len(selected)
    else:
        target_value_per_name = 0.0

    target_shares = {}
    for _, ticker in selected:
        price = current_prices.get(ticker)
        if price is not None and price > 0.0:
            target_shares[ticker] = int(math.floor(target_value_per_name / price))

    sell_orders = []
    buy_candidates = []

    # Exit names that lost eligibility and trim oversized holdings.
    for ticker, held in list(_positions.items()):
        held = int(held)
        if held <= 0:
            continue

        desired = int(target_shares.get(ticker, 0))

        # If a ticker is absent today, still request liquidation. It will fill
        # next bar only if that ticker has an available execution price.
        if ticker not in selected_tickers:
            sell_orders.append((ticker, "SELL", held))
        elif held > desired:
            sell_orders.append((ticker, "SELL", held - desired))

    # Do not assume today's sales will fill tomorrow. Buy allocations therefore
    # use only currently known cash, less a modest cash reserve.
    cash_reserve = 0.05 * equity
    buy_budget = max(0.0, _cash - cash_reserve)

    for score, ticker in selected:
        desired = int(target_shares.get(ticker, 0))
        held = int(_positions.get(ticker, 0))
        needed = desired - held
        if needed > 0:
            buy_candidates.append((score, ticker, needed))

    buy_orders = []
    for _, ticker, needed in buy_candidates:
        price = current_prices.get(ticker)
        if price is None or price <= 0.0 or buy_budget < price:
            continue

        affordable = int(math.floor((buy_budget + 1e-9) / price))
        quantity = min(needed, affordable)

        if quantity > 0:
            buy_orders.append((ticker, "BUY", quantity))
            buy_budget -= quantity * price

    return sell_orders + buy_orders


def on_bar(history):
    """
    Called once per timestamp. Returns orders that will execute on the next
    timestamp's open.
    """
    global _initialized_history, _pending_orders, _has_traded

    if history is None or history.empty:
        return []

    # Rows are sorted by timestamp then ticker. The last timestamp is the
    # currently tradable bar.
    timestamps = history["timestamp"]
    current_timestamp = timestamps.iloc[-1]

    # Find the first row of the current timestamp without scanning/grouping the
    # whole history on every call.
    values = timestamps.to_numpy()
    try:
        start = int(np.searchsorted(values, current_timestamp, side="left"))
    except TypeError:
        # Fallback for unusual object timestamp dtypes.
        start = len(history) - 1
        while start > 0 and timestamps.iloc[start - 1] == current_timestamp:
            start -= 1

    current_rows = history.iloc[start:]

    # Normally the first call contains only one timestamp. This bootstrap also
    # handles platforms that provide a preloaded historical warm-up dataset.
    if not _initialized_history:
        if start > 0:
            prior = history.iloc[:start]
            prior_timestamp = prior["timestamp"]

            # Process historical timestamps for indicator warm-up only; no
            # hypothetical orders are generated for these old bars.
            boundaries = np.flatnonzero(
                prior_timestamp.ne(prior_timestamp.shift()).to_numpy()
            )

            for i, begin in enumerate(boundaries):
                end = boundaries[i + 1] if i + 1 < len(boundaries) else len(prior)
                _update_indicators_from_rows(prior.iloc[begin:end])

        _initialized_history = True

    # First reconcile orders submitted on the previous timestamp at this bar's
    # opening price, then incorporate this bar's close into the signal.
    _apply_pending_orders(current_rows)
    _update_indicators_from_rows(current_rows)

    # Rebalance every five bars. The first eligible opportunity always trades.
    rebalance_now = (not _has_traded) or (_bar_number % 5 == 0)
    if not rebalance_now:
        return []

    orders = _make_rebalance_orders(current_rows)
    _pending_orders = list(orders)
    _has_traded = True

    return orders
