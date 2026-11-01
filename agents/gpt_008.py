from collections import defaultdict, deque
import math
import numpy as np


# Portfolio/account state
_cash = 0.0
_positions = defaultdict(int)
_last_price = {}
_peak_close = {}

# Market-data state
_close_history = defaultdict(lambda: deque(maxlen=170))
_volume_history = defaultdict(lambda: deque(maxlen=30))

# Pending orders submitted on the prior bar.  These are simulated at the
# current bar's open, matching the platform's execution convention.
_pending_orders = []

# History bookkeeping
_last_history_length = 0
_bar_number = 0


def init(cash):
    """
    Initialize the strategy state.

    Called exactly once by the platform before the first bar.
    """
    global _cash, _positions, _last_price, _peak_close
    global _close_history, _volume_history
    global _pending_orders, _last_history_length, _bar_number

    _cash = float(cash)
    _positions = defaultdict(int)
    _last_price = {}
    _peak_close = {}

    _close_history = defaultdict(lambda: deque(maxlen=170))
    _volume_history = defaultdict(lambda: deque(maxlen=30))

    _pending_orders = []
    _last_history_length = 0
    _bar_number = 0


def _valid_price(value):
    try:
        value = float(value)
        return math.isfinite(value) and value > 0.0
    except (TypeError, ValueError):
        return False


def _simulate_pending_fills(current_rows):
    """
    Update internal cash and positions by reproducing next-open fills for
    orders submitted on the prior call.
    """
    global _cash, _pending_orders

    if not _pending_orders:
        return

    open_prices = {}
    for row in current_rows:
        ticker = row["ticker"]
        price = row["open"]
        if _valid_price(price):
            open_prices[ticker] = float(price)

    # Orders are processed in returned order.  The strategy intentionally
    # emits sells first, then buys.
    for ticker, direction, size in _pending_orders:
        price = open_prices.get(ticker)

        # Platform discards an order if no execution price is available.
        if price is None or size <= 0:
            continue

        if direction == "SELL":
            filled = min(int(size), int(_positions.get(ticker, 0)))
            if filled > 0:
                _positions[ticker] -= filled
                _cash += filled * price

                if _positions[ticker] <= 0:
                    _positions[ticker] = 0
                    _peak_close.pop(ticker, None)

        elif direction == "BUY":
            affordable = int(_cash // price)
            filled = min(int(size), affordable)

            if filled > 0:
                _positions[ticker] += filled
                _cash -= filled * price

    _pending_orders = []


def _portfolio_value():
    """
    Estimate account equity using the latest known close for each holding.
    """
    value = _cash

    for ticker, shares in _positions.items():
        if shares <= 0:
            continue

        price = _last_price.get(ticker)
        if _valid_price(price):
            value += shares * float(price)

    return value


def _ticker_signal(ticker):
    """
    Return a tuple (score, volatility) for a ticker, or None if there is
    insufficient valid history.

    Score combines medium-term momentum with a volatility penalty.
    """
    closes = _close_history[ticker]
    volumes = _volume_history[ticker]

    # Need enough observations for 6-month momentum and a 100-bar trend.
    if len(closes) < 127 or len(volumes) < 20:
        return None

    prices = np.asarray(closes, dtype=float)
    vols = np.asarray(volumes, dtype=float)

    if not np.all(np.isfinite(prices)) or np.any(prices <= 0):
        return None

    current_price = prices[-1]
    price_63_bars_ago = prices[-64]
    price_126_bars_ago = prices[-127]

    if price_63_bars_ago <= 0 or price_126_bars_ago <= 0:
        return None

    momentum_63 = current_price / price_63_bars_ago - 1.0
    momentum_126 = current_price / price_126_bars_ago - 1.0

    sma_100 = float(np.mean(prices[-100:]))
    if current_price <= sma_100:
        return None

    # Require positive intermediate and medium-term momentum.
    if momentum_63 <= 0.0 or momentum_126 <= 0.0:
        return None

    recent_prices = prices[-61:]
    returns = np.diff(recent_prices) / recent_prices[:-1]
    returns = returns[np.isfinite(returns)]

    if len(returns) < 40:
        return None

    volatility = float(np.std(returns, ddof=1))
    if not math.isfinite(volatility) or volatility <= 0.0005:
        return None

    # Avoid thin or suddenly inactive instruments.
    average_volume = float(np.mean(vols[-20:]))
    current_volume = float(vols[-1])

    if (
        not math.isfinite(average_volume)
        or average_volume <= 0
        or not math.isfinite(current_volume)
        or current_volume < 0.5 * average_volume
    ):
        return None

    # Favor strong momentum but penalize unstable instruments.
    combined_momentum = 0.60 * momentum_63 + 0.40 * momentum_126
    score = combined_momentum / (volatility * math.sqrt(63.0))

    if not math.isfinite(score) or score <= 0:
        return None

    return score, volatility


def _select_targets(current_tickers):
    """
    Select the strongest candidates and return desired portfolio weights.
    """
    candidates = []

    for ticker in current_tickers:
        signal = _ticker_signal(ticker)
        if signal is None:
            continue

        score, volatility = signal
        candidates.append((score, volatility, ticker))

    candidates.sort(reverse=True, key=lambda x: x[0])

    # Concentrated enough to matter, diversified enough to avoid one-name risk.
    selected = candidates[:5]

    if not selected:
        return {}

    # Inverse-volatility allocation.  Very low vol values are capped so that
    # a single low-volatility name cannot dominate the portfolio.
    raw_weights = []
    for _, volatility, ticker in selected:
        raw_weight = min(1.0 / max(volatility, 0.005), 200.0)
        raw_weights.append((ticker, raw_weight))

    total_raw_weight = sum(weight for _, weight in raw_weights)
    if total_raw_weight <= 0:
        return {}

    return {
        ticker: weight / total_raw_weight
        for ticker, weight in raw_weights
    }


def on_bar(history):
    """
    Process the current bar and return a list of orders:
        [(ticker, "BUY" or "SELL", positive_integer_share_count), ...]

    Orders returned now are assumed to execute on the following bar's open.
    """
    global _last_history_length, _bar_number, _pending_orders

    if history is None or len(history) == 0:
        return []

    # Since history is append-only and the platform calls once per time step,
    # rows added since the prior call are the current timestamp's rows.
    start = min(_last_history_length, len(history))
    new_data = history.iloc[start:]
    _last_history_length = len(history)

    if len(new_data) == 0:
        return []

    current_rows = []
    current_tickers = set()

    for row in new_data.itertuples(index=False):
        ticker = getattr(row, "ticker")
        current_tickers.add(ticker)

        current_rows.append(
            {
                "ticker": ticker,
                "open": getattr(row, "open"),
                "close": getattr(row, "close"),
                "volume": getattr(row, "volume"),
            }
        )

    # First account for fills of orders sent on the prior bar.
    _simulate_pending_fills(current_rows)

    # Incorporate current close and volume data.
    for row in current_rows:
        ticker = row["ticker"]
        close = row["close"]
        volume = row["volume"]

        if _valid_price(close):
            close = float(close)
            _close_history[ticker].append(close)
            _last_price[ticker] = close

            if _positions.get(ticker, 0) > 0:
                old_peak = _peak_close.get(ticker, close)
                _peak_close[ticker] = max(old_peak, close)

        try:
            volume = float(volume)
            if not math.isfinite(volume) or volume < 0:
                volume = 0.0
        except (TypeError, ValueError):
            volume = 0.0

        _volume_history[ticker].append(volume)

    _bar_number += 1

    # Need a substantial history before the momentum model can operate.
    if _bar_number < 127:
        return []

    sell_orders = []
    buy_orders = []

    # Daily risk exits.  These are based on current-bar closing information
    # and become sell orders for next-bar open.
    forced_exits = set()

    for ticker, shares in list(_positions.items()):
        if shares <= 0:
            continue

        closes = _close_history.get(ticker)
        current_price = _last_price.get(ticker)

        if closes is None or len(closes) < 50 or not _valid_price(current_price):
            continue

        prices = np.asarray(closes, dtype=float)
        sma_50 = float(np.mean(prices[-50:]))
        peak = _peak_close.get(ticker, float(current_price))

        # 12% trailing close stop, or a meaningful break below the 50-bar trend.
        trailing_stop = float(current_price) < 0.88 * peak
        trend_stop = float(current_price) < 0.96 * sma_50

        if trailing_stop or trend_stop:
            forced_exits.add(ticker)

    # Rebalance every five bars.  On other bars only risk exits are used.
    should_rebalance = (_bar_number % 5 == 0)

    if should_rebalance:
        desired_weights = _select_targets(current_tickers)
        equity = _portfolio_value()

        # Keep an intentional cash reserve for overnight gaps and imperfect
        # target-price estimates.
        invested_fraction = 0.92
        desired_shares = {}

        for ticker, weight in desired_weights.items():
            price = _last_price.get(ticker)

            if not _valid_price(price):
                continue

            target_dollars = equity * invested_fraction * weight
            desired_shares[ticker] = int(target_dollars // float(price))

        # Risk-exited positions should not be repurchased this rebalance.
        for ticker in forced_exits:
            desired_shares[ticker] = 0

        # Sell positions no longer selected, risk-exited positions, and excess
        # shares in positions whose target allocation has declined.
        for ticker, shares in list(_positions.items()):
            if shares <= 0:
                continue

            target = desired_shares.get(ticker, 0)
            if ticker in forced_exits:
                target = 0

            amount_to_sell = shares - target
            if amount_to_sell > 0:
                sell_orders.append((ticker, "SELL", int(amount_to_sell)))

        # Buy after sells.  The platform may partially fill purchases if next
        # opening prices gap upward or sale proceeds differ from estimates.
        for ticker, target in desired_shares.items():
            if ticker in forced_exits:
                continue

            current_shares = int(_positions.get(ticker, 0))
            amount_to_buy = target - current_shares

            if amount_to_buy > 0:
                buy_orders.append((ticker, "BUY", int(amount_to_buy)))

    else:
        # Non-rebalance bars only submit stop/trend exits.
        for ticker in forced_exits:
            shares = int(_positions.get(ticker, 0))
            if shares > 0:
                sell_orders.append((ticker, "SELL", shares))

    orders = sell_orders + buy_orders

    # Save submitted orders so their next-open fills can be mirrored internally
    # on the following invocation.
    _pending_orders = list(orders)

    return orders
