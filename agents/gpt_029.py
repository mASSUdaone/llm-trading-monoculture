import math
from collections import defaultdict, deque

import numpy as np


# Strategy parameters
MAX_HOLDINGS = 8
REBALANCE_EVERY = 5
HISTORY_LENGTH = 75

MIN_PRICE = 2.0
MIN_DOLLAR_VOLUME = 250_000.0

TARGET_INVESTED_FRACTION = 0.90
TRAILING_STOP = 0.12
REDUCTION_BAND = 0.12
BUY_BAND = 0.10


# Module state. It is initialized by init().
_cash = 0.0
_positions = {}
_peaks = {}

_closes = defaultdict(lambda: deque(maxlen=HISTORY_LENGTH))
_volumes = defaultdict(lambda: deque(maxlen=HISTORY_LENGTH))
_last_close = {}

_pending_orders = []
_processed_rows = 0
_bar_number = 0


def init(cash):
    """
    Initialize the agent.

    The platform calls this once before the first bar.
    """
    global _cash, _positions, _peaks
    global _closes, _volumes, _last_close
    global _pending_orders, _processed_rows, _bar_number

    _cash = float(cash)
    _positions = {}
    _peaks = {}

    _closes = defaultdict(lambda: deque(maxlen=HISTORY_LENGTH))
    _volumes = defaultdict(lambda: deque(maxlen=HISTORY_LENGTH))
    _last_close = {}

    _pending_orders = []
    _processed_rows = 0
    _bar_number = 0


def _valid_number(value):
    try:
        value = float(value)
        return math.isfinite(value)
    except (TypeError, ValueError):
        return False


def _apply_pending_orders(current_open_prices):
    """
    Reconstruct fills from the orders sent on the previous bar.

    Orders execute on this bar's open. Since the platform does not provide
    fill reports, the agent maintains its own conservative cash/position ledger.
    """
    global _cash, _positions, _peaks, _pending_orders

    for ticker, direction, requested_size in _pending_orders:
        if ticker not in current_open_prices:
            continue

        price = current_open_prices[ticker]
        if not _valid_number(price) or price <= 0:
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
                    _peaks.pop(ticker, None)

        elif direction == "BUY":
            affordable = int(_cash // price)
            filled = min(requested_size, affordable)

            if filled > 0:
                _cash -= filled * price
                _positions[ticker] = int(_positions.get(ticker, 0)) + filled

                previous_peak = _peaks.get(ticker, price)
                _peaks[ticker] = max(previous_peak, price)

    _pending_orders = []


def _signal_for_ticker(ticker):
    """
    Return a risk-adjusted trend score for a ticker, or None when there is not
    enough history or the ticker fails liquidity/trend filters.

    The strategy favors securities with:
      * positive medium-term momentum,
      * positive short-term momentum,
      * price above its medium-term trend,
      * rising short-term trend,
      * lower realized volatility for a given return.
    """
    close_history = _closes[ticker]
    volume_history = _volumes[ticker]

    if len(close_history) < 65 or len(volume_history) < 20:
        return None

    prices = np.asarray(close_history, dtype=float)
    volumes = np.asarray(volume_history, dtype=float)

    if not np.all(np.isfinite(prices)) or not np.all(np.isfinite(volumes)):
        return None

    price = float(prices[-1])
    if price < MIN_PRICE:
        return None

    average_dollar_volume = float(np.mean(volumes[-20:]) * price)
    if average_dollar_volume < MIN_DOLLAR_VOLUME:
        return None

    sma20 = float(np.mean(prices[-20:]))
    sma50 = float(np.mean(prices[-50:]))

    if sma20 <= 0 or sma50 <= 0:
        return None

    # Approximately three-week and two-month momentum.
    ret15 = price / float(prices[-16]) - 1.0
    ret45 = price / float(prices[-46]) - 1.0

    # Realized volatility of recent log returns.
    recent = prices[-21:]
    log_returns = np.diff(np.log(recent))
    volatility = float(np.std(log_returns, ddof=1))

    if not math.isfinite(volatility) or volatility <= 1e-8:
        return None

    # Trend filter: avoid buying names in established downtrends.
    if price <= sma50:
        return None
    if sma20 <= sma50:
        return None
    if ret45 <= 0.0:
        return None

    # Risk-adjusted momentum. A small short-term component helps rank leaders.
    score = (ret45 + 0.35 * ret15) / volatility

    if not math.isfinite(score):
        return None

    return score


def _portfolio_value():
    """
    Estimate current equity using the most recent observed close for each held
    asset. The estimate is used only for position sizing.
    """
    value = _cash

    for ticker, shares in _positions.items():
        price = _last_close.get(ticker)
        if price is not None and _valid_number(price) and price > 0:
            value += int(shares) * float(price)

    return value


def on_bar(history):
    """
    Process one timestamp and return next-open orders.

    The strategy is a diversified long-only cross-sectional momentum system:
      - every few bars, rank liquid securities by risk-adjusted momentum;
      - hold the strongest up to MAX_HOLDINGS names;
      - size holdings approximately equally;
      - exit quickly on a material trailing loss or broken trend;
      - maintain an internal fill ledger because fill reports are unavailable.
    """
    global _processed_rows, _bar_number, _pending_orders

    if history is None or len(history) == 0:
        return []

    # history is append-only, so only inspect newly appended rows.
    if _processed_rows >= len(history):
        return []

    new_rows = history.iloc[_processed_rows:]
    _processed_rows = len(history)

    if len(new_rows) == 0:
        return []

    # Current timestamp open prices are needed to reconstruct execution of
    # prior orders before any new decisions are made.
    current_open_prices = {}
    current_close_prices = {}
    current_tickers = set()

    tickers = new_rows["ticker"].to_numpy()
    opens = new_rows["open"].to_numpy()
    closes = new_rows["close"].to_numpy()
    volumes = new_rows["volume"].to_numpy()

    for ticker, open_price, close_price, volume in zip(
        tickers, opens, closes, volumes
    ):
        current_tickers.add(ticker)

        if _valid_number(open_price) and float(open_price) > 0:
            current_open_prices[ticker] = float(open_price)

        if (
            _valid_number(close_price)
            and _valid_number(volume)
            and float(close_price) > 0
            and float(volume) >= 0
        ):
            close_value = float(close_price)
            volume_value = float(volume)

            _closes[ticker].append(close_value)
            _volumes[ticker].append(volume_value)
            _last_close[ticker] = close_value
            current_close_prices[ticker] = close_value

    # Previous orders execute at the current open.
    _apply_pending_orders(current_open_prices)

    _bar_number += 1

    # Update trailing-high marks for positions with a current close.
    for ticker in list(_positions.keys()):
        if ticker in current_close_prices:
            price = current_close_prices[ticker]
            _peaks[ticker] = max(_peaks.get(ticker, price), price)

    sell_orders = []
    buy_orders = []
    forced_exits = set()

    # Protective exits are evaluated every bar, not only at rebalance time.
    for ticker, shares in list(_positions.items()):
        if shares <= 0 or ticker not in current_close_prices:
            continue

        price = current_close_prices[ticker]
        peak = _peaks.get(ticker, price)
        closes_for_ticker = _closes[ticker]

        trailing_stop_hit = price <= peak * (1.0 - TRAILING_STOP)

        trend_break = False
        if len(closes_for_ticker) >= 50:
            price_array = np.asarray(closes_for_ticker, dtype=float)
            sma50 = float(np.mean(price_array[-50:]))
            ret15 = price / float(price_array[-16]) - 1.0

            # Require both weak momentum and a trend break to reduce noise.
            trend_break = price < sma50 and ret15 < 0.0

        if trailing_stop_hit or trend_break:
            sell_orders.append((ticker, "SELL", int(shares)))
            forced_exits.add(ticker)

    is_rebalance_bar = (_bar_number % REBALANCE_EVERY) == 0

    if is_rebalance_bar:
        scored = []

        # Only rank securities available on this current bar. This avoids
        # intentionally selecting names that may not be tradeable next bar.
        for ticker in current_tickers:
            if ticker in forced_exits:
                continue

            signal = _signal_for_ticker(ticker)
            if signal is not None:
                scored.append((signal, ticker))

        scored.sort(reverse=True)
        selected = [ticker for _, ticker in scored[:MAX_HOLDINGS]]
        selected_set = set(selected)

        equity = _portfolio_value()
        slot_count = len(selected)

        if slot_count > 0 and equity > 0:
            target_dollar_per_name = (
                equity * TARGET_INVESTED_FRACTION / float(slot_count)
            )

            desired_shares = {}
            for ticker in selected:
                price = current_close_prices.get(ticker)
                if price is not None and price > 0:
                    desired_shares[ticker] = int(target_dollar_per_name // price)

            # Sell positions that are no longer selected, or trim positions that
            # became materially larger than their equal-weight target.
            already_selling = set(forced_exits)

            for ticker, held in list(_positions.items()):
                if ticker in already_selling:
                    continue

                target = desired_shares.get(ticker, 0)

                if ticker not in selected_set:
                    sell_orders.append((ticker, "SELL", int(held)))
                    already_selling.add(ticker)
                elif held > int(target * (1.0 + REDUCTION_BAND)):
                    quantity = int(held - target)
                    if quantity > 0:
                        sell_orders.append((ticker, "SELL", quantity))
                        already_selling.add(ticker)

            # Buys are appended after sells. This ordering helps the internal
            # ledger match common exchange simulators that execute submitted
            # orders sequentially.
            for ticker in selected:
                if ticker in forced_exits:
                    continue

                target = desired_shares.get(ticker, 0)
                held = int(_positions.get(ticker, 0))

                if target > held:
                    deficit = target - held

                    # For existing positions, ignore tiny rebalance changes.
                    # New positions are always allowed.
                    if held == 0 or deficit >= max(1, int(target * BUY_BAND)):
                        buy_orders.append((ticker, "BUY", int(deficit)))

    # Sell orders precede buy orders, which makes future fill reconstruction
    # deterministic and maximizes cash made available by exits.
    orders = sell_orders + buy_orders

    # Save the exact order sequence so next bar's opening fills can be inferred.
    _pending_orders = list(orders)

    return orders
