import math
from collections import defaultdict, deque

import numpy as np


# Module-level state. The platform calls init() once before any bars arrive.
_cash = 0.0
_positions = {}
_pending_orders = []
_last_close = {}
_peak_close = {}
_close_history = defaultdict(lambda: deque(maxlen=130))
_volume_history = defaultdict(lambda: deque(maxlen=130))
_seen_rows = 0
_bar_number = 0


def init(cash):
    """
    Initialize the agent's internal shadow account.

    Since the platform does not provide fills, positions, or cash balances, the
    strategy reconstructs them from its own submitted orders and subsequent
    next-bar opening prices.
    """
    global _cash, _positions, _pending_orders, _last_close, _peak_close
    global _close_history, _volume_history, _seen_rows, _bar_number

    _cash = float(cash)
    _positions = {}
    _pending_orders = []
    _last_close = {}
    _peak_close = {}
    _close_history = defaultdict(lambda: deque(maxlen=130))
    _volume_history = defaultdict(lambda: deque(maxlen=130))
    _seen_rows = 0
    _bar_number = 0


def _valid_price(value):
    try:
        value = float(value)
        return math.isfinite(value) and value > 0.0
    except (TypeError, ValueError):
        return False


def _apply_pending_orders(open_prices):
    """
    Simulate the fills that occur at this bar's open.

    Orders are processed in submission order. The strategy always submits sells
    before buys, allowing sale proceeds to fund purchases on the same open.
    """
    global _cash, _positions, _pending_orders, _peak_close

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
            affordable = int((_cash + 1e-9) // price)
            filled = min(requested_size, affordable)

            if filled > 0:
                _cash -= filled * price
                _positions[ticker] = int(_positions.get(ticker, 0)) + filled

                # A newly purchased position starts its stop reference at fill.
                prior_peak = _peak_close.get(ticker, price)
                _peak_close[ticker] = max(prior_peak, price)

    # Eliminate tiny floating point residue.
    if _cash < 0.0 and _cash > -1e-6:
        _cash = 0.0

    _pending_orders = []


def _portfolio_value():
    """Estimate current account equity using latest observed closing prices."""
    value = _cash

    for ticker, shares in _positions.items():
        if shares <= 0:
            continue

        price = _last_close.get(ticker)
        if _valid_price(price):
            value += shares * float(price)

    return max(value, 0.0)


def _select_candidates(current_closes):
    """
    Cross-sectional momentum selection.

    A ticker is eligible when:
      * at least 64 observed closes exist,
      * price is above its 50-observation moving average,
      * 63-observation momentum is positive,
      * it has meaningful recent dollar volume,
      * its short-term realized volatility is finite.

    Candidates are ranked by volatility-adjusted momentum. The strategy holds
    up to eight names, which diversifies single-company risk while retaining a
    focused momentum portfolio.
    """
    scored = []

    for ticker, close in current_closes.items():
        if not _valid_price(close):
            continue

        closes_deque = _close_history.get(ticker)
        volumes_deque = _volume_history.get(ticker)

        if closes_deque is None or volumes_deque is None:
            continue

        if len(closes_deque) < 64 or len(volumes_deque) < 20:
            continue

        closes = np.asarray(closes_deque, dtype=float)
        volumes = np.asarray(volumes_deque, dtype=float)

        if not np.all(np.isfinite(closes[-64:])):
            continue

        current = closes[-1]
        old_price = closes[-64]
        sma50 = float(np.mean(closes[-50:]))

        if old_price <= 0.0 or sma50 <= 0.0:
            continue

        momentum = current / old_price - 1.0
        if momentum <= 0.0 or current <= sma50:
            continue

        recent_prices = closes[-21:]
        if np.any(recent_prices <= 0.0):
            continue

        daily_returns = np.diff(np.log(recent_prices))
        volatility = float(np.std(daily_returns, ddof=1))

        if not math.isfinite(volatility) or volatility <= 1e-6:
            continue

        dollar_volume = float(np.mean(closes[-20:] * np.maximum(volumes[-20:], 0.0)))

        # Avoid names that are too illiquid for a portfolio strategy.
        if not math.isfinite(dollar_volume) or dollar_volume < 100000.0:
            continue

        # Moderate volatility adjustment prevents highly volatile "lottery"
        # stocks from dominating pure momentum rankings.
        score = momentum / (volatility * math.sqrt(63.0))

        if math.isfinite(score):
            scored.append((score, ticker))

    scored.sort(reverse=True)
    return [ticker for _, ticker in scored[:8]]


def _make_rebalance_orders(current_closes, force_exits):
    """
    Create a mostly equal-weight portfolio of the strongest momentum names.

    The strategy keeps roughly 10% of equity in cash. It only trades normal
    rebalance changes when the desired share count differs by at least 10%;
    that threshold reduces turnover caused by small daily price changes.
    """
    equity = _portfolio_value()

    if equity <= 0.0:
        return []

    selected = _select_candidates(current_closes)
    selected = [ticker for ticker in selected if ticker not in force_exits]

    # Allocate 90% of equity across up to eight positions.
    capital_per_name = 0.90 * equity / len(selected) if selected else 0.0

    target_shares = {}

    for ticker in selected:
        price = current_closes.get(ticker)
        if _valid_price(price):
            target_shares[ticker] = int(capital_per_name // float(price))

    # Existing holdings that are no longer selected should be exited,
    # provided a current tradable price exists.
    for ticker in list(_positions):
        if ticker in force_exits:
            target_shares[ticker] = 0
        elif ticker not in target_shares and _valid_price(current_closes.get(ticker)):
            target_shares[ticker] = 0

    sell_orders = []
    buy_orders = []

    all_tickers = set(target_shares) | set(_positions)

    for ticker in sorted(all_tickers):
        held = int(_positions.get(ticker, 0))
        desired = int(target_shares.get(ticker, 0))
        price_available = _valid_price(current_closes.get(ticker))

        if not price_available:
            continue

        if desired < held:
            reduction = held - desired

            # Always fully exit when desired position is zero or a stop fired.
            must_trade = desired == 0 or ticker in force_exits

            # Otherwise ignore small rebalance adjustments.
            material_change = reduction >= max(1, int(0.10 * held))

            if must_trade or material_change:
                sell_orders.append((ticker, "SELL", reduction))

        elif desired > held:
            increase = desired - held
            material_change = increase >= max(1, int(0.10 * max(desired, 1)))

            if material_change:
                buy_orders.append((ticker, "BUY", increase))

    # Sells first. This makes the shadow fill simulation conservative and lets
    # realized sale proceeds fund buys at the next open.
    return sell_orders + buy_orders


def on_bar(history):
    """
    Process the newest timestamp and return orders to execute at the next open.

    Strategy summary:
      * Maintains a shadow account because external fill reports are unavailable.
      * Uses 63-observation cross-sectional momentum.
      * Requires positive trend versus the 50-observation moving average.
      * Ranks by momentum adjusted for recent volatility.
      * Holds up to eight liquid names, approximately equally weighted.
      * Rebalances every five bars.
      * Exits positions after a 12% trailing-close drawdown.
      * Retains approximately 10% cash to reduce gap-risk partial fills.
    """
    global _seen_rows, _bar_number, _pending_orders

    # The history DataFrame is append-only, so rows after _seen_rows are the
    # newest timestamp supplied by this call.
    if len(history) <= _seen_rows:
        return []

    new_rows = history.iloc[_seen_rows:]
    _seen_rows = len(history)

    open_prices = {}
    current_closes = {}
    current_volumes = {}

    # Read only the newly appended bars rather than repeatedly scanning all
    # history. This keeps runtime practical for long simulations.
    for row in new_rows.itertuples(index=False):
        ticker = row.ticker

        try:
            open_price = float(row.open)
            if math.isfinite(open_price) and open_price > 0.0:
                open_prices[ticker] = open_price
        except (TypeError, ValueError):
            pass

        try:
            close_price = float(row.close)
            if math.isfinite(close_price) and close_price > 0.0:
                current_closes[ticker] = close_price
        except (TypeError, ValueError):
            pass

        try:
            volume = float(row.volume)
            if not math.isfinite(volume) or volume < 0.0:
                volume = 0.0
            current_volumes[ticker] = volume
        except (TypeError, ValueError):
            current_volumes[ticker] = 0.0

    # Orders submitted last bar fill at this bar's open.
    _apply_pending_orders(open_prices)

    # Update observed close and volume histories.
    for ticker, close in current_closes.items():
        _last_close[ticker] = close
        _close_history[ticker].append(close)
        _volume_history[ticker].append(current_volumes.get(ticker, 0.0))

    _bar_number += 1

    # Update trailing peak values and identify drawdown-stop exits.
    force_exits = set()

    for ticker, shares in list(_positions.items()):
        if shares <= 0:
            continue

        close = current_closes.get(ticker)
        if not _valid_price(close):
            continue

        close = float(close)
        previous_peak = _peak_close.get(ticker, close)
        peak = max(previous_peak, close)
        _peak_close[ticker] = peak

        # 12% trailing close stop. The sell still executes next open, matching
        # the platform's execution model.
        if close <= 0.88 * peak:
            force_exits.add(ticker)

    # Rebalance every five bars. Stop exits are evaluated every bar.
    should_rebalance = (_bar_number % 5 == 0)

    if should_rebalance:
        orders = _make_rebalance_orders(current_closes, force_exits)
    else:
        orders = []

        # On non-rebalance bars, only submit protective exits.
        for ticker in sorted(force_exits):
            held = int(_positions.get(ticker, 0))
            if held > 0 and _valid_price(current_closes.get(ticker)):
                orders.append((ticker, "SELL", held))

    _pending_orders = orders
    return orders
