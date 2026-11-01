"""
Cross-sectional trend-following trading agent.

Strategy summary
----------------
- Tracks its own cash, positions, and pending orders because the platform does
  not provide account statements or fill reports.
- Uses the next bar's open to simulate fills for orders submitted on the prior
  bar, matching the stated execution model.
- After a warm-up period, ranks currently available tickers by:
    * 40-bar momentum
    * 60-bar EMA trend
    * 20-bar volatility-adjusted momentum
- Holds up to six strong-trending tickers, equally weighted.
- Targets roughly 90% invested capital, retaining a cash buffer for overnight
  price gaps between signal close and next execution open.
- Rebalances every five bars.
- Applies trailing-stop and EMA trend exits between scheduled rebalances.
"""

from collections import deque
import math

import numpy as np


# Module-level state. init() resets all of this before each platform run.
_cash = 0.0
_positions = {}
_pending_orders = []
_last_history_len = 0
_bar_number = 0

_state = {}
_last_prices = {}
_peak_prices = {}

WARMUP_BARS = 65
REBALANCE_EVERY = 5
MAX_HOLDINGS = 6
INVESTMENT_FRACTION = 0.90

FAST_EMA_SPAN = 20
SLOW_EMA_SPAN = 60
MOMENTUM_LOOKBACK = 40
VOLATILITY_LOOKBACK = 20

TRAILING_STOP_FRACTION = 0.88
EMA_EXIT_FRACTION = 0.97


def init(cash):
    """Initialize the agent's internal account and indicator state."""
    global _cash, _positions, _pending_orders, _last_history_len
    global _bar_number, _state, _last_prices, _peak_prices

    _cash = float(cash)
    _positions = {}
    _pending_orders = []
    _last_history_len = 0
    _bar_number = 0

    _state = {}
    _last_prices = {}
    _peak_prices = {}


def _valid_price(value):
    """Return True when value is a usable positive numeric price."""
    try:
        return np.isfinite(value) and float(value) > 0.0
    except (TypeError, ValueError):
        return False


def _get_state(ticker):
    """Create or retrieve rolling indicator state for a ticker."""
    if ticker not in _state:
        _state[ticker] = {
            "closes": deque(maxlen=70),
            "returns": deque(maxlen=VOLATILITY_LOOKBACK),
            "ema_fast": None,
            "ema_slow": None,
            "previous_close": None,
        }
    return _state[ticker]


def _process_pending_orders(current_rows):
    """
    Simulate fills for the orders returned on the previous bar.

    The platform executes orders at the current bar's open. Orders are processed
    in submitted order, which is why sell orders are deliberately emitted before
    buy orders by the strategy.
    """
    global _cash, _pending_orders, _positions, _peak_prices

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
            # The platform discards an order if no execution price exists.
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
                    _peak_prices.pop(ticker, None)

        elif direction == "BUY":
            affordable = int(math.floor((_cash + 1e-10) / price))
            filled = min(requested_size, max(0, affordable))

            if filled > 0:
                old_position = int(_positions.get(ticker, 0))
                _positions[ticker] = old_position + filled
                _cash -= filled * price

                if old_position == 0:
                    _peak_prices[ticker] = price

    # Orders are valid for only one execution opportunity.
    _pending_orders = []


def _update_indicators(current_rows):
    """Update prices, EMAs, returns, and trailing highs using current closes."""
    alpha_fast = 2.0 / (FAST_EMA_SPAN + 1.0)
    alpha_slow = 2.0 / (SLOW_EMA_SPAN + 1.0)

    for row in current_rows.itertuples(index=False):
        ticker = row.ticker
        close = row.close

        if not _valid_price(close):
            continue

        close = float(close)
        st = _get_state(ticker)

        previous_close = st["previous_close"]
        if previous_close is not None and previous_close > 0.0:
            daily_return = (close / previous_close) - 1.0
            if np.isfinite(daily_return):
                st["returns"].append(float(daily_return))

        st["closes"].append(close)
        st["previous_close"] = close

        if st["ema_fast"] is None:
            st["ema_fast"] = close
            st["ema_slow"] = close
        else:
            st["ema_fast"] = (
                alpha_fast * close + (1.0 - alpha_fast) * st["ema_fast"]
            )
            st["ema_slow"] = (
                alpha_slow * close + (1.0 - alpha_slow) * st["ema_slow"]
            )

        _last_prices[ticker] = close

        if _positions.get(ticker, 0) > 0:
            high = row.high
            if _valid_price(high):
                high = float(high)
            else:
                high = close

            prior_peak = _peak_prices.get(ticker, high)
            _peak_prices[ticker] = max(prior_peak, high)


def _portfolio_value():
    """Estimate marked-to-market portfolio equity using most recent closes."""
    total = _cash

    for ticker, shares in _positions.items():
        price = _last_prices.get(ticker)
        if price is not None and _valid_price(price):
            total += int(shares) * float(price)

    return max(total, 0.0)


def _should_exit(ticker):
    """Determine whether a currently held position has breached risk controls."""
    if _positions.get(ticker, 0) <= 0:
        return False

    st = _state.get(ticker)
    close = _last_prices.get(ticker)

    if st is None or close is None or not _valid_price(close):
        return False

    peak = _peak_prices.get(ticker, close)
    ema_slow = st.get("ema_slow")

    trailing_stop_hit = close < peak * TRAILING_STOP_FRACTION

    ema_exit_hit = False
    if ema_slow is not None and len(st["closes"]) >= 30:
        ema_exit_hit = close < ema_slow * EMA_EXIT_FRACTION

    return trailing_stop_hit or ema_exit_hit


def _select_candidates(current_tickers, excluded):
    """
    Rank eligible tickers using trend and volatility-adjusted momentum.

    Only tickers present on the current bar are considered for new allocation,
    avoiding orders based on stale price data.
    """
    ranked = []

    for ticker in current_tickers:
        if ticker in excluded:
            continue

        st = _state.get(ticker)
        if st is None:
            continue

        closes = st["closes"]
        returns = st["returns"]

        if len(closes) < MOMENTUM_LOOKBACK + 1:
            continue
        if len(returns) < VOLATILITY_LOOKBACK:
            continue

        close = _last_prices.get(ticker)
        ema_slow = st["ema_slow"]

        if close is None or ema_slow is None or close <= 0.0:
            continue

        past_close = list(closes)[-1 - MOMENTUM_LOOKBACK]
        if past_close <= 0.0:
            continue

        momentum = (close / past_close) - 1.0
        trend = (close / ema_slow) - 1.0

        volatility = float(np.std(np.asarray(returns, dtype=float), ddof=1))
        if not np.isfinite(volatility) or volatility <= 0.0001:
            volatility = 0.0001

        # Approximate risk-adjusted return over the lookback period.
        risk_adjusted_momentum = momentum / (volatility * math.sqrt(MOMENTUM_LOOKBACK))

        # Require both positive intermediate momentum and a positive long trend.
        if momentum <= 0.0 or trend <= 0.0:
            continue

        # Favor momentum, while giving a modest preference to stronger trends.
        score = risk_adjusted_momentum + 0.5 * trend

        if np.isfinite(score):
            ranked.append((score, ticker))

    ranked.sort(reverse=True, key=lambda item: item[0])
    return [ticker for _, ticker in ranked[:MAX_HOLDINGS]]


def _build_rebalance_orders(current_tickers):
    """
    Build sell-first, then buy orders for an equal-weight target portfolio.

    Sell-first ordering lets proceeds from position exits be available for buys
    under the platform's sequential fill behavior.
    """
    forced_exits = {ticker for ticker in _positions if _should_exit(ticker)}
    selected = _select_candidates(current_tickers, forced_exits)

    portfolio_value = _portfolio_value()
    target_value_per_position = (
        portfolio_value * INVESTMENT_FRACTION / len(selected)
        if selected
        else 0.0
    )

    target_shares = {}
    for ticker in selected:
        price = _last_prices.get(ticker)
        if price is None or not _valid_price(price):
            continue

        target_shares[ticker] = max(0, int(target_value_per_position // float(price)))

    sell_orders = []
    buy_orders = []

    # First remove non-selected holdings and trim overweight selected holdings.
    for ticker, held in list(_positions.items()):
        desired = target_shares.get(ticker, 0)

        if held > desired:
            sell_orders.append((ticker, "SELL", int(held - desired)))

    # Then add underweight selected holdings.
    for ticker in selected:
        held = int(_positions.get(ticker, 0))
        desired = int(target_shares.get(ticker, 0))

        if desired > held:
            buy_orders.append((ticker, "BUY", desired - held))

    return sell_orders + buy_orders


def _build_stop_orders():
    """Generate protective exits between normal rebalance dates."""
    orders = []

    for ticker, shares in list(_positions.items()):
        if shares > 0 and _should_exit(ticker):
            orders.append((ticker, "SELL", int(shares)))

    return orders


def on_bar(history):
    """
    Process the latest timestamp and return orders for next-bar-open execution.
    """
    global _last_history_len, _bar_number, _pending_orders

    # The platform appends exactly one timestamp's rows before each call.
    current_rows = history.iloc[_last_history_len:]
    _last_history_len = len(history)

    if len(current_rows) == 0:
        return []

    _bar_number += 1

    # First account for fills of prior-bar orders at this bar's open.
    _process_pending_orders(current_rows)

    # Then incorporate this bar's close into all indicators and risk state.
    _update_indicators(current_rows)

    if _bar_number < WARMUP_BARS:
        return []

    current_tickers = set(current_rows["ticker"].tolist())

    if _bar_number % REBALANCE_EVERY == 0:
        orders = _build_rebalance_orders(current_tickers)
    else:
        orders = _build_stop_orders()

    # Store submitted orders so their fills can be reconstructed next bar.
    _pending_orders = list(orders)
    return orders
