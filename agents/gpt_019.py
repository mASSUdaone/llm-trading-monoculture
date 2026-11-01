import math
from collections import defaultdict, deque

import numpy as np


# --- Module state ------------------------------------------------------------

_cash = 0.0
_positions = defaultdict(int)       # ticker -> shares
_last_price = {}                   # ticker -> most recent close
_peaks = {}                        # ticker -> highest price since entry
_price_history = {}                # ticker -> deque of closes
_pending_orders = []               # orders submitted on prior bar
_history_rows_seen = 0
_bar_number = 0
_last_timestamp = None
_cooldown_until = {}               # ticker -> bar number before re-entry allowed


# --- Strategy parameters -----------------------------------------------------

LOOKBACK_TREND = 101
LOOKBACK_MOMENTUM = 64
LOOKBACK_VOL = 22

REBALANCE_EVERY = 5
MAX_HOLDINGS = 6

# Intentionally leave some cash available for overnight gaps because orders
# execute at the following open rather than at the closing price used to size.
TARGET_INVESTED_FRACTION = 0.85

STOP_LOSS_FRACTION = 0.12
STOP_COOLDOWN_BARS = 10


def init(cash):
    """
    Called once by the platform before the first bar.
    """
    global _cash, _positions, _last_price, _peaks
    global _price_history, _pending_orders, _history_rows_seen
    global _bar_number, _last_timestamp, _cooldown_until

    _cash = float(cash)
    _positions = defaultdict(int)
    _last_price = {}
    _peaks = {}
    _price_history = {}
    _pending_orders = []
    _history_rows_seen = 0
    _bar_number = 0
    _last_timestamp = None
    _cooldown_until = {}


def _valid_price(value):
    try:
        return math.isfinite(float(value)) and float(value) > 0.0
    except (TypeError, ValueError):
        return False


def _apply_pending_orders(current_rows):
    """
    Maintain an internal account ledger.

    Orders returned on the prior call execute at this bar's open. The platform
    does not provide fill reports, so this mirrors the stated fill rules.
    """
    global _cash, _pending_orders

    if not _pending_orders:
        return

    open_prices = {}
    for row in current_rows.itertuples(index=False):
        ticker = row.ticker
        price = row.open
        if _valid_price(price):
            open_prices[ticker] = float(price)

    # Orders are processed in returned order. Rebalance construction always
    # places sells before buys, allowing sale proceeds to fund purchases.
    for ticker, direction, requested_size in _pending_orders:
        price = open_prices.get(ticker)
        if price is None:
            continue

        requested_size = int(requested_size)
        if requested_size <= 0:
            continue

        if direction == "SELL":
            filled = min(requested_size, _positions.get(ticker, 0))
            if filled > 0:
                _positions[ticker] -= filled
                _cash += filled * price

                if _positions[ticker] <= 0:
                    _positions[ticker] = 0
                    _peaks.pop(ticker, None)

        elif direction == "BUY":
            affordable = int(_cash // price)
            filled = min(requested_size, affordable)

            if filled > 0:
                old_position = _positions.get(ticker, 0)
                _positions[ticker] += filled
                _cash -= filled * price

                if old_position <= 0:
                    _peaks[ticker] = price
                else:
                    _peaks[ticker] = max(_peaks.get(ticker, price), price)

    _pending_orders = []


def _update_market_data(current_rows):
    """
    Update rolling per-ticker close histories and return currently tradable
    tickers plus their current high/close data.
    """
    current_tickers = set()
    current_highs = {}
    current_closes = {}

    for row in current_rows.itertuples(index=False):
        ticker = row.ticker
        close = row.close

        if not _valid_price(close):
            continue

        close = float(close)
        current_tickers.add(ticker)
        current_closes[ticker] = close
        _last_price[ticker] = close

        if ticker not in _price_history:
            _price_history[ticker] = deque(maxlen=130)
        _price_history[ticker].append(close)

        high = row.high
        if _valid_price(high):
            current_highs[ticker] = float(high)
        else:
            current_highs[ticker] = close

    # Update trailing highs only for positions that have a fresh bar today.
    for ticker, shares in list(_positions.items()):
        if shares <= 0:
            continue
        if ticker in current_highs:
            _peaks[ticker] = max(
                _peaks.get(ticker, current_highs[ticker]),
                current_highs[ticker],
            )

    return current_tickers, current_highs, current_closes


def _portfolio_value():
    """
    Mark positions using the latest available adjusted close.
    """
    value = _cash
    for ticker, shares in _positions.items():
        if shares > 0:
            price = _last_price.get(ticker)
            if _valid_price(price):
                value += shares * float(price)
    return value


def _select_candidates(current_tickers):
    """
    Select strong, liquid-enough trend-following candidates.

    Ranking combines:
      * 63-bar risk-adjusted momentum
      * 21-bar confirmation momentum
      * price above its 100-bar moving average

    Only positive-momentum assets above their long-term trend are eligible.
    """
    ranked = []

    for ticker in current_tickers:
        if _cooldown_until.get(ticker, -1) > _bar_number:
            continue

        closes_deque = _price_history.get(ticker)
        if closes_deque is None or len(closes_deque) < LOOKBACK_TREND:
            continue

        closes = np.asarray(closes_deque, dtype=float)
        price = closes[-1]

        if not np.isfinite(price) or price <= 0.5:
            continue

        sma_100 = float(np.mean(closes[-100:]))
        if not np.isfinite(sma_100) or price <= sma_100:
            continue

        # 63-bar and 21-bar returns.
        ret_63 = price / closes[-64] - 1.0
        ret_21 = price / closes[-22] - 1.0

        if not np.isfinite(ret_63) or not np.isfinite(ret_21):
            continue

        # Use recent daily log-return volatility to avoid simply selecting the
        # most volatile assets.
        recent = closes[-22:]
        if np.any(recent <= 0):
            continue

        log_returns = np.diff(np.log(recent))
        volatility = float(np.std(log_returns, ddof=1))

        if not np.isfinite(volatility) or volatility < 0.002:
            continue

        # Require a non-negative medium-term trend. This keeps the strategy
        # from aggressively buying broad downtrends merely because of a bounce.
        if ret_63 <= -0.02:
            continue

        score_63 = ret_63 / (volatility * math.sqrt(63.0))
        score_21 = ret_21 / (volatility * math.sqrt(21.0))

        # Favor sustained momentum, with a smaller weight on recent strength.
        score = 0.75 * score_63 + 0.25 * score_21

        if np.isfinite(score) and score > 0.0:
            ranked.append((score, ticker))

    ranked.sort(reverse=True)
    return [ticker for _, ticker in ranked[:MAX_HOLDINGS]]


def _stop_loss_orders(current_tickers, current_closes):
    """
    Exit positions that have materially declined from their post-entry peak.
    """
    orders = []

    for ticker, shares in list(_positions.items()):
        if shares <= 0 or ticker not in current_tickers:
            continue

        close = current_closes.get(ticker)
        peak = _peaks.get(ticker)

        if not _valid_price(close) or not _valid_price(peak):
            continue

        if close <= float(peak) * (1.0 - STOP_LOSS_FRACTION):
            orders.append((ticker, "SELL", int(shares)))
            _cooldown_until[ticker] = _bar_number + STOP_COOLDOWN_BARS

    return orders


def _rebalance_orders(current_tickers):
    """
    Build a target portfolio of equal-dollar positions in the selected names.

    Sells are emitted before buys so sale proceeds can be used at execution.
    """
    selected = _select_candidates(current_tickers)
    selected_set = set(selected)

    equity = _portfolio_value()
    if equity <= 0:
        return []

    orders = []

    # First remove positions no longer selected.
    for ticker, shares in list(_positions.items()):
        if shares <= 0:
            continue

        if ticker not in selected_set:
            # If a ticker is absent today, a sell would fill zero next bar if
            # still absent. Avoid needless orders until it appears again.
            if ticker in current_tickers:
                orders.append((ticker, "SELL", int(shares)))

    if not selected:
        return orders

    target_dollars = (equity * TARGET_INVESTED_FRACTION) / len(selected)
    target_shares = {}

    for ticker in selected:
        price = _last_price.get(ticker)
        if not _valid_price(price):
            continue
        target_shares[ticker] = max(0, int(target_dollars // float(price)))

    # Reduce oversized selected positions before submitting any buys.
    for ticker in selected:
        desired = target_shares.get(ticker, 0)
        held = _positions.get(ticker, 0)

        if held > desired and ticker in current_tickers:
            orders.append((ticker, "SELL", int(held - desired)))

    # Then add underweight positions. Actual fills are constrained by cash at
    # the following open, matching the platform's partial-fill rule.
    for ticker in selected:
        desired = target_shares.get(ticker, 0)
        held = _positions.get(ticker, 0)

        if desired > held:
            orders.append((ticker, "BUY", int(desired - held)))

    return orders


def on_bar(history):
    """
    Called once per timestamp. Returns orders to execute at the next bar's open.
    """
    global _history_rows_seen, _bar_number, _last_timestamp, _pending_orders

    if history is None or len(history) == 0:
        return []

    current_timestamp = history.iloc[-1]["timestamp"]

    # Under the append-only contract, rows after this index are the newly
    # supplied bar. The filter also makes this robust if several rows were
    # appended together.
    new_rows = history.iloc[_history_rows_seen:]
    if len(new_rows) == 0:
        return []

    current_rows = new_rows[new_rows["timestamp"] == current_timestamp]
    if len(current_rows) == 0:
        return []

    # Prior orders execute at this timestamp's open.
    _apply_pending_orders(current_rows)

    # Incorporate the just-closed prices for signal generation.
    current_tickers, _, current_closes = _update_market_data(current_rows)

    _history_rows_seen = len(history)

    if _last_timestamp != current_timestamp:
        _bar_number += 1
        _last_timestamp = current_timestamp

    # Risk exits are checked every bar, not merely on rebalance days.
    stop_orders = _stop_loss_orders(current_tickers, current_closes)
    if stop_orders:
        _pending_orders = stop_orders
        return stop_orders

    # Rebalance periodically once sufficient history exists.
    if _bar_number % REBALANCE_EVERY != 0:
        return []

    orders = _rebalance_orders(current_tickers)
    _pending_orders = orders
    return orders
