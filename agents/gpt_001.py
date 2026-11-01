import math
from collections import deque

import numpy as np


# Internal account state. The platform does not report fills, so this module
# reproduces the documented next-open fill rules internally.
_cash = 0.0
_positions = {}
_pending_orders = []
_last_price = {}
_peak_price = {}
_series = {}
_processed_rows = 0
_step = 0


def init(cash):
    global _cash, _positions, _pending_orders, _last_price
    global _peak_price, _series, _processed_rows, _step

    _cash = float(cash)
    _positions = {}
    _pending_orders = []
    _last_price = {}
    _peak_price = {}
    _series = {}
    _processed_rows = 0
    _step = 0


def _valid_price(x):
    try:
        return math.isfinite(float(x)) and float(x) > 0.0
    except (TypeError, ValueError):
        return False


def _state_for(ticker):
    state = _series.get(ticker)
    if state is None:
        state = {
            "close": deque(maxlen=70),
            "volume": deque(maxlen=25),
        }
        _series[ticker] = state
    return state


def _apply_pending_orders(open_prices):
    """
    Apply orders submitted on the previous bar at this bar's open.

    Orders are processed in returned order. This is why the strategy always
    emits sales before purchases: released sale proceeds can fund purchases.
    """
    global _cash, _pending_orders, _positions

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
                    _peak_price.pop(ticker, None)

        elif direction == "BUY":
            affordable = int(_cash // price)
            filled = min(requested_size, affordable)

            if filled > 0:
                _cash -= filled * price
                _positions[ticker] = int(_positions.get(ticker, 0)) + filled

    # Unfilled portions are discarded by platform rules.
    _pending_orders = []


def _portfolio_value():
    value = _cash

    for ticker, shares in _positions.items():
        price = _last_price.get(ticker)
        if _valid_price(price):
            value += shares * float(price)

    return value


def _candidate_scores(excluded):
    """
    Cross-sectional long-only momentum selection.

    A stock qualifies when:
      * it has at least 60 bars of history;
      * price is above its 50-bar average;
      * medium-term momentum is positive;
      * its recent dollar volume is not in the least-liquid 40% of names.

    Ranking favors strong 20/60 bar momentum with a volatility penalty.
    """
    raw = []

    for ticker, state in _series.items():
        if ticker in excluded:
            continue

        closes = state["close"]
        volumes = state["volume"]

        if len(closes) < 65 or len(volumes) < 20:
            continue

        prices = np.asarray(closes, dtype=float)
        vols = np.asarray(volumes, dtype=float)

        if not np.all(np.isfinite(prices)) or prices[-1] <= 0:
            continue

        close = float(prices[-1])
        sma50 = float(np.mean(prices[-50:]))

        if sma50 <= 0 or close < sma50:
            continue

        ret20 = close / float(prices[-21]) - 1.0
        ret60 = close / float(prices[-61]) - 1.0

        # Require positive medium-term trend and avoid sharp recent weakness.
        if ret60 <= 0.02 or ret20 < -0.03:
            continue

        recent = prices[-21:]
        if np.any(recent <= 0):
            continue

        log_returns = np.diff(np.log(recent))
        vol = float(np.std(log_returns, ddof=1))

        if not math.isfinite(vol):
            continue

        vol = max(vol, 0.003)
        dollar_volume = float(np.mean(vols[-20:]) * close)

        if not math.isfinite(dollar_volume) or dollar_volume <= 0:
            continue

        # Momentum weighted toward the longer horizon, then risk-adjusted.
        trend = 0.60 * ret60 + 0.40 * ret20
        score = trend / vol

        raw.append((ticker, score, dollar_volume, close))

    if not raw:
        return []

    # Avoid the most illiquid names relative to the currently available universe.
    dollar_volumes = np.asarray([x[2] for x in raw], dtype=float)
    liquidity_cutoff = float(np.percentile(dollar_volumes, 40.0))

    liquid = [x for x in raw if x[2] >= liquidity_cutoff]

    # If the universe is very small, do not accidentally exclude nearly all names.
    if len(liquid) < min(3, len(raw)):
        liquid = raw

    liquid.sort(key=lambda x: x[1], reverse=True)
    return liquid[:5]


def _build_rebalance_orders(forced_exits):
    """
    Build a target portfolio of up to five equally weighted momentum leaders.

    About 90% of marked portfolio value is invested, leaving an approximate
    10% reserve for opening gaps and partial-fill robustness.
    """
    portfolio_value = _portfolio_value()

    if portfolio_value <= 0:
        return []

    candidates = _candidate_scores(forced_exits)
    selected = {ticker: close for ticker, _, _, close in candidates}

    # Equal weights are intentionally used instead of concentration in the
    # highest score: cross-sectional momentum scores are noisy.
    target_weight = 0.90 / max(len(selected), 1)
    target_shares = {}

    for ticker, close in selected.items():
        target_value = portfolio_value * target_weight
        target_shares[ticker] = max(0, int(target_value // close))

    sell_orders = []
    estimated_sale_proceeds = 0.0

    # First remove positions stopped out or no longer selected.
    for ticker, held in list(_positions.items()):
        if held <= 0:
            continue

        if ticker in forced_exits:
            quantity = held
        elif ticker not in target_shares:
            quantity = held
        else:
            desired = target_shares[ticker]
            quantity = max(0, held - desired)

        if quantity > 0:
            sell_orders.append((ticker, "SELL", int(quantity)))
            price = _last_price.get(ticker)
            if _valid_price(price):
                estimated_sale_proceeds += quantity * float(price)

    # Estimate available capital after sales. Sales are emitted first.
    estimated_cash = _cash + estimated_sale_proceeds
    reserve = portfolio_value * 0.05
    buy_orders = []

    # Buy stronger ranked names first if opening gaps make capital tighter.
    for ticker, _, _, close in candidates:
        desired = target_shares[ticker]
        held = int(_positions.get(ticker, 0))
        needed = max(0, desired - held)

        # Ignore trivial adjustments below about 3% of target value.
        if needed > 0 and held > 0:
            difference_value = needed * close
            target_value = max(desired * close, 1.0)
            if difference_value < 0.03 * target_value:
                continue

        if needed <= 0:
            continue

        available_for_buy = max(0.0, estimated_cash - reserve)
        affordable = int(available_for_buy // close)
        quantity = min(needed, affordable)

        if quantity > 0:
            buy_orders.append((ticker, "BUY", int(quantity)))
            estimated_cash -= quantity * close

    return sell_orders + buy_orders


def _build_stop_orders():
    """
    Defensive exits are checked every bar, independently of the five-bar
    rebalance schedule.

    Exit if either:
      * close is 12% below the best close observed while held; or
      * price is materially below its 50-bar trend average and short-term
        momentum is also negative.
    """
    exits = set()

    for ticker, shares in _positions.items():
        if shares <= 0:
            continue

        state = _series.get(ticker)
        price = _last_price.get(ticker)

        if state is None or not _valid_price(price):
            continue

        closes = state["close"]
        if len(closes) < 50:
            continue

        prices = np.asarray(closes, dtype=float)
        close = float(price)
        sma50 = float(np.mean(prices[-50:]))

        peak = max(float(_peak_price.get(ticker, close)), close)
        _peak_price[ticker] = peak

        drawdown_exit = close <= 0.88 * peak

        momentum_exit = False
        if len(prices) >= 21:
            ret20 = close / float(prices[-21]) - 1.0
            momentum_exit = close < 0.97 * sma50 and ret20 < 0.0

        if drawdown_exit or momentum_exit:
            exits.add(ticker)

    return exits


def on_bar(history):
    global _processed_rows, _step, _pending_orders

    if history is None or len(history) == 0:
        return []

    # A repeated call with no newly appended data should not submit duplicates.
    if len(history) <= _processed_rows:
        return []

    new_rows = history.iloc[_processed_rows:]
    latest_timestamp = new_rows["timestamp"].iloc[-1]
    current_rows = new_rows.loc[new_rows["timestamp"] == latest_timestamp]

    # Fill prior-bar orders using this timestamp's open prices.
    open_prices = {}
    for row in current_rows.itertuples(index=False):
        if _valid_price(row.open):
            open_prices[row.ticker] = float(row.open)

    _apply_pending_orders(open_prices)

    # Update market data state using closes from the completed current bar.
    for row in current_rows.itertuples(index=False):
        ticker = row.ticker

        if not _valid_price(row.close):
            continue

        close = float(row.close)
        volume = float(row.volume) if _valid_price(row.volume) else 0.0

        state = _state_for(ticker)
        state["close"].append(close)
        state["volume"].append(max(0.0, volume))

        _last_price[ticker] = close

        if int(_positions.get(ticker, 0)) > 0:
            prior_peak = float(_peak_price.get(ticker, close))
            _peak_price[ticker] = max(prior_peak, close)

    _processed_rows = len(history)
    _step += 1

    forced_exits = _build_stop_orders()

    # Rebalance every five bars. Stop exits happen immediately.
    if _step % 5 == 0:
        orders = _build_rebalance_orders(forced_exits)
    else:
        orders = [
            (ticker, "SELL", int(_positions[ticker]))
            for ticker in sorted(forced_exits)
            if int(_positions.get(ticker, 0)) > 0
        ]

    # Orders submitted now execute at the next bar's open.
    _pending_orders = list(orders)
    return orders
