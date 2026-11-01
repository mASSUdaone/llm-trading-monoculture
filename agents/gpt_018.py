import math
from collections import deque

import numpy as np
import pandas as pd


# Portfolio/account state maintained locally because the platform does not
# provide fills, cash balances, or positions.
_cash = 0.0
_positions = {}
_avg_cost = {}
_peak_price = {}
_last_close = {}

# Rolling per-ticker market data.
_closes = {}
_volumes = {}

# Orders submitted on the prior bar, expected to execute on this bar's open.
_pending_orders = []

# Strategy state.
_targets = set()
_cooldown_until = {}
_bar_number = 0
_seen_rows = 0


def init(cash):
    """Initialize strategy state. Called exactly once by the platform."""
    global _cash, _positions, _avg_cost, _peak_price, _last_close
    global _closes, _volumes, _pending_orders
    global _targets, _cooldown_until, _bar_number, _seen_rows

    _cash = float(cash)
    _positions = {}
    _avg_cost = {}
    _peak_price = {}
    _last_close = {}

    _closes = {}
    _volumes = {}

    _pending_orders = []
    _targets = set()
    _cooldown_until = {}
    _bar_number = 0
    _seen_rows = 0


def _valid_price(x):
    try:
        return math.isfinite(float(x)) and float(x) > 0.0
    except (TypeError, ValueError):
        return False


def _execute_pending(current_rows):
    """
    Reconstruct fills from the prior submitted order list.

    The platform's documented behavior is modeled directly:
    - fills occur at current-bar open;
    - absent/invalid prices cause zero fill;
    - sells cannot exceed held shares;
    - buys cannot exceed available cash;
    - orders are processed in returned order order.
    """
    global _cash, _pending_orders

    if not _pending_orders:
        return

    open_prices = {}
    for row in current_rows.itertuples(index=False):
        ticker = row.ticker
        if _valid_price(row.open):
            open_prices[ticker] = float(row.open)

    for ticker, direction, requested_size in _pending_orders:
        size = int(requested_size)
        if size <= 0:
            continue

        price = open_prices.get(ticker)
        if price is None:
            continue

        if direction == "SELL":
            held = int(_positions.get(ticker, 0))
            filled = min(size, held)
            if filled <= 0:
                continue

            _cash += filled * price
            remaining = held - filled

            if remaining <= 0:
                _positions.pop(ticker, None)
                _avg_cost.pop(ticker, None)
                _peak_price.pop(ticker, None)
            else:
                _positions[ticker] = remaining

        elif direction == "BUY":
            affordable = int((_cash + 1e-10) // price)
            filled = min(size, affordable)
            if filled <= 0:
                continue

            old_shares = int(_positions.get(ticker, 0))
            old_cost = float(_avg_cost.get(ticker, 0.0))
            new_shares = old_shares + filled

            _avg_cost[ticker] = (
                (old_shares * old_cost + filled * price) / new_shares
            )
            _positions[ticker] = new_shares
            _cash -= filled * price

            prior_peak = float(_peak_price.get(ticker, price))
            _peak_price[ticker] = max(prior_peak, price)

    _pending_orders = []


def _update_market_data(current_rows):
    """Append this bar's close, volume, and high information."""
    for row in current_rows.itertuples(index=False):
        ticker = row.ticker

        if _valid_price(row.close):
            close = float(row.close)

            if ticker not in _closes:
                _closes[ticker] = deque(maxlen=140)
                _volumes[ticker] = deque(maxlen=140)

            _closes[ticker].append(close)

            try:
                volume = float(row.volume)
                if not math.isfinite(volume) or volume < 0:
                    volume = 0.0
            except (TypeError, ValueError):
                volume = 0.0
            _volumes[ticker].append(volume)

            _last_close[ticker] = close

        # Update trailing high for open positions.
        if ticker in _positions and _positions[ticker] > 0:
            high = float(row.high) if _valid_price(row.high) else _last_close.get(ticker)
            if high is not None and _valid_price(high):
                _peak_price[ticker] = max(
                    float(_peak_price.get(ticker, high)),
                    float(high),
                )


def _rank(values):
    """Return percentile-like ranks in [0, 1], higher is better."""
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        return np.array([], dtype=float)
    if n == 1:
        return np.array([1.0], dtype=float)

    order = np.argsort(arr)
    result = np.empty(n, dtype=float)
    result[order] = np.arange(n, dtype=float) / float(n - 1)
    return result


def _choose_targets():
    """
    Select liquid stocks/ETFs with:
    - positive intermediate momentum,
    - positive shorter-term momentum,
    - price above its 100-bar trend,
    - lower recent volatility preferred.

    The portfolio is intentionally long-only and will hold cash if few
    instruments qualify.
    """
    candidates = []

    for ticker, close_deque in _closes.items():
        if _cooldown_until.get(ticker, -1) > _bar_number:
            continue

        closes = np.asarray(close_deque, dtype=float)
        volumes = np.asarray(_volumes.get(ticker, []), dtype=float)

        # Need enough history for 100-bar trend and 63-bar momentum.
        if len(closes) < 110 or len(volumes) < 20:
            continue

        price = closes[-1]
        if not math.isfinite(price) or price < 3.0:
            continue

        ma100 = float(np.mean(closes[-100:]))
        if not math.isfinite(ma100) or price <= ma100:
            continue

        base63 = closes[-64]
        base21 = closes[-22]
        if base63 <= 0 or base21 <= 0:
            continue

        momentum_63 = price / base63 - 1.0
        momentum_21 = price / base21 - 1.0

        # Avoid buying clear downtrends even if relative ranking is good.
        if momentum_63 <= 0.0 or momentum_21 < -0.08:
            continue

        returns = np.diff(np.log(closes[-22:]))
        volatility = float(np.std(returns, ddof=1)) if len(returns) > 1 else np.nan
        if not math.isfinite(volatility) or volatility <= 0.0:
            continue

        dollar_volume = float(np.mean(closes[-20:] * volumes[-20:]))
        if not math.isfinite(dollar_volume) or dollar_volume <= 0.0:
            continue

        candidates.append(
            (ticker, momentum_63, momentum_21, volatility, dollar_volume)
        )

    if not candidates:
        return []

    # Exclude the least liquid fifth of the currently eligible universe.
    liquidity = np.asarray([x[4] for x in candidates], dtype=float)
    liquidity_cutoff = float(np.percentile(liquidity, 20.0))
    candidates = [x for x in candidates if x[4] >= liquidity_cutoff]

    if not candidates:
        return []

    mom63 = np.asarray([x[1] for x in candidates], dtype=float)
    mom21 = np.asarray([x[2] for x in candidates], dtype=float)
    vol = np.asarray([x[3] for x in candidates], dtype=float)

    # Cross-sectional composite:
    # 55% intermediate momentum, 25% short momentum, 20% low volatility.
    score = 0.55 * _rank(mom63) + 0.25 * _rank(mom21) + 0.20 * _rank(-vol)

    ranked = sorted(
        zip(candidates, score),
        key=lambda x: x[1],
        reverse=True,
    )

    # A modest number of names helps diversification while limiting turnover.
    return [item[0][0] for item in ranked[:8]]


def _portfolio_equity():
    """Estimate equity using latest known closing prices."""
    equity = float(_cash)
    for ticker, shares in _positions.items():
        price = _last_close.get(ticker)
        if price is not None and _valid_price(price):
            equity += int(shares) * float(price)
    return max(equity, 0.0)


def _weights_for_targets(targets):
    """Inverse-volatility weights, with concentration limits."""
    if not targets:
        return {}

    inverse_vol = {}
    for ticker in targets:
        closes = np.asarray(_closes.get(ticker, []), dtype=float)
        if len(closes) < 22:
            continue

        returns = np.diff(np.log(closes[-22:]))
        vol = float(np.std(returns, ddof=1)) if len(returns) > 1 else np.nan
        if math.isfinite(vol) and vol > 0.0:
            inverse_vol[ticker] = 1.0 / max(vol, 0.008)

    if not inverse_vol:
        return {}

    total = sum(inverse_vol.values())
    weights = {ticker: value / total for ticker, value in inverse_vol.items()}

    # No position may initially receive more than 23% of capital.
    capped = {ticker: min(weight, 0.23) for ticker, weight in weights.items()}
    return capped


def _make_rebalance_orders(targets, forced_sells):
    """
    Convert desired holdings to sell-first orders followed by buy orders.
    Sells are emitted before buys so released cash can fund purchases at the
    next opening auction.
    """
    equity = _portfolio_equity()
    if equity <= 0.0:
        return []

    weights = _weights_for_targets(targets)

    # Use up to 90% capital, but respect the 23% maximum per-position weight.
    # This naturally leaves cash when the qualifying universe is small.
    max_exposure = min(0.90, 0.23 * len(weights))
    target_dollars = {
        ticker: max_exposure * equity * weight
        for ticker, weight in weights.items()
    }

    desired_shares = {}
    for ticker, dollars in target_dollars.items():
        price = _last_close.get(ticker)
        if price is not None and _valid_price(price):
            desired_shares[ticker] = int(dollars // float(price))

    # Anything not in the new target list is exited at rebalance.
    all_relevant = set(_positions) | set(desired_shares) | set(forced_sells)

    sells = []
    buys = []
    estimated_sale_cash = 0.0

    for ticker in all_relevant:
        held = int(_positions.get(ticker, 0))
        desired = 0 if ticker in forced_sells else int(desired_shares.get(ticker, 0))

        if held > desired:
            quantity = held - desired
            sells.append((ticker, "SELL", quantity))

            price = _last_close.get(ticker)
            if price is not None and _valid_price(price):
                estimated_sale_cash += quantity * float(price)

    # Reserve at least 5% estimated portfolio value as cash.
    available_buy_cash = max(
        0.0,
        _cash + estimated_sale_cash - 0.05 * equity,
    )

    raw_buys = []
    total_buy_cost = 0.0

    for ticker, desired in desired_shares.items():
        held = int(_positions.get(ticker, 0))
        if desired > held:
            quantity = desired - held
            price = _last_close.get(ticker)
            if price is not None and _valid_price(price):
                raw_buys.append((ticker, quantity, float(price)))
                total_buy_cost += quantity * float(price)

    # Scale purchases down before submitting them. This reduces the chance
    # that an opening gap leaves most later orders partially unfilled.
    scale = 1.0
    if total_buy_cost > available_buy_cash and total_buy_cost > 0.0:
        scale = available_buy_cash / total_buy_cost

    for ticker, quantity, _price in raw_buys:
        adjusted = int(quantity * scale)
        if adjusted > 0:
            buys.append((ticker, "BUY", adjusted))

    return sells + buys


def on_bar(history):
    """
    Main platform callback.

    Signals use only current and past closes. Orders generated here execute at
    the next bar open, as required by the platform.
    """
    global _seen_rows, _bar_number, _targets, _pending_orders, _cooldown_until

    if history is None or len(history) == 0:
        return []

    # The frame is append-only. Reading only appended rows avoids repeatedly
    # processing the entire history on every callback.
    if _seen_rows >= len(history):
        return []

    new_rows = history.iloc[_seen_rows:].copy()
    _seen_rows = len(history)

    # Under the platform contract, newly appended rows belong to this step.
    # Keep the latest timestamp defensively if a caller supplied more.
    current_timestamp = new_rows["timestamp"].iloc[-1]
    current_rows = new_rows.loc[new_rows["timestamp"] == current_timestamp]

    if len(current_rows) == 0:
        return []

    _bar_number += 1

    # First reconstruct fills for orders submitted on the preceding bar.
    _execute_pending(current_rows)

    # Then current closes become available for signal generation.
    _update_market_data(current_rows)

    forced_sells = set()

    # Daily trailing stop and catastrophic loss stop.
    # A stopped ticker is excluded from new selections for 20 bars.
    for ticker, shares in list(_positions.items()):
        if shares <= 0:
            continue

        close = _last_close.get(ticker)
        peak = _peak_price.get(ticker)
        cost = _avg_cost.get(ticker)

        if close is None:
            continue

        trailing_stop = (
            peak is not None
            and _valid_price(peak)
            and float(close) < 0.88 * float(peak)
        )
        catastrophic_stop = (
            cost is not None
            and _valid_price(cost)
            and float(close) < 0.82 * float(cost)
        )

        if trailing_stop or catastrophic_stop:
            forced_sells.add(ticker)
            _targets.discard(ticker)
            _cooldown_until[ticker] = _bar_number + 20

    # Re-rank weekly. The strategy still checks stops every bar.
    rebalance_now = (_bar_number % 5 == 0)

    if rebalance_now:
        selected = _choose_targets()
        _targets = set(selected)

    # Stops must be submitted immediately even between scheduled rebalances.
    if forced_sells and not rebalance_now:
        orders = []
        for ticker in sorted(forced_sells):
            held = int(_positions.get(ticker, 0))
            if held > 0:
                orders.append((ticker, "SELL", held))
        _pending_orders = orders[:]
        return orders

    if rebalance_now:
        orders = _make_rebalance_orders(_targets, forced_sells)
        _pending_orders = orders[:]
        return orders

    return []
