import math
from collections import deque

import numpy as np


# Strategy state. All state is reset by init().
_cash = 0.0
_positions = {}
_avg_cost = {}
_peak_price = {}
_pending_orders = []
_seen_rows = 0
_bar_number = 0

_close_history = {}
_volume_history = {}

_MAX_HISTORY = 70
_REBALANCE_EVERY = 5
_MAX_HOLDINGS = 6
_PORTFOLIO_FRACTION = 0.90
_TRAILING_STOP = 0.88


def init(cash):
    """
    Called once by the platform before the first bar.
    """
    global _cash, _positions, _avg_cost, _peak_price
    global _pending_orders, _seen_rows, _bar_number
    global _close_history, _volume_history

    _cash = float(cash)
    _positions = {}
    _avg_cost = {}
    _peak_price = {}
    _pending_orders = []
    _seen_rows = 0
    _bar_number = 0

    _close_history = {}
    _volume_history = {}


def _valid_price(value):
    try:
        value = float(value)
        return math.isfinite(value) and value > 0.0
    except (TypeError, ValueError):
        return False


def _execute_pending_orders(open_prices):
    """
    Reconstruct fills locally. The platform executes orders at the next
    bar's open, so at the start of each on_bar call we can simulate the
    fills of orders submitted on the previous call.

    Orders are intentionally submitted with sells before buys. This makes
    sale proceeds available for later buy orders on the same execution bar.
    """
    global _cash, _positions, _avg_cost, _peak_price, _pending_orders

    for ticker, direction, requested_size in _pending_orders:
        requested_size = int(requested_size)
        if requested_size <= 0:
            continue

        price = open_prices.get(ticker)
        if not _valid_price(price):
            continue

        price = float(price)

        if direction == "SELL":
            held = int(_positions.get(ticker, 0))
            filled = min(held, requested_size)

            if filled > 0:
                _cash += filled * price
                remaining = held - filled

                if remaining > 0:
                    _positions[ticker] = remaining
                else:
                    _positions.pop(ticker, None)
                    _avg_cost.pop(ticker, None)
                    _peak_price.pop(ticker, None)

        elif direction == "BUY":
            affordable = int(_cash // price)
            filled = min(requested_size, affordable)

            if filled > 0:
                old_shares = int(_positions.get(ticker, 0))
                old_cost = float(_avg_cost.get(ticker, 0.0))

                new_shares = old_shares + filled
                new_cost = (
                    (old_shares * old_cost + filled * price) / new_shares
                )

                _cash -= filled * price
                _positions[ticker] = new_shares
                _avg_cost[ticker] = new_cost
                _peak_price[ticker] = max(
                    float(_peak_price.get(ticker, price)), price
                )

    _pending_orders = []


def _update_histories(current_rows):
    """
    Add the current close and volume for each ticker to rolling histories.
    """
    close_prices = {}
    open_prices = {}

    rows = current_rows[
        ["ticker", "open", "close", "volume"]
    ].itertuples(index=False, name=None)

    for ticker, open_price, close_price, volume in rows:
        if _valid_price(open_price):
            open_prices[ticker] = float(open_price)

        if not _valid_price(close_price):
            continue

        close_price = float(close_price)
        close_prices[ticker] = close_price

        if ticker not in _close_history:
            _close_history[ticker] = deque(maxlen=_MAX_HISTORY)
            _volume_history[ticker] = deque(maxlen=_MAX_HISTORY)

        try:
            volume = float(volume)
            if not math.isfinite(volume) or volume < 0:
                volume = 0.0
        except (TypeError, ValueError):
            volume = 0.0

        _close_history[ticker].append(close_price)
        _volume_history[ticker].append(volume)

    return open_prices, close_prices


def _rank_candidates(close_prices):
    """
    Cross-sectional momentum model.

    A ticker is eligible when:
      * it has enough history,
      * its 50-bar trend is positive,
      * its 60-bar momentum is positive,
      * it is not among the least liquid quarter of the current universe.

    Score combines 20-bar and 60-bar momentum, normalized by recent
    realized volatility. This prefers persistent trends while penalizing
    highly volatile names.
    """
    raw = []

    for ticker, current_price in close_prices.items():
        closes = _close_history.get(ticker)
        volumes = _volume_history.get(ticker)

        if closes is None or volumes is None or len(closes) < 61:
            continue

        prices = np.asarray(closes, dtype=float)
        vols = np.asarray(volumes, dtype=float)

        if np.any(~np.isfinite(prices)) or current_price <= 0:
            continue

        price_20 = prices[-21]
        price_60 = prices[-61]

        if price_20 <= 0 or price_60 <= 0:
            continue

        momentum_20 = current_price / price_20 - 1.0
        momentum_60 = current_price / price_60 - 1.0
        sma_50 = float(np.mean(prices[-50:]))

        if sma_50 <= 0:
            continue

        trend = current_price / sma_50 - 1.0

        recent_prices = prices[-21:]
        returns = np.diff(np.log(recent_prices))
        volatility = float(np.std(returns))

        # Avoid extreme scores caused by nearly zero historical volatility.
        volatility = max(volatility, 0.015)

        average_volume = float(np.mean(vols[-20:]))
        dollar_volume = max(0.0, average_volume * current_price)

        # Long-only trend filter.
        if trend <= 0.0 or momentum_60 <= 0.0:
            continue

        score = (0.35 * momentum_20 + 0.65 * momentum_60) / volatility

        if math.isfinite(score):
            raw.append((ticker, score, dollar_volume))

    if not raw:
        return []

    # Remove the least liquid quarter rather than relying on a fixed dollar
    # volume threshold, which makes the agent usable across different markets.
    liquidities = np.array([item[2] for item in raw], dtype=float)
    liquidity_cutoff = float(np.quantile(liquidities, 0.25))

    candidates = [
        (ticker, score)
        for ticker, score, dollar_volume in raw
        if dollar_volume >= liquidity_cutoff
    ]

    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates


def _estimate_equity(close_prices):
    """
    Mark known positions to the current close. If a ticker is absent from
    the current timestamp, use its latest observed close as a conservative
    bookkeeping fallback.
    """
    equity = _cash

    for ticker, shares in _positions.items():
        price = close_prices.get(ticker)

        if not _valid_price(price):
            history = _close_history.get(ticker)
            if history:
                price = history[-1]

        if _valid_price(price):
            equity += int(shares) * float(price)

    return max(0.0, equity)


def on_bar(history):
    """
    Generate orders for execution at the next bar's open.

    The strategy:
      1. Tracks fills from prior submitted orders internally.
      2. Maintains rolling price/volume histories.
      3. Holds up to six liquid stocks with the strongest volatility-adjusted
         20/60-bar momentum and positive 50-bar trend.
      4. Rebalances every five bars.
      5. Exits immediately on a 12% trailing stop from the highest close
         observed since entering the position.
      6. Keeps roughly 10% cash reserve to reduce failed buys after gaps.
    """
    global _seen_rows, _bar_number, _pending_orders, _peak_price

    if history is None or len(history) == 0:
        return []

    # history is append-only and the platform invokes on_bar once per step.
    # Reading only the newly appended rows avoids repeatedly scanning the
    # entire historical DataFrame.
    if _seen_rows >= len(history):
        return []

    current_rows = history.iloc[_seen_rows:]
    _seen_rows = len(history)

    if len(current_rows) == 0:
        return []

    open_prices, close_prices = _update_histories(current_rows)

    # Prior orders fill at this timestamp's open, before today's signal.
    _execute_pending_orders(open_prices)

    _bar_number += 1

    # Update high-water marks for held positions using today's close.
    for ticker, shares in list(_positions.items()):
        if shares <= 0:
            continue

        price = close_prices.get(ticker)
        if _valid_price(price):
            _peak_price[ticker] = max(
                float(_peak_price.get(ticker, price)),
                float(price),
            )

    sell_orders = []
    buy_orders = []
    planned_sells = set()

    def add_full_sell(ticker):
        if ticker in planned_sells:
            return
        shares = int(_positions.get(ticker, 0))
        if shares > 0 and ticker in close_prices:
            sell_orders.append((ticker, "SELL", shares))
            planned_sells.add(ticker)

    # Trailing stop: sell positions that have dropped 12% from their
    # highest observed close since entry.
    stopped_tickers = set()
    for ticker, shares in list(_positions.items()):
        if shares <= 0:
            continue

        price = close_prices.get(ticker)
        peak = _peak_price.get(ticker)

        if _valid_price(price) and _valid_price(peak):
            if float(price) < float(peak) * _TRAILING_STOP:
                stopped_tickers.add(ticker)
                add_full_sell(ticker)

    # Rebalance periodically. Stops are still evaluated every bar.
    if _bar_number % _REBALANCE_EVERY == 0:
        ranked = _rank_candidates(close_prices)

        selected = []
        for ticker, score in ranked:
            if ticker in stopped_tickers:
                continue
            selected.append(ticker)
            if len(selected) >= _MAX_HOLDINGS:
                break

        selected_set = set(selected)

        # Leave stocks that are no longer selected. A ticker absent from the
        # current bar cannot be sold because it would fill for zero anyway.
        for ticker, shares in list(_positions.items()):
            if shares <= 0:
                continue
            if ticker not in selected_set:
                add_full_sell(ticker)

        if selected:
            equity = _estimate_equity(close_prices)
            target_value = (
                equity * _PORTFOLIO_FRACTION / float(len(selected))
            )

            target_shares = {}
            for ticker in selected:
                price = close_prices.get(ticker)
                if _valid_price(price):
                    target_shares[ticker] = int(target_value // float(price))

            # First reduce oversized holdings. These sells are placed before
            # buys so their proceeds become usable at next open.
            for ticker in selected:
                current_shares = int(_positions.get(ticker, 0))
                desired_shares = int(target_shares.get(ticker, 0))

                if current_shares > desired_shares and ticker in close_prices:
                    reduction = current_shares - desired_shares
                    if reduction > 0:
                        sell_orders.append((ticker, "SELL", reduction))
                        planned_sells.add(ticker)

            # Then buy underweight selected names.
            for ticker in selected:
                current_shares = int(_positions.get(ticker, 0))
                desired_shares = int(target_shares.get(ticker, 0))

                # If a full stop sell was placed, do not repurchase it on
                # the same signal bar.
                if ticker in stopped_tickers:
                    continue

                if desired_shares > current_shares:
                    quantity = desired_shares - current_shares
                    if quantity > 0:
                        buy_orders.append((ticker, "BUY", quantity))

    # Sells first is important because the execution simulator and intended
    # platform behavior can use proceeds from sells for subsequent buys.
    orders = sell_orders + buy_orders
    _pending_orders = list(orders)

    return orders
