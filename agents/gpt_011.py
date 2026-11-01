import math
from collections import deque

import numpy as np


# Module-level state. init() resets all of this before a run.
_cash = 0.0
_positions = {}
_avg_cost = {}
_prices = {}
_close_history = {}
_volume_history = {}
_last_seen = {}
_pending_orders = []
_step = 0
_last_rebalance = -10**9
_blocked_until = {}


def init(cash):
    """
    Initialize the agent's internally maintained account state.

    The platform does not provide fills, cash, or positions, so this strategy
    reconstructs them using the prior step's submitted orders and the current
    bar's opening prices.
    """
    global _cash, _positions, _avg_cost, _prices
    global _close_history, _volume_history, _last_seen
    global _pending_orders, _step, _last_rebalance, _blocked_until

    _cash = float(cash)
    _positions = {}
    _avg_cost = {}
    _prices = {}
    _close_history = {}
    _volume_history = {}
    _last_seen = {}
    _pending_orders = []
    _step = 0
    _last_rebalance = -10**9
    _blocked_until = {}


def _valid_price(x):
    try:
        return np.isfinite(x) and float(x) > 0.0
    except (TypeError, ValueError):
        return False


def _current_rows(history):
    """
    Return only the rows belonging to the most recent timestamp.

    history is sorted by timestamp, so searchsorted avoids scanning the entire
    historical DataFrame with a boolean mask at every callback.
    """
    if history is None or len(history) == 0:
        return history

    last_timestamp = history["timestamp"].iloc[-1]
    try:
        first_row = int(history["timestamp"].searchsorted(last_timestamp, side="left"))
        return history.iloc[first_row:]
    except Exception:
        # Conservative fallback for unusual timestamp dtypes.
        return history[history["timestamp"] == last_timestamp]


def _apply_pending_orders(current_opens):
    """
    Reconstruct fills of orders submitted on the previous callback.

    Orders execute at this bar's open. Orders are processed in the same order
    returned by on_bar: sells are deliberately placed before buys.
    """
    global _cash, _positions, _avg_cost, _pending_orders

    if not _pending_orders:
        return

    for ticker, direction, requested_size in _pending_orders:
        requested_size = int(requested_size)
        if requested_size <= 0:
            continue

        open_price = current_opens.get(ticker)
        if not _valid_price(open_price):
            continue

        open_price = float(open_price)

        if direction == "SELL":
            owned = int(_positions.get(ticker, 0))
            filled = min(owned, requested_size)

            if filled > 0:
                _cash += filled * open_price
                remaining = owned - filled

                if remaining > 0:
                    _positions[ticker] = remaining
                else:
                    _positions.pop(ticker, None)
                    _avg_cost.pop(ticker, None)

        elif direction == "BUY":
            affordable = int(math.floor(_cash / open_price))
            filled = min(requested_size, affordable)

            if filled > 0:
                old_shares = int(_positions.get(ticker, 0))
                old_cost = float(_avg_cost.get(ticker, 0.0))

                new_shares = old_shares + filled
                if new_shares > 0:
                    _avg_cost[ticker] = (
                        old_shares * old_cost + filled * open_price
                    ) / new_shares

                _positions[ticker] = new_shares
                _cash -= filled * open_price

    # An unfilled remainder is discarded by the platform, not carried forward.
    _pending_orders = []


def _feature_for_ticker(ticker):
    """
    Produce momentum, volatility, trend, and liquidity features.

    Returns None until a ticker has enough observed bars.
    """
    closes = _close_history.get(ticker)
    volumes = _volume_history.get(ticker)

    if closes is None or volumes is None or len(closes) < 65 or len(volumes) < 15:
        return None

    c = np.asarray(closes, dtype=float)
    v = np.asarray(volumes, dtype=float)

    if not np.all(np.isfinite(c[-65:])) or c[-1] <= 0:
        return None

    current = float(c[-1])
    prior_60 = float(c[-61])
    prior_20 = float(c[-21])

    if prior_60 <= 0 or prior_20 <= 0:
        return None

    # Daily return volatility over approximately one trading month.
    recent = c[-22:]
    returns = np.diff(recent) / recent[:-1]
    volatility = float(np.std(returns, ddof=1))

    if not np.isfinite(volatility) or volatility <= 0:
        return None

    sma_50 = float(np.mean(c[-50:]))
    sma_60 = float(np.mean(c[-60:]))

    # Median dollar volume is less sensitive to isolated volume spikes.
    recent_volume = v[-20:]
    recent_volume = recent_volume[np.isfinite(recent_volume)]
    if len(recent_volume) == 0:
        return None

    dollar_volume = float(np.median(recent_volume) * current)

    return {
        "price": current,
        "ret_60": current / prior_60 - 1.0,
        "ret_20": current / prior_20 - 1.0,
        "vol": volatility,
        "sma_50": sma_50,
        "sma_60": sma_60,
        "dollar_volume": dollar_volume,
    }


def _portfolio_value():
    """
    Mark internal holdings to their most recently observed closing prices.
    """
    value = float(_cash)

    for ticker, shares in _positions.items():
        price = _prices.get(ticker)
        if _valid_price(price):
            value += int(shares) * float(price)

    return value


def _make_rebalance_orders(current_tickers):
    """
    Cross-sectional momentum portfolio construction.

    Selection:
      * Require positive 60-bar momentum and price above its 60-bar average.
      * Prefer liquid names.
      * Rank by 60-bar and 20-bar risk-adjusted momentum.
      * Hold up to eight names.
      * Stay mostly in cash when the broad cross-section is weak.

    Sizing:
      * Inverse-volatility weights.
      * 85% maximum invested capital, retaining a cash buffer for opening gaps.
      * Sell orders are returned before buy orders.
    """
    global _last_rebalance

    features = {}
    broad_features = []

    for ticker in current_tickers:
        if _blocked_until.get(ticker, -1) > _step:
            continue

        feat = _feature_for_ticker(ticker)
        if feat is None:
            continue

        if feat["dollar_volume"] <= 0:
            continue

        broad_features.append((ticker, feat))

    # Broad market / universe risk filter.
    if len(broad_features) >= 4:
        above_50 = sum(
            1 for _, f in broad_features if f["price"] > f["sma_50"]
        ) / len(broad_features)
        median_momentum = float(np.median([f["ret_60"] for _, f in broad_features]))
        risk_on = above_50 >= 0.45 and median_momentum > -0.01
    else:
        risk_on = False

    selected = []

    if risk_on and broad_features:
        # Limit the candidate pool to the more liquid portion of the universe.
        by_liquidity = sorted(
            broad_features,
            key=lambda item: item[1]["dollar_volume"],
            reverse=True,
        )
        liquid_count = min(
            len(by_liquidity),
            max(10, int(math.ceil(len(by_liquidity) * 0.70))),
        )
        liquid = by_liquidity[:liquid_count]

        for ticker, feat in liquid:
            if feat["ret_60"] <= 0.0:
                continue
            if feat["price"] <= feat["sma_60"]:
                continue

            # Volatility flooring prevents one unusually quiet series from
            # consuming nearly all portfolio weight.
            effective_vol = max(0.008, min(feat["vol"], 0.12))
            score = (0.70 * feat["ret_60"] + 0.30 * feat["ret_20"]) / effective_vol
            features[ticker] = (feat, score)

        ranked = sorted(
            features.items(),
            key=lambda item: item[1][1],
            reverse=True,
        )
        selected = [ticker for ticker, _ in ranked[:8]]

    portfolio_value = _portfolio_value()
    if portfolio_value <= 0:
        return []

    # Inverse-volatility target weights.
    target_weights = {}
    if selected:
        inv_vol = []
        for ticker in selected:
            vol = features[ticker][0]["vol"]
            effective_vol = max(0.008, min(vol, 0.12))
            inv_vol.append(1.0 / effective_vol)

        inv_vol = np.asarray(inv_vol, dtype=float)
        inv_vol /= inv_vol.sum()

        for ticker, weight in zip(selected, inv_vol):
            target_weights[ticker] = float(weight)

    # Keep 15% cash. It makes internal accounting more robust against large
    # overnight gaps between the signal close and the execution open.
    invested_fraction = 0.85
    target_shares = {}

    for ticker in selected:
        price = _prices.get(ticker)
        if not _valid_price(price):
            continue

        target_dollars = portfolio_value * invested_fraction * target_weights[ticker]
        shares = int(math.floor(target_dollars / float(price)))
        if shares > 0:
            target_shares[ticker] = shares

    sell_orders = []
    buy_orders = []

    # First reduce positions that exceed target, including names no longer
    # selected. Sells come first so their cash can fund buys at the next open.
    all_position_tickers = list(_positions.keys())
    for ticker in all_position_tickers:
        owned = int(_positions.get(ticker, 0))
        desired = int(target_shares.get(ticker, 0))

        if owned > desired:
            sell_orders.append((ticker, "SELL", owned - desired))

    for ticker, desired in target_shares.items():
        owned = int(_positions.get(ticker, 0))
        if desired > owned:
            buy_orders.append((ticker, "BUY", desired - owned))

    _last_rebalance = _step
    return sell_orders + buy_orders


def _make_defensive_exit_orders(current_tickers):
    """
    Between scheduled rebalances, exit a holding when its trend materially
    breaks. A short cooldown reduces immediate re-entry whipsaw.
    """
    orders = []

    for ticker in list(_positions.keys()):
        if ticker not in current_tickers:
            continue

        shares = int(_positions.get(ticker, 0))
        if shares <= 0:
            continue

        feat = _feature_for_ticker(ticker)
        if feat is None:
            continue

        closes = _close_history.get(ticker)
        if closes is None or len(closes) < 60:
            continue

        trailing_high = max(closes[-60:])
        price = feat["price"]

        # Exit after a meaningful moving-average break or a substantial
        # drawdown from the recent high.
        trend_break = price < 0.965 * feat["sma_60"]
        drawdown_break = price < 0.88 * trailing_high

        if trend_break or drawdown_break:
            orders.append((ticker, "SELL", shares))
            _blocked_until[ticker] = _step + 15

    return orders


def on_bar(history):
    """
    Main platform callback.

    Orders returned here execute at the next bar's open.
    """
    global _step, _pending_orders

    if history is None or len(history) == 0:
        return []

    current = _current_rows(history)
    if current is None or len(current) == 0:
        return []

    # Use current opens to reconstruct fills from the previous callback.
    current_opens = {}
    for row in current.itertuples(index=False):
        ticker = getattr(row, "ticker")
        open_price = getattr(row, "open")
        if _valid_price(open_price):
            current_opens[ticker] = float(open_price)

    _apply_pending_orders(current_opens)

    _step += 1
    current_tickers = set()

    # Update rolling per-ticker histories with the current completed bars.
    for row in current.itertuples(index=False):
        ticker = getattr(row, "ticker")
        close_price = getattr(row, "close")
        volume = getattr(row, "volume")

        if not _valid_price(close_price):
            continue

        close_price = float(close_price)
        try:
            volume = float(volume)
        except (TypeError, ValueError):
            volume = 0.0

        if not np.isfinite(volume) or volume < 0:
            volume = 0.0

        if ticker not in _close_history:
            _close_history[ticker] = deque(maxlen=130)
            _volume_history[ticker] = deque(maxlen=25)

        _close_history[ticker].append(close_price)
        _volume_history[ticker].append(volume)
        _prices[ticker] = close_price
        _last_seen[ticker] = _step
        current_tickers.add(ticker)

    # Wait until enough data exists for the 60-bar signals.
    if _step < 65:
        return []

    # Rebalance every ten bars. Stop-loss / trend exits are checked on the
    # intervening bars.
    if (_step - _last_rebalance) >= 10:
        orders = _make_rebalance_orders(current_tickers)
    else:
        orders = _make_defensive_exit_orders(current_tickers)

    # Store submitted orders so next callback can reconstruct their fills.
    # Filter defensively to guarantee the required order contract.
    clean_orders = []
    for ticker, direction, size in orders:
        size = int(size)
        if direction in ("BUY", "SELL") and size > 0:
            clean_orders.append((ticker, direction, size))

    _pending_orders = clean_orders
    return clean_orders
