import math
from collections import defaultdict, deque

import numpy as np


# All mutable state is created by init().
_STATE = None


def init(cash):
    """
    Called once by the platform before the first bar.
    """
    global _STATE

    _STATE = {
        "cash": float(cash),
        "positions": defaultdict(int),
        "last_close": {},
        "series": defaultdict(lambda: deque(maxlen=70)),
        "pending_orders": [],
        "bar_count": 0,
    }


def _finite_positive(value):
    try:
        return math.isfinite(float(value)) and float(value) > 0.0
    except (TypeError, ValueError):
        return False


def _execute_pending_orders(current_rows):
    """
    Reconstruct fills from the known next-bar open.

    Orders returned on the prior call execute at this call's open. Since the
    platform provides no account statements, this internal ledger mirrors the
    documented execution rules: sells first if they were emitted first, then
    buys, with partial fills where necessary.
    """
    state = _STATE
    pending = state["pending_orders"]

    if not pending:
        return

    opens = {}
    for row in current_rows.itertuples(index=False):
        ticker = row.ticker
        price = row.open
        if _finite_positive(price):
            opens[ticker] = float(price)

    for ticker, direction, requested_size in pending:
        price = opens.get(ticker)
        if price is None:
            continue

        requested_size = int(requested_size)
        if requested_size <= 0:
            continue

        if direction == "SELL":
            available = int(state["positions"].get(ticker, 0))
            filled = min(requested_size, available)
            if filled > 0:
                state["positions"][ticker] -= filled
                state["cash"] += filled * price
                if state["positions"][ticker] <= 0:
                    state["positions"].pop(ticker, None)

        elif direction == "BUY":
            affordable = int(math.floor((state["cash"] + 1e-9) / price))
            filled = min(requested_size, max(0, affordable))
            if filled > 0:
                state["positions"][ticker] += filled
                state["cash"] -= filled * price

    # An order gets exactly one execution opportunity.
    state["pending_orders"] = []


def _update_market_data(current_rows):
    """
    Add current closes and volumes to compact rolling per-ticker histories.
    """
    state = _STATE

    for row in current_rows.itertuples(index=False):
        ticker = row.ticker
        close = row.close
        volume = row.volume

        if not _finite_positive(close):
            continue

        close = float(close)
        if not _finite_positive(volume):
            volume = 0.0
        else:
            volume = float(volume)

        state["series"][ticker].append((close, volume))
        state["last_close"][ticker] = close


def _portfolio_equity():
    state = _STATE
    equity = state["cash"]

    for ticker, shares in state["positions"].items():
        if shares <= 0:
            continue
        price = state["last_close"].get(ticker)
        if _finite_positive(price):
            equity += shares * price

    return max(0.0, float(equity))


def _build_targets():
    """
    Cross-sectional momentum strategy.

    A ticker qualifies when:
      * enough history exists for 20/60 bar momentum,
      * it is above its 50-bar moving average,
      * both short and medium momentum are positive.

    Candidate ranking favors high 60-bar momentum per unit of 20-bar
    volatility. A broad cross-sectional regime filter moves fully to cash
    when the market is weak.

    The selected basket is equal weighted, capped at 8 names, with a 10%
    cash reserve while invested.
    """
    state = _STATE
    candidates = []
    short_returns = []

    for ticker, values in state["series"].items():
        if len(values) < 61:
            continue

        closes = np.asarray([x[0] for x in values], dtype=float)
        volumes = np.asarray([x[1] for x in values], dtype=float)

        if not np.all(np.isfinite(closes)) or closes[-1] <= 0:
            continue

        current = closes[-1]
        close_20 = closes[-21]
        close_60 = closes[-61]

        if close_20 <= 0 or close_60 <= 0:
            continue

        ret_20 = current / close_20 - 1.0
        ret_60 = current / close_60 - 1.0
        ma_50 = float(np.mean(closes[-50:]))

        # Daily/bar-to-bar volatility from the recent 20 returns.
        daily_returns = np.diff(closes[-21:]) / closes[-21:-1]
        volatility = float(np.std(daily_returns, ddof=1)) if len(daily_returns) > 1 else 0.0

        if not math.isfinite(volatility):
            continue

        short_returns.append(ret_20)

        if current <= ma_50:
            continue
        if ret_20 <= 0.0 or ret_60 <= 0.0:
            continue

        # Avoid allowing extremely low measured volatility to dominate.
        risk = max(volatility, 0.008)

        # Dollar volume is used only as a weak liquidity preference.
        dollar_volume = float(np.mean(closes[-20:] * volumes[-20:]))
        score = ret_60 / risk

        candidates.append(
            {
                "ticker": ticker,
                "score": score,
                "liquidity": dollar_volume,
                "price": current,
            }
        )

    # Broad risk regime: if most securities have weak 20-bar momentum,
    # liquidate rather than attempting to pick relative winners in a decline.
    if len(short_returns) < 10:
        return {}

    median_return = float(np.median(short_returns))
    positive_fraction = float(np.mean(np.asarray(short_returns) > 0.0))

    risk_on = median_return > -0.015 and positive_fraction >= 0.45
    if not risk_on or not candidates:
        return {}

    # Remove the least liquid fifth only when the universe is sufficiently
    # large. This helps avoid stale or thinly traded names without requiring
    # a fixed dollar-volume threshold that may not fit every dataset.
    if len(candidates) >= 15:
        liquidity_values = np.asarray([x["liquidity"] for x in candidates], dtype=float)
        cutoff = float(np.quantile(liquidity_values, 0.20))
        candidates = [x for x in candidates if x["liquidity"] >= cutoff]

    candidates.sort(key=lambda x: (-x["score"], str(x["ticker"])))
    selected = candidates[:8]

    if not selected:
        return {}

    equity = _portfolio_equity()

    # Keep a small permanent reserve for execution gaps between close and open.
    investable_capital = equity * 0.90
    per_position_value = investable_capital / len(selected)

    targets = {}
    for item in selected:
        shares = int(math.floor(per_position_value / item["price"]))
        if shares > 0:
            targets[item["ticker"]] = shares

    return targets


def _rebalance_orders(targets):
    """
    Generate sell orders first, then buy orders.

    Selling first is important because the platform processes returned orders
    in list order, allowing sale proceeds to fund later purchases.
    """
    state = _STATE
    positions = state["positions"]

    orders = []

    # Exit names not selected, or trim names whose target became smaller.
    held_tickers = list(positions.keys())
    for ticker in sorted(held_tickers, key=str):
        held = int(positions.get(ticker, 0))
        desired = int(targets.get(ticker, 0))
        if held > desired:
            orders.append((ticker, "SELL", held - desired))

    # Add or increase selected positions.
    for ticker in sorted(targets.keys(), key=str):
        held = int(positions.get(ticker, 0))
        desired = int(targets[ticker])
        if desired > held:
            orders.append((ticker, "BUY", desired - held))

    return orders


def on_bar(history):
    """
    Called once per timestamp.

    The strategy:
      - maintains an internal cash/position ledger by simulating documented
        next-open execution rules;
      - tracks 70 bars of close/volume data per ticker;
      - rebalances every fifth bar after enough history has accumulated;
      - holds up to eight liquid, positive-trend momentum names;
      - moves fully to cash during a weak broad-market regime.
    """
    global _STATE

    if _STATE is None:
        raise RuntimeError("init(cash) must be called before on_bar(history).")

    if history is None or len(history) == 0:
        return []

    # Rows are sorted timestamp then ticker. searchsorted finds the beginning
    # of the final timestamp block without scanning all prior bars.
    current_timestamp = history.iloc[-1]["timestamp"]
    start = history["timestamp"].searchsorted(current_timestamp, side="left")
    current_rows = history.iloc[start:]

    # First, execute orders submitted on the preceding timestamp at this
    # timestamp's open.
    _execute_pending_orders(current_rows)

    # Then incorporate the just-completed bar's close into indicators.
    _update_market_data(current_rows)

    state = _STATE
    state["bar_count"] += 1

    # Need at least 61 completed closes for 60-bar momentum. Rebalance every
    # five bars to limit turnover and avoid reacting to single-bar noise.
    if state["bar_count"] < 61 or state["bar_count"] % 5 != 0:
        return []

    targets = _build_targets()
    orders = _rebalance_orders(targets)

    # The platform executes these on the next timestamp's open.
    state["pending_orders"] = orders
    return orders
