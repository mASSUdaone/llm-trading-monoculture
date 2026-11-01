import numpy as np
import pandas as pd

# Global state variables for internal portfolio tracking
_cash = 100000.0
_positions = {}       # ticker -> share count (int)
_pending_orders = []  # list of (ticker, direction, size)
_entry_prices = {}    # ticker -> float average entry price
_highest_prices = {}  # ticker -> float peak price since entry


def init(cash: float):
    """
    Called once before the first bar by the platform.
    Initializes portfolio state and tracking variables.
    """
    global _cash, _positions, _pending_orders, _entry_prices, _highest_prices
    _cash = float(cash)
    _positions = {}
    _pending_orders = []
    _entry_prices = {}
    _highest_prices = {}


def on_bar(history: pd.DataFrame):
    """
    Called once per timestep with full historical data up to current bar.
    Returns list of orders: [(ticker, direction, size), ...]
    """
    global _cash, _positions, _pending_orders, _entry_prices, _highest_prices

    if history.empty:
        return []

    # Identify current timestamp and latest bar data
    latest_ts = history['timestamp'].iloc[-1]
    current_bar = history[history['timestamp'] == latest_ts]

    open_prices = current_bar.set_index('ticker')['open'].to_dict()
    close_prices = current_bar.set_index('ticker')['close'].to_dict()

    # ------------------------------------------------------------------
    # Step 1: Process executed pending orders from previous bar at Open
    # ------------------------------------------------------------------
    if _pending_orders:
        # First pass: Process SELL orders to liberate cash
        for ticker, direction, size in _pending_orders:
            if direction == "SELL" and ticker in open_prices:
                exec_price = open_prices[ticker]
                held_qty = _positions.get(ticker, 0)
                actual_size = min(int(size), held_qty)

                if actual_size > 0:
                    _cash += actual_size * exec_price
                    _positions[ticker] -= actual_size
                    if _positions[ticker] <= 0:
                        _positions.pop(ticker, None)
                        _entry_prices.pop(ticker, None)
                        _highest_prices.pop(ticker, None)

        # Second pass: Process BUY orders constrained by current cash balance
        for ticker, direction, size in _pending_orders:
            if direction == "BUY" and ticker in open_prices:
                exec_price = open_prices[ticker]
                if exec_price > 0:
                    max_shares_affordable = int(_cash // exec_price)
                    actual_size = min(int(size), max_shares_affordable)

                    if actual_size > 0:
                        cost = actual_size * exec_price
                        _cash -= cost
                        prev_qty = _positions.get(ticker, 0)
                        new_qty = prev_qty + actual_size
                        
                        # Update average entry price
                        prev_total_cost = _entry_prices.get(ticker, exec_price) * prev_qty
                        _entry_prices[ticker] = (prev_total_cost + cost) / new_qty
                        _positions[ticker] = new_qty
                        _highest_prices[ticker] = max(_highest_prices.get(ticker, exec_price), exec_price)

        _pending_orders = []

    # Update highest observed prices for held positions using current bar close
    for ticker in list(_positions.keys()):
        if ticker in close_prices:
            _highest_prices[ticker] = max(_highest_prices.get(ticker, close_prices[ticker]), close_prices[ticker])

    # ------------------------------------------------------------------
    # Step 2: Compute Total Portfolio Equity
    # ------------------------------------------------------------------
    portfolio_value = _cash
    for ticker, qty in _positions.items():
        price = close_prices.get(ticker, _entry_prices.get(ticker, 0.0))
        portfolio_value += qty * price

    # ------------------------------------------------------------------
    # Step 3: Signal Generation (Tail Slice for High Performance)
    # ------------------------------------------------------------------
    unique_ts = history['timestamp'].drop_duplicates()
    lookback_window = 120
    if len(unique_ts) > lookback_window:
        recent_ts = unique_ts.iloc[-lookback_window]
        recent_df = history[history['timestamp'] >= recent_ts]
    else:
        recent_df = history

    # Pivot to timestamp x ticker matrix for close prices
    close_matrix = recent_df.pivot(index='timestamp', columns='ticker', values='close')

    if len(close_matrix) < 20:
        return []

    # Indicator calculations
    # 20-period return
    ret_20 = (close_matrix.iloc[-1] / close_matrix.iloc[max(0, len(close_matrix) - 21)]) - 1.0

    # 20-period volatility
    returns = close_matrix.pct_change()
    vol_20 = returns.iloc[-20:].std()

    # Volatility-adjusted return (Sharpe proxy)
    sharpe = ret_20 / (vol_20 + 1e-6)

    # 50-period SMA for trend filtering
    sma_len = min(50, len(close_matrix))
    sma_50 = close_matrix.iloc[-sma_len:].mean()

    # 10-period EMA for pullback detection
    ema_10 = close_matrix.ewm(span=10).mean().iloc[-1]

    # Pullback score: distance below EMA10 in an overall uptrend
    latest_close = close_matrix.iloc[-1]
    pullback = (ema_10 - latest_close) / (latest_close + 1e-6)

    # Eligible Universe filtering: Price > SMA50 and 20-period return > 0
    valid_mask = (latest_close > sma_50) & (ret_20 > 0)
    eligible_tickers = valid_mask[valid_mask].index.tolist()

    # Score calculation for eligible assets
    scores = {}
    for ticker in eligible_tickers:
        s_val = sharpe.get(ticker, -np.inf)
        p_val = pullback.get(ticker, -np.inf)
        if not np.isnan(s_val) and not np.isnan(p_val):
            scores[ticker] = 0.6 * s_val + 0.4 * p_val

    # Rank target candidates
    sorted_candidates = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    max_positions = 8
    target_candidates = set(sorted_candidates[:max_positions])

    # ------------------------------------------------------------------
    # Step 4: Generate Orders & Risk Management Rules
    # ------------------------------------------------------------------
    new_orders = []
    planned_sells = set()

    # Risk Management & Exits
    for ticker in list(_positions.keys()):
        c_price = close_prices.get(ticker, None)
        if c_price is None or c_price <= 0:
            continue

        entry_p = _entry_prices.get(ticker, c_price)
        high_p = _highest_prices.get(ticker, c_price)
        sma_p = sma_50.get(ticker, 0.0)

        # Exit conditions
        trailing_stop_hit = c_price < (high_p * 0.92)   # 8% trailing drawdown
        hard_stop_hit = c_price < (entry_p * 0.90)       # 10% hard stop loss
        trend_broken = c_price < sma_p                   # Dropped below SMA50
        no_longer_target = ticker not in target_candidates

        if trailing_stop_hit or hard_stop_hit or trend_broken or no_longer_target:
            qty = _positions[ticker]
            if qty > 0:
                new_orders.append((ticker, "SELL", qty))
                planned_sells.add(ticker)

    # Estimate expected cash after sell orders execute
    est_released_cash = sum(
        _positions[t] * close_prices.get(t, 0.0) for t in planned_sells if t in close_prices
    )
    est_available_cash = _cash + est_released_cash
    cash_budget = est_available_cash * 0.95  # Preserve 5% cash buffer

    # Calculate target allocation size per active position
    target_value_per_asset = cash_budget / max_positions

    # Entry / Rebalance Buy Orders
    for ticker in sorted_candidates[:max_positions]:
        if ticker in planned_sells:
            continue

        c_price = close_prices.get(ticker, None)
        if c_price is None or c_price <= 0:
            continue

        held_shares = _positions.get(ticker, 0)
        current_val = held_shares * c_price
        needed_val = target_value_per_asset - current_val

        if needed_val > (c_price * 2):  # Minimum order threshold
            shares_to_buy = int(needed_val // c_price)
            buy_cost = shares_to_buy * c_price

            if buy_cost <= cash_budget:
                new_orders.append((ticker, "BUY", shares_to_buy))
                cash_budget -= buy_cost

    _pending_orders = list(new_orders)
    return new_orders
