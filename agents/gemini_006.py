import numpy as np
import pandas as pd

# Global state variables
my_cash = 0.0
my_positions = {}
pending_orders = []
step_counter = 0


def init(cash: float):
    """
    Called once before the first bar.
    Initializes internal portfolio state.
    """
    global my_cash, my_positions, pending_orders, step_counter
    my_cash = float(cash)
    my_positions = {}
    pending_orders = []
    step_counter = 0


def _process_pending_orders(latest_bar_df: pd.DataFrame):
    """
    Simulates the execution of orders placed in the previous bar at the current bar's open price.
    """
    global my_cash, my_positions, pending_orders

    if not pending_orders:
        return

    # Map current bar open prices
    open_prices = dict(zip(latest_bar_df["ticker"], latest_bar_df["open"]))

    # Execute SELL orders first to free up cash balance
    for ticker, direction, size in pending_orders:
        if direction == "SELL":
            curr_pos = my_positions.get(ticker, 0)
            fill_size = min(int(size), curr_pos)
            if fill_size > 0 and ticker in open_prices:
                price = open_prices[ticker]
                if not np.isnan(price) and price > 0:
                    my_cash += fill_size * price
                    my_positions[ticker] -= fill_size
                    if my_positions[ticker] <= 0:
                        del my_positions[ticker]

    # Execute BUY orders second
    for ticker, direction, size in pending_orders:
        if direction == "BUY":
            if ticker in open_prices:
                price = open_prices[ticker]
                if not np.isnan(price) and price > 0:
                    max_affordable = int(my_cash // price)
                    fill_size = min(int(size), max_affordable)
                    if fill_size > 0:
                        my_cash -= fill_size * price
                        my_positions[ticker] = (
                            my_positions.get(ticker, 0) + fill_size
                        )

    # Reset pending orders after fill simulation
    pending_orders = []


def on_bar(history: pd.DataFrame) -> list:
    """
    Called once per time step with history up to current bar.
    Returns list of orders as (ticker, direction, size).
    """
    global my_cash, my_positions, pending_orders, step_counter

    # Extract unique timestamps and slice recent history for optimal speed
    unique_ts = history["timestamp"].unique()
    latest_ts = unique_ts[-1]
    latest_bar_df = history[history["timestamp"] == latest_ts]

    # 1. Update internal state by filling pending orders from previous bar
    _process_pending_orders(latest_bar_df)

    step_counter += 1

    # Keep a window of up to 220 bars to optimize memory and computation time
    max_lookback = 220
    if len(unique_ts) > max_lookback:
        cutoff_ts = unique_ts[-max_lookback]
        recent_history = history[history["timestamp"] >= cutoff_ts]
    else:
        recent_history = history

    # Pivot to get a time-series of close prices (rows: timestamps, columns: tickers)
    close_prices = recent_history.pivot(
        index="timestamp", columns="ticker", values="close"
    )

    num_bars = len(close_prices)
    if num_bars < 20:
        # Not enough historical data to generate signals yet
        return []

    # Current available market prices at bar close
    current_closes = close_prices.iloc[-1].dropna()

    # Calculate alpha factors
    # 21-bar and 63-bar momentum
    lb_short = min(21, num_bars - 1)
    lb_long = min(63, num_bars - 1)

    ret_short = (close_prices.iloc[-1] / close_prices.iloc[-1 - lb_short]) - 1.0
    ret_long = (close_prices.iloc[-1] / close_prices.iloc[-1 - lb_long]) - 1.0

    # Moving average trend filter (50-bar SMA)
    lb_sma = min(50, num_bars)
    sma_50 = close_prices.tail(lb_sma).mean()

    # 20-bar realized volatility
    daily_returns = close_prices.pct_change()
    lb_vol = min(20, num_bars)
    vol_20 = daily_returns.tail(lb_vol).std()

    # Asset selection: Price above SMA50 and positive short/long momentum
    trend_mask = close_prices.iloc[-1] > sma_50
    momentum_score = (0.5 * ret_short + 0.5 * ret_long) / (vol_20 + 1e-4)

    # Apply filters
    valid_mask = (
        trend_mask
        & (ret_short > 0)
        & (ret_long > 0)
        & (vol_20 > 0)
        & current_closes.notna()
    )
    eligible_scores = momentum_score[valid_mask].dropna()

    # Rebalance periodically (every 5 bars) or when position count drops
    target_top_k = 5
    selected_tickers = []
    if not eligible_scores.empty:
        selected_tickers = list(
            eligible_scores.nlargest(target_top_k).index
        )

    # Rebalance trigger
    should_rebalance = (step_counter % 5 == 0) or (
        set(my_positions.keys()) - set(selected_tickers)
    )

    if not should_rebalance:
        return []

    # Calculate total portfolio equity at current close prices
    total_portfolio_value = my_cash
    for ticker, shares in my_positions.items():
        if ticker in current_closes and not np.isnan(current_closes[ticker]):
            total_portfolio_value += shares * current_closes[ticker]

    # Target position sizing with 4% cash buffer for open price variance
    allocatable_value = total_portfolio_value * 0.96
    target_positions = {}

    if selected_tickers:
        weight_per_asset = 1.0 / len(selected_tickers)
        for ticker in selected_tickers:
            price = current_closes[ticker]
            if price > 0:
                target_value = allocatable_value * weight_per_asset
                target_positions[ticker] = int(target_value // price)

    # Generate BUY and SELL orders
    orders = []

    # 1. SELL orders first (to release capital)
    all_known_tickers = set(my_positions.keys()).union(target_positions.keys())
    for ticker in all_known_tickers:
        curr_shares = my_positions.get(ticker, 0)
        targ_shares = target_positions.get(ticker, 0)

        if curr_shares > targ_shares:
            sell_qty = curr_shares - targ_shares
            if sell_qty > 0:
                orders.append((ticker, "SELL", int(sell_qty)))

    # 2. BUY orders second
    for ticker in selected_tickers:
        curr_shares = my_positions.get(ticker, 0)
        targ_shares = target_positions.get(ticker, 0)

        if targ_shares > curr_shares:
            buy_qty = targ_shares - curr_shares
            if buy_qty > 0:
                orders.append((ticker, "BUY", int(buy_qty)))

    # Save pending orders to track execution at step t+1 open
    pending_orders = list(orders)
    return orders
