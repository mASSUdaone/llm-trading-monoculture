import numpy as np
import pandas as pd

# Global state variables
CASH = 100000.0
POSITIONS = {}        # ticker -> int shares
PENDING_ORDERS = []   # list of (ticker, direction, size)


def init(cash: float):
    """
    Called once before the first bar. Initializes cash and state.
    """
    global CASH, POSITIONS, PENDING_ORDERS
    CASH = float(cash)
    POSITIONS = {}
    PENDING_ORDERS = []


def zscore(series: pd.Series) -> pd.Series:
    """
    Cross-sectional z-score standardization.
    """
    valid = series.dropna()
    if len(valid) < 2:
        return pd.Series(0.0, index=series.index)
    std = valid.std()
    if pd.isna(std) or std < 1e-8:
        return pd.Series(0.0, index=series.index)
    mean = valid.mean()
    res = (series - mean) / std
    return res.fillna(0.0)


def on_bar(history: pd.DataFrame) -> list[tuple[str, str, int]]:
    """
    Called once per time step in chronological order.
    Returns list of orders: [(ticker, direction, size), ...]
    """
    global CASH, POSITIONS, PENDING_ORDERS

    if history is None or history.empty:
        return []

    # 1. Update internal portfolio state by simulating execution of previous step's orders
    current_timestamp = history['timestamp'].iloc[-1]
    
    # Fast slice of recent rows for current bar prices
    history_tail = history.tail(min(len(history), 2000))
    current_bar_df = history_tail[history_tail['timestamp'] == current_timestamp]

    current_opens = {}
    current_closes = {}
    for _, row in current_bar_df.iterrows():
        t, o, c = row['ticker'], row['open'], row['close']
        if pd.notna(o) and o > 0:
            current_opens[t] = float(o)
        if pd.notna(c) and c > 0:
            current_closes[t] = float(c)

    if PENDING_ORDERS:
        for ticker, direction, size in PENDING_ORDERS:
            if ticker in current_opens:
                exec_price = current_opens[ticker]
                if direction == "BUY":
                    max_affordable = int(CASH // exec_price)
                    fill_qty = min(int(size), max_affordable)
                    if fill_qty > 0:
                        CASH -= fill_qty * exec_price
                        POSITIONS[ticker] = POSITIONS.get(ticker, 0) + fill_qty
                elif direction == "SELL":
                    curr_qty = POSITIONS.get(ticker, 0)
                    fill_qty = min(int(size), curr_qty)
                    if fill_qty > 0:
                        CASH += fill_qty * exec_price
                        POSITIONS[ticker] = curr_qty - fill_qty
                        if POSITIONS[ticker] <= 0:
                            POSITIONS.pop(ticker, None)
        PENDING_ORDERS = []

    # 2. Compute current portfolio market value
    portfolio_value = CASH
    for ticker, qty in POSITIONS.items():
        price = current_closes.get(ticker, 0.0)
        if price > 0:
            portfolio_value += qty * price

    if portfolio_value <= 0:
        return []

    # 3. Alpha Factor Calculation
    recent_data = history.tail(25000)

    try:
        if recent_data.duplicated(subset=['timestamp', 'ticker']).any():
            pivot_close = recent_data.pivot_table(index='timestamp', columns='ticker', values='close', aggfunc='last')
        else:
            pivot_close = recent_data.pivot(index='timestamp', columns='ticker', values='close')
    except Exception:
        return []

    n_bars = len(pivot_close)
    if n_bars < 21:
        return []  # Wait until enough historical bars are available

    latest_close_series = pivot_close.iloc[-1].dropna()

    # Factor 1: 20-bar Momentum
    ret_20 = (pivot_close.iloc[-1] / pivot_close.iloc[-21]) - 1.0

    # Factor 2: 3-bar Reversal
    ret_3 = (pivot_close.iloc[-1] / pivot_close.iloc[-4]) - 1.0

    # Factor 3: 20-bar Return Volatility
    returns = pivot_close.pct_change()
    vol_20 = returns.tail(20).std()

    # Align factors across available universe
    valid_tickers = latest_close_series.index.intersection(vol_20.dropna().index)
    if len(valid_tickers) == 0:
        return []

    mom_series = ret_20.reindex(valid_tickers)
    rev_series = -ret_3.reindex(valid_tickers)
    vol_series = vol_20.reindex(valid_tickers)

    # Standardized composite score
    mom_z = zscore(mom_series)
    rev_z = zscore(rev_series)
    combined_score = 0.6 * mom_z + 0.4 * rev_z

    # Filter eligible positive alpha assets
    mask = (combined_score > 0) & (vol_series > 1e-5) & (latest_close_series.reindex(valid_tickers) > 0)
    eligible_scores = combined_score[mask]

    target_positions = {}

    if not eligible_scores.empty:
        # Select top K assets
        max_assets = min(5, len(eligible_scores))
        top_tickers = eligible_scores.nlargest(max_assets).index

        # Inverse-volatility risk weighting
        weights = eligible_scores.loc[top_tickers] / vol_series.loc[top_tickers]
        if weights.sum() > 0:
            weights = 0.95 * (weights / weights.sum())

            for ticker, weight in weights.items():
                p = latest_close_series.loc[ticker]
                if pd.notna(p) and p > 0:
                    target_dollar = portfolio_value * weight
                    target_shares = int(target_dollar // p)
                    if target_shares > 0:
                        target_positions[ticker] = target_shares

    # 4. Generate Orders (SELLs first, then BUYs)
    orders = []
    expected_cash_after_sells = CASH

    # Process SELLs
    for ticker, curr_qty in list(POSITIONS.items()):
        target_qty = target_positions.get(ticker, 0)
        p = current_closes.get(ticker, latest_close_series.get(ticker, 0.0))

        if target_qty < curr_qty:
            qty_to_sell = curr_qty - target_qty
            if qty_to_sell > 0:
                orders.append((ticker, "SELL", int(qty_to_sell)))
                if p > 0:
                    expected_cash_after_sells += qty_to_sell * p

    # Process BUYs within available budget
    available_buy_budget = expected_cash_after_sells * 0.95

    for ticker, target_qty in target_positions.items():
        curr_qty = POSITIONS.get(ticker, 0)

        if target_qty > curr_qty:
            diff = target_qty - curr_qty
            # Rebalance condition to reduce turnover
            if curr_qty == 0 or (diff / float(curr_qty)) >= 0.10:
                p = current_closes.get(ticker, latest_close_series.get(ticker, 0.0))
                if p > 0:
                    cost = diff * p
                    if cost > available_buy_budget:
                        diff = int(available_buy_budget // p)

                    if diff > 0:
                        orders.append((ticker, "BUY", int(diff)))
                        available_buy_budget -= diff * p

    # Save pending orders for next step's execution calculation
    PENDING_ORDERS = list(orders)

    return orders
