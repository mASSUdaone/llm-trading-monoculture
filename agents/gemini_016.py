import numpy as np
import pandas as pd

# Module-level state tracking
_initial_cash = 100000.0
_cash = 100000.0
_shares = {}
_entry_prices = {}
_pending_orders = []

# Strategy Hyperparameters
MAX_POSITIONS = 5
LOOKBACK = 65
MIN_PRICE = 2.0
STOP_LOSS_PCT = 0.08


def init(cash: float) -> None:
    """Initialize portfolio state before the first bar."""
    global _initial_cash, _cash, _shares, _entry_prices, _pending_orders
    _initial_cash = float(cash)
    _cash = float(cash)
    _shares = {}
    _entry_prices = {}
    _pending_orders = []


def _process_pending_orders(current_bar_df: pd.DataFrame) -> None:
    """
    Simulates fill execution of step t-1 orders at step t's open prices.
    Sells execute first to release liquidity.
    """
    global _cash, _shares, _entry_prices, _pending_orders
    if not _pending_orders:
        return

    # Map ticker -> open price for current bar
    open_prices = current_bar_df.set_index('ticker')['open'].to_dict()

    sell_orders = [o for o in _pending_orders if o[1] == 'SELL']
    buy_orders = [o for o in _pending_orders if o[1] == 'BUY']

    # 1. Process Sells
    for ticker, _, qty in sell_orders:
        p_open = open_prices.get(ticker, None)
        if p_open is not None and not np.isnan(p_open) and p_open > 0:
            held = _shares.get(ticker, 0)
            actual_qty = min(int(qty), held)
            if actual_qty > 0:
                _cash += actual_qty * p_open
                _shares[ticker] = held - actual_qty
                if _shares[ticker] <= 0:
                    _shares.pop(ticker, None)
                    _entry_prices.pop(ticker, None)

    # 2. Process Buys
    for ticker, _, qty in buy_orders:
        p_open = open_prices.get(ticker, None)
        if p_open is not None and not np.isnan(p_open) and p_open > 0:
            max_qty = int(_cash // p_open)
            actual_qty = min(int(qty), max_qty)
            if actual_qty > 0:
                _cash -= actual_qty * p_open
                old_qty = _shares.get(ticker, 0)
                new_qty = old_qty + actual_qty
                _shares[ticker] = new_qty
                
                # Update weighted average entry price
                old_cost = _entry_prices.get(ticker, p_open) * old_qty
                _entry_prices[ticker] = (old_cost + actual_qty * p_open) / new_qty

    _pending_orders = []


def on_bar(history: pd.DataFrame) -> list[tuple[str, str, int]]:
    """
    Called once per time step.
    Returns list of orders [(ticker, "BUY"/"SELL", quantity), ...].
    """
    global _cash, _shares, _entry_prices, _pending_orders

    if history.empty:
        return []

    # Get current timestamp and bar rows
    timestamps = history['timestamp'].drop_duplicates().values
    current_ts = timestamps[-1]
    current_bar_df = history[history['timestamp'] == current_ts]

    # Process execution of orders submitted on previous bar
    _process_pending_orders(current_bar_df)

    # Check minimum history requirement
    if len(timestamps) < 25:
        return []

    # Slice recent history for fast computation
    recent_ts = timestamps[-LOOKBACK:]
    df_recent = history[history['timestamp'].isin(recent_ts)]

    # Pivot close prices: rows = timestamps, columns = tickers
    pivoted_close = df_recent.pivot(index='timestamp', columns='ticker', values='close')
    if pivoted_close.shape[0] < 21:
        return []

    current_prices = pivoted_close.iloc[-1]

    # Portfolio Valuation
    portfolio_value = _cash + sum(
        _shares[t] * current_prices[t]
        for t in _shares
        if t in current_prices and not np.isnan(current_prices[t])
    )

    # Technical Indicators (Vectorized)
    ret_20 = pivoted_close.iloc[-1] / pivoted_close.iloc[-21] - 1.0
    ret_5 = (
        pivoted_close.iloc[-1] / pivoted_close.iloc[-6] - 1.0
        if pivoted_close.shape[0] >= 6
        else ret_20
    )
    
    pct_changes = pivoted_close.pct_change()
    vol_20 = pct_changes.iloc[-20:].std(axis=0)
    sma_20 = pivoted_close.iloc[-20:].mean(axis=0)

    # Vectorized 14-period RSI
    delta = pivoted_close.diff().iloc[-14:]
    gain = np.where(delta > 0, delta, 0.0).mean(axis=0)
    loss = np.where(delta < 0, -delta, 0.0).mean(axis=0)
    rs = gain / (loss + 1e-8)
    rsi = pd.Series(100.0 - (100.0 / (1.0 + rs)), index=pivoted_close.columns)

    # Signal Composite Calculation
    sharpe_20 = ret_20 / (vol_20 + 1e-5)
    
    valid_mask = (
        (current_prices >= MIN_PRICE) &
        (current_prices > sma_20) &
        (ret_20 > 0) &
        (rsi < 70) &
        (~current_prices.isna()) &
        (~sharpe_20.isna())
    )

    scores = sharpe_20.rank(pct=True) * 0.6 + ret_5.rank(pct=True) * 0.4
    valid_scores = scores[valid_mask]
    top_candidates = valid_scores.nlargest(MAX_POSITIONS).index.tolist()

    orders = []

    # 1. Determine Exits (SELL)
    for ticker in list(_shares.keys()):
        if ticker not in current_prices or np.isnan(current_prices[ticker]):
            continue

        c_price = current_prices[ticker]
        entry_p = _entry_prices.get(ticker, c_price)
        holding_qty = _shares[ticker]

        # Exit criteria: Stop Loss, Trend Breakdown, Reversal, or Out of Top Scores
        exit_signal = False
        if c_price < entry_p * (1.0 - STOP_LOSS_PCT):
            exit_signal = True
        elif ticker in sma_20 and c_price < sma_20[ticker]:
            exit_signal = True
        elif ticker in ret_5 and ret_5[ticker] < -0.06:
            exit_signal = True

        if exit_signal and holding_qty > 0:
            orders.append((ticker, "SELL", int(holding_qty)))

    # Estimate available cash assuming pending SELLs fill at current prices
    estimated_cash = _cash + sum(
        qty * current_prices[t]
        for t, direction, qty in orders
        if direction == "SELL" and t in current_prices
    )

    # 2. Determine Entries / Rebalance (BUY)
    target_pos_value = portfolio_value / MAX_POSITIONS

    for ticker in top_candidates:
        c_price = current_prices[ticker]
        if c_price <= 0 or np.isnan(c_price):
            continue

        held_qty = _shares.get(ticker, 0)
        
        # Subtract shares already queued for sale
        sell_qty_queued = sum(
            q for t, d, q in orders if t == ticker and d == "SELL"
        )
        effective_held = held_qty - sell_qty_queued

        target_qty = int(target_pos_value // c_price)
        diff_qty = target_qty - effective_held

        if diff_qty > 0:
            required_cash = diff_qty * c_price
            if required_cash <= estimated_cash:
                orders.append((ticker, "BUY", int(diff_qty)))
                estimated_cash -= required_cash

    _pending_orders = list(orders)
    return orders
