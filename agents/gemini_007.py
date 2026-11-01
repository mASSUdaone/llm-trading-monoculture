import numpy as np
import pandas as pd

# -------------------------------------------------------------------------
# Global State Management
# -------------------------------------------------------------------------
_cash = 100000.0
_positions = {}         # dict[ticker, int] -> quantity held
_pending_orders = []    # list of (ticker, direction, size) submitted on bar t

def init(cash: float):
    """
    Called once before the first bar.
    Initializes portfolio cash and resets tracking state.
    """
    global _cash, _positions, _pending_orders
    _cash = float(cash)
    _positions = {}
    _pending_orders = []

def on_bar(history: pd.DataFrame) -> list:
    """
    Called once per time step in chronological order.
    Returns a list of orders [(ticker, direction, size), ...].
    """
    global _cash, _positions, _pending_orders

    if history.empty:
        return []

    # Get the latest timestamp and bar data
    latest_ts = history['timestamp'].iloc[-1]
    current_bar = history[history['timestamp'] == latest_ts]

    # ---------------------------------------------------------------------
    # 1. Process Fills for Pending Orders Submitted on Bar t-1
    # ---------------------------------------------------------------------
    if _pending_orders:
        opens = dict(zip(current_bar['ticker'], current_bar['open']))
        
        for ticker, direction, size in _pending_orders:
            if ticker in opens and not np.isnan(opens[ticker]) and opens[ticker] > 0:
                open_p = float(opens[ticker])
                
                if direction == "BUY":
                    max_buy = int(_cash // open_p)
                    fill_qty = min(int(size), max_buy)
                    if fill_qty > 0:
                        _cash -= fill_qty * open_p
                        _positions[ticker] = _positions.get(ticker, 0) + fill_qty
                        
                elif direction == "SELL":
                    held = _positions.get(ticker, 0)
                    fill_qty = min(int(size), held)
                    if fill_qty > 0:
                        _cash += fill_qty * open_p
                        _positions[ticker] = held - fill_qty
                        if _positions[ticker] <= 0:
                            _positions.pop(ticker, None)

        _pending_orders = []

    # ---------------------------------------------------------------------
    # 2. Extract Recent Data Window for Indicator Calculations
    # ---------------------------------------------------------------------
    unique_ts = pd.unique(history['timestamp'])
    num_ts = len(unique_ts)
    
    # Require minimal warmup period (at least 10 bars)
    if num_ts < 10:
        return []

    # Slice tail history (last 120 bars max for high efficiency)
    cutoff_ts = unique_ts[-120] if num_ts >= 120 else unique_ts[0]
    recent_df = history[history['timestamp'] >= cutoff_ts]

    # Pivot price matrices
    close_pivot = recent_df.pivot(index='timestamp', columns='ticker', values='close')
    vol_pivot = recent_df.pivot(index='timestamp', columns='ticker', values='volume')

    if close_pivot.empty:
        return []

    current_closes = close_pivot.iloc[-1]

    # ---------------------------------------------------------------------
    # 3. Portfolio Net Asset Value (NAV) Estimation
    # ---------------------------------------------------------------------
    nav = _cash
    for t, qty in _positions.items():
        if t in current_closes and not np.isnan(current_closes[t]):
            nav += qty * current_closes[t]

    if nav <= 0:
        return []

    # ---------------------------------------------------------------------
    # 4. Multi-Factor Alpha Signal Generation
    # ---------------------------------------------------------------------
    n_bars = len(close_pivot)
    
    sma_len = min(50, n_bars)
    sma_50 = close_pivot.iloc[-sma_len:].mean(axis=0)

    mom_30_len = min(30, n_bars)
    mom_30 = close_pivot.iloc[-1] / close_pivot.iloc[-mom_30_len] - 1.0

    mom_90_len = min(90, n_bars)
    mom_90 = close_pivot.iloc[-1] / close_pivot.iloc[-mom_90_len] - 1.0

    ret_3_len = min(3, n_bars)
    ret_3 = close_pivot.iloc[-1] / close_pivot.iloc[-ret_3_len] - 1.0

    vol_len = min(20, n_bars)
    returns_20 = close_pivot.iloc[-vol_len:].pct_change()
    volatility = returns_20.std(axis=0)

    dollar_vol = (close_pivot * vol_pivot).iloc[-vol_len:].mean(axis=0)

    # Filter universe for trending and liquid assets
    valid_mask = (
        (current_closes > 2.0) & 
        (current_closes >= sma_50 * 0.97) & 
        (dollar_vol > 50000) & 
        (volatility > 0.0005) & 
        (~current_closes.isna())
    )

    valid_tickers = current_closes.index[valid_mask].tolist()

    target_positions = {}

    if valid_tickers:
        # Cross-sectional factor ranking
        m30_rank = mom_30[valid_tickers].rank(pct=True)
        m90_rank = mom_90[valid_tickers].rank(pct=True)
        ret3_rank = ret_3[valid_tickers].rank(pct=True)

        composite_score = m30_rank + m90_rank - 0.8 * ret3_rank

        # Select top K assets (e.g., top 10)
        top_k = min(10, len(valid_tickers))
        top_tickers = composite_score.nlargest(top_k).index.tolist()

        # Inverse-volatility weighting for position sizing
        inv_vol = 1.0 / (volatility[top_tickers] + 1e-5)
        weights = inv_vol / inv_vol.sum()

        # Cap individual weights at 15% and total exposure at 90%
        weights = np.minimum(weights, 0.15)
        tot_w = weights.sum()
        if tot_w > 0:
            weights = (weights / tot_w) * 0.90

        # Compute target share counts
        for t, w in zip(top_tickers, weights):
            p = current_closes[t]
            if p > 0:
                target_dollar = nav * w
                target_qty = int(target_dollar / p)
                if target_qty > 0:
                    target_positions[t] = target_qty

    # ---------------------------------------------------------------------
    # 5. Order Generation Logic (SELLs first, BUYs second)
    # ---------------------------------------------------------------------
    orders = []

    # Sell orders first to release cash
    for t, held_qty in list(_positions.items()):
        tgt_qty = target_positions.get(t, 0)
        if tgt_qty < held_qty:
            sell_qty = held_qty - tgt_qty
            if sell_qty > 0:
                orders.append((t, "SELL", int(sell_qty)))

    # Buy orders second using available cash
    for t, tgt_qty in target_positions.items():
        held_qty = _positions.get(t, 0)
        if tgt_qty > held_qty:
            buy_qty = tgt_qty - held_qty
            if buy_qty > 0:
                orders.append((t, "BUY", int(buy_qty)))

    _pending_orders = orders
    return orders
