import numpy as np
import pandas as pd

# Global state variables persistent across on_bar calls
_CASH = 100000.0
_POSITIONS = {}        # Maps ticker (str) -> quantity (int)
_PENDING_ORDERS = []   # Orders submitted at bar t to execute at bar t+1


def init(cash: float) -> None:
    """
    Called exactly once before the first bar. Initializes cash balance and portfolio tracking.
    """
    global _CASH, _POSITIONS, _PENDING_ORDERS
    _CASH = float(cash)
    _POSITIONS = {}
    _PENDING_ORDERS = []


def on_bar(history: pd.DataFrame) -> list:
    """
    Called once per time step with full historical bar DataFrame.
    Returns list of order tuples: (ticker, direction, size).
    """
    global _CASH, _POSITIONS, _PENDING_ORDERS

    if history.empty:
        return []

    # Get array of timestamps to efficiently locate the current bar slice
    timestamps = history['timestamp'].values
    current_ts = timestamps[-1]

    # Binary search to find start index of the current bar step
    start_curr_bar = np.searchsorted(timestamps, current_ts, side='left')
    current_bar = history.iloc[start_curr_bar:]

    # -------------------------------------------------------------------------
    # 1. Update Internal Portfolio Accounting with Last Bar's Pending Orders
    # -------------------------------------------------------------------------
    if _PENDING_ORDERS:
        open_prices = {}
        for _, row in current_bar.iterrows():
            tk = row['ticker']
            op = row['open']
            if pd.notna(op) and op > 0:
                open_prices[tk] = float(op)

        # Process SELL orders first to credit cash before BUY execution
        sells = [o for o in _PENDING_ORDERS if o[1] == 'SELL']
        buys = [o for o in _PENDING_ORDERS if o[1] == 'BUY']

        for tk, _, size in sells:
            if tk in open_prices:
                op = open_prices[tk]
                held = _POSITIONS.get(tk, 0)
                fill_qty = min(size, held)
                if fill_qty > 0:
                    _CASH += fill_qty * op
                    _POSITIONS[tk] = held - fill_qty
                    if _POSITIONS[tk] <= 0:
                        _POSITIONS.pop(tk, None)

        for tk, _, size in buys:
            if tk in open_prices:
                op = open_prices[tk]
                max_affordable = int(_CASH // op)
                fill_qty = min(size, max_affordable)
                if fill_qty > 0:
                    _CASH -= fill_qty * op
                    _POSITIONS[tk] = _POSITIONS.get(tk, 0) + fill_qty

        _PENDING_ORDERS = []

    # -------------------------------------------------------------------------
    # 2. Extract Data Window for Indicator Calculations (Last 65 bars max)
    # -------------------------------------------------------------------------
    unique_ts = pd.unique(timestamps)
    if len(unique_ts) < 21:
        return []  # Insufficient history for 20-bar indicators

    lookback_ts = unique_ts[-65] if len(unique_ts) >= 65 else unique_ts[0]
    lookback_idx = np.searchsorted(timestamps, lookback_ts, side='left')
    recent_df = history.iloc[lookback_idx:]

    # Deduplicate to handle any duplicate rows defensively
    dedup_df = recent_df.drop_duplicates(subset=['timestamp', 'ticker'], keep='last')

    # Pivot prices and volumes for fast vectorized series operations
    pivot_close = dedup_df.pivot(index='timestamp', columns='ticker', values='close').ffill()
    pivot_vol = dedup_df.pivot(index='timestamp', columns='ticker', values='volume').fillna(0)

    if len(pivot_close) < 21:
        return []

    # -------------------------------------------------------------------------
    # 3. Factor Calculations & Asset Ranking
    # -------------------------------------------------------------------------
    returns = pivot_close.pct_change()

    # Dynamic trend filter length (50-bar SMA if available, else 20)
    sma_len = 50 if len(pivot_close) >= 50 else 20
    sma_long = pivot_close.rolling(window=sma_len).mean()

    # 20-day returns, short-term 5-day return, and 20-day volatility
    ret20 = pivot_close.pct_change(periods=20)
    ret5 = pivot_close.pct_change(periods=5)
    vol20 = returns.rolling(window=20).std()

    # Alpha Score: (1-month Momentum - 1-week Return) / Volatility
    score = (ret20 - ret5) / (vol20 + 1e-4)

    # Liquidity: 20-day average dollar volume
    dollar_vol = (pivot_close * pivot_vol).rolling(window=20).mean()

    # Snapshot at current timestamp
    latest_close = pivot_close.iloc[-1]
    latest_sma_long = sma_long.iloc[-1]
    latest_score = score.iloc[-1]
    latest_dvol = dollar_vol.iloc[-1]

    # Liquidity hurdle: top 70% of dollar volume
    valid_dvol = latest_dvol.dropna()
    dvol_threshold = valid_dvol.quantile(0.30) if len(valid_dvol) > 0 else 0

    # Trend + Liquidity + Quality conditions
    valid_mask = (
        (latest_close > latest_sma_long) &
        (latest_close > 1.0) &
        (latest_dvol >= dvol_threshold) &
        (latest_score.notna()) &
        (np.isfinite(latest_score)) &
        (latest_score > 0)
    )

    eligible_scores = latest_score[valid_mask]

    # Select top 5 candidate tickers
    if not eligible_scores.empty:
        top_tickers = eligible_scores.sort_values(ascending=False).head(5).index.tolist()
    else:
        top_tickers = []

    # -------------------------------------------------------------------------
    # 4. Target Allocation & Order Generation
    # -------------------------------------------------------------------------
    # Estimate total portfolio liquidation value
    est_equity = _CASH
    for tk, qty in list(_POSITIONS.items()):
        if tk in latest_close and pd.notna(latest_close[tk]) and latest_close[tk] > 0:
            est_equity += qty * float(latest_close[tk])

    # Target up to 92% capital deployment (leaves 8% cushion for execution gap)
    target_alloc_total = est_equity * 0.92
    alloc_per_stock = target_alloc_total / 5.0  # Max 20% equity per position slot

    target_shares = {}
    for tk in top_tickers:
        p = float(latest_close[tk])
        if p > 0:
            target_shares[tk] = int(alloc_per_stock // p)

    sell_orders = []
    buy_orders = []

    # Exits & Rebalancing for currently held positions
    for tk, held_qty in list(_POSITIONS.items()):
        if held_qty <= 0:
            continue

        tgt_qty = target_shares.get(tk, 0)
        curr_price = float(latest_close[tk]) if (tk in latest_close and pd.notna(latest_close[tk])) else 0.0

        if tk not in top_tickers or tgt_qty == 0:
            # Complete Exit
            sell_orders.append((str(tk), "SELL", int(held_qty)))
        else:
            diff = tgt_qty - held_qty
            if diff < 0:
                abs_diff = abs(diff)
                # Rebalance down if position size shifted significantly
                if (abs_diff * curr_price >= 0.10 * held_qty * curr_price) or (abs_diff >= 10):
                    sell_orders.append((str(tk), "SELL", int(abs_diff)))

    # New Entries & Position Size Adjustments
    for tk in top_tickers:
        tgt_qty = target_shares.get(tk, 0)
        held_qty = _POSITIONS.get(tk, 0)
        diff = tgt_qty - held_qty
        curr_price = float(latest_close[tk])

        if diff > 0 and curr_price > 0:
            if held_qty == 0 or (diff * curr_price >= 0.10 * tgt_qty * curr_price):
                buy_orders.append((str(tk), "BUY", int(diff)))

    # Always submit SELLs before BUYs to optimize execution cash flow
    orders = sell_orders + buy_orders
    _PENDING_ORDERS = list(orders)

    return orders
