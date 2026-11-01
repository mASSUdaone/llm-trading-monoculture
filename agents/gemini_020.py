import numpy as np
import pandas as pd

# Global state variables
_initial_cash = 100000.0
_current_cash = 100000.0
_positions = {}  # ticker -> int share count
_pending_orders = []  # list of (ticker, direction, size) submitted at step t-1


def init(cash: float) -> None:
    """Initialize the agent's internal state.

    Called exactly once before the first bar.
    """
    global _initial_cash, _current_cash, _positions, _pending_orders
    _initial_cash = float(cash)
    _current_cash = float(cash)
    _positions = {}
    _pending_orders = []


def on_bar(history: pd.DataFrame) -> list[tuple[str, str, int]]:
    """Called once per step with the chronological history up to current bar.

    Returns a list of orders formatted as (ticker, direction, size).
    """
    global _current_cash, _positions, _pending_orders

    if history.empty:
        return []

    # 1. Identify current bar slice
    last_ts = history["timestamp"].iloc[-1]
    # Fast backward scan to locate the start of current bar rows
    idx = len(history) - 1
    while idx >= 0 and history["timestamp"].iloc[idx] == last_ts:
        idx -= 1
    curr_bar_df = history.iloc[idx + 1 :]

    # Map current OPEN prices for pending order execution
    open_prices = dict(zip(curr_bar_df["ticker"], curr_bar_df["open"]))

    # 2. Synchronize internal state with step t-1 orders executing at step t OPEN
    if _pending_orders:
        sells = [o for o in _pending_orders if o[1] == "SELL"]
        buys = [o for o in _pending_orders if o[1] == "BUY"]

        # Process SELLs first to release cash
        for ticker, _, size in sells:
            if ticker in open_prices and ticker in _positions:
                p_open = open_prices[ticker]
                if pd.notna(p_open) and p_open > 0:
                    actual_sell = min(size, _positions[ticker])
                    if actual_sell > 0:
                        _positions[ticker] -= actual_sell
                        _current_cash += actual_sell * p_open
                        if _positions[ticker] <= 0:
                            _positions.pop(ticker, None)

        # Process BUYs with remaining cash constraint
        for ticker, _, size in buys:
            if ticker in open_prices:
                p_open = open_prices[ticker]
                if pd.notna(p_open) and p_open > 0:
                    max_affordable = int(_current_cash // p_open)
                    actual_buy = min(size, max_affordable)
                    if actual_buy > 0:
                        _positions[ticker] = (
                            _positions.get(ticker, 0) + actual_buy
                        )
                        _current_cash -= actual_buy * p_open

        _pending_orders = []

    # Ensure cash is non-negative due to float rounding
    _current_cash = max(0.0, _current_cash)

    # 3. Slicing recent history window (max 120 bars) for fast factor computation
    # Limit tail size assuming <= 200 tickers per bar
    tail_df = history.tail(24000)
    unique_ts = tail_df["timestamp"].drop_duplicates().tail(120)
    recent_df = tail_df[tail_df["timestamp"] >= unique_ts.iloc[0]]

    # Pivot matrix: index=timestamp, columns=ticker, values=close
    piv_close = (
        recent_df.drop_duplicates(subset=["timestamp", "ticker"])
        .pivot(index="timestamp", columns="ticker", values="close")
        .ffill()
    )

    n_bars = len(piv_close)
    if n_bars < 10:
        return []

    # Dynamic lookback windows
    w_short = min(3, n_bars - 1)
    w_med = min(20, n_bars - 1)
    w_long = min(50, n_bars - 1)

    c_last = piv_close.iloc[-1]
    active_tickers = curr_bar_df["ticker"].tolist()
    c_last = c_last.reindex(active_tickers).dropna()

    if c_last.empty:
        return []

    # Calculate returns and volatility
    ret1 = piv_close.pct_change()
    vol = ret1.tail(w_med).std().reindex(c_last.index)

    c_med_prev = (
        piv_close.iloc[-w_med - 1] if n_bars > w_med else piv_close.iloc[0]
    )
    c_short_prev = (
        piv_close.iloc[-w_short - 1] if n_bars > w_short else piv_close.iloc[0]
    )

    ret_med = (c_last - c_med_prev.reindex(c_last.index)) / (
        c_med_prev.reindex(c_last.index) + 1e-8
    )
    ret_short = (c_last - c_short_prev.reindex(c_last.index)) / (
        c_short_prev.reindex(c_last.index) + 1e-8
    )
    sma_long = piv_close.tail(w_long).mean().reindex(c_last.index)

    # Filter qualified tickers (trend filter + valid data)
    valid_mask = (
        (vol > 1e-6)
        & (c_last > 0)
        & (c_last >= sma_long * 0.95)
        & vol.notna()
        & ret_med.notna()
        & ret_short.notna()
    )
    qualified = c_last.index[valid_mask].tolist()

    if not qualified:
        # Fallback filter without strict trend constraint
        valid_mask = (vol > 1e-6) & (c_last > 0) & vol.notna()
        qualified = c_last.index[valid_mask].tolist()

    if not qualified:
        return []

    # Factor computation
    vol_q = vol[qualified]
    ret_m_q = ret_med[qualified]
    ret_s_q = ret_short[qualified]
    c_q = c_last[qualified]
    sma_l_q = sma_long[qualified]

    z_mom = ret_m_q / (vol_q + 1e-8)
    z_trend = (c_q - sma_l_q) / (sma_l_q + 1e-8)
    z_rev = -ret_s_q

    def _zscore(s: pd.Series) -> pd.Series:
        if len(s) <= 1:
            return pd.Series(0.0, index=s.index)
        std_val = s.std()
        if pd.isna(std_val) or std_val < 1e-8:
            return pd.Series(0.0, index=s.index)
        return (s - s.mean()) / std_val

    composite_score = (
        0.5 * _zscore(z_mom) + 0.3 * _zscore(z_trend) + 0.2 * _zscore(z_rev)
    )

    # Select top assets
    max_portfolio_size = max(1, min(10, len(qualified)))
    top_tickers = composite_score.nlargest(max_portfolio_size).index.tolist()

    # Risk parity weighting (Inverse Volatility)
    inv_vol = 1.0 / (vol_q[top_tickers] + 1e-8)
    w_raw = inv_vol / inv_vol.sum()
    w_capped = np.minimum(w_raw, 0.25)  # Cap weight per asset at 25%

    target_total_exposure = 0.95
    if w_capped.sum() > 0:
        target_weights = (w_capped / w_capped.sum()) * target_total_exposure
    else:
        target_weights = pd.Series(
            target_total_exposure / len(top_tickers), index=top_tickers
        )

    # Estimate current total portfolio value
    est_portfolio_val = _current_cash
    for tkr, qty in _positions.items():
        if qty > 0 and tkr in c_last and pd.notna(c_last[tkr]):
            est_portfolio_val += qty * c_last[tkr]

    est_portfolio_val = max(est_portfolio_val, _current_cash)

    # Target share count per selected asset
    target_shares = {}
    for tkr in top_tickers:
        w = target_weights[tkr]
        p = c_last[tkr]
        if p > 0:
            # 1% price buffer to avoid out-of-cash on next bar open gap-ups
            target_shares[tkr] = int((est_portfolio_val * w) // (p * 1.01))

    # Evaluate target differences
    all_relevant_tickers = set(_positions.keys()).union(set(target_shares.keys()))

    sells = []
    buys = []

    for tkr in all_relevant_tickers:
        cur_qty = _positions.get(tkr, 0)
        tgt_qty = target_shares.get(tkr, 0)
        diff = tgt_qty - cur_qty

        if diff == 0:
            continue

        p = c_last.get(tkr, 0)
        if pd.isna(p) or p <= 0:
            if diff < 0 and cur_qty > 0:
                sells.append((str(tkr), "SELL", int(min(abs(diff), cur_qty))))
            continue

        trade_val = abs(diff) * p
        # Minimum trade threshold to avoid churn
        if trade_val < 100.0 and tgt_qty != 0:
            continue

        if diff < 0:
            sell_qty = min(abs(diff), cur_qty)
            if sell_qty > 0:
                sells.append((str(tkr), "SELL", int(sell_qty)))
        else:
            buys.append((str(tkr), "BUY", int(diff)))

    # Order priority: SELLs first, then BUYs
    orders = sells + buys
    _pending_orders = orders

    return orders
