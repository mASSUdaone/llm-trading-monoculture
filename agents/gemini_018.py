import numpy as np
import pandas as pd

# Global account tracking state
INITIAL_CASH = 100000.0
current_cash = 100000.0
positions = {}  # ticker -> integer share balance
pending_orders = []  # [(ticker, direction, requested_size)]


def init(cash: float) -> None:
    """Initialize agent state before the first bar."""
    global current_cash, INITIAL_CASH, positions, pending_orders
    INITIAL_CASH = float(cash)
    current_cash = float(cash)
    positions = {}
    pending_orders = []


def on_bar(history: pd.DataFrame) -> list[tuple[str, str, int]]:
    """Process bar data, update internal state, and return new orders."""
    global current_cash, positions, pending_orders

    if history.empty:
        return []

    # Identify timestamps and current bar timestamp
    timestamps = history["timestamp"].drop_duplicates()
    curr_ts = timestamps.iloc[-1]

    # Map current bar's tick data for fast lookups
    curr_bar_df = history[history["timestamp"] == curr_ts]
    curr_bar_dict = curr_bar_df.set_index("ticker").to_dict("index")

    # -------------------------------------------------------------------------
    # 1. Process Pending Orders from Step t-1 using Step t Open Prices
    # -------------------------------------------------------------------------
    if pending_orders:
        for ticker, direction, req_size in pending_orders:
            if ticker in curr_bar_dict:
                open_price = float(curr_bar_dict[ticker]["open"])
                if open_price > 0:
                    if direction == "BUY":
                        max_affordable = int(current_cash // open_price)
                        fill_qty = min(req_size, max_affordable)
                        if fill_qty > 0:
                            current_cash -= fill_qty * open_price
                            positions[ticker] = (
                                positions.get(ticker, 0) + fill_qty
                            )
                    elif direction == "SELL":
                        held_qty = positions.get(ticker, 0)
                        fill_qty = min(req_size, held_qty)
                        if fill_qty > 0:
                            current_cash += fill_qty * open_price
                            positions[ticker] = held_qty - fill_qty
                            if positions[ticker] <= 0:
                                positions.pop(ticker, None)
        pending_orders = []

    # Clean zero or negative positions
    positions = {k: v for k, v in positions.items() if v > 0}

    # -------------------------------------------------------------------------
    # 2. Extract Sliced Data Window for Efficient Signal Computation
    # -------------------------------------------------------------------------
    lookback_window = 120
    if len(timestamps) > lookback_window:
        cutoff_ts = timestamps.iloc[-lookback_window]
        recent_df = history[history["timestamp"] >= cutoff_ts]
    else:
        recent_df = history

    active_tickers = list(curr_bar_dict.keys())
    if not active_tickers:
        return []

    # Vectorized close matrix [timestamps x tickers]
    pivoted_close = recent_df.pivot(
        index="timestamp", columns="ticker", values="close"
    )
    pivoted_close = pivoted_close.reindex(columns=active_tickers)

    n_bars = len(pivoted_close)
    if n_bars < 10:
        return []

    # -------------------------------------------------------------------------
    # 3. Alpha & Risk Signal Generation
    # -------------------------------------------------------------------------
    returns = pivoted_close.pct_change(fill_method=None)
    vol_20 = returns.tail(20).std(axis=0)

    lookback_mom1 = min(20, n_bars - 1)
    lookback_mom2 = min(60, n_bars - 1)
    lookback_rev = min(3, n_bars - 1)

    close_curr = pivoted_close.iloc[-1]
    close_mom1 = pivoted_close.iloc[-1 - lookback_mom1]
    close_mom2 = pivoted_close.iloc[-1 - lookback_mom2]
    close_rev = pivoted_close.iloc[-1 - lookback_rev]

    mom1 = (close_curr - close_mom1) / (close_mom1 + 1e-8)
    mom2 = (close_curr - close_mom2) / (close_mom2 + 1e-8)
    rev = (close_curr - close_rev) / (close_rev + 1e-8)

    vol_safe = vol_20.fillna(vol_20.median()).replace(0, 1e-4)
    sig_mom1 = mom1 / (vol_safe + 1e-4)
    sig_mom2 = mom2 / (vol_safe + 1e-4)
    sig_rev = rev / (vol_safe + 1e-4)

    # Composite Signal: Momentum with Short-Term Pullback Bonus
    composite_signal = 0.5 * sig_mom1 + 0.5 * sig_mom2 - 0.3 * sig_rev
    composite_signal = composite_signal.dropna()

    if composite_signal.empty:
        return []

    # Estimate Portfolio Value
    est_portfolio_val = current_cash
    for t, qty in positions.items():
        if t in curr_bar_dict:
            est_portfolio_val += qty * curr_bar_dict[t]["close"]

    # Top Universe Selection
    positive_signals = composite_signal[composite_signal > 0]
    max_holdings = max(3, min(10, len(active_tickers) // 2))

    if not positive_signals.empty:
        top_tickers = positive_signals.nlargest(max_holdings)
    else:
        top_tickers = pd.Series(dtype=float)

    # Volatility Parity Position Sizing
    target_shares = {}
    if not top_tickers.empty:
        inv_vols = 1.0 / (
            vol_safe.reindex(top_tickers.index).fillna(0.02) + 1e-4
        )
        weights = inv_vols / inv_vols.sum()
        usable_capital = est_portfolio_val * 0.97  # 3% cash safety buffer

        for ticker, weight in weights.items():
            if ticker in curr_bar_dict:
                c_price = curr_bar_dict[ticker]["close"]
                if c_price > 0:
                    alloc_cash = usable_capital * weight
                    target_shares[ticker] = int(alloc_cash // c_price)

    # -------------------------------------------------------------------------
    # 4. Generate & Cash-Budget Orders
    # -------------------------------------------------------------------------
    all_tickers = set(positions.keys()).union(target_shares.keys())
    sells = []
    buys = []

    for ticker in all_tickers:
        curr_qty = positions.get(ticker, 0)
        tgt_qty = target_shares.get(ticker, 0)

        if curr_qty > tgt_qty:
            sell_qty = curr_qty - tgt_qty
            if sell_qty > 0:
                sells.append((ticker, "SELL", int(sell_qty)))
        elif tgt_qty > curr_qty:
            buy_qty = tgt_qty - curr_qty
            if buy_qty > 0:
                buys.append((ticker, "BUY", int(buy_qty)))

    orders = []
    orders.extend(sells)

    # Estimate available cash after sells
    est_cash_available = current_cash
    for ticker, _, sz in sells:
        if ticker in curr_bar_dict:
            est_cash_available += sz * curr_bar_dict[ticker]["close"]

    # Budget buys within cash constraints
    buy_cash_budget = est_cash_available * 0.97
    for ticker, _, sz in buys:
        if ticker in curr_bar_dict:
            c_price = curr_bar_dict[ticker]["close"]
            if c_price > 0:
                req_cost = sz * c_price
                if req_cost <= buy_cash_budget:
                    orders.append((ticker, "BUY", int(sz)))
                    buy_cash_budget -= req_cost
                else:
                    max_affordable_sz = int(buy_cash_budget // c_price)
                    if max_affordable_sz > 0:
                        orders.append((ticker, "BUY", max_affordable_sz))
                        buy_cash_budget -= max_affordable_sz * c_price

    # Record submitted orders for next bar fill estimation
    pending_orders = orders.copy()

    return orders
