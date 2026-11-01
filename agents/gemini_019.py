import pandas as pd
import numpy as np


class TradingAgent:
    def __init__(self):
        self.cash = 100000.0
        self.positions = {}          # ticker -> int (shares held)
        self.pending_orders = []     # list of (ticker, direction, size)
        self.lookback = 40          # Rolling lookback window in bars
        self.min_history = 20        # Minimum required bars to compute signals
        self.max_assets = 10         # Maximum number of concurrent positions
        self.target_leverage = 0.92  # Exposure cap to maintain liquid cash buffer
        self.min_trade_val = 100.0   # Min dollar size threshold to execute a order


# Global state instance
_agent = TradingAgent()


def init(cash: float):
    """
    Initializes or resets the trading agent with starting cash balance.
    Called exactly once before the first bar.
    """
    global _agent
    _agent = TradingAgent()
    _agent.cash = float(cash)
    _agent.positions = {}
    _agent.pending_orders = []


def on_bar(history: pd.DataFrame) -> list:
    """
    Called on every timestamp with full history up to current bar.
    Returns list of orders: (ticker, "BUY"/"SELL", size)
    """
    global _agent

    if history.empty:
        return []

    current_ts = history['timestamp'].iloc[-1]

    # -------------------------------------------------------------------------
    # 1. Resolve pending orders placed on previous bar against current bar OPEN
    # -------------------------------------------------------------------------
    if _agent.pending_orders:
        current_bar_df = history[history['timestamp'] == current_ts]
        open_prices = current_bar_df.set_index('ticker')['open'].to_dict()

        sells = [o for o in _agent.pending_orders if o[1] == "SELL"]
        buys = [o for o in _agent.pending_orders if o[1] == "BUY"]

        # Process SELLs first to release cash
        for ticker, _, size in sells:
            if ticker in open_prices and not np.isnan(open_prices[ticker]):
                price = float(open_prices[ticker])
                current_pos = _agent.positions.get(ticker, 0)
                fill_qty = min(size, current_pos)
                if fill_qty > 0:
                    _agent.cash += fill_qty * price
                    _agent.positions[ticker] = current_pos - fill_qty
                    if _agent.positions[ticker] == 0:
                        del _agent.positions[ticker]

        # Process BUYs with remaining available cash
        for ticker, _, size in buys:
            if ticker in open_prices and not np.isnan(open_prices[ticker]):
                price = float(open_prices[ticker])
                if price > 0:
                    max_affordable = int(_agent.cash // price)
                    fill_qty = min(size, max_affordable)
                    if fill_qty > 0:
                        _agent.cash -= fill_qty * price
                        _agent.positions[ticker] = _agent.positions.get(ticker, 0) + fill_qty

        _agent.pending_orders = []

    # -------------------------------------------------------------------------
    # 2. Slice recent data window for calculation
    # -------------------------------------------------------------------------
    approx_rows = _agent.lookback * 1500
    recent_data = history.tail(max(5000, approx_rows))

    unique_ts = recent_data['timestamp'].unique()
    if len(unique_ts) < _agent.min_history:
        return []

    if len(unique_ts) > _agent.lookback:
        target_ts = unique_ts[-_agent.lookback:]
        recent_data = recent_data[recent_data['timestamp'].isin(target_ts)]

    close_pivot = recent_data.pivot(index='timestamp', columns='ticker', values='close')
    latest_closes = close_pivot.iloc[-1].dropna()
    valid_tickers = latest_closes.index

    close_pivot = close_pivot[valid_tickers]
    if len(close_pivot) < _agent.min_history:
        return []

    # -------------------------------------------------------------------------
    # 3. Factor Signal Scoring & Target Portfolio Allocation
    # -------------------------------------------------------------------------
    ret_5 = close_pivot.pct_change(5).iloc[-1]
    ret_20 = close_pivot.pct_change(20).iloc[-1]

    daily_returns = close_pivot.pct_change()
    vol_20 = daily_returns.iloc[-20:].std()

    sma_10 = close_pivot.iloc[-10:].mean()
    sma_20 = close_pivot.iloc[-20:].mean()

    # Risk-adjusted momentum score
    composite_mom = 0.4 * ret_5 + 0.6 * ret_20
    score = composite_mom / (vol_20 + 1e-5)

    # Filter for positive trend alignment
    trend_filter = (latest_closes > sma_20) & (sma_10 > sma_20) & (score > 0)
    candidate_tickers = trend_filter[trend_filter].index

    if len(candidate_tickers) > 0:
        candidate_scores = score[candidate_tickers]
        top_candidates = candidate_scores.nlargest(_agent.max_assets).index
    else:
        top_candidates = pd.Index([])

    # Total estimated portfolio liquid value
    current_portfolio_val = _agent.cash
    for tkr, shares in _agent.positions.items():
        if tkr in latest_closes:
            current_portfolio_val += shares * float(latest_closes[tkr])

    target_positions = {}

    if len(top_candidates) > 0:
        candidate_vols = vol_20[top_candidates].replace(0, np.nan).fillna(vol_20.mean())
        inv_vols = 1.0 / (candidate_vols + 1e-5)
        weights = inv_vols / inv_vols.sum()

        target_total_capital = current_portfolio_val * _agent.target_leverage

        for tkr in top_candidates:
            target_alloc = target_total_capital * weights[tkr]
            price = float(latest_closes[tkr])
            if price > 0:
                target_positions[tkr] = int(target_alloc // price)

    # -------------------------------------------------------------------------
    # 4. Generate Rebalancing Orders
    # -------------------------------------------------------------------------
    sell_orders = []
    buy_orders = []

    all_tickers = set(_agent.positions.keys()).union(set(target_positions.keys()))
    estimated_cash = _agent.cash

    # Generate SELL orders first to project cash release
    for tkr in all_tickers:
        curr_qty = _agent.positions.get(tkr, 0)
        targ_qty = target_positions.get(tkr, 0)
        diff = targ_qty - curr_qty

        if diff < 0:
            sell_qty = abs(diff)
            px = float(latest_closes.get(tkr, 0))
            if targ_qty == 0 or (sell_qty * px >= _agent.min_trade_val):
                actual_sell_qty = min(sell_qty, curr_qty)
                if actual_sell_qty > 0:
                    sell_orders.append((tkr, "SELL", int(actual_sell_qty)))
                    if px > 0:
                        estimated_cash += actual_sell_qty * px

    # Generate BUY orders ranked by signal score
    buy_candidates = []
    for tkr in all_tickers:
        curr_qty = _agent.positions.get(tkr, 0)
        targ_qty = target_positions.get(tkr, 0)
        diff = targ_qty - curr_qty

        if diff > 0:
            px = float(latest_closes.get(tkr, 0))
            if px > 0:
                tkr_score = float(score.get(tkr, 0))
                buy_candidates.append((tkr_score, tkr, diff, px))

    buy_candidates.sort(key=lambda x: x[0], reverse=True)

    for _, tkr, buy_qty, px in buy_candidates:
        cost = buy_qty * px
        if cost > estimated_cash:
            buy_qty = int(estimated_cash // px)

        if buy_qty > 0 and (buy_qty * px >= _agent.min_trade_val):
            buy_orders.append((tkr, "BUY", int(buy_qty)))
            estimated_cash -= buy_qty * px

    orders = sell_orders + buy_orders
    _agent.pending_orders = orders

    return orders
