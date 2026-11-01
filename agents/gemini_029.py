"""
Autonomous Multi-Factor Quantitative Trading Agent
Target Platform: Python 3.11 with pandas and numpy
"""

import numpy as np
import pandas as pd


class FactorTradingAgent:
    """
    Quantitative trading agent utilizing cross-sectional risk-adjusted momentum,
    trend efficiency, moving-average filters, and inverse-volatility weighting.
    """

    def __init__(self):
        self.cash = 0.0
        self.positions = {}  # ticker -> int (shares held)
        self.pending_orders = []  # list of (ticker, direction, size) submitted last bar
        self.last_known_prices = {}  # ticker -> float (last close price)
        self.initialized = False

    def init(self, cash: float) -> None:
        """Initialize or reset agent state with initial cash balance."""
        self.cash = float(cash)
        self.positions = {}
        self.pending_orders = []
        self.last_known_prices = {}
        self.initialized = True

    def update_state(self, current_bar_df: pd.DataFrame) -> None:
        """
        Simulate platform execution of pending orders at the current bar's open price.
        Updates internal cash and share counts deterministically.
        """
        if not self.pending_orders:
            return

        # Extract valid open prices for current bar
        valid_opens = current_bar_df[current_bar_df['open'] > 0]
        open_prices = dict(zip(valid_opens['ticker'], valid_opens['open']))

        sells = [o for o in self.pending_orders if o[1] == "SELL"]
        buys = [o for o in self.pending_orders if o[1] == "BUY"]

        # Process SELL orders first to release cash
        for ticker, _, size in sells:
            if ticker in open_prices and pd.notna(open_prices[ticker]):
                price = float(open_prices[ticker])
                curr_pos = self.positions.get(ticker, 0)
                fill_qty = min(size, curr_pos)
                if fill_qty > 0:
                    new_pos = curr_pos - fill_qty
                    if new_pos > 0:
                        self.positions[ticker] = new_pos
                    else:
                        self.positions.pop(ticker, None)
                    self.cash += fill_qty * price

        # Process BUY orders second using available cash
        for ticker, _, size in buys:
            if ticker in open_prices and pd.notna(open_prices[ticker]):
                price = float(open_prices[ticker])
                max_qty_cash = max(0, int(self.cash // price))
                fill_qty = min(size, max_qty_cash)
                if fill_qty > 0:
                    self.positions[ticker] = self.positions.get(ticker, 0) + fill_qty
                    self.cash -= fill_qty * price

        self.pending_orders = []

    def on_bar(self, history: pd.DataFrame) -> list[tuple[str, str, int]]:
        """
        Called once per timestep. Processes new price data, updates state,
        computes factor signals, and outputs orders as (ticker, direction, size).
        """
        if history.empty:
            return []

        ts_series = history['timestamp']
        latest_ts = ts_series.iloc[-1]

        # Extract current bar slice to update fill simulation and last prices
        current_bar_df = history[ts_series == latest_ts]

        # 1. Update internal execution state from pending orders
        self.update_state(current_bar_df)

        # 2. Update price cache
        valid_closes = current_bar_df[current_bar_df['close'] > 0]
        self.last_known_prices.update(
            dict(zip(valid_closes['ticker'], valid_closes['close']))
        )

        # 3. Efficiently slice the recent lookback window (last 70 distinct bars)
        unique_ts = ts_series.drop_duplicates()
        n_ts = len(unique_ts)

        # Minimum required bars for signal calculation
        if n_ts < 15:
            return []

        if n_ts > 70:
            start_ts = unique_ts.iloc[-70]
            start_idx = ts_series.searchsorted(start_ts)
            window_df = history.iloc[start_idx:]
        else:
            window_df = history

        # Pivot close prices: index=timestamp, columns=ticker
        pivot_close = window_df.pivot(
            index='timestamp', columns='ticker', values='close'
        )
        pivot_filled = pivot_close.ffill()

        # Require continuous trading history in recent 15 bars
        valid_tickers = pivot_filled.iloc[-15:].notna().all(axis=0)
        df_valid = pivot_filled.loc[:, valid_tickers]

        if df_valid.empty or df_valid.shape[1] == 0:
            return []

        # 4. Factor Calculations
        price_now = df_valid.iloc[-1]
        lookback_mom = min(30, len(df_valid) - 1)
        price_past = df_valid.iloc[-1 - lookback_mom]

        # Price momentum return
        ret_mom = (price_now - price_past) / (price_past + 1e-6)

        # Realized volatility
        log_returns = np.log(df_valid / df_valid.shift(1))
        lookback_vol = min(20, len(df_valid) - 1)
        vol_20 = log_returns.iloc[-lookback_vol:].std(axis=0) + 1e-6

        # Risk-adjusted momentum (Sharpe-like ratio)
        sharpe_mom = ret_mom / vol_20

        # Trend filter (50-bar SMA)
        lookback_sma = min(50, len(df_valid))
        sma_50 = df_valid.iloc[-lookback_sma:].mean(axis=0)
        above_sma = price_now > sma_50

        # Kaufman Efficiency Ratio (net path / total path)
        step_diffs = (df_valid - df_valid.shift(1)).abs()
        total_path = step_diffs.iloc[-lookback_mom:].sum(axis=0)
        net_path = (price_now - price_past).abs()
        efficiency = net_path / (total_path + 1e-6)

        # Composite score
        score = sharpe_mom * (0.6 + 0.4 * efficiency)

        # Disqualify weak/downtrend assets
        valid_signals = above_sma & (ret_mom > 0) & (price_now > 0.5)
        score = score.where(valid_signals, -np.inf)

        # 5. Asset Selection & Weighting
        candidates = score[score > -np.inf].sort_values(ascending=False)

        if len(candidates) == 0:
            target_tickers = []
        else:
            max_holdings = min(8, max(1, len(candidates)))
            target_tickers = list(candidates.index[:max_holdings])

        target_shares = {}
        if target_tickers:
            sel_vols = vol_20[target_tickers]
            inv_vols = 1.0 / sel_vols
            weights = inv_vols / inv_vols.sum()

            # Cap individual position weights at 30%
            weights = np.minimum(weights, 0.30)
            weights = weights / weights.sum()

            # Market Breadth regime adjustment (scale exposure between 50% and 95%)
            market_breadth = (score > -np.inf).sum() / max(1, len(score))
            target_exposure = 0.50 + 0.45 * min(1.0, market_breadth * 3.0)

            # Portfolio Valuation
            current_prices_dict = price_now.to_dict()
            portfolio_val = self.cash
            for t, shares in self.positions.items():
                if shares > 0:
                    p = current_prices_dict.get(
                        t, self.last_known_prices.get(t, 0.0)
                    )
                    portfolio_val += shares * p

            allocated_cash = portfolio_val * target_exposure

            for t in target_tickers:
                w = weights[t]
                p = current_prices_dict[t]
                dollar_alloc = allocated_cash * w
                target_shares[t] = int(dollar_alloc // p)

        # 6. Order Generation
        orders = []

        # Emit SELL orders first to free up liquidity
        for ticker, curr_shares in list(self.positions.items()):
            if curr_shares <= 0:
                continue
            tgt_shares = target_shares.get(ticker, 0)
            if curr_shares > tgt_shares:
                sell_qty = curr_shares - tgt_shares
                orders.append((ticker, "SELL", int(sell_qty)))

        # Emit BUY orders second
        for ticker, tgt_shares in target_shares.items():
            curr_shares = self.positions.get(ticker, 0)
            if tgt_shares > curr_shares:
                buy_qty = tgt_shares - curr_shares
                orders.append((ticker, "BUY", int(buy_qty)))

        self.pending_orders = orders
        return orders


# Module-level agent instance
_agent = FactorTradingAgent()


def init(cash: float) -> None:
    """Platform hook called once before the first bar."""
    _agent.init(cash)


def on_bar(history: pd.DataFrame) -> list[tuple[str, str, int]]:
    """Platform hook called once per bar."""
    return _agent.on_bar(history)
