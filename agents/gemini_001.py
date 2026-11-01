"""
Trading Agent Implementation
Python 3.11 Compatible
"""

from collections import defaultdict
import numpy as np
import pandas as pd


class PortfolioAgent:
    def __init__(self, cash: float):
        self.cash = float(cash)
        self.positions = defaultdict(int)  # ticker -> integer shares
        self.pending_orders = []           # list of (ticker, direction, size) placed at t-1

    def execute_pending_orders(self, current_bar_df: pd.DataFrame) -> None:
        """
        Executes pending orders submitted on the previous bar using current open prices.
        Update internal cash and share positions accordingly.
        """
        if not self.pending_orders:
            return

        # Map ticker to current open price
        open_prices = current_bar_df.set_index('ticker')['open'].to_dict()

        for ticker, direction, size in self.pending_orders:
            if ticker not in open_prices:
                continue

            price = open_prices[ticker]
            if pd.isna(price) or price <= 0:
                continue

            price = float(price)

            if direction == "BUY":
                max_affordable = int(self.cash // price)
                fill_qty = min(size, max_affordable)
                if fill_qty > 0:
                    cost = fill_qty * price
                    self.cash -= cost
                    self.positions[ticker] += fill_qty

            elif direction == "SELL":
                held_qty = self.positions.get(ticker, 0)
                fill_qty = min(size, held_qty)
                if fill_qty > 0:
                    revenue = fill_qty * price
                    self.cash += revenue
                    self.positions[ticker] -= fill_qty
                    if self.positions[ticker] <= 0:
                        del self.positions[ticker]

        self.pending_orders = []

    def on_bar(self, history: pd.DataFrame) -> list:
        # Fast extraction of recent history (last 60 bars) for speed optimization
        ts_values = history['timestamp'].values
        unique_ts = pd.unique(ts_values)

        if len(unique_ts) > 60:
            cutoff_ts = unique_ts[-60]
            idx = np.searchsorted(ts_values, cutoff_ts)
            recent_df = history.iloc[idx:]
        else:
            recent_df = history

        current_ts = unique_ts[-1]
        current_bar_df = recent_df[recent_df['timestamp'] == current_ts]

        # 1. Update internal state based on order fills at today's open price
        self.execute_pending_orders(current_bar_df)

        # Warm-up period check (need at least 22 bars for 20-period lookback)
        if len(unique_ts) < 22:
            return []

        # Pivot close prices to calculate factors efficiently
        try:
            pivoted_close = recent_df.pivot(index='timestamp', columns='ticker', values='close')
        except Exception:
            return []

        if pivoted_close.shape[0] < 22:
            return []

        latest_close = pivoted_close.iloc[-1].dropna()
        if latest_close.empty:
            return []

        # 2. Factor Computations
        rets = pivoted_close.pct_change()
        ret_20 = (pivoted_close.iloc[-1] / pivoted_close.iloc[-21]) - 1.0
        ret_3 = (pivoted_close.iloc[-1] / pivoted_close.iloc[-4]) - 1.0
        vol_20 = rets.iloc[-20:].std()
        sma_20 = pivoted_close.iloc[-20:].mean()

        scores = {}
        for ticker in latest_close.index:
            c = latest_close[ticker]
            s20 = sma_20.get(ticker)
            v20 = vol_20.get(ticker)
            r20 = ret_20.get(ticker)
            r3 = ret_3.get(ticker)

            if pd.isna(c) or pd.isna(s20) or pd.isna(v20) or pd.isna(r20) or pd.isna(r3):
                continue

            # Filter out non-trending assets or unstable volatility
            if c <= s20 or v20 <= 1e-5:
                continue

            # Factor score calculation
            sharpe_score = r20 / (v20 + 1e-5)
            rev_score = r3 / (v20 + 1e-5)
            score = sharpe_score - 0.5 * rev_score

            if score > 0:
                scores[ticker] = score

        # Liquidate positions if market environment produces no positive candidates
        if not scores:
            orders = []
            for ticker, qty in list(self.positions.items()):
                if qty > 0:
                    orders.append((ticker, "SELL", qty))
            self.pending_orders = orders
            return orders

        # Select top N candidates (up to 6)
        ranked_tickers = sorted(scores.keys(), key=lambda t: scores[t], reverse=True)
        top_n = min(6, len(ranked_tickers))
        selected_tickers = set(ranked_tickers[:top_n])

        # Target Portfolio Construction
        close_map = latest_close.to_dict()
        portfolio_val = self.cash + sum(qty * close_map.get(t, 0.0) for t, qty in self.positions.items())

        if portfolio_val <= 0:
            return []

        # Allocate 92% of total equity evenly among selected assets (8% cash buffer for price drift)
        target_weight = 0.92 / top_n
        target_shares = {}
        for t in selected_tickers:
            price = close_map.get(t, 0.0)
            if price > 0:
                target_shares[t] = int((portfolio_val * target_weight) // price)
            else:
                target_shares[t] = 0

        sell_orders = []
        buy_orders = []

        # Generate SELL orders first to release capital
        for t, held_qty in list(self.positions.items()):
            tgt_qty = target_shares.get(t, 0)
            if held_qty > tgt_qty:
                sell_orders.append((t, "SELL", held_qty - tgt_qty))

        # Estimate available cash after pending sells complete
        est_sell_revenue = sum(qty * close_map.get(t, 0.0) for t, _, qty in sell_orders)
        est_available_cash = max(0.0, self.cash + est_sell_revenue)

        # Generate BUY orders within cash budget
        for t in selected_tickers:
            held_qty = self.positions.get(t, 0)
            tgt_qty = target_shares.get(t, 0)
            if tgt_qty > held_qty:
                needed_qty = tgt_qty - held_qty
                price = close_map.get(t, 0.0)
                if price > 0:
                    max_affordable = int(est_available_cash // price)
                    buy_qty = min(needed_qty, max_affordable)
                    if buy_qty > 0:
                        buy_orders.append((t, "BUY", buy_qty))
                        est_available_cash -= buy_qty * price

        all_orders = sell_orders + buy_orders
        self.pending_orders = all_orders
        return all_orders


# Global instance reference required by the platform API interface
_agent_instance = None


def init(cash: float) -> None:
    """Platform initialization function called once before the first bar."""
    global _agent_instance
    _agent_instance = PortfolioAgent(cash)


def on_bar(history: pd.DataFrame) -> list:
    """Platform step handler called once per timestamp bar."""
    global _agent_instance
    if _agent_instance is None:
        return []
    return _agent_instance.on_bar(history)
