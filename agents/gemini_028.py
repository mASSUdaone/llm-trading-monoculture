import math
import numpy as np
import pandas as pd
from collections import deque


class TradingAgent:
    """
    State manager for cross-sectional momentum strategy with market regime filter.
    """

    def __init__(self, initial_cash: float):
        self.cash = float(initial_cash)
        self.positions = {}          # ticker -> int shares
        self.pending_orders = []     # list of (ticker, direction, size)
        self.price_history = {}      # ticker -> deque of close prices
        self.last_processed_len = 0  # row offset for fast incremental history updates
        self.bar_counter = 0

    def process_bar(self, history: pd.DataFrame) -> list:
        # 1. Fast incremental ingestion of new rows
        new_data = history.iloc[self.last_processed_len:]
        self.last_processed_len = len(history)

        if new_data.empty:
            return []

        # Extract current open and close prices for available tickers
        current_opens = {}
        current_closes = {}

        for row in new_data[['ticker', 'open', 'close', 'volume']].itertuples(index=False):
            ticker, o, c, v = row.ticker, row.open, row.close, row.volume

            if pd.isna(o) or pd.isna(c) or float(o) <= 0 or float(c) <= 0:
                continue

            current_opens[ticker] = float(o)
            current_closes[ticker] = float(c)

            if ticker not in self.price_history:
                self.price_history[ticker] = deque(maxlen=100)

            self.price_history[ticker].append(float(c))

        # 2. Simulate fills for pending orders placed on previous bar
        if self.pending_orders:
            # Process SELL orders first to free up cash
            for ticker, direction, size in self.pending_orders:
                if direction == "SELL" and ticker in current_opens:
                    open_price = current_opens[ticker]
                    held = self.positions.get(ticker, 0)
                    actual_sell = min(held, size)
                    if actual_sell > 0:
                        self.positions[ticker] = held - actual_sell
                        self.cash += actual_sell * open_price
                        if self.positions[ticker] <= 0:
                            self.positions.pop(ticker, None)

            # Process BUY orders next
            for ticker, direction, size in self.pending_orders:
                if direction == "BUY" and ticker in current_opens:
                    open_price = current_opens[ticker]
                    affordable = int(self.cash // open_price)
                    actual_buy = min(size, affordable)
                    if actual_buy > 0:
                        self.positions[ticker] = self.positions.get(ticker, 0) + actual_buy
                        self.cash -= actual_buy * open_price

            self.pending_orders = []

        self.bar_counter += 1

        # Periodic rebalancing check (every 5 bars)
        if self.bar_counter % 5 != 0:
            return []

        # 3. Portfolio valuation estimate
        est_portfolio_val = self.cash
        for t, qty in self.positions.items():
            if t in current_closes:
                est_portfolio_val += qty * current_closes[t]
            elif t in self.price_history and len(self.price_history[t]) > 0:
                est_portfolio_val += qty * self.price_history[t][-1]

        # 4. Signal generation across eligible tickers
        scores = {}
        vols = {}
        trend_pass_count = 0
        total_valid = 0

        for ticker, closes_deque in self.price_history.items():
            closes = np.array(closes_deque, dtype=float)
            n = len(closes)
            if n < 20:
                continue

            total_valid += 1
            curr_close = closes[-1]
            sma20 = np.mean(closes[-20:])

            # Volatility estimate from daily returns
            rets = np.diff(closes) / closes[:-1]
            vol = float(np.std(rets)) if len(rets) > 1 else 0.02
            vol = max(vol, 0.005)
            vols[ticker] = vol

            # Momentum measures
            m20 = (curr_close - closes[-20]) / closes[-20]
            m_long = (curr_close - closes[0]) / closes[0]
            m_combo = 0.6 * m20 + 0.4 * m_long

            # Trend & Quality filter
            is_trending = (curr_close > sma20) and (m_combo > 0)
            if is_trending:
                trend_pass_count += 1
                # Risk-adjusted momentum score
                scores[ticker] = m_combo / vol
            else:
                scores[ticker] = -999.0

        if total_valid == 0:
            return []

        # 5. Market Regime Detection
        bullish_ratio = trend_pass_count / total_valid
        if bullish_ratio >= 0.35:
            target_market_alloc = 0.95
        elif bullish_ratio >= 0.20:
            target_market_alloc = 0.50
        else:
            target_market_alloc = 0.0  # Bearish market regime: move to cash

        # 6. Target Selection & Asset Sizing
        valid_candidates = [
            t for t, s in sorted(scores.items(), key=lambda x: x[1], reverse=True)
            if s > 0
        ]

        TOP_K = 5
        target_tickers = valid_candidates[:TOP_K]

        target_weights = {}
        if target_tickers and target_market_alloc > 0:
            inv_vols = {t: 1.0 / vols.get(t, 0.02) for t in target_tickers}
            total_inv_vol = sum(inv_vols.values())
            for t in target_tickers:
                w = (inv_vols[t] / total_inv_vol) * target_market_alloc
                target_weights[t] = min(w, 0.22)  # Cap single position at 22%

        # Target shares per ticker
        target_shares = {}
        for t in target_tickers:
            if t in current_closes and current_closes[t] > 0:
                target_dollars = est_portfolio_val * target_weights.get(t, 0.0)
                target_shares[t] = int(target_dollars // current_closes[t])

        # 7. Order Generation
        orders = []
        held_tickers = set(self.positions.keys())

        # Step A: Generate SELL orders
        for t in held_tickers:
            curr_qty = self.positions[t]
            tgt_qty = target_shares.get(t, 0)
            if tgt_qty < curr_qty:
                sell_qty = curr_qty - tgt_qty
                if sell_qty > 0:
                    orders.append((t, "SELL", int(sell_qty)))

        # Estimate available cash after expected sell proceeds (using 2% conservative discount)
        est_cash_after_sells = self.cash
        for t, direction, size in orders:
            if direction == "SELL":
                p = current_closes.get(
                    t,
                    self.price_history[t][-1] if t in self.price_history else 0.0
                )
                est_cash_after_sells += size * p * 0.98

        # Step B: Generate BUY orders constrained by estimated cash
        buy_candidates = []
        for t in target_tickers:
            curr_qty = self.positions.get(t, 0)
            tgt_qty = target_shares.get(t, 0)
            if tgt_qty > curr_qty:
                buy_qty = tgt_qty - curr_qty
                p = current_closes.get(t, 0.0)
                if p > 0:
                    buy_candidates.append((t, buy_qty, p))

        for t, req_qty, p in buy_candidates:
            max_affordable = int((est_cash_after_sells * 0.95) // p)
            actual_buy_qty = min(req_qty, max_affordable)
            if actual_buy_qty > 0:
                orders.append((t, "BUY", int(actual_buy_qty)))
                est_cash_after_sells -= actual_buy_qty * p

        self.pending_orders = orders
        return orders


# Global agent instance
_agent = None


def init(cash: float) -> None:
    """
    Called once by the backtest platform before the first bar.
    """
    global _agent
    _agent = TradingAgent(cash)


def on_bar(history: pd.DataFrame) -> list:
    """
    Called once per time step in chronological order.
    Returns a list of (ticker, direction, size) tuples.
    """
    global _agent
    if _agent is None:
        _agent = TradingAgent(100000.0)
    return _agent.process_bar(history)
