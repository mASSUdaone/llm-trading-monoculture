import numpy as np
import pandas as pd


class CrossSectionalRiskParityAgent:
    """
    Quantitative trading agent executing a Cross-Sectional Momentum strategy
    with Inverse-Volatility Risk Parity and Moving Average Trend Filters.
    
    Includes internal order execution simulation to accurately track portfolio
    positions and cash balance in real time without external feedback.
    """

    def __init__(self):
        self.cash = 0.0
        self.positions = {}  # ticker -> int shares held
        self.pending_orders = []  # list of (ticker, direction, size)
        self.max_lookback = 120

    def init(self, cash: float) -> None:
        """Initialize starting cash balance and reset portfolio state."""
        self.cash = float(cash)
        self.positions = {}
        self.pending_orders = []

    def on_bar(self, history: pd.DataFrame) -> list:
        """
        Process the current bar history and return new orders.
        Executes pending orders from the previous bar at current open prices.
        """
        if history.empty:
            return []

        timestamps = history["timestamp"].values
        latest_ts = timestamps[-1]

        # Fast slice for current bar data using searchsorted
        start_curr_idx = np.searchsorted(timestamps, latest_ts, side="left")
        curr_bar_df = history.iloc[start_curr_idx:]

        curr_open = dict(zip(curr_bar_df["ticker"], curr_bar_df["open"]))
        curr_close = dict(zip(curr_bar_df["ticker"], curr_bar_df["close"]))

        # ------------------------------------------------------------------
        # 1. Update Internal Portfolio State (Simulate Pending Orders Fill)
        # ------------------------------------------------------------------
        if self.pending_orders:
            # Process SELLs first to realize cash gains before BUY executions
            sells = [o for o in self.pending_orders if o[1] == "SELL"]
            buys = [o for o in self.pending_orders if o[1] == "BUY"]

            for ticker, direction, size in sells:
                if ticker in curr_open and not pd.isna(curr_open[ticker]) and curr_open[ticker] > 0:
                    price = float(curr_open[ticker])
                    held = self.positions.get(ticker, 0)
                    fill_qty = min(size, held)
                    if fill_qty > 0:
                        self.cash += fill_qty * price
                        self.positions[ticker] = held - fill_qty
                        if self.positions[ticker] <= 0:
                            self.positions.pop(ticker, None)

            for ticker, direction, size in buys:
                if ticker in curr_open and not pd.isna(curr_open[ticker]) and curr_open[ticker] > 0:
                    price = float(curr_open[ticker])
                    max_affordable = int(self.cash // price)
                    fill_qty = min(size, max_affordable)
                    if fill_qty > 0:
                        self.cash -= fill_qty * price
                        self.positions[ticker] = self.positions.get(ticker, 0) + fill_qty

            self.pending_orders = []

        # ------------------------------------------------------------------
        # 2. Slice Historical Data for Fast Indicator Computation
        # ------------------------------------------------------------------
        unique_ts = pd.unique(timestamps)
        if len(unique_ts) > self.max_lookback:
            cutoff_ts = unique_ts[-self.max_lookback]
            start_lookback_idx = np.searchsorted(timestamps, cutoff_ts, side="left")
            recent_df = history.iloc[start_lookback_idx:]
        else:
            recent_df = history

        active_tickers = set(curr_bar_df["ticker"])

        # Pivot prices into contiguous matrix
        try:
            pivoted_close = recent_df.pivot_table(
                index="timestamp", columns="ticker", values="close", aggfunc="last"
            ).ffill().bfill()
        except Exception:
            return []

        n_bars = len(pivoted_close)
        if n_bars < 3:
            return []

        # ------------------------------------------------------------------
        # 3. Vectorized Factor Calculation
        # ------------------------------------------------------------------
        p_curr = pivoted_close.iloc[-1]

        idx_5 = max(0, n_bars - 5)
        idx_20 = max(0, n_bars - 20)
        idx_60 = max(0, n_bars - 60)

        p_5 = pivoted_close.iloc[idx_5]
        p_20 = pivoted_close.iloc[idx_20]
        p_60 = pivoted_close.iloc[idx_60]

        ret_5 = (p_curr - p_5) / np.maximum(p_5, 1e-5)
        ret_20 = (p_curr - p_20) / np.maximum(p_20, 1e-5)
        ret_60 = (p_curr - p_60) / np.maximum(p_60, 1e-5)

        sma_20 = pivoted_close.iloc[max(0, n_bars - 20):].mean()

        pct_changes = pivoted_close.pct_change()
        vol_20 = pct_changes.iloc[max(0, n_bars - 20):].std().fillna(0.02).replace(0, 0.02)

        def zscore(s: pd.Series) -> pd.Series:
            std = s.std()
            if pd.isna(std) or std < 1e-6:
                return s - s.mean()
            return (s - s.mean()) / std

        # Composite Factor Score: Multi-period momentum + short-term pullback filter
        composite_score = 0.5 * zscore(ret_20) + 0.3 * zscore(ret_60) - 0.2 * zscore(ret_5)

        # ------------------------------------------------------------------
        # 4. Selection & Risk Parity Capital Allocation
        # ------------------------------------------------------------------
        active_mask = composite_score.index.isin(active_tickers)
        composite_score = composite_score[active_mask]
        p_curr_active = p_curr[active_mask]
        sma_20_active = sma_20[active_mask]
        ret_20_active = ret_20[active_mask]

        valid_mask = (p_curr_active > 0) & (~p_curr_active.isna())

        # Trend filter condition
        trend_mask = valid_mask & (p_curr_active >= sma_20_active)
        selectable = composite_score[trend_mask]

        # Broad market pullback fallback
        if len(selectable) < 3:
            selectable = composite_score[valid_mask & (ret_20_active > -0.05)]
        if len(selectable) == 0:
            selectable = composite_score[valid_mask]

        num_to_select = min(6, len(selectable))
        target_shares = {}

        if num_to_select > 0:
            top_tickers = selectable.nlargest(num_to_select).index.tolist()

            # Inverse volatility weighting (Risk Parity)
            vols = vol_20.reindex(top_tickers).fillna(0.02).replace(0, 0.02).values
            inv_vols = 1.0 / vols
            weights = inv_vols / np.sum(inv_vols)

            # Concentration cap at 25% max weight per ticker
            weights = np.minimum(weights, 0.25)
            weights = (weights / np.sum(weights)) * 0.95  # Retain 5% cash buffer

            # Calculate total liquidation value
            portfolio_val = self.cash
            for t, qty in self.positions.items():
                if t in curr_close and not pd.isna(curr_close[t]):
                    portfolio_val += qty * float(curr_close[t])

            for i, ticker in enumerate(top_tickers):
                p = float(curr_close.get(ticker, p_curr.get(ticker, 0)))
                if p > 0:
                    target_alloc = portfolio_val * weights[i]
                    target_shares[ticker] = int(target_alloc // p)

        # ------------------------------------------------------------------
        # 5. Order Generation & Safety Validation
        # ------------------------------------------------------------------
        new_orders = []
        est_sell_revenue = 0.0

        # Phase A: Generate SELL orders for reduced/exited holdings
        for ticker, held_qty in list(self.positions.items()):
            desired_qty = target_shares.get(ticker, 0)
            if desired_qty < held_qty:
                sell_qty = held_qty - desired_qty
                if sell_qty > 0:
                    new_orders.append((ticker, "SELL", int(sell_qty)))
                    p = float(curr_close.get(ticker, 0))
                    if p > 0:
                        est_sell_revenue += sell_qty * p

        # Phase B: Generate BUY orders
        est_buy_cost = 0.0
        buy_orders = []
        for ticker, desired_qty in target_shares.items():
            held_qty = self.positions.get(ticker, 0)
            if desired_qty > held_qty:
                buy_qty = desired_qty - held_qty
                p = float(curr_close.get(ticker, 0))
                if buy_qty > 0 and p > 0:
                    buy_orders.append((ticker, buy_qty, p))
                    est_buy_cost += buy_qty * p

        # Scale BUY sizes dynamically if total estimated cost exceeds available cash
        avail_cash = max(0.0, (self.cash + est_sell_revenue) * 0.95)
        if est_buy_cost > avail_cash and est_buy_cost > 0:
            scale = avail_cash / est_buy_cost
            for ticker, buy_qty, p in buy_orders:
                scaled_qty = int(buy_qty * scale)
                if scaled_qty > 0:
                    new_orders.append((ticker, "BUY", scaled_qty))
        else:
            for ticker, buy_qty, p in buy_orders:
                new_orders.append((ticker, "BUY", int(buy_qty)))

        self.pending_orders = new_orders
        return new_orders


# ----------------------------------------------------------------------
# Module Level Entrypoints (Called by the Platform)
# ----------------------------------------------------------------------

_agent = CrossSectionalRiskParityAgent()


def init(cash: float) -> None:
    """Platform hook: Invoked exactly once before the first bar."""
    _agent.init(cash)


def on_bar(history: pd.DataFrame) -> list:
    """Platform hook: Invoked on every bar step in chronological order."""
    return _agent.on_bar(history)
