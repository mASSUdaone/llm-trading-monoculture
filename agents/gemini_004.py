"""
Quantitative Trading Agent Module

Strategy Architecture:
- Cross-Sectional Risk-Adjusted Momentum with Trend Filtering & Inverse Volatility Weighting.
- Internal Portfolio State Tracking: Simulates execution at next-bar open prices to track 
  cash, position holdings, and pending orders accurately without official platform fill reports.
- Risk & Execution Controls: Trailing trend breakdown exits, max position cap (25%), 
  cash buffer reserve (5%), sell-first order prioritization, and strict data validation.
- Optimization: Fast tail-slicing on incoming DataFrames to maintain O(1) per-bar compute time.
"""

import numpy as np
import pandas as pd

# Module-level state tracking
_INITIAL_CASH: float = 100000.0
_CASH: float = 100000.0
_POSITIONS: dict[str, int] = {}
_PENDING_ORDERS: list[tuple[str, str, int]] = []
_BAR_COUNT: int = 0


def init(cash: float) -> None:
    """
    Called exactly once before the first bar by the platform.
    Initializes trading state and cash balances.
    """
    global _INITIAL_CASH, _CASH, _POSITIONS, _PENDING_ORDERS, _BAR_COUNT
    _INITIAL_CASH = float(cash)
    _CASH = float(cash)
    _POSITIONS = {}
    _PENDING_ORDERS = []
    _BAR_COUNT = 0


def _process_pending_orders(current_bar_df: pd.DataFrame) -> None:
    """
    Simulates execution of orders submitted at bar t-1 using bar t's open prices.
    Executes SELL orders first to free up capital before executing BUY orders.
    """
    global _CASH, _POSITIONS, _PENDING_ORDERS

    if not _PENDING_ORDERS or current_bar_df.empty:
        _PENDING_ORDERS = []
        return

    # Map available tickers to their open prices for the current timestamp
    open_prices = dict(zip(current_bar_df["ticker"], current_bar_df["open"]))

    sells = [o for o in _PENDING_ORDERS if o[1] == "SELL"]
    buys = [o for o in _PENDING_ORDERS if o[1] == "BUY"]

    # Process SELL orders
    for ticker, _, size in sells:
        if ticker in open_prices and not np.isnan(open_prices[ticker]):
            price = open_prices[ticker]
            if price > 0:
                current_qty = _POSITIONS.get(ticker, 0)
                fill_qty = min(size, current_qty)
                if fill_qty > 0:
                    _CASH += fill_qty * price
                    _POSITIONS[ticker] = current_qty - fill_qty
                    if _POSITIONS[ticker] <= 0:
                        _POSITIONS.pop(ticker, None)

    # Process BUY orders
    for ticker, _, size in buys:
        if ticker in open_prices and not np.isnan(open_prices[ticker]):
            price = open_prices[ticker]
            if price > 0:
                max_affordable = int(_CASH // price)
                fill_qty = min(size, max_affordable)
                if fill_qty > 0:
                    _CASH -= fill_qty * price
                    _POSITIONS[ticker] = _POSITIONS.get(ticker, 0) + fill_qty

    _PENDING_ORDERS = []


def on_bar(history: pd.DataFrame) -> list[tuple[str, str, int]]:
    """
    Main callback invoked per timestamp in chronological order.
    Returns a list of orders as (ticker, direction, size).
    """
    global _CASH, _POSITIONS, _PENDING_ORDERS, _BAR_COUNT

    _BAR_COUNT += 1

    if history.empty:
        return []

    # 1. Extract current bar data efficiently
    latest_ts = history["timestamp"].iat[-1]
    
    # Fast backward slice to get current bar's rows
    tail_len = min(len(history), 2000)
    history_tail = history.iloc[-tail_len:]
    current_bar_df = history_tail[history_tail["timestamp"] == latest_ts]

    # 2. Process pending orders from previous step at current open prices
    _process_pending_orders(current_bar_df)

    # Map current close prices for valuation and signal processing
    current_closes = dict(zip(current_bar_df["ticker"], current_bar_df["close"]))

    # 3. Calculate portfolio equity estimate
    portfolio_value = _CASH
    for ticker, qty in _POSITIONS.items():
        price = current_closes.get(ticker, 0.0)
        if price <= 0:
            # Fallback to last non-null close in history tail if missing in current bar
            sub = history_tail[history_tail["ticker"] == ticker]
            if not sub.empty:
                price = sub["close"].iat[-1]
        if price > 0:
            portfolio_value += qty * price

    # 4. Extract lookback window (last ~60 bars) for factor computation
    max_rows = min(len(history), 25000)
    sub_hist = history.iloc[-max_rows:]
    unique_ts = sub_hist["timestamp"].unique()
    
    lookback_bars = 60
    if len(unique_ts) > lookback_bars:
        cutoff_ts = unique_ts[-lookback_bars]
        window_df = sub_hist[sub_hist["timestamp"] >= cutoff_ts]
    else:
        window_df = sub_hist

    # 5. Compute indicators and rank candidates
    min_required_bars = 20
    candidates = []

    grouped = window_df.groupby("ticker")
    for ticker, group in grouped:
        closes = group["close"].dropna().values
        if len(closes) < min_required_bars:
            continue

        curr_close = closes[-1]
        if curr_close <= 0:
            continue

        # Moving averages
        sma_long_len = min(len(closes), 50)
        sma_short_len = min(len(closes), 20)
        
        sma_long = np.mean(closes[-sma_long_len:])
        sma_short = np.mean(closes[-sma_short_len:])

        # Momentum calculation over lookback
        mom_period = min(len(closes) - 1, 20)
        past_close = closes[-1 - mom_period]
        raw_mom = (curr_close - past_close) / past_close if past_close > 0 else 0.0

        # Rolling return volatility
        returns = np.diff(closes[-mom_period - 1:]) / closes[-mom_period - 1:-1] if len(closes) > mom_period else np.array([])
        vol = np.std(returns) if len(returns) > 1 else 0.02
        if np.isnan(vol) or vol <= 0:
            vol = 0.02

        # Filter: Long trend confirmation (Close above SMA50 and SMA20 above SMA50)
        trend_pass = (curr_close > sma_long) and (sma_short >= sma_long)

        # Risk-adjusted momentum score
        score = (raw_mom / vol) if (trend_pass and raw_mom > 0) else -999.0

        if score > 0:
            candidates.append({
                "ticker": ticker,
                "score": score,
                "vol": vol,
                "close": curr_close
            })

    # Sort candidate tickers by risk-adjusted momentum score
    candidates.sort(key=lambda x: x["score"], reverse=True)

    # 6. Target Allocation Logic
    max_holdings = min(5, len(candidates)) if candidates else 0
    selected_candidates = candidates[:max_holdings]
    selected_tickers = {c["ticker"] for c in selected_candidates}

    target_shares: dict[str, int] = {}

    if selected_candidates:
        # Inverse-volatility target weighting
        inv_vols = [1.0 / c["vol"] for c in selected_candidates]
        sum_inv_vol = sum(inv_vols)

        # Invest up to 95% of equity (keep 5% cash buffer)
        allocatable_equity = portfolio_value * 0.95

        for c, inv_v in zip(selected_candidates, inv_vols):
            ticker = c["ticker"]
            close_price = c["close"]
            
            # Weight proportional to inverse vol, capped at max 25% per position
            raw_weight = (inv_v / sum_inv_vol) if sum_inv_vol > 0 else (1.0 / len(selected_candidates))
            weight = min(0.25, raw_weight)
            
            target_cash = allocatable_equity * weight
            target_shares[ticker] = int(target_cash // close_price)

    # Tickers currently held but not selected get target 0 shares (liquidate)
    for ticker in list(_POSITIONS.keys()):
        if ticker not in selected_tickers:
            target_shares[ticker] = 0

    # 7. Generate Rebalance Orders
    orders: list[tuple[str, str, int]] = []

    # Emit SELL orders first
    for ticker, current_qty in list(_POSITIONS.items()):
        tgt = target_shares.get(ticker, 0)
        if current_qty > tgt:
            sell_qty = current_qty - tgt
            if sell_qty > 0:
                orders.append((ticker, "SELL", int(sell_qty)))

    # Emit BUY orders second
    for ticker, tgt in target_shares.items():
        current_qty = _POSITIONS.get(ticker, 0)
        if tgt > current_qty:
            buy_qty = tgt - current_qty
            if buy_qty > 0:
                orders.append((ticker, "BUY", int(buy_qty)))

    _PENDING_ORDERS = list(orders)
    return orders
