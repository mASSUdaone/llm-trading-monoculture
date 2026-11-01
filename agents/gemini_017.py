import numpy as np
import pandas as pd

# Global portfolio tracking state
GLOBAL_STATE = {
    "cash": 100000.0,
    "positions": {},  # ticker -> int (shares held)
    "pending_orders": [],  # list of (ticker, direction, size) submitted last bar
    "initialized": False,
}


def init(cash: float) -> None:
    """Initialize trading state before the first bar."""
    GLOBAL_STATE["cash"] = float(cash)
    GLOBAL_STATE["positions"] = {}
    GLOBAL_STATE["pending_orders"] = []
    GLOBAL_STATE["initialized"] = True


def on_bar(history: pd.DataFrame) -> list[tuple[str, str, int]]:
    """Process incoming market data bar and return orders for execution at next open."""
    if not GLOBAL_STATE["initialized"]:
        init(100000.0)

    if history.empty:
        return []

    # Identify the current timestamp
    latest_ts = history["timestamp"].iloc[-1]

    # Efficiently extract latest bar slice
    latest_rows = history[history["timestamp"] == latest_ts]

    # Extract available execution (open) and valuation (close) prices for current bar
    latest_opens = {}
    latest_closes = {}
    for _, row in latest_rows.iterrows():
        t = row["ticker"]
        o = float(row["open"]) if not pd.isna(row["open"]) else 0.0
        c = float(row["close"]) if not pd.isna(row["close"]) else 0.0
        if o > 0:
            latest_opens[t] = o
        if c > 0:
            latest_closes[t] = c

    # ------------------------------------------------------------------
    # 1. Update Internal Ledger (Reconcile pending orders at current OPEN)
    # ------------------------------------------------------------------
    pending = GLOBAL_STATE["pending_orders"]
    if pending:
        # Execute SELLs first to maximize available cash for BUYs
        sells = [o for o in pending if o[1] == "SELL"]
        buys = [o for o in pending if o[1] == "BUY"]

        for ticker, _, size in sells:
            if ticker in latest_opens:
                p_open = latest_opens[ticker]
                held = GLOBAL_STATE["positions"].get(ticker, 0)
                fill_qty = min(size, held)
                if fill_qty > 0:
                    GLOBAL_STATE["positions"][ticker] = held - fill_qty
                    GLOBAL_STATE["cash"] += fill_qty * p_open
                    if GLOBAL_STATE["positions"][ticker] == 0:
                        del GLOBAL_STATE["positions"][ticker]

        for ticker, _, size in buys:
            if ticker in latest_opens:
                p_open = latest_opens[ticker]
                affordable = int(GLOBAL_STATE["cash"] // p_open)
                fill_qty = min(size, affordable)
                if fill_qty > 0:
                    held = GLOBAL_STATE["positions"].get(ticker, 0)
                    GLOBAL_STATE["positions"][ticker] = held + fill_qty
                    GLOBAL_STATE["cash"] -= fill_qty * p_open

        GLOBAL_STATE["pending_orders"] = []

    # Floor cash against precision anomalies
    if GLOBAL_STATE["cash"] < 0:
        GLOBAL_STATE["cash"] = 0.0

    # ------------------------------------------------------------------
    # 2. Extract Window Data for Fast Signal Computation
    # ------------------------------------------------------------------
    unique_ts = history["timestamp"].drop_duplicates().tail(60)
    if len(unique_ts) < 15:
        return []

    sub_df = history[history["timestamp"].isin(unique_ts)]

    # Compute total portfolio valuation
    total_equity = GLOBAL_STATE["cash"]
    for ticker, pos in list(GLOBAL_STATE["positions"].items()):
        if pos > 0 and ticker in latest_closes:
            total_equity += pos * latest_closes[ticker]

    if total_equity <= 0:
        return []

    # ------------------------------------------------------------------
    # 3. Factor Scoring & Alpha Ranking
    # ------------------------------------------------------------------
    tickers = sub_df["ticker"].unique()
    scores = {}
    vols = {}

    for ticker in tickers:
        t_data = sub_df[sub_df["ticker"] == ticker]
        if len(t_data) < 15:
            continue

        c = t_data["close"].to_numpy(dtype=float)
        c = c[~np.isnan(c)]
        c = c[c > 0]

        if len(c) < 15:
            continue

        curr_price = c[-1]

        # Multi-horizon returns
        ret_20 = (c[-1] / c[-20]) - 1.0 if len(c) >= 20 else (c[-1] / c[0]) - 1.0
        ret_5 = (c[-1] / c[-5]) - 1.0 if len(c) >= 5 else (c[-1] / c[0]) - 1.0

        # Trend strength indicator
        sma_10 = np.mean(c[-10:])
        sma_base = np.mean(c)
        trend_factor = (sma_10 - sma_base) / curr_price

        # Volatility estimation (std dev of log returns)
        if len(c) > 5:
            log_returns = np.diff(np.log(c))
            vol = float(np.std(log_returns))
        else:
            vol = 0.02

        vol = max(vol, 0.002)

        # Volatility-adjusted composite score
        raw_signal = 0.50 * ret_20 - 0.25 * ret_5 + 0.50 * trend_factor
        score = raw_signal / vol

        scores[ticker] = score
        vols[ticker] = vol

    # Filter candidates with positive risk-adjusted momentum
    eligible = [t for t, s in scores.items() if s > 0 and t in latest_closes]

    # Target asset weights
    target_weights = {}

    if eligible:
        eligible.sort(key=lambda x: scores[x], reverse=True)
        num_assets = min(5, len(eligible))
        top_eligible = eligible[:num_assets]

        # Risk parity / Inverse-volatility weighting
        inv_vols = [1.0 / vols[t] for t in top_eligible]
        total_inv_vol = sum(inv_vols)

        target_invested = 0.95  # Retain 5% liquid cash buffer
        max_weight = 0.40 if num_assets >= 3 else (1.0 / num_assets)

        raw_weights = [
            (iv / total_inv_vol) * target_invested for iv in inv_vols
        ]
        capped_weights = [min(w, max_weight) for w in raw_weights]
        sum_capped = sum(capped_weights)

        if sum_capped > 0:
            final_weights = [
                (w / sum_capped) * target_invested for w in capped_weights
            ]
            for t, w in zip(top_eligible, final_weights):
                target_weights[t] = w

    # ------------------------------------------------------------------
    # 4. Target Allocation & Order Generation
    # ------------------------------------------------------------------
    orders = []
    all_tickers = set(GLOBAL_STATE["positions"].keys()).union(
        target_weights.keys()
    )

    for ticker in all_tickers:
        if ticker not in latest_closes:
            continue

        p_close = latest_closes[ticker]
        if p_close <= 0:
            continue

        target_w = target_weights.get(ticker, 0.0)
        target_val = total_equity * target_w
        target_qty = int(target_val // p_close)

        current_qty = GLOBAL_STATE["positions"].get(ticker, 0)
        diff_qty = target_qty - current_qty

        if diff_qty == 0:
            continue

        # Check rebalance noise threshold to prevent unnecessary churn
        trade_val = abs(diff_qty) * p_close
        min_trade_thresh = max(200.0, 0.015 * total_equity)

        if target_qty == 0 or trade_val >= min_trade_thresh:
            if diff_qty > 0:
                orders.append((ticker, "BUY", int(diff_qty)))
            else:
                orders.append((ticker, "SELL", int(abs(diff_qty))))

    # Save orders to internal tracking ledger
    GLOBAL_STATE["pending_orders"] = orders
    return orders
