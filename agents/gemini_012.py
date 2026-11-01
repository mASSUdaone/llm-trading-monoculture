import numpy as np
import pandas as pd

# Module-level state tracking
state = {
    "cash": 100000.0,
    "positions": {},       # ticker -> share_count (int)
    "pending_orders": [],  # list of (ticker, direction, size)
    "peak_prices": {},     # ticker -> max close price seen while holding
}


def init(cash: float):
    """
    Called once before the first bar by the evaluation platform.
    Initializes trading state and cash balance.
    """
    global state
    state = {
        "cash": float(cash),
        "positions": {},
        "pending_orders": [],
        "peak_prices": {},
    }


def on_bar(history: pd.DataFrame) -> list:
    """
    Called once per time step in chronological order.
    Returns a list of orders formatted as (ticker, direction, size).
    """
    global state

    if history is None or history.empty:
        return []

    # 1. Get current timestamp and bar rows
    ts_array = history["timestamp"].values
    curr_ts = ts_array[-1]

    curr_bar_start = np.searchsorted(ts_array, curr_ts, side="left")
    curr_bar_df = history.iloc[curr_bar_start:]

    open_prices = dict(zip(curr_bar_df["ticker"], curr_bar_df["open"]))
    close_prices = dict(zip(curr_bar_df["ticker"], curr_bar_df["close"]))

    # 2. Simulate fills for pending orders submitted at previous bar
    for ticker, direction, size in state["pending_orders"]:
        if ticker in open_prices:
            p_open = open_prices[ticker]
            if pd.notna(p_open) and p_open > 0:
                if direction == "BUY":
                    max_buy = int(state["cash"] // p_open)
                    filled = min(int(size), max_buy)
                    if filled > 0:
                        state["cash"] -= filled * p_open
                        state["positions"][ticker] = (
                            state["positions"].get(ticker, 0) + filled
                        )
                elif direction == "SELL":
                    held = state["positions"].get(ticker, 0)
                    filled = min(int(size), held)
                    if filled > 0:
                        state["positions"][ticker] = held - filled
                        state["cash"] += filled * p_open
                        if state["positions"][ticker] <= 0:
                            state["positions"].pop(ticker, None)

    state["pending_orders"] = []

    # Remove zero/empty positions
    for t in list(state["positions"].keys()):
        if state["positions"][t] <= 0:
            state["positions"].pop(t, None)
            state["peak_prices"].pop(t, None)

    # Update trailing peak prices for held assets
    for t, pos in state["positions"].items():
        if t in close_prices and pd.notna(close_prices[t]) and close_prices[t] > 0:
            cp = float(close_prices[t])
            state["peak_prices"][t] = max(state["peak_prices"].get(t, cp), cp)

    # 3. Fast history slicing (limit to recent 120 timestamps)
    unique_ts = pd.unique(ts_array)
    lookback = 120
    if len(unique_ts) > lookback:
        cutoff_ts = unique_ts[-lookback]
        start_idx = np.searchsorted(ts_array, cutoff_ts, side="left")
        recent_df = history.iloc[start_idx:]
    else:
        recent_df = history

    # Wait for minimum history required for signals
    if len(unique_ts) < 15:
        return []

    # Clean duplicates if any and pivot close prices
    recent_df = recent_df.drop_duplicates(subset=["timestamp", "ticker"])
    try:
        pivot_close = recent_df.pivot(
            index="timestamp", columns="ticker", values="close"
        )
    except Exception:
        return []

    if pivot_close.empty or pivot_close.shape[0] < 10:
        return []

    latest_close = pivot_close.iloc[-1]
    returns = pivot_close.pct_change()

    # Technical Metrics
    sma_20 = pivot_close.tail(20).mean()
    sma_50 = pivot_close.tail(min(50, pivot_close.shape[0])).mean()
    vol_20 = returns.tail(20).std()

    n_obs = pivot_close.shape[0]
    idx_10 = max(0, n_obs - 10)
    idx_30 = max(0, n_obs - 30)

    ret_10 = (pivot_close.iloc[-1] - pivot_close.iloc[idx_10]) / (
        pivot_close.iloc[idx_10] + 1e-8
    )
    ret_30 = (pivot_close.iloc[-1] - pivot_close.iloc[idx_30]) / (
        pivot_close.iloc[idx_30] + 1e-8
    )
    momentum = 0.5 * ret_10 + 0.5 * ret_30

    delta = pivot_close.diff()
    gain = delta.clip(lower=0).tail(14).mean()
    loss = (-delta.clip(upper=0)).tail(14).mean()
    rs = gain / (loss + 1e-8)
    rsi_14 = 100.0 - (100.0 / (1.0 + rs))

    risk_adj_score = momentum / (vol_20 + 0.001)

    # 4. Filter Long Candidates
    candidates = []
    for ticker in pivot_close.columns:
        c = latest_close.get(ticker)
        s20 = sma_20.get(ticker)
        s50 = sma_50.get(ticker)
        mom = momentum.get(ticker)
        rsi = rsi_14.get(ticker)
        v20 = vol_20.get(ticker)

        if pd.isna(c) or c <= 0:
            continue

        cond_trend = (pd.notna(s20) and c > s20 * 0.99) and (
            pd.notna(s50) and c > s50 * 0.98
        )
        cond_mom = pd.notna(mom) and (mom > 0)
        cond_rsi = pd.notna(rsi) and (35 < rsi < 75)

        if cond_trend and cond_mom and cond_rsi:
            score = risk_adj_score.get(ticker)
            if pd.notna(score):
                candidates.append((ticker, score, v20, c))

    candidates.sort(key=lambda x: x[1], reverse=True)

    max_pos = 5
    top_candidates = candidates[:max_pos]

    # Target Weight Allocation (Inverse Volatility)
    target_portfolio = {}
    if top_candidates:
        inv_vols = []
        for ticker, score, v20, price in top_candidates:
            v = v20 if (pd.notna(v20) and v20 > 1e-5) else 0.02
            inv_vols.append(1.0 / v)

        tot_inv = sum(inv_vols)
        raw_weights = [inv / tot_inv for inv in inv_vols]
        capped_weights = [min(0.30, w) for w in raw_weights]
        sum_capped = sum(capped_weights)

        if sum_capped > 0:
            final_weights = [(w / sum_capped) * 0.95 for w in capped_weights]
        else:
            final_weights = [0.95 / len(top_candidates)] * len(top_candidates)

        for i, (ticker, score, v20, price) in enumerate(top_candidates):
            target_portfolio[ticker] = {
                "weight": final_weights[i],
                "price": price,
            }

    # 5. Compute Estimated Total Equity
    equity = state["cash"]
    for t, pos in state["positions"].items():
        if t in close_prices and pd.notna(close_prices[t]) and close_prices[t] > 0:
            equity += pos * float(close_prices[t])

    equity = max(0.0, equity)

    sell_orders = []
    buy_orders = []

    # 6. Evaluate Exits for Held Assets
    for ticker in list(state["positions"].keys()):
        pos = state["positions"][ticker]
        cp = float(close_prices.get(ticker, 0.0))

        should_sell = False
        peak = state["peak_prices"].get(ticker, cp)

        # Exit 1: Trailing stop (10% drop from peak)
        if cp > 0 and peak > 0 and cp < 0.90 * peak:
            should_sell = True
        # Exit 2: Trend breakdown (4% below 20-bar SMA)
        elif (
            ticker in sma_20
            and pd.notna(sma_20[ticker])
            and cp < 0.96 * float(sma_20[ticker])
        ):
            should_sell = True
        # Exit 3: Loss of momentum rank
        elif ticker not in target_portfolio:
            mom_val = momentum.get(ticker, 0)
            if pd.isna(mom_val) or mom_val < 0 or len(target_portfolio) == max_pos:
                should_sell = True

        if should_sell and pos > 0:
            sell_orders.append((ticker, "SELL", int(pos)))

    # 7. Evaluate Entries & Rebalancing
    for ticker, info in target_portfolio.items():
        price = info["price"]
        w = info["weight"]
        target_val = equity * w
        target_shares = int(target_val // price)

        curr_shares = state["positions"].get(ticker, 0)
        already_selling = any(o[0] == ticker for o in sell_orders)

        if already_selling:
            curr_shares = 0

        diff = target_shares - curr_shares
        if diff > 0:
            buy_orders.append((ticker, "BUY", int(diff)))
        elif diff < 0 and not already_selling:
            sell_qty = abs(diff)
            if sell_qty > 0.15 * curr_shares and sell_qty > 0:
                sell_orders.append((ticker, "SELL", int(sell_qty)))

    # SELL orders first to free capital before BUY orders execute
    orders = sell_orders + buy_orders
    state["pending_orders"] = list(orders)

    return orders
