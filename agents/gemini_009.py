import numpy as np
import pandas as pd

# Module-level state
_initial_cash: float = 100000.0
_current_cash: float = 100000.0
_current_positions: dict[str, int] = {}
_pending_orders: list[tuple[str, str, int]] = []
_lookback_window: int = 150


def init(cash: float) -> None:
    """Initialize agent state before the first bar."""
    global _initial_cash, _current_cash, _current_positions, _pending_orders
    _initial_cash = float(cash)
    _current_cash = float(cash)
    _current_positions = {}
    _pending_orders = []


def on_bar(history: pd.DataFrame) -> list[tuple[str, str, int]]:
    """Process incoming bar history and return trade orders for the current step."""
    global _initial_cash, _current_cash, _current_positions, _pending_orders

    if history.empty:
        return []

    # 1. Identify latest timestamp & latest prices
    latest_ts = history["timestamp"].iloc[-1]
    latest_rows = history[history["timestamp"] == latest_ts]

    latest_opens = dict(zip(latest_rows["ticker"], latest_rows["open"]))
    latest_closes = dict(zip(latest_rows["ticker"], latest_rows["close"]))

    # 2. Process pending orders from previous bar using execution open prices
    if _pending_orders:
        for ticker, direction, size in _pending_orders:
            if size <= 0:
                continue
            open_price = latest_opens.get(ticker, np.nan)
            if np.isnan(open_price) or open_price <= 0:
                # Execution unavailable on this bar
                continue

            if direction == "BUY":
                max_buyable = int(_current_cash // open_price)
                fill_qty = min(size, max_buyable)
                if fill_qty > 0:
                    _current_cash -= fill_qty * open_price
                    _current_positions[ticker] = (
                        _current_positions.get(ticker, 0) + fill_qty
                    )
            elif direction == "SELL":
                held_qty = _current_positions.get(ticker, 0)
                fill_qty = min(size, held_qty)
                if fill_qty > 0:
                    _current_cash += fill_qty * open_price
                    _current_positions[ticker] = held_qty - fill_qty
                    if _current_positions[ticker] <= 0:
                        _current_positions.pop(ticker, None)

        _pending_orders = []

    # Clean up zero positions
    _current_positions = {t: q for t, q in _current_positions.items() if q > 0}

    # 3. Fast Data Windowing for Quantitative Features
    unique_ts = history["timestamp"].unique()
    if len(unique_ts) > _lookback_window:
        cutoff_ts = unique_ts[-_lookback_window]
        df_win = history[history["timestamp"] >= cutoff_ts]
    else:
        df_win = history

    piv_close = df_win.pivot_table(
        index="timestamp", columns="ticker", values="close", aggfunc="last"
    )

    if len(piv_close) < 10:
        return []

    active_tickers = list(latest_closes.keys())

    # Calculate Current Estimated Portfolio Equity
    total_equity = _current_cash
    for t, q in _current_positions.items():
        p = latest_closes.get(t, 0.0)
        total_equity += q * p

    if total_equity <= 0:
        return []

    # 4. Market Regime Filter (Macro Universe Trend)
    mkt_index = piv_close.mean(axis=1)
    if len(mkt_index) >= 30:
        mkt_sma30 = mkt_index.iloc[-30:].mean()
        mkt_bullish = mkt_index.iloc[-1] > mkt_sma30
    else:
        mkt_bullish = True

    # Adjust target portfolio market exposure based on regime
    target_market_exposure = 0.95 if mkt_bullish else 0.40

    # 5. Asset Scoring & Volatility Estimation
    scores = {}
    vols = {}
    eligible_tickers = []

    for ticker in active_tickers:
        if ticker not in piv_close.columns:
            continue

        s_close = piv_close[ticker].dropna()
        if len(s_close) < 15:
            continue

        last_price = s_close.iloc[-1]
        if last_price < 1.0:
            continue  # Exclude penny stocks

        # Volatility estimation (annualized 20-bar daily volatility)
        rets = s_close.pct_change().dropna()
        if len(rets) < 5:
            continue

        vol_20 = (
            rets.iloc[-20:].std() * np.sqrt(252)
            if len(rets) >= 20
            else rets.std() * np.sqrt(252)
        )
        if np.isnan(vol_20) or vol_20 < 0.01:
            vol_20 = 0.05
        vols[ticker] = vol_20

        # Multi-period return signals
        ret_5 = (last_price / s_close.iloc[-min(5, len(s_close))]) - 1.0
        ret_20 = (last_price / s_close.iloc[-min(20, len(s_close))]) - 1.0

        lookback_long = min(120, len(s_close))
        lookback_lag = min(5, lookback_long - 1)
        if lookback_long > lookback_lag:
            ret_long = (
                s_close.iloc[-1 - lookback_lag] / s_close.iloc[-lookback_long]
            ) - 1.0
        else:
            ret_long = ret_20

        # Moving Averages
        sma_20 = s_close.iloc[-min(20, len(s_close)) :].mean()
        sma_50 = s_close.iloc[-min(50, len(s_close)) :].mean()

        # Trend eligibility check
        trend_ok = (last_price > sma_50) or (sma_20 > sma_50)

        # Signal score: reward strong momentum, penalize recent overextended spikes
        score = 0.6 * ret_long + 0.4 * ret_20 - 0.3 * max(0.0, ret_5 - 0.10)
        scores[ticker] = score

        if trend_ok:
            eligible_tickers.append(ticker)

    # Fallback to top scored assets if too few pass trend checks
    if len(eligible_tickers) < 3:
        sorted_active = sorted(
            scores.keys(), key=lambda x: scores[x], reverse=True
        )
        eligible_tickers = sorted_active[:5]

    # Rank eligible tickers by score
    eligible_tickers.sort(key=lambda x: scores.get(x, -999.0), reverse=True)

    max_positions = min(8, max(3, len(eligible_tickers)))
    selected_tickers = eligible_tickers[:max_positions]

    if not selected_tickers:
        return []

    # 6. Risk Parity Sizing & Weighting
    inv_vols = {t: 1.0 / vols.get(t, 0.20) for t in selected_tickers}
    sum_inv_vols = sum(inv_vols.values())

    raw_weights = {t: inv_vols[t] / sum_inv_vols for t in selected_tickers}

    # Cap single position weight at 25%
    capped_weights = {t: min(w, 0.25) for t, w in raw_weights.items()}
    weight_sum = sum(capped_weights.values())

    if weight_sum > 0:
        norm_weights = {
            t: (w / weight_sum) * target_market_exposure
            for t, w in capped_weights.items()
        }
    else:
        norm_weights = {
            t: target_market_exposure / len(selected_tickers)
            for t in selected_tickers
        }

    # Calculate target shares per ticker
    target_shares = {}
    for t in selected_tickers:
        p = latest_closes.get(t, 0.0)
        if p > 0:
            target_dollar = total_equity * norm_weights[t]
            target_shares[t] = int(target_dollar // p)

    # 7. Order Generation with Rebalance Buffers
    sell_orders: list[tuple[str, str, int]] = []
    raw_buy_orders: list[tuple[str, str, int, float, float]] = []

    est_cash_available = _current_cash

    # Determine SELL orders first to release capital
    for t, current_q in _current_positions.items():
        if current_q <= 0:
            continue
        p = latest_closes.get(t, 0.0)
        t_shares = target_shares.get(t, 0)

        if t not in selected_tickers:
            # Liquidate positions no longer in top selection
            sell_orders.append((str(t), "SELL", int(current_q)))
            est_cash_available += current_q * p * 0.99
        elif t_shares < current_q:
            diff = current_q - t_shares
            # Trade buffer: execute if change is >= 1.5% of equity or >= 50% of position size
            if (diff * p) >= (0.015 * total_equity) or (diff >= 0.5 * current_q):
                sell_orders.append((str(t), "SELL", int(diff)))
                est_cash_available += diff * p * 0.99

    # Determine BUY orders
    for t in selected_tickers:
        t_shares = target_shares.get(t, 0)
        current_q = _current_positions.get(t, 0)
        p = latest_closes.get(t, 0.0)

        if p <= 0:
            continue

        if t_shares > current_q:
            diff = t_shares - current_q
            trade_value = diff * p
            if current_q == 0 or trade_value >= (0.015 * total_equity):
                raw_buy_orders.append(
                    (str(t), "BUY", int(diff), scores.get(t, 0.0), p)
                )

    # Prioritize buys by composite factor score
    raw_buy_orders.sort(key=lambda x: x[3], reverse=True)

    # Budget allocation for buy orders with safety margin
    final_buy_orders: list[tuple[str, str, int]] = []
    spendable_cash = est_cash_available * 0.96

    for t, direction, size, _, p in raw_buy_orders:
        cost = size * p
        if cost <= spendable_cash:
            final_buy_orders.append((t, direction, size))
            spendable_cash -= cost
        else:
            affordable_size = int(spendable_cash // p)
            if affordable_size > 0:
                final_buy_orders.append((t, direction, affordable_size))
                spendable_cash -= affordable_size * p

    # Place SELL orders before BUY orders in returned list
    orders = sell_orders + final_buy_orders
    _pending_orders = orders

    return orders
