from collections import defaultdict, deque
import numpy as np
import pandas as pd

# Global state variables
cash: float = 100000.0
positions: dict[str, int] = {}
pending_orders: list[tuple[str, str, int]] = []
last_processed_idx: int = 0
bar_count: int = 0

# Historical data buffers per ticker
ticker_closes: dict[str, deque] = defaultdict(lambda: deque(maxlen=60))
ticker_opens: dict[str, deque] = defaultdict(lambda: deque(maxlen=60))
ticker_volumes: dict[str, deque] = defaultdict(lambda: deque(maxlen=60))

# Strategy parameters
REBALANCE_INTERVAL: int = 5
TOP_K: int = 10
MAX_POS_WEIGHT: float = 0.12
CASH_BUFFER: float = 0.05
LOOKBACK: int = 20


def init(starting_cash: float) -> None:
    """Initialize agent state before the first bar."""
    global cash, positions, pending_orders, last_processed_idx, bar_count
    global ticker_closes, ticker_opens, ticker_volumes

    cash = float(starting_cash)
    positions = {}
    pending_orders = []
    last_processed_idx = 0
    bar_count = 0

    ticker_closes = defaultdict(lambda: deque(maxlen=60))
    ticker_opens = defaultdict(lambda: deque(maxlen=60))
    ticker_volumes = defaultdict(lambda: deque(maxlen=60))


def on_bar(history: pd.DataFrame) -> list[tuple[str, str, int]]:
    """Called on every bar timestep to process history and return trading orders."""
    global cash, positions, pending_orders, last_processed_idx, bar_count
    global ticker_closes, ticker_opens, ticker_volumes

    # 1. Incrementally ingest newly arrived rows from append-only history DataFrame
    new_rows = history.iloc[last_processed_idx:]
    last_processed_idx = len(history)

    current_opens: dict[str, float] = {}
    current_closes: dict[str, float] = {}

    for row in new_rows.itertuples(index=False):
        t = str(row.ticker)
        c = float(row.close)
        o = float(row.open)
        v = float(row.volume) if hasattr(row, "volume") else 1.0

        ticker_closes[t].append(c)
        ticker_opens[t].append(o)
        ticker_volumes[t].append(v)

        current_opens[t] = o
        current_closes[t] = c

    # 2. Simulate platform execution of orders submitted at t-1 using bar t open prices
    if pending_orders:
        sells = [order for order in pending_orders if order[1] == "SELL"]
        buys = [order for order in pending_orders if order[1] == "BUY"]

        # Platform processes SELLs first to realize cash
        for ticker, _, size in sells:
            if ticker in current_opens:
                open_px = current_opens[ticker]
                held = positions.get(ticker, 0)
                actual_sell = min(size, held)
                if actual_sell > 0:
                    cash += actual_sell * open_px
                    positions[ticker] = held - actual_sell
                    if positions[ticker] == 0:
                        del positions[ticker]

        # Platform processes BUYs next
        for ticker, _, size in buys:
            if ticker in current_opens:
                open_px = current_opens[ticker]
                if open_px > 0:
                    max_buy = int(cash // open_px)
                    actual_buy = min(size, max_buy)
                    if actual_buy > 0:
                        cost = actual_buy * open_px
                        cash -= cost
                        positions[ticker] = positions.get(ticker, 0) + actual_buy

        pending_orders = []

    bar_count += 1

    # Need minimum lookback history before active trading
    if bar_count < LOOKBACK:
        return []

    # 3. Calculate current total portfolio mark-to-market value
    total_val = cash
    for ticker, pos in positions.items():
        if ticker in current_closes:
            total_val += pos * current_closes[ticker]
        elif len(ticker_closes[ticker]) > 0:
            total_val += pos * ticker_closes[ticker][-1]

    # Rebalance periodically or on initial deployment
    if bar_count % REBALANCE_INTERVAL != 0 and len(positions) > 0:
        return []

    # 4. Cross-Sectional Alpha Signal & Target Ranking
    candidates = []
    for ticker, closes in ticker_closes.items():
        if len(closes) >= LOOKBACK:
            c_arr = np.array(closes)[-LOOKBACK:]
            rets = np.diff(c_arr) / c_arr[:-1]
            if len(rets) < 5:
                continue

            mean_ret = float(np.mean(rets))
            std_ret = float(np.std(rets))

            vols = np.array(ticker_volumes[ticker])[-LOOKBACK:]
            avg_dollar_vol = float(np.mean(c_arr * vols))

            # Quality, trend, and liquidity filters
            if mean_ret > 0 and std_ret > 1e-6 and avg_dollar_vol > 1000:
                sharpe = mean_ret / std_ret
                candidates.append((sharpe, std_ret, ticker, c_arr[-1]))

    # Rank candidates by risk-adjusted momentum (Sharpe ratio)
    candidates.sort(key=lambda x: x[0], reverse=True)
    top_candidates = candidates[:TOP_K]

    # 5. Target Position Calculation via Risk Parity (Inverse Volatility)
    target_positions: dict[str, int] = {}
    if top_candidates:
        inv_vols = [1.0 / (c[1] + 1e-5) for c in top_candidates]
        tot_inv_vol = sum(inv_vols)

        for (sharpe, std_ret, ticker, curr_px), inv_v in zip(
            top_candidates, inv_vols
        ):
            weight = min(
                MAX_POS_WEIGHT, (inv_v / tot_inv_vol) * (1.0 - CASH_BUFFER)
            )
            target_val = total_val * weight
            if curr_px > 0:
                target_shares = int(target_val // curr_px)
                if target_shares > 0:
                    target_positions[ticker] = target_shares

    # 6. Order Generation (SELLs first, followed by BUYs)
    est_freed_cash = 0.0
    sell_orders: list[tuple[str, str, int]] = []

    for ticker, current_shares in list(positions.items()):
        target_shares = target_positions.get(ticker, 0)
        if current_shares > target_shares:
            shares_to_sell = current_shares - target_shares
            sell_orders.append((ticker, "SELL", shares_to_sell))
            curr_px = current_closes.get(
                ticker,
                ticker_closes[ticker][-1] if ticker_closes[ticker] else 0.0,
            )
            est_freed_cash += shares_to_sell * curr_px

    avail_cash = cash + est_freed_cash
    buy_orders: list[tuple[str, str, int]] = []

    for ticker, target_shares in target_positions.items():
        current_shares = positions.get(ticker, 0)
        if target_shares > current_shares:
            shares_to_buy = target_shares - current_shares
            curr_px = current_closes.get(
                ticker,
                ticker_closes[ticker][-1] if ticker_closes[ticker] else 0.0,
            )
            if curr_px > 0:
                max_buy = int((avail_cash * 0.98) // curr_px)
                actual_buy = min(shares_to_buy, max_buy)
                if actual_buy > 0:
                    buy_orders.append((ticker, "BUY", actual_buy))
                    avail_cash -= actual_buy * curr_px

    orders = sell_orders + buy_orders
    pending_orders = list(orders)

    return orders
