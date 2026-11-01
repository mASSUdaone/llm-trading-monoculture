import numpy as np
import pandas as pd

# ----------------------- Strategy parameters -----------------------
LOOKBACK = 20          # bars used to measure momentum / volatility
REBALANCE_FREQ = 5     # rebalance the portfolio every N bars
TOP_K = 8              # number of names to hold
MAX_WEIGHT = 0.20      # cap on any single position's weight
INVEST_FRACTION = 0.95 # fraction of equity to deploy (keep a cash buffer)
STOP_LOSS = 0.12       # exit a position if it falls 12% below avg cost
MIN_PRICE = 1.0        # ignore very low priced / illiquid-looking names

_state = {}


def init(cash):
    _state.clear()
    _state['cash'] = float(cash)
    _state['positions'] = {}       # ticker -> shares held (int)
    _state['entry_price'] = {}     # ticker -> volume-weighted avg cost
    _state['pending_orders'] = []  # orders submitted last bar, to fill now
    _state['timestamps'] = []      # unique timestamps seen, in order
    _state['bar_count'] = 0


def _process_fills(current_bar_df):
    """Apply orders submitted on the previous bar at this bar's open,
    exactly mirroring the platform's fill rules, keeping our ledger in sync."""
    prices = dict(zip(current_bar_df['ticker'], current_bar_df['open']))
    orders = _state['pending_orders']
    cash = _state['cash']
    positions = _state['positions']
    entry_price = _state['entry_price']

    for ticker, direction, size in orders:
        price = prices.get(ticker)
        if price is None or not np.isfinite(price) or price <= 0 or size <= 0:
            continue  # no price available this bar -> zero fill

        if direction == 'BUY':
            max_affordable = int(cash // price)
            qty = min(int(size), max_affordable)
            if qty <= 0:
                continue
            old_shares = positions.get(ticker, 0)
            old_cost = entry_price.get(ticker, 0.0)
            new_shares = old_shares + qty
            new_cost = (old_cost * old_shares + price * qty) / new_shares
            positions[ticker] = new_shares
            entry_price[ticker] = new_cost
            cash -= qty * price

        elif direction == 'SELL':
            held = positions.get(ticker, 0)
            qty = min(int(size), held)
            if qty <= 0:
                continue
            remaining = held - qty
            cash += qty * price
            if remaining <= 0:
                positions.pop(ticker, None)
                entry_price.pop(ticker, None)
            else:
                positions[ticker] = remaining

    _state['cash'] = cash
    _state['pending_orders'] = []


def on_bar(history):
    ts_values = history['timestamp'].values
    current_ts = ts_values[-1]

    start_idx = np.searchsorted(ts_values, current_ts, side='left')
    current_bar = history.iloc[start_idx:]

    # 1) settle orders placed on the previous bar, at this bar's open
    _process_fills(current_bar)

    # 2) bookkeeping
    timestamps = _state['timestamps']
    if not timestamps or timestamps[-1] != current_ts:
        timestamps.append(current_ts)
    _state['bar_count'] += 1

    close_map = dict(zip(current_bar['ticker'], current_bar['close']))
    positions = _state['positions']
    entry_price = _state['entry_price']
    cash = _state['cash']

    orders = []
    stopped = set()

    # 3) per-bar stop-loss check on all current holdings
    for ticker, shares in list(positions.items()):
        price = close_map.get(ticker)
        cost = entry_price.get(ticker)
        if price is None or cost is None or shares <= 0:
            continue
        if price <= cost * (1.0 - STOP_LOSS):
            orders.append((ticker, 'SELL', shares))
            stopped.add(ticker)

    # 4) periodic momentum rebalance
    do_rebalance = (
        _state['bar_count'] % REBALANCE_FREQ == 0
        and len(timestamps) >= LOOKBACK + 1
    )

    if do_rebalance:
        cutoff_ts = timestamps[-(LOOKBACK + 1)]
        start_idx2 = np.searchsorted(ts_values, cutoff_ts, side='left')
        recent = history.iloc[start_idx2:]

        pivot = recent.pivot_table(index='timestamp', columns='ticker', values='close')

        if pivot.shape[0] >= 2:
            first_row = pivot.iloc[0]
            last_row = pivot.iloc[-1]
            momentum = (last_row / first_row - 1.0).dropna()

            rets = pivot.pct_change().dropna(how='all')
            vol = rets.std(skipna=True)

            candidates = []
            for t in momentum.index:
                v = vol.get(t, np.nan)
                p = last_row.get(t, np.nan)
                if (
                    t not in stopped
                    and pd.notna(v) and v > 1e-8
                    and pd.notna(p) and p >= MIN_PRICE
                ):
                    candidates.append(t)

            momentum = momentum[candidates]
            longs = momentum[momentum > 0].sort_values(ascending=False).head(TOP_K)

            equity = cash + sum(
                positions.get(t, 0) * close_map.get(t, 0.0) for t in positions
            )

            target_shares = {}
            if len(longs) > 0:
                inv_vol = 1.0 / vol[longs.index]
                weights = inv_vol / inv_vol.sum()
                weights = weights.clip(upper=MAX_WEIGHT)
                weights = weights / weights.sum()

                for t in longs.index:
                    price = close_map.get(t)
                    if price is None or price <= 0:
                        continue
                    alloc = weights[t] * equity * INVEST_FRACTION
                    target_shares[t] = int(alloc // price)

            all_tickers = set(positions.keys()) | set(target_shares.keys())
            sell_orders = []
            buy_candidates = []

            for t in all_tickers:
                if t in stopped:
                    continue  # already fully exiting this bar
                current_shares = positions.get(t, 0)
                target = target_shares.get(t, 0)
                diff = target - current_shares
                if diff < 0:
                    sell_orders.append((t, 'SELL', -diff))
                elif diff > 0:
                    price = close_map.get(t)
                    if price and price > 0:
                        buy_candidates.append((t, diff, price))

            # only spend cash we currently have (don't assume same-bar sell proceeds)
            total_cost = sum(diff * price for _, diff, price in buy_candidates)
            if total_cost > cash and total_cost > 0:
                scale = (cash / total_cost) * 0.98
            else:
                scale = 1.0

            buy_orders = []
            for t, diff, price in buy_candidates:
                qty = int(diff * scale)
                if qty > 0:
                    buy_orders.append((t, 'BUY', qty))

            orders.extend(sell_orders)
            orders.extend(buy_orders)

    _state['pending_orders'] = orders
    return orders
