import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Strategy parameters
# ---------------------------------------------------------------------------
LOOKBACK_MOM   = 63     # bars for momentum signal
LOOKBACK_VOL   = 20     # bars for volatility estimate (position sizing)
REBALANCE_FREQ = 5      # rebalance every N bars
TOP_K          = 8      # max number of concurrent holdings
TARGET_INVESTED = 0.95  # fraction of equity we try to keep deployed
MAX_WEIGHT     = 0.25   # cap on any single position's weight of equity
STOP_LOSS_PCT  = 0.15   # hard stop loss from average cost basis
MIN_TRADE_VALUE = 50.0  # skip trades smaller than this (avoid noise churn)
MIN_PRICE      = 2.0    # ignore very low priced / penny names

_STATE = {}


def init(cash):
    _STATE.clear()
    _STATE['cash'] = float(cash)
    _STATE['positions'] = {}      # ticker -> shares held (internal ledger)
    _STATE['avg_cost'] = {}       # ticker -> average cost basis
    _STATE['last_price'] = {}     # ticker -> most recently observed close
    _STATE['pending_orders'] = [] # orders submitted last bar, fill this bar
    _STATE['bar_count'] = 0


def _process_pending_orders(open_map):
    """Mirror the platform's fill logic on our own internal ledger."""
    cash = _STATE['cash']
    positions = _STATE['positions']
    avg_cost = _STATE['avg_cost']

    for ticker, direction, size in _STATE['pending_orders']:
        if size <= 0:
            continue
        price = open_map.get(ticker)
        if price is None or not np.isfinite(price) or price <= 0:
            continue  # no price available on execution bar -> fills for zero

        if direction == 'BUY':
            max_afford = int(cash // price)
            fill = min(int(size), max_afford)
            if fill > 0:
                cost = fill * price
                prev_shares = positions.get(ticker, 0)
                prev_cost_total = avg_cost.get(ticker, 0.0) * prev_shares
                new_shares = prev_shares + fill
                avg_cost[ticker] = (prev_cost_total + cost) / new_shares
                positions[ticker] = new_shares
                cash -= cost

        elif direction == 'SELL':
            held = positions.get(ticker, 0)
            fill = min(int(size), held)
            if fill > 0:
                remaining = held - fill
                cash += fill * price
                if remaining <= 0:
                    positions.pop(ticker, None)
                    avg_cost.pop(ticker, None)
                else:
                    positions[ticker] = remaining

    _STATE['cash'] = cash
    _STATE['pending_orders'] = []


def _slice_from_ts(df, ts_array, target_ts):
    """Fast slice of an append-only, timestamp-sorted frame."""
    idx = np.searchsorted(ts_array, target_ts, side='left')
    return df.iloc[idx:]


def on_bar(history):
    if history.empty:
        return []

    ts_full = history['timestamp'].to_numpy()
    current_ts = ts_full[-1]
    current = _slice_from_ts(history, ts_full, current_ts)

    current_open_map = dict(zip(current['ticker'], current['open']))
    current_close_map = dict(zip(current['ticker'], current['close']))

    # keep last known prices fresh for valuation purposes
    for t, p in current_close_map.items():
        if pd.notna(p):
            _STATE['last_price'][t] = float(p)

    # 1) settle orders placed last bar using this bar's open prices
    _process_pending_orders(current_open_map)

    _STATE['bar_count'] += 1
    positions = _STATE['positions']
    avg_cost = _STATE['avg_cost']
    last_price = _STATE['last_price']

    orders = []

    # 2) stop-loss check, every bar, independent of rebalance schedule
    stop_sell_tickers = set()
    for t, shares in list(positions.items()):
        if shares <= 0:
            continue
        price_today = current_close_map.get(t)
        if price_today is None or pd.isna(price_today):
            continue
        cb = avg_cost.get(t)
        if cb and price_today <= cb * (1 - STOP_LOSS_PCT):
            orders.append((t, 'SELL', int(shares)))
            stop_sell_tickers.add(t)

    # 3) periodic momentum rebalance
    if _STATE['bar_count'] % REBALANCE_FREQ == 0:
        n_needed = LOOKBACK_MOM + 2
        ts_unique = history['timestamp'].unique()

        if len(ts_unique) >= n_needed:
            cutoff_ts = ts_unique[-n_needed]
            recent = _slice_from_ts(history, ts_full, cutoff_ts)
            price_pivot = recent.pivot_table(index='timestamp', columns='ticker',
                                              values='close').sort_index()

            today_tickers = [t for t, p in current_close_map.items()
                              if pd.notna(p) and p >= MIN_PRICE]

            mom_scores = {}
            for t in today_tickers:
                if t not in price_pivot.columns:
                    continue
                series = price_pivot[t].dropna()
                if len(series) < LOOKBACK_MOM + 1:
                    continue
                p_now = series.iloc[-1]
                p_then = series.iloc[-(LOOKBACK_MOM + 1)]
                if p_then > 0:
                    mom_scores[t] = p_now / p_then - 1.0

            ranked = sorted(mom_scores, key=lambda t: mom_scores[t], reverse=True)
            selected = [t for t in ranked if mom_scores[t] > 0][:TOP_K]

            weights = {}
            if selected:
                vols = {}
                for t in selected:
                    series = price_pivot[t].dropna()
                    rets = series.pct_change().dropna().iloc[-LOOKBACK_VOL:]
                    v = rets.std()
                    if v is None or pd.isna(v) or v <= 1e-6:
                        v = 0.02
                    vols[t] = v
                inv_vol = {t: 1.0 / vols[t] for t in selected}
                total_inv = sum(inv_vol.values())
                for t in selected:
                    raw_w = inv_vol[t] / total_inv * TARGET_INVESTED
                    weights[t] = min(raw_w, MAX_WEIGHT)

            # mark-to-market equity using last known prices
            holdings_value = sum(
                shares * last_price.get(t, 0.0) for t, shares in positions.items()
            )
            total_equity = _STATE['cash'] + holdings_value

            target_shares = {}
            for t in selected:
                w = weights.get(t, 0.0)
                p = current_close_map.get(t) or last_price.get(t)
                if not p or p <= 0:
                    continue
                target_shares[t] = int((w * total_equity) // p)

            all_relevant = set(positions.keys()) | set(target_shares.keys())
            sell_orders, buy_orders = [], []
            for t in all_relevant:
                if t in stop_sell_tickers:
                    continue
                cur_shares = positions.get(t, 0)
                tgt_shares = target_shares.get(t, 0)
                delta = tgt_shares - cur_shares
                if delta == 0:
                    continue
                p = current_close_map.get(t) or last_price.get(t) or 0
                if abs(delta) * p < MIN_TRADE_VALUE:
                    continue
                if delta < 0:
                    sell_orders.append((t, 'SELL', int(-delta)))
                else:
                    buy_orders.append((t, 'BUY', int(delta)))

            orders.extend(sell_orders)
            orders.extend(buy_orders)

    _STATE['pending_orders'] = orders
    return orders
