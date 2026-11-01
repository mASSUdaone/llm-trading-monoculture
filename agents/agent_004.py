import numpy as np
import pandas as pd

_STATE = {}


def init(cash):
    global _STATE
    _STATE = {
        'cash': float(cash),
        'positions': {},        # ticker -> shares held (int)
        'pending_orders': [],   # orders submitted last bar, to fill at this bar's open
        'peak_price': {},       # ticker -> peak close price since position opened
        'day_count': 0,
        'last_rebalance_day': -10_000,
        'rebalance_interval': 21,
        'lookback_mom': 126,
        'lookback_vol': 20,
        'top_n': 8,
        'max_weight': 0.25,
        'invested_fraction': 0.95,
        'min_dollar_volume': 500_000.0,
        'trailing_stop_pct': 0.15,
        'min_history_for_trade': 130,
    }


def _process_fills(current_prices_open):
    """Deterministically replay the platform's fill rule against known open
    prices, keeping our internal cash/position ledger in sync."""
    st = _STATE
    for ticker, direction, size in st['pending_orders']:
        if size <= 0:
            continue
        price = current_prices_open.get(ticker)
        if price is None or not np.isfinite(price) or price <= 0:
            continue  # no price available -> fills for zero shares

        if direction == 'BUY':
            affordable = int(st['cash'] // price)
            fill = min(size, affordable)
            if fill > 0:
                st['cash'] -= fill * price
                st['positions'][ticker] = st['positions'].get(ticker, 0) + fill
        elif direction == 'SELL':
            held = st['positions'].get(ticker, 0)
            fill = min(size, held)
            if fill > 0:
                remaining = held - fill
                if remaining > 0:
                    st['positions'][ticker] = remaining
                else:
                    st['positions'].pop(ticker, None)
                st['cash'] += fill * price

    st['pending_orders'] = []


def _cap_and_renormalize(weights, max_w, iterations=15):
    w = dict(weights)
    for _ in range(iterations):
        over = {t: v for t, v in w.items() if v > max_w + 1e-12}
        if not over:
            break
        excess = sum(v - max_w for v in over.values())
        for t in over:
            w[t] = max_w
        uncapped = {t: v for t, v in w.items() if v < max_w - 1e-12}
        if not uncapped:
            break
        total_uncapped = sum(uncapped.values())
        if total_uncapped <= 0:
            break
        for t in uncapped:
            w[t] += excess * (uncapped[t] / total_uncapped)
    total = sum(w.values())
    if total > 0:
        w = {t: v / total for t, v in w.items()}
    return w


def _compute_rebalance_orders(history, current, current_prices_close, excluded_tickers):
    st = _STATE
    lookback_mom = st['lookback_mom']
    lookback_vol = st['lookback_vol']
    top_n = st['top_n']

    ts_all = history['timestamp'].unique()
    ts_all = np.sort(ts_all)
    needed = lookback_mom + 2
    recent_ts = ts_all[-needed:] if len(ts_all) >= needed else ts_all
    recent = history[history['timestamp'].isin(recent_ts)]

    closes = recent.pivot_table(index='timestamp', columns='ticker', values='close').sort_index()
    volumes = recent.pivot_table(index='timestamp', columns='ticker', values='volume').sort_index()

    if len(closes) < lookback_mom + 1:
        return []

    tickers_now = set(current['ticker']) - set(excluded_tickers)

    first_row = closes.iloc[0]
    last_row = closes.iloc[-1]

    vol_window = closes.tail(lookback_vol + 1)
    rets = vol_window.pct_change().iloc[1:]

    dv_closes = closes.tail(lookback_vol)
    dv_vols = volumes.tail(lookback_vol)
    dollar_vol_window = dv_closes * dv_vols

    momentum_scores = {}
    vol_scores = {}

    for ticker in tickers_now:
        if ticker not in closes.columns:
            continue
        col = closes[ticker]
        if col.notna().sum() < lookback_mom + 1:
            continue
        c0 = first_row.get(ticker, np.nan)
        c1 = last_row.get(ticker, np.nan)
        if pd.isna(c0) or pd.isna(c1) or c0 <= 0:
            continue

        mom = c1 / c0 - 1.0

        if ticker not in rets.columns:
            continue
        r = rets[ticker].dropna()
        if len(r) < max(2, lookback_vol // 2):
            continue
        vol = r.std()
        if not np.isfinite(vol) or vol <= 0:
            continue

        if ticker not in dollar_vol_window.columns:
            continue
        dv = dollar_vol_window[ticker].dropna()
        if dv.empty:
            continue
        dv_mean = dv.mean()
        if not np.isfinite(dv_mean) or dv_mean < st['min_dollar_volume']:
            continue

        momentum_scores[ticker] = mom
        vol_scores[ticker] = vol

    candidates = [(t, m) for t, m in momentum_scores.items() if m > 0]
    candidates.sort(key=lambda x: x[1], reverse=True)
    selected = [t for t, _ in candidates[:top_n]]

    equity = st['cash'] + sum(
        shares * current_prices_close.get(t, 0.0)
        for t, shares in st['positions'].items()
        if current_prices_close.get(t) is not None
    )

    orders = []

    if not selected:
        for ticker, shares in list(st['positions'].items()):
            if ticker in excluded_tickers:
                continue
            if shares > 0:
                orders.append((ticker, 'SELL', shares))
        return orders

    inv_vol = {t: 1.0 / vol_scores[t] for t in selected}
    total_inv_vol = sum(inv_vol.values())
    weights = {t: inv_vol[t] / total_inv_vol for t in selected}
    weights = _cap_and_renormalize(weights, st['max_weight'])

    invested_equity = max(equity, 0.0) * st['invested_fraction']

    target_shares = {}
    for t in selected:
        price = current_prices_close.get(t)
        if price is None or not np.isfinite(price) or price <= 0:
            continue
        target_dollar = weights[t] * invested_equity
        target_shares[t] = int(target_dollar // price)

    for ticker, shares in list(st['positions'].items()):
        if ticker in excluded_tickers:
            continue
        if ticker not in target_shares and shares > 0:
            orders.append((ticker, 'SELL', shares))

    for t in selected:
        if t in excluded_tickers:
            continue
        current_shares = st['positions'].get(t, 0)
        tgt = target_shares.get(t, 0)
        diff = tgt - current_shares
        if diff > 0:
            orders.append((t, 'BUY', diff))
        elif diff < 0:
            orders.append((t, 'SELL', -diff))

    return orders


def on_bar(history):
    st = _STATE

    ts_col = history['timestamp']
    current_ts = ts_col.iloc[-1]
    current = history[ts_col == current_ts]

    current_prices_open = dict(zip(current['ticker'], current['open']))
    current_prices_close = dict(zip(current['ticker'], current['close']))

    # 1) Sync internal ledger with the platform's deterministic fill of
    #    orders we submitted on the previous bar.
    _process_fills(current_prices_open)

    st['day_count'] += 1

    # 2) Housekeeping on peak-price tracking (drop stale, update live).
    for ticker in list(st['peak_price'].keys()):
        if ticker not in st['positions']:
            del st['peak_price'][ticker]

    for ticker in st['positions'].keys():
        price = current_prices_close.get(ticker)
        if price is not None and np.isfinite(price):
            prev_peak = st['peak_price'].get(ticker, price)
            st['peak_price'][ticker] = max(prev_peak, price)

    sell_orders = []
    buy_orders = []

    # 3) Trailing stop check, every bar.
    stopped_out = []
    for ticker, shares in list(st['positions'].items()):
        price = current_prices_close.get(ticker)
        if price is None or not np.isfinite(price):
            continue
        peak = st['peak_price'].get(ticker, price)
        if peak > 0 and (price / peak - 1.0) <= -st['trailing_stop_pct']:
            stopped_out.append(ticker)

    for ticker in stopped_out:
        shares = st['positions'].get(ticker, 0)
        if shares > 0:
            sell_orders.append((ticker, 'SELL', shares))

    # 4) Periodic rebalance.
    due = (st['day_count'] - st['last_rebalance_day']) >= st['rebalance_interval']
    if due and st['day_count'] >= st['min_history_for_trade']:
        st['last_rebalance_day'] = st['day_count']
        rebalance_orders = _compute_rebalance_orders(
            history, current, current_prices_close, set(stopped_out)
        )
        for t, d, s in rebalance_orders:
            if d == 'SELL':
                sell_orders.append((t, d, s))
            else:
                buy_orders.append((t, d, s))

    orders = sell_orders + buy_orders

    # 5) Queue these orders to be replayed against next bar's open.
    st['pending_orders'] = list(orders)

    return orders
