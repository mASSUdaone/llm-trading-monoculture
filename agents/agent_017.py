import numpy as np
import pandas as pd

_STATE = {}


def init(cash):
    """Called once before the first bar."""
    _STATE.clear()
    _STATE['cash'] = float(cash)
    _STATE['positions'] = {}        # ticker -> shares held (int), our own ledger
    _STATE['pending_orders'] = []   # orders submitted last bar, filled at this bar's open
    _STATE['bar_count'] = 0
    _STATE['peak_price'] = {}       # ticker -> peak close since entry (trailing stop)
    _STATE['last_close'] = {}       # ticker -> most recently observed close (for valuation)

    # --- strategy parameters ---
    _STATE['rebalance_period'] = 10     # bars between full rebalances
    _STATE['lookback'] = 126            # bars of history used for momentum/vol
    _STATE['top_n'] = 10                # max number of names held
    _STATE['max_alloc'] = 0.95          # max fraction of portfolio value deployed
    _STATE['max_weight'] = 0.25         # cap on any single name's weight
    _STATE['stop_loss'] = 0.15          # trailing stop, fraction off peak close
    _STATE['min_order_value'] = 50.0    # skip rebalancing trades smaller than this


def _valid_price(p):
    return p is not None and not (isinstance(p, float) and np.isnan(p)) and p > 0


def _fill_pending(current_open):
    """Simulate execution of last bar's orders at this bar's open, mirroring the
    platform's partial-fill / affordability rules, to keep our ledger accurate."""
    st = _STATE
    cash = st['cash']
    positions = st['positions']

    for ticker, direction, size in st['pending_orders']:
        price = current_open.get(ticker)
        if not _valid_price(price) or size <= 0:
            continue
        if direction == 'BUY':
            affordable = int(cash // price)
            fill = min(int(size), affordable)
            if fill > 0:
                cash -= fill * price
                positions[ticker] = positions.get(ticker, 0) + fill
        elif direction == 'SELL':
            held = positions.get(ticker, 0)
            fill = min(int(size), held)
            if fill > 0:
                new_held = held - fill
                if new_held > 0:
                    positions[ticker] = new_held
                else:
                    positions.pop(ticker, None)
                    st['peak_price'].pop(ticker, None)
                cash += fill * price

    st['cash'] = cash
    st['pending_orders'] = []


def _portfolio_value():
    st = _STATE
    val = st['cash']
    for t, sh in st['positions'].items():
        p = st['last_close'].get(t)
        if _valid_price(p):
            val += sh * p
    return val


def on_bar(history):
    st = _STATE
    st['bar_count'] += 1
    orders = []

    try:
        ts_values = history['timestamp'].values
        current_ts = ts_values[-1]
        start_idx = np.searchsorted(ts_values, current_ts, side='left')
        current_rows = history.iloc[start_idx:]

        current_open = dict(zip(current_rows['ticker'], current_rows['open']))
        current_close = dict(zip(current_rows['ticker'], current_rows['close']))

        # 1. settle orders submitted on the previous bar, at this bar's open
        if st['pending_orders']:
            _fill_pending(current_open)

        # refresh last-known-close cache (used for valuation of stale/absent tickers)
        for t, p in current_close.items():
            if _valid_price(p):
                st['last_close'][t] = p

        # 2. trailing stop-loss, checked every bar
        stop_sell_tickers = set()
        for ticker, shares in list(st['positions'].items()):
            price = current_close.get(ticker)
            if not _valid_price(price):
                continue
            peak = max(st['peak_price'].get(ticker, price), price)
            st['peak_price'][ticker] = peak
            if price <= peak * (1.0 - st['stop_loss']):
                orders.append((ticker, 'SELL', int(shares)))
                stop_sell_tickers.add(ticker)

        # 3. periodic momentum rebalance
        do_rebalance = (
            st['bar_count'] % st['rebalance_period'] == 0
            and st['bar_count'] >= st['lookback']
        )

        if do_rebalance:
            unique_ts = pd.unique(history['timestamp'].values)
            if len(unique_ts) >= st['lookback'] + 1:
                window_start_ts = unique_ts[-(st['lookback'] + 1)]
                window = history[history['timestamp'] >= window_start_ts]
                close_pivot = window.pivot_table(
                    index='timestamp', columns='ticker', values='close'
                )
                valid = close_pivot.dropna(axis=1, how='any')

                if valid.shape[1] > 0:
                    rets = valid.pct_change().dropna(how='all')
                    mean_ret = rets.mean(axis=0)
                    vol = rets.std(axis=0).replace(0, np.nan)
                    score = (mean_ret / vol).dropna().sort_values(ascending=False)
                    selected = score[score > 0].head(st['top_n'])

                    weights = {}
                    if len(selected) > 0:
                        inv_vol = (1.0 / vol.loc[selected.index]).replace(
                            [np.inf, -np.inf], np.nan
                        ).dropna()
                        if inv_vol.sum() > 0:
                            w = (inv_vol / inv_vol.sum()).clip(upper=st['max_weight'])
                            weights = (w * st['max_alloc']).to_dict()

                    pv = _portfolio_value()

                    target_shares = {}
                    for ticker, weight in weights.items():
                        price = current_close.get(ticker)
                        if not _valid_price(price):
                            continue
                        target_shares[ticker] = int((pv * weight) // price)

                    all_relevant = set(st['positions'].keys()) | set(target_shares.keys())
                    sell_list, buy_list = [], []

                    for ticker in all_relevant:
                        if ticker in stop_sell_tickers:
                            continue  # already fully exited this bar
                        current_sh = st['positions'].get(ticker, 0)
                        target_sh = target_shares.get(ticker, 0)
                        diff = target_sh - current_sh
                        price = current_close.get(ticker)
                        if diff == 0 or not _valid_price(price):
                            continue
                        if abs(diff) * price < st['min_order_value']:
                            continue
                        if diff < 0:
                            sell_list.append((ticker, 'SELL', int(-diff)))
                        else:
                            buy_list.append((ticker, 'BUY', int(diff)))

                    # sells first so freed cash can support buys at the same execution bar
                    orders.extend(sell_list)
                    orders.extend(buy_list)

    except Exception:
        # never let a data hiccup crash the run; just skip trading this bar
        orders = []

    st['pending_orders'] = orders
    return orders
