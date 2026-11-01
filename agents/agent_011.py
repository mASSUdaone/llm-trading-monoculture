import numpy as np
from collections import deque

STATE = {}


def init(cash):
    global STATE
    STATE = {
        'cash': float(cash),
        'positions': {},          # ticker -> shares held (internal ledger)
        'pending_orders': [],     # orders returned last call; filled this call at open
        'price_history': {},      # ticker -> deque of recent closes
        'last_price': {},         # ticker -> most recent known close
        'peak_price': {},         # ticker -> peak close since entry (trailing stop)
        'processed_rows': 0,      # how many rows of history we've already consumed
        'bar_count': 0,

        # --- strategy parameters ---
        'lookback': 20,           # bars used for momentum / vol estimate
        'rebalance_every': 5,     # rebalance cadence, in bars
        'top_n': 10,              # max number of concurrent positions
        'max_weight': 0.20,       # cap per-position weight
        'invest_fraction': 0.95,  # keep some cash uninvested as buffer
        'stop_loss_pct': 0.12,    # trailing stop, fraction off peak
        'min_trade_value': 50.0,  # skip rebalancing trades smaller than this
    }


def _simulate_fills(open_prices):
    """Apply orders submitted on the previous bar using this bar's open prices,
    replicating the platform's fill rule exactly so our ledger stays accurate."""
    orders = STATE['pending_orders']
    STATE['pending_orders'] = []
    if not orders:
        return

    cash = STATE['cash']
    positions = STATE['positions']

    for ticker, direction, size in orders:
        if size <= 0:
            continue
        price = open_prices.get(ticker)
        if price is None or not np.isfinite(price) or price <= 0:
            continue  # no price available -> fills for zero shares

        if direction == 'BUY':
            affordable = int(cash // price)
            qty = min(size, affordable)
            if qty > 0:
                cash -= qty * price
                positions[ticker] = positions.get(ticker, 0) + qty

        elif direction == 'SELL':
            held = positions.get(ticker, 0)
            qty = min(size, held)
            if qty > 0:
                cash += qty * price
                remaining = held - qty
                if remaining <= 0:
                    positions.pop(ticker, None)
                else:
                    positions[ticker] = remaining

    STATE['cash'] = cash


def _update_price_history(closes):
    lookback = STATE['lookback']
    ph = STATE['price_history']
    for ticker, price in closes.items():
        if price is None or not np.isfinite(price) or price <= 0:
            continue
        dq = ph.get(ticker)
        if dq is None:
            dq = deque(maxlen=lookback + 1)
            ph[ticker] = dq
        dq.append(price)
        STATE['last_price'][ticker] = price


def _portfolio_value():
    value = STATE['cash']
    lp = STATE['last_price']
    for ticker, shares in STATE['positions'].items():
        price = lp.get(ticker)
        if price is not None:
            value += shares * price
    return value


def _select_candidates():
    """Rank tickers by risk-adjusted momentum; keep only positive-momentum names."""
    lookback = STATE['lookback']
    scores = {}
    vols = {}

    for ticker, dq in STATE['price_history'].items():
        if len(dq) < lookback + 1:
            continue
        arr = np.array(dq, dtype=float)
        ret = arr[-1] / arr[0] - 1.0
        rets = np.diff(arr) / arr[:-1]
        vol = float(np.std(rets))
        if vol <= 1e-6:
            vol = 1e-6
        scores[ticker] = ret / vol   # risk-adjusted momentum score
        vols[ticker] = vol

    candidates = [t for t, s in scores.items() if s > 0]
    candidates.sort(key=lambda t: scores[t], reverse=True)
    top = candidates[: STATE['top_n']]
    return top, vols


def _compute_target_positions():
    top, vols = _select_candidates()
    if not top:
        return {}

    inv_vol = {t: 1.0 / vols[t] for t in top}
    total_inv_vol = sum(inv_vol.values())
    weights = {t: inv_vol[t] / total_inv_vol for t in top}

    # cap weights at max_weight, redistribute excess to the rest, iterate to convergence
    max_w = STATE['max_weight']
    for _ in range(10):
        over = {t: w for t, w in weights.items() if w > max_w}
        if not over:
            break
        excess = sum(w - max_w for w in over.values())
        for t in over:
            weights[t] = max_w
        under = {t: w for t, w in weights.items() if w < max_w}
        under_total = sum(under.values())
        if under_total <= 0:
            break
        for t in under:
            weights[t] += excess * (under[t] / under_total)

    portfolio_value = _portfolio_value()
    invest_fraction = STATE['invest_fraction']
    lp = STATE['last_price']

    target_shares = {}
    for t, w in weights.items():
        price = lp.get(t)
        if price and price > 0:
            target_value = portfolio_value * invest_fraction * w
            target_shares[t] = int(target_value // price)
    return target_shares


def on_bar(history):
    STATE['bar_count'] += 1

    n_prev = STATE['processed_rows']
    new_rows = history.iloc[n_prev:]
    STATE['processed_rows'] = len(history)

    if new_rows.empty:
        return []

    current_opens = dict(zip(new_rows['ticker'], new_rows['open']))
    current_closes = dict(zip(new_rows['ticker'], new_rows['close']))

    # 1. settle orders placed on the previous bar, using this bar's open
    _simulate_fills(current_opens)

    # 2. update rolling signal data
    _update_price_history(current_closes)

    sell_orders = []
    buy_orders = []
    sold_tickers = set()

    # 3. trailing stop-loss check on current holdings (every bar)
    stop_pct = STATE['stop_loss_pct']
    for ticker, shares in list(STATE['positions'].items()):
        price = current_closes.get(ticker, STATE['last_price'].get(ticker))
        if price is None:
            continue
        peak = STATE['peak_price'].get(ticker, price)
        if price > peak:
            peak = price
        STATE['peak_price'][ticker] = peak
        if price <= peak * (1.0 - stop_pct):
            sell_orders.append((ticker, 'SELL', shares))
            sold_tickers.add(ticker)

    # 4. periodic rebalance toward momentum-based targets
    if STATE['bar_count'] % STATE['rebalance_every'] == 0:
        target = _compute_target_positions()
        current_positions = STATE['positions']
        lp = STATE['last_price']

        all_tickers = set(target.keys()) | set(current_positions.keys())
        for ticker in all_tickers:
            if ticker in sold_tickers:
                continue
            tgt = target.get(ticker, 0)
            cur = current_positions.get(ticker, 0)
            diff = tgt - cur
            price = lp.get(ticker)
            if price is None or price <= 0:
                continue
            if abs(diff) * price < STATE['min_trade_value']:
                continue
            if diff > 0:
                buy_orders.append((ticker, 'BUY', diff))
            elif diff < 0:
                sell_orders.append((ticker, 'SELL', -diff))

        # maintain peak-price tracking only for names we currently care about
        keep = set(target.keys()) | set(current_positions.keys())
        for ticker in target:
            STATE['peak_price'].setdefault(ticker, lp.get(ticker))
        for ticker in list(STATE['peak_price'].keys()):
            if ticker not in keep:
                STATE['peak_price'].pop(ticker, None)

    orders = sell_orders + buy_orders  # sells first to free up cash for buys
    STATE['pending_orders'] = orders
    return orders
