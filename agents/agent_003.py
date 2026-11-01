import numpy as np
import pandas as pd
from collections import deque

# ---------------- strategy parameters ----------------
LOOKBACK_MOM      = 63     # bars for momentum measurement window
MOM_SKIP          = 5      # bars to skip at the end (avoid short-term reversal)
LOOKBACK_VOL      = 21     # bars for volatility estimate
MAX_HIST          = 260    # max closes kept per ticker
N_TOP             = 8      # max number of concurrent positions
REBALANCE_EVERY   = 10     # bars between rebalances
MAX_WEIGHT        = 0.20   # max weight per position
TARGET_INVEST     = 0.95   # target fraction of equity invested
TRAILING_STOP_PCT = 0.15   # sell if price falls this much from peak since held
WARMUP            = LOOKBACK_MOM + MOM_SKIP + 2

_state = {}


def init(cash):
    _state.clear()
    _state['cash'] = float(cash)
    _state['positions'] = {}        # ticker -> int shares
    _state['pending_orders'] = []   # orders submitted last bar, to fill at this bar's open
    _state['closes'] = {}           # ticker -> deque of recent closes
    _state['peak'] = {}             # ticker -> peak close since position opened
    _state['last_len'] = 0          # rows of history already consumed
    _state['bar_count'] = 0


def _execute_pending(open_prices):
    """Replicate the platform's fill logic exactly, using this bar's open prices,
    to keep our internal cash/position ledger consistent with reality."""
    cash = _state['cash']
    positions = _state['positions']
    for ticker, direction, size in _state['pending_orders']:
        price = open_prices.get(ticker)
        if price is None or pd.isna(price) or price <= 0 or size <= 0:
            continue
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
                if remaining > 0:
                    positions[ticker] = remaining
                else:
                    positions.pop(ticker, None)
    _state['cash'] = cash
    _state['pending_orders'] = []


def on_bar(history):
    st = _state
    n = len(history)
    new_rows = history.iloc[st['last_len']:n]
    st['last_len'] = n
    if new_rows.empty:
        return []

    open_prices = dict(zip(new_rows['ticker'], new_rows['open']))
    close_prices_raw = dict(zip(new_rows['ticker'], new_rows['close']))
    close_prices = {t: p for t, p in close_prices_raw.items() if p is not None and not pd.isna(p) and p > 0}

    # 1. settle orders queued from the previous bar, using this bar's open prices
    _execute_pending(open_prices)

    # 2. update rolling close-price history
    closes = st['closes']
    for t, c in close_prices.items():
        dq = closes.get(t)
        if dq is None:
            dq = deque(maxlen=MAX_HIST)
            closes[t] = dq
        dq.append(c)

    st['bar_count'] += 1
    positions = st['positions']
    orders = []
    stopped_out = set()

    # 3. trailing-stop risk overlay (every bar, independent of rebalance schedule)
    peak = st['peak']
    for t in list(positions.keys()):
        price = close_prices.get(t)
        if price is None:
            continue
        pk = max(peak.get(t, price), price)
        peak[t] = pk
        if price <= pk * (1 - TRAILING_STOP_PCT):
            held = positions.get(t, 0)
            if held > 0:
                orders.append((t, 'SELL', held))
                stopped_out.add(t)
                peak.pop(t, None)

    # 4. periodic momentum rebalance
    if st['bar_count'] >= WARMUP and st['bar_count'] % REBALANCE_EVERY == 0:
        scores, vols = {}, {}
        need = LOOKBACK_MOM + MOM_SKIP
        for t, dq in closes.items():
            if t in stopped_out or len(dq) < need:
                continue
            price_now = dq[-MOM_SKIP] if MOM_SKIP > 0 else dq[-1]
            price_ref = dq[-need]
            if price_ref <= 0:
                continue
            mom = price_now / price_ref - 1.0
            if mom <= 0:
                continue
            arr = np.asarray(dq)[-(LOOKBACK_VOL + 1):]
            if len(arr) >= 2:
                rets = np.diff(arr) / arr[:-1]
                vol = float(np.std(rets))
            else:
                vol = np.nan
            if not np.isfinite(vol) or vol <= 1e-6:
                vol = 1e-2
            scores[t] = mom
            vols[t] = vol

        selected = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:N_TOP]
        sel_tickers = [t for t, _ in selected]

        # portfolio equity valuation (cash + mark-to-market positions)
        equity = st['cash']
        for t, sh in positions.items():
            p = close_prices.get(t)
            if p is None:
                dq = closes.get(t)
                p = dq[-1] if dq else 0.0
            equity += sh * p

        weights = {}
        if sel_tickers:
            inv_vol = {t: 1.0 / vols[t] for t in sel_tickers}
            tot = sum(inv_vol.values())
            for t in sel_tickers:
                w = (inv_vol[t] / tot) * TARGET_INVEST
                weights[t] = min(w, MAX_WEIGHT)

        target_shares = {}
        for t, w in weights.items():
            price = close_prices.get(t)
            if price is None or price <= 0:
                continue
            target_shares[t] = int((w * equity) // price)

        all_relevant = set(positions.keys()) | set(target_shares.keys())
        for t in all_relevant:
            if t in stopped_out:
                continue
            cur = positions.get(t, 0)
            tgt = target_shares.get(t, 0)
            price = close_prices.get(t)
            if price is None or price <= 0:
                continue
            if tgt > cur:
                orders.append((t, 'BUY', tgt - cur))
            elif tgt < cur:
                orders.append((t, 'SELL', cur - tgt))

    st['pending_orders'] = orders
    return orders
