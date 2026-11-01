import numpy as np
import pandas as pd

# ---------------- Strategy configuration ----------------
LOOKBACK_MOM   = 63     # bars used for momentum score
LOOKBACK_VOL   = 20     # bars used for volatility estimate
TREND_SMA      = 50     # trend filter moving-average window
REBALANCE_EVERY = 10    # bars between full rebalances
TOP_N          = 8      # number of names to hold
MAX_WEIGHT     = 0.20   # cap per position (fraction of portfolio)
GROSS_TARGET   = 0.95   # fraction of portfolio value to deploy
MIN_HISTORY    = 60     # min bars of history required for a ticker
STOP_LOSS      = 0.15   # trailing stop vs highest close since entry
HISTORY_WINDOW = 400    # bars of history considered at each rebalance (perf cap)

_state = {}


def init(cash):
    _state.clear()
    _state['cash'] = float(cash)
    _state['positions'] = {}       # ticker -> shares (internal ledger)
    _state['pending_orders'] = []  # orders submitted last call, to fill this bar
    _state['bar_count'] = 0
    _state['entry_high'] = {}      # ticker -> highest close since entry (for stop)
    _state['last_prices'] = {}     # ticker -> last known close


def _apply_fills(open_prices):
    """Replicate the platform's fill logic for orders queued on the previous bar,
    using the current bar's open prices. Keeps our internal ledger exact."""
    orders = _state['pending_orders']
    _state['pending_orders'] = []
    cash = _state['cash']
    positions = _state['positions']

    for ticker, direction, size in orders:
        price = open_prices.get(ticker)
        if price is None or price <= 0 or size <= 0:
            continue  # no price available -> fills for zero shares
        if direction == 'BUY':
            max_afford = int(cash // price)
            qty = min(size, max_afford)
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


def on_bar(history):
    try:
        return _on_bar_impl(history)
    except Exception:
        # Never let an unexpected data issue crash the whole run.
        return []


def _on_bar_impl(history):
    _state['bar_count'] += 1

    ts = history['timestamp'].iloc[-1]
    current = history[history['timestamp'] == ts]
    open_prices = dict(zip(current['ticker'], current['open']))
    close_prices = dict(zip(current['ticker'], current['close']))

    # 1) Settle orders queued on the previous call using this bar's open.
    _apply_fills(open_prices)

    # 2) Update price memory (used for portfolio valuation when a ticker is absent).
    for t, p in close_prices.items():
        if pd.notna(p):
            _state['last_prices'][t] = float(p)

    positions = _state['positions']
    cash = _state['cash']

    port_value = cash + sum(
        sh * _state['last_prices'].get(t, 0.0) for t, sh in positions.items()
    )
    if port_value <= 0:
        port_value = cash

    orders = []

    # 3) Always-on trailing stop-loss check.
    stopped = []
    for t, sh in list(positions.items()):
        price = close_prices.get(t)
        if price is None or pd.isna(price):
            continue
        hi = _state['entry_high'].get(t, price)
        if price > hi:
            hi = price
        _state['entry_high'][t] = hi
        if price <= hi * (1.0 - STOP_LOSS):
            stopped.append(t)

    for t in stopped:
        sh = positions.get(t, 0)
        if sh > 0:
            orders.append((t, 'SELL', sh))
            _state['entry_high'].pop(t, None)

    # 4) Periodic rebalance.
    if _state['bar_count'] % REBALANCE_EVERY == 0:
        unique_ts = history['timestamp'].unique()
        if len(unique_ts) > HISTORY_WINDOW:
            cutoff_ts = unique_ts[-HISTORY_WINDOW]
            hist_recent = history[history['timestamp'] >= cutoff_ts]
        else:
            hist_recent = history

        pivot = hist_recent.pivot_table(index='timestamp', columns='ticker', values='close')
        pivot = pivot.sort_index()

        available_now = set(current['ticker'])
        stopped_set = set(stopped)

        scores, vols, trend_ok = {}, {}, {}
        for t in pivot.columns:
            if t not in available_now:
                continue
            series = pivot[t].dropna()
            if len(series) < MIN_HISTORY or series.iloc[-1] <= 0:
                continue

            look = min(LOOKBACK_MOM, len(series) - 1)
            if look < 5:
                continue
            mom = series.iloc[-1] / series.iloc[-1 - look] - 1.0

            rets = series.pct_change().dropna()
            vol_window = rets.iloc[-LOOKBACK_VOL:] if len(rets) >= LOOKBACK_VOL else rets
            vol = vol_window.std()
            if vol is None or vol == 0 or pd.isna(vol):
                continue

            sma_window = min(TREND_SMA, len(series))
            sma = series.iloc[-sma_window:].mean()

            scores[t] = mom
            vols[t] = vol
            trend_ok[t] = series.iloc[-1] > sma

        candidates = [t for t in scores if trend_ok.get(t, False) and t not in stopped_set]
        candidates.sort(key=lambda t: scores[t], reverse=True)
        top = candidates[:TOP_N]

        weights = {}
        if top:
            inv_vol = {t: 1.0 / vols[t] for t in top}
            total_inv = sum(inv_vol.values())
            raw_weights = {t: inv_vol[t] / total_inv for t in top}
            capped = {t: min(w, MAX_WEIGHT) for t, w in raw_weights.items()}
            total_capped = sum(capped.values())
            if total_capped > 0:
                weights = {t: (capped[t] / total_capped) * GROSS_TARGET for t in capped}

        current_after_stop = {t: sh for t, sh in positions.items() if t not in stopped_set}

        # Exit anything no longer in the target set.
        for t, sh in current_after_stop.items():
            if t not in weights and sh > 0:
                orders.append((t, 'SELL', sh))

        # Move each target name toward its target weight.
        for t, w in weights.items():
            price = close_prices.get(t)
            if price is None or pd.isna(price) or price <= 0:
                continue
            target_dollar = port_value * w
            target_shares = int(target_dollar // price)
            held = current_after_stop.get(t, 0)
            diff = target_shares - held
            if diff > 0:
                orders.append((t, 'BUY', diff))
            elif diff < 0:
                orders.append((t, 'SELL', -diff))
            _state['entry_high'].setdefault(t, price)

    _state['pending_orders'] = list(orders)
    return orders
