import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Module-level state (persists across on_bar calls)
# ----------------------------------------------------------------------
_state = {}


def init(cash):
    """Called once before the first bar."""
    _state.clear()
    _state['cash'] = float(cash)
    _state['positions'] = {}       # ticker -> shares held
    _state['entry_price'] = {}     # ticker -> avg cost basis
    _state['peak_price'] = {}      # ticker -> peak close since entry
    _state['pending_orders'] = []  # orders submitted last on_bar call

    _state['bar_count'] = 0
    _state['last_rebalance'] = -10 ** 9

    # --- strategy parameters ---
    _state['lookback'] = 20            # bars for momentum/vol calc
    _state['rebalance_every'] = 5       # bars between rebalances
    _state['top_n'] = 8                 # max number of names held
    _state['max_invest_frac'] = 0.90    # cap on invested fraction of equity
    _state['stop_loss_frac'] = 0.12     # hard stop from avg cost
    _state['trailing_stop_frac'] = 0.15 # trailing stop from peak
    _state['min_history'] = 25          # bars required before first trade


def _safe_price(p):
    if p is None:
        return None
    try:
        if pd.isna(p) or p <= 0:
            return None
    except TypeError:
        return None
    return float(p)


def _apply_fills(current_open_prices):
    """
    Simulate execution of orders submitted on the previous bar,
    at this bar's open, using the platform's stated fill rules.
    """
    orders = _state['pending_orders']
    cash = _state['cash']
    positions = _state['positions']
    entry_price = _state['entry_price']
    peak_price = _state['peak_price']

    for ticker, direction, size in orders:
        price = _safe_price(current_open_prices.get(ticker))
        if price is None or size <= 0:
            continue  # zero fill: no price available or bad order

        if direction == "BUY":
            affordable = int(cash // price)
            qty = min(int(size), affordable)
            if qty > 0:
                cost = qty * price
                cash -= cost
                prev_shares = positions.get(ticker, 0)
                prev_cost_basis = entry_price.get(ticker, price) * prev_shares
                new_shares = prev_shares + qty
                entry_price[ticker] = (prev_cost_basis + cost) / new_shares
                positions[ticker] = new_shares
                peak_price[ticker] = max(peak_price.get(ticker, price), price)

        elif direction == "SELL":
            held = positions.get(ticker, 0)
            qty = min(int(size), held)
            if qty > 0:
                cash += qty * price
                remaining = held - qty
                if remaining <= 0:
                    positions.pop(ticker, None)
                    entry_price.pop(ticker, None)
                    peak_price.pop(ticker, None)
                else:
                    positions[ticker] = remaining

    _state['cash'] = cash
    _state['pending_orders'] = []


def _compute_orders(history):
    ts_all = history['timestamp'].unique()
    current_ts = ts_all[-1]
    current_rows = history[history['timestamp'] == current_ts]
    current_close = dict(zip(current_rows['ticker'], current_rows['close']))

    positions = _state['positions']
    cash = _state['cash']
    orders = []
    selling_now = set()

    # ---------------- risk management: stops ----------------
    for ticker in list(positions.keys()):
        price = _safe_price(current_close.get(ticker))
        if price is None:
            continue
        entry = _state['entry_price'].get(ticker, price)
        peak = max(_state['peak_price'].get(ticker, price), price)
        _state['peak_price'][ticker] = peak

        loss_from_entry = (entry - price) / entry if entry > 0 else 0.0
        drawdown_from_peak = (peak - price) / peak if peak > 0 else 0.0

        if loss_from_entry >= _state['stop_loss_frac'] or \
           drawdown_from_peak >= _state['trailing_stop_frac']:
            qty = positions[ticker]
            if qty > 0:
                orders.append((ticker, "SELL", qty))
                selling_now.add(ticker)

    # ---------------- periodic rebalance ----------------
    ready = _state['bar_count'] >= _state['min_history']
    due = (_state['bar_count'] - _state['last_rebalance']) >= _state['rebalance_every']

    if ready and due:
        _state['last_rebalance'] = _state['bar_count']
        lookback = _state['lookback']

        if len(ts_all) >= lookback + 1:
            boundary_ts = ts_all[-(lookback + 1)]
        else:
            boundary_ts = ts_all[0]

        recent_hist = history[history['timestamp'] >= boundary_ts]
        pivot = recent_hist.pivot_table(index='timestamp', columns='ticker',
                                         values='close', aggfunc='last')
        pivot = pivot.sort_index()

        if len(pivot) >= 2:
            first_row = pivot.iloc[0]
            last_row = pivot.iloc[-1]
            returns = (last_row / first_row - 1.0).dropna()
            vol = pivot.pct_change().std()

            valid = [t for t in returns.index if t not in selling_now]
            returns = returns.loc[valid]

            candidates = returns[returns > 0].sort_values(ascending=False)
            top_tickers = list(candidates.index[:_state['top_n']])

            # breadth-based exposure scaling
            if len(returns) > 0:
                frac_positive = float((returns > 0).mean())
            else:
                frac_positive = 0.0
            exposure_scale = min(1.0, frac_positive * 1.5)

            # current equity estimate
            equity = cash
            for t, sh in positions.items():
                p = _safe_price(current_close.get(t))
                if p is not None:
                    equity += sh * p

            invest_total = equity * _state['max_invest_frac'] * exposure_scale

            # sell names that fell out of favor
            for ticker in list(positions.keys()):
                if ticker not in top_tickers and ticker not in selling_now:
                    qty = positions[ticker]
                    if qty > 0:
                        orders.append((ticker, "SELL", qty))
                        selling_now.add(ticker)

            if top_tickers:
                top_vol = vol.reindex(top_tickers)
                positive_vol = top_vol[top_vol > 0]
                fill_val = positive_vol.min() if not positive_vol.empty else 1e-4
                top_vol = top_vol.fillna(fill_val).replace(0, fill_val)

                inv_vol = 1.0 / top_vol
                weights = inv_vol / inv_vol.sum()

                for ticker in top_tickers:
                    price = _safe_price(current_close.get(ticker))
                    if price is None:
                        continue
                    target_dollars = invest_total * float(weights[ticker])
                    target_shares = int(target_dollars // price)
                    current_shares = positions.get(ticker, 0)
                    diff = target_shares - current_shares

                    if diff > 0:
                        orders.append((ticker, "BUY", diff))
                    elif diff < 0:
                        qty = min(-diff, current_shares)
                        if qty > 0:
                            orders.append((ticker, "SELL", qty))

    # clean, positive-size only
    cleaned = []
    for t, d, s in orders:
        s_int = int(s)
        if s_int > 0 and d in ("BUY", "SELL"):
            cleaned.append((t, d, s_int))
    return cleaned


def on_bar(history):
    """Called once per time step, in chronological order."""
    _state['bar_count'] += 1

    try:
        ts_all = history['timestamp'].unique()
        current_ts = ts_all[-1]
        current_rows = history[history['timestamp'] == current_ts]
        current_open = dict(zip(current_rows['ticker'], current_rows['open']))

        # 1) settle orders placed on the previous bar at this bar's open
        _apply_fills(current_open)

        # 2) decide new orders based on updated state
        orders = _compute_orders(history)

    except Exception:
        # Fail safe: never let an error kill the whole run.
        orders = []

    _state['pending_orders'] = orders
    return orders
