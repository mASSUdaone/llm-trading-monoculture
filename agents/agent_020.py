import numpy as np
import pandas as pd

# ----------------------------- Strategy parameters -----------------------------
LOOKBACK        = 126   # momentum lookback window (bars)
SKIP            = 5     # skip most recent bars (avoid short-term reversal)
TREND_WINDOW    = 200   # long moving average window for trend filter
VOL_WINDOW      = 20    # realized volatility window for weighting
REBALANCE_EVERY = 10    # rebalance cadence (bars)
TOP_FRACTION    = 0.15  # fraction of eligible universe to hold
MAX_POSITIONS   = 20
MIN_POSITIONS   = 3
MAX_WEIGHT      = 0.15  # cap per-name weight
CASH_BUFFER     = 0.05  # keep this fraction of equity uninvested
MIN_PRICE       = 1.0   # ignore penny-priced garbage

# ----------------------------- Internal state -----------------------------
STATE = {}


def init(cash):
    STATE.clear()
    STATE["cash"] = float(cash)
    STATE["positions"] = {}      # ticker -> shares held (int)
    STATE["last_price"] = {}     # ticker -> last known close
    STATE["pending_orders"] = [] # orders submitted last bar, to settle now
    STATE["bar_count"] = 0
    STATE["last_rebalance_bar"] = -REBALANCE_EVERY


def _settle_pending(cur_open_map):
    """Apply last bar's orders using this bar's open prices, mirroring the
    platform's fill rule exactly, so our ledger stays in sync."""
    cash = STATE["cash"]
    positions = STATE["positions"]

    for ticker, direction, size in STATE["pending_orders"]:
        price = cur_open_map.get(ticker)
        if price is None or not np.isfinite(price) or price <= 0 or size <= 0:
            continue  # zero fill, no price available

        if direction == "BUY":
            affordable = int(np.floor(cash / price))
            fill = min(int(size), affordable)
            if fill > 0:
                cash -= fill * price
                positions[ticker] = positions.get(ticker, 0) + fill
        elif direction == "SELL":
            held = positions.get(ticker, 0)
            fill = min(int(size), held)
            if fill > 0:
                positions[ticker] = held - fill
                cash += fill * price

    STATE["cash"] = cash
    STATE["pending_orders"] = []


def _recent_slice(history, n_timestamps):
    """Efficiently slice history to the last n_timestamps distinct timestamps,
    relying on the fact that history is append-only and time-sorted."""
    ts_values = history["timestamp"].values
    unique_ts = pd.unique(ts_values)
    if len(unique_ts) <= n_timestamps:
        return history
    start_ts = unique_ts[-n_timestamps]
    start_idx = np.searchsorted(ts_values, start_ts, side="left")
    return history.iloc[start_idx:]


def _compute_targets(history, equity, current_time):
    """Return dict ticker -> target share count."""
    window_needed = max(LOOKBACK + SKIP, TREND_WINDOW) + 2
    recent = _recent_slice(history, window_needed)

    pivot = recent.pivot_table(index="timestamp", columns="ticker", values="close")
    pivot = pivot.sort_index()

    n_rows = len(pivot)
    if n_rows < max(LOOKBACK + SKIP, TREND_WINDOW) + 1:
        return {}  # not enough history yet

    last_row = pivot.iloc[-1]
    cur_tickers = history.loc[history["timestamp"] == current_time, "ticker"].unique()

    # Momentum
    mom_now_idx = n_rows - 1 - SKIP
    mom_past_idx = n_rows - 1 - SKIP - LOOKBACK
    if mom_now_idx < 0 or mom_past_idx < 0:
        return {}
    price_now = pivot.iloc[mom_now_idx]
    price_past = pivot.iloc[mom_past_idx]
    with np.errstate(divide="ignore", invalid="ignore"):
        momentum = (price_now / price_past) - 1.0

    # Trend filter
    sma = pivot.tail(TREND_WINDOW).mean()

    # Volatility (for weighting)
    rets = pivot.pct_change()
    vol = rets.tail(VOL_WINDOW).std()

    candidates = []
    for ticker in cur_tickers:
        if ticker not in pivot.columns:
            continue
        px = last_row.get(ticker, np.nan)
        m = momentum.get(ticker, np.nan)
        s = sma.get(ticker, np.nan)
        v = vol.get(ticker, np.nan)

        if not np.isfinite(px) or px < MIN_PRICE:
            continue
        if not np.isfinite(m) or not np.isfinite(s) or not np.isfinite(v):
            continue
        if m <= 0:
            continue
        if px <= s:
            continue
        if v <= 0:
            continue

        candidates.append((ticker, m, v, px))

    if not candidates:
        return {}

    candidates.sort(key=lambda x: x[1], reverse=True)
    n_select = int(round(len(candidates) * TOP_FRACTION))
    n_select = max(MIN_POSITIONS, min(MAX_POSITIONS, n_select))
    n_select = min(n_select, len(candidates))
    selected = candidates[:n_select]

    if not selected:
        return {}

    inv_vol = np.array([1.0 / c[2] for c in selected])
    weights = inv_vol / inv_vol.sum()

    # cap and renormalize (couple of passes to approximate a proper cap)
    for _ in range(3):
        capped = np.minimum(weights, MAX_WEIGHT)
        s = capped.sum()
        if s <= 0:
            break
        weights = capped / s

    weights = weights * (1.0 - CASH_BUFFER)

    targets = {}
    for (ticker, m, v, px), w in zip(selected, weights):
        dollars = equity * w
        shares = int(np.floor(dollars / px))
        if shares > 0:
            targets[ticker] = shares

    return targets


def on_bar(history):
    try:
        return _on_bar_impl(history)
    except Exception:
        # Never let an internal error break the run.
        return []


def _on_bar_impl(history):
    STATE.setdefault("cash", 0.0)
    STATE.setdefault("positions", {})
    STATE.setdefault("last_price", {})
    STATE.setdefault("pending_orders", [])
    STATE.setdefault("bar_count", 0)
    STATE.setdefault("last_rebalance_bar", -REBALANCE_EVERY)

    current_time = history["timestamp"].iloc[-1]
    cur = history[history["timestamp"] == current_time]

    open_map = dict(zip(cur["ticker"], cur["open"]))
    close_map = dict(zip(cur["ticker"], cur["close"]))

    # 1. Settle orders issued on the previous bar using this bar's open.
    _settle_pending(open_map)

    # 2. Update last known prices.
    for t, p in close_map.items():
        if np.isfinite(p):
            STATE["last_price"][t] = p

    STATE["bar_count"] += 1
    bar_count = STATE["bar_count"]

    orders = []

    min_history_needed = max(LOOKBACK + SKIP, TREND_WINDOW) + 5
    ready_for_rebalance = (
        bar_count - STATE["last_rebalance_bar"] >= REBALANCE_EVERY
        and bar_count >= min_history_needed
    )

    if ready_for_rebalance:
        equity = STATE["cash"] + sum(
            sh * STATE["last_price"].get(t, 0.0)
            for t, sh in STATE["positions"].items()
            if sh > 0
        )
        if equity > 0:
            targets = _compute_targets(history, equity, current_time)

            positions = STATE["positions"]
            all_tickers = set(targets.keys()) | {t for t, s in positions.items() if s > 0}

            sells, buys = [], []
            for t in all_tickers:
                target = targets.get(t, 0)
                held = positions.get(t, 0)
                diff = target - held
                if diff < 0:
                    sells.append((t, "SELL", int(-diff)))
                elif diff > 0:
                    buys.append((t, "BUY", int(diff)))

            orders = sells + buys
            STATE["last_rebalance_bar"] = bar_count

    STATE["pending_orders"] = list(orders)
    return orders
