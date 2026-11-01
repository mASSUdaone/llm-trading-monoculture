#!/usr/bin/env python3
"""
score.py -- turn trades.csv files into positions, behaviour stats, and
pairwise overlap between agents.

    python3 score.py runs/gemini_001                 # one agent
    python3 score.py runs/*                          # table across agents
    python3 score.py runs/* --overlap                # + pairwise Jaccard/cosine

What it does
------------
1. Replays every order against the tape using the platform's own fill rules
   (next bar's open, partial fills, no shorting, no borrowing) to reconstruct
   what each agent actually HELD at every bar. trades.csv records intentions;
   this recovers outcomes.
2. Summarises each agent's behaviour: how often it traded, how many names it
   held, how long it held them, how concentrated it was, how often it sat flat.
3. With two or more agents, computes per-bar holdings overlap between every
   pair -- binary Jaccard on held names, and cosine similarity on position
   value weights.

Scoring decisions baked in here (write these into the pre-commitment doc):
  * SELLs are processed before BUYs within a bar, so sale proceeds are
    available to same-bar buys. The contract does not specify an order; this
    one is stated rather than left implicit.
  * Only bars from 2025 onward are scored. 2024 is burn-in.
  * A pair's Jaccard is averaged over bars where AT LEAST ONE of the two holds
    something. Bars where both are flat are undefined (0/0) and excluded, but
    counted separately -- simultaneous flatness is itself a coordination
    signal and is reported as flat_frac, not silently dropped.
  * A ticker with no price on a bar keeps its last known price for valuation.
    It is still held; absence of a quote is not a sale.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

CASH = 100_000.0
SCORE_FROM = "2025-01-01"
RESULTS_DIR = "results"


# ---------------------------------------------------------------- tape

def load_tape():
    if os.path.exists("market.parquet"):
        df = pd.read_parquet("market.parquet")
    elif os.path.exists(os.path.join("data", "market.csv")):
        df = pd.read_csv(os.path.join("data", "market.csv"))
    else:
        sys.exit("error: no market.parquet or data/market.csv found")
    return df.sort_values(["timestamp", "ticker"], kind="mergesort")


def build_matrices(df):
    """bars x tickers matrices of open (fills) and close (valuation)."""
    bars = np.array(sorted(df["timestamp"].unique()))
    tickers = np.array(sorted(df["ticker"].unique()))
    bi = {b: i for i, b in enumerate(bars)}
    ti = {t: i for i, t in enumerate(tickers)}

    opens = np.full((len(bars), len(tickers)), np.nan, dtype=np.float32)
    closes = np.full((len(bars), len(tickers)), np.nan, dtype=np.float32)
    r = df["timestamp"].map(bi).to_numpy()
    c = df["ticker"].map(ti).to_numpy()
    opens[r, c] = df["open"].to_numpy(dtype=np.float32)
    closes[r, c] = df["close"].to_numpy(dtype=np.float32)

    # carry the last known close forward for valuation only
    for j in range(closes.shape[1]):
        col = closes[:, j]
        idx = np.where(~np.isnan(col))[0]
        if len(idx):
            filled = pd.Series(col).ffill().to_numpy()
            closes[:, j] = filled
    return bars, tickers, bi, ti, opens, closes


# ------------------------------------------------------- reconstruction

def reconstruct(trades, bars, tickers, bi, ti, opens, closes):
    """Replay orders under the platform's fill rules. Returns shares matrix."""
    nb, nt = len(bars), len(tickers)
    shares = np.zeros((nb, nt), dtype=np.int32)
    cash = np.zeros(nb, dtype=np.float64)

    # group orders by decision bar
    pend = {}
    for ts, tk, d, sz in trades.itertuples(index=False):
        if ts in bi and tk in ti:
            pend.setdefault(bi[ts], []).append((ti[tk], d, int(sz)))

    held = np.zeros(nt, dtype=np.int64)
    c = CASH
    fill_bar = []          # (bar, kind) so fills can be windowed like orders

    for b in range(nb):
        orders = pend.get(b - 1)          # decided last bar, fills at this open
        if orders:
            px = opens[b]
            # sells first, so proceeds fund same-bar buys
            for j, d, sz in orders:
                if d != "SELL":
                    continue
                p = px[j]
                if not np.isfinite(p) or p <= 0:
                    fill_bar.append((b, "zero")); continue
                q = min(sz, held[j])
                if q <= 0:
                    fill_bar.append((b, "zero")); continue
                held[j] -= q
                c += q * float(p)
                fill_bar.append((b, "partial" if q < sz else "full"))
            for j, d, sz in orders:
                if d != "BUY":
                    continue
                p = px[j]
                if not np.isfinite(p) or p <= 0:
                    fill_bar.append((b, "zero")); continue
                q = min(sz, int(c // float(p)))
                if q <= 0:
                    fill_bar.append((b, "zero")); continue
                held[j] += q
                c -= q * float(p)
                fill_bar.append((b, "partial" if q < sz else "full"))
        shares[b] = held
        cash[b] = c

    values = shares.astype(np.float32) * np.nan_to_num(closes)
    equity = cash + values.sum(axis=1)
    return shares, cash, values, equity, fill_bar


# ------------------------------------------------------------ behaviour

def benchmark(closes, keep):
    """Equal-weighted buy & hold across every ticker with a price at the start."""
    c = closes[keep]
    first = c[0]
    live = np.isfinite(first) & (first > 0)
    if not live.any():
        return None
    per = CASH / live.sum()
    sh = per / first[live]
    return (np.nan_to_num(c[:, live]) * sh).sum(axis=1)


def risk_stats(equity, bars_per_year):
    """Annualised vol, Sharpe (rf=0), and worst peak-to-trough drop."""
    if len(equity) < 3:
        return {}
    r = np.diff(equity) / np.maximum(equity[:-1], 1e-9)
    sd = float(np.std(r))
    return {
        "ann_vol": round(sd * np.sqrt(bars_per_year), 4),
        "sharpe": round(float(np.mean(r)) / sd * np.sqrt(bars_per_year), 2) if sd > 1e-12 else 0.0,
        "max_drawdown": round(float(np.min(equity / np.maximum.accumulate(equity)) - 1), 4),
    }


def holding_periods(shares):
    """Bars between a name going from zero to nonzero and back to zero."""
    out = []
    nz = shares > 0
    for j in range(shares.shape[1]):
        col = nz[:, j]
        if not col.any():
            continue
        d = np.diff(col.astype(np.int8))
        starts = list(np.where(d == 1)[0] + 1)
        ends = list(np.where(d == -1)[0] + 1)
        if col[0]:
            starts = [0] + starts
        if col[-1]:
            ends = ends + [len(col)]
        out.extend(e - s for s, e in zip(starts, ends))
    return np.array(out) if out else np.array([0])


def window_fills(fill_bar, first_bar):
    """Count fills only in the scored window, so rates match the order count."""
    sel = [k for b, k in fill_bar if b >= first_bar]
    return {"filled": sum(k != "zero" for k in sel),
            "partial": sum(k == "partial" for k in sel),
            "zero": sum(k == "zero" for k in sel)}


def summarise(name, trades, shares, values, equity, fills, bars, bench=None,
              shares_traded=0, bars_per_year=3250):
    nz = shares > 0
    n_held = nz.sum(axis=1)
    flat = n_held == 0
    hp = holding_periods(shares)

    w = values / np.where(values.sum(axis=1, keepdims=True) > 0,
                          values.sum(axis=1, keepdims=True), 1.0)
    hhi = (w ** 2).sum(axis=1)
    active = ~flat

    return {
        "agent": name,
        "orders": len(trades),
        "orders_per_bar": round(len(trades) / max(len(bars), 1), 2),
        "buys": int((trades["direction"] == "BUY").sum()),
        "sells": int((trades["direction"] == "SELL").sum()),
        "fill_rate": round(fills["filled"] / max(len(trades), 1), 3),
        "partial_fills": fills["partial"],
        "zero_fills": fills["zero"],
        "tickers_ever_held": int(nz.any(axis=0).sum()),
        "mean_names_held": round(float(n_held.mean()), 1),
        "median_names_held": int(np.median(n_held)),
        "max_names_held": int(n_held.max()),
        "flat_frac": round(float(flat.mean()), 3),
        "median_hold_bars": int(np.median(hp)),
        "mean_hold_bars": round(float(hp.mean()), 1),
        "concentration_hhi": round(float(hhi[active].mean()) if active.any() else 0.0, 4),
        "final_equity": round(float(equity[-1]), 2),
        "total_return": round(float(equity[-1] / CASH - 1), 4),
        **risk_stats(equity, bars_per_year),
        "benchmark_return": round(float(bench[-1] / CASH - 1), 4) if bench is not None else None,
        "excess_vs_benchmark": (round(float(equity[-1] / CASH - bench[-1] / CASH), 4)
                                if bench is not None else None),
        "shares_traded": shares_traded,
        **{f"net_ret_{int(r*1000)}mils": round(
            float((equity[-1] - shares_traded * r) / CASH - 1), 4)
           for r in (0.005, 0.01, 0.02)},
    }


# -------------------------------------------------------------- overlap

def overlap(a_shares, b_shares, a_val, b_val):
    A, B = a_shares > 0, b_shares > 0
    inter = (A & B).sum(axis=1)
    union = (A | B).sum(axis=1)
    ok = union > 0                                  # both flat -> undefined
    jac = float((inter[ok] / union[ok]).mean()) if ok.any() else float("nan")

    na = np.linalg.norm(a_val, axis=1)
    nb = np.linalg.norm(b_val, axis=1)
    good = (na > 0) & (nb > 0)
    cos = float(((a_val[good] * b_val[good]).sum(axis=1) /
                 (na[good] * nb[good])).mean()) if good.any() else float("nan")

    both_flat = float((~ok).mean())
    return jac, cos, both_flat, int(ok.sum())


def crowding(store, names):
    """Population-level concentration of holdings, per bar.

    Pairwise overlap says two agents resemble each other. Crowding says how
    much of the whole population is piled into the same names at the same
    moment -- which is the quantity the systemic-risk story actually needs.
    Linear in the number of agents, not quadratic.
    """
    H = np.stack([store[n][0] > 0 for n in names])        # agents x bars x tickers
    n_agents = H.shape[0]
    per_bar_holders = H.sum(axis=0)                       # bars x tickers

    # share of the population holding the single most crowded name
    top1 = per_bar_holders.max(axis=1) / n_agents

    # share of all held positions sitting in the 5 most crowded names
    total = per_bar_holders.sum(axis=1)
    top5 = np.sort(per_bar_holders, axis=1)[:, -5:].sum(axis=1)
    top5_share = np.divide(top5, total, out=np.zeros_like(top5, dtype=float),
                           where=total > 0)

    # Herfindahl across tickers: 1/HHI is the effective number of distinct
    # names the population is spread over at that bar
    p = np.divide(per_bar_holders, np.where(total[:, None] > 0, total[:, None], 1))
    hhi = (p ** 2).sum(axis=1)
    eff_names = np.divide(1.0, hhi, out=np.zeros_like(hhi), where=hhi > 0)

    return {
        "agents": n_agents,
        "mean_top1_share": round(float(top1.mean()), 4),
        "median_top1_share": round(float(np.median(top1)), 4),
        "max_top1_share": round(float(top1.max()), 4),
        "mean_top5_share": round(float(top5_share.mean()), 4),
        "mean_effective_names": round(float(eff_names[total > 0].mean()), 1),
        "median_effective_names": round(float(np.median(eff_names[total > 0])), 1),
    }


def distribution(vals, label):
    v = np.array([x for x in vals if np.isfinite(x)])
    if not len(v):
        return None
    q = np.percentile(v, [0, 25, 50, 75, 100])
    print(f"\n{label}  (n={len(v)})")
    print(f"  mean {v.mean():.4f}   sd {v.std():.4f}")
    print(f"  min {q[0]:.4f}   q1 {q[1]:.4f}   median {q[2]:.4f}   "
          f"q3 {q[3]:.4f}   max {q[4]:.4f}")
    # crude histogram, 10 buckets from 0 to 1
    hist, _ = np.histogram(v, bins=10, range=(0, 1))
    for i, h in enumerate(hist):
        if h:
            print(f"    {i/10:.1f}-{(i+1)/10:.1f}  {'#' * min(h, 50)} {h}")
    return {"metric": label, "n_pairs": len(v),
            "mean": round(float(v.mean()), 4), "sd": round(float(v.std(ddof=1)), 4),
            "sd_over_mean": round(float(v.std(ddof=1) / v.mean()), 3) if v.mean() else None,
            "min": round(float(q[0]), 4), "q1": round(float(q[1]), 4),
            "median": round(float(q[2]), 4), "q3": round(float(q[3]), 4),
            "max": round(float(q[4]), 4)}


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run directories containing trades.csv")
    ap.add_argument("--overlap", action="store_true", help="pairwise comparison")
    ap.add_argument("--from", dest="start", default=SCORE_FROM)
    ap.add_argument("--save", action="store_true", help="write positions to npz")
    ap.add_argument("--min-active", type=float, default=0.05, dest="min_active",
                    help="exclude an agent from overlap/crowding if it holds "
                         "something on fewer than this fraction of scored bars "
                         "(default 0.05). Behaviour rows are always kept.")
    args = ap.parse_args()

    print("loading tape...")
    df = load_tape()
    bars, tickers, bi, ti, opens, closes = build_matrices(df)
    keep = bars >= args.start
    print(f"{len(bars):,} bars, {len(tickers)} tickers "
          f"({keep.sum():,} bars scored from {args.start})\n")

    rows, store = [], {}
    for run in args.runs:
        path = os.path.join(run, "trades.csv")
        if not os.path.exists(path):
            print(f"  skip {run} (no trades.csv)")
            continue
        name = os.path.basename(run.rstrip("/"))
        trades = pd.read_csv(path, dtype={"timestamp": str})
        shares, cash, values, equity, fill_bar = reconstruct(
            trades, bars, tickers, bi, ti, opens, closes)
        fills = window_fills(fill_bar, int(np.argmax(keep)))

        s, v, e = shares[keep], values[keep], equity[keep]
        scored = trades[trades["timestamp"] >= args.start]
        bench = benchmark(closes, keep)
        shtr = int(scored["size"].sum()) if len(scored) else 0
        row = summarise(name, scored, s, v, e, fills, bars[keep],
                        bench=bench, shares_traded=shtr,
                        bars_per_year=int(keep.sum()))
        rows.append(row)
        pd.DataFrame([row]).set_index("agent").T.to_csv(
            os.path.join(run, "summary.csv"), header=["value"])
        store[name] = (s, v)
        if args.save:
            np.savez_compressed(os.path.join(run, "positions.npz"),
                                shares=s, values=v, equity=e, bars=bars[keep])
        print(f"  scored {name}")

    if not rows:
        return 1

    tab = pd.DataFrame(rows).set_index("agent")
    print("\n" + "=" * 70)
    print("PER-AGENT BEHAVIOUR (2025 only)")
    print("=" * 70)
    def fmt(v):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "-"
        if isinstance(v, (int, np.integer)) or (isinstance(v, float) and v == int(v)
                                                and abs(v) >= 1000):
            return f"{int(v):,}"
        if isinstance(v, float):
            return f"{v:,.4f}".rstrip("0").rstrip(".") if abs(v) < 1000 else f"{v:,.2f}"
        return str(v)

    disp = tab.T.map(fmt)
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(disp.to_string())

    os.makedirs(RESULTS_DIR, exist_ok=True)
    combined = os.path.join(RESULTS_DIR, "agent_summary.csv")
    if os.path.exists(combined):
        old = pd.read_csv(combined).set_index("agent")
        old = old[~old.index.isin(tab.index)]        # rescored agents win
        tab = pd.concat([old, tab])
    tab = tab.sort_index()
    tab.to_csv(combined)
    print(f"\nwrote {combined}  ({len(tab)} agents)")
    print(f"wrote per-agent summary.csv into each run folder")

    if args.overlap and len(store) > 1:
        active_frac = {n: float((store[n][0] > 0).any(axis=1).mean()) for n in store}
        excluded = sorted(n for n, f in active_frac.items() if f < args.min_active)
        names = [n for n in store if n not in excluded]
        if excluded:
            print(f"\nexcluded from overlap (active on <{args.min_active:.0%} "
                  f"of bars): {', '.join(excluded)}")
            print(f"  {len(excluded)}/{len(store)} agents "
                  f"({len(excluded)/len(store):.1%} inactive rate)")
        if len(names) < 2:
            print("fewer than two active agents; nothing to compare")
            return 0
        print("\n" + "=" * 70)
        print("PAIRWISE HOLDINGS OVERLAP")
        print("=" * 70)
        print(f"{'pair':<34}{'jaccard':>9}{'cosine':>9}{'both flat':>11}")
        pairs = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                jac, cos, bf, n = overlap(store[a][0], store[b][0],
                                          store[a][1], store[b][1])
                pairs.append({"a": a, "b": b, "jaccard": jac,
                              "cosine": cos, "both_flat_frac": bf, "bars_used": n})
                print(f"{a[:16]} / {b[:16]:<16}{jac:>9.4f}{cos:>9.4f}{bf:>11.3f}")
        pw = pd.DataFrame(pairs)
        pwpath = os.path.join(RESULTS_DIR, "pairwise_overlap.csv")
        if os.path.exists(pwpath):
            old = pd.read_csv(pwpath)
            fresh = set(zip(pw["a"], pw["b"]))
            old = old[[(a, b) not in fresh for a, b in zip(old["a"], old["b"])]]
            pw = pd.concat([old, pw], ignore_index=True)
        pw = pw.sort_values(["a", "b"]).reset_index(drop=True)
        pw.to_csv(pwpath, index=False)
        print(f"\nwrote {pwpath}  ({len(pw)} pairs)")

        summaries = [
            distribution([p["jaccard"] for p in pairs], "jaccard"),
            distribution([p["cosine"] for p in pairs], "cosine"),
        ]
        summaries = [x for x in summaries if x]
        for x in summaries:
            x["excluded_agents"] = ";".join(excluded) if excluded else ""
            x["n_agents_used"] = len(names)
            x["inactive_rate"] = round(len(excluded) / len(store), 4)
        pd.DataFrame(summaries).to_csv(
            os.path.join(RESULTS_DIR, "overlap_summary.csv"), index=False)

        print("\n" + "=" * 70)
        print("POPULATION CROWDING")
        print("=" * 70)
        cr = crowding(store, names)
        cr["excluded_agents"] = ";".join(excluded) if excluded else ""
        cr["inactive_rate"] = round(len(excluded) / len(store), 4)
        for k, v in cr.items():
            print(f"  {k:<26}{v}")
        pd.DataFrame([cr]).to_csv(os.path.join(RESULTS_DIR, "crowding.csv"), index=False)
        print(f"\nwrote {os.path.join(RESULTS_DIR, 'crowding.csv')}")
        print(f"wrote {os.path.join(RESULTS_DIR, 'overlap_summary.csv')}")
    elif args.overlap:
        print("\n(need at least two agents for pairwise overlap)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
