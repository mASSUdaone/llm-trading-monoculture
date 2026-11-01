#!/usr/bin/env python3
"""
baseline.py -- synthetic agents that pick stocks at random.

    python3 baseline.py --n 30                    # uniform random picks
    python3 baseline.py --n 30 --mode weighted --match runs/*
    python3 baseline.py --n 30 --names 6 --rotate 2

Why this exists
---------------
Every group in the pool is an LLM. So the label-shuffle null can only answer
"does Claude cluster more than a random regrouping of LLM agents" -- it cannot
say whether 0.19 is high in absolute terms, because there is nothing in the
pool that fails to cluster.

These agents supply that floor. They hold the same number of names and rotate
at the same speed as the real ones, but choose which names by coin flip. Any
overlap they show is what arises from chance alone at that portfolio size.

Two modes, and the difference between them matters:

  uniform    picks from all tickers with equal probability. The pure floor.

  weighted   picks with probability proportional to how often the real agents
             actually held each ticker. This separates two explanations that
             uniform cannot: agents may overlap because they coordinate on the
             same picks, or merely because they all favour the same liquid
             megacaps. If LLM overlap sits far above WEIGHTED random, the
             overlap is coordination. If it sits near it, the overlap is
             shared taste in stocks -- still monoculture, different mechanism.

Output goes to runs/random_NNN/trades.csv, so score.py and report.py pick
these up exactly like generated agents. They are labelled `random`.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

try:
    import score
except ImportError:
    sys.exit("error: score.py must sit next to baseline.py")

CASH = 100_000.0


def params_from_summary(path="report/csv/agent_summary.csv"):
    """Preferred source: the reconstructed portfolio sizes, not order counts.

    Counting distinct tickers bought per rebalance is NOT portfolio size -- an
    agent holding 12 names and swapping 2 each time looks like 2. That
    understates the baseline's portfolio, and smaller portfolios overlap less
    by chance, so the floor comes out too low and the gap to the real agents
    looks bigger than it is. mean_names_held is the honest number.
    """
    if not os.path.exists(path):
        return None, None
    t = pd.read_csv(path)
    if "mean_names_held" not in t.columns:
        return None, None
    t = t[t["mean_names_held"] > 0]
    if not len(t):
        return None, None
    hold = t["median_hold_bars"] if "median_hold_bars" in t.columns else None
    return (t["mean_names_held"].to_numpy(dtype=float),
            hold.clip(lower=1).to_numpy(dtype=float) if hold is not None else None)


def observed_params(runs, bars_set):
    """Fallback: infer from trades.csv when no summary exists."""
    sizes, rots = [], []
    for run in runs:
        p = os.path.join(run, "trades.csv")
        if not os.path.exists(p):
            continue
        tr = pd.read_csv(p, dtype={"timestamp": str})
        tr = tr[tr["timestamp"].isin(bars_set)]
        if not len(tr):
            continue
        buys = tr[tr["direction"] == "BUY"]
        if not len(buys):
            continue
        per_bar = buys.groupby("timestamp")["ticker"].nunique()
        sizes.append(float(per_bar.mean()))
        active = buys["timestamp"].nunique()
        rots.append(len(bars_set) / active if active else np.nan)
    return np.array(sizes), np.array(rots)


def ticker_popularity(runs, tickers, bars_set):
    """How often each ticker was bought by the real agents."""
    c = pd.Series(0, index=tickers, dtype=float)
    for run in runs:
        p = os.path.join(run, "trades.csv")
        if not os.path.exists(p):
            continue
        tr = pd.read_csv(p, dtype={"timestamp": str})
        tr = tr[(tr["direction"] == "BUY") & tr["timestamp"].isin(bars_set)]
        v = tr["ticker"].value_counts()
        c = c.add(v.reindex(c.index).fillna(0), fill_value=0)
    if c.sum() <= 0:
        return None
    return (c / c.sum()).to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="how many agents")
    ap.add_argument("--mode", choices=["uniform", "weighted"], default="uniform")
    ap.add_argument("--match", nargs="*", default=None,
                    help="run dirs to copy portfolio size / rotation speed from")
    ap.add_argument("--names", type=int, default=None,
                    help="fixed portfolio size (overrides --match)")
    ap.add_argument("--rotate", type=int, default=None,
                    help="fixed bars between rotations (overrides --match)")
    ap.add_argument("--prefix", default="random")
    ap.add_argument("--summary", default="report/csv/agent_summary.csv",
                    help="where to read real portfolio sizes from")
    ap.add_argument("--from", dest="start", default="2025-01-01")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("loading tape...")
    df = score.load_tape()
    bars = np.array(sorted(df["timestamp"].unique()))
    tickers = np.array(sorted(df["ticker"].unique()))
    scored = bars[bars >= args.start]
    bars_set = set(scored)
    print(f"{len(bars):,} bars ({len(scored):,} scored), {len(tickers)} tickers")

    # opens, so an order placed on bar t is affordable at t+1
    px = df.pivot_table(index="timestamp", columns="ticker", values="open",
                        aggfunc="first").reindex(index=bars, columns=tickers)
    px = px.ffill().to_numpy(dtype=float)

    # --- parameters
    if args.names and args.rotate:
        sizes = np.array([float(args.names)])
        rots = np.array([float(args.rotate)])
        print(f"fixed: {args.names} names, rotate every {args.rotate} bars")
    elif args.match:
        sizes, rots = params_from_summary(args.summary)
        if sizes is not None:
            print(f"matched to {len(sizes)} agents in {args.summary}")
        else:
            sizes, rots = observed_params(args.match, bars_set)
            print(f"no {args.summary}; inferred from trades.csv instead "
                  f"(run report.py first for a better match)")
        if sizes is None or not len(sizes):
            sys.exit("error: could not determine portfolio parameters")
        if rots is None or not len(rots):
            rots = np.array([5.0])
        print(f"  names held {sizes.mean():.1f} "
              f"(median {np.median(sizes):.1f}, range {sizes.min():.0f}-{sizes.max():.0f})")
        print(f"  rotate every {np.median(rots):.1f} bars "
              f"(range {rots.min():.0f}-{rots.max():.0f})")
    else:
        sizes, rots = np.array([6.0]), np.array([2.0])
        print("defaults: 6 names, rotate every 2 bars "
              "(pass --match to copy the real agents instead)")

    probs = None
    if args.mode == "weighted":
        if not args.match:
            sys.exit("error: --mode weighted needs --match to read real holdings")
        probs = ticker_popularity(args.match, tickers, bars_set)
        if probs is None:
            sys.exit("error: no buys found in the matched runs")
        top = np.argsort(-probs)[:5]
        print(f"weighted by real popularity; most-bought: "
              f"{', '.join(tickers[t] for t in top)}")

    rng = np.random.default_rng(args.seed)
    b0 = int(np.searchsorted(bars, scored[0]))
    made = 0

    for i in range(1, args.n + 1):
        name = f"{args.prefix}_{i:03d}"
        outdir = os.path.join("runs", name)
        if os.path.exists(os.path.join(outdir, "trades.csv")):
            print(f"  {name} exists, skipping")
            continue
        os.makedirs(outdir, exist_ok=True)

        k = max(1, int(round(rng.choice(sizes))))
        step = max(1, int(round(rng.choice(rots))))

        cash, held, rows = CASH, {}, []
        for b in range(b0, len(bars) - 1):
            if (b - b0) % step:
                continue
            ts = bars[b]
            # sell everything held, then buy a fresh random basket
            for tk, q in held.items():
                rows.append((ts, tk, "SELL", int(q)))
                p = px[b + 1, np.where(tickers == tk)[0][0]]
                if np.isfinite(p):
                    cash += q * p
            held = {}

            pick = rng.choice(len(tickers), size=k, replace=False, p=probs)
            budget = cash / k
            for j in pick:
                p = px[b + 1, j]
                if not np.isfinite(p) or p <= 0:
                    continue
                q = int(budget // p)
                if q > 0:
                    rows.append((ts, tickers[j], "BUY", q))
                    held[tickers[j]] = q
                    cash -= q * p

        pd.DataFrame(rows, columns=["timestamp", "ticker", "direction", "size"]) \
            .to_csv(os.path.join(outdir, "trades.csv"), index=False)
        print(f"  {name}  {k} names, rotate {step}  ->  {len(rows):,} orders")
        made += 1

    print(f"\nwrote {made} agents to runs/{args.prefix}_*  (labelled `{args.prefix}`)")
    print("score them alongside the real ones:")
    print(f"  python3 report.py all runs/*")
    print("\nTheir within-group overlap is the zero-coordination floor: what")
    print("agents holding this many names at this rotation speed score when")
    print("nothing links their picks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
