#!/usr/bin/env python3
"""
analyze.py -- the three analyses beyond pairwise overlap.

    python3 analyze.py classify runs/*        # strategy family per agent
    python3 analyze.py flow runs/*            # population signed flow + null
    python3 analyze.py timeseries runs/*      # overlap over the year

Imports score.py so position reconstruction and the fill rules are shared,
not duplicated.

1. CLASSIFY -- what kind of strategy is each agent running?
   Combines behaviour (holding period, portfolio size, turnover, flat time)
   with a keyword scan of the agent's own source. Heuristic: it produces a
   label and the evidence for it, and you should spot-check a sample by hand
   before quoting the counts. Source is read from agents/<name>.py.

2. FLOW -- do the agents move together, or do they cancel out?
   Holdings overlap is a stock; a crash is a flow event. A hundred agents
   holding identical portfolios forever move no prices; a hundred agents
   selling in the same bar do. For each model, per ticker per bar, sum signed
   share flow across all its agents. The headline is directional imbalance:
       |net flow| / gross flow
   1.0 means every agent traded the same way; 0 means they perfectly offset.
   Compared against a label-shuffled null, same logic as null.py.

3. TIMESERIES -- is overlap constant, or does it spike?
   Per-bar mean within-model Jaccard across the year, so you can see whether
   agents converge under stress and disperse in calm markets. Speaks directly
   to the flash-crash motivation, which a single annual average cannot.
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

try:
    import score
except ImportError:
    sys.exit("error: score.py must be in the same folder as analyze.py")

RESULTS_DIR = "results"
AGENT_DIR = "agents"


def label_of(name):
    n = name.lower()
    if n.startswith("gemini"):
        return "gemini"
    if n.startswith("claude") or n.startswith("agent"):
        return "claude"
    if n.startswith("gpt") or n.startswith("openai"):
        return "gpt"
    return "unknown"


def load_all(runs, start):
    """Reconstruct positions and trades for every run directory."""
    df = score.load_tape()
    bars, tickers, bi, ti, opens, closes = score.build_matrices(df)
    keep = bars >= start
    out = {}
    for run in runs:
        p = os.path.join(run, "trades.csv")
        if not os.path.exists(p):
            continue
        name = os.path.basename(run.rstrip("/"))
        tr = pd.read_csv(p, dtype={"timestamp": str})
        shares, cash, values, equity, _ = score.reconstruct(
            tr, bars, tickers, bi, ti, opens, closes)
        out[name] = {"shares": shares[keep], "values": values[keep],
                     "trades": tr[tr["timestamp"] >= start], "bi": bi, "ti": ti}
        print(f"  loaded {name}")
    return out, bars[keep], tickers, ti


# --------------------------------------------------------------- 1. classify

KEYWORDS = {
    "momentum": [r"\bmomentum\b", r"\bmom\b", r"roc\b", r"rate_of_change"],
    "mean_reversion": [r"mean[_ ]?revers", r"\bzscore\b", r"z_score",
                       r"bollinger", r"\brsi\b", r"oversold"],
    "trend": [r"\btrend\b", r"\bsma\b", r"\bema\b", r"moving[_ ]average",
              r"crossover", r"\bmacd\b"],
    "volatility": [r"\bvolatilit", r"\batr\b", r"inverse[_ ]vol", r"vol_target"],
    "breakout": [r"breakout", r"\bhigh_?52\b", r"donchian", r"channel"],
    "buy_hold": [r"buy[_ ]and[_ ]hold", r"equal[_ ]weight"],
}


def read_source(name):
    for cand in (os.path.join(AGENT_DIR, name + ".py"),
                 os.path.join(AGENT_DIR, name.replace("run_", "") + ".py")):
        if os.path.exists(cand):
            return open(cand, encoding="utf-8", errors="replace").read().lower()
    return ""


def classify(data, bars):
    rows = []
    for name, d in data.items():
        sh = d["shares"]
        nz = sh > 0
        n_held = nz.sum(axis=1)
        flat = float((n_held == 0).mean())
        hp = score.holding_periods(sh)
        med_hold = float(np.median(hp))
        ever = int(nz.any(axis=0).sum())
        mean_names = float(n_held.mean())
        orders_per_bar = len(d["trades"]) / max(len(bars), 1)

        src = read_source(name)
        hits = {k: sum(bool(re.search(p, src)) for p in pats)
                for k, pats in KEYWORDS.items()}
        top = max(hits, key=hits.get) if any(hits.values()) else None

        # behavioural family, independent of what the source calls itself
        if flat > 0.95:
            behav = "inactive"
        elif med_hold >= 200 and orders_per_bar < 0.05:
            behav = "buy_hold"
        elif med_hold <= 4 and ever > 100:
            behav = "high_turnover_rotator"
        elif mean_names > 40:
            behav = "broad_portfolio"
        elif med_hold <= 26:
            behav = "short_horizon"
        else:
            behav = "medium_horizon"

        rows.append({
            "agent": name, "model": label_of(name),
            "behavioural_family": behav,
            "source_family": top if top and hits[top] else "unlabelled",
            "keyword_hits": ";".join(f"{k}:{v}" for k, v in hits.items() if v),
            "median_hold_bars": round(med_hold, 1),
            "mean_names_held": round(mean_names, 1),
            "tickers_ever_held": ever,
            "orders_per_bar": round(orders_per_bar, 2),
            "flat_frac": round(flat, 3),
            "source_found": bool(src),
        })

    tab = pd.DataFrame(rows).sort_values(["model", "agent"])
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tab.to_csv(os.path.join(RESULTS_DIR, "strategy_families.csv"), index=False)

    print("\n" + "=" * 70)
    print("BEHAVIOURAL FAMILY x MODEL")
    print("=" * 70)
    print(pd.crosstab(tab["behavioural_family"], tab["model"]).to_string())
    print("\n" + "=" * 70)
    print("SOURCE-KEYWORD FAMILY x MODEL")
    print("=" * 70)
    print(pd.crosstab(tab["source_family"], tab["model"]).to_string())
    missing = (~tab["source_found"]).sum()
    if missing:
        print(f"\nnote: source not found for {missing} agents "
              f"(expected at {AGENT_DIR}/<name>.py) -- keyword columns blank")
    print(f"\nwrote {RESULTS_DIR}/strategy_families.csv")
    print("These labels are heuristic. Spot-check a handful before quoting counts.")


# ------------------------------------------------------------------- 2. flow

def flow(data, bars, tickers, ti, iters):
    names = sorted(data)
    labels = np.array([label_of(n) for n in names])
    nb, nt = len(bars), len(tickers)
    bpos = {b: i for i, b in enumerate(bars)}

    # signed share flow per agent per bar per ticker
    F = np.zeros((len(names), nb, nt), dtype=np.float32)
    for ai, n in enumerate(names):
        tr = data[n]["trades"]
        for ts, tk, d, sz in tr.itertuples(index=False):
            if ts in bpos and tk in ti:
                F[ai, bpos[ts], ti[tk]] += sz if d == "BUY" else -sz

    def imbalance(mask):
        """Gross-weighted mean of |net| / gross over active ticker-bars."""
        sub = F[mask]
        net = np.abs(sub.sum(axis=0))
        gross = np.abs(sub).sum(axis=0)
        live = gross > 0
        return float(net[live].sum() / gross[live].sum()) if live.any() else np.nan

    groups = sorted(set(labels))
    print("\n" + "=" * 70)
    print("POPULATION SIGNED FLOW")
    print("=" * 70)
    print("directional imbalance = |net flow| / gross flow")
    print("  1.0 = all agents traded the same way   0.0 = perfectly offsetting\n")

    obs = {}
    for g in groups:
        m = labels == g
        obs[g] = imbalance(m)
        print(f"  {g:<8} {obs[g]:.4f}   ({m.sum()} agents)")
    obs_all = imbalance(np.ones(len(names), dtype=bool))
    print(f"  {'pooled':<8} {obs_all:.4f}")

    rng = np.random.default_rng(0)
    rows = []
    for g in groups:
        size = int((labels == g).sum())
        nd = np.empty(iters)
        for k in range(iters):
            pick = rng.choice(len(names), size=size, replace=False)
            m = np.zeros(len(names), dtype=bool)
            m[pick] = True
            nd[k] = imbalance(m)
        p = (np.sum(nd >= obs[g]) + 1) / (iters + 1)
        gate = np.percentile(nd, 95)
        print(f"\n  {g}: null mean {nd.mean():.4f}  gate {gate:.4f}  "
              f"p {p:.4f}  {'PASS' if obs[g] > gate else 'fail'}")
        rows.append({"model": g, "n_agents": size,
                     "observed_imbalance": round(obs[g], 4),
                     "null_mean": round(float(nd.mean()), 4),
                     "gate_95pct": round(float(gate), 4),
                     "p_value": round(p, 4), "passes": bool(obs[g] > gate)})

    os.makedirs(RESULTS_DIR, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "signed_flow.csv"), index=False)
    print(f"\nwrote {RESULTS_DIR}/signed_flow.csv")
    print("Note: fewer shuffles than the overlap null -- this one is expensive.")


# ------------------------------------------------------------- 3. timeseries

def timeseries(data, bars, min_active):
    names = [n for n in sorted(data)
             if (data[n]["shares"] > 0).any(axis=1).mean() >= min_active]
    labels = np.array([label_of(n) for n in names])
    H = np.stack([data[n]["shares"] > 0 for n in names])      # agents x bars x tickers
    na, nb, _ = H.shape
    groups = sorted(set(labels))

    series = {g: np.full(nb, np.nan) for g in groups}
    series["cross"] = np.full(nb, np.nan)

    for b in range(nb):
        X = H[:, b, :].astype(np.float32)
        inter = X @ X.T
        sizes = X.sum(axis=1)
        union = sizes[:, None] + sizes[None, :] - inter
        with np.errstate(invalid="ignore", divide="ignore"):
            J = np.where(union > 0, inter / union, np.nan)
        iu = np.triu_indices(na, 1)
        same = labels[iu[0]] == labels[iu[1]]
        v = J[iu]
        for g in groups:
            m = same & (labels[iu[0]] == g) & np.isfinite(v)
            if m.any():
                series[g][b] = v[m].mean()
        m = (~same) & np.isfinite(v)
        if m.any():
            series["cross"][b] = v[m].mean()

    out = pd.DataFrame({"timestamp": bars, **series})
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out.to_csv(os.path.join(RESULTS_DIR, "overlap_timeseries.csv"), index=False)

    print("\n" + "=" * 70)
    print("OVERLAP OVER TIME")
    print("=" * 70)
    for col in list(groups) + ["cross"]:
        v = out[col].to_numpy()
        v = v[np.isfinite(v)]
        if not len(v):
            continue
        rng_ratio = (v.max() - v.min()) / v.mean() if v.mean() > 0 else float("nan")
        print(f"  {col:<8} mean {v.mean():.4f}  sd {v.std():.4f}  "
              f"min {v.min():.4f}  max {v.max():.4f}  range/mean {rng_ratio:.2f}")

    # monthly means, so the shape is visible without plotting
    out["month"] = pd.to_datetime(out["timestamp"]).dt.to_period("M")
    monthly = out.groupby("month")[list(groups) + ["cross"]].mean()
    print("\nMONTHLY MEAN")
    print(monthly.round(4).to_string())

    for g in groups:
        v = out[g].to_numpy()
        v = v[np.isfinite(v)]
        if len(v) > 20:
            hi = np.percentile(v, 95)
            print(f"\n  {g}: top 5% of bars exceed {hi:.4f} "
                  f"({(v > hi).sum()} bars) vs mean {v.mean():.4f}")
    print(f"\nwrote {RESULTS_DIR}/overlap_timeseries.csv")


# ------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["classify", "flow", "timeseries"])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--from", dest="start", default="2025-01-01")
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--min-active", type=float, default=0.05, dest="min_active")
    args = ap.parse_args()

    print("loading tape and reconstructing positions...")
    data, bars, tickers, ti = load_all(args.runs, args.start)
    if not data:
        sys.exit("no runs with trades.csv found")
    print(f"{len(data)} agents, {len(bars):,} scored bars")

    if args.mode == "classify":
        classify(data, bars)
    elif args.mode == "flow":
        flow(data, bars, tickers, ti, args.iters)
    else:
        timeseries(data, bars, args.min_active)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
