#!/usr/bin/env python3
"""
flow_mixed.py -- flow imbalance for size-matched mixed groups.

    python3 flow_mixed.py runs/*

Same measure as report.py's flow section (|net| / gross per ticker-bar,
ticker-bars with fewer than 2 traders dropped, weighted by traders - 1),
computed for:
    each model on its own              (~30 agents)
    mixed groups, 10 per model         (30 agents, 200 random draws)
    all LLM agents pooled              (~90 agents, description only)
    random / popular comparison agents

The mixed groups are the same size as a single-model group, so a lower
score there means mixing models reduces one-sided trading, not just that
the group got bigger.

Output: report/csv/flow_mixed.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

try:
    import report
except ImportError:
    sys.exit("error: report.py and score.py must sit next to flow_mixed.py")

LLM = ["claude", "gemini", "gpt"]


class Quiet:
    def __call__(self, *a):
        pass

    def head(self, t):
        pass


def imbalance(D, idx, min_traders):
    if len(idx) < 2:
        return np.nan
    net = D.flow[idx[0]].copy()
    gro = D.aflow[idx[0]].copy()
    cnt = (D.flow[idx[0]] != 0).astype(np.float32)
    for i in idx[1:]:
        net = net + D.flow[i]
        gro = gro + D.aflow[i]
        cnt = cnt + (D.flow[i] != 0).astype(np.float32)
    nz = gro.nonzero()
    if not len(nz[0]):
        return np.nan
    g = np.asarray(gro[nz]).ravel()
    n = np.abs(np.asarray(net[nz]).ravel())
    m = np.asarray(cnt[nz]).ravel()
    keep = m >= min_traders
    if not keep.any():
        return np.nan
    w = m[keep] - 1.0
    return float((w * (n[keep] / g[keep])).sum() / w.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--from", dest="start", default="2025-01-01")
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--per-model", type=int, default=10, dest="per_model")
    ap.add_argument("--min-traders", type=int, default=2, dest="min_traders")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print("loading agents...")
    D = report.Data(args.runs, args.start, Quiet())
    labels = np.array(D.labels)
    rng = np.random.default_rng(args.seed)
    rows = []

    def add(group, val, n, note=""):
        rows.append({"group": group, "agents": n, "imbalance": round(val, 4), "note": note})
        print(f"  {group:<22}{n:>5}   {val:.4f}   {note}")

    print(f"\n  {'group':<22}{'n':>5}   imbalance")
    for g in LLM + ["popular", "random"]:
        idx = np.where(labels == g)[0]
        if len(idx):
            add(g, imbalance(D, idx, args.min_traders), len(idx))

    pools = {g: np.where(labels == g)[0] for g in LLM}
    if min(len(p) for p in pools.values()) >= args.per_model:
        vals = []
        for _ in range(args.draws):
            idx = np.concatenate([rng.choice(pools[g], args.per_model, replace=False)
                                  for g in LLM])
            vals.append(imbalance(D, idx, args.min_traders))
        vals = np.array(vals)
        lo, hi = np.nanpercentile(vals, [2.5, 97.5])
        add(f"mixed {args.per_model}+{args.per_model}+{args.per_model}",
            float(np.nanmean(vals)), 3 * args.per_model,
            f"mean of {args.draws} draws, 95% range {lo:.4f}-{hi:.4f}")
        single = [r["imbalance"] for r in rows if r["group"] in LLM]
        print(f"\n  average single-model group: {np.mean(single):.4f}")
        print(f"  mixed group:                {np.nanmean(vals):.4f}")

    idx = np.where(np.isin(labels, LLM))[0]
    add("all LLM pooled", imbalance(D, idx, args.min_traders), len(idx),
        "bigger group -- description only")

    os.makedirs(report.CSVDIR if hasattr(report, "CSVDIR") else "report/csv", exist_ok=True)
    out = os.path.join(getattr(report, "CSVDIR", "report/csv"), "flow_mixed.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    raise SystemExit(main())
