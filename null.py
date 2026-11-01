#!/usr/bin/env python3
"""
null.py -- the label-shuffled permutation test.

    python3 null.py                      # reads results/pairwise_overlap.csv
    python3 null.py --iters 5000 --metric jaccard

The question this answers
-------------------------
Same-model agents overlap at some rate. So do different-model agents, because
momentum strategies chase the same stocks no matter who wrote them. The test
asks whether the SAME-MODEL rate is higher than you would get by grouping the
same agents at random.

Method: keep every pairwise overlap value exactly as measured, throw away the
model labels, reassign them at random preserving group sizes, and recompute
the mean within-group overlap. Repeat many times. That distribution is what
chance produces. The observed value is compared against it.

Why a permutation test rather than a t-test: the pairs are not independent.
With 30 agents each agent appears in 29 pairs, so 435 pairs carry nowhere near
435 pairs' worth of information. Shuffling labels preserves that dependency
structure and randomises only the thing under test.

Pre-committed gate: observed must exceed the 95th percentile of the null.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

RESULTS_DIR = "results"
DEFAULT_IN = os.path.join(RESULTS_DIR, "pairwise_overlap.csv")


def label_of(name):
    """Model label from the agent's filename prefix."""
    n = name.lower()
    if n.startswith("gemini"):
        return "gemini"
    if n.startswith("claude") or n.startswith("agent"):
        return "claude"
    if n.startswith("gpt") or n.startswith("openai"):
        return "gpt"
    return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=DEFAULT_IN)
    ap.add_argument("--metric", default="jaccard", choices=["jaccard", "cosine"])
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not os.path.exists(args.input):
        sys.exit(f"error: {args.input} not found -- run score.py with --overlap first")

    df = pd.read_csv(args.input)
    df = df.dropna(subset=[args.metric])

    agents = sorted(set(df["a"]) | set(df["b"]))
    labels = {a: label_of(a) for a in agents}
    groups = sorted(set(labels.values()))
    if len(groups) < 2:
        sys.exit("error: need agents from at least two models")

    idx = {a: i for i, a in enumerate(agents)}
    n = len(agents)

    # dense pair matrix, computed once; shuffling only re-slices it
    M = np.full((n, n), np.nan)
    for a, b, v in zip(df["a"], df["b"], df[args.metric]):
        i, j = idx[a], idx[b]
        M[i, j] = M[j, i] = v
    iu = np.triu_indices(n, 1)
    vals = M[iu]
    valid = np.isfinite(vals)

    lab = np.array([labels[a] for a in agents])
    counts = pd.Series(lab).value_counts()

    print(f"metric: {args.metric}")
    print(f"agents: {n}  ({', '.join(f'{k} {v}' for k, v in counts.items())})")
    print(f"pairs:  {valid.sum()}\n")

    def within_mean(assignment):
        same = assignment[iu[0]] == assignment[iu[1]]
        sel = same & valid
        return vals[sel].mean() if sel.any() else np.nan

    def cross_mean(assignment):
        diff = assignment[iu[0]] != assignment[iu[1]]
        sel = diff & valid
        return vals[sel].mean() if sel.any() else np.nan

    obs_within = within_mean(lab)
    obs_cross = cross_mean(lab)

    print("OBSERVED")
    for g in groups:
        m = (lab[iu[0]] == g) & (lab[iu[1]] == g) & valid
        print(f"  within {g:<8} {vals[m].mean():.4f}   (n={m.sum()})")
    print(f"  within (pooled)  {obs_within:.4f}")
    print(f"  cross-model      {obs_cross:.4f}")
    print(f"  difference       {obs_within - obs_cross:+.4f}")

    # ---- permutation
    rng = np.random.default_rng(args.seed)
    perm = lab.copy()
    null_within = np.empty(args.iters)
    null_gap = np.empty(args.iters)
    for k in range(args.iters):
        rng.shuffle(perm)                 # preserves group sizes
        null_within[k] = within_mean(perm)
        null_gap[k] = null_within[k] - cross_mean(perm)

    p_within = (np.sum(null_within >= obs_within) + 1) / (args.iters + 1)
    p_gap = (np.sum(null_gap >= obs_within - obs_cross) + 1) / (args.iters + 1)
    gate = np.percentile(null_within, 95)

    print(f"\nNULL ({args.iters:,} label shuffles, group sizes preserved)")
    print(f"  mean            {null_within.mean():.4f}")
    print(f"  sd              {null_within.std(ddof=1):.4f}")
    print(f"  95th pct (gate) {gate:.4f}")
    print(f"  max             {null_within.max():.4f}")

    z = (obs_within - null_within.mean()) / null_within.std(ddof=1)
    print(f"\nRESULT")
    print(f"  observed within-model  {obs_within:.4f}")
    print(f"  pre-committed gate     {gate:.4f}")
    print(f"  {'PASSES' if obs_within > gate else 'DOES NOT PASS'} the 95th-percentile gate")
    print(f"  p (within-model mean)  {p_within:.4f}")
    print(f"  p (within minus cross) {p_gap:.4f}")
    print(f"  effect size (z)        {z:.2f}")
    if p_within <= 1.5 / (args.iters + 1):
        print(f"  note: p is at the resolution floor for {args.iters:,} iterations;"
              f" report as p < {2/(args.iters+1):.4f}")

    # ---- per-model, each against the rest
    print(f"\nPER MODEL (each model's within-pairs vs its own shuffled null)")
    rows = []
    for g in groups:
        m = (lab[iu[0]] == g) & (lab[iu[1]] == g) & valid
        obs_g = vals[m].mean()
        size = int((lab == g).sum())
        nd = np.empty(args.iters)
        fake = np.zeros(n, dtype=bool)
        for k in range(args.iters):
            pick = rng.choice(n, size=size, replace=False)
            fake[:] = False
            fake[pick] = True
            mm = fake[iu[0]] & fake[iu[1]] & valid
            nd[k] = vals[mm].mean()
        pg = (np.sum(nd >= obs_g) + 1) / (args.iters + 1)
        gate_g = np.percentile(nd, 95)
        print(f"  {g:<8} observed {obs_g:.4f}  null mean {nd.mean():.4f}  "
              f"gate {gate_g:.4f}  p {pg:.4f}  "
              f"{'PASS' if obs_g > gate_g else 'fail'}")
        rows.append({"model": g, "n_agents": size, "observed": round(obs_g, 4),
                     "null_mean": round(float(nd.mean()), 4),
                     "null_sd": round(float(nd.std(ddof=1)), 4),
                     "gate_95pct": round(float(gate_g), 4), "p_value": round(pg, 4),
                     "passes": bool(obs_g > gate_g)})

    out = pd.DataFrame(rows)
    summary = {"model": "POOLED", "n_agents": n, "observed": round(obs_within, 4),
               "null_mean": round(float(null_within.mean()), 4),
               "null_sd": round(float(null_within.std(ddof=1)), 4),
               "gate_95pct": round(float(gate), 4), "p_value": round(p_within, 4),
               "passes": bool(obs_within > gate)}
    out = pd.concat([out, pd.DataFrame([summary])], ignore_index=True)
    out["metric"] = args.metric
    out["iterations"] = args.iters
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"null_test_{args.metric}.csv")
    out.to_csv(path, index=False)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
