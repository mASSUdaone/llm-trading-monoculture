#!/usr/bin/env python3
"""
checks.py -- the remaining checks before writing, in one run.

    python3 checks.py

Reads results already on disk (no re-scoring), so it takes seconds.
Needs: report/csv/agent_summary.csv (or results/agent_summary.csv), report/csv/pairs_all.csv,
       report/csv/flow_imbalance.csv, report/csv/flow_by_regime.csv

1. Buy-and-hold agents per model, and whether they drive the overlap
2. Fill rate, with "no orders" shown as undefined instead of 0
3. Convergence ratio, LLM-only (the corrected version)
4. Flow imbalance overall and by market regime
"""
import os
import sys

import numpy as np
import pandas as pd

LLM = ["claude", "gemini", "gpt"]


def label(name):
    n = name.lower()
    for p, g in (("random", "random"), ("baseline", "random"), ("popular", "popular"),
                 ("weighted", "popular"), ("gemini", "gemini"), ("gpt", "gpt"),
                 ("openai", "gpt"), ("claude", "claude"), ("agent", "claude")):
        if n.startswith(p):
            return g
    return "unknown"


def need(path):
    if not os.path.exists(path):
        sys.exit(f"missing {path} -- run report.py all runs/* first")
    return pd.read_csv(path)


def head(t):
    print("\n" + "=" * 70 + "\n" + t + "\n" + "=" * 70)


S = need("report/csv/agent_summary.csv" if os.path.exists("report/csv/agent_summary.csv")
         else "results/agent_summary.csv")
S["model"] = S["agent"].map(label)
P = need("report/csv/pairs_all.csv")
P = P[~P["excluded"].astype(bool)] if "excluded" in P.columns else P

# ------------------------------------------------------------------ 1
head("1. BUY-AND-HOLD AGENTS  (median holding period over 1,000 bars)")
act = S[S["mean_names_held"] > 0].copy()
act["long_hold"] = act["median_hold_bars"] > 1000
t = act.groupby("model").agg(agents=("agent", "size"),
                             long_hold=("long_hold", "sum"),
                             median_hold_bars=("median_hold_bars", "median"))
print(t.reindex([m for m in LLM + ["popular", "random"] if m in t.index]).to_string())

longs = set(act.loc[act["long_hold"], "agent"])
print(f"\nlong-hold agents: {', '.join(sorted(longs)) if longs else 'none'}")

print("\nwithin-model Jaccard, all agents vs long-hold agents removed:")
print(f"  {'model':<8}{'all':>9}{'without':>10}{'change':>9}")
for m in LLM:
    w = P[(P.model_a == m) & (P.model_b == m)]
    if not len(w):
        continue
    w2 = w[~w.a.isin(longs) & ~w.b.isin(longs)]
    a, b = w.jaccard.mean(), w2.jaccard.mean() if len(w2) else np.nan
    print(f"  {m:<8}{a:>9.4f}{b:>10.4f}{b - a:>+9.4f}")
print("\n  if 'without' stays close to 'all', buy-and-hold agents are not what")
print("  drives a model's overlap and the confound can be ruled out.")

# ------------------------------------------------------------------ 2
head("2. FILL RATE  (agents with no orders shown as undefined, not 0)")
fr = S.copy()
fr.loc[fr["orders"] == 0, "fill_rate"] = np.nan
g = fr.groupby("model")["fill_rate"].agg(["count", "median", "min"])
g["no_orders"] = fr[fr["orders"] == 0].groupby("model").size()
g["no_orders"] = g["no_orders"].fillna(0).astype(int)
print(g.reindex([m for m in LLM + ["popular", "random"] if m in g.index]).round(3).to_string())
low = fr[fr["fill_rate"] < 0.9].sort_values("fill_rate")
if len(low):
    print("\nagents below 0.9:")
    print(low[["agent", "orders", "fill_rate", "zero_fills"]].to_string(index=False))
over = fr[fr["fill_rate"] > 1]
if len(over):
    print(f"\nabove 1.0 (the 2024/2025 boundary leak): "
          f"{', '.join(f'{a} {v:.3f}' for a, v in zip(over.agent, over.fill_rate))}")

# ------------------------------------------------------------------ 3
head("3. CONVERGENCE RATIO  (within / cross, LLM pairs only)")
print(f"  {'model':<8}{'within':>9}{'cross':>9}{'ratio':>8}   {'old report.py':>13}")
for m in LLM:
    w = P[(P.model_a == m) & (P.model_b == m)].jaccard.mean()
    c = P[((P.model_a == m) ^ (P.model_b == m))
          & P.model_a.isin(LLM) & P.model_b.isin(LLM)].jaccard.mean()
    old = P[((P.model_a == m) ^ (P.model_b == m))].jaccard.mean()
    print(f"  {m:<8}{w:>9.4f}{c:>9.4f}{w / c:>8.2f}   {w / old:>13.2f}")
print("\n  report the LLM-only ratio; the old one counts pairs with synthetic")
print("  agents as 'cross', which inflates it.")

# ------------------------------------------------------------------ 4
head("4. FLOW IMBALANCE")
F = need("report/csv/flow_imbalance.csv")
cols = [c for c in ["group", "observed_imbalance", "null_mean", "gate_95pct",
                    "p_value", "z_score", "passes"] if c in F.columns]
print(F[cols].to_string(index=False))
print("\nby market regime:")
R = need("report/csv/flow_by_regime.csv").set_index("regime")
R = R.reindex([r for r in ["calm", "normal", "stressed"] if r in R.index])
print(R.round(4).to_string())
if {"calm", "stressed"} <= set(R.index):
    print("\n  stressed minus calm:")
    for c in R.columns:
        d = R.loc["stressed", c] - R.loc["calm", c]
        print(f"    {c:<8} {d:+.4f}  ({d / R.loc['calm', c]:+.1%})")
    print("\n  compare each model's change with random's: a model is only")
    print("  more one-sided under stress if it moves clearly more than random.")
print()
