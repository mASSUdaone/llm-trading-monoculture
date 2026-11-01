#!/usr/bin/env python3
"""
power.py -- how big an effect can this design actually detect?

    python3 power.py                                  # default sweep
    python3 power.py --sizes 20 30 50 100 --reps 200
    python3 power.py --target 0.194 --cross 0.129     # match your real numbers

The problem this solves
-----------------------
A "fail" from the permutation test has two possible meanings, and the test
cannot tell them apart:

    (a) this model's agents genuinely do not cluster
    (b) they cluster a little, and the sample is too small to see it

Reporting (b) as (a) is the standard way an underpowered study misleads. So
before trusting any null result -- Gemini's, most obviously -- you have to
know what size of effect the design can see at the sample size you ran.

Method
------
Build agents whose overlap is set by construction, so the true answer is known
in advance. Each group has a "core" of favoured tickers. An agent draws a
fraction p of its holdings from its group's core and the rest from the whole
universe. Raising p raises within-group overlap without touching anything
else. Then run the real permutation test on these fabricated agents and record
whether it fired.

Repeat many times. The fraction of runs where the test fired is its POWER at
that effect size and sample size.

    power 0.80 means: if an effect that size is real, this design catches it
    four times in five. power 0.30 means it misses it seven times in ten, and
    a "fail" tells you almost nothing.

Reading the output
------------------
Find the row whose `within` column matches your observed overlap. The `power`
column on that row is how often a real effect of that size would be caught at
each sample size. If power at n=30 is low but n=100 is high, the extra agents
are worth generating; if n=30 already reads 0.95, they are not.
"""

import argparse
import sys

import numpy as np
import pandas as pd


def build(rng, n_per, n_groups, k, core, frac, universe, reps_bars):
    """Fabricate agents with a known within-group tendency.

    Returns a pairwise Jaccard matrix averaged over `reps_bars` independent
    redraws, mimicking the way the real statistic averages over 3,250 bars --
    which is what makes the observed value stable enough to test.
    """
    n = n_per * n_groups
    labels = np.repeat(np.arange(n_groups), n_per)
    cores = [rng.choice(universe, size=core, replace=False) for _ in range(n_groups)]

    acc = np.zeros((n, n))
    for _ in range(reps_bars):
        sets = []
        for a in range(n):
            n_core = rng.binomial(k, frac)
            pick = list(rng.choice(cores[labels[a]], size=min(n_core, core),
                                   replace=False))
            rest = k - len(pick)
            if rest > 0:
                pool = np.setdiff1d(universe, pick, assume_unique=False)
                pick += list(rng.choice(pool, size=rest, replace=False))
            sets.append(set(pick))
        M = np.zeros((n, n))
        for a in range(n):
            for b in range(a + 1, n):
                inter = len(sets[a] & sets[b])
                union = len(sets[a] | sets[b])
                M[a, b] = M[b, a] = inter / union if union else 0.0
        acc += M
    return acc / reps_bars, labels


def test(J, labels, iters, rng):
    """The real permutation test: does within-group beat the 95th percentile?"""
    n = len(labels)
    iu = np.triu_indices(n, 1)
    vals = J[iu]
    same = labels[iu[0]] == labels[iu[1]]
    obs = vals[same].mean()
    cross = vals[~same].mean()

    perm = labels.copy()
    nd = np.empty(iters)
    for i in range(iters):
        rng.shuffle(perm)
        nd[i] = vals[perm[iu[0]] == perm[iu[1]]].mean()
    gate = np.percentile(nd, 95)
    return obs, cross, bool(obs > gate)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[20, 30, 50, 100],
                    help="agents per group to test")
    ap.add_argument("--fracs", type=float, nargs="+",
                    default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                    help="share of each holding drawn from the group core")
    ap.add_argument("--groups", type=int, default=3)
    ap.add_argument("--k", type=int, default=6, help="names held per agent")
    ap.add_argument("--core", type=int, default=40, help="size of a group's core")
    ap.add_argument("--universe", type=int, default=486)
    ap.add_argument("--reps", type=int, default=60, help="replicate runs per cell")
    ap.add_argument("--iters", type=int, default=300, help="shuffles per test")
    ap.add_argument("--bars", type=int, default=40,
                    help="redraws averaged into each pair matrix")
    ap.add_argument("--target", type=float, default=None,
                    help="your observed within-model overlap, for the guide line")
    ap.add_argument("--cross", type=float, default=None,
                    help="your observed cross-model overlap")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    universe = np.arange(args.universe)
    rng = np.random.default_rng(args.seed)

    print(f"{args.groups} groups | {args.k} names per agent | "
          f"universe {args.universe} | core {args.core}")
    print(f"{args.reps} replicates per cell, {args.iters} shuffles per test, "
          f"{args.bars} redraws averaged\n")
    if args.target:
        print(f"your observed within-model overlap: {args.target}"
              + (f", cross-model {args.cross}" if args.cross else ""))
        print("find the row whose `within` is closest to it\n")

    rows = []
    for frac in args.fracs:
        for n_per in args.sizes:
            hits, ws, cs = 0, [], []
            for _ in range(args.reps):
                J, labels = build(rng, n_per, args.groups, args.k, args.core,
                                  frac, universe, args.bars)
                w, c, fired = test(J, labels, args.iters, rng)
                hits += fired
                ws.append(w); cs.append(c)
            power = hits / args.reps
            rows.append({"core_frac": frac, "n_per_group": n_per,
                         "within": round(float(np.mean(ws)), 4),
                         "cross": round(float(np.mean(cs)), 4),
                         "effect": round(float(np.mean(ws) - np.mean(cs)), 4),
                         "power": round(power, 3)})
            bar = "#" * int(power * 30)
            print(f"  core_frac {frac:.2f}  n={n_per:<4} within {np.mean(ws):.4f}  "
                  f"cross {np.mean(cs):.4f}  effect {np.mean(ws)-np.mean(cs):+.4f}  "
                  f"power {power:.2f} {bar}")
        print()

    t = pd.DataFrame(rows)
    t.to_csv("power_analysis.csv", index=False)
    print("wrote power_analysis.csv\n")

    print("=" * 68)
    print("SMALLEST DETECTABLE EFFECT (power >= 0.80)")
    print("=" * 68)
    for n_per in args.sizes:
        s = t[(t.n_per_group == n_per) & (t.power >= 0.80)]
        if len(s):
            r = s.nsmallest(1, "effect").iloc[0]
            print(f"  n={n_per:<4} detects an effect of {r.effect:+.4f} "
                  f"(within {r.within:.4f} vs cross {r.cross:.4f})")
        else:
            print(f"  n={n_per:<4} no tested effect reached power 0.80")

    if args.target and args.cross:
        eff = args.target - args.cross
        print(f"\n  your observed effect is {eff:+.4f}")
        near = t.iloc[(t.effect - eff).abs().argsort()[:len(args.sizes)]]
        for _, r in near.sort_values("n_per_group").iterrows():
            verdict = ("reliable" if r.power >= 0.8 else
                       "marginal" if r.power >= 0.5 else "underpowered")
            print(f"    at n={int(r.n_per_group):<4} power {r.power:.2f}  ({verdict})")
        print("\n  power below 0.80 means a 'fail' for that model is not")
        print("  evidence of no effect -- it is evidence of not enough agents.")

    print("\nNote: this models holdings as independent redraws. Real agents")
    print("hold positions across consecutive bars, which makes the observed")
    print("statistic somewhat more stable than simulated here. Treat these")
    print("numbers as a conservative lower bound on the true power.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
