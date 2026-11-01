#!/usr/bin/env python3
"""
floor.py -- the diversification floor: how much crowding survives provider mixing.

    python3 floor.py
    python3 floor.py --metric cosine
    python3 floor.py --csv report/csv/pairs_all.csv --out report/csv
    python3 floor.py --boot 2000            # agent-level confidence intervals

The question this answers
-------------------------
Suppose you run a book of LLM trading agents and you are worried about
crowding. The obvious defence is "don't buy all your agents from one vendor" --
mix Claude, Gemini and GPT and the correlation should wash out.

This script asks how much of it actually washes out.

Two numbers already exist in the report:

    within-provider overlap   two agents from the SAME model          0.189
    cross-provider overlap    two agents from DIFFERENT models        0.129

Naively that looks like mixing removes about a third. But both numbers contain
overlap that any two portfolios of this size would share by pure chance -- two
agents each holding ~9 names out of 486 will collide sometimes with no
mechanism at all. That chance component is the RANDOM FLOOR, measured directly
by the synthetic agents from baseline.py:

    random-random overlap                                            0.012

Everything above the floor is the part that needs explaining. So:

    excess_within = within - floor          crowding when you buy one vendor
    excess_cross  = cross  - floor          crowding you keep after mixing

    removed  = (excess_within - excess_cross) / excess_within
    FLOOR    =  excess_cross / excess_within          <- the headline

FLOOR is the share of excess crowding that provider diversification CANNOT
remove. It is a floor in the portfolio sense: the irreducible piece, the
analogue of systematic risk that survives however many names you add.

Why this is worth a number
--------------------------
Ross et al. (2026) show that correlation across LLM agents has a component
that does not shrink as you add agents: Var(mean) = (1-rho)sigma^2/N + rho
sigma^2, and the second term is flat in N. They frame it at the population
level. This is the same idea aimed at the decision an allocator actually
faces -- not "how many agents" but "how many vendors" -- and it is measured
rather than assumed.

If FLOOR were near 0, monoculture would be a procurement problem with a
procurement fix. It is not near 0.

What it does NOT say
--------------------
It is a statement about this universe, this contract, this bar length, and
these three model versions. It is not a claim about LLMs in general, and the
cross-provider number is bounded below by whatever the three models happen to
share -- megacap liquidity, S&P membership, momentum being the obvious
strategy to write. The `popular` floor row in the robustness table is there
to show how much of the answer that shared taste accounts for.

Reading the output
------------------
  removed_pct    how much of the excess goes away when you mix vendors
  floor_pct      how much stays -- the headline
  cross_x_floor  residual overlap as a multiple of uncoordinated agents

Per-model rows read the same way, but "cross" means pairs where exactly one
side is that model. A model with a high floor_pct is one whose crowding is
mostly shared with the other providers rather than particular to it.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

LLM = ("claude", "gemini", "gpt")
BASE = ("random", "popular")


def load(path, metric):
    if not os.path.exists(path):
        sys.exit(f"not found: {path}\nrun `python3 report.py overlap` first")
    p = pd.read_csv(path)
    need = {"a", "b", "model_a", "model_b", metric}
    miss = need - set(p.columns)
    if miss:
        sys.exit(f"{path} is missing columns: {sorted(miss)}")
    if "excluded" in p.columns:
        n0 = len(p)
        p = p[~p.excluded.astype(bool)]
        if len(p) < n0:
            print(f"dropped {n0 - len(p)} pairs involving inactive agents")
    return p.dropna(subset=[metric])


def matrix(p, metric):
    """Pair list -> dense symmetric agent x agent matrix, NaN where no pair."""
    names = sorted(set(p.a) | set(p.b))
    ix = {n: i for i, n in enumerate(names)}
    lab = {}
    for c, m in (("a", "model_a"), ("b", "model_b")):
        for nm, ml in zip(p[c], p[m]):
            lab[nm] = ml
    n = len(names)
    M = np.full((n, n), np.nan)
    i = p.a.map(ix).to_numpy()
    j = p.b.map(ix).to_numpy()
    v = p[metric].to_numpy(float)
    M[i, j] = v
    M[j, i] = v
    labels = np.array([lab[nm] for nm in names])
    return M, labels, np.array(names)


def within(M, labels, g):
    """Mean over distinct same-group pairs."""
    idx = np.where(labels == g)[0]
    if len(idx) < 2:
        return np.nan, 0
    sub = M[np.ix_(idx, idx)]
    iu = np.triu_indices(len(idx), 1)
    v = sub[iu]
    v = v[np.isfinite(v)]
    return (float(v.mean()), len(v)) if len(v) else (np.nan, 0)


def between(M, labels, ga, gb):
    """Mean over pairs with one agent in ga and the other in gb (ga != gb)."""
    ia = np.where(np.isin(labels, ga))[0]
    ib = np.where(np.isin(labels, gb))[0]
    if not len(ia) or not len(ib):
        return np.nan, 0
    sub = M[np.ix_(ia, ib)]
    v = sub[np.isfinite(sub)]
    return (float(v.mean()), len(v)) if len(v) else (np.nan, 0)


def pooled_within(M, labels, groups, weight):
    """Same-model overlap pooled across the LLM groups."""
    vals, ns = [], []
    for g in groups:
        mu, n = within(M, labels, g)
        if np.isfinite(mu):
            vals.append(mu)
            ns.append(n)
    if not vals:
        return np.nan, 0
    vals, ns = np.array(vals), np.array(ns, float)
    if weight == "equal":
        return float(vals.mean()), int(ns.sum())
    return float((vals * ns).sum() / ns.sum()), int(ns.sum())


def pooled_cross(M, labels, groups, weight):
    """Different-model overlap pooled across the LLM group pairs."""
    vals, ns = [], []
    for k, ga in enumerate(groups):
        for gb in groups[k + 1:]:
            mu, n = between(M, labels, [ga], [gb])
            if np.isfinite(mu):
                vals.append(mu)
                ns.append(n)
    if not vals:
        return np.nan, 0
    vals, ns = np.array(vals), np.array(ns, float)
    if weight == "equal":
        return float(vals.mean()), int(ns.sum())
    return float((vals * ns).sum() / ns.sum()), int(ns.sum())


def split(w, c, floor):
    """The decomposition itself. Returns removed / residual shares."""
    ew = w - floor
    ec = c - floor
    if not np.isfinite(ew) or ew <= 0:
        return dict(excess_within=ew, excess_cross=ec,
                    removed_pct=np.nan, floor_pct=np.nan)
    return dict(excess_within=ew, excess_cross=ec,
                removed_pct=(ew - ec) / ew, floor_pct=ec / ew)


def stat(M, labels, groups, floor_groups, weight):
    """One full evaluation. Split out so the bootstrap can reuse it."""
    w, nw = pooled_within(M, labels, groups, weight)
    c, nc = pooled_cross(M, labels, groups, weight)
    if floor_groups is None:
        f, nf = 0.0, 0
    else:
        f, nf = within(M, labels, floor_groups[0]) if len(floor_groups) == 1 \
            else pooled_within(M, labels, floor_groups, weight)
    return w, c, f, nw, nc, nf


def bootstrap(M, labels, groups, floor_groups, weight, reps, seed):
    """
    Resample AGENTS, not pairs.

    Pairs are not independent -- each agent sits in n-1 of them -- so a
    bootstrap over the pair list would produce intervals several times too
    narrow. Resampling the roster and rebuilding the pair set from it keeps
    the dependency structure intact. Slots drawn twice give a self-pair, which
    is NaN in the matrix and drops out.
    """
    rng = np.random.default_rng(seed)
    groups_all = sorted(set(labels))
    idx_by = {g: np.where(labels == g)[0] for g in groups_all}
    out = {"removed_pct": [], "floor_pct": [], "cross_x_floor": []}
    for _ in range(reps):
        roster = np.concatenate([rng.choice(idx_by[g], len(idx_by[g]), True)
                                 for g in groups_all])
        blab = labels[roster]
        B = M[np.ix_(roster, roster)]
        np.fill_diagonal(B, np.nan)
        w, c, f, *_ = stat(B, blab, groups, floor_groups, weight)
        d = split(w, c, f)
        out["removed_pct"].append(d["removed_pct"])
        out["floor_pct"].append(d["floor_pct"])
        out["cross_x_floor"].append(c / f if f > 0 else np.nan)
    return {k: np.array(v, float) for k, v in out.items()}


def ci(a, lo=2.5, hi=97.5):
    a = a[np.isfinite(a)]
    if len(a) < 20:
        return np.nan, np.nan
    return float(np.percentile(a, lo)), float(np.percentile(a, hi))


def pct(x):
    return "n/a" if not np.isfinite(x) else f"{100 * x:5.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="report/csv/pairs_all.csv")
    ap.add_argument("--out", default="report/csv")
    ap.add_argument("--metric", default="jaccard",
                    choices=["jaccard", "cosine"])
    ap.add_argument("--models", nargs="*", default=list(LLM),
                    help="the provider groups being mixed")
    ap.add_argument("--floor", default="random",
                    choices=["random", "popular", "zero"],
                    help="which baseline counts as uncoordinated")
    ap.add_argument("--weight", default="pairs", choices=["pairs", "equal"],
                    help="pool relations by pair count or give each equal say")
    ap.add_argument("--boot", type=int, default=2000,
                    help="agent-level bootstrap reps; 0 to skip")
    ap.add_argument("--seed", type=int, default=20260919)
    args = ap.parse_args()

    p = load(args.csv, args.metric)
    M, labels, names = matrix(p, args.metric)
    have = sorted(set(labels))
    groups = [g for g in args.models if g in have]
    missing = [g for g in args.models if g not in have]
    if len(groups) < 2:
        sys.exit(f"need at least 2 provider groups; found {have}")
    if missing:
        print(f"note: not in the pool, skipped: {', '.join(missing)}")

    fg = None if args.floor == "zero" else [args.floor]
    if fg and fg[0] not in have:
        sys.exit(f"floor group '{args.floor}' is not in the pool ({have}).\n"
                 f"generate it with baseline.py and rescore, or use --floor zero")

    print(f"\nmetric        {args.metric}")
    print(f"agents        {len(names)}  ({', '.join(f'{g}:{(labels == g).sum()}' for g in have)})")
    print(f"providers     {', '.join(groups)}")
    print(f"floor         {args.floor}")
    print(f"pooling       by {args.weight}")

    w, c, f, nw, nc, nf = stat(M, labels, groups, fg, args.weight)
    d = split(w, c, f)

    print("\n" + "=" * 68)
    print("DIVERSIFICATION FLOOR -- pooled across providers")
    print("=" * 68)
    print(f"  within-provider overlap   {w:.4f}   ({nw} pairs)")
    print(f"  cross-provider overlap    {c:.4f}   ({nc} pairs)")
    print(f"  uncoordinated floor       {f:.4f}   ({nf} pairs)")
    print(f"  excess above floor        {d['excess_within']:.4f} within   "
          f"{d['excess_cross']:.4f} cross")
    print()
    print(f"  removed by mixing vendors {pct(d['removed_pct'])}")
    print(f"  FLOOR (cannot be removed) {pct(d['floor_pct'])}")
    if f > 0:
        print(f"  residual overlap is       {c / f:.1f}x the uncoordinated level")

    rows = [dict(scope="POOLED", within=w, cross=c, floor=f,
                 n_within=nw, n_cross=nc, **d,
                 cross_x_floor=(c / f if f > 0 else np.nan))]

    # ---- per provider: one side of the pair is this model
    print("\n" + "-" * 68)
    print("BY PROVIDER  (cross = pairs with exactly one agent from this model)")
    print("-" * 68)
    print(f"{'model':<9}{'within':>9}{'cross':>9}{'removed':>10}{'floor':>10}"
          f"{'x floor':>9}")
    for g in groups:
        wm, nwm = within(M, labels, g)
        others = [o for o in groups if o != g]
        cm, ncm = between(M, labels, [g], others)
        dm = split(wm, cm, f)
        xf = cm / f if f > 0 else np.nan
        print(f"{g:<9}{wm:>9.4f}{cm:>9.4f}{pct(dm['removed_pct']):>10}"
              f"{pct(dm['floor_pct']):>10}"
              f"{('n/a' if not np.isfinite(xf) else f'{xf:.1f}x'):>9}")
        rows.append(dict(scope=g, within=wm, cross=cm, floor=f,
                         n_within=nwm, n_cross=ncm, **dm, cross_x_floor=xf))

    # ---- uncertainty
    if args.boot:
        print("\n" + "-" * 68)
        print(f"AGENT-LEVEL BOOTSTRAP  ({args.boot} resamples of the roster)")
        print("-" * 68)
        b = bootstrap(M, labels, groups, fg, args.weight, args.boot, args.seed)
        for k, lab in (("removed_pct", "removed by mixing"),
                       ("floor_pct", "FLOOR"),
                       ("cross_x_floor", "residual / floor")):
            lo, hi = ci(b[k])
            if k == "cross_x_floor":
                print(f"  {lab:<20} 95% CI  {lo:.1f}x  to  {hi:.1f}x")
            else:
                print(f"  {lab:<20} 95% CI  {pct(lo)}  to  {pct(hi)}")
        lo, hi = ci(b["floor_pct"])
        rows[0]["floor_pct_lo"] = lo
        rows[0]["floor_pct_hi"] = hi

    # ---- robustness: does the answer depend on which floor you pick?
    print("\n" + "-" * 68)
    print("ROBUSTNESS -- same calculation under each choice of floor")
    print("-" * 68)
    print(f"{'floor':<12}{'value':>9}{'removed':>10}{'FLOOR':>10}")
    rob = []
    for choice in ("random", "popular", "zero"):
        g2 = None if choice == "zero" else [choice]
        if g2 and g2[0] not in have:
            continue
        w2, c2, f2, *_ = stat(M, labels, groups, g2, args.weight)
        d2 = split(w2, c2, f2)
        print(f"{choice:<12}{f2:>9.4f}{pct(d2['removed_pct']):>10}"
              f"{pct(d2['floor_pct']):>10}")
        rob.append(dict(floor_source=choice, floor=f2, within=w2, cross=c2,
                        **d2))
    print("\n  zero is the worst case for the claim: it credits chance overlap")
    print("  to the models and so understates how much mixing removes.")

    os.makedirs(args.out, exist_ok=True)
    t1 = os.path.join(args.out, f"diversification_floor_{args.metric}.csv")
    t2 = os.path.join(args.out, f"diversification_floor_{args.metric}_robustness.csv")
    pd.DataFrame(rows).round(6).to_csv(t1, index=False)
    pd.DataFrame(rob).round(6).to_csv(t2, index=False)
    print(f"\nwrote {t1}")
    print(f"wrote {t2}\n")


if __name__ == "__main__":
    main()
