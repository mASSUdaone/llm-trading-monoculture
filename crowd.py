#!/usr/bin/env python3
"""
crowd.py -- group-level overlap, unweighted and weighted, yearly and per bar.

    python3 crowd.py runs/*
    python3 crowd.py runs/* --smooth 5 --mixes 200

Two families of measure, each computed two ways (unweighted / weighted):

A. PAIRWISE JACCARD, averaged over every pair in a group
     unweighted   J  = |A ∩ B| / |A ∪ B|                     (shares names?)
     weighted     Jw = Σ min(wa, wb) / Σ max(wa, wb)         (shares money?)
   Groups: claude, gemini, gpt (within-model), all_llm (every pair among
   the LLM agents, within and cross together), random, popular.

B. EFFECTIVE NAMES OF THE WHOLE CROWD, all agents at once, no pairs
     pool the group's holdings, p_j = share of the pile in stock j,
     effective names = 1 / Σ p_j²
     unweighted   pile = number of agents holding each stock
     weighted     pile = sum of the agents' portfolio weights in each stock
   Groups: claude, gemini, gpt (30 each), mixed (10 + 10 + 10 drawn from the
   three providers, redrawn --mixes times), random (30), popular (30),
   all_llm (every active LLM agent).

   Effective names grows with crowd size even with no strategy at all, so
   only equal-size crowds are compared: a single-provider 30 against a
   mixed 30 against a random 30. all_llm is larger and is descriptive only.

Everything is computed at every scored bar, then averaged by day for the
charts and over the year for the headline tables. Days are tagged calm /
normal / stressed exactly as in report.py.

Outputs
    report/csv/crowd_annual.csv
    report/csv/crowd_bars.csv
    report/csv/crowd_days.csv
    report/figures/crowd_jaccard.png
    report/figures/crowd_effective_names.png
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

try:
    import report
except ImportError:
    sys.exit("error: report.py and score.py must sit next to crowd.py")

LLM = ("claude", "gemini", "gpt")
COLORS = {"claude": "#C4652C", "gemini": "#2F6FBF", "gpt": "#2E8B57",
          "all_llm": "#222222", "mixed": "#8A4FB8",
          "random": "#A8A8A8", "popular": "#C9A227"}
ORDER = ["claude", "gemini", "gpt", "mixed", "all_llm", "random", "popular"]


class Say:
    def __call__(self, *a):
        print(" ".join(str(x) for x in a))

    def head(self, t):
        print("\n" + "=" * 70 + "\n" + t + "\n" + "=" * 70)


def eff_names(M, X):
    """Effective names for each roster (row of M) given a bar's matrix X."""
    C = M @ X                                   # rosters x tickers
    tot = C.sum(axis=1)
    p = np.divide(C, tot[:, None], out=np.zeros_like(C), where=tot[:, None] > 0)
    hhi = (p ** 2).sum(axis=1)
    return np.divide(1.0, hhi, out=np.full_like(hhi, np.nan), where=hhi > 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--from", dest="start", default="2025-01-01")
    ap.add_argument("--min-active", type=float, default=0.05, dest="min_active")
    ap.add_argument("--per", type=int, default=10,
                    help="agents drawn from each provider for a mixed crowd")
    ap.add_argument("--mixes", type=int, default=200,
                    help="how many mixed crowds to draw")
    ap.add_argument("--smooth", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260927)
    args = ap.parse_args()

    say = Say()
    D = report.Data(args.runs, args.start, say)
    keep = D.active >= args.min_active
    lab = D.labels
    if (~keep).any():
        say("excluded as inactive: " + ", ".join(np.array(D.names)[~keep]))

    have = [m for m in LLM if ((lab == m) & keep).sum() >= 2]
    if len(have) < 2:
        sys.exit("need at least two LLM groups")

    # ------------------------------------------------ pair masks (family A)
    n = D.na
    iu = np.triu_indices(n, 1)
    la, lb = lab[iu[0]], lab[iu[1]]
    live = keep[iu[0]] & keep[iu[1]]
    pairs = {}
    for g in LLM + ("random", "popular"):
        m = live & (la == g) & (lb == g)
        if m.any():
            pairs[g] = m
    pairs["all_llm"] = live & np.isin(la, have) & np.isin(lb, have)

    # ---------------------------------------------- crowd rosters (family B)
    rng = np.random.default_rng(args.seed)
    idx = {g: np.where((lab == g) & keep)[0] for g in set(lab)}
    named = [g for g in LLM + ("random", "popular") if len(idx.get(g, [])) >= 2]
    named.append("all_llm")
    idx["all_llm"] = np.concatenate([idx[m] for m in have])

    rows = []
    for g in named:
        r = np.zeros(n, np.float32); r[idx[g]] = 1; rows.append(r)
    can_mix = all(len(idx[m]) >= args.per for m in have)
    n_mix = args.mixes if can_mix else 0
    for _ in range(n_mix):
        r = np.zeros(n, np.float32)
        for m in have:
            r[rng.choice(idx[m], args.per, replace=False)] = 1
        rows.append(r)
    M = np.stack(rows)
    k_named = len(named)
    mix_size = args.per * len(have)
    if not can_mix:
        say(f"note: a provider has fewer than {args.per} active agents; "
            "mixed crowds skipped")

    say(f"\ncrowd sizes: " + ", ".join(f"{g} {len(idx[g])}" for g in named)
        + (f", mixed {mix_size} ({n_mix} draws)" if n_mix else ""))

    # ------------------------------------------------------------- main loop
    say("\ncomputing every bar...")
    series = {}
    for fam in ("jac", "jacw"):
        for g in pairs:
            series[f"{fam}_{g}"] = np.full(D.nb, np.nan)
    for fam in ("eff", "effw"):
        for g in named:
            series[f"{fam}_{g}"] = np.full(D.nb, np.nan)
        if n_mix:
            for s in ("mixed", "mixed_lo", "mixed_hi"):
                series[f"{fam}_{s}"] = np.full(D.nb, np.nan)
    mix_sum = {f: np.zeros(n_mix) for f in ("eff", "effw")}
    mix_cnt = {f: np.zeros(n_mix) for f in ("eff", "effw")}

    for b0, b1 in D.chunks():
        H = D.dense_hold(b0, b1)
        V = D.dense_value(b0, b1)
        tot = V.sum(axis=2, keepdims=True)
        W = np.divide(V, tot, out=np.zeros_like(V), where=tot > 0)
        for k in range(b1 - b0):
            b = b0 + k
            X = H[:, k, :]
            Wk = W[:, k, :]

            # A. pairwise, unweighted
            inter = X @ X.T
            sz = X.sum(axis=1)
            union = sz[:, None] + sz[None, :] - inter
            Jb = np.divide(inter, union, out=np.full_like(inter, np.nan),
                           where=union > 0)[iu]
            # A. pairwise, weighted: Σmin = (sa+sb-L1)/2, Σmax = (sa+sb+L1)/2
            L1 = cdist(Wk, Wk, "cityblock")
            s = Wk.sum(axis=1)
            ss = s[:, None] + s[None, :]
            Jw = np.divide(ss - L1, ss + L1, out=np.full_like(L1, np.nan),
                           where=(ss + L1) > 0)[iu]
            for g, m in pairs.items():
                for fam, J in (("jac", Jb), ("jacw", Jw)):
                    v = J[m]; v = v[np.isfinite(v)]
                    if len(v):
                        series[f"{fam}_{g}"][b] = v.mean()

            # B. whole-crowd effective names
            for fam, Z in (("eff", X), ("effw", Wk)):
                e = eff_names(M, Z)
                for i, g in enumerate(named):
                    series[f"{fam}_{g}"][b] = e[i]
                if n_mix:
                    em = e[k_named:]
                    ok = np.isfinite(em)
                    if ok.any():
                        series[f"{fam}_mixed"][b] = em[ok].mean()
                        series[f"{fam}_mixed_lo"][b] = np.percentile(em[ok], 2.5)
                        series[f"{fam}_mixed_hi"][b] = np.percentile(em[ok], 97.5)
                    mix_sum[fam][ok] += em[ok]
                    mix_cnt[fam][ok] += 1
        say(f"  bars {b1:,}/{D.nb:,}")

    # ------------------------------------------------------------ assemble
    tag, *_ = D.regimes()
    bars = pd.DataFrame({"timestamp": D.bars, **series})
    bars["regime"] = tag
    bars["day"] = pd.to_datetime(bars["timestamp"]).dt.date
    num = [c for c in bars.columns if c not in ("timestamp", "regime", "day")]
    days = bars.groupby("day").agg({**{c: "mean" for c in num}, "regime": "first"})

    os.makedirs(report.CSVDIR, exist_ok=True)
    bars.drop(columns="day").to_csv(os.path.join(report.CSVDIR, "crowd_bars.csv"),
                                    index=False)
    days.reset_index().to_csv(os.path.join(report.CSVDIR, "crowd_days.csv"),
                              index=False)

    # ------------------------------------------------------- annual tables
    ann = []
    say.head("A. PAIRWISE JACCARD -- year average over every pair in the group")
    say("unweighted = shares the same names?   weighted = shares the same money?\n")
    say(f"  {'group':<10}{'pairs':>7}{'unweighted':>13}{'weighted':>11}")
    for g in [g for g in ORDER if g in pairs]:
        ju, jw = np.nanmean(bars[f"jac_{g}"]), np.nanmean(bars[f"jacw_{g}"])
        say(f"  {g:<10}{int(pairs[g].sum()):>7}{ju:>13.4f}{jw:>11.4f}")
        ann.append({"measure": "pairwise_jaccard", "group": g,
                    "size": int(pairs[g].sum()), "unweighted": round(ju, 4),
                    "weighted": round(jw, 4)})
    say("\n  all_llm mixes within-model and cross-model pairs together:")
    say("  it is the average resemblance of any two AI agents, whoever made them.")

    say.head("B. EFFECTIVE NAMES OF THE WHOLE CROWD -- year average")
    say("how many stocks the crowd is really spread across (lower = more crowded)\n")
    say(f"  {'crowd':<10}{'agents':>7}{'unweighted':>13}{'weighted':>11}")
    out_b = {}
    for g in [g for g in ORDER if g in named or (g == "mixed" and n_mix)]:
        if g == "mixed":
            size = mix_size
            eu = np.nanmean(bars["eff_mixed"]); ew = np.nanmean(bars["effw_mixed"])
        else:
            size = len(idx[g])
            eu = np.nanmean(bars[f"eff_{g}"]); ew = np.nanmean(bars[f"effw_{g}"])
        out_b[g] = (eu, ew)
        note = "   (larger crowd, not comparable)" if g == "all_llm" else ""
        say(f"  {g:<10}{size:>7}{eu:>13.1f}{ew:>11.1f}{note}")
        ann.append({"measure": "effective_names", "group": g, "size": size,
                    "unweighted": round(eu, 2), "weighted": round(ew, 2)})

    if n_mix:
        say(f"\n  mixed crowds, spread across the {n_mix} draws (95% range):")
        for fam, lbl in (("eff", "unweighted"), ("effw", "weighted")):
            a = mix_sum[fam] / np.maximum(mix_cnt[fam], 1)
            lo, hi = np.percentile(a, [2.5, 97.5])
            say(f"    {lbl:<11} {lo:.1f} to {hi:.1f}")
            ann.append({"measure": f"effective_names_mixed_range_{lbl}",
                        "group": "mixed", "size": mix_size,
                        "unweighted": round(lo, 2), "weighted": round(hi, 2)})

        singles = [g for g in LLM if g in out_b]
        say.head("DOES MIXING PROVIDERS SPREAD THE CROWD?  (equal-size crowds)")
        for j, lbl in ((0, "unweighted"), (1, "weighted")):
            s_avg = np.mean([out_b[g][j] for g in singles])
            mx = out_b["mixed"][j]
            say(f"\n  {lbl}")
            say(f"    single-provider crowd (avg)  {s_avg:8.1f} names")
            say(f"    mixed crowd                  {mx:8.1f} names   "
                f"({mx / s_avg:.2f}x a single provider)")
            if "random" in out_b:
                rd = out_b["random"][j]
                say(f"    random crowd                 {rd:8.1f} names   "
                    f"({rd / s_avg:.1f}x a single provider)")

    say.head("BY MARKET REGIME (daily means)")
    keycols = [c for c in ["jac_all_llm", "jacw_all_llm", "eff_mixed",
                           "effw_mixed", "eff_random"] if c in days.columns]
    keycols += [f"jac_{g}" for g in LLM if f"jac_{g}" in days.columns]
    g = days.groupby("regime")[keycols].mean().reindex(["calm", "normal", "stressed"])
    say(g.round(4).T.to_string())

    pd.DataFrame(ann).to_csv(os.path.join(report.CSVDIR, "crowd_annual.csv"),
                             index=False)

    # -------------------------------------------------------------- charts
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        say("\nmatplotlib missing -- charts skipped")
        return 0

    t = pd.to_datetime(days.index)
    stressed = (days["regime"] == "stressed").to_numpy()

    def shade(ax):
        for i, s in enumerate(stressed):
            if s:
                ax.axvspan(t[i] - pd.Timedelta(hours=12), t[i] + pd.Timedelta(hours=12),
                           color="#E8A0A0", alpha=0.22, lw=0)

    def line(ax, col, g, lw=1.3, ls="-"):
        if col not in days.columns:
            return
        c = COLORS.get(g, "#444")
        y = days[col]
        if args.smooth > 1:
            ax.plot(t, y, color=c, alpha=0.22, lw=0.7)
            y = y.rolling(args.smooth, min_periods=1).mean()
        ax.plot(t, y, color=c, lw=lw, ls=ls, label=g)

    def style(ax, ylabel, title):
        ax.set_ylabel(ylabel); ax.set_title(title, fontsize=10.5, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#E6E6E6", lw=0.6)

    # figure 1: pairwise Jaccard, unweighted and weighted
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.4), sharex=True)
    for ax, fam, name in ((axes[0], "jac", "unweighted"), (axes[1], "jacw", "weighted")):
        shade(ax)
        for g in ["claude", "gemini", "gpt"]:
            line(ax, f"{fam}_{g}", g)
        line(ax, f"{fam}_all_llm", "all_llm", lw=2.0)
        line(ax, f"{fam}_random", "random", lw=1, ls="--")
        line(ax, f"{fam}_popular", "popular", lw=1, ls="--")
        ax.set_ylim(bottom=0)
        style(ax, "mean pairwise Jaccard",
              f"Pairwise Jaccard, {name}  ·  shaded = stressed days")
    axes[0].legend(fontsize=8, ncol=6, loc="upper left", frameon=False)
    p1 = os.path.join(report.FIGDIR, "crowd_jaccard.png")
    os.makedirs(report.FIGDIR, exist_ok=True)
    fig.savefig(p1, dpi=150, bbox_inches="tight"); plt.close(fig)

    # figure 2: effective names of the crowd, unweighted and weighted
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.4), sharex=True)
    for ax, fam, name in ((axes[0], "eff", "unweighted"), (axes[1], "effw", "weighted")):
        shade(ax)
        if n_mix and f"{fam}_mixed_lo" in days.columns:
            lo, hi = days[f"{fam}_mixed_lo"], days[f"{fam}_mixed_hi"]
            if args.smooth > 1:
                lo = lo.rolling(args.smooth, min_periods=1).mean()
                hi = hi.rolling(args.smooth, min_periods=1).mean()
            ax.fill_between(t, lo, hi, color=COLORS["mixed"], alpha=0.15, lw=0)
        for g in ["claude", "gemini", "gpt"]:
            line(ax, f"{fam}_{g}", g)
        line(ax, f"{fam}_mixed", "mixed", lw=2.0)
        line(ax, f"{fam}_random", "random", lw=1, ls="--")
        line(ax, f"{fam}_popular", "popular", lw=1, ls="--")
        ax.set_yscale("log")
        from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
        ax.yaxis.set_major_locator(FixedLocator([2, 5, 10, 20, 50, 100, 200, 500]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        style(ax, "effective names (log scale)",
              f"Whole-crowd effective names, {name}, crowds of {mix_size or 30}"
              "  ·  band = 95% of mixed crowds")
    axes[0].legend(fontsize=8, ncol=6, loc="upper left", frameon=False)
    p2 = os.path.join(report.FIGDIR, "crowd_effective_names.png")
    fig.savefig(p2, dpi=150, bbox_inches="tight"); plt.close(fig)

    say(f"\nwrote {report.CSVDIR}/crowd_annual.csv, crowd_bars.csv, crowd_days.csv")
    say(f"wrote {p1}")
    say(f"wrote {p2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
