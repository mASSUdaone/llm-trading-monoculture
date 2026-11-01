#!/usr/bin/env python3
"""
timeline.py -- binary Jaccard through the scored year, bar by bar.

    python3 timeline.py runs/*
    python3 timeline.py runs/* --smooth 5       # 5-day rolling line on top

What it does
------------
The headline Jaccard is one number per group for the whole year. This keeps
the time dimension instead: at every bar, average the Jaccard over all pairs
in a group, giving one value per bar per group.

Groups:
    claude / gemini / gpt   within-model pairs
    cross                   pairs from two DIFFERENT LLMs only
                            (synthetic agents excluded, unlike report.py)
    random / popular        synthetic floors, if present

Bars are then averaged into days so the chart is readable (13 bars a day,
~250 days), and each day is tagged calm / normal / stressed by the same rule
report.py uses: terciles of 5-day volatility of an equal-weighted index.

Outputs
    report/csv/jaccard_timeline_bars.csv    one row per bar
    report/csv/jaccard_timeline_days.csv    one row per day, with regime
    report/figures/jaccard_timeline.png     the chart

A note on the averages
----------------------
The yearly headline averages each PAIR over the bars, then averages pairs.
This averages each BAR over the pairs, then averages bars. The two orders
give slightly different numbers when some pairs have more usable bars than
others (both-flat bars are skipped). Expect the means here to be close to
the headline figures, not identical.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

try:
    import report
except ImportError:
    sys.exit("error: report.py and score.py must sit next to timeline.py")

LLM = ("claude", "gemini", "gpt")
COLORS = {"claude": "#C4652C", "gemini": "#2F6FBF", "gpt": "#2E8B57",
          "cross": "#6B6B6B", "random": "#B0B0B0", "popular": "#C9A227"}


class Say:
    def __call__(self, *a):
        print(" ".join(str(x) for x in a))

    def head(self, t):
        print("\n" + "=" * 68 + "\n" + t + "\n" + "=" * 68)


def series_per_bar(D, keep):
    """Mean pairwise Jaccard at every bar, per group."""
    n = D.na
    iu = np.triu_indices(n, 1)
    la, lb = D.labels[iu[0]], D.labels[iu[1]]
    live = keep[iu[0]] & keep[iu[1]]

    groups = {}
    for g in sorted(set(D.labels)):
        m = live & (la == g) & (lb == g)
        if m.any():
            groups[g] = m
    xm = live & (la != lb) & np.isin(la, LLM) & np.isin(lb, LLM)
    if xm.any():
        groups["cross"] = xm

    out = {g: np.full(D.nb, np.nan) for g in groups}
    for b0, b1 in D.chunks():
        H = D.dense_hold(b0, b1)
        for k in range(b1 - b0):
            X = H[:, k, :]
            inter = X @ X.T
            sz = X.sum(axis=1)
            union = sz[:, None] + sz[None, :] - inter
            J = np.divide(inter, union, out=np.full_like(inter, np.nan),
                          where=union > 0)[iu]
            for g, m in groups.items():
                v = J[m]
                v = v[np.isfinite(v)]
                if len(v):
                    out[g][b0 + k] = v.mean()
        print(f"  bars {b1:,}/{D.nb:,}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--from", dest="start", default="2025-01-01")
    ap.add_argument("--min-active", type=float, default=0.05, dest="min_active")
    ap.add_argument("--smooth", type=int, default=0,
                    help="optional rolling window in days drawn over the daily line")
    args = ap.parse_args()

    say = Say()
    D = report.Data(args.runs, args.start, say)
    keep = D.active >= args.min_active
    dropped = [nm for nm, k in zip(D.names, keep) if not k]
    if dropped:
        say(f"excluded as inactive (<{args.min_active:.0%} of bars held): "
            f"{', '.join(dropped)}")

    say.head("PER-BAR JACCARD")
    S = series_per_bar(D, keep)

    tag, *_ = D.regimes()
    bars = pd.DataFrame({"timestamp": D.bars, **S})
    bars["regime"] = tag
    bars["day"] = pd.to_datetime(bars["timestamp"]).dt.date

    cols = [c for c in ["claude", "gemini", "gpt", "cross", "random", "popular"]
            if c in bars.columns]
    days = bars.groupby("day").agg({**{c: "mean" for c in cols}, "regime": "first"})

    os.makedirs(report.CSVDIR, exist_ok=True)
    bars.drop(columns="day").to_csv(
        os.path.join(report.CSVDIR, "jaccard_timeline_bars.csv"), index=False)
    days.reset_index().to_csv(
        os.path.join(report.CSVDIR, "jaccard_timeline_days.csv"), index=False)

    # ---------------------------------------------------------------- tables
    say.head("YEAR MEAN OF THE PER-BAR SERIES")
    say("(close to the headline numbers, not identical -- see the docstring)\n")
    for c in cols:
        say(f"  {c:<8} {np.nanmean(bars[c]):.4f}")

    say.head("BY MARKET REGIME")
    say("mean daily Jaccard in calm / normal / stressed days\n")
    g = days.groupby("regime")[cols].mean().reindex(["calm", "normal", "stressed"])
    say(g.round(4).to_string())
    if "stressed" in g.index and "calm" in g.index:
        say("\n  stressed minus calm:")
        for c in cols:
            d = g.loc["stressed", c] - g.loc["calm", c]
            base = g.loc["calm", c]
            pct = f"  ({d / base:+.1%})" if base else ""
            say(f"    {c:<8} {d:+.4f}{pct}")

    say.head("HIGHEST-OVERLAP DAYS")
    for c in [c for c in cols if c in LLM + ("cross",)]:
        top = days[c].nlargest(5)
        say(f"\n  {c}")
        for d, v in top.items():
            say(f"    {d}  {v:.4f}  ({days.loc[d, 'regime']})")

    shared = set.intersection(*[set(days[c].nlargest(10).index)
                                for c in LLM if c in days.columns])
    if shared:
        say(f"\n  in the top 10 for every model: "
            f"{', '.join(str(d) for d in sorted(shared))}")

    # ----------------------------------------------------------------- chart
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        say("\nmatplotlib missing -- chart skipped (pip3 install matplotlib)")
        return 0

    t = pd.to_datetime(days.index)
    fig, ax = plt.subplots(figsize=(11, 4.6))

    stressed = (days["regime"] == "stressed").to_numpy()
    for i, s in enumerate(stressed):
        if s:
            ax.axvspan(t[i] - pd.Timedelta(hours=12), t[i] + pd.Timedelta(hours=12),
                       color="#E8A0A0", alpha=0.25, lw=0)

    for c in cols:
        style = dict(color=COLORS.get(c, "#444"), label=c)
        if c in ("random", "popular"):
            style.update(lw=1, ls="--")
        elif c == "cross":
            style.update(lw=1.6, ls="-")
        else:
            style.update(lw=1.3)
        if args.smooth > 1:
            ax.plot(t, days[c], alpha=0.25, lw=0.8, color=style["color"])
            ax.plot(t, days[c].rolling(args.smooth, min_periods=1).mean(), **style)
        else:
            ax.plot(t, days[c], **style)

    ax.set_ylabel("mean pairwise Jaccard (daily)")
    ax.set_ylim(bottom=0)
    ax.set_title("Holdings overlap through the scored year  ·  shaded = stressed days")
    ax.legend(fontsize=8, ncol=len(cols), loc="upper left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E6E6E6", lw=0.6)

    os.makedirs(report.FIGDIR, exist_ok=True)
    p = os.path.join(report.FIGDIR, "jaccard_timeline.png")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)

    say(f"\nwrote {report.CSVDIR}/jaccard_timeline_bars.csv")
    say(f"wrote {report.CSVDIR}/jaccard_timeline_days.csv")
    say(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
