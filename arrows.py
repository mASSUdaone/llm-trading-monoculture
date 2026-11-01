#!/usr/bin/env python3
"""
arrows.py -- see your agents' portfolios as arrows.

    python3 arrows.py runs/*                      # whole-year average mix
    python3 arrows.py runs/* --date 2025-04-04    # one day's mix

Each agent's portfolio is an arrow in 486-dimensional space, one direction
per stock. You cannot draw 486 dimensions, so this finds the two directions
that the agents' arrows use the most and draws the shadow of every arrow on
that flat plane.

How to read the left chart
    Arrows pointing the same way  -> similar portfolio mix (high cosine)
    Arrows fanned apart           -> different mixes
    A SHORT arrow                 -> this portfolio points mostly into the
                                     other 484 directions the picture cannot
                                     show. Random agents spread over hundreds
                                     of stocks, so their arrows come out short.

How to read the right chart
    The cosine between every pair of agents, sorted by model. Each square is
    one pair. Bright blocks along the diagonal = agents from the same model
    point the same way.

Every arrow is scaled to length 1 first, so only the mix matters, not the
amount invested -- exactly what cosine does.

Outputs
    report/figures/arrows.png          (or arrows_<date>.png)
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

try:
    import report
except ImportError:
    sys.exit("error: report.py and score.py must sit next to arrows.py")

ORDER = ["claude", "gemini", "gpt", "popular", "random"]
COLORS = {"claude": "#C4652C", "gemini": "#2F6FBF", "gpt": "#2E8B57",
          "popular": "#C9A227", "random": "#9A9A9A"}


class Quiet:
    def __call__(self, *a):
        pass

    def head(self, t):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--from", dest="start", default="2025-01-01")
    ap.add_argument("--date", default=None, help="YYYY-MM-DD: use one day's mix")
    ap.add_argument("--min-active", type=float, default=0.05, dest="min_active")
    args = ap.parse_args()

    print("loading agents...")
    D = report.Data(args.runs, args.start, Quiet())

    if args.date:
        sel = np.array([str(b).startswith(args.date) for b in D.bars])
        if not sel.any():
            sys.exit(f"no scored bars on {args.date}")
    else:
        sel = np.ones(D.nb, bool)
    rows = np.where(sel)[0]

    # average portfolio mix of each agent over the selected bars
    names, labels, vecs = [], [], []
    for i, nm in enumerate(D.names):
        if D.active[i] < args.min_active:
            continue
        V = D.value[i][rows].toarray()
        tot = V.sum(axis=1, keepdims=True)
        live = tot[:, 0] > 0
        if not live.any():
            continue
        w = (V[live] / tot[live]).mean(axis=0)          # average share per stock
        n = np.linalg.norm(w)
        if n == 0:
            continue
        names.append(nm); labels.append(D.labels[i]); vecs.append(w / n)
    X = np.array(vecs)
    labels = np.array(labels)
    print(f"{len(X)} agents with holdings, {X.shape[1]} stocks")

    # the two directions the arrows use most (uncentred SVD keeps the origin,
    # so the shadows are still arrows from the same starting point)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    P = X @ Vt[:2].T
    if P[:, 0].mean() < 0:
        P[:, 0] *= -1
    shown = (S[:2] ** 2).sum() / (S ** 2).sum()

    # the stocks that define each drawn direction
    tk = D.tickers
    top = [", ".join(tk[np.argsort(-np.abs(Vt[k]))[:4]]) for k in range(2)]

    C = X @ X.T                                            # cosine, every pair
    order = np.concatenate([np.where(labels == g)[0] for g in ORDER if (labels == g).any()])

    print(f"\nthe flat picture shows {shown:.0%} of what separates the arrows")
    print(f"  direction 1 is mostly: {top[0]}")
    print(f"  direction 2 is mostly: {top[1]}")
    print("\naverage shadow length (1.0 = arrow lies fully in the picture):")
    for g in ORDER:
        m = labels == g
        if m.any():
            print(f"  {g:<8} {np.linalg.norm(P[m], axis=1).mean():.2f}")
    print("\naverage cosine within each group (this chart's version -- the mix")
    print("is averaged over the period first, so it runs higher than report.py):")
    for g in ORDER:
        m = np.where(labels == g)[0]
        if len(m) > 1:
            sub = C[np.ix_(m, m)][np.triu_indices(len(m), 1)]
            print(f"  {g:<8} {sub.mean():.3f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("matplotlib missing (pip3 install matplotlib)")

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(14, 6.4),
                                 gridspec_kw={"width_ratios": [1.1, 1]})

    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(th), np.sin(th), color="#DDDDDD", lw=0.8, zorder=0)
    for g in ["random", "popular", "gemini", "claude", "gpt"]:
        m = labels == g
        if not m.any():
            continue
        for x, y in P[m]:
            ax.annotate("", xy=(x, y), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color=COLORS[g],
                                        lw=1.1, alpha=0.75, mutation_scale=9))
        ax.plot([], [], color=COLORS[g], lw=2, label=g)
    ax.set_aspect("equal")
    lim = 1.08
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.axhline(0, color="#EEEEEE", lw=0.8, zorder=0)
    ax.axvline(0, color="#EEEEEE", lw=0.8, zorder=0)
    ax.set_xlabel(f"direction 1  ({top[0]} ...)", fontsize=9)
    ax.set_ylabel(f"direction 2  ({top[1]} ...)", fontsize=9)
    when = args.date or "whole scored year"
    ax.set_title(f"Every portfolio as an arrow  ·  {when}\n"
                 f"flat picture shows {shown:.0%} of the differences · "
                 "grey circle = full length", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)

    im = bx.imshow(C[np.ix_(order, order)], cmap="viridis", vmin=0, vmax=1,
                   interpolation="nearest")
    edges, pos = [], 0
    for g in ORDER:
        k = (labels == g).sum()
        if k:
            if pos:
                bx.axhline(pos - 0.5, color="white", lw=0.8)
                bx.axvline(pos - 0.5, color="white", lw=0.8)
            edges.append((pos + k / 2, g)); pos += k
    bx.set_xticks([e[0] for e in edges]); bx.set_xticklabels([e[1] for e in edges], fontsize=9)
    bx.set_yticks([e[0] for e in edges]); bx.set_yticklabels([e[1] for e in edges], fontsize=9)
    bx.set_title("Cosine between every pair of agents\n(bright = arrows point the same way)",
                 fontsize=10)
    fig.colorbar(im, ax=bx, fraction=0.046, pad=0.03, label="cosine")

    os.makedirs(report.FIGDIR, exist_ok=True)
    fn = f"arrows_{args.date}.png" if args.date else "arrows.png"
    p = os.path.join(report.FIGDIR, fn)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
