#!/usr/bin/env python3
"""
report.py -- the full monoculture analysis, in one place.

    python3 report.py all runs/*              # everything
    python3 report.py overlap runs/*          # pairwise + nulls + clusters
    python3 report.py flow runs/*             # signed flow + null
    python3 report.py time runs/*             # everything split by time/regime
    python3 report.py crowding runs/*         # population concentration
    python3 report.py strategy runs/*         # behaviour + source classification
    python3 report.py returns runs/*          # side analysis, not a headline

Everything lands in one folder:
    report/summary.txt      every line this program printed
    report/csv/*.csv        every table
    report/figures/*.png    every plot

Imports score.py for position reconstruction so the fill rules live in exactly
one place and cannot drift between programs.

A note on what these metrics can and cannot show. Holdings overlap measures
what agents HOLD. A crash is caused by what agents DO -- many of them selling
in the same bar. Flow imbalance is the measure closer to that mechanism, and
the time block exists because an annual average hides precisely the episodes
that matter. Read the flow and time sections before quoting the overlap number.
"""

import argparse
import os
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd
import scipy.sparse as sp

OUT = "report"
CSVDIR = os.path.join(OUT, "csv")
FIGDIR = os.path.join(OUT, "figures")
AGENT_DIR = "agents"
CASH = 100_000.0
ITERS = 5000
FLOW_ITERS = 500
CHUNK = 250
EXTREME = (0.5, 0.7, 0.9)

try:
    import score
except ImportError:
    sys.exit("error: score.py must sit next to report.py")


# ------------------------------------------------------------------ plumbing

class Tee:
    """Print to the terminal and to report/summary.txt at the same time."""

    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "a")

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s)
        self.f.write(s + "\n")
        self.f.flush()

    def head(self, title):
        self("\n" + "=" * 72)
        self(title)
        self("=" * 72)


def save(df, name, out):
    os.makedirs(CSVDIR, exist_ok=True)
    p = os.path.join(CSVDIR, name)
    df.to_csv(p, index=False)
    out(f"  -> csv/{name}")


def label_of(name):
    n = name.lower()
    if n.startswith("random") or n.startswith("baseline"):
        return "random"
    if n.startswith("popular") or n.startswith("weighted"):
        return "popular"
    if n.startswith("gemini"):
        return "gemini"
    if n.startswith("gpt") or n.startswith("openai"):
        return "gpt"
    if n.startswith("claude") or n.startswith("agent"):
        return "claude"
    return "unknown"


def figure(fn, name, out):
    """Run a plotting function, save the png, never let a plot crash the run."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    os.makedirs(FIGDIR, exist_ok=True)
    try:
        fig = fn(plt)
        if fig is None:
            return
        p = os.path.join(FIGDIR, name)
        fig.savefig(p, dpi=140, bbox_inches="tight")
        plt.close(fig)
        out(f"  -> figures/{name}")
    except Exception as e:
        out(f"  (figure {name} failed: {type(e).__name__}: {e})")


# --------------------------------------------------------------------- data

class Data:
    """Loads the tape once, reconstructs every agent, keeps it all sparse.

    Holdings and position values are stored as one sparse matrix per agent
    (bars x tickers). Dense would be 87 x 3250 x 486 floats -- half a gigabyte.
    Sparse is a few megabytes because an agent holds 5-40 names out of 486.
    Bars are densified in chunks only when a calculation needs them.
    """

    def __init__(self, runs, start, out):
        out("loading tape and reconstructing positions...")
        df = score.load_tape()
        bars, tickers, bi, ti, opens, closes = score.build_matrices(df)
        keep = bars >= start

        self.bars = bars[keep]
        self.tickers = np.array(tickers)
        self.ti = ti
        self.nb, self.nt = len(self.bars), len(self.tickers)
        self.closes = closes[keep]
        bpos = {b: i for i, b in enumerate(self.bars)}

        self.names, self.hold, self.value = [], [], []
        self.flow, self.aflow, self.summary = [], [], []

        for run in sorted(runs):
            p = os.path.join(run, "trades.csv")
            if not os.path.exists(p):
                continue
            name = os.path.basename(run.rstrip("/"))
            tr = pd.read_csv(p, dtype={"timestamp": str})
            shares, cash, values, equity, fill_bar = score.reconstruct(
                tr, bars, tickers, bi, ti, opens, closes)

            s, v, e = shares[keep], values[keep], equity[keep]
            self.names.append(name)
            self.hold.append(sp.csr_matrix(s > 0))
            self.value.append(sp.csr_matrix(v))

            scored = tr[tr["timestamp"] >= start]
            f = sp.lil_matrix((self.nb, self.nt), dtype=np.float32)
            for ts, tk, d, sz in scored.itertuples(index=False):
                if ts in bpos and tk in ti:
                    f[bpos[ts], ti[tk]] += sz if d == "BUY" else -sz
            f = f.tocsr()
            self.flow.append(f)
            self.aflow.append(abs(f))

            fills = score.window_fills(fill_bar, int(np.argmax(keep)))
            row = score.summarise(name, scored, s, v, e, fills, self.bars,
                                  bench=score.benchmark(closes, keep),
                                  shares_traded=int(scored["size"].sum()) if len(scored) else 0,
                                  bars_per_year=int(keep.sum()))
            row["model"] = label_of(name)
            self.summary.append(row)
            out(f"  {name}")

        if not self.names:
            sys.exit("no runs with trades.csv found")

        self.labels = np.array([label_of(n) for n in self.names])
        self.models = sorted(set(self.labels))
        self.na = len(self.names)
        self.active = np.array([h.getnnz(axis=1).astype(bool).mean()
                                for h in self.hold])
        out(f"\n{self.na} agents | {self.nb:,} scored bars | {self.nt} tickers")
        out(f"models: {', '.join(f'{m} {(self.labels == m).sum()}' for m in self.models)}")

    def chunks(self):
        for b0 in range(0, self.nb, CHUNK):
            yield b0, min(b0 + CHUNK, self.nb)

    def dense_hold(self, b0, b1):
        return np.stack([h[b0:b1].toarray() for h in self.hold]).astype(np.float32)

    def dense_value(self, b0, b1):
        return np.stack([v[b0:b1].toarray() for v in self.value]).astype(np.float32)

    def regimes(self):
        """Tag every bar calm / normal / stressed from the tape itself.

        An equal-weighted index of all tickers gives a daily return series;
        realised volatility over a 5-day window splits the year into terciles.
        This is a property of the market, not of the agents, so every model is
        tagged identically.
        """
        c = np.nan_to_num(self.closes)
        live = (c[0] > 0)
        idx = (c[:, live] / np.where(c[0, live] > 0, c[0, live], 1)).mean(axis=1)
        day = pd.to_datetime(pd.Series(self.bars)).dt.date
        dfi = pd.DataFrame({"day": day, "idx": idx})
        daily = dfi.groupby("day")["idx"].last()
        ret = daily.pct_change().fillna(0.0)
        vol = ret.rolling(5, min_periods=1).std().fillna(0.0)
        lo, hi = vol.quantile([1 / 3, 2 / 3])
        tag = pd.Series(np.where(vol <= lo, "calm",
                        np.where(vol >= hi, "stressed", "normal")), index=vol.index)
        return np.array([tag[d] for d in day]), daily, ret, vol


# ------------------------------------------------------------------ pairwise

def pair_matrices(D, out):
    """One pass over bars gives both the annual pair means and the time series.

    For each bar: intersection = X @ X.T, union = sizes + sizes - intersection.
    Bars where both agents hold nothing are undefined (0/0) and excluded from
    that pair's average; they are reported separately as simultaneous flatness,
    which is itself a coordination signal.
    """
    n = D.na
    jsum = np.zeros((n, n)); jcnt = np.zeros((n, n))
    csum = np.zeros((n, n)); ccnt = np.zeros((n, n))
    iu = np.triu_indices(n, 1)
    same = D.labels[iu[0]] == D.labels[iu[1]]
    jt = {m: np.full(D.nb, np.nan) for m in D.models}
    jt["cross"] = np.full(D.nb, np.nan)

    for b0, b1 in D.chunks():
        H = D.dense_hold(b0, b1)
        V = D.dense_value(b0, b1)
        for k in range(b1 - b0):
            X = H[:, k, :]
            inter = X @ X.T
            sz = X.sum(axis=1)
            union = sz[:, None] + sz[None, :] - inter
            ok = union > 0
            J = np.where(ok, inter / np.where(union > 0, union, 1), 0.0)
            jsum += np.where(ok, J, 0.0); jcnt += ok

            Y = V[:, k, :]
            nv = np.linalg.norm(Y, axis=1)
            good = nv > 0
            gg = good[:, None] & good[None, :]
            denom = nv[:, None] * nv[None, :]
            C = np.where(gg, (Y @ Y.T) / np.where(denom > 0, denom, 1), 0.0)
            csum += np.where(gg, C, 0.0); ccnt += gg

            v = J[iu]; m = ok[iu]
            for mo in D.models:
                s = same & (D.labels[iu[0]] == mo) & m
                if s.any():
                    jt[mo][b0 + k] = v[s].mean()
            s = (~same) & m
            if s.any():
                jt["cross"][b0 + k] = v[s].mean()
        out(f"  bars {b1:,}/{D.nb:,}")

    with np.errstate(invalid="ignore", divide="ignore"):
        Jm = np.where(jcnt > 0, jsum / np.where(jcnt > 0, jcnt, 1), np.nan)
        Cm = np.where(ccnt > 0, csum / np.where(ccnt > 0, ccnt, 1), np.nan)
    flat = 1.0 - jcnt / D.nb
    return Jm, Cm, flat, pd.DataFrame({"timestamp": D.bars, **jt})


def describe(v, label):
    v = np.asarray([x for x in v if np.isfinite(x)])
    if not len(v):
        return None
    q = np.percentile(v, [0, 25, 50, 75, 100])
    hist, _ = np.histogram(v, bins=10, range=(0, 1))
    d = {"group": label, "n_pairs": len(v),
         "mean": round(float(v.mean()), 4),
         "sd": round(float(v.std(ddof=1)), 4) if len(v) > 1 else 0.0,
         "min": round(float(q[0]), 4), "q1": round(float(q[1]), 4),
         "median": round(float(q[2]), 4), "q3": round(float(q[3]), 4),
         "max": round(float(q[4]), 4)}
    d["sd_over_mean"] = round(d["sd"] / d["mean"], 3) if d["mean"] else None
    for t in EXTREME:
        d[f"pairs_above_{t}"] = int((v > t).sum())
    for i in range(10):
        d[f"bin_{i/10:.1f}_{(i+1)/10:.1f}"] = int(hist[i])
    return d


def permute(vals, valid, iu, labels, models, iters, rng, out, tee):
    """Shuffle model labels, recompute within-group means. Group sizes preserved.

    Pairs are not independent -- each agent appears in n-1 of them -- so a
    t-test would be badly overconfident. Shuffling preserves that dependency
    and randomises only the label, which is the thing under test.
    """
    rows = []
    n = len(labels)
    for mo in list(models) + ["POOLED"]:
        if mo == "POOLED":
            m = (labels[iu[0]] == labels[iu[1]]) & valid
            size = None
        else:
            m = (labels[iu[0]] == mo) & (labels[iu[1]] == mo) & valid
            size = int((labels == mo).sum())
        if not m.any():
            continue
        obs = vals[m].mean()

        nd = np.empty(iters)
        perm = labels.copy()
        fake = np.zeros(n, dtype=bool)
        for k in range(iters):
            if size is None:
                rng.shuffle(perm)
                s = (perm[iu[0]] == perm[iu[1]]) & valid
            else:
                fake[:] = False
                fake[rng.choice(n, size=size, replace=False)] = True
                s = fake[iu[0]] & fake[iu[1]] & valid
            nd[k] = vals[s].mean() if s.any() else np.nan
        nd = nd[np.isfinite(nd)]
        p = (np.sum(nd >= obs) + 1) / (len(nd) + 1)
        gate = float(np.percentile(nd, 95))
        z = (obs - nd.mean()) / nd.std(ddof=1) if nd.std(ddof=1) > 0 else np.nan
        rows.append({"group": mo, "n_agents": size or n,
                     "observed": round(float(obs), 4),
                     "null_mean": round(float(nd.mean()), 4),
                     "null_sd": round(float(nd.std(ddof=1)), 4),
                     "gate_95pct": round(gate, 4),
                     "p_value": round(float(p), 5),
                     "z_score": round(float(z), 2),
                     "passes": bool(obs > gate)})
        tee(f"  {mo:<8} observed {obs:.4f}  null {nd.mean():.4f}  "
            f"gate {gate:.4f}  p {p:.5f}  z {z:5.2f}  "
            f"{'PASS' if obs > gate else 'fail'}")
    return pd.DataFrame(rows)


def cmd_overlap(D, out, args):
    out.head("PAIRWISE HOLDINGS OVERLAP")
    out("computing per-bar pair matrices (this is the slow part)")
    Jm, Cm, flat, ts = pair_matrices(D, out)
    D._J, D._C, D._ts = Jm, Cm, ts

    n = D.na
    iu = np.triu_indices(n, 1)
    excl = set(np.array(D.names)[D.active < args.min_active])
    if excl:
        out(f"\nexcluded from overlap (held something on <{args.min_active:.0%} "
            f"of bars): {', '.join(sorted(excl))}")
        out(f"  inactive rate {len(excl)}/{n} = {len(excl)/n:.1%}")
    keep = np.array([nm not in excl for nm in D.names])
    valid = keep[iu[0]] & keep[iu[1]] & np.isfinite(Jm[iu])

    # ---- full pair list
    pairs = pd.DataFrame({
        "a": np.array(D.names)[iu[0]], "b": np.array(D.names)[iu[1]],
        "model_a": D.labels[iu[0]], "model_b": D.labels[iu[1]],
        "same_model": D.labels[iu[0]] == D.labels[iu[1]],
        "jaccard": Jm[iu], "cosine": Cm[iu],
        "both_flat_frac": flat[iu], "excluded": ~valid})
    pairs["relation"] = [f"{a}-{b}" if a == b else "-".join(sorted([a, b]))
                         for a, b in zip(pairs.model_a, pairs.model_b)]
    save(pairs, "pairs_all.csv", out)

    live = pairs[~pairs.excluded]

    # ---- distributions per relation
    for metric in ("jaccard", "cosine"):
        out.head(f"{metric.upper()} DISTRIBUTIONS")
        rows = []
        for rel in sorted(live.relation.unique()):
            v = live[live.relation == rel][metric]
            d = describe(v, rel)
            if d:
                d["kind"] = "within" if len(set(rel.split("-"))) == 1 else "cross"
                rows.append(d)
        d = describe(live[live.same_model][metric], "ALL within")
        if d:
            d["kind"] = "within"; rows.append(d)
        d = describe(live[~live.same_model][metric], "ALL cross")
        if d:
            d["kind"] = "cross"; rows.append(d)
        tab = pd.DataFrame(rows)
        cols = ["group", "kind", "n_pairs", "mean", "sd", "sd_over_mean",
                "min", "q1", "median", "q3", "max"] + \
               [f"pairs_above_{t}" for t in EXTREME]
        out(tab[cols].to_string(index=False))
        save(tab, f"distribution_{metric}.csv", out)

    # ---- model matrix
    out.head("MODEL MATRIX (mean jaccard)")
    mat = pd.DataFrame(index=D.models, columns=D.models, dtype=float)
    for i in D.models:
        for j in D.models:
            m = (((live.model_a == i) & (live.model_b == j)) |
                 ((live.model_a == j) & (live.model_b == i)))
            if i == j:
                m = (live.model_a == i) & (live.model_b == i)
            mat.loc[i, j] = round(live[m].jaccard.mean(), 4) if m.any() else np.nan
    out(mat.to_string())
    out("\ndiagonal = within-model, off-diagonal = between two models")
    save(mat.reset_index().rename(columns={"index": "model"}), "model_matrix.csv", out)

    # ---- convergence ratio
    out.head("CONVERGENCE RATIO (within / cross)")
    out("how much more a model's agents resemble each other than outsiders")
    out("1.0 = no self-similarity at all;  higher = more monocultural\n")
    rows = []
    for mo in D.models:
        w = live[(live.model_a == mo) & (live.model_b == mo)].jaccard.mean()
        c = live[((live.model_a == mo) ^ (live.model_b == mo))].jaccard.mean()
        rows.append({"model": mo, "within": round(w, 4), "cross": round(c, 4),
                     "ratio": round(w / c, 3) if c else np.nan})
        out(f"  {mo:<8} within {w:.4f}  cross {c:.4f}  ratio {w/c:.3f}")
    save(pd.DataFrame(rows), "convergence_ratio.csv", out)

    # ---- cross-model near-duplicates
    out.head("CROSS-MODEL NEAR-DUPLICATES (jaccard > 0.7)")
    out("a high-scoring pair spanning two models means the strategy space")
    out("itself has an attractor, not that one model has a house style\n")
    cd = live[(~live.same_model) & (live.jaccard > 0.7)].sort_values("jaccard", ascending=False)
    if len(cd):
        out(cd[["a", "b", "jaccard", "cosine"]].head(20).to_string(index=False))
    else:
        out("  none -- every near-duplicate pair is within a single model")
    save(cd, "cross_model_near_duplicates.csv", out)

    # ---- nulls
    rng = np.random.default_rng(args.seed)
    for metric in ("jaccard", "cosine"):
        vals = (Jm if metric == "jaccard" else Cm)[iu]
        v = valid & np.isfinite(vals)
        out.head(f"PERMUTATION NULL -- {metric} ({args.iters:,} shuffles)")
        t = permute(vals, v, iu, D.labels, D.models, args.iters, rng, out, out)
        t["metric"] = metric
        save(t, f"null_{metric}.csv", out)
        if metric == "jaccard":
            null_tab = t

    # ---- per-agent typicality
    out.head("PER-AGENT TYPICALITY")
    out("each agent's mean overlap with the others from its own model")
    out("high = a central, representative agent;  low = an outlier\n")
    rows = []
    for i, nm in enumerate(D.names):
        if nm in excl:
            continue
        m = (D.labels == D.labels[i]) & keep
        m[i] = False
        if not m.any():
            continue
        rows.append({"agent": nm, "model": D.labels[i],
                     "typicality": round(float(np.nanmean(Jm[i][m])), 4),
                     "max_partner": round(float(np.nanmax(Jm[i][m])), 4),
                     "closest": np.array(D.names)[m][int(np.nanargmax(Jm[i][m]))]})
    typ = pd.DataFrame(rows).sort_values(["model", "typicality"], ascending=[True, False])
    for mo in D.models:
        s = typ[typ.model == mo]
        if len(s):
            out(f"  {mo:<8} range {s.typicality.min():.3f} - {s.typicality.max():.3f}"
                f"   most typical {s.iloc[0].agent}   least {s.iloc[-1].agent}")
    save(typ, "agent_typicality.csv", out)

    # ---- clustering
    cluster(D, Jm, keep, out)

    figure(lambda plt: fig_null(plt, null_tab), "null_test.png", out)
    figure(lambda plt: fig_dist(plt, live), "overlap_distribution.png", out)
    figure(lambda plt: fig_matrix(plt, mat), "model_matrix.png", out)
    figure(lambda plt: fig_typicality(plt, typ), "agent_typicality.png", out)
    return ts


def cluster(D, Jm, keep, out):
    """Hierarchical clustering on the similarity matrix.

    Better than thresholding by hand: merging the two most similar agents
    repeatedly records structure at every level of strictness, so the answer
    does not hinge on one arbitrary cutoff.
    """
    out.head("HOLDINGS CLUSTERS")
    try:
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform
    except ImportError:
        out("  scipy not installed -- skipping (pip3 install scipy)")
        return
    idx = np.where(keep)[0]
    if len(idx) < 3:
        out("  too few active agents")
        return
    S = np.nan_to_num(Jm[np.ix_(idx, idx)])
    Dm = 1.0 - S
    np.fill_diagonal(Dm, 0.0)
    Dm = (Dm + Dm.T) / 2
    Z = linkage(squareform(Dm, checks=False), method="average")

    rows = []
    for cut in (0.3, 0.4, 0.5, 0.6):
        lab = fcluster(Z, t=1.0 - cut, criterion="distance")
        sizes = Counter(lab)
        multi = {k: v for k, v in sizes.items() if v > 1}
        rows.append({"similarity_cut": cut, "clusters": len(multi),
                     "agents_clustered": sum(multi.values()),
                     "largest_cluster": max(multi.values()) if multi else 0,
                     "singletons": sum(1 for v in sizes.values() if v == 1)})
        out(f"  at similarity >= {cut}: {len(multi)} clusters covering "
            f"{sum(multi.values())} agents, largest {max(multi.values()) if multi else 0}, "
            f"{sum(1 for v in sizes.values() if v == 1)} singletons")
    save(pd.DataFrame(rows), "cluster_counts.csv", out)

    CUT = 0.3
    lab = fcluster(Z, t=1.0 - CUT, criterion="distance")
    memb = pd.DataFrame({"agent": np.array(D.names)[idx],
                         "model": D.labels[idx], "cluster": lab})
    big = [c for c, v in Counter(lab).items() if v > 1]
    out(f"\nmembership at similarity >= {CUT} ({len(big)} multi-agent clusters):")
    for c in sorted(big, key=lambda c: -Counter(lab)[c]):
        s = memb[memb.cluster == c]
        mix = Counter(s.model)
        out(f"  cluster {c}: {len(s)} agents  [{', '.join(f'{k} {v}' for k, v in mix.items())}]"
            f"  {' '.join(s.agent)}")
    save(memb, "cluster_membership.csv", out)

    mb = memb[memb.cluster.isin(big)]
    if len(mb):
        ct = pd.crosstab(mb.cluster, mb.model)
        out("\ncluster x model (multi-agent clusters only)")
        out(ct.to_string())
        pure = sum(1 for c in big if (ct.loc[c] > 0).sum() == 1)
        out(f"\n{pure} of {len(big)} clusters contain agents from only one model")
    else:
        ct = pd.DataFrame()
        out("\nno multi-agent clusters at this cut")
    if len(ct):
        save(ct.reset_index(), "cluster_by_model.csv", out)
    figure(lambda plt: fig_dendro(plt, Z, np.array(D.names)[idx], D.labels[idx], CUT),
           "dendrogram.png", out)


# --------------------------------------------------------------------- flow

def cmd_flow(D, out, args):
    out.head("POPULATION SIGNED FLOW")
    out("Do agents push the same direction at the same moment, or cancel out?")
    out("For one ticker in one bar: net = buys minus sells across the group,")
    out("gross = all trading ignoring direction. imbalance = |net| / gross.")
    out("1.0 = everyone traded the same way   0.0 = perfectly offsetting\n")
    out("Weighted by (m - 1), where m is how many agents traded that ticker-bar,")
    out("so a ticker-bar with 20 participants counts 19x a ticker-bar with 2.")
    out(f"Ticker-bars with fewer than {args.min_traders} traders are dropped: a")
    out("lone agent has nobody to disagree with and scores 1.0 automatically,")
    out("which measures how thinly trading is spread, not whether agents herd.\n")

    ind = [(f != 0).astype(np.float32) for f in D.flow]

    def sums(idx):
        net = D.flow[idx[0]].copy()
        gro = D.aflow[idx[0]].copy()
        cnt = ind[idx[0]].copy()
        for i in idx[1:]:
            net = net + D.flow[i]
            gro = gro + D.aflow[i]
            cnt = cnt + ind[i]
        return net, gro, cnt

    def imbalance(mask, want_parts=False):
        idx = np.where(mask)[0]
        if len(idx) < 2:
            return (np.nan, None) if want_parts else np.nan
        net, gro, cnt = sums(idx)
        nz = gro.nonzero()
        if not len(nz[0]):
            return (np.nan, None) if want_parts else np.nan
        g = np.asarray(gro[nz]).ravel()
        n = np.abs(np.asarray(net[nz]).ravel())
        m = np.asarray(cnt[nz]).ravel()
        keep = m >= args.min_traders
        if not keep.any():
            return (np.nan, None) if want_parts else np.nan
        w = (m[keep] - 1.0)
        val = float((w * (n[keep] / g[keep])).sum() / w.sum())
        if want_parts:
            return val, m
        return val

    # how much data survives the filter -- report it, do not hide it
    _, m_all = imbalance(np.ones(D.na, bool), want_parts=True)
    if m_all is not None:
        tot = len(m_all)
        out("ticker-bars by number of participating agents (whole population):")
        for lo, hi, lab in [(1, 1, "1 (dropped)"), (2, 2, "2"), (3, 5, "3-5"),
                            (6, 10, "6-10"), (11, 10**9, "11+")]:
            c = int(((m_all >= lo) & (m_all <= hi)).sum())
            out(f"    {lab:<12} {c:>9,}  ({c/tot:5.1%})")
        kept = int((m_all >= args.min_traders).sum())
        out(f"  {kept:,} of {tot:,} ticker-bars used ({kept/tot:.1%})\n")

    rows, rng = [], np.random.default_rng(args.seed)
    pooled = imbalance(np.ones(D.na, bool))
    out(f"  {'ALL':<8} observed {pooled:.4f}   (whole population; no null --")
    out("           shuffling everyone into one group reproduces it exactly)\n")

    for mo in D.models:
        m = D.labels == mo
        obs = imbalance(m)
        size = int(m.sum())
        nd = np.empty(args.flow_iters)
        for k in range(args.flow_iters):
            f = np.zeros(D.na, bool)
            f[rng.choice(D.na, size=size, replace=False)] = True
            nd[k] = imbalance(f)
        nd = nd[np.isfinite(nd)]
        p = (np.sum(nd >= obs) + 1) / (len(nd) + 1)
        gate = float(np.percentile(nd, 95))
        z = (obs - nd.mean()) / nd.std(ddof=1) if nd.std(ddof=1) > 0 else np.nan
        rows.append({"group": mo, "n_agents": size,
                     "observed_imbalance": round(float(obs), 4),
                     "null_mean": round(float(nd.mean()), 4),
                     "null_sd": round(float(nd.std(ddof=1)), 4),
                     "gate_95pct": round(gate, 4), "p_value": round(float(p), 5),
                     "z_score": round(float(z), 2), "passes": bool(obs > gate)})
        out(f"  {mo:<8} observed {obs:.4f}  null {nd.mean():.4f}  gate {gate:.4f}"
            f"  p {p:.5f}  z {z:5.2f}  {'PASS' if obs > gate else 'fail'}")

    rows.append({"group": "ALL", "n_agents": D.na,
                 "observed_imbalance": round(float(pooled), 4),
                 "null_mean": np.nan, "null_sd": np.nan, "gate_95pct": np.nan,
                 "p_value": np.nan, "z_score": np.nan, "passes": None})
    tab = pd.DataFrame(rows)
    tab["min_traders"] = args.min_traders
    save(tab, "flow_imbalance.csv", out)

    out("\nIf a synthetic random group is in the pool, compare against it: that")
    out("is what this metric reads when nothing links the agents' decisions.")

    # ---- per-bar series, same weighting
    out.head("FLOW IMBALANCE OVER TIME")
    series = {}
    for mo in D.models:
        idx = np.where(D.labels == mo)[0]
        if len(idx) < 2:
            continue
        net, gro, cnt = sums(idx)
        net = abs(net).tocsr(); gro = gro.tocsr(); cnt = cnt.tocsr()
        vals = np.full(D.nb, np.nan)
        for b in range(D.nb):
            g = gro.getrow(b)
            if g.nnz == 0:
                continue
            cols = g.indices
            gv = g.data
            nv = np.asarray(net.getrow(b).todense()).ravel()[cols]
            mv = np.asarray(cnt.getrow(b).todense()).ravel()[cols]
            k = mv >= args.min_traders
            if not k.any():
                continue
            w = mv[k] - 1.0
            if w.sum() > 0:
                vals[b] = float((w * (nv[k] / gv[k])).sum() / w.sum())
        series[mo] = vals
        v = vals[np.isfinite(vals)]
        if len(v):
            out(f"  {mo:<8} mean {v.mean():.4f}  sd {v.std():.4f}  "
                f"min {v.min():.4f}  max {v.max():.4f}  "
                f"bars measurable: {len(v):,}/{D.nb:,}")
    ft = pd.DataFrame({"timestamp": D.bars, **series})
    save(ft, "flow_timeseries.csv", out)
    figure(lambda plt: fig_flow(plt, tab), "flow_imbalance.png", out)
    return ft


# ----------------------------------------------------------------- crowding

def cmd_crowding(D, out, args):
    out.head("POPULATION CROWDING")
    out("how piled-up the whole population is, rather than pair by pair")
    out("effective names = distinct stocks the population really spans,")
    out("after duplicate holdings are discounted\n")

    keep = D.active >= args.min_active
    rows, series = [], {}
    for mo in list(D.models) + ["ALL"]:
        m = keep if mo == "ALL" else (keep & (D.labels == mo))
        idx = np.where(m)[0]
        if len(idx) < 2:
            continue
        top1 = np.zeros(D.nb); top5 = np.zeros(D.nb); eff = np.full(D.nb, np.nan)
        counts_total = np.zeros(D.nt)
        for b0, b1 in D.chunks():
            H = np.stack([D.hold[i][b0:b1].toarray() for i in idx])
            hb = H.sum(axis=0)
            counts_total += hb.sum(axis=0)
            tot = hb.sum(axis=1)
            top1[b0:b1] = hb.max(axis=1) / len(idx)
            top5[b0:b1] = np.sort(hb, axis=1)[:, -5:].sum(axis=1) / np.where(tot > 0, tot, 1)
            p = hb / np.where(tot[:, None] > 0, tot[:, None], 1)
            hhi = (p ** 2).sum(axis=1)
            eff[b0:b1] = np.where(hhi > 0, 1.0 / np.where(hhi > 0, hhi, 1), np.nan)
        rows.append({"group": mo, "n_agents": len(idx),
                     "mean_top1_share": round(float(top1.mean()), 4),
                     "max_top1_share": round(float(top1.max()), 4),
                     "mean_top5_share": round(float(np.nanmean(top5)), 4),
                     "mean_effective_names": round(float(np.nanmean(eff)), 1),
                     "min_effective_names": round(float(np.nanmin(eff)), 1)})
        series[mo] = eff
        out(f"  {mo:<8} top1 {top1.mean():.3f} (peak {top1.max():.3f})  "
            f"top5 {np.nanmean(top5):.3f}  effective names {np.nanmean(eff):.1f}"
            f" (min {np.nanmin(eff):.1f})")
        top = np.argsort(-counts_total)[:10]
        D.__dict__.setdefault("top_names", {})[mo] = [
            (D.tickers[t], int(counts_total[t])) for t in top]
    save(pd.DataFrame(rows), "crowding.csv", out)
    save(pd.DataFrame({"timestamp": D.bars, **series}), "crowding_timeseries.csv", out)

    out.head("MOST-CROWDED TICKERS")
    out("ranked by total agent-bars held\n")
    tn = D.__dict__.get("top_names", {})
    for mo in list(D.models) + ["ALL"]:
        if mo in tn:
            out(f"  {mo:<8} {', '.join(t for t, _ in tn[mo])}")
    if len(D.models) > 1:
        sets = {m: set(t for t, _ in tn[m]) for m in D.models if m in tn}
        if len(sets) > 1:
            common = set.intersection(*sets.values())
            out(f"\n  in every model's top ten: {', '.join(sorted(common)) if common else 'none'}")
            out("  (heavy overlap means one universal attractor;")
            out("   disjoint lists mean separate monocultures per model)")
    save(pd.DataFrame([{"group": m, "rank": i + 1, "ticker": t, "agent_bars": c}
                       for m, lst in tn.items() for i, (t, c) in enumerate(lst)]),
         "top_tickers.csv", out)
    figure(lambda plt: fig_crowd(plt, D, series), "crowding_over_time.png", out)


# --------------------------------------------------------------------- time

def cmd_time(D, out, args, ts=None, ft=None):
    out.head("TIME AND MARKET REGIME")
    out("every number elsewhere in this report is a yearly average.")
    out("this section splits the year, because an average hides the episodes.\n")

    tag, daily, ret, vol = D.regimes()
    counts = Counter(tag)
    out(f"  regime bars: {', '.join(f'{k} {v:,}' for k, v in sorted(counts.items()))}")
    out("  (terciles of 5-day realised volatility of an equal-weighted index)")

    if ts is None:
        p = os.path.join(CSVDIR, "overlap_timeseries.csv")
        if os.path.exists(p):
            ts = pd.read_csv(p)
    if ts is not None:
        ts = ts.copy()
        ts["regime"] = tag
        ts["month"] = pd.to_datetime(ts["timestamp"]).dt.to_period("M").astype(str)
        cols = [c for c in ts.columns if c not in ("timestamp", "regime", "month")]

        out.head("OVERLAP BY MARKET REGIME")
        g = ts.groupby("regime")[cols].mean().round(4)
        out(g.to_string())
        save(g.reset_index(), "overlap_by_regime.csv", out)

        out.head("OVERLAP BY MONTH")
        m = ts.groupby("month")[cols].mean().round(4)
        out(m.to_string())
        save(m.reset_index(), "overlap_by_month.csv", out)
        save(ts, "overlap_timeseries.csv", out)

        out.head("PEAK CROWDING BARS")
        for c in cols:
            if c == "cross" or ts[c].isna().all():
                continue
            hi = ts.nlargest(10, c)
            out(f"\n  {c}: top 10 bars, mean {ts[c].mean():.4f}, peak {ts[c].max():.4f}")
            out(f"    dates: {', '.join(sorted(set(str(x)[:10] for x in hi.timestamp)))}")
            out(f"    regimes: {dict(Counter(hi.regime))}")

    if ft is None:
        p = os.path.join(CSVDIR, "flow_timeseries.csv")
        if os.path.exists(p):
            ft = pd.read_csv(p)
    if ft is not None:
        ft = ft.copy()
        ft["regime"] = tag
        cols = [c for c in ft.columns if c not in ("timestamp", "regime")]
        out.head("FLOW IMBALANCE BY MARKET REGIME")
        out("if agents herd harder under stress, it shows here and nowhere else\n")
        g = ft.groupby("regime")[cols].mean().round(4)
        out(g.to_string())
        save(g.reset_index(), "flow_by_regime.csv", out)

    # simultaneous flatness
    out.head("SIMULTANEOUS RISK-OFF")
    out("fraction of the population holding nothing at all, per bar")
    out("a spike means many agents went to cash at the same moment\n")
    keep = D.active >= args.min_active
    rows = {}
    for mo in D.models:
        idx = np.where(keep & (D.labels == mo))[0]
        if not len(idx):
            continue
        fl = np.zeros(D.nb)
        for b0, b1 in D.chunks():
            H = np.stack([D.hold[i][b0:b1].toarray() for i in idx])
            fl[b0:b1] = (H.sum(axis=2) == 0).mean(axis=0)
        rows[mo] = fl
        out(f"  {mo:<8} mean {fl.mean():.4f}  max {fl.max():.4f}  "
            f"bars with >50% flat: {(fl > 0.5).sum()}")
    if rows:
        save(pd.DataFrame({"timestamp": D.bars, **rows}), "flat_fraction_timeseries.csv", out)
    if ts is not None:
        figure(lambda plt: fig_time(plt, ts, cols if ft is None else None),
               "overlap_over_time.png", out)


# ----------------------------------------------------------------- strategy

KEYWORDS = {
    "momentum": [r"\bmomentum\b", r"\bmom\b", r"roc\b", r"rate_of_change"],
    "mean_reversion": [r"mean[_ ]?revers", r"\bzscore\b", r"z_score", r"bollinger",
                       r"\brsi\b", r"oversold"],
    "trend": [r"\btrend\b", r"\bsma\b", r"\bema\b", r"moving[_ ]average",
              r"crossover", r"\bmacd\b"],
    "volatility": [r"\bvolatilit", r"\batr\b", r"inverse[_ ]vol", r"vol_target"],
    "breakout": [r"breakout", r"donchian", r"channel"],
    "buy_hold": [r"buy[_ ]and[_ ]hold", r"equal[_ ]weight"],
}


def cmd_strategy(D, out, args):
    out.head("STRATEGY CLASSIFICATION")
    out("two independent views: how each agent traded, and what its code says.")
    out("neither is authoritative -- the keyword scan counts words, and any")
    out("behavioural label depends on where the boundaries are drawn.\n")

    S = pd.DataFrame(D.summary)
    feats = ["mean_names_held", "median_hold_bars", "orders_per_bar",
             "flat_frac", "tickers_ever_held", "concentration_hhi"]
    feats = [f for f in feats if f in S.columns]

    out("BEHAVIOURAL SPREAD PER MODEL")
    g = S.groupby("model")[feats].agg(["mean", "std"]).round(3)
    out(g.to_string())
    save(g.reset_index(), "behavioural_spread.csv", out)

    # behavioural clustering -- no hand-picked thresholds
    try:
        from scipy.cluster.hierarchy import linkage, fcluster
        X = S[feats].to_numpy(dtype=float)
        X = np.log1p(np.clip(X, 0, None))
        X = (X - X.mean(axis=0)) / np.where(X.std(axis=0) > 0, X.std(axis=0), 1)
        Z = linkage(X, method="ward")
        S["behaviour_cluster"] = fcluster(Z, t=4, criterion="maxclust")
        out.head("BEHAVIOUR CLUSTERS x MODEL")
        out("clusters found in the behaviour numbers themselves, not by cutoffs\n")
        ct = pd.crosstab(S.behaviour_cluster, S.model)
        out(ct.to_string())
        save(ct.reset_index(), "behaviour_clusters.csv", out)
        out("\ncluster profiles (median):")
        out(S.groupby("behaviour_cluster")[feats].median().round(2).to_string())
    except ImportError:
        out("  scipy missing -- behavioural clustering skipped")

    # source keywords
    def src(name):
        p = os.path.join(AGENT_DIR, name + ".py")
        return open(p, encoding="utf-8", errors="replace").read().lower() \
            if os.path.exists(p) else ""

    rows = []
    for nm in D.names:
        s = src(nm)
        hits = {k: sum(bool(re.search(p, s)) for p in pats)
                for k, pats in KEYWORDS.items()}
        top = max(hits, key=hits.get) if any(hits.values()) else None
        rows.append({"agent": nm, "model": label_of(nm), "source_found": bool(s),
                     "source_family": top if top and hits[top] else "unlabelled",
                     **{f"kw_{k}": v for k, v in hits.items()}})
    K = pd.DataFrame(rows)
    out.head("SOURCE-KEYWORD FAMILY x MODEL")
    out("crude: counts word occurrences in the generated code\n")
    out(pd.crosstab(K.source_family, K.model).to_string())
    missing = (~K.source_found).sum()
    if missing:
        out(f"\n  source not found for {missing} agents (expected {AGENT_DIR}/<name>.py)")
    merged = S.merge(K, on=["agent", "model"], how="left")
    save(merged, "strategy_per_agent.csv", out)

    # do holdings clusters line up with behaviour clusters?
    p = os.path.join(CSVDIR, "cluster_membership.csv")
    if os.path.exists(p) and "behaviour_cluster" in S.columns:
        hc = pd.read_csv(p)
        j = S.merge(hc[["agent", "cluster"]], on="agent", how="inner")
        if len(j):
            out.head("HOLDINGS CLUSTER x BEHAVIOUR CLUSTER")
            out("agents grouped by holdings but NOT by behaviour are running the")
            out("same strategy and landing on the same names. grouped by both")
            out("means genuinely distinct strategy families.\n")
            out(pd.crosstab(j.cluster, j.behaviour_cluster).to_string())
    figure(lambda plt: fig_behaviour(plt, S), "behaviour_scatter.png", out)


# ------------------------------------------------------------------ returns

def cmd_returns(D, out, args):
    out.head("RETURNS -- SIDE ANALYSIS, NOT A HEADLINE")
    out("the contract states zero commission and zero slippage, so these are")
    out("gross by construction. an agent that trades constantly looks far")
    out("better here than it would anywhere real. kept because it is")
    out("interesting, excluded from every conclusion.\n")
    S = pd.DataFrame(D.summary)
    cols = [c for c in ["total_return", "benchmark_return", "excess_vs_benchmark",
                        "sharpe", "ann_vol", "max_drawdown", "shares_traded",
                        "net_ret_5mils", "net_ret_10mils", "net_ret_20mils"]
            if c in S.columns]
    out(S.groupby("model")[cols].mean().round(4).to_string())
    if "net_ret_10mils" in S.columns:
        neg = (S.net_ret_10mils < 0).sum()
        out(f"\n  {neg}/{len(S)} agents lose money at one cent per share traded")
    save(S, "agent_summary.csv", out)
    figure(lambda plt: fig_returns(plt, S), "cost_sensitivity.png", out)


# ------------------------------------------------------------------ figures

C = {"claude": "#2a78d6", "gemini": "#eb6834", "gpt": "#1baf7a",
     "random": "#B4B2A9", "popular": "#c9a227",
     "unknown": "#898781", "cross": "#B4B2A9", "POOLED": "#4a3aa7"}


def fig_null(plt, t):
    t = t[t.group != "POOLED"]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(t))
    ax.bar(x, t.observed, color=[C.get(g, "#888") for g in t.group], width=.55,
           label="observed")
    for i, r in enumerate(t.itertuples()):
        ax.plot([i - .32, i + .32], [r.gate_95pct] * 2, color="#d03b3b", lw=2)
        ax.text(i, r.observed + .006, f"p={r.p_value:g}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(t.group)
    ax.set_ylabel("mean within-model jaccard")
    ax.set_title("Observed overlap vs the 95% chance threshold (red)")
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def fig_dist(plt, live):
    fig, ax = plt.subplots(figsize=(7, 4))
    for rel in sorted(live.relation.unique()):
        v = live[live.relation == rel].jaccard
        within = len(set(rel.split("-"))) == 1
        ax.hist(v, bins=np.linspace(0, 1, 41), histtype="step", density=True,
                lw=2 if within else 1, ls="-" if within else "--",
                label=f"{rel} ({len(v)})")
    ax.set_xlabel("jaccard"); ax.set_ylabel("density")
    ax.set_title("Pair overlap distributions (solid = within model)")
    ax.legend(fontsize=8); ax.spines[["top", "right"]].set_visible(False)
    return fig


def fig_matrix(plt, mat):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    m = mat.to_numpy(dtype=float)
    im = ax.imshow(m, cmap="Blues", vmin=0)
    ax.set_xticks(range(len(mat))); ax.set_xticklabels(mat.columns)
    ax.set_yticks(range(len(mat))); ax.set_yticklabels(mat.index)
    for i in range(len(mat)):
        for j in range(len(mat)):
            if np.isfinite(m[i, j]):
                ax.text(j, i, f"{m[i,j]:.3f}", ha="center", va="center",
                        color="white" if m[i, j] > np.nanmax(m) * .6 else "black")
    ax.set_title("Mean jaccard by model pair")
    fig.colorbar(im, ax=ax, shrink=.8)
    return fig


def fig_typicality(plt, typ):
    fig, ax = plt.subplots(figsize=(7, 4))
    for mo in typ.model.unique():
        s = typ[typ.model == mo].sort_values("typicality").reset_index()
        ax.plot(range(len(s)), s.typicality, "o-", ms=4, color=C.get(mo, "#888"),
                label=mo)
    ax.set_xlabel("agents, ranked within model")
    ax.set_ylabel("mean overlap with own model")
    ax.set_title("Core vs outlier agents")
    ax.legend(); ax.spines[["top", "right"]].set_visible(False)
    return fig


def fig_dendro(plt, Z, names, labels, cut=0.3):
    from scipy.cluster.hierarchy import dendrogram
    fig, ax = plt.subplots(figsize=(9, 4.5))
    d = dendrogram(Z, labels=names, ax=ax, color_threshold=1.0 - cut,
                   leaf_font_size=6)
    lut = {n: labels[i] for i, n in enumerate(names)}
    for t in ax.get_xticklabels():
        t.set_color(C.get(lut.get(t.get_text(), ""), "#333"))
    ax.set_ylabel("1 - jaccard")
    ax.set_title("Holdings similarity tree (colour = model)")
    return fig


def fig_flow(plt, t):
    t = t[t.group != "ALL"].dropna(subset=["gate_95pct"])
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(t))
    ax.bar(x, t.observed_imbalance, color=[C.get(g, "#888") for g in t.group], width=.55)
    for i, r in enumerate(t.itertuples()):
        ax.plot([i - .32, i + .32], [r.gate_95pct] * 2, color="#d03b3b", lw=2)
    ax.set_xticks(x); ax.set_xticklabels(t.group)
    ax.set_ylim(0, 1)
    ax.set_ylabel("|net| / gross")
    ax.set_title("Directional flow imbalance vs chance threshold (red)")
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def fig_crowd(plt, D, series):
    fig, ax = plt.subplots(figsize=(9, 3.6))
    t = pd.to_datetime(pd.Series(D.bars))
    for k, v in series.items():
        ax.plot(t, pd.Series(v).rolling(65, min_periods=1).mean(),
                color=C.get(k, "#555"), lw=1.2, label=k)
    ax.set_ylabel("effective distinct names")
    ax.set_title("How many stocks the population really spans (5-day mean)")
    ax.legend(fontsize=8); ax.spines[["top", "right"]].set_visible(False)
    return fig


def fig_time(plt, ts, cols=None):
    cols = cols or [c for c in ts.columns
                    if c not in ("timestamp", "regime", "month")]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    t = pd.to_datetime(ts["timestamp"])
    for c in cols:
        ax.plot(t, ts[c].rolling(65, min_periods=1).mean(),
                color=C.get(c, "#555"), lw=1.2, label=c)
    ax.set_ylabel("mean pair jaccard")
    ax.set_title("Overlap through the year (5-day mean)")
    ax.legend(fontsize=8); ax.spines[["top", "right"]].set_visible(False)
    return fig


def fig_behaviour(plt, S):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for mo in S.model.unique():
        s = S[S.model == mo]
        ax.scatter(s.median_hold_bars.clip(lower=.5), s.mean_names_held.clip(lower=.5),
                   s=34, alpha=.75, color=C.get(mo, "#888"), label=mo)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("median holding period (bars)")
    ax.set_ylabel("mean names held")
    ax.set_title("Behaviour space")
    ax.legend(); ax.spines[["top", "right"]].set_visible(False)
    return fig


def fig_returns(plt, S):
    cols = [c for c in ["total_return", "net_ret_5mils", "net_ret_10mils",
                        "net_ret_20mils"] if c in S.columns]
    if len(cols) < 2:
        return None
    x = [0, .5, 1, 2][:len(cols)]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for _, r in S.iterrows():
        ax.plot(x, [r[c] for c in cols], lw=.8, alpha=.5,
                color=C.get(r["model"], "#888"))
    ax.axhline(0, color="#d03b3b", lw=1)
    ax.set_xlabel("assumed cost, cents per share")
    ax.set_ylabel("return")
    ax.set_title("Where each agent's edge dies")
    ax.spines[["top", "right"]].set_visible(False)
    return fig


# --------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["all", "overlap", "flow", "time",
                                     "crowding", "strategy", "returns"])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--from", dest="start", default="2025-01-01")
    ap.add_argument("--iters", type=int, default=ITERS)
    ap.add_argument("--flow-iters", type=int, default=FLOW_ITERS, dest="flow_iters")
    ap.add_argument("--min-active", type=float, default=0.05, dest="min_active")
    ap.add_argument("--min-traders", type=int, default=2, dest="min_traders",
                    help="drop ticker-bars with fewer than this many agents "
                         "trading; a lone trader scores 1.0 by construction")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(CSVDIR, exist_ok=True)
    os.makedirs(FIGDIR, exist_ok=True)
    out = Tee(os.path.join(OUT, "summary.txt"))
    out.head(f"REPORT -- {args.mode} -- {pd.Timestamp.now():%Y-%m-%d %H:%M}")

    D = Data(args.runs, args.start, out)
    ts = ft = None
    if args.mode in ("all", "overlap"):
        ts = cmd_overlap(D, out, args)
    if args.mode in ("all", "flow"):
        ft = cmd_flow(D, out, args)
    if args.mode in ("all", "crowding"):
        cmd_crowding(D, out, args)
    if args.mode in ("all", "strategy"):
        cmd_strategy(D, out, args)
    if args.mode in ("all", "time"):
        cmd_time(D, out, args, ts, ft)
    if args.mode in ("all", "returns"):
        cmd_returns(D, out, args)

    out.head("DONE")
    out(f"  {OUT}/summary.txt   everything printed above")
    out(f"  {OUT}/csv/          {len(os.listdir(CSVDIR))} tables")
    out(f"  {OUT}/figures/      {len(os.listdir(FIGDIR))} plots")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
