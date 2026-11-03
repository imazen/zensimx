"""E36 revised plan R5 (method review M4): a linear-probe transfer matrix over human training sources.

For each training set (a human source, a distortion tier of one, or a dataset leg's TRAIN rows), fit a ridge head on
F3's 492 direct input features and score the EVAL MOS collections: SROCC per collection with a whole-reference 95%
interval (dataset_eval.mos_interval, the canonical `panel rank-interval`), and within-reference SROCC. A probe is a
cheap stand-in for a retrain: it shows which labels, mapped linearly from the same features, rank other collections'
images well. It does not predict what a jointly trained network does; experiment 5's retrains are the check on it.

Preprocessing per training set: features scaled by the set's median and interquartile range and clipped to +-6;
targets z-scored within each source (pooled sets z-score each source first). Ridge strength from 5-fold
whole-reference cross-validation inside the training set (grid 1e-2 .. 1e5, mean fold SROCC).

Three objectives: `global` regresses the z-scored targets directly; `within` first subtracts each reference's mean
from both targets and features, so only differences inside a reference teach the head; `rank` fits a linear
pairwise-logistic ranker on pairs inside each reference (up to 300 per reference, ties skipped), which, like the
trainer's human leg, sees only orderings inside one reference and is blind to each source's label scale.

    probe_transfer.py OUT.json        (EVAL rows and features from ~/tmp/e36test/eval; ZEN_PANEL_BIN set)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import dataset_eval as de  # noqa: E402
from e36_exp56.m3_rows import TIER_C, TIER_N  # noqa: E402

ROOT = Path("/var/tmp/byc-fchroma-20261029/r3/w100/root")
HUMAN = ROOT / "wide/main/real/human_all_fit.parquet"
BANK = Path("/var/tmp/rev4-featbank/bank")
EVAL = Path.home() / "tmp/e36test/eval"
LEGS = ["mdid", "live-md", "cid-iq", "cid-oa", "irccyn-ivc", "colourlab-rating-variability-2022"]
SOURCE = {"kadid_train": "kadid", "kadid_select": "kadid", "tid2013": "tid2013", "konfig_train": "konfig",
          "konfig_val": "konfig", "cid22_a25": "cid22_a25"}
COMMON = ["iqa-odi", "nits-iqa", "vcl-fer"]


def f3_columns():
    argv = json.loads((Path.home() / "tmp/e36test/arms/p30L-s1.cmd.json").read_text())["argv"]
    return [f"f{c}" for c in argv[argv.index("--columns") + 1].split(",")]


def zscore(y):
    return (y - y.mean()) / y.std()


def demean(A, groups):
    out = np.array(A, dtype=float, copy=True)
    for g in np.unique(groups):
        m = groups == g
        out[m] -= out[m].mean(axis=0)
    return out


def fit_ridge(X, y, groups, within=False):
    med = np.median(X, axis=0)
    iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
    iqr[iqr <= 1e-12] = 1.0
    scale = lambda A: np.clip((A - med) / iqr, -6, 6)  # noqa: E731
    Xs = scale(X)
    mu = Xs.mean(axis=0)
    Xc = Xs - mu
    if within:
        Xc, y = demean(Xc, groups), demean(y, groups)
    uniq = np.unique(groups)
    rng = np.random.default_rng(20261103)
    fold = dict(zip(uniq, rng.permutation(len(uniq)) % 5))
    f = np.array([fold[g] for g in groups])
    best = None
    for alpha in 10.0 ** np.arange(-2, 6):
        s = []
        for k in range(5):
            tr, te = f != k, f == k
            if te.sum() < 3:
                continue
            A = Xc[tr]
            w = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ (y[tr] - y[tr].mean()))
            s.append(spearmanr(Xc[te] @ w, y[te]).statistic)
        m = float(np.mean(s))
        if best is None or m > best[0]:
            best = (m, alpha)
    w = np.linalg.solve(Xc.T @ Xc + best[1] * np.eye(Xc.shape[1]), Xc.T @ (y - y.mean()))
    return (lambda A: (scale(A) - mu) @ w), dict(alpha=best[1], cv_srocc=best[0], rows=len(y), refs=len(uniq))


def fit_rank(X, y, groups, per_ref=300):
    from scipy.optimize import minimize
    from scipy.special import expit
    med = np.median(X, axis=0)
    iqr = np.subtract(*np.percentile(X, [75, 25], axis=0))
    iqr[iqr <= 1e-12] = 1.0
    scale = lambda A: np.clip((A - med) / iqr, -6, 6)  # noqa: E731
    Xs = scale(X)
    rng = np.random.default_rng(20261103)
    uniq = np.unique(groups)
    pi, pj, pg = [], [], []
    for g in uniq:
        ii = np.flatnonzero(groups == g)
        a, b = np.triu_indices(len(ii), 1)
        keep = y[ii[a]] != y[ii[b]]
        a, b = a[keep], b[keep]
        if len(a) > per_ref:
            sel = rng.choice(len(a), per_ref, replace=False)
            a, b = a[sel], b[sel]
        hi = np.where(y[ii[a]] > y[ii[b]], ii[a], ii[b])
        lo = np.where(y[ii[a]] > y[ii[b]], ii[b], ii[a])
        pi.append(hi); pj.append(lo); pg.append(np.full(len(hi), g, dtype=object))
    pi, pj, pg = np.concatenate(pi), np.concatenate(pj), np.concatenate(pg)
    D = Xs[pi] - Xs[pj]

    def solve(Dm, c):
        lam = c * len(Dm)
        f = lambda w: (np.logaddexp(0, -(Dm @ w)).sum() + 0.5 * lam * w @ w,  # noqa: E731
                       -(Dm.T @ expit(-(Dm @ w))) + lam * w)
        return minimize(f, np.zeros(Dm.shape[1]), jac=True, method="L-BFGS-B").x

    fold = dict(zip(uniq, rng.permutation(len(uniq)) % 5))
    f_row = np.array([fold[g] for g in groups])
    f_pair = np.array([fold[g] for g in pg])
    best = None
    for c in 10.0 ** np.arange(-5, 1):
        sc = []
        for k in range(5):
            te = f_row == k
            if te.sum() < 3:
                continue
            w = solve(D[f_pair != k], c)
            sc.append(spearmanr(demean(Xs[te] @ w, groups[te]), demean(y[te], groups[te])).statistic)
        m = float(np.mean(sc))
        if best is None or m > best[0]:
            best = (m, c)
    w = solve(D, best[1])
    return (lambda A: scale(A) @ w), dict(alpha=best[1], cv_srocc=best[0], rows=len(y), refs=len(uniq), pairs=len(D))


def training_sets(cols):
    h = pq.read_table(HUMAN, columns=["ref_basename", "human_score", *cols]).to_pandas()
    keys = pq.read_table(HUMAN.with_suffix(".keys.parquet")).to_pandas()
    bk = pd.concat([pq.read_table(BANK / s / "keys.parquet", columns=["pair_key", "codec"]).to_pandas().assign(member_set=s)
                    for s in sorted(keys.member_set.unique())])
    k = keys.merge(bk, on=["member_set", "pair_key"], how="left", validate="many_to_one")
    assert len(k) == len(h) and k.codec.notna().all()
    h["source"] = k.member_set.map(SOURCE).to_numpy()
    h["group"] = (k.member_set.map(SOURCE) + "/" + k.ref_basename).to_numpy()
    tier = k.codec.map(lambda c: "C" if c in TIER_C else "N" if c in TIER_N else "S").to_numpy()
    sets = {}
    for s in ["kadid", "tid2013", "konfig", "cid22_a25"]:
        m = h.source.to_numpy() == s
        sets[s] = m
        if s in ("kadid", "tid2013"):
            sets[f"{s} C"] = m & (tier == "C")
            sets[f"{s} C+N"] = m & (tier != "S")
            sets[f"{s} S"] = m & (tier == "S")
    sets["human, all four"] = np.ones(len(h), bool)
    sets["human C (R2 m3c)"] = tier == "C"
    sets["human C+N (R2 m3cn)"] = tier != "S"
    out = {}
    for name, m in sets.items():
        d = h[m]
        y = np.concatenate([zscore(g.human_score.to_numpy(float)) for _, g in d.groupby("source", sort=False)])
        d = pd.concat([g for _, g in d.groupby("source", sort=False)])
        out[name] = (d[cols].to_numpy(float), y, d.group.to_numpy())
    for c in LEGS:
        d = pq.read_table(ROOT / "datasets" / f"{c}.parquet", columns=["ref_basename", "human_score", *cols]).to_pandas()
        out[f"leg {c}"] = (d[cols].to_numpy(float), zscore(d.human_score.to_numpy(float)), d.ref_basename.to_numpy())
    return out


def main(dest):
    cols = f3_columns()
    rows = [r for r in de.read_eval_rows(EVAL, "e36-r5-probe") if r["kind"] == "mos"]
    fe = pq.read_table(EVAL / "features.parquet", columns=["ref_basename", *cols]).to_pandas().set_index("ref_basename")
    fe = fe[~fe.index.duplicated()]
    Xe = fe.loc[[r["a"] for r in rows], cols].to_numpy(float)
    colls = sorted({r["collection"] for r in rows})
    result = {}
    scores = de.load_scores(EVAL)
    for s0 in range(3):  # the experiment 6 walk control, for scale
        m = f"x6_walk_s{s0}"
        per = {c: spearmanr([scores[m][r["a"]] for r in rows if r["collection"] == c],
                            [r["target"] for r in rows if r["collection"] == c]).statistic for c in colls}
        print(f"{'F3 walk control s' + str(s0):34s} mean {np.mean(list(per.values())):.3f} common-3 "
              f"{np.mean([per[c] for c in COMMON]):.3f}", flush=True)
    for objective, (name, (X, y, g)) in [(o, t) for o in ("global", "within", "rank")
                                         for t in training_sets(cols).items()]:
        if objective == "rank":
            predict, info = fit_rank(X, y, g)
        else:
            predict, info = fit_ridge(X, y, g, within=objective == "within")
        name = f"{objective}: {name}"
        p = predict(Xe)
        s = {r["a"]: float(v) for r, v in zip(rows, p)}
        per = {}
        for c in colls:
            rr = [r for r in rows if r["collection"] == c]
            ci, n = de.mos_interval(rr, s)
            per[c] = dict(srocc=ci["signed"]["estimate"], lo=ci["signed"]["lo"], hi=ci["signed"]["hi"],
                          within=de.within(rr, s), n=n)
        trained = {c for c in colls if name == f"leg {c}"}
        held = [c for c in colls if c not in trained]
        result[name] = dict(fit=info, per=per, mean_heldout=float(np.mean([per[c]["srocc"] for c in held])),
                            mean_common3=float(np.mean([per[c]["srocc"] for c in COMMON])))
        print(f"{name:34s} rows {info['rows']:5d} alpha {info['alpha']:g} cv {info['cv_srocc']:.3f} | held-out "
              f"{result[name]['mean_heldout']:.3f} common-3 {result[name]['mean_common3']:.3f} | "
              + " ".join(f"{c[:8]} {per[c]['srocc']:.3f}" for c in colls), flush=True)
    Path(dest).write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
