#!/usr/bin/env python3
"""Corruption-head theory tests T1-T5 (pre-registered 2026-09-06).

Pre-registration: `docs/PLAN_CORRHEAD_THEORIES_2026-09-06.md`. Read it before
reading any number here — the split, the metric definitions, the matched-
operating-point rule and the falsification criteria are all fixed there.

This is the STUDY DRIVER, not a second trainer. Every piece of model machinery
comes from the owner `train_corruption_head.py` (`load_X`, `make_classifier`,
`assemble_dataset`); every statistic comes from sklearn or numpy group-bys.
Nothing here re-implements a fit or a metric.

The split is not re-derived: it is READ from the incumbent's own `split.tsv`,
so an RNG-consumption-order difference cannot silently move a source between
folds between arms.

  corrhead_theories.py prep            # cache features + D's dial + the frozen split
  corrhead_theories.py t1 t2 t3 t4 t5  # run tests (any subset, in any order)
"""
import argparse, json, os, struct, subprocess, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_corruption_head import load_X, make_classifier  # THE owner

# Every path below is env-overridable so this driver can be pointed at a
# DIFFERENT extraction era's tables without a second copy of the driver
# (rev2 refit lane, 2026-09-06). Omitting the env vars leaves the rev1
# invocation byte-identical to the one that produced the T1-T9 record.
#
# CORRTH_SPLIT_ROOT is separate from CORRTH_ROOT on purpose: the FROZEN split
# and its parity `metrics.json` live with the INCUMBENT, and a re-extraction at
# another era must reuse them (the split keys -- ref_id / severe/<source_id> /
# ladder/<image_id> -- are era-invariant), or the two eras are not paired.
ROOT = os.environ.get("CORRTH_ROOT",
                      "/mnt/v/output/zensim/corruption-head-2026-09-05")
SPLIT_ROOT = os.environ.get("CORRTH_SPLIT_ROOT", ROOT)
OUT = os.environ.get("CORRTH_OUT", f"{ROOT}/theories")
LADDER = os.environ.get(
    "CORRTH_LADDER",
    "/mnt/v/output/zensim/ladder-2026-09-05/instruments/"
    "dial_grid_372col_ladder.parquet")
CORPUS = os.environ.get("CORRTH_CORPUS",
                        f"{ROOT}/im26_corruption_372_postC.parquet")
NEGRICH = os.environ.get("CORRTH_NEGRICH", f"{ROOT}/negrich_372_postC.parquet")
GATE = os.environ.get("CORRTH_GATE",
                      f"{ROOT}/corruption_grid_372col_postC_2026-09-05.parquet")
DBAKE = os.environ.get("CORRTH_DBAKE", os.path.expanduser(
    "~/work/zen/zensimx/zensim/weights/"
    "d_sdr_add156_id100_negrich_dial_2026-09-05.bin"))
FWD = os.environ.get("CORRTH_FWD", os.path.expanduser(
    "~/work/zen/zensimx/target/release/predict_features_with_bake"))
CACHE = os.environ.get("CORRTH_CACHE", f"{OUT}/dataset_rev1.npz")
GATE_CACHE = os.environ.get("CORRTH_GATE_CACHE", f"{OUT}/gate_rev1.npz")
NFEAT_SLICE = 228          # d228: f0..f227, free at D's V1PoolsMode::Peaks walk
FP_TARGETS = (0.0025, 0.005, 0.01, 0.05)
QBANDS = (("q<50", -1e9, 50.0), ("50-85", 50.0, 85.0),
          ("85-95", 85.0, 95.0), ("q>=95", 95.0, 1e9))

_t0 = time.time()


def log(msg):
    line = f"[{time.time()-_t0:7.1f}s] {msg}"
    print(line, flush=True)
    with open(os.path.expanduser("~/tmp/corrtheories_progress.log"), "a") as f:
        f.write(line + "\n")


# --------------------------------------------------------------------------
# prep
# --------------------------------------------------------------------------
def dial_scores(X372, tag):
    """Profile D's dial on pre-extracted 372-feature rows.

    Goes through `predict_features_with_bake` — the owner for "forward a bake
    over feature rows" — with `--bake-post raw`, which is the post-spline value
    without the [0,100] clamp, i.e. the dial the 2026-09-05 record quotes
    (its flagged corruptions sit at p5 = -53.5, so the clamp would destroy the
    quantity being measured).
    """
    blob = os.path.expanduser(f"~/tmp/corrtheories/{tag}.bin")
    os.makedirs(os.path.dirname(blob), exist_ok=True)
    with open(blob, "wb") as f:
        f.write(struct.pack("<II", X372.shape[1], len(X372)))
        f.write(np.ascontiguousarray(X372, dtype=np.float32).tobytes(order="C"))
    r = subprocess.run([FWD, "--bake", DBAKE, "--bake-post", "raw",
                        "--features-file", blob], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"{FWD} rc={r.returncode}: {r.stderr.strip()[:400]}")
    s = np.array([float(x) for x in r.stdout.split()])
    assert len(s) == len(X372), (len(s), len(X372))
    os.remove(blob)
    return s


def prep():
    os.makedirs(OUT, exist_ok=True)
    idx = list(range(372))

    log("loading corpus (positives + matched anchors)")
    Xc, exc = load_X(CORPUS, idx,
                     extra=("is_corruption", "family", "content_class", "severity",
                            "ref_id", "region", "kind"))
    isc = exc["is_corruption"].astype(int)

    log("loading negrich (severe-honest negatives)")
    Xn, exn = load_X(NEGRICH, idx,
                     extra=("source_id", "dist_name", "severity_level"))

    log("loading ladder (broad-honest negatives)")
    Xb, exb = load_X(LADDER, idx, extra=("image_id", "codec", "q"))

    n = len(Xc) + len(Xn) + len(Xb)
    X = np.empty((n, 372), dtype=np.float32)
    X[:len(Xc)] = Xc
    X[len(Xc):len(Xc)+len(Xn)] = Xn
    X[len(Xc)+len(Xn):] = Xb
    del Xc, Xn, Xb

    blank = lambda v, k: np.array([v] * k)
    y = np.concatenate([isc, np.zeros(len(exn["source_id"]), int),
                        np.zeros(len(exb["image_id"]), int)])
    # source keys EXACTLY as the owner builds them, so split.tsv keys match
    src = np.concatenate([
        exc["ref_id"],
        np.array([f"severe/{v}" for v in exn["source_id"]]),
        np.array([f"ladder/{v}" for v in exb["image_id"]])])
    sub = np.concatenate([
        np.where(isc == 1, "corruption", "matched_anchor"),
        blank("severe_honest", len(exn["source_id"])),
        blank("broad_honest", len(exb["image_id"]))])
    fam = np.concatenate([exc["family"],
                          blank("severe_honest", len(exn["source_id"])),
                          blank("broad_honest", len(exb["image_id"]))])
    cc = np.concatenate([exc["content_class"],
                         blank("severe_honest", len(exn["source_id"])),
                         blank("broad_honest", len(exb["image_id"]))])
    sev = np.concatenate([exc["severity"].astype(str),
                          exn["severity_level"].astype(str),
                          blank("", len(exb["image_id"]))])
    region = np.concatenate([exc["region"].astype(str),
                             blank("", len(exn["source_id"])),
                             blank("", len(exb["image_id"]))])
    kind = np.concatenate([exc["kind"].astype(str),
                           blank("", len(exn["source_id"])),
                           blank("", len(exb["image_id"]))])
    codec = np.concatenate([blank("", len(exc["ref_id"])),
                            blank("", len(exn["source_id"])),
                            exb["codec"].astype(str)])
    q = np.concatenate([np.full(len(exc["ref_id"]), np.nan),
                        np.full(len(exn["source_id"]), np.nan),
                        exb["q"].astype(float)])

    # --- the FROZEN split, read not re-derived -----------------------------
    fmap = {}
    with open(f"{SPLIT_ROOT}/d228/split.tsv") as f:
        next(f)
        for line in f:
            s_, fo = line.rstrip("\n").split("\t")
            fmap[s_] = fo
    missing = sorted({s_ for s_ in src if s_ not in fmap})
    if missing:
        sys.exit(f"{len(missing)} sources absent from split.tsv, e.g. {missing[:3]}")
    fold = np.array([fmap[s_] for s_ in src])

    # --- parity gate against the incumbent's own metrics.json --------------
    m = json.load(open(f"{SPLIT_ROOT}/d228/metrics.json"))
    got = {k: int(v) for k, v in zip(*np.unique(sub, return_counts=True))}
    if got != m["subclass_counts"]:
        sys.exit(f"PARITY FAIL subclass_counts {got} != {m['subclass_counts']}")
    sizes = {k: int((fold == k).sum()) for k in ("train", "val", "test")}
    if sizes["train"] != m["n_train"] or sizes["test"] != m["n_test"]:
        sys.exit(f"PARITY FAIL fold sizes {sizes} vs "
                 f"train={m['n_train']} test={m['n_test']}")
    log(f"PARITY OK — subclasses {got}, folds {sizes}")

    log("scoring Profile D's dial on every row")
    dial = np.concatenate([
        dial_scores(X[:len(exc['ref_id'])], "corpus"),
        dial_scores(X[len(exc['ref_id']):len(exc['ref_id'])+len(exn['source_id'])], "negrich"),
        dial_scores(X[len(exc['ref_id'])+len(exn['source_id']):], "ladder")])

    np.savez_compressed(CACHE, X=X, y=y, src=src, sub=sub, fam=fam, cc=cc,
                        sev=sev, region=region, kind=kind, codec=codec, q=q,
                        fold=fold, dial=dial)
    log(f"cached -> {CACHE}  ({os.path.getsize(CACHE)/1e6:.0f} MB)")

    # gate grid, scored separately (never trained on)
    Xg, exg = load_X(GATE, idx, extra=("entry",))
    np.savez_compressed(GATE_CACHE, X=Xg.astype(np.float32),
                        entry=exg["entry"].astype(str),
                        dial=dial_scores(Xg, "gate"))
    log("cached gate grid")


# --------------------------------------------------------------------------
# shared machinery
# --------------------------------------------------------------------------
class D:
    """The cached dataset + the standard train/val/test masks."""

    def __init__(self):
        z = np.load(CACHE, allow_pickle=False)
        for k in z.files:
            setattr(self, k, z[k])
        self.tr = self.fold == "train"
        self.va = self.fold == "val"
        self.te = self.fold == "test"
        self.Xs = self.X[:, :NFEAT_SLICE].astype(np.float64)
        self.corr_te = self.te & (self.y == 1)

    def masks_te(self):
        return {
            "severe_honest": self.te & (self.y == 0) & (self.sub == "severe_honest"),
            "broad_honest": self.te & (self.y == 0) & (self.sub == "broad_honest"),
            "matched_anchor": self.te & (self.y == 0) & (self.sub == "matched_anchor"),
        }


def standardize(Xtr, Xall, clip=8.0):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    return np.clip(sc.transform(Xall), -clip, clip)


def fit_head(Xall, y, tr, va, model="logistic", seed=0):
    """Fit the SHIPPED head shape and return the three fitted objects.

    Factored out 2026-09-06 so that `fit_probs`, `t6`'s gate-grid scoring and
    the ZCTH `export` command all fit the head ONE way instead of three. The
    body is the code `t6` already ran, in the same order with the same calls,
    so the refactor is arithmetic-neutral by construction and gated by
    re-running `t6` and diffing its TSVs (see the serving record).

    Returns `(sc, clf, iso, raw)` where `raw(M)` is the estimator's own
    decision function over RAW (unstandardized) rows, so a caller holding a
    matrix that was never part of `Xall` -- the gate grid -- can score it
    without rebuilding the scaler.
    """
    from sklearn.isotonic import IsotonicRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xall[tr])
    Z = lambda M: np.clip(sc.transform(M), -8.0, 8.0)          # noqa: E731
    clf = make_classifier(model, seed=seed).fit(Z(Xall[tr]), y[tr])
    raw = lambda M: (clf.decision_function(Z(M))               # noqa: E731
                     if hasattr(clf, "decision_function")
                     else np.log(np.clip(clf.predict_proba(Z(M))[:, 1], 1e-12, 1 - 1e-12) /
                                 np.clip(1 - clf.predict_proba(Z(M))[:, 1], 1e-12, 1 - 1e-12)))
    pv = 1.0 / (1.0 + np.exp(-raw(Xall[va])))
    iso = IsotonicRegression(out_of_bounds="clip").fit(pv, y[va])
    return sc, clf, iso, raw


def fit_probs(Xall, y, tr, va, model="logistic", seed=0, calibrate=True):
    """Fit `model` on train, isotonic-calibrate on val, return P over all rows.

    Mirrors the SHIPPED form of the owner exactly (plain estimator refit on
    train + IsotonicRegression fit on val), not its `CalibratedClassifierCV`
    reporting form — the 2026-09-05 record §4.1 shows the two differ by up to
    4.6 points of ladder FP and that the bake is the one the product runs.
    """
    from sklearn.isotonic import IsotonicRegression
    Z = standardize(Xall[tr], Xall)
    clf = make_classifier(model, seed=seed).fit(Z[tr], y[tr])
    raw = (clf.decision_function(Z) if hasattr(clf, "decision_function")
           else np.log(np.clip(clf.predict_proba(Z)[:, 1], 1e-12, 1 - 1e-12) /
                       np.clip(1 - clf.predict_proba(Z)[:, 1], 1e-12, 1 - 1e-12)))
    p = 1.0 / (1.0 + np.exp(-raw))
    if not calibrate:
        return p, clf
    iso = IsotonicRegression(out_of_bounds="clip").fit(p[va], y[va])
    return rank_break(iso.predict(p), p), clf


def rank_break(p_cal, p_raw):
    """Calibrated probability with isotonic PLATEAUS broken by the raw score.

    Isotonic regression is a step function, so many rows share an identical
    calibrated probability. That is harmless for an absolute deadband (`P > 0.9`)
    and fatal for a matched-operating-point sweep: the achievable FP grid becomes
    coarse and arms land on different achieved FPs, so "matched" stops being
    matched (MEASURED: the per-band arm reported FP exactly 0.00 % at three
    different targets because its top plateau held more rows than the 1 % budget).

    Adding `eps * normalized_rank(p_raw)` is strictly monotone within each
    plateau and cannot reorder ACROSS plateaus (eps << the smallest isotonic
    step), so the ROC is the ordering the model actually produces, the
    cross-band calibrated ordering is preserved, and `P > 0.9` is unchanged to
    9 decimals.
    """
    r = np.empty(len(p_raw))
    r[np.argsort(p_raw, kind="stable")] = np.arange(len(p_raw))
    return p_cal + 1e-9 * (r / max(len(r) - 1, 1))


def pauc(s, pos_mask, neg_mask, fp_max=0.05):
    """Partial AUC of detection vs FP over FP in [0, fp_max], normalized to 1.

    Plateau-robust single number for ranking arms whose achievable FP grids
    differ. Trapezoidal over the empirical ROC restricted to the low-FP region
    that a closed loop actually operates in.
    """
    sp, sn = np.sort(s[pos_mask]), np.sort(s[neg_mask])
    thr = np.unique(np.concatenate([sn, [np.inf]]))
    fp = 1.0 - np.searchsorted(sn, thr, side="right") / len(sn)
    tp = 1.0 - np.searchsorted(sp, thr, side="right") / len(sp)
    o = np.argsort(fp)
    fp, tp = fp[o], tp[o]
    k = fp <= fp_max
    if k.sum() < 2:
        return float("nan")
    f, t = fp[k], tp[k]
    if f[-1] < fp_max:
        f = np.append(f, fp_max)
        t = np.append(t, t[-1])
    return float(np.trapezoid(t, f) / fp_max)


def threshold_for_fp(p, mask, target):
    """Smallest T with FP(mask) <= target. None if unreachable."""
    v = np.sort(p[mask])[::-1]
    k = int(np.floor(target * len(v)))
    if k >= len(v):
        return None
    t = v[k]
    return float(np.nextafter(t, 1.0)) if k > 0 else float(np.nextafter(v[0], 1.0))


def op_table(d, p, label, extra_fire=None):
    """Detection + FP at each matched FP_honest target, plus T=0.9/0.95."""
    mk = d.masks_te()
    bh = mk["broad_honest"]
    fire = (lambda pp, T: pp > T) if extra_fire is None else extra_fire
    rows = []
    for tgt in FP_TARGETS:
        T = threshold_for_fp(p, bh, tgt) if extra_fire is None else None
        if extra_fire is not None:
            # sweep T on the composed rule
            cand = np.unique(np.quantile(p[bh], np.linspace(0, 1, 4001)))
            T = None
            for t in cand[::-1]:
                if fire(p, t)[bh].mean() <= tgt:
                    T = float(t)
                    break
        if T is None:
            rows.append(dict(arm=label, target=tgt, T=None, detection=None))
            continue
        f = fire(p, T)
        rows.append(dict(arm=label, target=tgt, T=T,
                         detection=float(f[d.corr_te].mean()),
                         fp_honest=float(f[bh].mean()),
                         fp_severe=float(f[mk["severe_honest"]].mean()),
                         fp_anchor=float(f[mk["matched_anchor"]].mean()),
                         **qband_fp(d, f)))
    for T in (0.9, 0.95):
        f = fire(p, T)
        rows.append(dict(arm=label, target=f"T={T}", T=T,
                         detection=float(f[d.corr_te].mean()),
                         fp_honest=float(f[bh].mean()),
                         fp_severe=float(f[mk["severe_honest"]].mean()),
                         fp_anchor=float(f[mk["matched_anchor"]].mean()),
                         **qband_fp(d, f)))
    return rows


def qband_fp(d, fired):
    """FP on test-fold ladder rows, per q-band."""
    bh = d.te & (d.sub == "broad_honest")
    out = {}
    for name, lo, hi in QBANDS:
        m = bh & (d.q >= lo) & (d.q < hi)
        out[f"fp_{name}"] = float(fired[m].mean()) if m.sum() else float("nan")
        out[f"n_{name}"] = int(m.sum())
    return out


def boot_detection_ci(d, p, T, n=1000, seed=0):
    """Paired bootstrap over test-fold SOURCES (clustered), detection CI."""
    rng = np.random.default_rng(seed)
    m = d.corr_te
    s = d.src[m]
    hit = (p[m] > T).astype(float)
    uniq, inv = np.unique(s, return_inverse=True)
    sums = np.bincount(inv, weights=hit, minlength=len(uniq))
    cnts = np.bincount(inv, minlength=len(uniq))
    out = np.empty(n)
    for i in range(n):
        k = rng.integers(0, len(uniq), len(uniq))
        out[i] = sums[k].sum() / max(cnts[k].sum(), 1)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def write_tsv(path, rows, cols=None):
    if not rows:
        return
    cols = cols or list(rows[0].keys())
    with open(path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join("" if r.get(c) is None else
                              (f"{r[c]:.6g}" if isinstance(r.get(c), float) else str(r.get(c)))
                              for c in cols) + "\n")
    log(f"wrote {path}")


def export(model="hgb", widths=(372, 944), deadband_t=0.9):
    """Export THIS lane's own fitted head through the owner's ZCTH writer.

    Exists because a fresh training run through `train_corruption_head.py`
    CANNOT bit-reproduce the trees measured here, and saying so is cheaper than
    discovering it: `HistGradientBoostingClassifier(early_stopping=True)` draws
    its internal validation split with `train_test_split(random_state=0)` over
    the rows **in the order they are stacked**, and the owner permutes its
    broad-honest block (`rng.choice(n, n, replace=False)`) while `prep()` does
    not. Same rows, different order, different trees.

    So the artifact the serving lane gates end-to-end is the model whose gate
    numbers are published in `t6_gate_pass.tsv`, exported by the owner's
    `emit_zcth` -- the format writer stays single-owner even though the fit
    lives here.
    """
    import train_corruption_head as OWN            # THE owner (format writer)
    d = D()
    log(f"export: fitting {model} on the frozen split")
    sc, clf, iso, _raw = fit_head(d.Xs, d.y, d.tr, d.va, model=model)
    ids = list(range(NFEAT_SLICE))
    for w in widths:
        path = f"{OUT}/corrhead_{model}_theoryfit_w{w}.zcth"
        OWN.emit_zcth(path, w, ids, sc.mean_, sc.scale_, 8.0, clf, iso,
                      deadband_t,
                      {"lane": "corrserve", "fit": "corrhead_theories.export",
                       "model": model, "sklearn": OWN._sklearn_version(),
                       "dataset": CACHE, "gate_grid": GATE, "caller_width": w,
                       "feature_slice": f"f0..f{NFEAT_SLICE - 1}",
                       "era": "rev1",
                       "split": "corruption-head-2026-09-05/d228/split.tsv",
                       "deadband_t": deadband_t,
                       "record": "benchmarks/corruption_head_serving_2026-09-06.md"})
    # The parity vectors the Rust evaluator is gated against: raw decision
    # function AND calibrated probability, on the frozen TEST fold plus the
    # whole gate grid, in f64.
    z = np.load(f"{OUT}/gate_rev1.npz", allow_pickle=False)
    Xg = z["X"][:, :NFEAT_SLICE].astype(np.float64)
    Zc = lambda M: np.clip(sc.transform(M), -8.0, 8.0)         # noqa: E731
    out = {}
    for name, X in (("test", d.Xs[d.te]), ("gate", Xg)):
        rawv = clf.decision_function(Zc(X))
        praw = 1.0 / (1.0 + np.exp(-rawv))
        out[f"{name}_X"] = X.astype(np.float64)
        out[f"{name}_raw"] = rawv.astype(np.float64)
        out[f"{name}_p"] = iso.predict(praw).astype(np.float64)
    out["gate_entry"] = z["entry"].astype(str)
    out["gate_dial"] = z["dial"].astype(np.float64)
    np.savez(f"{OUT}/parity_{model}.npz", **out)
    log(f"export: parity vectors -> {OUT}/parity_{model}.npz "
        f"(test {out['test_X'].shape}, gate {out['gate_X'].shape})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tests", nargs="+")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for t in a.tests:
        if t == "prep":
            prep()
        elif t == "export":
            export()
        else:
            import corrhead_tests
            getattr(corrhead_tests, t)(D())


if __name__ == "__main__":
    main()
