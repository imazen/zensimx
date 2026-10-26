"""AIC2026 TRAIN (the supplied 42-reference JND reconstruction, PTC crops) for the by-construction exploration.

Exploration record: benchmarks/byconstruction_2026-10-22.md. Every number this script prints about AIC2026 is either a
frozen model's score (no fit on AIC2026) or an out-of-fold prediction: the references are split into folds, and each
reference's rows are predicted by a model fitted without any row of that reference. In-sample fits are never reported.

The table is `mixture_table.py build-pairs` output: `key` (distorted id), `ref_basename` (source), `human_score`
(= -JND_curve, higher is better) and the computed Rev5 vector f0..f719 plus the byc_globaff descriptor f1867..f1890.

  zeroshot --table T.parquet --pairs PAIRS.tsv --bin-dir BIN --bake NAME=PATH [...] [--decl DECL.json] --out OUT.json
  cv       --table T.parquet --pairs PAIRS.tsv --model NAME [...] [--folds 6] [--repeats 3] --out OUT.json
  analyze  (cv's input options) --model NAME [--labels JND_curve.csv] --out OUT.json

Input options (cv, analyze): --globaff adds the descriptor, --extra P.parquet adds damage-feature tables,
--no-fx1 drops B's inputs, --peer COL adds a supplied peer metric column (diagnostic only).
Models: nnls (monotone linear), gamcvx / gammono (monotone additive, convex / any shape), ridge:A, hgbmono, hgb.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.optimize import nnls
from scipy.stats import rankdata, spearmanr

V1_WIDTH, V2_START, V2_CELLS = 372, 372, 12
V2 = ("ssim_mean ssim_dev2 ssim_dev4 art det mse hf_gain hf_loss hf_mag_loss ssim_soft_peak art_soft_peak "
      "det_soft_peak masked_ssim masked_art masked_det masked_mse iw_ssim iw_art iw_det iw_mse pjnd_transducer "
      "pjnd_fragility gms transducer_lowk transducer_highk blockiness ringing banding edge_width").split()
GLOBAFF = list(range(1867, 1891))
DECL = Path(__file__).resolve().parents[1] / "rev4_featpot" / "byc_fx1g_declaration.json"


def prepare_rev7_folds(table, pairs, assignment, output, build_commit):
    """D6 public_train only: preserve the registered source folds and Rev7 rows.

    All products, targets and source identities stay in private output files.
    Input hashes and the authorization ledger are persisted before target reads.
    """
    import pyarrow as pa
    table, pairs, assignment, output = map(Path, (table, pairs, assignment, output))
    if output.exists():
        raise ValueError("preserve existing AIC fold artifacts; choose a fresh output")
    if pairs.name != "pairs_train.tsv" or assignment.name != "assignment.json":
        raise ValueError("D6 requires the explicit public_train pair list and assignment")
    a = json.loads(assignment.read_text())
    if (a.get("population") != "WG1 m112107 public_train" or a.get("folds") != 7
            or a.get("labels") != "JND_score (human_score = -JND_score)"
            or len(a.get("assignment", {})) != 28
            or any(list(a["assignment"].values()).count(k) != 4 for k in range(7))):
        raise ValueError("D6 public_train source-fold contract differs")
    producer = json.loads(table.with_suffix(".json").read_text())
    if (producer.get("dump_bake_sha256") !=
            "5fe0a96230badfa4228e314f12b962b3c18408612d8da9a6390d6259c5a9bcd4"
            or producer.get("ring_form") != "raw"):
        raise ValueError("AIC features must come from the registered B7 extraction")
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    bindings = {"source_table_sha256": digest(table), "pairs_sha256": digest(pairs),
                "assignment_sha256": digest(assignment), "producer_sha256": digest(table.with_suffix(".json"))}
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rev4_featpot"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from v2_human_role import D6_INPUT_PINS, aic_public_train_declaration
    if any(bindings[k] != v for k, v in D6_INPUT_PINS.items()):
        raise ValueError("D6 public_train input pin differs")
    output.mkdir(parents=True)
    def write(path, value):
        path.write_text(json.dumps(value, indent=1, allow_nan=False) + "\n")
    write(output / "LABEL_READS.json", dict(role="TRAIN", ruling="D6", reads=[dict(
        path=str(pairs), columns=["jnd"], population=a["population"],
        state="authorized-before-read", **bindings)]))
    p = pd.read_csv(pairs, sep="\t")
    names = ["key", "ref_basename", *(f"f{i}" for i in range(1853))]
    t = pq.read_table(table, columns=names).to_pandas()
    if (len(p) != 3920 or len(t) != len(p) or p.key.duplicated().any()
            or not np.array_equal(t.key.astype(str), p.key.astype(str))
            or not np.array_equal(t.ref_basename.astype(str), p.group.astype(str))
            or set(p.group) != set(a["assignment"])):
        raise ValueError("AIC public_train ordered rows/source coverage differ")
    y = -p.jnd.to_numpy(dtype=np.float64)
    if not np.isfinite(y).all():
        raise ValueError("nonfinite public_train JND")
    t["human_score"] = y
    f = p.group.map(a["assignment"]).to_numpy(dtype=int)
    records = {}
    for k in [*range(7), None]:
        for split in (["fit", "test"] if k is not None else ["fit"]):
            mask = np.ones(len(p), dtype=bool) if k is None else (f != k if split == "fit" else f == k)
            name = f"aic_k{k}_{split}" if k is not None else "aic_full_fit"
            dest = output / f"{name}.parquet"
            keys = [dict(row=j, key=str(p.key.iloc[i]), source=str(p.group.iloc[i]),
                         split="public_train", source_fold=int(f[i]))
                    for j, i in enumerate(np.flatnonzero(mask))]
            kp = dest.with_suffix(".keys.json")
            write(kp, keys)
            frame = pa.Table.from_pandas(t.loc[mask].reset_index(drop=True), preserve_index=False)
            frame = frame.replace_schema_metadata({b"feature_set_id":
                b"basic+peaks+v2@w1825/rev7_rowlanes#36c3f3af", b"formula_revision": b"7"})
            pq.write_table(frame, dest, compression="zstd")
            rec = dict(schema="aic2026-public-train-rev7-v1", ruling="D6", split="public_train",
                role="TRAIN", use="fit" if split == "fit" else "out-of-fold-score", omitted_source_fold=k,
                formula_revision=7, feature_set_id="basic+peaks+v2@w1825/rev7_rowlanes#36c3f3af",
                rows=len(keys), sources=sorted({r["source"] for r in keys}),
                assignment_path=str(assignment.resolve()),
                table_sha256=digest(dest), keys_path=str(kp), keys_sha256=digest(kp),
                target_column="human_score", target_transform="negate-JND_score", build_commit=build_commit,
                **bindings)
            aic_public_train_declaration(rec, keys, fit=split == "fit")
            write(Path(str(dest) + ".manifest.json"), rec)
            records[name] = dict(path=str(dest), manifest_sha256=digest(Path(str(dest) + ".manifest.json")), **rec)
    manifest = dict(schema="aic2026-public-train-rev7-folds-v1", ruling="D6", build_commit=build_commit,
                    population=a["population"], assignment=a["assignment"], tables=records, **bindings)
    write(output / "_MANIFEST.json", manifest)
    return manifest


def verify_rev7_folds(output):
    """Replay all metadata/key checks, then verify every output byte digest.

    No target column is opened or interpreted by this independent readback.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rev4_featpot"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from v2_human_role import D6_INPUT_PINS, aic_public_train_declaration
    output = Path(output)
    manifest = json.loads((output / "_MANIFEST.json").read_text())
    expected = {f"aic_k{k}_{split}" for k in range(7) for split in ("fit", "test")} | {"aic_full_fit"}
    if (manifest.get("schema") != "aic2026-public-train-rev7-folds-v1"
            or set(manifest.get("tables", {})) != expected
            or any(manifest.get(k) != v for k, v in D6_INPUT_PINS.items())):
        raise ValueError("D6 complete revision-7 fold inventory differs")
    payloads = []
    for name, record in manifest["tables"].items():
        path = output / (name + ".parquet")
        sidecar = Path(str(path) + ".manifest.json")
        if (path.resolve() != Path(record["path"]).resolve()
                or hashlib.sha256(sidecar.read_bytes()).hexdigest() != record["manifest_sha256"]):
            raise ValueError("D6 fold output declaration differs")
        d = json.loads(sidecar.read_text())
        kp = Path(d["keys_path"])
        if hashlib.sha256(kp.read_bytes()).hexdigest() != d["keys_sha256"]:
            raise ValueError("D6 fold output keys differ")
        keys = json.loads(kp.read_text())
        aic_public_train_declaration(d, keys, fit=name.endswith("_fit"))
        payloads.append((path, d["table_sha256"]))
    for path, expected_sha in payloads:
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha:
            raise ValueError("D6 fold output bytes differ")
    return dict(status="PASS", tables=len(payloads), targets_interpreted=False,
                manifest_sha256=hashlib.sha256((output / "_MANIFEST.json").read_bytes()).hexdigest())


def v2_name(i: int) -> str:
    cell, k = divmod(i - V2_START, len(V2))
    return f"{V2[k]}_s{cell // 3}{'XYB'[cell % 3]}"


def load(table: str, pairs: str):
    """The table rows of the pairs list, in its order (a pairs list may select a subset, e.g. one WG1 split)."""
    t = pq.read_table(table).to_pandas()
    p = pd.read_csv(pairs, sep="\t")
    keys = p.key.astype(str)
    if keys.duplicated().any():
        raise ValueError("duplicate keys in the pairs list")
    t = t.set_index("key").loc[keys].reset_index()
    return t, p


def panel(pred: np.ndarray, p: pd.DataFrame) -> dict:
    """Quality SROCC against -JND: global, mean within reference, mean within ladder, per codec (pooled)."""
    q = -p.jnd.to_numpy()
    out = {"global": float(spearmanr(pred, q).correlation)}
    wr = [spearmanr(pred[g.index], q[g.index]).correlation for _, g in p.groupby("group")]
    out["within_ref_mean"], out["within_ref_min"] = float(np.mean(wr)), float(np.min(wr))
    wl = [spearmanr(pred[g.index], q[g.index]).correlation for _, g in p.groupby(["group", "codec"]) if len(g) > 2]
    out["within_ladder_mean"] = float(np.nanmean(wl))
    out["per_codec"] = {c: round(float(spearmanr(pred[g.index], q[g.index]).correlation), 4)
                        for c, g in p.groupby("codec")}
    return out


def scale_of(i: int) -> int:
    """Pyramid scale of a Rev5 slot (basic 13, peaks 6, masked 6, IW 6 per cell, then v2 29 per cell; 3 cells/scale)."""
    for start, per in ((0, 13), (156, 6), (228, 6), (300, 6), (372, 29)):
        if start <= i < start + 12 * per:
            return (i - start) // per // 3
    if i in GLOBAFF:
        return (i - 1867) // 6
    raise ValueError(i)


def fx1(t: pd.DataFrame, globaff: bool, min_scale: int = 0):
    """B's fx1 inputs: 410 direct damage features, 410 products with their pjnd_fragility factor; content slots.
    `min_scale` keeps only slots measured at that pyramid scale or coarser (a cheaper extraction)."""
    d = json.loads(DECL.read_text())
    direct = [i for i in d["direct"] if i not in GLOBAFF and scale_of(i) >= min_scale]
    d["products"] = [(i, f) for i, f in d["products"] if scale_of(i) >= min_scale]
    frag = sorted({f for _, f in d["products"]})
    cols = [t[f"f{i}"].to_numpy(np.float64) for i in direct]
    cols += [t[f"f{i}"].to_numpy(np.float64) * t[f"f{f}"].to_numpy(np.float64) for i, f in d["products"]]
    names = [f"f{i}" for i in direct] + [f"f{i}*f{f}" for i, f in d["products"]]
    if globaff:
        keep = [i for i in GLOBAFF if scale_of(i) >= min_scale]
        cols += [t[f"f{i}"].to_numpy(np.float64) for i in keep]
        names += [f"g{i - 1867}" for i in keep]
    content = [t[f"f{f}"].to_numpy(np.float64) for f in frag]
    return np.stack(cols, 1), names, np.stack(content, 1), [f"f{f}" for f in frag]


def folds(groups: np.ndarray, k: int, seed: int) -> np.ndarray:
    u = np.array(sorted(set(groups)))
    perm = np.random.default_rng(seed).permutation(len(u))
    fold_of = {g: int(i % k) for g, i in zip(u[perm], range(len(u)))}
    return np.array([fold_of[g] for g in groups])


def fit_predict(model: str, Xtr, ytr, Xte, Ctr, Cte):
    """Return predicted JND (higher = worse) for Xte. Damage features are >= 0."""
    if model == "nnls":  # monotone linear: jnd = c . x, c >= 0, plus intercept
        A = np.c_[Xtr, np.ones(len(Xtr))]
        s = np.maximum(A.std(0), 1e-12)
        c, _ = nnls(A / s, ytr, maxiter=20 * A.shape[1])
        return (np.c_[Xte, np.ones(len(Xte))] / s) @ c
    if model.startswith("gam"):  # monotone additive: NNLS over per-feature hinge (convex) and ramp (saturating) bases
        Ztr, Zte = np.log1p(Xtr / np.maximum(Xtr.std(0), 1e-12)), np.log1p(Xte / np.maximum(Xtr.std(0), 1e-12))
        q = np.quantile(Ztr, [0.25, 0.5, 0.75, 0.9], axis=0)

        def basis(Z):
            cols = [Z] + [np.maximum(Z - k, 0.0) for k in q]
            if model == "gammono":
                cols += [np.minimum(Z, k) for k in q]
            return np.concatenate(cols, 1)
        A, B = basis(Ztr), basis(Zte)
        s = np.maximum(A.std(0), 1e-12)
        A = np.c_[A / s, np.ones(len(A))]
        c, _ = nnls(A, ytr, maxiter=10 * A.shape[1])
        return np.c_[B / s, np.ones(len(B))] @ c
    if model.startswith("ridge"):
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(np.log1p(Xtr))
        r = Ridge(alpha=float(model.split(":")[1]) if ":" in model else 10.0).fit(sc.transform(np.log1p(Xtr)), ytr)
        return r.predict(sc.transform(np.log1p(Xte)))
    if model.startswith("hgb"):
        from sklearn.ensemble import HistGradientBoostingRegressor
        mono = model.startswith("hgbmono")
        Xa, Xb = np.c_[Xtr, Ctr], np.c_[Xte, Cte]
        cst = [1] * Xtr.shape[1] + [0] * Ctr.shape[1] if mono else None
        h = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40,
                                          l2_regularization=1.0, monotonic_cst=cst, random_state=0)
        return h.fit(Xa, ytr).predict(Xb)
    raise ValueError(model)


PEER_TABLE = "/mnt/v/datasets/aic2026/combined-july-drop-2026-10-20/combined.csv"


def peer_columns(p: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Supplied (published) peer metric columns, cropped view, joined on the distorted id; oriented so higher = worse."""
    m = pd.read_csv(PEER_TABLE, low_memory=False)
    m = m[m.view == "cropped"]
    m = m.set_index(m.distorted.str.removesuffix(".png"))
    out = []
    for c in cols:
        name, sign = (c[1:], -1.0) if c.startswith("-") else (c, 1.0)
        out.append(sign * pd.to_numeric(m.loc[p.key.astype(str), name], errors="raise").to_numpy(np.float64))
    return np.stack(out, 1)


def inputs(a, t: pd.DataFrame, p: pd.DataFrame):
    X, names, C, _ = fx1(t, a.globaff, a.min_scale)
    if a.only_peers or a.no_fx1:
        X, names = X[:, :0], []
    for path in a.extra:  # extra damage-feature tables (key + columns), e.g. jnd_band_proto.py output
        e = pd.read_parquet(path).set_index("key").loc[p.key.astype(str)].reset_index()
        cols = [c for c in e.columns if c != "key" and (not a.extra_match or any(m in c for m in a.extra_match))]
        E = e[cols].to_numpy(np.float64)
        if not np.isfinite(E).all() or (E < 0).any():
            raise ValueError(f"{path}: extra features must be finite and >= 0")
        X, names = np.c_[X, E], names + cols
    if a.peer:  # diagnostic only: which information is missing from the features
        P = peer_columns(p, a.peer)
        X, names = np.c_[X, P - P.min(0)], names + a.peer
    return X, names, C


def oof_predict(model: str, X, C, y, groups, n_folds: int, seed: int, target: str = "none") -> np.ndarray:
    """Out-of-fold JND predictions: every reference's rows come from a model fitted without that reference."""
    f = folds(groups, n_folds, seed)
    oof = np.full(len(y), np.nan)
    for k in range(n_folds):
        tr, te = f != k, f == k
        yt = {"none": y, "log1p": np.log1p(y), "sqrt": np.sqrt(y), "rank": None}[target]
        if target == "rank":  # fold-internal ranks of the training labels
            yt = np.zeros(len(y))
            yt[tr] = rankdata(y[tr]) / tr.sum()
        oof[te] = fit_predict(model, X[tr], yt[tr], X[te], C[tr], C[te])
    return oof


def cmd_cv(a) -> int:
    t, p = load(a.table, a.pairs)
    X, names, C = inputs(a, t, p)
    y = p.jnd.to_numpy(np.float64)
    groups = p.group.to_numpy()
    res = {"table": a.table, "inputs": X.shape[1], "content": C.shape[1], "folds": a.folds, "repeats": a.repeats,
           "globaff": a.globaff, "peer": a.peer, "only_peers": a.only_peers,
           "extra": a.extra, "extra_match": a.extra_match, "no_fx1": a.no_fx1,
           "target": a.target, "models": {}}
    for model in a.model:
        reps = []
        for r in range(a.repeats):
            reps.append(panel(-oof_predict(model, X, C, y, groups, a.folds, 1000 + r, a.target), p))
        g = [x["global"] for x in reps]
        res["models"][model] = {"global_mean": float(np.mean(g)), "global_per_repeat": g,
                                "within_ref_mean": float(np.mean([x["within_ref_mean"] for x in reps])),
                                "within_ladder_mean": float(np.mean([x["within_ladder_mean"] for x in reps])),
                                "per_codec_repeat0": reps[0]["per_codec"]}
        print(f"{model:16s} OOF global {np.mean(g):.4f} {['%.4f' % v for v in g]}  within-ref "
              f"{res['models'][model]['within_ref_mean']:.4f}  within-ladder {res['models'][model]['within_ladder_mean']:.4f}",
              flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    return 0


def cmd_analyze(a) -> int:
    """Where the out-of-fold error lives: oracle offset removal, discordant-pair classes, worst ladders, and the
    SROCC the supplied label confidence intervals allow (labels resampled from their CIs, ladder-shared or not)."""
    t, p = load(a.table, a.pairs)
    X, _, C = inputs(a, t, p)
    y, groups = p.jnd.to_numpy(np.float64), p.group.to_numpy()
    oof = oof_predict(a.model[0], X, C, y, groups, a.folds, 1000, a.target)
    p = p.assign(r=y - oof)
    rep = {"model": a.model[0], "oof_global": float(spearmanr(oof, y).correlation), "oracle": {}, "discordance": {}}
    for keys in (["codec"], ["group"], ["group", "codec"]):
        off = p.groupby(keys).r.transform("mean").to_numpy()
        rep["oracle"]["remove mean residual by " + "+".join(keys)] = float(spearmanr(oof + off, y).correlation)
    lm = p.groupby(["group", "codec"]).r.transform("mean").to_numpy()
    rep["oracle"]["perfect within-ladder shape, ladder offsets kept"] = float(spearmanr(y - lm, y).correlation)
    rng = np.random.default_rng(0)
    i, j = rng.integers(0, len(y), 2_000_000), rng.integers(0, len(y), 2_000_000)
    k = y[i] != y[j]
    i, j = i[k], j[k]
    disc = np.sign(y[i] - y[j]) != np.sign(oof[i] - oof[j])
    gap = np.abs(y[i] - y[j])
    rep["discordance"]["all"] = float(disc.mean())
    same_src = groups[i] == groups[j]
    rep["discordance"]["cross_source_share_of_discordant"] = float(disc[~same_src].sum() / disc.sum())
    for lo, hi in ((0, .25), (.25, .5), (.5, 1), (1, 2), (2, 9)):
        m = (gap >= lo) & (gap < hi)
        rep["discordance"][f"|dJND| [{lo},{hi})"] = {"pair_share": float(m.mean()), "discordant": float(disc[m].mean()),
                                                      "share_of_discordant": float(disc[m].sum() / disc.sum())}
    lad = p.groupby(["group", "codec"]).r.mean().sort_values()
    rep["worst_ladders"] = {"too_harsh": {f"{g}/{c}": round(v, 3) for (g, c), v in lad.head(8).items()},
                            "too_lenient": {f"{g}/{c}": round(v, 3) for (g, c), v in lad.tail(8).items()}}
    if a.labels:
        lab = pd.read_csv(a.labels)
        lab = lab.set_index("distorted").loc[p.key.astype(str)]
        sd = ((lab.JND_curve_CI_high - lab.JND_curve_CI_low) / 3.92).to_numpy()
        ladder = p.groupby(["group", "codec"]).ngroup().to_numpy()
        for name, frac in (("independent", 0.0), ("ladder_shared", 1.0)):
            r = []
            for _ in range(20):
                z = np.sqrt(frac) * rng.normal(size=ladder.max() + 1)[ladder] + np.sqrt(1 - frac) * rng.normal(size=len(y))
                r.append(spearmanr(y, y + z * sd).correlation)
            rep[f"label_ci_ceiling_{name}"] = float(np.mean(r))
    print(json.dumps(rep, indent=1))
    Path(a.out).write_text(json.dumps(rep, indent=1) + "\n")
    return 0


def cmd_zeroshot(a) -> int:
    t, p = load(a.table, a.pairs)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rev4_featpot"))
    os.environ.setdefault("REV4_V2_BIN_DIR", a.bin_dir)
    from v2_lodo_mlp import predict  # noqa: E402
    res = {"table": a.table, "fit_on_aic2026": False, "models": {}}
    for spec in a.bake:
        name, path = spec.split("=", 1)
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as td:
            if a.decl:
                subprocess.run([str(Path(a.bin_dir) / "byc_predict"), "--bake", path, "--decl", a.decl, "--corpus",
                                a.table, "--out", str(Path(td) / "pred.tsv")], check=True, capture_output=True)
                pred = pd.read_csv(Path(td) / "pred.tsv", sep="\t").pred.to_numpy(np.float64)
            else:
                pred = predict(Path(path), Path(a.table), Path(td) / "pred.tsv")
        full_keys = pq.read_table(a.table, columns=["key"])["key"].to_pylist()
        pred = pd.Series(pred, index=full_keys).loc[p.key.astype(str)].to_numpy()  # the pairs list's rows
        res["models"][name] = {"bake": path, **panel(pred, p)}
        m = res["models"][name]
        print(f"{name:20s} global {m['global']:.4f} within-ref {m['within_ref_mean']:.4f} "
              f"within-ladder {m['within_ladder_mean']:.4f}", flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("rev7-folds", help="D6: private public_train tables in the registered seven source folds")
    for option in ("table", "pairs", "assignment", "output", "build-commit"):
        f.add_argument("--" + option, required=True)
    f.set_defaults(fn=lambda a: (prepare_rev7_folds(a.table, a.pairs, a.assignment, a.output, a.build_commit), 0)[1])
    v = sub.add_parser("verify-rev7-folds", help="D6 metadata/key and output byte readback; no targets interpreted")
    v.add_argument("output")
    v.set_defaults(fn=lambda a: (print(json.dumps(verify_rev7_folds(a.output))), 0)[1])
    z = sub.add_parser("zeroshot")
    z.add_argument("--table", required=True)
    z.add_argument("--pairs", required=True)
    z.add_argument("--bin-dir", required=True)
    z.add_argument("--bake", action="append", required=True)
    z.add_argument("--decl")
    z.add_argument("--out", required=True)
    z.set_defaults(fn=cmd_zeroshot)
    for name in ("cv", "analyze"):
        c = sub.add_parser(name)
        add_input_args(c)
        c.set_defaults(fn=cmd_cv if name == "cv" else cmd_analyze)
    a = ap.parse_args()
    return a.fn(a)


def add_input_args(c) -> None:
    c.add_argument("--table", required=True)
    c.add_argument("--pairs", required=True)
    c.add_argument("--model", action="append", required=True)
    c.add_argument("--folds", type=int, default=6)
    c.add_argument("--repeats", type=int, default=3)
    c.add_argument("--globaff", action="store_true")
    c.add_argument("--min-scale", type=int, default=0, help="keep only fx1 slots at this pyramid scale or coarser")
    c.add_argument("--peer", action="append", default=[],
                   help="supplied peer column as an extra input (diagnostic); prefix '-' for higher-is-better metrics")
    c.add_argument("--only-peers", action="store_true")
    c.add_argument("--extra", action="append", default=[], help="parquet of extra damage features (key + columns)")
    c.add_argument("--extra-match", action="append", default=[], help="keep extra columns containing any of these")
    c.add_argument("--no-fx1", action="store_true", help="drop B's fx1 inputs (use only --extra / --peer)")
    c.add_argument("--target", default="none", choices=("none", "log1p", "sqrt", "rank"),
                   help="monotone transform of the JND target before fitting (SROCC is invariant to it)")
    c.add_argument("--labels", help="analyze: the supplied JND_curve.csv, for the label-CI ceiling")
    c.add_argument("--out", required=True)


if __name__ == "__main__":
    sys.exit(main())
