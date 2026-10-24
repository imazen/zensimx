"""Mohammadi 2025 panel (SROCC, PLCC, KROCC, OR, PWRC, Z-RMSE through the native `panel --batch`) per dataset, model
and sample role, for the by-construction AIC2026 TRAIN study (benchmarks/byconstruction_2026-10-22.md).

Sample roles:
- E21 sets (KADID TRAIN+SELECT, TID2013, KonFiG TRAIN+VAL, CID22-A 25 refs): `out of sample` is each LODO cell's
  prediction of the dataset it was trained without (seeds 0-2, statistics averaged); `in sample` is the seed-0 cell of
  another fold predicting this dataset, which it trained on (fit and development rows).
- AIC2026 public_train (WG1 JND_score, quality = -JND): `zero-shot` (no AIC fit), `out of fold` (source-held-out
  folds), `in sample` (fitted on all of public_train).
- AIC2026 public_test: frozen B only (the one read of 2026-10-23); no candidate is read.

Options: `--dvifm` adds DVIFM 0.3 (dvifm_e21.py outputs; zero-shot on E21, its constants were fitted on AIC2026);
`--geo` adds the scatter geometry (zenstats `diagnose`: rank-mapped residual envelope, tails, coverage, clamp mass) and
the absolute-error tails around the panel's own 4-parameter logistic fit; `--per-type` adds every E21 statistic per
distortion type (KADID 25, TID2013 24, KonFiG 7, CID22-A 8 codecs).

  python3 aic_panel.py --out PANEL.json [--arms base-c,Tghpm,Tga,Thpm,Mhpm] [--dvifm] [--geo] [--per-type]
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from lib.zen_stats import logistic_fit, panel_batch, scatter  # noqa: E402

BIN = Path(os.environ.get("AIC_BIN", "/var/tmp/byconstruction-bin/58a4d100ad36"))
os.environ.setdefault("REV4_V2_BIN_DIR", str(BIN))
os.environ.setdefault("ZEN_PANEL_BIN", str(BIN / "panel"))
OUT = Path("/mnt/v/output/zensim/byconstruction-2026-10-22/aic2026train")
F = OUT / "folds_train"
CELLS = Path.home() / "tmp/byc/cells"
E34 = Path("/mnt/v/output/zensim/e34-launch-2026-10-15/evaluation/panel-input.json")
E34_CELLS = Path("/var/tmp/rev4-featpot/e34-controlc-results/cells/sel:3b7bd5ebe929@h32:H128:cv16:cf98:fx1__N")
ROOTS = {"raw": Path("/mnt/v/output/zensim/v40r4-2026-10-08/v2e29"),
         "hpm": Path("/mnt/v/output/zensim/byconstruction-2026-10-22/ringhpm/root"),
         "g": Path("/mnt/v/output/zensim/byconstruction-2026-10-22/ringhpm/root_globaff"),
         "gg": Path("/mnt/v/output/zensim/byconstruction-2026-10-22/ringhpm/root_globaff_gated"),
         "linpyr": Path("/mnt/v/output/zensim/byconstruction-2026-10-22/linpyr/root")}


def arm_root(arm: str, spec: str) -> str:
    """The training root of an arm's cells: B's control is the raw root; fx1g/fx1gs1 arms the hpm + descriptor root;
    fx1gg arms the gated root."""
    if arm == "base-c":
        return "raw"
    decl = research_decl(spec)
    return "hpm" if decl is None else "gg" if decl.name == "byc_fx1gg_declaration.json" else "g"
def cell_root(cell: Path, arm: str, spec: str) -> Path:
    """The root a cell actually trained on, read from its trainer command line (train.log's first line); falls back
    to `arm_root` for B's control, whose E34 cells keep no such log here."""
    log = cell / "train.log"
    if arm != "base-c" and log.is_file():
        first = log.read_text().split("\n", 1)[0]
        hits = [r for r in ROOTS.values() if f"{r}/wide/" in first]
        if len(hits) != 1:
            raise ValueError(f"{cell}: cannot tell its training root")
        return hits[0]
    return ROOTS[arm_root(arm, spec)]


FOLDS = ("kadid", "tid2013", "konfig", "cid22_a25")
POP = {"kadid": "KADID TRAIN+SELECT", "tid2013": "TID2013 (train-only by ruling)", "konfig": "KonFiG TRAIN+VAL",
       "cid22_a25": "CID22-A, 25 references"}
STATS = ("srocc_signed", "plcc", "krocc", "or", "pwrc", "z_rmse")
DVIFM = Path("/mnt/v/output/zensim/byconstruction-2026-10-22/dvifm03")
# Readable distortion-type names: KADID-10k (Lin, Hosu and Saupe 2019, Table 1) and TID2013 type order.
TYPE_NAMES = {
    **{f"kadid_{i:02d}": n for i, n in enumerate((
        "Gaussian blur", "lens blur", "motion blur", "color diffusion", "color shift", "color quantization",
        "color saturation 1", "color saturation 2", "JPEG 2000", "JPEG", "white noise", "white noise in color",
        "impulse noise", "multiplicative noise", "denoise", "brighten", "darken", "mean shift", "jitter",
        "non-eccentricity patch", "pixelate", "quantization", "color block", "high sharpen", "contrast change"), 1)},
    **{f"tid_{i:02d}": n for i, n in enumerate((
        "additive Gaussian noise", "noise in color components", "spatially correlated noise", "masked noise",
        "high-frequency noise", "impulse noise", "quantization noise", "Gaussian blur", "denoising", "JPEG",
        "JPEG 2000", "JPEG transmission errors", "JPEG 2000 transmission errors", "non-eccentricity pattern noise",
        "local block-wise distortion", "mean shift", "contrast change", "color saturation change",
        "multiplicative Gaussian noise", "comfort noise", "lossy compression of noisy images",
        "color quantization with dither", "chromatic aberration", "sparse sampling and reconstruction"), 1)},
    "colordiffusion": "color diffusion", "highsharpen": "high sharpen", "multinoise": "multiplicative noise",
    "jpeg2000": "JPEG 2000", "lensblur": "lens blur", "motionblur": "motion blur", "jitter": "jitter",
}
GEO = ("out4", "p99d", "maxd", "mad", "cov", "clump", "clampLo", "clampHi")
GEOMETRY = GEO + ("raw_chart_p99", "fit_p99", "fit_max", "fit_mad", "fit_out4")


def dvifm_e21():
    """E21-ordered DVIFM 0.3 quality (-JND) and distortion type per dataset (dvifm_e21.py outputs; no labels there)."""
    rows = pd.read_parquet(DVIFM / "rows.parquet")
    jnd = pd.read_csv(DVIFM / "dvifm.tsv", sep="\t").set_index(["ref_path", "dist_path"]).jnd
    q = -jnd.loc[list(zip(rows.ref_path, rows.dist_path))].to_numpy(np.float64)
    return ({ds: q[(rows.dataset == ds).to_numpy()] for ds in FOLDS},
            {ds: rows.type[rows.dataset == ds].to_numpy() for ds in FOLDS})


def geometry(x, y, arrays: bool = False):
    """zenstats scatter geometry (rank-mapped residual: share outside the ±4 robust-scale envelope, p99 and max over
    the target span, robust scale, coverage, clump, clamp mass; raw chart-z p99) and the absolute-error tails around
    the panel's 4-parameter logistic fit (p99 and max over the target span, robust scale, share beyond 4 robust
    scales). Polarity is the caller's: `x` and `y` must agree in direction. With `arrays`, also returns the
    rank-mapped and logistic-fitted predictions in target units, in row order."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    g = scatter(x, y)
    ok = g["status"] == "MEASURED"
    out = {k: (g["geo"][k] if ok else None) for k in GEO}
    out["raw_chart_p99"] = g["raw"]["p99"] if ok else None
    fitted = np.asarray(logistic_fit(x, y))
    r = fitted - y
    span = float(np.quantile(y, 0.99) - np.quantile(y, 0.01))
    mad = 1.4826 * float(np.median(np.abs(r - np.median(r))))
    out.update(fit_p99=float(np.quantile(np.abs(r), 0.99) / span), fit_max=float(np.abs(r).max() / span),
               fit_mad=mad, fit_out4=float((np.abs(r) > 4 * mad).mean()))
    if arrays:
        return out, {"rank": np.asarray(g["normalized_pred"]) if ok else None, "fit": fitted}
    return out


def research_decl(spec: str):
    for tok in ("fx1gs1", "fx1gg", "fx1g"):
        if f":{tok}:" in f"{spec}:":
            return HERE.parent / "rev4_featpot" / f"byc_{tok}_declaration.json"
    return None


def predict(bake: Path, table: Path, spec: str) -> np.ndarray:
    with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as td:
        decl = research_decl(spec)
        if decl is not None:
            subprocess.run([str(BIN / "byc_predict"), "--bake", str(bake), "--decl", str(decl), "--corpus", str(table),
                            "--out", f"{td}/p.tsv"], check=True, capture_output=True)
            return pd.read_csv(f"{td}/p.tsv", sep="\t").pred.to_numpy(np.float64)
        sys.path.insert(0, str(HERE.parent / "rev4_featpot"))
        from v2_lodo_mlp import predict as owner_predict  # noqa: E402
        return owner_predict(bake, table, Path(td) / "p.tsv")


def e21_jobs(arms: list[str], dvifm=None):
    panel_in = {d["name"]: d for d in json.loads(E34.read_text())["datasets"]}
    jobs, meta = [], []
    for fold in FOLDS:
        y = np.asarray(panel_in[fold]["target"], np.float64)
        if dvifm is not None:
            jobs.append((f"{fold}|DVIFM03|zero|0", dvifm[fold], y))
            meta.append(dict(dataset=fold, model="DVIFM 0.3", sample="zero-shot (constants fitted on AIC2026)", seed=0))
        for arm in arms:
            for s in (0, 1, 2):  # out of sample: the cell trained without this dataset
                if arm == "base-c":
                    x = np.asarray(panel_in[fold]["predictions"]["base-c"][s], np.float64)
                else:
                    p = CELLS / arm / f"without_{fold}_s{s}" / f"pred_{fold}.tsv"
                    if not p.is_file():
                        continue
                    x = pd.read_csv(p, sep="\t").pred.to_numpy(np.float64)
                jobs.append((f"{fold}|{arm}|out|{s}", x, y))
                meta.append(dict(dataset=fold, model=arm, sample="out of sample (dataset held out)", seed=s))
            other = next(f for f in FOLDS if f != fold)  # in sample: a seed-0 cell that trained on this dataset
            cell = (E34_CELLS if arm == "base-c" else CELLS / arm) / f"without_{other}_s0"
            bake = cell / "refit" / "last.bin"
            if not bake.is_file():
                continue
            spec = "" if arm == "base-c" else json.loads((cell / "byc_record.json").read_text())["spec"]
            table = cell_root(cell, arm, spec) / "wide/main/real" / f"{fold}.parquet"
            x = predict(bake, table, spec)
            jobs.append((f"{fold}|{arm}|in|0", x, y))
            meta.append(dict(dataset=fold, model=arm, sample=f"in sample (cell without {other}, seed 0)", seed=0))
    return jobs, meta


def aic_jobs(public_test: bool = True):
    ptr = pd.read_csv(F / "pairs_train.tsv", sep="\t")
    keys = ptr.key.astype(str)
    y = -ptr.jnd.to_numpy(np.float64)
    b = pd.read_csv("/mnt/v/output/zensim/jnd-reconstruction-2026-10-19-b/JND_curve.scored.csv").set_index("distorted").score_b
    jobs = [("aic_train|B|zero", b.loc[keys].to_numpy(), y)]
    meta = [dict(dataset="aic2026_public_train", model="B", sample="zero-shot (no AIC fit)", seed=None)]
    full = pd.read_parquet(OUT / "table_raw/table.parquet", columns=["key"]).key.astype(str).tolist()
    for name, path in (("Tfull", "full/mono-thr_s0/refit/production-f16.bin"),
                       ("Mfull", "full/mono-thr-mp32_s0/refit/production-f16.bin")):
        sys.path.insert(0, str(HERE.parent / "rev4_featpot"))
        from v2_lodo_mlp import predict as owner_predict  # noqa: E402
        with tempfile.TemporaryDirectory(dir=Path.home() / "tmp") as td:
            pr = owner_predict(Path("/mnt/v/output/zensim/byconstruction-2026-10-22") / path,
                               OUT / "table_raw/table.parquet", Path(td) / "p.tsv")
        jobs.append((f"aic_train|{name}|zero", pd.Series(pr, index=full).loc[keys].to_numpy(), y))
        meta.append(dict(dataset="aic2026_public_train", model=f"{name} (old data, no AIC)", sample="zero-shot (no AIC fit)", seed=None))
    for arm in ("T", "M", "S"):
        d = OUT / "rust/folds_train" / arm
        if not (d / "oof_panel.json").is_file():
            continue
        s = pd.Series(np.nan, index=keys)
        for k in range(7):
            kk = pd.read_parquet(F / f"aic_k{k}_test.parquet", columns=["key"]).key.astype(str).to_numpy()
            s.loc[kk] = pd.read_csv(d / f"k{k}" / "pred.tsv", sep="\t").pred.to_numpy()
        jobs.append((f"aic_train|{arm}|oof", s.to_numpy(), y))
        meta.append(dict(dataset="aic2026_public_train", model=f"{arm} (AIC only)", sample="out of fold (7 source folds)", seed=None))
    ins = OUT / "rust/folds_train/T_full/pred_insample.tsv"
    if ins.is_file():
        order = pd.read_parquet(F / "aic_train_all.parquet", columns=["key"]).key.astype(str)
        s = pd.Series(pd.read_csv(ins, sep="\t").pred.to_numpy(), index=order).loc[keys]
        jobs.append(("aic_train|T|in", s.to_numpy(), y))
        meta.append(dict(dataset="aic2026_public_train", model="T (AIC only)", sample="in sample (fitted on all of it)", seed=None))
    for arm in sorted(p.name for p in CELLS.iterdir() if (p / "without_kadid_s0" / "byc_record.json").is_file()):
        recs = [json.loads(Path(r).read_text()) for r in glob.glob(str(CELLS / arm / "*/byc_record.json"))]
        if not recs or "aic_fold" not in recs[0] or len(recs) < 12:
            continue
        by = {}
        for r in recs:
            by.setdefault(r["aic_fold"], []).append(
                pd.read_csv(CELLS / arm / f"without_{r['fold']}_s{r['seed']}" / f"pred_aic_k{r['aic_fold']}.tsv",
                            sep="\t").pred.to_numpy())
        s = pd.Series(np.nan, index=keys)
        for k, ps in by.items():
            kk = pd.read_parquet(F / f"aic_k{k}_test.parquet", columns=["key"]).key.astype(str).to_numpy()
            s.loc[kk] = np.mean(ps, 0)
        jobs.append((f"aic_train|{arm}|oof", s.to_numpy(), y))
        meta.append(dict(dataset="aic2026_public_train", model=f"{arm} (E21 + AIC)", sample="out of fold (rotating source fold)", seed=None))
    if public_test:  # the one frozen-B read; pairs_test.tsv is opened only here
        pte = pd.read_csv(F / "pairs_test.tsv", sep="\t")
        tkeys = pte.key.astype(str)
        jobs.append(("aic_test|B|zero", b.loc[tkeys].to_numpy(), -pte.jnd.to_numpy(np.float64)))
        meta.append(dict(dataset="aic2026_public_test", model="B", sample="zero-shot, frozen; the one public_test read",
                         seed=None))
    return jobs, meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default="base-c,Tghpm,Tga,Thpm,Mhpm")
    ap.add_argument("--dvifm", action="store_true", help="add DVIFM 0.3 on the E21 sets (dvifm_e21.py outputs)")
    ap.add_argument("--geo", action="store_true", help="add scatter geometry and logistic-fit error tails per row")
    ap.add_argument("--per-type", action="store_true", help="add E21 statistics per distortion type")
    ap.add_argument("--no-public-test", action="store_true",
                    help="leave out the frozen-B public_test row (reuse panel_public_test_frozen.json instead)")
    a = ap.parse_args()
    dv, types = dvifm_e21() if (a.dvifm or a.per_type) else (None, None)
    j1, m1 = e21_jobs(a.arms.split(","), dv if a.dvifm else None)
    j2, m2 = aic_jobs(public_test=not a.no_public_test)
    res = panel_batch(j1 + j2, stats="full")
    rows = []
    for m, r, (label, x, y) in zip(m1 + m2, res, j1 + j2):
        rows.append({**m, "n": int(r["n"]), **{k: float(r[k]) for k in STATS},
                     **(geometry(x, y) if a.geo else {})})
    keep = STATS + (GEOMETRY if a.geo else ())

    def seed_mean(df, by):  # E21 out of sample: mean over seeds
        return [{**dict(zip(by, k)), "n": int(g.n.iloc[0]), "seeds": int(len(g)),
                 **{s_: (None if g[s_].isna().all() else float(g[s_].mean())) for s_ in keep if s_ in g}}
                for k, g in df.groupby(list(by), sort=False)]
    out = seed_mean(pd.DataFrame(rows), ("dataset", "model", "sample"))
    per_type = []
    if a.per_type:
        tj, tm = [], []
        for (label, x, y), m in zip(j1, m1):
            if not m["sample"].startswith(("out", "zero")):
                continue
            t = types[m["dataset"]]
            for ty in sorted(set(t)):
                idx = t == ty
                tj.append((f"{label}|{ty}", np.asarray(x)[idx], np.asarray(y)[idx]))
                tm.append({**m, "type": ty})
        tr = [{**m, "n": int(r["n"]), **{k: float(r[k]) for k in STATS}}
              for m, r in zip(tm, panel_batch(tj, stats="full"))]
        per_type = seed_mean(pd.DataFrame(tr), ("dataset", "type", "model", "sample"))
    Path(a.out).write_text(json.dumps({"stats": list(keep), "rows": out, "per_type": per_type, "per_seed": rows},
                                      indent=1) + "\n")
    for r in out:
        print(f"{r['dataset']:22s} {r['model']:24s} {r['sample'][:38]:38s} n={r['n']:5d} srocc={r['srocc_signed']:+.4f} "
              f"plcc={r['plcc']:.4f} krocc={r['krocc']:.4f} or={r['or']:.4f} pwrc={r['pwrc']:.4f} zrmse={r['z_rmse']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
