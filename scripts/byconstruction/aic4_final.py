"""Final read: frozen bakes on the AIC-4 public testing population, the protocol of the owner's comparison table (AIC-4
framework benchmark, screenshot 2026-10-22 011958): SRCC, PLCC and RMSE against `JND_curve` of
`jnd_reconstruction_2026-10-21.csv` on the 1,848 public testing crops (14 sources).

Exploration record: benchmarks/byconstruction_2026-10-22.md ("Final test"). Each bake is served through
`score_pair_with_bake --bake-post raw` (revision-6 bakes stamped by `bake_stamp_revision`). Its raw score is mapped to
JND units by a monotone 4-parameter logistic fitted on the public *training* crops only (the AIC-4 split file), then
frozen; SRCC needs no mapping. Public testing labels are read once per frozen model, here.

  python3 aic4_final.py --bin-dir DIR --bake NAME=PATH [--bake ...] --out OUT.json [--jobs 12]
"""

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr

DATA = Path("/mnt/v/zenmetrics-private/aic4/models/ecec4e0/data")
PAIRS = Path("/mnt/v/output/zensim/byconstruction-2026-10-22/aic2026train/folds_train")


def logistic(x, a, b, c, d):
    return a + (b - a) / (1.0 + np.exp(-(x - c) / d))


def pairs():
    splits = pd.read_csv(DATA / "aic4_splits.csv")
    gt = pd.read_csv(DATA / "jnd_reconstruction_2026-10-21.csv", usecols=["source", "distorted", "JND_curve"])
    paths = pd.concat([pd.read_csv(PAIRS / f, sep="\t", usecols=["key", "ref_path", "dist_path"])
                       for f in ("pairs_train.tsv", "pairs_test.tsv")])
    df = gt.merge(splits, on="source").merge(paths, left_on="distorted", right_on="key", how="left")
    if df.ref_path.isna().any():
        raise SystemExit(f"{int(df.ref_path.isna().sum())} ground-truth rows have no image pair")
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin-dir", required=True)
    ap.add_argument("--bake", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=12)
    a = ap.parse_args()
    bin_dir = Path(a.bin_dir)
    df = pairs()
    env = {k: v for k, v in os.environ.items() if k not in ("ZENSIM_RING_FORM", "ZENSIM_FORMULA_REV")}
    res = {"population": {"n_train": int((df.split == "public_training").sum()),
                          "n_test": int((df.split == "public_testing").sum()),
                          "ground_truth": str(DATA / "jnd_reconstruction_2026-10-21.csv"),
                          "ground_truth_sha256": hashlib.sha256((DATA / "jnd_reconstruction_2026-10-21.csv").read_bytes()).hexdigest()},
           "models": {}}
    for spec in a.bake:
        name, path = spec.split("=", 1)

        def one(i, path=path):
            r = subprocess.run([str(bin_dir / "score_pair_with_bake"), "--bake", path, "--bake-post", "raw",
                                "--ref", df.ref_path.iloc[i], "--dist", df.dist_path.iloc[i]],
                               capture_output=True, text=True, check=True, env=env)
            return float(r.stdout.strip().splitlines()[-1])

        with cf.ThreadPoolExecutor(a.jobs) as ex:
            raw = np.array(list(ex.map(one, range(len(df)))))
        tr, te = (df.split == "public_training").to_numpy(), (df.split == "public_testing").to_numpy()
        y = df.JND_curve.to_numpy(dtype=np.float64)
        p0 = [y[tr].max(), y[tr].min(), float(np.median(raw[tr])), float(np.std(raw[tr]) / 4 or 1.0)]
        coef, _ = curve_fit(logistic, raw[tr], y[tr], p0=p0, maxfev=200000)
        jnd = logistic(raw, *coef)
        m = {"bake": path, "bake_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
             "remap": {"form": "4-parameter logistic, fitted on public_training", "coef": [float(c) for c in coef]},
             "train": {"srcc": float(spearmanr(-raw[tr], y[tr]).statistic),
                       "plcc": float(pearsonr(jnd[tr], y[tr]).statistic),
                       "rmse": float(np.sqrt(np.mean((jnd[tr] - y[tr]) ** 2)))},
             "test": {"srcc": float(spearmanr(-raw[te], y[te]).statistic),
                      "plcc": float(pearsonr(jnd[te], y[te]).statistic),
                      "rmse": float(np.sqrt(np.mean((jnd[te] - y[te]) ** 2)))}}
        res["models"][name] = m
        print(name, json.dumps({"train": m["train"], "test": m["test"]}))
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
