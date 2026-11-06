"""Run one Rev4 POTENTIAL linear baseline cell via the Rust fit/stat owners.

Run this entire script under ``~/tmp/devin/heavy``. It never opens bank files:
input is the already admitted, pair_key-checked diagnostic table. Per-fit
normalization is learned only on fit references, then fed to the Rust Gram and
lasso/BVLS owners. No bake is emitted.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# The shaped-study dispatch must set BLAS limits before NumPy initializes.
if len(sys.argv) > 1 and sys.argv[1] == "shaped-study":
    for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS'):
        os.environ[_key] = '1'

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from data import FEATURES, load

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.zen_stats import panel_batch  # noqa: E402


ROOT = Path("/var/tmp/rev4-featpot")
BIN = Path(os.environ.get("REV4_FIT_BIN", str(ROOT / "target/debug/bake_dial_refit")))
REPO = Path(__file__).resolve().parents[2]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(cmd: list[str], log: Path) -> None:
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log.write_text("$ " + " ".join(cmd) + "\n" + proc.stdout)
    if proc.returncode:
        raise RuntimeError(f"{cmd[1]} failed rc={proc.returncode}; see {log}")


def fold_table(data: pd.DataFrame, fit_idx: np.ndarray, out: Path) -> tuple[float, float]:
    y = data.loc[fit_idx, "target"].to_numpy(dtype=np.float64)
    lo, hi = np.quantile(y, [0.001, 0.999])
    if not np.isfinite([lo, hi]).all() or hi <= lo:
        raise ValueError(f"bad fit-fold target bounds {lo}, {hi}")
    tab = data.loc[fit_idx, ["ref_basename"] + FEATURES].copy()
    tab["target"] = y
    tab["target_norm"] = np.clip((y - lo) / (hi - lo), 0.0, 1.0)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(tab, preserve_index=False), out, compression="zstd")
    return float(lo), float(hi)


def gram(table: Path, out: Path, per_ref: bool = False, n_features: int = 944) -> None:
    if per_ref:
        raw = out.with_name(out.stem + "_raw.npz")
        run([str(BIN), "gram", "--parquet", str(table), "--target", "target",
             "--expect-n-feat", str(n_features), "--out", str(raw),
             "--per-reference-out-dir", str(out.with_suffix(".refs"))],
            raw.with_suffix(".log"))
    cmd = [str(BIN), "gram", "--parquet", str(table), "--target", "target",
           "--target-minmax01", "--expect-n-feat", str(n_features), "--out", str(out)]
    run(cmd, out.with_suffix(".log"))


def lasso_path(g: Path, out: Path, arm: str, slice_file: Path) -> dict:
    cmd = [str(BIN), "fit-lasso", "--gram", str(g), "--space", "raw",
           "--target", "target__mm01", "--lam", "0", "--path-out", str(out),
           "--out", str(out.with_suffix(".unused.bin"))]
    if arm == "minus_basic" or arm.removesuffix("_perm") == "a1w":
        cmd += ["--slice-file", str(slice_file)]
    run(cmd, out.with_suffix(".log"))
    value = json.loads(out.read_text())
    assert value["schema"] == "rev4-featpot-lasso-path-v1" and len(value["fits"]) == 50
    return value


def predict_path(path: dict, x: np.ndarray) -> np.ndarray:
    w = np.asarray([fit["w"] for fit in path["fits"]], dtype=np.float64)
    mu = np.asarray(path["mu"], dtype=np.float64)
    sd = np.asarray(path["sd"], dtype=np.float64)
    return (x - mu) / sd @ w.T + path["bias"]


def choose_1se(inner_scores: list[list[float]]) -> tuple[int, dict]:
    values = np.asarray(inner_scores, dtype=np.float64)
    if values.shape != (4, 50) or not np.isfinite(values).all():
        raise ValueError(f"inner score shape/nonfinite: {values.shape}")
    mean = values.mean(axis=0)
    best = int(np.argmax(mean))
    se = values[:, best].std(ddof=1) / np.sqrt(4)
    eligible = np.flatnonzero(mean >= mean[best] - se)
    # The path runs lambda_max -> lambda_min, so first eligible is most sparse.
    selected = int(eligible[0])
    return selected, {"best_index": best, "selected_index": selected,
                      "best_mean": float(mean[best]), "one_se": float(se),
                      "mean_srocc_by_index": mean.tolist()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", required=True)
    ap.add_argument("--arm", choices=["r0", "minus_basic"], required=True)
    args = ap.parse_args()
    if not BIN.is_file():
        raise FileNotFoundError(f"build {BIN} through the heavy runner first")
    folds = json.loads((ROOT / "folds.json").read_text())["sets"][args.set]
    admitted = ROOT / "admitted" / f"POT_{args.set}_rev3_944.parquet"
    data, label_meta = load(args.set)
    label = label_meta["target"]
    x = data[FEATURES].to_numpy(dtype=np.float64, copy=False)
    y = data["target"].to_numpy(dtype=np.float64)
    refs = data["ref_basename"].astype(str).to_numpy()
    dest = ROOT / "fits" / f"POT_{args.set}_{args.arm}_linear"
    dest.mkdir(parents=True, exist_ok=True)
    slice_file = dest / "keep_f228_to_f943.txt"
    slice_file.write_text("\n".join(map(str, range(228, 944))) + "\n")
    result = {"schema": "rev4-featpot-linear-v1", "set": args.set, "arm": args.arm,
              "input_sha256": sha(admitted), "target": label,
              "label_source": label_meta,
              "label": "POTENTIAL — ceiling, not a model score", "outer": []}

    def indices(allowed: list[str]) -> np.ndarray:
        return np.flatnonzero(np.isin(refs, allowed))

    for outer_i, outer in enumerate(folds["outer"]):
        test = indices(outer["test_refs"])
        fit_refs = sorted(set(folds["full_refs"]) - set(outer["test_refs"]))
        fit = indices(fit_refs)
        inner_scores = []
        for inner_i, val_refs in enumerate(outer["inner_val_refs"]):
            train = indices(sorted(set(fit_refs) - set(val_refs)))
            val = indices(val_refs)
            stem = dest / f"o{outer_i}_i{inner_i}"
            bounds = fold_table(data, train, stem.with_suffix(".parquet"))
            gram(stem.with_suffix(".parquet"), stem.with_suffix(".npz"))
            path = lasso_path(stem.with_suffix(".npz"), stem.with_suffix(".json"), args.arm, slice_file)
            preds = predict_path(path, x[val])
            jobs = [(f"lambda_{i}", preds[:, i], y[val]) for i in range(50)]
            scores = panel_batch(jobs, stats="srocc")
            inner_scores.append([float(s["srocc"]) for s in scores])
            print(json.dumps({"outer": outer_i, "inner": inner_i, "train_rows": len(train),
                              "val_rows": len(val), "bounds": bounds,
                              "gram_sha256": sha(stem.with_suffix(".npz")),
                              "path_sha256": sha(stem.with_suffix(".json"))}), flush=True)
        chosen, selection = choose_1se(inner_scores)
        stem = dest / f"o{outer_i}_fit"
        bounds = fold_table(data, fit, stem.with_suffix(".parquet"))
        gram(stem.with_suffix(".parquet"), stem.with_suffix(".npz"))
        path = lasso_path(stem.with_suffix(".npz"), stem.with_suffix(".json"), args.arm, slice_file)
        pred = predict_path(path, x[test])[:, chosen]
        score = panel_batch([(f"outer_{outer_i}", pred, y[test])], stats="full")[0]
        result["outer"].append({"fold": outer_i, "fit_rows": len(fit), "test_rows": len(test),
                                "fit_refs": len(fit_refs), "test_refs": len(outer["test_refs"]),
                                "bounds": bounds, "selection": selection, "score": score,
                                "prediction": pred.tolist(), "test_index": test.tolist(),
                                "gram_sha256": sha(stem.with_suffix(".npz")),
                                "path_sha256": sha(stem.with_suffix(".json"))})
        print(json.dumps({"outer": outer_i, "selected_index": chosen,
                          "test_rows": len(test), "srocc": score["srocc"]}), flush=True)
    oof = np.full(len(data), np.nan, dtype=np.float64)
    for fold in result["outer"]:
        oof[np.asarray(fold["test_index"], dtype=np.int64)] = fold["prediction"]
    if not np.isfinite(oof).all():
        raise ValueError("outer folds did not cover every row exactly once")
    result["nested"] = {"score": panel_batch([("nested", oof, y)], stats="full")[0],
                        "prediction": oof.tolist()}

    full_inner_scores = []
    for inner_i, val_refs in enumerate(folds["full_inner_val_refs"]):
        train = indices(sorted(set(folds["full_refs"]) - set(val_refs)))
        val = indices(val_refs)
        stem = dest / f"full_i{inner_i}"
        bounds = fold_table(data, train, stem.with_suffix(".parquet"))
        gram(stem.with_suffix(".parquet"), stem.with_suffix(".npz"))
        path = lasso_path(stem.with_suffix(".npz"), stem.with_suffix(".json"), args.arm, slice_file)
        preds = predict_path(path, x[val])
        scores = panel_batch([(f"lambda_{i}", preds[:, i], y[val]) for i in range(50)], stats="srocc")
        full_inner_scores.append([float(s["srocc"]) for s in scores])
        print(json.dumps({"full_inner": inner_i, "train_rows": len(train),
                          "val_rows": len(val), "bounds": bounds,
                          "gram_sha256": sha(stem.with_suffix(".npz")),
                          "path_sha256": sha(stem.with_suffix(".json"))}), flush=True)
    chosen, selection = choose_1se(full_inner_scores)
    stem = dest / "full_fit"
    bounds = fold_table(data, np.arange(len(data)), stem.with_suffix(".parquet"))
    gram(stem.with_suffix(".parquet"), stem.with_suffix(".npz"), per_ref=True)
    path = lasso_path(stem.with_suffix(".npz"), stem.with_suffix(".json"), args.arm, slice_file)
    pred = predict_path(path, x)[:, chosen]
    result["in_sample"] = {"selection": selection, "bounds": bounds,
                           "score": panel_batch([("in_sample", pred, y)], stats="full")[0],
                           "prediction": pred.tolist(), "gram_sha256": sha(stem.with_suffix(".npz")),
                           "path_sha256": sha(stem.with_suffix(".json"))}
    result["gap_srocc"] = result["in_sample"]["score"]["srocc"] - result["nested"]["score"]["srocc"]
    out = dest / "result.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"result": str(out), "sha256": sha(out),
                      "nested_srocc": result["nested"]["score"]["srocc"],
                      "in_sample_srocc": result["in_sample"]["score"]["srocc"],
                      "gap_srocc": result["gap_srocc"]}), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "shaped-study":
        from shaped_linear_study import main as shaped_main
        shaped_main(sys.argv[2:])
    else:
        tmpdir = Path.home() / "tmp/rev4-featpot"
        tmpdir.mkdir(parents=True, exist_ok=True)
        os.environ["TMPDIR"] = str(tmpdir)
        tempfile.tempdir = str(tmpdir)
        main()
