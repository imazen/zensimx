"""TRAIN-only analysis of the two preregistered Mlic continuation arms.

Requires the completed guarded launch. Reuses frozen E21 and mixture reference
predictions; computes only the missing same-source-fold Mlic AIC reference.
No EVAL or public_test reader is called. The existing statistic and diagnostic
owners provide the reported metrics.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ARMS = ("MlicFT8x4", "MlicFT16x8")
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--jobs", choices=range(1, 9), type=int, default=8)
    args = parser.parse_args()
    study = args.study.resolve()
    packet = json.loads(args.packet.read_text())
    if json.loads((study / "TRAIN_COMPLETE.json").read_text())["registered_cells"] != 24:
        raise ValueError("all 24 registered cells must complete before analysis")
    frozen = json.loads((study / "BIN_FROZEN.json").read_text())
    for name, pin in frozen["binaries"].items():
        if sha(study / "bin" / name) != pin:
            raise ValueError(f"frozen binary differs: {name}")
    os.environ.update(BYC_ROOT=packet["root"], REV4_V2_ROOT=packet["root"],
        BYC_CELLS=str(study / "cells"), REV4_V2_BIN_DIR=str(study / "bin"),
        AIC_BIN=str(study / "bin"), ZEN_PANEL_BIN=str(study / "bin/panel"),
        BYC_LABEL_LEDGER_DIR=str(study / "analysis/label-reads"),
        ZENSIM_MAX_TIER="v3", ZENSIM_RING_FORM="raw", ZENSIM_PYRAMID="xyb",
        RAYON_NUM_THREADS="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    # Environment must precede imports: the recipe owners bind roots and tools.
    import numpy as np
    import pandas as pd
    import byc_cells as cells
    sys.path.insert(0, str(HERE.parent))
    sys.path.insert(0, str(HERE.parent / "byconstruction"))
    from lib.zen_stats import panel_batch
    from aic_train import panel
    from aic_panel import geometry
    from subjective_training_leg import log_dataset_read
    from v2_human_role import D6_INPUT_PINS

    out = study / "analysis"
    out.mkdir()
    baseline = study / "cells/Mlic"
    baseline.symlink_to(Path(packet["initial_cells"]).resolve(), target_is_directory=True)
    pairs_path = Path(os.environ["AIC_PAIRS_TRAIN"])
    folds = Path(os.environ["AIC_FOLDS"])
    if sha(folds / "_MANIFEST.json") != packet["fold_manifest_sha256"]:
        raise ValueError("frozen AIC fold inventory differs")
    if sha(pairs_path) != D6_INPUT_PINS["pairs_sha256"]:
        raise ValueError("public_train pairs pin differs")
    log_dataset_read(out, pairs_path, 3920, ["key", "group", "codec", "jnd"],
                     "D6-public_train-continuation-OOF-analysis", table_sha256=sha(pairs_path))
    pairs = pd.read_csv(pairs_path, sep="\t")
    if len(pairs) != 3920 or pairs.key.duplicated().any():
        raise ValueError("public_train population differs")

    def command(name, argv):
        print(json.dumps(dict(stage=name, command=[str(x) for x in argv])), flush=True)
        with (out / f"{name}.log").open("x") as stream:
            subprocess.run([str(x) for x in argv], check=True, stdout=stream,
                           stderr=subprocess.STDOUT)

    def prediction(path, count):
        table = pd.read_csv(path, sep="\t")
        if list(table.columns) != ["row_idx", "pred"] or not np.array_equal(table.row_idx, np.arange(count)):
            raise ValueError(f"prediction order differs: {path}")
        values = table.pred.to_numpy(np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"nonfinite prediction: {path}")
        return values

    aic_results, metrics_jobs, metrics_labels = {}, [], []
    reference_cache = out / "Mlic-same-fold"
    reference_cache.mkdir()
    fold_indices = {}
    for k in range(7):
        keys = pd.read_parquet(folds / f"aic_k{k}_test.parquet", columns=["key"]).key.astype(str)
        index = pairs.key.astype(str).to_list()
        position = {key: i for i, key in enumerate(index)}
        fold_indices[k] = np.asarray([position[key] for key in keys], dtype=int)
    if sorted(np.concatenate(list(fold_indices.values())).tolist()) != list(range(3920)):
        raise ValueError("OOF key partition differs")
    old_predictions = {}
    for seed in range(3):
        for fold in cells.FOLDS:
            identity = f"without_{fold}_s{seed}"
            k = cells.aic_fold(fold, seed)
            initial = Path(packet["initial_cells"]) / identity / "refit/last.bin"
            dest = reference_cache / identity
            dest.mkdir()
            path = dest / f"pred_aic_k{k}.tsv"
            values = cells.predict_cell(initial, folds / f"aic_k{k}_test.parquet", path,
                                       cells.spec_of("mono:thr:mp32"))
            old_predictions[identity] = prediction(path, len(fold_indices[k]))
            if not np.array_equal(values, old_predictions[identity]):
                raise ValueError("Mlic same-fold reference changed during readback")
            write(dest / "PROVENANCE.json", dict(checkpoint=str(initial), checkpoint_sha256=sha(initial),
                table_sha256=sha(folds / f"aic_k{k}_test.parquet"), prediction_sha256=sha(path),
                aic_fold=k, tool_sha256=frozen["binaries"]["bake_dial_refit"]))
    for arm in ("Mlic", *ARMS):
        per_cell, fold_predictions = {}, {k: [] for k in range(7)}
        for seed in range(3):
            for fold in cells.FOLDS:
                identity = f"without_{fold}_s{seed}"
                k = cells.aic_fold(fold, seed)
                idx = fold_indices[k]
                subset = pairs.iloc[idx].reset_index(drop=True)
                if arm == "Mlic":
                    values = old_predictions[identity]
                else:
                    dest = study / "cells" / arm / identity
                    record = json.loads((dest / "byc_record.json").read_text())
                    path = dest / f"pred_aic_k{k}.tsv"
                    if record["aic_fold"] != k or sha(path) != record["aic_pred_sha256"]:
                        raise ValueError("registered AIC cell prediction pin differs")
                    values = prediction(path, len(idx))
                reference = panel(old_predictions[identity], subset)
                result = panel(values, subset)
                result["global_delta"] = result["global"] - reference["global"]
                result["aic_fold"] = k
                result["codec_bias"] = cells.codec_rank_bias(values, -subset.jnd.to_numpy(), subset.codec.to_numpy())
                per_cell[identity] = result
                fold_predictions[k].append(values)
                label = f"aic|{arm}|{identity}"
                metrics_jobs.append((label, values, -subset.jnd.to_numpy()))
                metrics_labels.append(dict(label=label, geometry=geometry(values, -subset.jnd.to_numpy())))
        pooled = np.full(3920, np.nan)
        for k, values in fold_predictions.items():
            pooled[fold_indices[k]] = np.mean(values, axis=0)
        if not np.isfinite(pooled).all():
            raise ValueError("OOF prediction coverage is incomplete")
        row_file = out / f"{arm}-OOF.tsv"
        pd.DataFrame(dict(key=pairs.key, pred=pooled)).to_csv(row_file, sep="\t", index=False)
        aic_results[arm] = dict(per_cell=per_cell,
            mean_cell_global_gain=float(np.mean([v["global_delta"] for v in per_cell.values()])),
            mean_cell_global=float(np.mean([v["global"] for v in per_cell.values()])),
            oof=panel(pooled, pairs), cells_per_fold={k: len(v) for k, v in fold_predictions.items()},
            oof_prediction_sha256=sha(row_file))
    write(out / "AIC_OOF.json", aic_results)
    statistics = panel_batch(metrics_jobs, stats="full")
    write(out / "AIC_DIAGNOSTICS.json", [dict(**m, statistics=r) for m, r in zip(metrics_labels, statistics, strict=True)])
    command("e21", [sys.executable, "-B", HERE / "byc_cells.py", "score", "--arms", ",".join(("Mlic", *ARMS)),
        "--control", "base-c"])
    # The owner emits JSON directly; preserve stdout as the measured E21 receipt.
    measured = json.loads((out / "e21.log").read_text())
    e21 = {}
    for arm in ARMS:
        absolute = measured[arm]["per_dataset_per_seed_srocc"]
        old = measured["Mlic"]["per_dataset_per_seed_srocc"]
        seed_delta = [float(np.mean([absolute[f][s] - old[f][s] for f in cells.FOLDS])) for s in range(3)]
        e21[arm] = dict(control="Mlic", per_seed_delta=seed_delta, delta=float(np.mean(seed_delta)),
            seed_unit_se=float(np.std(seed_delta, ddof=1) / np.sqrt(3)),
            per_dataset_delta={f: float(np.mean(np.asarray(absolute[f]) - np.asarray(old[f]))) for f in cells.FOLDS},
            per_dataset_per_seed_srocc=absolute, control_per_dataset_per_seed_srocc=old)
    write(out / "E21.json", e21)
    command("knob-inversions", [sys.executable, "-B", HERE / "byc_cells.py", "knobinv",
        "--arms", ",".join(("Mlic", *ARMS)), "--out", out / "KNOB_INVERSIONS.json"])
    command("type-bias", [sys.executable, "-B", HERE.parent / "byconstruction/type_bias.py",
        "Mlic", *ARMS, "--out", out / "TYPE_BIAS.json"])
    command("e21-panel", [sys.executable, "-B", HERE.parent / "byconstruction/aic_panel.py",
        "--arms", ",".join(("Mlic", *ARMS)), "--geo", "--no-aic", "--no-public-test",
        "--out", out / "E21_DIAGNOSTICS.json"])
    mixtures = {}
    for name in ("mixture", "mixture_global"):
        source = Path(packet["initial_cells"]).parents[1] / name
        dest = out / name
        dest.mkdir()
        shutil.copyfile(source / "manifest.json", dest / "manifest.json")
        (dest / "img").symlink_to((source / "img").resolve(), target_is_directory=True)
        for arm in ("Mlic", *ARMS):
            for seed in range(3):
                for fold in cells.FOLDS:
                    label = f"{arm}_{fold}_s{seed}"
                    score_file = f"scores_{label}.json"
                    if arm == "Mlic":
                        shutil.copyfile(source / score_file, dest / score_file)
                    else:
                        cell = study / "cells" / arm / f"without_{fold}_s{seed}"
                        dense = list((cell / "dense").glob("*.bin"))
                        if len(dense) != 1:
                            raise ValueError("one owner-produced dense bake required")
                        command(f"{name}-{label}", [sys.executable, "-B", HERE.parent / "byconstruction/mixture_test.py",
                            "score", "--out", dest, "--bake", f"{label}={dense[0]}",
                            "--bin-dir", study / "bin", "--jobs", str(args.jobs)])
        command(name + "-summary", [sys.executable, "-B", HERE.parent / "byconstruction/mixture_test.py",
            "summary", "--out", dest, "--json-out", out / f"{name}.json", "Mlic", *ARMS])
        mixtures[name] = json.loads((out / f"{name}.json").read_text())
    qualified = []
    decisions = {}
    for arm, epochs, weight in ((ARMS[0], 4, 8), (ARMS[1], 8, 16)):
        e = e21[arm]
        zero = mixtures["mixture_global"][arm]["inversions_max"] == 0
        old_ok = e["delta"] >= -2 * e["seed_unit_se"]
        decisions[arm] = dict(zero_global_inversions=zero, e21_within_two_seed_se=old_ok,
            e21_delta=e["delta"], e21_seed_se=e["seed_unit_se"],
            aic_mean_same_fold_global_gain=aic_results[arm]["mean_cell_global_gain"],
            qualifies=zero and old_ok)
        if zero and old_ok:
            qualified.append((aic_results[arm]["mean_cell_global_gain"], -epochs, -weight, arm))
    winner = max(qualified)[3] if qualified else None
    write(out / "TRAIN_SELECTION.json", dict(winner=winner, arms=decisions,
        rule="zero global-edit inversions in every cell; E21 mean >= -2 paired seed SE; largest mean of 12 same-fold AIC global gains; exact ties fewer epochs then lower weight",
        preflight_sha256=sha(study / "PREFLIGHT_PASSED.json"),
        training_complete_sha256=sha(study / "TRAIN_COMPLETE.json")))
    print(json.dumps(dict(winner=winner, arms=decisions), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
