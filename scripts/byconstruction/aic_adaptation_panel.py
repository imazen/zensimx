"""Separate fitted-source and held-out-source panels for AIC recipe adaptation.

All statistics use the existing canonical owners. This report has no selector
and cannot promote a model or authorize an EVAL/public_test read.
"""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aic_adaptation import FOLDS, RECIPES, SPEC, STRENGTHS, bound_worker, compute_lock, process_snapshot, process_tree, sha, write
from aic_adaptation import __file__ as coordinator_source


def finite_json(value):
    if isinstance(value, dict):
        return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [finite_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def role_masks(folds, heldout):
    import numpy as np
    folds = np.asarray(folds)
    if heldout not in range(7) or set(folds.tolist()) != set(range(7)):
        raise ValueError("all seven declared source folds required")
    return {"fitted": folds != heldout, "heldout": folds == heldout}


def role_average(rows, folds, role):
    import numpy as np
    if role not in ("fitted", "heldout") or len(rows) != 12:
        raise ValueError("a role average requires twelve matched omission cells")
    total, count = np.zeros(len(folds)), np.zeros(len(folds), dtype=int)
    seen = set()
    for row in rows:
        key = row["fold"], row["seed"]
        if key in seen or key[0] not in FOLDS or key[1] not in (0, 1, 2):
            raise ValueError("duplicated or unexpected omission cell")
        seen.add(key)
        k = (4 * row["seed"] + FOLDS.index(row["fold"])) % 7
        if row["aic_fold"] != k:
            raise ValueError("source fold rotation changed")
        values = np.asarray(row["values"], dtype=float)
        if values.shape != total.shape or not np.isfinite(values).all():
            raise ValueError("complete finite ordered prediction required")
        mask = role_masks(folds, k)[role]
        total[mask] += values[mask]
        count[mask] += 1
    if (count == 0).any():
        raise ValueError("role-specific source coverage incomplete")
    return total / count, count


def fits_finished(root, plan_pin):
    if (root / "FOLD_FITS_COMPLETE.json").exists():
        complete = json.loads((root / "FOLD_FITS_COMPLETE.json").read_text())
        if complete["plan_sha256"] != plan_pin or len(complete["cells"]) != 96:
            raise ValueError("preceding fit completion differs")
        return True
    state = json.loads((root / "RUN_STATE.json").read_text())
    if state["failures"]:
        raise RuntimeError("preceding adaptation failed; inspect its preserved failures")
    worker = state.get("worker", state["queue_process"])
    if not bound_worker(worker["pid"], worker["start_ticks"]):
        raise RuntimeError("preceding adaptation exited without completion")
    return False


def queue(plan_path, output):
    root, pin, owner_pin = plan_path.parent, sha(plan_path), sha(__file__)
    plan = json.loads(plan_path.read_text())
    if sha(coordinator_source) != plan["coordinator_sha256"]:
        raise ValueError("panel imported a different continuation owner")
    state = dict(worker=process_snapshot(os.getpid()), plan_sha256=pin, panel_owner_sha256=owner_pin,
                 output=str(output), stage="waiting-for-fold-fits", failure=None,
                 checked_utc=datetime.now(timezone.utc).isoformat())
    write(root / "PANEL_QUEUE.json", state, fresh=True)
    try:
        while not fits_finished(root, pin):
            time.sleep(30)
        if (sha(plan_path) != pin or sha(__file__) != owner_pin
                or sha(coordinator_source) != plan["coordinator_sha256"]):
            raise ValueError("queued panel source or input plan changed")
        command = ["flock", str(Path.home() / "tmp/heavy.lock"),
            str(Path.home() / "work/claudehints/scripts/run-heavy"), "--mem", "24G", "--jobs", "4", "--",
            sys.executable, "-B", str(Path(__file__).resolve()), "run", str(plan_path), str(output)]
        state.update(stage="queued-for-panels", command=command, checked_utc=datetime.now(timezone.utc).isoformat())
        write(root / "PANEL_QUEUE.json", state)
        with (root / "panel-launch.log").open("x") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    except BaseException as error:
        state.update(stage="failed", failure=repr(error), checked_utc=datetime.now(timezone.utc).isoformat())
        write(root / "PANEL_QUEUE.json", state)
        raise


def status(plan_path, output):
    root = plan_path.parent
    path = output / "PANEL_STATE.json"
    state = json.loads(path.read_text()) if path.exists() else json.loads((root / "PANEL_QUEUE.json").read_text())
    worker = state["worker"]
    live = bound_worker(worker["pid"], worker["start_ticks"])
    failures = state.get("failures", [state["failure"]] if state.get("failure") else [])
    if state["stage"] == "fold-panels-complete":
        if sha(output / "PANEL_COMPLETE.json") != state["completion_sha256"]:
            failures = [*failures, "panel completion receipt changed"]
    elif not live and not failures:
        failures = ["recorded panel process exited without completion"]
    value = dict(state, failures=failures, checked_utc=datetime.now(timezone.utc).isoformat(),
                 live_processes=process_tree(worker["pid"]) if live else [],
                 compute_lock=compute_lock(worker["pid"] if live else None))
    write(root / "CURRENT_PANEL.json", value)
    print(json.dumps({k: value.get(k) for k in ("stage", "completed_models", "required_models", "failures", "compute_lock")}))


def run(plan_path, output):
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    if not compute_lock(os.getpid())["held_by_worker_ancestor"]:
        raise ValueError("panel evaluation requires the shared compute lock")
    plan = json.loads(plan_path.read_text())
    root = plan_path.parent
    fits = json.loads((root / "FOLD_FITS_COMPLETE.json").read_text())
    if fits["plan_sha256"] != sha(plan_path) or len(fits["cells"]) != 96:
        raise ValueError("all registered continuation cells required")
    initial_path = root / "INITIAL_CHECKPOINTS.json"
    if sha(initial_path) != fits["initial_checkpoints_sha256"]:
        raise ValueError("starting inventory changed")
    initial = json.loads(initial_path.read_text())
    for name, pin in plan["input_sha256"].items():
        if sha(name) != pin:
            raise ValueError(f"frozen adaptation input changed: {name}")
    table = Path(plan["aic_source_table"])
    if sha(table) != plan["aic_source_table_sha256"]:
        raise ValueError("original revision-7 AIC extraction changed")
    source = Path(plan["warmstart_source"])
    os.environ.update(BYC_ROOT=plan["common_root"], REV4_V2_ROOT=plan["common_root"],
        REV4_V2_BIN_DIR=plan["bin"], AIC_BIN=plan["bin"], ZEN_PANEL_BIN=str(Path(plan["bin"]) / "panel"),
        BYC_COLS=plan["columns"], AIC_FOLDS=plan["aic_folds"], AIC_PAIRS_TRAIN=plan["aic_pairs"],
        AIC_PREDICT_FOLDS=plan["prediction_views"], BYC_CELLS=str(output / "cells"),
        BYC_LABEL_LEDGER_DIR=str(output / "label-reads"), TMPDIR=str(Path.home() / "tmp"),
        ZENSIM_MAX_TIER="v3", ZENSIM_RING_FORM="raw", ZENSIM_PYRAMID="xyb",
        RAYON_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    sys.path[:0] = [str(source / "scripts/rev4_featpot"), str(source / "scripts/byconstruction"), str(source / "scripts")]
    import byc_cells as cells
    from aic_panel import geometry
    from lib.zen_stats import panel_batch, panel_batch_indexed
    from subjective_training_leg import log_dataset_read
    from v2_human_role import D6_INPUT_PINS
    pairs_path = Path(plan["aic_pairs"])
    if sha(pairs_path) != D6_INPUT_PINS["pairs_sha256"]:
        raise ValueError("D6 public_train source differs")
    output.mkdir()
    state = dict(stage="read-admitted-TRAIN", worker=process_snapshot(os.getpid()), completed_models=0, required_models=144,
                 plan_sha256=sha(plan_path), panel_owner_sha256=sha(__file__), binary_sha256=plan["trainer_sha256"],
                 failures=[], objective_complete=False)
    def stage(name, **details):
        state.update(stage=name, checked_utc=datetime.now(timezone.utc).isoformat(), **details)
        write(output / "PANEL_STATE.json", state)
    def save(name, value):
        write(output / name, finite_json(value), fresh=True)
    def command(name, argv):
        stage(name)
        with (output / f"{name}.log").open("x") as log:
            subprocess.run([str(x) for x in argv], check=True, stdout=log, stderr=subprocess.STDOUT)
    try:
        stage("read-admitted-TRAIN")
        log_dataset_read(output, pairs_path, 3920, ["key", "group", "codec", "quality", "jnd"],
                         "D6-registered-recipe-adaptation-fitted-and-heldout-panels", table_sha256=sha(pairs_path))
        pairs = pd.read_csv(pairs_path, sep="\t")
        manifest = json.loads((Path(plan["aic_folds"]) / "_MANIFEST.json").read_text())
        assignment = manifest["assignment"]
        keys = pq.read_table(table, columns=["key", "ref_basename"])
        if (len(pairs) != 3920 or pairs.key.duplicated().any() or len(assignment) != 28
                or keys["key"].to_pylist() != pairs.key.tolist()
                or keys["ref_basename"].to_pylist() != pairs.group.tolist()
                or set(pairs.group) != set(assignment)
                or any(list(assignment.values()).count(k) != 4 for k in range(7))):
            raise ValueError("ordered public_train source/key population differs")
        folds = pairs.group.map(assignment).to_numpy(dtype=int)
        target = -pairs.jnd.to_numpy(dtype=float)
        if not np.isfinite(target).all():
            raise ValueError("nonfinite public_train response")
        (output / "cells").mkdir()
        (output / "predictions").mkdir()
        records = []
        for recipe in RECIPES:
            for row in initial[recipe]:
                if any(sha(name) != pin for name, pin in row["input_sha256"].items()):
                    raise ValueError("initial checkpoint evidence changed")
                records.append(dict(row, recipe=recipe, arm=recipe, aic_fold=(4 * row["seed"] + FOLDS.index(row["fold"])) % 7))
        for row in fits["cells"]:
            d = Path(row["directory"])
            if (sha(d / "refit/last.bin") != row["checkpoint_sha256"]
                    or sha(d / "byc_record.json") != row["record_sha256"]
                    or sha(d / "result.json") != row["result_sha256"]):
                raise ValueError("completed continuation evidence changed")
            records.append(row)
        by_arm, per_cell, batch = {}, [], []
        for row in records:
            d, arm = Path(row["directory"]), row["arm"]
            key = f"without_{row['fold']}_s{row['seed']}"
            link = output / "cells" / arm / key
            link.parent.mkdir(exist_ok=True)
            link.symlink_to(d, target_is_directory=True)
            dest = output / "predictions" / f"{arm}_{key}.tsv"
            stage("predict-all-public-train", current_model=f"{arm}/{key}")
            cells.predict_cell(d / "refit/last.bin", table, dest, SPEC)
            frame = pd.read_csv(dest, sep="\t")
            if list(frame.columns) != ["row_idx", "pred"] or not np.array_equal(frame.row_idx, np.arange(3920)):
                raise ValueError("AIC prediction order differs")
            values = frame.pred.to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError("nonfinite AIC prediction")
            if "epochs" in row:
                k = row["aic_fold"]
                old = pd.read_csv(d / f"pred_aic_k{k}.tsv", sep="\t").pred.to_numpy(dtype=float)
                if not np.array_equal(old, values[folds == k]):
                    raise ValueError("original-feature prediction differs from frozen held-out view")
            by_arm.setdefault(arm, []).append(dict(row, values=values))
            for role, mask in role_masks(folds, row["aic_fold"]).items():
                label = f"{arm}|{key}|{role}"
                batch.append((label, values[mask], target[mask]))
                per_cell.append(dict(arm=arm, recipe=row["recipe"], fold=row["fold"], seed=row["seed"], role=role,
                    sources=int(pairs.group[mask].nunique()), rows=int(mask.sum()), label=label,
                    geometry=geometry(values[mask], target[mask]), prediction_sha256=sha(dest)))
            state["completed_models"] += 1
        stage("canonical-cell-panels")
        statistics = panel_batch(batch, stats="full")
        for row, stats in zip(per_cell, statistics, strict=True):
            if row["label"] != stats["label"]:
                raise ValueError("canonical cell panel order differs")
            row["statistics"] = stats
        save("AIC_CELL_PANELS.json", per_cell)
        aggregates, jobs, descriptions = {}, [], []
        for arm, rows in by_arm.items():
            aggregates[arm] = {}
            for role in ("fitted", "heldout"):
                value, count = role_average(rows, folds, role)
                aggregates[arm][role] = value
                path = output / f"{arm}-{role}.tsv"
                pd.DataFrame(dict(key=pairs.key, pred=value, contributing_models=count)).to_csv(path, sep="\t", index=False)
                groups = [("all", "ALL", np.arange(3920))]
                for category, columns in (("source", ["group"]), ("codec", ["codec"]), ("ladder", ["group", "codec"])):
                    for group, index in pairs.groupby(columns, sort=True).indices.items():
                        groups.append((category, repr(group), np.asarray(index, dtype=int)))
                for category, group, index in groups:
                    label = f"{arm}|{role}|{category}|{group}"
                    jobs.append((label, value[index], target[index]))
                    descriptions.append(dict(arm=arm, role=role, category=category, group=group, label=label,
                        rows=len(index), geometry=geometry(value[index], target[index]) if category in ("all", "codec") else None,
                        contributions_min=int(count[index].min()), contributions_max=int(count[index].max())))
                save(f"{arm}-{role}-bias.json", cells.codec_rank_bias(value, target, pairs.codec.to_numpy()))
        stage("canonical-source-codec-ladder-panels")
        full = panel_batch(jobs, stats="full")
        for description, stats in zip(descriptions, full, strict=True):
            if description["label"] != stats["label"]:
                raise ValueError("canonical aggregate panel order differs")
            description["statistics"] = stats
        save("AIC_AGGREGATE_PANELS.json", dict(panels=descriptions,
            undefined_numeric_values="JSON null; denominators and native n_dropped retained; no invented replacement"))
        del batch, jobs
        stage("paired-seed-summaries")
        seed_results, contrasts = {}, []
        for recipe in RECIPES:
            for weight, epochs in STRENGTHS:
                arm = f"{recipe}FT{weight}x{epochs}"
                contrasts.append((arm, recipe))
                if recipe != "Mlic":
                    contrasts.append((arm, f"MlicFT{weight}x{epochs}"))
        for arm, control in contrasts:
            contrast = f"{arm}-minus-{control}"
            seed_results[contrast] = {}
            for role in ("fitted", "heldout"):
                baseline = {(r["fold"], r["seed"]): r for r in per_cell if r["arm"] == control and r["role"] == role}
                candidate = {(r["fold"], r["seed"]): r for r in per_cell if r["arm"] == arm and r["role"] == role}
                result = {}
                for metric in ("srocc_signed", "krocc", "plcc", "plcc_raw", "or", "pwrc", "z_rmse", "mae"):
                    delta = [float(np.mean([candidate[f, s]["statistics"][metric] - baseline[f, s]["statistics"][metric]
                                            for f in FOLDS])) for s in range(3)]
                    result[metric] = dict(per_seed_delta=delta, mean=float(np.mean(delta)),
                                          seed_unit_se=float(np.std(delta, ddof=1) / np.sqrt(3)))
                seed_results[contrast][role] = result
        save("AIC_PAIRED_SEEDS.json", seed_results)
        # Whole-source paired bootstrap over role-specific aggregate predictions.
        clusters = [np.asarray(index, dtype=int) for index in pairs.groupby("group", sort=True).indices.values()]
        rng = np.random.default_rng(1)
        draws = [np.concatenate([clusters[i] for i in rng.integers(0, 28, 28)]) for _ in range(2000)]
        intervals = {}
        for arm, control in contrasts:
            for role in ("fitted", "heldout"):
                stage("source-cluster-interval", current_model=f"{arm}-minus-{control}/{role}")
                bases = dict(target=target, baseline=aggregates[control][role], candidate=aggregates[arm][role])
                tasks = [(f"{i}-{name}", name, "target", index) for i, index in enumerate(draws)
                         for name in ("baseline", "candidate")]
                stats = panel_batch_indexed(bases, tasks, stats="srocc")
                delta = np.asarray([stats[i+1]["srocc_signed"] - stats[i]["srocc_signed"] for i in range(0, len(stats), 2)])
                good = delta[np.isfinite(delta)]
                intervals[f"{arm}-minus-{control}/{role}"] = dict(resamples=2000, seed=1, source_clusters=28,
                    defined_resamples=int(len(good)), undefined_resamples=int(2000-len(good)),
                    delta_interval_95=np.quantile(good, [0.025, 0.975]).tolist() if len(good) else None,
                    descriptive_only=True, selection_gate=False)
        save("AIC_SOURCE_INTERVALS.json", intervals)
        arms = ",".join(by_arm)
        command("e21-score", [sys.executable, "-B", source / "scripts/rev4_featpot/byc_cells.py",
            "score", "--arms", arms, "--control", "base-c"])
        e21 = json.loads((output / "e21-score.log").read_text())
        paired = {}
        for arm, control in contrasts:
            a = e21[arm]["per_dataset_per_seed_srocc"]
            b = e21[control]["per_dataset_per_seed_srocc"]
            delta = [float(np.mean([a[fold][s] - b[fold][s] for fold in FOLDS])) for s in range(3)]
            paired[f"{arm}-minus-{control}"] = dict(per_seed_delta=delta, mean=float(np.mean(delta)),
                seed_unit_se=float(np.std(delta, ddof=1) / np.sqrt(3)), selection_gate=False,
                per_dataset_delta={fold: float(np.mean(np.asarray(a[fold]) - np.asarray(b[fold]))) for fold in FOLDS})
        save("E21_PAIRED_SROCC.json", paired)
        command("e21-panel", [sys.executable, "-B", source / "scripts/byconstruction/aic_panel.py",
            "--arms", arms, "--geo", "--per-type", "--no-aic", "--no-public-test", "--out", output / "E21_PANELS.json"])
        command("knob-inversions", [sys.executable, "-B", source / "scripts/rev4_featpot/byc_cells.py",
            "knobinv", "--arms", arms, "--out", output / "KNOB_INVERSIONS.json"])
        command("type-bias", [sys.executable, "-B", source / "scripts/byconstruction/type_bias.py",
            *by_arm, "--out", output / "TYPE_BIAS.json"])
        prior_train = {}
        for directory in map(Path, plan["starting_studies"]):
            complete = json.loads((directory / "TRAIN_SCREEN_COMPLETE.json").read_text())
            selection = directory / "TRAIN_SELECTION.json"
            if sha(selection) != complete["selection_sha256"]:
                raise ValueError("starting TRAIN panel changed")
            prior_train[str(directory)] = dict(selection_sha256=sha(selection),
                stage="before-AIC-continuation", report=json.loads(selection.read_text()))
        save("INITIAL_TRAIN_PANELS.json", dict(studies=prior_train,
            post_AIC_new_collection_omission_cells="not part of this 12-cell E21 continuation stage; not measured"))
        # Reuse exact baseline and prior Mlic-continuation mixtures; score the new continuations.
        for name in ("mixture", "mixture_global"):
            original = Path(plan["initial_cells"]["Mlic"]).parents[1] / name
            dest = output / name
            dest.mkdir()
            (dest / "manifest.json").write_bytes((original / "manifest.json").read_bytes())
            (dest / "img").symlink_to((original / "img").resolve(), target_is_directory=True)
            for row in records:
                label = f"{row['arm']}_{row['fold']}_s{row['seed']}"
                if "epochs" not in row:
                    prior = Path(plan["initial_cells"][row["recipe"]]).parents[1] / name
                elif row["recipe"] == "Mlic":
                    prior = Path(plan["reuse_mlic"]) / name
                else:
                    prior = None
                if prior is not None:
                    if (prior / "manifest.json").read_bytes() != (original / "manifest.json").read_bytes():
                        raise ValueError("mixture inventories differ")
                    (dest / f"scores_{label}.json").write_bytes((prior / f"scores_{label}.json").read_bytes())
                else:
                    model = Path(row["directory"])
                    dense = list((model / "dense").glob("*.bin"))
                    if len(dense) != 1:
                        raise ValueError("one owner-produced dense checkpoint required")
                    command(f"{name}-{label}", [sys.executable, "-B", source / "scripts/byconstruction/mixture_test.py",
                        "score", "--out", dest, "--bake", f"{label}={dense[0]}", "--bin-dir", plan["bin"], "--jobs", "3"])
            command(name + "-summary", [sys.executable, "-B", source / "scripts/byconstruction/mixture_test.py",
                "summary", "--out", dest, "--json-out", output / f"{name}.json", *by_arm])
        save("PANEL_COMPLETE.json", dict(plan_sha256=sha(plan_path), fold_fits_sha256=sha(root / "FOLD_FITS_COMPLETE.json"),
            panel_owner_sha256=sha(__file__), selector=None, full_data_followthrough="pending",
            production_qualified=False, objective_complete=False,
            outputs_sha256={p.name: sha(p) for p in output.glob("*.json") if p.name != "PANEL_STATE.json"}))
        stage("fold-panels-complete", completion_sha256=sha(output / "PANEL_COMPLETE.json"),
              next_action="compare all tradeoffs and execute registered full-data follow-through")
    except BaseException as error:
        stage("failed", failures=[repr(error)])
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("queue", "run", "status"))
    p.add_argument("plan", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    globals()[a.mode](a.plan.resolve(), a.output.resolve())
