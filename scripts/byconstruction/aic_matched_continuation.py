"""Registered matched-continuation diagnostic (benchmarks/byconstruction_2026-10-22.md, 2026-10-27).

Original Mlic full seed 0 continues for 60 epochs (LR 1e-5 constant, fresh Adam, 50,000 draws per
epoch, the existing seed-0 streams) under two populations: the original mixture plus AIC-2026
public_train at weight 16, and public_train alone. The latter is AIC-only continuation of a
mixed-data pretrained model, not training from scratch. Epochs 7 and 59 are the fixed assessment
checkpoints. The mixed arm's epoch 7 must equal the existing eight-epoch MlicFT16x8 seed-0
continuation except for zentrain.repro; a mismatch blocks interpretation of the contrast, not the
preservation of its artifacts. Raw f32 panels on the five admitted TRAIN populations reuse the full
study's frozen owners. No packing, output calibration, EVAL or public_test read; no selector.
Seed 0 diagnoses a mechanism; it cannot establish stability or adoption.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from aic_adaptation import FOLDS, SPEC, bound_worker, compute_lock, process_snapshot, require_clean_factors, sha, write
from aic_adaptation_diagnostics import checked_statistics, ladder_diagnostics
from aic_adaptation_full_panel import full_populations, pixel_mixtures
from aic_adaptation_panel import finite_json

PROTOCOL = "aic-matched-continuation-diagnostic-v1"
REGISTRATION = dict(record="benchmarks/byconstruction_2026-10-22.md",
                    section="Prospective bounded screen: population versus continuation budget",
                    commit="6cfd69ad7bd12390f1ea73cc631f2160632900c7")
ARMS = {"MixedAIC16": "mixed-aic16", "AICOnly": "aic-only"}
EPOCHS, CHECKPOINTS, AIC_WEIGHT, LR = 60, (7, 59), 16.0, 0.00001
FROZEN_TOOLS = ("bake_dial_refit", "byc_predict", "panel", "score_pair_with_bake", "byc_fsid")
METRICS = ("srocc_signed", "krocc", "plcc", "plcc_raw", "or", "pwrc", "z_rmse", "mae")
# Registered inputs (dated observations on dev, 2026-10-27; every one is rehashed by plan and run).
INPUTS = dict(
    fold_plan=("/mnt/v/output/zensim/mlic-aic-adaptation-20261027-run03/PLAN.json",
               "98f72bda706481f885ddd1c295e0ce3ab9c1f38f29d8afd0c59115022a8afa54"),
    full_plan=("/mnt/v/output/zensim/mlic-aic-full-20261027-run03/PLAN.json",
               "4684cbb8cd653f5409e292a4c6233ac74a4e797aa696f55be6a64a273a0fa4da"),
    full_fits=("/mnt/v/output/zensim/mlic-aic-full-20261027-run03/FULL_FITS_COMPLETE.json",
               "18e4c56c28089f340355caaf6d39660087ce1a208ac2bc41d40482cb475cdb25"),
    full_panel=("/mnt/v/output/zensim/mlic-aic-full-20261027-run03/panels/FULL_PANEL_COMPLETE.json",
                "fd787030c17f916c06dc8207b30dd353873e983befdea305754cbbb6ce0f805a"),
    aic_table=("/mnt/v/output/zensim/aic-finetune-20261026/folds-rev7/aic_full_fit.parquet",
               "58a0af71f8ce69ddad0314850f2b0069bab5158ded6213d5fc7dacc77aa34c99"),
    aic_authority=("/mnt/v/output/zensim/aic-finetune-20261026/folds-rev7/_MANIFEST.json",
                   "0c9595f65d0ad08587c04821fc734733c0582033cbb96ad902eb4406556b3d18"),
    followthrough=("/mnt/v/output/zensim/mlic-confidence-followthrough-20261027-run01/CONFIG.json",
                   "60cb9ed63f18ef6e8f0af6f27adbac3fb1e8f78f63c77a5c6ec101279eb557a3"))
FOLLOWTHROUGH_DONE = "confidence-AIC-panels-complete"


def now():
    return datetime.now(timezone.utc).isoformat()


def labels():
    return ["Initial", *(f"{arm}@e{e}" for arm in ARMS for e in CHECKPOINTS)]


def full_record(fits, arm):
    rows = fits["initial"]["Mlic"] if arm == "Mlic" else [r for r in fits["continued"] if r["arm"] == arm]
    rows = [r for r in rows if r["seed"] == 0]
    if len(rows) != 1 or any(sha(p) != pin for p, pin in rows[0]["input_sha256"].items()):
        raise ValueError(f"one unchanged completed seed-0 {arm} record required")
    return rows[0]


def plan(run, commit, trainer, meta_tool):
    """Freeze source, tools, inputs and the registered design before any fit."""
    run = Path(run).resolve()
    if run.exists():
        raise ValueError("preserve existing diagnostic attempts")
    for path, pin in INPUTS.values():
        if sha(path) != pin:
            raise ValueError(f"registered input changed: {path}")
    if subprocess.run(["git", "-C", str(REPO), "merge-base", "--is-ancestor", commit, "origin/main"]).returncode:
        raise ValueError("diagnostic source must be pushed to origin/main")
    fold = json.loads(Path(INPUTS["fold_plan"][0]).read_text())
    fits = json.loads(Path(INPUTS["full_fits"][0]).read_text())
    initial = full_record(fits, "Mlic")
    reference = full_record(fits, "MlicFT16x8")
    if reference["epochs"] != 8 or reference["weight"] != 16:
        raise ValueError("reference must be the eight-epoch weight-16 continuation")
    checkpoint = Path(initial["directory"]) / "refit/last.bin"
    reference_bake = Path(reference["directory"]) / "refit/last.bin"
    if sha(checkpoint) != initial["checkpoint_sha256"] or sha(reference_bake) != reference["checkpoint_sha256"]:
        raise ValueError("completed initial/reference model changed")
    run.mkdir(parents=True)
    archive = run / "source.tar"
    with archive.open("xb") as stream:
        subprocess.run(["git", "-C", str(REPO), "archive", "--format=tar", commit, "scripts"], stdout=stream, check=True)
    (run / "source").mkdir()
    subprocess.run(["tar", "-xf", str(archive), "-C", str(run / "source")], check=True)
    (run / "bin").mkdir()
    (run / "tools").mkdir()
    shutil.copy2(trainer, run / "bin/zensim_mlp_train")
    for name in FROZEN_TOOLS:
        shutil.copy2(Path(fold["bin"]) / name, run / "bin" / name)
    shutil.copy2(meta_tool, run / "tools/bake_dial_refit")
    root = Path(fold["common_root"])
    inputs = [checkpoint, reference_bake, *map(Path, initial["input_sha256"]), *map(Path, reference["input_sha256"]),
              Path(fold["columns"]),
              root / "human_role_decision.json", Path(INPUTS["aic_table"][0] + ".manifest.json")]
    value = dict(schema="aic-matched-continuation-plan-v1", protocol=PROTOCOL, registration=REGISTRATION,
                 source_commit=commit, source_archive_sha256=sha(archive), arms=ARMS, epochs=EPOCHS,
                 checkpoints=list(CHECKPOINTS), aic_weight=AIC_WEIGHT, lr=LR, seed_index=0, spec=SPEC,
                 root=str(root), columns=fold["columns"], initial=initial, reference=reference,
                 registered_inputs={k: dict(path=p, sha256=pin) for k, (p, pin) in INPUTS.items()},
                 bin_sha256={p.name: sha(p) for p in sorted((run / "bin").iterdir())},
                 frozen_tools_from=fold["bin"], meta_tool_sha256=sha(run / "tools/bake_dial_refit"),
                 input_sha256={str(p): sha(p) for p in inputs}, models=labels(), populations=["aic", *FOLDS],
                 representation="final-f32 checkpoint dumps", selector=None, public_test_opened=False,
                 eval_read=False, production_qualified=False,
                 priority="queued after the confidence omission study and its followthrough")
    write(run / "PLAN.json", value, fresh=True)
    print(json.dumps(dict(plan=str(run / "PLAN.json"), sha256=sha(run / "PLAN.json"))))


def checked(path):
    value = json.loads(path.read_text())
    if value["schema"] != "aic-matched-continuation-plan-v1" or value["protocol"] != PROTOCOL:
        raise ValueError("diagnostic plan schema differs")
    for item in value["registered_inputs"].values():
        if sha(item["path"]) != item["sha256"]:
            raise ValueError(f"registered input changed: {item['path']}")
    for name, pin in {**value["input_sha256"], **{str(path.parent / "bin" / k): v for k, v in value["bin_sha256"].items()},
                      str(path.parent / "tools/bake_dial_refit"): value["meta_tool_sha256"],
                      str(path.parent / "source.tar"): value["source_archive_sha256"]}.items():
        if sha(name) != pin:
            raise ValueError(f"diagnostic input changed: {name}")
    return value


def followthrough_state():
    state = json.loads((Path(INPUTS["followthrough"][0]).parent / "STATE.json").read_text())
    if state["stage"] == FOLLOWTHROUGH_DONE:
        return "complete"
    worker = state["worker"]
    if state["stage"] == "failed" or state["failures"] or not bound_worker(worker["pid"], worker["start_ticks"]):
        return "failed"
    return "running"


def queue(path):
    """Wait, holding no lock, until the confidence followthrough completes; then take the shared lock."""
    run_dir = path.parent
    state = dict(plan_sha256=sha(path), worker=process_snapshot(os.getpid()), stage="waiting-for-confidence-followthrough",
                 failures=[], checked_utc=now())
    write(run_dir / "QUEUE.json", state, fresh=True)
    try:
        while (current := followthrough_state()) == "running":
            time.sleep(60)
        if current != "complete":
            raise RuntimeError("confidence followthrough did not complete; inspect it before this diagnostic")
        checked(path)
        command = ["flock", str(Path.home() / "tmp/heavy.lock"), str(Path.home() / "work/claudehints/scripts/run-heavy"),
                   "--mem", "24G", "--jobs", "4", "--", sys.executable, "-B", str(Path(__file__).resolve()), "run", str(path)]
        state.update(stage="queued-for-compute-lock", command=command, checked_utc=now())
        write(run_dir / "QUEUE.json", state)
        with (run_dir / "launch.log").open("x") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        state.update(stage="run-returned", checked_utc=now())
        write(run_dir / "QUEUE.json", state)
    except BaseException as error:
        state.update(stage="failed", failures=[*state["failures"], repr(error)], checked_utc=now())
        write(run_dir / "QUEUE.json", state)
        raise


def fit_command(plan_value, run_dir, arm):
    source = run_dir / "source/scripts/rev4_featpot"
    return [sys.executable, "-B", str(source / "v2_confirm_fit.py"), "--root", plan_value["root"], "--spec", SPEC,
            "--columns", Path(plan_value["columns"]).read_text().strip(), "--head", "N", "--seed-index", "0",
            "--strict-admission", "--data-role-decision", str(Path(plan_value["root"]) / "human_role_decision.json"),
            "--dest", str(run_dir / "arms" / arm / "full_s0"), "--train-only",
            "--init-from", str(Path(plan_value["initial"]["directory"]) / "refit/last.bin"),
            "--fine-tune-epochs", str(EPOCHS), "--fine-tune-lr", str(LR),
            "--aic-public-train", plan_value["registered_inputs"]["aic_table"]["path"],
            "--aic-weight", str(AIC_WEIGHT), "--diagnostic-population", ARMS[arm]]


def nonrepro_identical(tool, first, second, output):
    """Exact model bytes after the native owner removes only zentrain.repro from copies."""
    output.mkdir()
    copies = []
    for label, path in (("diagnostic", first), ("reference", second)):
        dest = output / f"{label}.bin"
        with (output / f"{label}.log").open("x") as log:
            subprocess.run([str(tool), "strip", "--in", str(path), "--out", str(dest), "--key", "zentrain.repro"],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        copies.append(dest)
    return dict(identical=copies[0].read_bytes() == copies[1].read_bytes(), diagnostic_sha256=sha(first),
                reference_sha256=sha(second), nonrepro_sha256=[sha(p) for p in copies], native_owner_sha256=sha(tool))


def run(path):
    import numpy as np
    import pandas as pd
    if not compute_lock(os.getpid())["held_by_worker_ancestor"]:
        raise ValueError("the diagnostic requires the shared compute lock")
    run_dir = path.parent
    plan_value = checked(path)
    require_clean_factors(os.environ)
    fold = json.loads(Path(plan_value["registered_inputs"]["fold_plan"]["path"]).read_text())
    state = dict(stage="fit-arms", worker=process_snapshot(os.getpid()), plan_sha256=sha(path), failures=[],
                 objective_complete=False, production_qualified=False)
    def stage(name, **details):
        state.update(stage=name, checked_utc=now(), **details)
        write(run_dir / "STATE.json", state)
    def save(name, value):
        write(output / name, finite_json(value), fresh=True)
    if (run_dir / "STATE.json").exists():
        raise ValueError("preserve previous diagnostic run state")
    output = run_dir / "panels"
    try:
        stage("fit-arms")
        environment = dict(os.environ, REV4_V2_BIN_DIR=str(run_dir / "bin"), RAYON_NUM_THREADS="1",
                           OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", ZENSIM_MAX_TIER="v3")
        processes = {}
        for arm in ARMS:
            scratch = Path.home() / "tmp" / f"{run_dir.name}-{arm}"
            scratch.mkdir(parents=True)
            log = (run_dir / f"{arm}.log").open("x")
            processes[arm] = (subprocess.Popen(fit_command(plan_value, run_dir, arm), stdout=log, stderr=subprocess.STDOUT,
                              env=dict(environment, TMPDIR=str(scratch))), log)
        stage("fit-arms", fitters={arm: process_snapshot(p.pid) for arm, (p, _) in processes.items()})
        for arm, (process, log) in processes.items():
            if process.wait():
                raise RuntimeError(f"{arm} fit failed; preserved in {arm}.log")
            log.close()
        arms = {}
        for arm, population in ARMS.items():
            dest = run_dir / "arms" / arm / "full_s0"
            result = json.loads((dest / "result.json").read_text())
            diagnostic = result.get("diagnostic") or {}
            dumps = sorted(p.name for p in (dest / "ckpt").glob("ckpt_epoch*.bin"))
            if (result["epochs"] != EPOCHS or diagnostic.get("population") != population
                    or result["selection"]["selected_epoch"] != EPOCHS - 1
                    or dumps != [f"ckpt_epoch{e:03d}.bin" for e in range(EPOCHS)]
                    or any(sha(dest / "ckpt" / f"ckpt_epoch{e:03d}.bin") != diagnostic["fixed_checkpoints"][str(e)]["sha256"]
                           for e in CHECKPOINTS)):
                raise ValueError(f"{arm}: registered fixed-epoch outputs incomplete")
            arms[arm] = dict(directory=str(dest), result_sha256=sha(dest / "result.json"),
                             checkpoints={str(e): sha(dest / "ckpt" / f"ckpt_epoch{e:03d}.bin") for e in range(EPOCHS)})
        write(run_dir / "FITS_COMPLETE.json", dict(plan_sha256=sha(path), arms=arms), fresh=True)
        stage("reproduction-control")
        reference = Path(plan_value["reference"]["directory"])
        control = nonrepro_identical(run_dir / "bin/bake_dial_refit",
                                     run_dir / "arms/MixedAIC16/full_s0/ckpt/ckpt_epoch007.bin",
                                     reference / "refit/last.bin", run_dir / "control")
        mixed = json.loads((run_dir / "arms/MixedAIC16/full_s0/result.json").read_text())
        original = json.loads((reference / "result.json").read_text())
        shared = ("train_weights", "coverage_leg", "wide_receipt_sha256", "frozen_sha256", "human_sources",
                  "data_role_decision_sha256", "spec", "head", "init_seed", "sample_seed", "width", "kept_features",
                  "pairs_per_epoch", "derived_inputs")
        control.update(result_fields_equal={k: mixed.get(k) == original.get(k) for k in shared},
                       comparison="mixed epoch 7 versus the registered eight-epoch MlicFT16x8 seed 0")
        control["contrast_interpretable"] = control["identical"] and all(control["result_fields_equal"].values())
        write(run_dir / "CONTROL.json", control, fresh=True)
        stage("native-sampling-receipts")
        exposure = {}
        for arm in ARMS:
            for e in CHECKPOINTS:
                text = subprocess.run([str(run_dir / "tools/bake_dial_refit"), "read-meta", "--key", "zentrain.sample_coverage",
                                       "--in", str(run_dir / "arms" / arm / "full_s0/ckpt" / f"ckpt_epoch{e:03d}.bin")],
                                      check=True, capture_output=True, text=True).stdout
                coverage = json.loads(text)
                groups = {g["name"]: g for g in coverage["full"]["per_group"]}
                aic = groups["aic_public_train"]
                if coverage["epochs"] != e + 1 or coverage["pairs_per_epoch"] != 50000:
                    raise ValueError(f"{arm} epoch {e}: sampler receipt covers a different prefix")
                exposure[f"{arm}@e{e}"] = dict(epochs=coverage["epochs"], pairs=coverage["full"]["n_pairs"],
                    aic_pairs=aic["n_pairs"], aic_share=aic["n_pairs"] / coverage["full"]["n_pairs"],
                    aic_rows_touched=aic["rows_touched"], aic_row_coverage=aic["row_coverage"],
                    aic_refs_touched=aic["refs_touched"], per_group={k: g["n_pairs"] for k, g in groups.items()},
                    digest=coverage["digest"])
        write(run_dir / "EXPOSURE.json", dict(source="zentrain.sample_coverage prefix receipts, native replay", arms=exposure),
              fresh=True)
        stage("raw-f32-panels")
        source = Path(fold["warmstart_source"])
        os.environ.update(TMPDIR=str(Path.home() / "tmp"), REV4_V2_BIN_DIR=str(run_dir / "bin"), AIC_BIN=str(run_dir / "bin"),
                          BYC_ROOT=fold["common_root"], ZEN_PANEL_BIN=str(run_dir / "bin/panel"),
                          BYC_LABEL_LEDGER_DIR=str(output / "label-reads"), RAYON_NUM_THREADS="1", OMP_NUM_THREADS="1",
                          OPENBLAS_NUM_THREADS="1", ZENSIM_MAX_TIER="v3", ZENSIM_RING_FORM="raw", ZENSIM_PYRAMID="xyb")
        sys.path[:0] = [str(source / "scripts/rev4_featpot"), str(source / "scripts/byconstruction"), str(source / "scripts")]
        import byc_cells as cells
        import aic_panel as legacy
        from lib.zen_stats import logistic_fit, panel_batch, panel_batch_indexed
        output.mkdir()
        pairs, populations = full_populations(fold, output, "registered-matched-continuation-diagnostic-panels")
        save("POPULATIONS.json", {k: dict(table=str(v["table"]), sha256=sha(v["table"]), rows=len(v["target"]),
             exposure="in sample: every public_train source was used in continuation" if k == "aic"
             else "in sample: original full fit, including original development rows") for k, v in populations.items()})
        models = {"Initial": Path(plan_value["initial"]["directory"]) / "refit/last.bin",
                  **{f"{arm}@e{e}": run_dir / "arms" / arm / "full_s0/ckpt" / f"ckpt_epoch{e:03d}.bin"
                     for arm in ARMS for e in CHECKPOINTS}}
        (output / "predictions").mkdir()
        values, cells_out, jobs = {}, [], []
        for label, model in models.items():
            for name, population in populations.items():
                destination = output / "predictions" / f"{label}|{name}.tsv"
                cells.predict_cell(model, population["table"], destination, SPEC)
                frame = pd.read_csv(destination, sep="\t")
                prediction, target = frame.pred.to_numpy(dtype=float), population["target"]
                if (list(frame.columns) != ["row_idx", "pred"] or len(prediction) != len(target)
                        or not np.array_equal(frame.row_idx, np.arange(len(target))) or not np.isfinite(prediction).all()):
                    raise ValueError("ordered finite prediction coverage differs")
                values[label, name] = prediction
                jobs.append((f"{label}|{name}", prediction, target))
                row = dict(model=label, population=name, prediction_sha256=sha(destination),
                           geometry=legacy.geometry(prediction, target))
                if name == "aic":
                    row["ladder"] = ladder_diagnostics(pairs, prediction, target, np.ones(len(target), bool), logistic_fit)
                cells_out.append(row)
        stats = panel_batch(jobs, stats="full")
        checked_statistics(jobs, stats)
        for row, value in zip(cells_out, stats, strict=True):
            row["statistics"] = value
        save("CELL_PANELS.json", cells_out)
        stored = Path(plan_value["registered_inputs"]["full_panel"]["path"]).parent / "predictions"
        save("PREDICTION_CONTROLS.json", {name: dict(
            initial_equals_full_study=sha(output / "predictions" / f"Initial|{name}.tsv") == sha(stored / f"Mlic|s0|final-f32|{name}.tsv"),
            mixed_epoch7_equals_registered_16x8=sha(output / "predictions" / f"MixedAIC16@e7|{name}.tsv")
            == sha(stored / f"MlicFT16x8|s0|final-f32|{name}.tsv")) for name in populations})
        aggregate_jobs, aggregate_rows = [], []
        for (label, name), prediction in values.items():
            target, metadata = populations[name]["target"], populations[name]["metadata"]
            groups = [("all", "ALL", np.arange(len(prediction)))]
            for category, columns in (("source", ["group"]), ("codec", ["codec"]), ("ladder", ["group", "codec"])):
                for key, index in metadata.groupby(columns, sort=True).indices.items():
                    groups.append((category, repr(key), np.asarray(index, dtype=int)))
            for category, group, index in groups:
                aggregate_jobs.append((f"{label}|{name}|{category}|{group}", prediction[index], target[index]))
                aggregate_rows.append(dict(model=label, population=name, category=category, group=group,
                    geometry=legacy.geometry(prediction[index], target[index]) if category in ("all", "codec") else None))
            save(f"bias-{label}|{name}.json", cells.codec_rank_bias(prediction, target, metadata.codec.to_numpy()))
        stats = panel_batch(aggregate_jobs, stats="full")
        checked_statistics(aggregate_jobs, stats)
        for row, value in zip(aggregate_rows, stats, strict=True):
            row["statistics"] = value
        save("AGGREGATE_PANELS.json", dict(panels=aggregate_rows, serving_model=None,
             note="seed 0 only; final-f32 checkpoint dumps; no packing or TRAIN output calibration"))
        contrasts = [("AICOnly@e59", "MixedAIC16@e59"), ("AICOnly@e7", "MixedAIC16@e7"),
                     ("MixedAIC16@e59", "MixedAIC16@e7"), ("AICOnly@e59", "AICOnly@e7"),
                     *((label, "Initial") for label in models if label != "Initial")]
        by_key = {(r["model"], r["population"], r["category"], r["group"]): r["statistics"] for r in aggregate_rows}
        deltas = {}
        for candidate, baseline in contrasts:
            for (model, name, category, group), value in by_key.items():
                if model == candidate and category in ("all", "source", "codec"):
                    other = by_key[baseline, name, category, group]
                    deltas[f"{candidate}-minus-{baseline}|{name}|{category}|{group}"] = {
                        m: (value[m] - other[m] if value.get(m) is not None and other.get(m) is not None else None)
                        for m in METRICS}
        save("CONTRASTS.json", dict(deltas=deltas, selection_gate=False,
             interpretable=control["contrast_interpretable"], higher_is_better={m: m not in ("or", "z_rmse", "mae") for m in METRICS}))
        rng = np.random.default_rng(1)
        clusters = [np.asarray(index, int) for index in pairs.groupby("group", sort=True).indices.values()]
        draws = [np.concatenate([clusters[i] for i in rng.integers(0, 28, 28)]) for _ in range(2000)]
        intervals = {}
        for candidate, baseline in contrasts:
            stage("source-intervals", current=f"{candidate}-minus-{baseline}")
            bases = dict(target=populations["aic"]["target"], candidate=values[candidate, "aic"], baseline=values[baseline, "aic"])
            tasks = [(f"{i}-{name}", name, "target", index) for i, index in enumerate(draws) for name in ("baseline", "candidate")]
            result = panel_batch_indexed(bases, tasks, stats="srocc")
            checked_statistics([(label, index, index) for label, _, _, index in tasks], result)
            delta = np.asarray([result[i + 1]["srocc_signed"] - result[i]["srocc_signed"] for i in range(0, len(result), 2)])
            good = delta[np.isfinite(delta)]
            intervals[f"{candidate}-minus-{baseline}"] = dict(metric="srocc_signed", resamples=2000, source_clusters=28, seed=1,
                defined_resamples=len(good), quantile_method="linear",
                interval_95=np.quantile(good, [.025, .975], method="linear").tolist() if len(good) else None,
                uncertainty="whole-source resampling conditional on fitted seed-0 models; every AIC source is in sample",
                descriptive_only=True, selection_gate=False)
        save("SOURCE_INTERVALS.json", intervals)
        stage("pixel-mixtures")
        pixel_mixtures(fold, source, output, {label.replace("@", "_"): model for label, model in models.items()}, save)
        inventory = {str(p.relative_to(run_dir)): sha(p) for p in sorted(run_dir.rglob("*"))
                     if p.is_file() and p.name not in ("STATE.json", "QUEUE.json") and "source" not in p.relative_to(run_dir).parts[:1]}
        write(run_dir / "EVIDENCE_INVENTORY.json", inventory, fresh=True)
        write(run_dir / "COMPLETE.json", dict(plan_sha256=sha(path), evidence_inventory_sha256=sha(run_dir / "EVIDENCE_INVENTORY.json"),
              control_identical=control["identical"], contrast_interpretable=control["contrast_interpretable"], selector=None,
              assessment="seed-0 mechanism diagnostic; advancing needs a separate three-seed/source-fold registration",
              production_qualified=False, objective_complete=False), fresh=True)
        stage("diagnostic-complete", next_action="review raw-f32 panels; no adoption from this screen")
    except BaseException as error:
        stage("failed", failures=[*state["failures"], repr(error)])
        raise


def status(path):
    run_dir = path.parent
    state_path = run_dir / "STATE.json"
    state = json.loads((state_path if state_path.exists() else run_dir / "QUEUE.json").read_text())
    worker = state["worker"]
    live = bound_worker(worker["pid"], worker["start_ticks"])
    if not live and state["stage"] not in ("diagnostic-complete", "run-returned") and not state["failures"]:
        state["failures"].append("recorded diagnostic process exited without completion")
    epochs = {}
    for arm in ARMS:
        log = run_dir / "arms" / arm / "full_s0/train.log"
        if log.exists():
            done = [int(line.split()[1]) for line in log.read_text().splitlines() if line.startswith("  epoch ")]
            epochs[arm] = max(done) if done else None
    state.update(checked_utc=now(), live=live, logged_epochs=epochs, followthrough=followthrough_state(),
                 compute_lock=compute_lock(worker["pid"] if live else None))
    write(run_dir / "CURRENT.json", state)
    print(json.dumps({k: state.get(k) for k in ("stage", "live", "logged_epochs", "followthrough", "failures", "compute_lock")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    p = sub.add_parser("plan")
    p.add_argument("run", type=Path)
    p.add_argument("--source-commit", required=True)
    p.add_argument("--trainer", type=Path, required=True)
    p.add_argument("--meta-tool", type=Path, required=True)
    for mode in ("queue", "run", "status"):
        sub.add_parser(mode).add_argument("plan", type=Path)
    args = parser.parse_args()
    if args.mode == "plan":
        plan(args.run, args.source_commit, args.trainer, args.meta_tool)
    else:
        globals()[args.mode](args.plan.resolve())
