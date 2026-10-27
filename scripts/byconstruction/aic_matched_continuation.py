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
sys.path.insert(1, str(HERE.parent / "rev4_featpot"))
from aic_adaptation import FOLDS, SPEC, bound_worker, compute_lock, process_snapshot, require_clean_factors, sha, write
from aic_adaptation_diagnostics import checked_statistics, ladder_diagnostics
from aic_adaptation_full_panel import full_populations, pixel_mixtures, seed_delta
from aic_adaptation_panel import finite_json
from subjective_training_leg import diagnostic_protocol

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


def full_record(fits, arm, seed=0):
    rows = fits["initial"]["Mlic"] if arm == "Mlic" else [r for r in fits["continued"] if r["arm"] == arm]
    rows = [r for r in rows if r["seed"] == seed]
    if len(rows) != 1 or any(sha(p) != pin for p, pin in rows[0]["input_sha256"].items()):
        raise ValueError(f"one unchanged completed seed-{seed} {arm} record required")
    return rows[0]


def freeze(run, commit, trainer, meta_tool, fold):
    """Fresh run directory with the pushed source snapshot, the new trainer and the fold study's frozen tools."""
    if run.exists():
        raise ValueError("preserve existing diagnostic attempts")
    for path, pin in INPUTS.values():
        if sha(path) != pin:
            raise ValueError(f"registered input changed: {path}")
    if subprocess.run(["git", "-C", str(REPO), "merge-base", "--is-ancestor", commit, "origin/main"]).returncode:
        raise ValueError("diagnostic source must be pushed to origin/main")
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
    return dict(source_commit=commit, source_archive_sha256=sha(archive),
                bin_sha256={p.name: sha(p) for p in sorted((run / "bin").iterdir())},
                frozen_tools_from=fold["bin"], meta_tool_sha256=sha(run / "tools/bake_dial_refit"),
                registered_inputs={k: dict(path=p, sha256=pin) for k, (p, pin) in INPUTS.items()})


def plan(run, commit, trainer, meta_tool):
    """Freeze source, tools, inputs and the registered design before any fit."""
    run = Path(run).resolve()
    if run.exists():
        raise ValueError("preserve existing diagnostic attempts")
    for path, pin in INPUTS.values():
        if sha(path) != pin:
            raise ValueError(f"registered input changed: {path}")
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
    frozen = freeze(run, commit, trainer, meta_tool, fold)
    root = Path(fold["common_root"])
    inputs = [checkpoint, reference_bake, *map(Path, initial["input_sha256"]), *map(Path, reference["input_sha256"]),
              Path(fold["columns"]),
              root / "human_role_decision.json", Path(INPUTS["aic_table"][0] + ".manifest.json")]
    value = dict(schema="aic-matched-continuation-plan-v1", protocol=PROTOCOL, registration=REGISTRATION,
                 **frozen, arms=ARMS, epochs=EPOCHS,
                 checkpoints=list(CHECKPOINTS), aic_weight=AIC_WEIGHT, lr=LR, seed_index=0, spec=SPEC,
                 root=str(root), columns=fold["columns"], initial=initial, reference=reference,
                 input_sha256={str(p): sha(p) for p in inputs}, models=labels(), populations=["aic", *FOLDS],
                 representation="final-f32 checkpoint dumps", selector=None, public_test_opened=False,
                 eval_read=False, production_qualified=False,
                 priority="queued after the confidence omission study and its followthrough")
    write(run / "PLAN.json", value, fresh=True)
    print(json.dumps(dict(plan=str(run / "PLAN.json"), sha256=sha(run / "PLAN.json"))))


def checked(path, schema="aic-matched-continuation-plan-v1"):
    value = json.loads(path.read_text())
    protocol = PROTOCOL if schema == FOLD_SCHEMA or schema.startswith("aic-matched") else FRONTIER_PROTOCOL
    if schema in (RECIPE_SCHEMA, SEEDS_SCHEMA, OMISSION_SCHEMA, FULL_SCHEMA, XREF_SCHEMA):
        protocol = diagnostic_protocol(next(iter(value["arms"].values())))[0]
    if schema == WISE_SCHEMA:
        protocol = WISE_PROTOCOL
    if value["schema"] != schema or value["protocol"] != protocol:
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


def fit_command(plan_value, run_dir, arm, seed=0, initial=None, aic_table=None, dest=None, checkpoint=None):
    """The registered diagnostic route; the seed-0 screen uses the full table, the fold study one fold table.
    `checkpoint` continues from an explicit model (the recovery stage) instead of a full fit's final checkpoint."""
    source = run_dir / "source/scripts/rev4_featpot"
    initial = initial or plan_value["initial"]["directory"]
    population = plan_value.get("arms", ARMS)[arm]
    # A recipe's own training inputs (stage B: Order's matching schedule) ride only on populations that keep
    # the older legs; the AIC-only population has no group for them to act on.
    recipe = [] if population == "aic-only" else plan_value.get("fit_extra_args", [])
    return [sys.executable, "-B", str(source / "v2_confirm_fit.py"), "--root", plan_value["root"], "--spec", SPEC,
            "--columns", Path(plan_value["columns"]).read_text().strip(), "--head", "N", "--seed-index", str(seed),
            "--strict-admission", "--data-role-decision", str(Path(plan_value["root"]) / "human_role_decision.json"),
            "--dest", str(dest or run_dir / "arms" / arm / "full_s0"), "--train-only",
            "--init-from", str(checkpoint or Path(initial) / "refit/last.bin"),
            "--fine-tune-epochs", str(EPOCHS), "--fine-tune-lr", str(LR),
            "--aic-public-train", aic_table or plan_value["registered_inputs"]["aic_table"]["path"],
            "--aic-weight", str(diagnostic_protocol(population)[1]), "--diagnostic-population", population, *recipe]


def fixed_outputs(dest, population, label, checkpoints=CHECKPOINTS):
    """The registered every-epoch dumps and fixed checkpoints of one completed diagnostic fit."""
    result = json.loads((dest / "result.json").read_text())
    diagnostic = result.get("diagnostic") or {}
    dumps = sorted(p.name for p in (dest / "ckpt").glob("ckpt_epoch*.bin"))
    if (result["epochs"] != EPOCHS or diagnostic.get("population") != population
            or result["selection"]["selected_epoch"] != EPOCHS - 1
            or dumps != [f"ckpt_epoch{e:03d}.bin" for e in range(EPOCHS)]
            or not {str(e) for e in checkpoints} & set(diagnostic["fixed_checkpoints"])
            or any(sha(dest / "ckpt" / f"ckpt_epoch{int(e):03d}.bin") != record["sha256"]
                   for e, record in diagnostic["fixed_checkpoints"].items())):
        raise ValueError(f"{label}: registered fixed-epoch outputs incomplete")
    return dict(directory=str(dest), result_sha256=sha(dest / "result.json"),
                checkpoints={str(e): sha(dest / "ckpt" / f"ckpt_epoch{e:03d}.bin") for e in range(EPOCHS)})


def sampler_receipt(tool, checkpoint, epoch):
    """Native zentrain.sample_coverage prefix receipt of one stamped checkpoint."""
    coverage = json.loads(subprocess.run([str(tool), "read-meta", "--key", "zentrain.sample_coverage", "--in", str(checkpoint)],
                                         check=True, capture_output=True, text=True).stdout)
    groups = {g["name"]: g for g in coverage["full"]["per_group"]}
    aic = groups["aic_public_train"]
    if coverage["epochs"] != epoch + 1 or coverage["pairs_per_epoch"] != 50000:
        raise ValueError(f"{checkpoint}: sampler receipt covers a different prefix")
    return dict(epochs=coverage["epochs"], pairs=coverage["full"]["n_pairs"], aic_pairs=aic["n_pairs"],
                aic_share=aic["n_pairs"] / coverage["full"]["n_pairs"], aic_rows_touched=aic["rows_touched"],
                aic_row_coverage=aic["row_coverage"], aic_refs_touched=aic["refs_touched"],
                per_group={k: g["n_pairs"] for k, g in groups.items()}, digest=coverage["digest"])


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
        arms = {arm: fixed_outputs(run_dir / "arms" / arm / "full_s0", population, arm) for arm, population in ARMS.items()}
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
        exposure = {f"{arm}@e{e}": sampler_receipt(run_dir / "tools/bake_dial_refit",
                     run_dir / "arms" / arm / "full_s0/ckpt" / f"ckpt_epoch{e:03d}.bin", e) for arm in ARMS for e in CHECKPOINTS}
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


FOLD_SCHEMA = "aic-matched-continuation-fold-plan-v1"
FOLD_REGISTRATION = dict(record="benchmarks/byconstruction_2026-10-22.md",
                         section="Prospective source-fold matched continuation",
                         commit="fcb745ff6bc1f86d1750a44881a9d846c49d0f8f")
SEEDS, SOURCE_FOLDS = (0, 1, 2), tuple(range(7))
GATES_LOG = Path("/mnt/v/output/zensim/mlic-candidate-gates-20261027-run01/queue.log")
# Stage A of the AIC-share frontier (byconstruction record, registered 2026-10-27 before any frontier fit): Mlic
# seed 0 continued at public_train weights 64/256/1024 on the seven folds; the source-fold study's seed-0 x16 and
# AIC-only fits are reused at the same checkpoints.
FRONTIER_SCHEMA = "aic-share-frontier-plan-v1"
FRONTIER_PROTOCOL = "aic-share-frontier-v1"
FRONTIER_REGISTRATION = dict(record="benchmarks/byconstruction_2026-10-22.md",
                             section="AIC-share frontier: can rehearsal reach 0.95 out of fold and keep the older sets?",
                             commit="0a8a00e1dd396aa0aacba8eb64d1f1770bf04d5b")
FRONTIER_ARMS = {"MixedAIC64": "mixed-aic64", "MixedAIC256": "mixed-aic256", "MixedAIC1024": "mixed-aic1024"}
FRONTIER_SEEDS, FRONTIER_CHECKPOINTS = (0,), (7, 15, 29, 59)
FRONTIER_REUSED_RUN = Path("/mnt/v/output/zensim/mlic-matched-continuation-folds-20261027-run01")
FRONTIER_AFTER = Path("/mnt/v/output/zensim/mlic-candidate-gates-20261027-run02/queue.log")
# Stage A2 (registered 2026-10-27 before any stage A result): each fold's seed-0 AIC-only epoch-59 model continues on
# the original mixture plus the same fold's AIC table at W = 16, 64 or 256.
RECOVERY_SCHEMA = "aic-share-recovery-plan-v1"
RECOVERY_REGISTRATION = dict(record="benchmarks/byconstruction_2026-10-22.md",
                             section="Stage A2, recovery continuation", commit="b492099e78bbf1b69b5291d23c733b3d315fa09d")
RECOVERY_ARMS = {"RecoverAIC16": "mixed-aic16", "RecoverAIC64": "mixed-aic64", "RecoverAIC256": "mixed-aic256"}
RECOVERY_START = ("AICOnly", 59)
RECOVERY_AFTER = Path("/mnt/v/output/zensim/mlic-aic-share-frontier-20261027-run01")
# Stage B (registered with stage A; mechanics registered 2026-10-27 before any stage A2 result): one run per recipe,
# its full seed-0 model continued at the advanced budget with its own weighting active in its older legs.
RECIPE_SCHEMA = "aic-share-recipe-plan-v1"
RECIPE_REGISTRATION = dict(record="benchmarks/byconstruction_2026-10-22.md",
                           section="Stage B details", commit="219e15c183a762c688d8c3534d931567a8e6d966")
RECIPES = ("MlicNeff", "MlicVote", "MlicOrder", "MlicConfidence")
# Stage C (registered with stage A): the chosen Mlic point on the folds for seeds 0-2.
SEEDS_SCHEMA = "aic-share-seeds-plan-v1"
# Stage D (registered 2026-10-27 before any stage B result): the human leg rehearsed within each older dataset.
XREF_SCHEMA = "aic-share-xref-plan-v1"
XREF_REGISTRATION = dict(record="benchmarks/byconstruction_2026-10-22.md",
                         section="Stage D: within-dataset rehearsal of the older sets",
                         commit="58c56ef37b972fabb252fc75dd6add5e2d973a2a")
XREF_ARMS = {"XrefAIC256": "mixed-aic256-xref", "XrefAIC1024": "mixed-aic1024-xref"}
RECIPE_ARM = {"aic-only": "AICOnly", "mixed-aic16": "AIC16", "mixed-aic64": "AIC64", "mixed-aic256": "AIC256",
              "mixed-aic1024": "AIC1024", "mixed-aic256-xref": "XrefAIC256", "mixed-aic1024-xref": "XrefAIC1024"}
RECIPE_INPUTS = dict(
    confidence_full=("/mnt/v/output/zensim/mlic-confidence-followthrough-20261027-run01/full/FULL_FITS_COMPLETE.json",
                     "b8b6fb9240e84b1f7d3d375ef929c6f3dcd45d765d8f38f7ca1c962bc51b20dc"),
    confidence_plan=("/mnt/v/output/zensim/mlic-confidence-followthrough-20261027-run01/initial/PLAN.json",
                     "d56125a05e6862fcbb0211dc6a48e927320747a2c9c4887dac2599a99f4f5a7f"),
    matching=("/mnt/v/output/zensim/mlic-aic-full-20261027-run03/matching/MANIFESTS.json",
              "bcd3cf1e07effffc74744ba04211a652c969aa7eb45c5da3a2784863491bb6e2"))


def plan_schema(path):
    schema = json.loads(Path(path).read_text()).get("schema")
    if schema not in (FOLD_SCHEMA, FRONTIER_SCHEMA, RECOVERY_SCHEMA, RECIPE_SCHEMA, SEEDS_SCHEMA, OMISSION_SCHEMA, FULL_SCHEMA,
                      XREF_SCHEMA, WISE_SCHEMA):
        raise ValueError("source-fold or frontier plan required")
    return schema


def predecessor_finished(plan_value):
    """A plan waits for the gates log, or for a predecessor run's completion or failure."""
    run = plan_value.get("after_run")
    if run is None:
        return gates_finished(plan_value.get("after_log"))
    run = Path(run)
    state = json.loads((run / "STATE.json").read_text()) if (run / "STATE.json").exists() else {}
    return (run / "COMPLETE.json").exists() or state.get("stage") == "failed"


def fold_runs(plan_value=None):
    """The fits a fold plan runs: its own arms and seeds (default: the registered source-fold study)."""
    arms, seeds = (plan_value["arms"], plan_value["seeds"]) if plan_value else (ARMS, SEEDS)
    return [(arm, seed, k) for arm in arms for seed in seeds for k in SOURCE_FOLDS]


def paired_delta(candidate, baseline, seeds):
    """Seed-paired difference: the three-seed owner's summary, or single-seed deltas without an SE."""
    if list(seeds) == [0, 1, 2]:
        return seed_delta(candidate, baseline)
    values = [candidate[s] - baseline[s] for s in seeds]
    return dict(per_seed_delta=values, mean=sum(values) / len(values), seed_unit_se=None, selection_gate=False)


def fold_plan(run, commit, trainer, meta_tool):
    """Freeze the registered source-fold study: 2 populations x 3 Mlic full seeds x 7 D6 source folds."""
    seven_fold_plan(run, commit, trainer, meta_tool, schema=FOLD_SCHEMA, protocol=PROTOCOL,
                    registration=FOLD_REGISTRATION, arms=ARMS, seeds=SEEDS, checkpoints=CHECKPOINTS,
                    extra=dict(aic_weight=AIC_WEIGHT, priority="queued after the candidate gates"))


def frontier_plan(run, commit, trainer, meta_tool):
    """Freeze stage A of the AIC-share frontier, pinning the reused source-fold checkpoints."""
    reused_complete = FRONTIER_REUSED_RUN / "COMPLETE.json"
    if not reused_complete.exists():
        raise ValueError("the reused source-fold study must be complete")
    reused = {arm: dict(run_dir=str(FRONTIER_REUSED_RUN), population=ARMS[arm], checkpoint_sha256={
        f"{seed}|{k}|{e}": sha(FRONTIER_REUSED_RUN / "arms" / arm / f"s{seed}_k{k}/ckpt/ckpt_epoch{e:03d}.bin")
        for seed in FRONTIER_SEEDS for k in SOURCE_FOLDS for e in FRONTIER_CHECKPOINTS}) for arm in ARMS}
    groups = [f"{arm}@e{e}" for arm in [*FRONTIER_ARMS, *ARMS] for e in FRONTIER_CHECKPOINTS]
    contrasts = [*((g, "Initial") for g in groups),
                 *((f"{arm}@e{e}", f"MixedAIC16@e{e}") for arm in FRONTIER_ARMS for e in FRONTIER_CHECKPOINTS),
                 *((f"AICOnly@e{e}", f"{arm}@e{e}") for arm in FRONTIER_ARMS for e in FRONTIER_CHECKPOINTS)]
    seven_fold_plan(run, commit, trainer, meta_tool, schema=FRONTIER_SCHEMA, protocol=FRONTIER_PROTOCOL,
                    registration=FRONTIER_REGISTRATION, arms=FRONTIER_ARMS, seeds=FRONTIER_SEEDS,
                    checkpoints=FRONTIER_CHECKPOINTS,
                    extra=dict(aic_weights={arm: diagnostic_protocol(p)[1] for arm, p in FRONTIER_ARMS.items()},
                               control=dict(arm="MixedAIC16", population=ARMS["MixedAIC16"], seed=0, fold=0,
                                            epochs=list(FRONTIER_CHECKPOINTS), purpose="this trainer reproduces the "
                                            "reused fits: refit x16 seed 0 fold 0, byte-identical without zentrain.repro"),
                               reused_arms=reused, reused_complete_sha256=sha(reused_complete),
                               contrasts=[list(c) for c in contrasts], after_log=str(FRONTIER_AFTER),
                               assessment="stage A seed-0 AIC-share frontier; advancement by the registered rule; no adoption",
                               priority="queued after the candidate gates run02"))


def recovery_plan(run, commit, trainer, meta_tool):
    """Freeze stage A2: each fold's AIC-only epoch-59 model continues on the mixture at W = 16, 64 or 256."""
    if not (FRONTIER_REUSED_RUN / "COMPLETE.json").exists():
        raise ValueError("the source-fold study must be complete")
    arm, epoch = RECOVERY_START
    starts = {f"s{seed}_k{k}": FRONTIER_REUSED_RUN / "arms" / arm / f"s{seed}_k{k}/ckpt/ckpt_epoch{epoch:03d}.bin"
              for seed in FRONTIER_SEEDS for k in SOURCE_FOLDS}
    reused = {name: dict(run_dir=str(FRONTIER_REUSED_RUN), population=ARMS[name], checkpoint_sha256={
        f"{seed}|{k}|{e}": sha(FRONTIER_REUSED_RUN / "arms" / name / f"s{seed}_k{k}/ckpt/ckpt_epoch{e:03d}.bin")
        for seed in FRONTIER_SEEDS for k in SOURCE_FOLDS for e in FRONTIER_CHECKPOINTS}) for name in ARMS}
    groups = [f"{a}@e{e}" for a in RECOVERY_ARMS for e in FRONTIER_CHECKPOINTS]
    contrasts = [*((g, "Initial") for g in groups), *((g, f"{arm}@e{epoch}") for g in groups)]
    seven_fold_plan(run, commit, trainer, meta_tool, schema=RECOVERY_SCHEMA, protocol=FRONTIER_PROTOCOL,
                    registration=RECOVERY_REGISTRATION, arms=RECOVERY_ARMS, seeds=FRONTIER_SEEDS,
                    checkpoints=FRONTIER_CHECKPOINTS,
                    extra=dict(aic_weights={a: diagnostic_protocol(p)[1] for a, p in RECOVERY_ARMS.items()},
                               run_initial={k: dict(path=str(v), sha256=sha(v)) for k, v in starts.items()},
                               start=dict(arm=arm, epoch=epoch), reused_arms=reused,
                               reused_complete_sha256=sha(FRONTIER_REUSED_RUN / "COMPLETE.json"),
                               contrasts=[list(c) for c in contrasts], after_run=str(RECOVERY_AFTER),
                               assessment="stage A2 seed-0 recovery continuation; advancement by the registered rule; no adoption",
                               priority="queued after the stage A frontier run"))


def recipe_inputs(recipe, seed=0):
    """A recipe's completed full model, its own root and its weighting inputs (stage B)."""
    if recipe not in RECIPES:
        raise ValueError("registered stage B recipe required")
    for path, pin in RECIPE_INPUTS.values():
        if sha(path) != pin:
            raise ValueError(f"registered recipe input changed: {path}")
    study = RECIPE_INPUTS["confidence_full"][0] if recipe == "MlicConfidence" else INPUTS["full_fits"][0]
    rows = [r for r in json.loads(Path(study).read_text())["initial"][recipe] if r["seed"] == seed]
    if len(rows) != 1 or any(sha(p) != pin for p, pin in rows[0]["input_sha256"].items()):
        raise ValueError(f"one unchanged completed seed-{seed} {recipe} full model required")
    record = rows[0]
    directory = Path(record["directory"])
    selection = json.loads((directory / "DATASET_GROUP_SELECTION.json").read_text())
    root = Path(selection["root"])
    if (sha(directory / "refit/last.bin") != record["checkpoint_sha256"]
            or json.loads((directory / "result.json").read_text())["frozen_sha256"] != selection["root_sha256"]
            or sha(root / "wide/frozen.json") != selection["root_sha256"]
            or selection["requested_exclusions"] or selection["omitted_collections"]):
        raise ValueError(f"{recipe}: full model, root or omission authority differs")
    factors = {}
    if recipe == "MlicOrder":
        schedule = json.loads(Path(RECIPE_INPUTS["matching"][0]).read_text())["cells"][str(seed)]
        if sha(schedule["path"]) != schedule["sha256"]:
            raise ValueError("full matching schedule changed")
        factors.update(matching=schedule, sample_coverage=False, fit_extra_args=[
            "--matched-codec-full", "--matched-codec-manifest", schedule["path"],
            "--matched-codec-sha256", schedule["sha256"]])
    if recipe == "MlicConfidence":
        from mlic_confidence_full import full_binding
        prior = json.loads(Path(RECIPE_INPUTS["confidence_plan"][0]).read_text())
        factors["confidence"] = full_binding(prior["registry"], prior["input_sha256"][prior["registry"]], seed)
    return record, root, factors


def reused_arm(run_dir, arm, seeds=FRONTIER_SEEDS, checkpoints=FRONTIER_CHECKPOINTS):
    """Pin another completed run's fold checkpoints for this run's contrasts."""
    run_dir = Path(run_dir)
    if not (run_dir / "COMPLETE.json").exists():
        raise ValueError(f"{run_dir.name} must be complete")
    population = json.loads((run_dir / "PLAN.json").read_text())["arms"].get(arm)
    if population is None:
        raise ValueError(f"{arm} is not an arm of {run_dir.name}")
    return dict(run_dir=str(run_dir), population=population, checkpoint_sha256={
        f"{seed}|{k}|{e}": sha(run_dir / "arms" / arm / f"s{seed}_k{k}/ckpt/ckpt_epoch{e:03d}.bin")
        for seed in seeds for k in SOURCE_FOLDS for e in checkpoints})


# Stage E (registered 2026-10-27 after stage D's result): MlicVote with the rehearsal leg at x256 and x1024, both
# weights in one recipe plan.
RECIPE_E_REGISTRATION = dict(record="benchmarks/byconstruction_2026-10-22.md",
                             section="Stage E: Vote with the rehearsal leg",
                             commit="e68f7d4add0c12501b489210f86de9195c16f040")


def recipe_plan(run, commit, trainer, meta_tool, recipe, populations, after_run, compares, start=None, control=True):
    """Freeze one stage B run: the recipe's full seed-0 model continued under each of `populations` on the seven
    folds (stage E runs both rehearsal weights in one plan). `compares` = one (run, arm) per population: Mlic's point
    at the same budget. `start` = (run, arm, epoch): a two-phase schedule continues from that run's fold checkpoints
    instead of the full model (the recovery phase)."""
    if len(set(populations)) != len(populations) or len(compares) != len(populations):
        raise ValueError("one comparison point per distinct population")
    protocols = {diagnostic_protocol(p)[0] for p in populations}
    if len(protocols) != 1:
        raise ValueError("a recipe plan's populations share one registered protocol")
    if start and len(populations) != 1:
        raise ValueError("a recovery phase continues one population")
    record, root, factors = recipe_inputs(recipe)
    arms = {recipe + RECIPE_ARM[p]: p for p in populations}
    reused, contrasts = {}, []
    for (arm, population), (compare_run, compare_arm) in zip(arms.items(), compares):
        reused[compare_arm] = reused_arm(Path(compare_run), compare_arm)
        if diagnostic_protocol(reused[compare_arm]["population"])[1] != diagnostic_protocol(population)[1]:
            raise ValueError(f"{compare_arm} runs another AIC weight than {arm}")
        contrasts += [*((f"{arm}@e{e}", "Initial") for e in FRONTIER_CHECKPOINTS),
                      *((f"{arm}@e{e}", f"{compare_arm}@e{e}") for e in FRONTIER_CHECKPOINTS)]
    rehearsal = protocols == {diagnostic_protocol("mixed-aic256-xref")[0]}
    extra = dict(recipe=recipe, **factors, aic_weights={arm: diagnostic_protocol(p)[1] for arm, p in arms.items()},
                 reused_arms=reused, after_run=str(after_run),
                 assessment=f"stage {'E' if rehearsal else 'B'} seed-0 recipe continuation; registered rule against the "
                            "recipe's own initial model; no adoption", priority="queued after its predecessor run")
    if start:
        (arm, _), = arms.items()
        start_run, start_arm, start_epoch = Path(start[0]), start[1], int(start[2])
        if start_epoch not in FRONTIER_CHECKPOINTS:
            raise ValueError("the recovery phase starts from a studied checkpoint")
        reused[start_arm] = reused_arm(start_run, start_arm)
        starts = {f"s0_k{k}": start_run / "arms" / start_arm / f"s0_k{k}/ckpt/ckpt_epoch{start_epoch:03d}.bin"
                  for k in SOURCE_FOLDS}
        extra.update(run_initial={k: dict(path=str(v), sha256=sha(v)) for k, v in starts.items()},
                     start=dict(run=str(start_run), arm=start_arm, epoch=start_epoch))
        contrasts += [(f"{arm}@e{e}", f"{start_arm}@e{start_epoch}") for e in FRONTIER_CHECKPOINTS]
    if control:
        extra["control"] = dict(kind="full-fit-replay", recipe=recipe, seed=0, original=record["directory"],
                                purpose="this trainer replays the recipe's 120-epoch full fit byte for byte without "
                                        "zentrain.repro (existing E33 control owner)")
    extra["contrasts"] = [list(c) for c in contrasts]
    seven_fold_plan(run, commit, trainer, meta_tool, schema=RECIPE_SCHEMA, protocol=protocols.pop(),
                    registration=RECIPE_E_REGISTRATION if rehearsal else RECIPE_REGISTRATION, arms=arms,
                    seeds=FRONTIER_SEEDS, checkpoints=FRONTIER_CHECKPOINTS, extra=extra, initial={"0": record}, root=root)


def xref_plan(run, commit, trainer, meta_tool, after_run):
    """Freeze stage D: stage A's x256 and x1024 continuations plus the within-dataset human rehearsal leg."""
    frontier = Path("/mnt/v/output/zensim/mlic-aic-share-frontier-20261027-run01")
    reused = {name: reused_arm(frontier, name) for name in ("MixedAIC256", "MixedAIC1024")}
    contrasts = [*((f"{arm}@e{e}", "Initial") for arm in XREF_ARMS for e in FRONTIER_CHECKPOINTS),
                 *((f"XrefAIC{w}@e{e}", f"MixedAIC{w}@e{e}") for w in (256, 1024) for e in FRONTIER_CHECKPOINTS)]
    protocol = diagnostic_protocol("mixed-aic256-xref")[0]
    seven_fold_plan(run, commit, trainer, meta_tool, schema=XREF_SCHEMA, protocol=protocol,
                    registration=XREF_REGISTRATION, arms=XREF_ARMS, seeds=FRONTIER_SEEDS,
                    checkpoints=FRONTIER_CHECKPOINTS, extra=dict(
                        aic_weights={arm: diagnostic_protocol(p)[1] for arm, p in XREF_ARMS.items()},
                        reused_arms=reused, contrasts=[list(c) for c in contrasts], after_run=str(after_run),
                        control=dict(arm="MixedAIC256", population="mixed-aic256", seed=0, fold=0,
                                     epochs=list(FRONTIER_CHECKPOINTS), purpose="this trainer reproduces stage A's "
                                     "x256 seed 0 fold 0 fit, byte-identical without zentrain.repro"),
                        assessment="stage D seed-0 rehearsal frontier; in-sample rule reported, E21 out of sample "
                                   "decides; no adoption", priority="queued after stage B"))


# Stage F (registered 2026-10-27 before any interpolation result): Mlic seed 0 interpolated in weight space toward each
# fold's seed-0 model of three AIC-heavy arms; no training. The bake owner's `interpolate` emits every point.
WISE_SCHEMA = "aic-share-wise-plan-v1"
WISE_PROTOCOL = "aic-share-wise-v1"
WISE_REGISTRATION = dict(record="benchmarks/byconstruction_2026-10-22.md",
                         section="Stage F: weight-space interpolation between Mlic and AIC-heavy fold models",
                         commit="d75866aaeeeb6e769b1f318f264aa48047cccf44")
WISE_ENDS = {"AICOnly": (FRONTIER_REUSED_RUN, "AICOnly", 29),
             "MixedAIC1024": (Path("/mnt/v/output/zensim/mlic-aic-share-frontier-20261027-run01"), "MixedAIC1024", 59),
             "XrefAIC1024": (Path("/mnt/v/output/zensim/mlic-aic-xref-20261027-run01"), "XrefAIC1024", 59)}
WISE_ALPHAS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def wise_arm(end, alpha):
    return f"Wise{end}A{round(alpha * 100):02d}"


def wise_plan(run, commit, trainer, meta_tool, after_run):
    """Freeze stage F: every interpolation point as an arm whose one checkpoint (epoch 0) the bake owner emits."""
    ends = {}
    for name, (run_dir, arm, epoch) in WISE_ENDS.items():
        if not (run_dir / "COMPLETE.json").exists():
            raise ValueError(f"{run_dir.name} must be complete")
        ckpt = lambda k, e: run_dir / "arms" / arm / f"s0_k{k}/ckpt/ckpt_epoch{e:03d}.bin"
        # Every-epoch dumps before the end: the bake owner matches hidden units step by step along them.
        ends[name] = dict(run_dir=str(run_dir), arm=arm, epoch=epoch,
                          population=json.loads((run_dir / "PLAN.json").read_text())["arms"][arm],
                          checkpoint_sha256={f"0|{k}": sha(ckpt(k, epoch)) for k in SOURCE_FOLDS},
                          track_sha256={f"0|{k}": [sha(ckpt(k, e)) for e in range(epoch)] for k in SOURCE_FOLDS})
    interpolations = {wise_arm(end, alpha): dict(end=end, alpha=alpha) for end in WISE_ENDS for alpha in WISE_ALPHAS}
    seven_fold_plan(run, commit, trainer, meta_tool, schema=WISE_SCHEMA, protocol=WISE_PROTOCOL,
                    registration=WISE_REGISTRATION, arms={arm: "interpolation" for arm in interpolations},
                    seeds=FRONTIER_SEEDS, checkpoints=(0,), extra=dict(
                        ends=ends, interpolations=interpolations,
                        contrasts=[[f"{arm}@e0", "Initial"] for arm in interpolations], after_run=str(after_run),
                        assessment="stage F seed-0 weight interpolation; registered rule against Mlic; no adoption",
                        priority="queued between the omission runs"))


# Stage C recipe variant (registered 2026-10-27 after stage B): a recipe point runs stage C on the recipe's own full
# models, root and omission cells. Recipes with per-seed factors (Order, Confidence) need their own variant.
RECIPE_C_REGISTRATION = dict(record="benchmarks/byconstruction_2026-10-22.md",
                             section="Stage C recipe variant and the stage D comparator",
                             commit="a29bb316db8eb45824463be0825646fd4fe7421c")
RECIPE_CELLS = {"MlicVote": Path("/mnt/v/output/zensim/mlic-vote-20261027-run02/cells"),
                "MlicNeff": Path("/mnt/v/output/zensim/mlic-neff-20261026/cells")}
RECIPE_FT_CELLS = Path("/mnt/v/output/zensim/mlic-aic-adaptation-20261027-run03/cells")


def recipe_point(point):
    """(recipe, {seed: full record}, root) for a stage B recipe point; (None, None, None) for an Mlic point."""
    if point.get("schema") != RECIPE_SCHEMA:
        return None, None, None
    recipe, initial, roots = point["recipe"], {}, set()
    if recipe not in RECIPE_CELLS:
        raise ValueError(f"{recipe}: its per-seed factors need their own stage C variant")
    for seed in SEEDS:
        record, root, factors = recipe_inputs(recipe, seed)
        if factors:
            raise ValueError(f"{recipe}: its per-seed factors need their own stage C variant")
        initial[str(seed)] = record
        roots.add(str(root))
    if len(roots) != 1:
        raise ValueError(f"{recipe}: one root across seeds required")
    return recipe, initial, Path(roots.pop())


def seeds_plan(run, commit, trainer, meta_tool, point_run, arm, after_run):
    """Stage C, first step (registered with stage A): the advanced Mlic point on the seven folds for seeds 0-2.
    Its seed-0 fits must reproduce the advancing run's checkpoints byte for byte (all folds and epochs)."""
    point_run = Path(point_run)
    point = json.loads((point_run / "PLAN.json").read_text())
    if (point.get("schema") not in (FRONTIER_SCHEMA, RECOVERY_SCHEMA, XREF_SCHEMA, RECIPE_SCHEMA)
            or arm not in point["arms"]):
        raise ValueError("the advanced point must be a frontier, recovery, rehearsal or recipe arm (x16 already has three seeds)")
    population = point["arms"][arm]
    protocol, weight, _ = diagnostic_protocol(population)
    recipe, initial, root = recipe_point(point)
    reused = {name: reused_arm(FRONTIER_REUSED_RUN, name, seeds=SEEDS) for name in ARMS}
    extra = dict(aic_weights={arm: weight}, reused_arms=reused, after_run=str(after_run), recipe=recipe,
                 reproduces=dict(reused_arm(point_run, arm), arm=arm, seed=0),
                 assessment="stage C three-seed fold study of the advanced point; no adoption from it alone",
                 priority="queued after stage B")
    contrasts = [*((f"{arm}@e{e}", "Initial") for e in FRONTIER_CHECKPOINTS),
                 *((f"{arm}@e{e}", f"MixedAIC16@e{e}") for e in FRONTIER_CHECKPOINTS),
                 *((f"AICOnly@e{e}", f"{arm}@e{e}") for e in FRONTIER_CHECKPOINTS)]
    if point.get("start"):
        start_arm, start_epoch = point["start"]["arm"], point["start"]["epoch"]
        starts = {f"s{seed}_k{k}": FRONTIER_REUSED_RUN / "arms" / start_arm / f"s{seed}_k{k}/ckpt/ckpt_epoch{start_epoch:03d}.bin"
                  for seed in SEEDS for k in SOURCE_FOLDS}
        extra.update(run_initial={k: dict(path=str(v), sha256=sha(v)) for k, v in starts.items()},
                     start=dict(arm=start_arm, epoch=start_epoch))
        contrasts += [(f"{arm}@e{e}", f"{start_arm}@e{start_epoch}") for e in FRONTIER_CHECKPOINTS]
    extra["contrasts"] = [list(c) for c in contrasts]
    seven_fold_plan(run, commit, trainer, meta_tool, schema=SEEDS_SCHEMA, protocol=protocol,
                    registration=RECIPE_C_REGISTRATION if recipe else FRONTIER_REGISTRATION, arms={arm: population},
                    seeds=SEEDS, checkpoints=FRONTIER_CHECKPOINTS, extra=extra, initial=initial, root=root)


def seven_fold_plan(run, commit, trainer, meta_tool, schema, protocol, registration, arms, seeds, checkpoints, extra,
                    initial=None, root=None):
    """Freeze a continuation study over the seven D6 source folds (shared by the fold study and the frontier).
    `initial` and `root` replace full Mlic and the common root for a recipe's own study (stage B)."""
    run = Path(run).resolve()
    if run.exists():
        raise ValueError("preserve existing diagnostic attempts")
    for path, pin in INPUTS.values():
        if sha(path) != pin:
            raise ValueError(f"registered input changed: {path}")
    fold = json.loads(Path(INPUTS["fold_plan"][0]).read_text())
    if initial is None:
        fits = json.loads(Path(INPUTS["full_fits"][0]).read_text())
        initial = {str(seed): full_record(fits, "Mlic", seed) for seed in seeds}
    if any(sha(Path(r["directory"]) / "refit/last.bin") != r["checkpoint_sha256"] for r in initial.values()):
        raise ValueError("completed initial model changed")
    authority = json.loads(Path(INPUTS["aic_authority"][0]).read_text())
    tables, inputs = {}, []
    for k in SOURCE_FOLDS:
        record = authority["tables"][f"aic_k{k}_fit"]
        table = Path(record["path"])
        manifest, keys = Path(f"{table}.manifest.json"), table.with_name(f"aic_k{k}_fit.keys.json")
        if (record["omitted_source_fold"] != k or record["rows"] != 3360 or len(record["sources"]) != 24
                or sha(manifest) != record["manifest_sha256"]):
            raise ValueError(f"D6 fold {k} table differs from its pinned authority")
        tables[str(k)] = dict(path=str(table), sha256=sha(table), manifest_sha256=sha(manifest), keys_sha256=sha(keys))
        inputs += [table, manifest, keys]
    frozen = freeze(run, commit, trainer, meta_tool, fold)
    root = Path(root or fold["common_root"])
    for record in initial.values():
        inputs += [Path(record["directory"]) / "refit/last.bin", *map(Path, record["input_sha256"])]
    inputs += [Path(fold["columns"]), root / "human_role_decision.json", root / "wide/frozen.json"]
    value = dict(schema=schema, protocol=protocol, registration=registration, **frozen, arms=dict(arms),
                 seeds=list(seeds), source_folds=list(SOURCE_FOLDS), epochs=EPOCHS, checkpoints=list(checkpoints),
                 lr=LR, spec=SPEC, root=str(root), columns=fold["columns"], initial=initial,
                 fold_tables=tables, assignment_sha256=authority["assignment_sha256"],
                 runs=len(fold_runs(dict(arms=arms, seeds=seeds))),
                 input_sha256={str(p): sha(p) for p in inputs}, populations=["aic", *FOLDS],
                 representation="final-f32 checkpoint dumps", selector=None, public_test_opened=False,
                 eval_read=False, production_qualified=False, **extra)
    write(run / "PLAN.json", value, fresh=True)
    print(json.dumps(dict(plan=str(run / "PLAN.json"), sha256=sha(run / "PLAN.json"))))


def gates_finished(log=None):
    log = Path(GATES_LOG if log is None else log)
    lines = log.read_text().splitlines() if log.exists() else []
    return any(line.split(" ", 1)[-1].startswith(("GATES-DONE", "GATES-EXIT")) for line in lines)


def fold_queue(path):
    """Wait, holding no lock, for the candidate gates queue to end; then take the shared lock."""
    run_dir = path.parent
    schema = plan_schema(path)
    plan_value = json.loads(path.read_text())
    after = plan_value.get("after_run") or plan_value.get("after_log", str(GATES_LOG))
    state = dict(plan_sha256=sha(path), worker=process_snapshot(os.getpid()),
                 stage="waiting-for-predecessor" if "after_run" in plan_value else "waiting-for-candidate-gates",
                 after_log=after, failures=[], checked_utc=now())
    write(run_dir / "QUEUE.json", state, fresh=True)
    try:
        while not predecessor_finished(plan_value):
            time.sleep(60)
        checked(path, schema)
        command = ["flock", str(Path.home() / "tmp/heavy.lock"), str(Path.home() / "work/claudehints/scripts/run-heavy"),
                   "--mem", "40G", "--jobs", "12", "--", sys.executable, "-B", str(Path(__file__).resolve()),
                   {OMISSION_SCHEMA: "omission-run", FULL_SCHEMA: "full-run"}.get(schema, "fold-run"), str(path)]
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


def out_of_fold(values, folds, label):
    """One complete out-of-fold prediction from the seven fold models' held-out rows."""
    import numpy as np
    oof = np.full(len(folds), np.nan)
    for k in SOURCE_FOLDS:
        oof[folds == k] = values[f"{label}|k{k}"][folds == k]
    if not np.isfinite(oof).all():
        raise ValueError(f"{label}: incomplete out-of-fold assembly")
    return oof


def fold_run(path):
    import numpy as np
    import pandas as pd
    from concurrent.futures import ThreadPoolExecutor
    if not compute_lock(os.getpid())["held_by_worker_ancestor"]:
        raise ValueError("the diagnostic requires the shared compute lock")
    run_dir, output = path.parent, path.parent / "panels"
    plan_value = checked(path, plan_schema(path))
    arms, seeds, checkpoints = plan_value["arms"], plan_value["seeds"], plan_value["checkpoints"]
    reused = plan_value.get("reused_arms", {})
    every_arm = [*arms, *reused]
    mean_label = f"mean{len(seeds)}"
    def checkpoint(arm, seed, k, e):
        base = Path(reused[arm]["run_dir"]) if arm in reused else run_dir
        return base / "arms" / arm / f"s{seed}_k{k}/ckpt/ckpt_epoch{e:03d}.bin"
    require_clean_factors(os.environ)
    fold = json.loads(Path(plan_value["registered_inputs"]["fold_plan"]["path"]).read_text())
    if (run_dir / "STATE.json").exists():
        raise ValueError("preserve previous diagnostic run state")
    state = dict(stage="fit-folds", worker=process_snapshot(os.getpid()), plan_sha256=sha(path), completed=[],
                 failures=[], required=len(fold_runs(plan_value)), objective_complete=False, production_qualified=False)
    def stage(name, **details):
        state.update(stage=name, checked_utc=now(), **details)
        write(run_dir / "STATE.json", state)
    def save(name, value):
        write(output / name, finite_json(value), fresh=True)
    environment = dict(os.environ, REV4_V2_BIN_DIR=str(run_dir / "bin"), RAYON_NUM_THREADS="1",
                       OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", ZENSIM_MAX_TIER="v3")
    def interpolate(arm, seed, k, dest, label):
        """Stage F: the bake owner emits this fold's point between the initial model and the end arm's checkpoint."""
        spec = plan_value["interpolations"][arm]
        end = plan_value["ends"][spec["end"]]
        ckpt = lambda e: Path(end["run_dir"]) / "arms" / end["arm"] / f"s{seed}_k{k}/ckpt/ckpt_epoch{e:03d}.bin"
        model_b, track = ckpt(end["epoch"]), [ckpt(e) for e in range(end["epoch"])]
        if (sha(model_b) != end["checkpoint_sha256"][f"{seed}|{k}"]
                or [sha(c) for c in track] != end["track_sha256"][f"{seed}|{k}"]):
            raise ValueError(f"{label}: end checkpoint or its track changed")
        out = dest / "ckpt/ckpt_epoch000.bin"
        out.parent.mkdir(parents=True)
        with (run_dir / "arms" / arm / f"s{seed}_k{k}.log").open("x") as log:
            if subprocess.run([str(run_dir / "tools/bake_dial_refit"), "interpolate", "--a",
                               str(Path(plan_value["initial"][str(seed)]["directory"]) / "refit/last.bin"), "--b",
                               str(model_b), "--alpha", repr(spec["alpha"]), "--out", str(out),
                               *(arg for c in track for arg in ("--track", str(c)))],
                              stdout=log, stderr=subprocess.STDOUT).returncode:
                raise RuntimeError(f"{label} interpolation failed; preserved in its log")
        return dict(arm=arm, seed=seed, fold=k, directory=str(dest), checkpoints={"0": sha(out)})
    def fit(run):
        arm, seed, k = run
        label = f"{arm}|s{seed}|k{k}"
        dest = run_dir / "arms" / arm / f"s{seed}_k{k}"
        if arm in plan_value.get("interpolations", {}):
            return interpolate(arm, seed, k, dest, label)
        scratch = Path.home() / "tmp" / f"{run_dir.name}-{arm}-s{seed}-k{k}"
        scratch.mkdir(parents=True)
        start = plan_value.get("run_initial", {}).get(f"s{seed}_k{k}")
        if start and sha(start["path"]) != start["sha256"]:
            raise ValueError(f"{label}: continuation checkpoint changed")
        command = fit_command(plan_value, run_dir, arm, seed, plan_value["initial"][str(seed)]["directory"],
                              plan_value["fold_tables"][str(k)]["path"], dest, start["path"] if start else None)
        dest.parent.mkdir(parents=True, exist_ok=True)
        env, confidence = dict(environment, TMPDIR=str(scratch)), None
        if plan_value.get("confidence") and arms[arm] != "aic-only":
            from confidence_weights import continuation_manifest
            from mlic_confidence_full import child_environment
            initial = Path(start["path"] if start else Path(plan_value["initial"][str(seed)]["directory"]) / "refit/last.bin")
            (run_dir / "confidence").mkdir(exist_ok=True)
            confidence = continuation_manifest(plan_value["confidence"], initial, plan_value["aic_weights"][arm],
                                               run_dir / "confidence" / f"{arm}_s{seed}_k{k}.json")
            env = child_environment(env, confidence, dest)
        with (run_dir / "arms" / arm / f"s{seed}_k{k}.log").open("x") as log:
            if subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env).returncode:
                raise RuntimeError(f"{label} fit failed; preserved in its log")
        outputs = fixed_outputs(dest, arms[arm], label, checkpoints)
        if confidence:
            from confidence_weights import verify_continuation_receipt
            outputs["confidence"] = dict(confidence, receipt_sha256=verify_continuation_receipt(
                dest / "CONFIDENCE_NORMALIZERS.json", confidence, plan_value["confidence"]))
        return dict(arm=arm, seed=seed, fold=k, **outputs)
    control = plan_value.get("control")
    replay = bool(control) and control.get("kind") == "full-fit-replay"
    def replay_control():
        """Stage B: the recipe's original 120-epoch full fit through its wrapper route, with this trainer."""
        from aic_adaptation_full import fit_command as full_fit_command
        from mlic_confidence_full import child_environment
        dest = run_dir / "control" / f"{control['recipe']}_full_s{control['seed']}"
        scratch = Path.home() / "tmp" / f"{run_dir.name}-control"
        scratch.mkdir(parents=True)
        dest.parent.mkdir(parents=True, exist_ok=True)
        command = full_fit_command(run_dir / "source", Path(plan_value["root"]), dest,
                                   Path(plan_value["columns"]).read_text().strip(), control["seed"],
                                   matching=plan_value.get("matching"))
        env = child_environment(dict(environment, TMPDIR=str(scratch), ZEN_PANEL_BIN=str(run_dir / "bin/panel")),
                                plan_value.get("confidence"), dest)
        with (run_dir / "control" / "fit.log").open("x") as log:
            log.write(json.dumps(command) + "\n")
            log.flush()
            if subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env).returncode:
                raise RuntimeError("control replay failed; preserved in its log")
        return dest
    def fit_control():
        if replay:
            return replay_control()
        dest = run_dir / "control" / f"{control['arm']}_s{control['seed']}_k{control['fold']}"
        scratch = Path.home() / "tmp" / f"{run_dir.name}-control"
        scratch.mkdir(parents=True)
        command = fit_command(dict(plan_value, arms={control["arm"]: control["population"]}), run_dir, control["arm"],
                              control["seed"], plan_value["initial"][str(control["seed"])]["directory"],
                              plan_value["fold_tables"][str(control["fold"])]["path"], dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with (run_dir / "control" / "fit.log").open("x") as log:
            if subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=dict(environment, TMPDIR=str(scratch))).returncode:
                raise RuntimeError("control fit failed; preserved in its log")
        return dest
    try:
        stage("fit-folds")
        with ThreadPoolExecutor(max_workers=12) as pool:
            control_future = pool.submit(fit_control) if control else None
            futures = {pool.submit(fit, run): run for run in fold_runs(plan_value)}
            for future, run in futures.items():
                try:
                    state["completed"].append(future.result())
                except BaseException as error:
                    state["failures"].append(dict(run=list(run), error=repr(error)))
                stage("fit-folds")
            control_dest = control_future.result() if control_future else None
        if state["failures"]:
            raise RuntimeError("fold fit failures retained; inspect their complete logs")
        write(run_dir / "FITS_COMPLETE.json", dict(plan_sha256=sha(path), runs=state["completed"]), fresh=True)
        if replay:
            stage("trainer-identity-control")
            from aic_adaptation_full import check_fit, require_full_control_parity
            original = Path(control["original"])
            outcome = dict(control, replay=str(control_dest))
            try:
                outcome["replay_record"] = check_fit(control_dest, control["seed"], sha(Path(plan_value["root"]) / "wide/frozen.json"))
                outcome["parity"] = require_full_control_parity(original, control_dest, run_dir / "tools/bake_dial_refit",
                                                                run_dir / "control" / "parity")
                if plan_value.get("confidence"):
                    receipts = [sha(d / "CONFIDENCE_NORMALIZERS.json") for d in (original, control_dest)]
                    outcome["confidence_receipts_sha256"] = receipts
                    if receipts[0] != receipts[1]:
                        raise ValueError("native confidence normalizer receipt changed")
                outcome["identical"] = True
            except (ValueError, KeyError) as error:
                outcome.update(identical=False, error=repr(error))
            write(run_dir / "CONTROL.json", outcome, fresh=True)
            if not outcome["identical"]:
                raise ValueError("this trainer does not replay the recipe's full fit; its contrasts are not interpretable")
        elif plan_value.get("reproduces"):
            # Stage C: this run's seed-0 fits must equal the advancing run's, every fold and epoch.
            stage("reproduction-control")
            reproduces = plan_value["reproduces"]
            (run_dir / "control").mkdir()
            comparisons = {}
            for label, pin in reproduces["checkpoint_sha256"].items():
                seed, k, e = (int(x) for x in label.split("|"))
                original = (Path(reproduces["run_dir"]) / "arms" / reproduces["arm"] / f"s{seed}_k{k}/ckpt"
                            / f"ckpt_epoch{e:03d}.bin")
                if sha(original) != pin:
                    raise ValueError(f"reproduced checkpoint changed: {label}")
                comparisons[label] = nonrepro_identical(run_dir / "tools/bake_dial_refit",
                                                        checkpoint(reproduces["arm"], seed, k, e), original,
                                                        run_dir / "control" / f"s{seed}_k{k}_e{e:03d}")
            identical = all(c["identical"] for c in comparisons.values())
            write(run_dir / "CONTROL.json", dict(reproduces=reproduces["run_dir"], arm=reproduces["arm"],
                                                 comparisons=comparisons, identical=identical), fresh=True)
            if not identical:
                raise ValueError("seed 0 does not reproduce the advancing run; the three-seed study is not interpretable")
        elif control:
            stage("trainer-identity-control")
            comparisons = {str(e): nonrepro_identical(run_dir / "tools/bake_dial_refit",
                                                      control_dest / "ckpt" / f"ckpt_epoch{e:03d}.bin",
                                                      checkpoint(control["arm"], control["seed"], control["fold"], e),
                                                      run_dir / "control" / f"epoch{e:03d}")
                           for e in control["epochs"]}
            identical = all(c["identical"] for c in comparisons.values())
            write(run_dir / "CONTROL.json", dict(control, comparisons=comparisons, identical=identical), fresh=True)
            if not identical:
                raise ValueError("this trainer does not reproduce the reused fits; the frontier contrasts are not interpretable")
        for arm, info in reused.items():
            for label, pin in info["checkpoint_sha256"].items():
                seed, k, e = (int(x) for x in label.split("|"))
                if sha(checkpoint(arm, seed, k, e)) != pin:
                    raise ValueError(f"reused {arm} checkpoint changed: {label}")
        stage("native-sampling-receipts")
        tool = run_dir / "tools/bake_dial_refit"
        # Order's matched-codec route trains without sample-coverage replay; its own arm has no receipt. Stage F's
        # interpolation points are not trained at all.
        unrecorded = [a for a in arms if (plan_value.get("sample_coverage") is False and arms[a] != "aic-only")
                      or a in plan_value.get("interpolations", {})]
        exposure = {f"{arm}@e{e}|s{seed}|k{k}": sampler_receipt(tool, checkpoint(arm, seed, k, e), e)
                    for arm in every_arm if arm not in unrecorded
                    for seed in seeds for k in SOURCE_FOLDS for e in checkpoints}
        write(run_dir / "EXPOSURE.json", dict(source="zentrain.sample_coverage prefix receipts, native replay",
              runs=exposure, not_recorded={a: "weight interpolation, not trained" if a in plan_value.get("interpolations", {})
                                           else "matched-codec route trains without sample-coverage replay"
                                           for a in unrecorded}), fresh=True)
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
        pairs, populations = full_populations(fold, output, "registered-source-fold-matched-continuation-panels")
        assignment = json.loads(Path(INPUTS["aic_authority"][0]).read_text())["assignment"]
        folds = pairs.group.map(assignment).to_numpy(dtype=int)
        if sorted(np.bincount(folds).tolist()) != [560] * 7:
            raise ValueError("D6 seven-fold source assignment differs")
        save("POPULATIONS.json", {k: dict(table=str(v["table"]), sha256=sha(v["table"]), rows=len(v["target"]))
                                  for k, v in populations.items()})
        models = {f"Initial|s{seed}": Path(plan_value["initial"][str(seed)]["directory"]) / "refit/last.bin" for seed in seeds}
        models.update({f"{arm}@e{e}|s{seed}|k{k}": checkpoint(arm, seed, k, e)
                       for arm in every_arm for seed in seeds for k in SOURCE_FOLDS for e in checkpoints})
        (output / "predictions").mkdir()
        values = {}
        for label, model in models.items():
            stage("predict", current_model=label)
            for name, population in populations.items():
                destination = output / "predictions" / f"{label}|{name}.tsv"
                cells.predict_cell(model, population["table"], destination, SPEC)
                frame = pd.read_csv(destination, sep="\t")
                prediction = frame.pred.to_numpy(dtype=float)
                if (list(frame.columns) != ["row_idx", "pred"] or len(prediction) != len(population["target"])
                        or not np.array_equal(frame.row_idx, np.arange(len(prediction))) or not np.isfinite(prediction).all()):
                    raise ValueError("ordered finite prediction coverage differs")
                values[label, name] = prediction
        aic_target = populations["aic"]["target"]
        oof = {f"Initial|s{seed}": values[f"Initial|s{seed}", "aic"] for seed in seeds}
        for arm in every_arm:
            for e in checkpoints:
                for seed in seeds:
                    oof[f"{arm}@e{e}|s{seed}"] = out_of_fold({f"{arm}@e{e}|s{seed}|k{k}": values[f"{arm}@e{e}|s{seed}|k{k}", "aic"]
                                                              for k in SOURCE_FOLDS}, folds, f"{arm}@e{e}|s{seed}")
        groups = ["Initial", *(f"{arm}@e{e}" for arm in every_arm for e in checkpoints)]
        mean = {g: np.mean([oof[f"{g}|s{seed}"] for seed in seeds], axis=0) for g in groups}
        jobs, rows = [], []
        for label, prediction in [*oof.items(), *((f"{g}|{mean_label}", v) for g, v in mean.items())]:
            jobs.append((f"{label}|out-of-fold", prediction, aic_target))
            rows.append(dict(model=label, scope="out-of-fold (Initial: zero-shot)", geometry=legacy.geometry(prediction, aic_target),
                             ladder=ladder_diagnostics(pairs, prediction, aic_target, np.ones(len(prediction), bool), logistic_fit)))
        stage("out-of-fold-panels")
        stats = panel_batch(jobs, stats="full")
        checked_statistics(jobs, stats)
        for row, value in zip(rows, stats, strict=True):
            row["statistics"] = value
        save("OUT_OF_FOLD_PANELS.json", rows)
        aggregate_jobs, aggregate_rows = [], []
        for g, prediction in mean.items():
            for category, columns in (("source", ["group"]), ("codec", ["codec"]), ("ladder", ["group", "codec"])):
                for key, index in pairs.groupby(columns, sort=True).indices.items():
                    index = np.asarray(index, dtype=int)
                    aggregate_jobs.append((f"{g}|{mean_label}|{category}|{key!r}", prediction[index], aic_target[index]))
                    aggregate_rows.append(dict(model=f"{g}|{mean_label}", category=category, group=repr(key),
                                               geometry=legacy.geometry(prediction[index], aic_target[index]) if category == "codec" else None))
            save(f"bias-{g}|{mean_label}|out-of-fold.json", cells.codec_rank_bias(prediction, aic_target, pairs.codec.to_numpy()))
        stage("out-of-fold-source-codec-ladder-panels")
        stats = panel_batch(aggregate_jobs, stats="full")
        checked_statistics(aggregate_jobs, stats)
        for row, value in zip(aggregate_rows, stats, strict=True):
            row["statistics"] = value
        save("OUT_OF_FOLD_AGGREGATE_PANELS.json", aggregate_rows)
        jobs, rows = [], []
        for (label, name), prediction in values.items():
            if label.startswith("Initial") or name != "aic":
                index = np.arange(len(prediction))
                scope = "zero-shot" if name == "aic" else "in sample"
            else:
                index = np.flatnonzero(folds != int(label.rsplit("|k", 1)[1]))
                scope = "fitted sources, in sample"
            jobs.append((f"{label}|{name}|{scope}", prediction[index], populations[name]["target"][index]))
            rows.append(dict(model=label, population=name, scope=scope, rows=len(index)))
        stage("per-model-panels")
        stats = panel_batch(jobs, stats="full")
        checked_statistics(jobs, stats)
        for row, value in zip(rows, stats, strict=True):
            row["statistics"] = value
        save("MODEL_PANELS.json", rows)
        by_model = {(r["model"], r["population"]): r["statistics"] for r in rows}
        contrasts = [tuple(c) for c in plan_value.get("contrasts", [])] or [
            ("AICOnly@e59", "MixedAIC16@e59"), ("AICOnly@e7", "MixedAIC16@e7"),
            ("MixedAIC16@e59", "MixedAIC16@e7"), ("AICOnly@e59", "AICOnly@e7"),
            *((g, "Initial") for g in groups if g != "Initial")]
        oof_stats = {r["model"]: r["statistics"] for r in json.loads((output / "OUT_OF_FOLD_PANELS.json").read_text())}
        paired = {}
        for candidate, baseline in contrasts:
            paired[f"{candidate}-minus-{baseline}|out-of-fold"] = {m: paired_delta(
                {s: oof_stats[f"{candidate}|s{s}"][m] for s in seeds}, {s: oof_stats[f"{baseline}|s{s}"][m] for s in seeds},
                seeds) for m in METRICS}
            for name in FOLDS:
                def e21(group, seed):
                    if group == "Initial":
                        return by_model[f"Initial|s{seed}", name]
                    return {m: float(np.mean([by_model[f"{group}|s{seed}|k{k}", name][m] for k in SOURCE_FOLDS]))
                            for m in METRICS}
                paired[f"{candidate}-minus-{baseline}|{name}|mean over fold models"] = {m: paired_delta(
                    {s: e21(candidate, s)[m] for s in seeds}, {s: e21(baseline, s)[m] for s in seeds}, seeds)
                    for m in METRICS}
        save("PAIRED_SEEDS.json", dict(contrasts=paired, selection_gate=False,
             note="E21 populations are in sample for every model; out-of-fold covers AIC public_train only"))
        rng = np.random.default_rng(1)
        clusters = [np.asarray(index, int) for index in pairs.groupby("group", sort=True).indices.values()]
        draws = [np.concatenate([clusters[i] for i in rng.integers(0, 28, 28)]) for _ in range(2000)]
        intervals = {}
        for candidate, baseline in contrasts:
            stage("source-intervals", current=f"{candidate}-minus-{baseline}")
            bases = dict(target=aic_target, candidate=mean[candidate], baseline=mean[baseline])
            tasks = [(f"{i}-{name}", name, "target", index) for i, index in enumerate(draws) for name in ("baseline", "candidate")]
            result = panel_batch_indexed(bases, tasks, stats="srocc")
            checked_statistics([(label, index, index) for label, _, _, index in tasks], result)
            delta = np.asarray([result[i + 1]["srocc_signed"] - result[i]["srocc_signed"] for i in range(0, len(result), 2)])
            good = delta[np.isfinite(delta)]
            intervals[f"{candidate}-minus-{baseline}"] = dict(metric="srocc_signed", aggregation=f"{len(seeds)}-seed mean out-of-fold",
                resamples=2000, source_clusters=28, seed=1, defined_resamples=len(good), quantile_method="linear",
                interval_95=np.quantile(good, [.025, .975], method="linear").tolist() if len(good) else None,
                uncertainty="whole-source resampling conditional on fitted models", descriptive_only=True, selection_gate=False)
        save("SOURCE_INTERVALS.json", intervals)
        stage("pixel-mixtures")
        pixel_mixtures(fold, source, output, {label.replace("@", "_").replace("|", "_"): model for label, model in models.items()}, save)
        inventory = {str(p.relative_to(run_dir)): sha(p) for p in sorted(run_dir.rglob("*"))
                     if p.is_file() and p.name not in ("STATE.json", "QUEUE.json", "CURRENT.json")
                     and p.relative_to(run_dir).parts[0] != "source"}
        write(run_dir / "EVIDENCE_INVENTORY.json", inventory, fresh=True)
        write(run_dir / "COMPLETE.json", dict(plan_sha256=sha(path), evidence_inventory_sha256=sha(run_dir / "EVIDENCE_INVENTORY.json"),
              selector=None, production_qualified=False, objective_complete=False,
              assessment=plan_value.get("assessment", "three-seed source-fold mechanism/stability evidence; no adoption")),
              fresh=True)
        stage("fold-diagnostic-complete", next_action="review out-of-fold and drift panels; no adoption from this study")
    except BaseException as error:
        stage("failed", failures=[*state["failures"], repr(error)])
        raise


# Stage C, E21 out of sample (registered with stage A): the Mlic omission cells (one per E21 dataset and seed) continue
# at the advanced budget, each with the same rotating AIC fold table as the common adaptation study; their held-out
# dataset predictions at every studied epoch are scored by the existing E21 owner against Mlic and MlicFT16x8.
OMISSION_SCHEMA = "aic-share-omission-plan-v1"
MLIC_CELLS = Path("/var/tmp/byc-mixes-20261025/cells")
FT_CELLS = Path("/mnt/v/output/zensim/aic-finetune-20261026/run-05/cells")


def omission_cells():
    return [(fold, seed) for fold in FOLDS for seed in SEEDS]


def omission_aic_fold(fold, seed):
    """The common adaptation study's rotation, so no omission cell shares its AIC held-out sources across seeds."""
    return (4 * seed + FOLDS.index(fold)) % 7


def omission_plan(run, commit, trainer, meta_tool, point_run, arm, after_run):
    from aic_adaptation import initial_cell
    point = json.loads((Path(point_run) / "PLAN.json").read_text())
    if point.get("schema") not in (FRONTIER_SCHEMA, RECOVERY_SCHEMA, RECIPE_SCHEMA, XREF_SCHEMA) or arm not in point["arms"]:
        raise ValueError("the advanced point must be a frontier, recovery, rehearsal or recipe arm")
    recipe, _, recipe_root = recipe_point(point)
    run = Path(run).resolve()
    if run.exists():
        raise ValueError("preserve existing diagnostic attempts")
    population = point["arms"][arm]
    protocol, weight, _ = diagnostic_protocol(population)
    for path, pin in INPUTS.values():
        if sha(path) != pin:
            raise ValueError(f"registered input changed: {path}")
    fold = json.loads(Path(INPUTS["fold_plan"][0]).read_text())
    authority = json.loads(Path(INPUTS["aic_authority"][0]).read_text())
    root = recipe_root or Path(fold["common_root"])
    cell_dir = RECIPE_CELLS[recipe] / recipe if recipe else MLIC_CELLS / "Mlic"
    baselines = dict(Mlic=str(MLIC_CELLS), MlicFT16x8=str(FT_CELLS))
    if recipe:  # the recipe's own omission cells and its common 16x8 continuation
        baselines.update({recipe: str(RECIPE_CELLS[recipe]), f"{recipe}FT16x8": str(RECIPE_FT_CELLS)})
    from subjective_training_leg import linked_subset_exclusions
    cells, inputs, tables = {}, [Path(fold["columns"]), root / "human_role_decision.json", root / "wide/frozen.json"], {}
    for name, seed in omission_cells():
        cell = initial_cell(cell_dir / f"without_{name}_s{seed}", name, seed)
        selection = json.loads((cell_dir / f"without_{name}_s{seed}/DATASET_GROUP_SELECTION.json").read_text())
        if Path(selection["root"]) != root:
            raise ValueError(f"{name} s{seed}: omission cell trained on another root")
        k = omission_aic_fold(name, seed)
        record = authority["tables"][f"aic_k{k}_fit"]
        table = Path(record["path"])
        tables[str(k)] = dict(path=str(table), sha256=sha(table))
        cells[f"{name}_s{seed}"] = dict(cell, aic_fold=k, exclusions=list(linked_subset_exclusions(root, name)))
        inputs += [*map(Path, cell["input_sha256"]), table, Path(f"{table}.manifest.json")]
    frozen = freeze(run, commit, trainer, meta_tool, fold)
    value = dict(schema=OMISSION_SCHEMA, protocol=protocol,
                 registration=RECIPE_C_REGISTRATION if recipe else FRONTIER_REGISTRATION, **frozen, recipe=recipe,
                 arms={arm: population}, aic_weights={arm: weight}, seeds=list(SEEDS), e21_folds=list(FOLDS),
                 epochs=EPOCHS, checkpoints=list(FRONTIER_CHECKPOINTS), lr=LR, spec=SPEC, root=str(root),
                 columns=fold["columns"], cells=cells, aic_tables=tables, baselines=baselines,
                 point=dict(run=str(point_run), arm=arm),
                 phases=two_phases(point, arm, population),
                 after_run=str(after_run), input_sha256={str(p): sha(p) for p in inputs},
                 representation="final-f32 checkpoint dumps", selector=None, public_test_opened=False, eval_read=False,
                 production_qualified=False, assessment="stage C E21 out of sample for the advanced point; no adoption")
    write(run / "PLAN.json", value, fresh=True)
    print(json.dumps(dict(plan=str(run / "PLAN.json"), sha256=sha(run / "PLAN.json"))))


def two_phases(point, arm, population):
    """[(arm, population, start epoch of the previous phase or None)]: a recovery point first continues AIC-only."""
    if point["schema"] != RECOVERY_SCHEMA:
        return [[arm, population, None]]
    start = point["start"]
    return [[start["arm"], ARMS[start["arm"]], None], [arm, population, int(start["epoch"])]]


def omission_command(plan_value, run_dir, arm, name, seed, dest, init=None, population=None):
    cell = plan_value["cells"][f"{name}_s{seed}"]
    table = plan_value["aic_tables"][str(cell["aic_fold"])]["path"]
    population = population or plan_value["arms"][arm]
    weight = diagnostic_protocol(population)[1]
    root = Path(plan_value["root"])
    command = [sys.executable, "-B", str(run_dir / "source/scripts/rev4_featpot/v2_lodo_mlp.py"), "--root", str(root),
               "--spec", SPEC, "--columns", Path(plan_value["columns"]).read_text().strip(), "--head", "N",
               "--heldout", name, "--seed-index", str(seed), "--strict-admission",
               "--data-role-decision", str(root / "human_role_decision.json"), "--train-only", "--dest", str(dest),
               "--init-from", str(init or Path(cell["directory"]) / "refit/last.bin"), "--fine-tune-epochs", str(EPOCHS),
               "--fine-tune-lr", str(LR), "--aic-public-train", table,
               "--research-extra-group", f"aic_public_train:{table}:{weight!r}:0:rank",
               "--diagnostic-population", population]
    return command + (["--exclude-datasets", ",".join(cell["exclusions"])] if cell["exclusions"] else [])


def omission_run(path):
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor
    if not compute_lock(os.getpid())["held_by_worker_ancestor"]:
        raise ValueError("the omission study requires the shared compute lock")
    run_dir = path.parent
    plan_value = checked(path, OMISSION_SCHEMA)
    (arm, population), = plan_value["arms"].items()
    require_clean_factors(os.environ)
    if (run_dir / "STATE.json").exists():
        raise ValueError("preserve previous diagnostic run state")
    state = dict(stage="fit-omission-cells", worker=process_snapshot(os.getpid()), plan_sha256=sha(path),
                 completed=[], failures=[], required=len(omission_cells()), objective_complete=False,
                 production_qualified=False)
    def stage(name, **details):
        state.update(stage=name, checked_utc=now(), **details)
        write(run_dir / "STATE.json", state)
    environment = dict(os.environ, REV4_V2_BIN_DIR=str(run_dir / "bin"), RAYON_NUM_THREADS="1",
                       OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", ZENSIM_MAX_TIER="v3")
    def fit(cell):
        name, seed = cell
        label = f"{arm}|{name}|s{seed}"
        if any(sha(p) != pin for p, pin in plan_value["cells"][f"{name}_s{seed}"]["input_sha256"].items()):
            raise ValueError(f"{label}: starting cell changed")
        phases, init = [], None
        # A recovery point's cell continues AIC-only first, then recovers from that phase's start epoch.
        for phase_arm, phase_population, start_epoch in plan_value.get("phases", [[arm, population, None]]):
            if start_epoch is not None:
                init = phases[-1]["directory"] + f"/ckpt/ckpt_epoch{start_epoch:03d}.bin"
            dest = run_dir / "cells" / phase_arm / f"without_{name}_s{seed}"
            scratch = Path.home() / "tmp" / f"{run_dir.name}-{phase_arm}-{name}-s{seed}"
            scratch.mkdir(parents=True)
            dest.parent.mkdir(parents=True, exist_ok=True)
            with (run_dir / "cells" / phase_arm / f"without_{name}_s{seed}.log").open("x") as log:
                if subprocess.run(omission_command(plan_value, run_dir, phase_arm, name, seed, dest, init, phase_population),
                                  stdout=log, stderr=subprocess.STDOUT, env=dict(environment, TMPDIR=str(scratch))).returncode:
                    raise RuntimeError(f"{phase_arm}|{name}|s{seed} fit failed; preserved in its log")
            phases.append(fixed_outputs(dest, phase_population, f"{phase_arm}|{name}|s{seed}", plan_value["checkpoints"]))
        return dict(fold=name, seed=seed, **phases[-1], phases=phases)
    try:
        stage("fit-omission-cells")
        with ThreadPoolExecutor(max_workers=12) as pool:
            futures = {pool.submit(fit, cell): cell for cell in omission_cells()}
            for future, cell in futures.items():
                try:
                    state["completed"].append(future.result())
                except BaseException as error:
                    state["failures"].append(dict(cell=list(cell), error=repr(error)))
                stage("fit-omission-cells")
        if state["failures"]:
            raise RuntimeError("omission fit failures retained; inspect their complete logs")
        write(run_dir / "FITS_COMPLETE.json", dict(plan_sha256=sha(path), cells=state["completed"]), fresh=True)
        stage("held-out-predictions")
        fold = json.loads(Path(plan_value["registered_inputs"]["fold_plan"]["path"]).read_text())
        os.environ.update(TMPDIR=str(Path.home() / "tmp"), REV4_V2_BIN_DIR=str(run_dir / "bin"),
                          BYC_ROOT=plan_value["root"], ZEN_PANEL_BIN=str(run_dir / "bin/panel"),
                          BYC_LABEL_LEDGER_DIR=str(run_dir / "label-reads"), RAYON_NUM_THREADS="1",
                          OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", ZENSIM_MAX_TIER="v3",
                          ZENSIM_RING_FORM="raw", ZENSIM_PYRAMID="xyb")
        source = Path(fold["warmstart_source"])
        sys.path[:0] = [str(source / "scripts/rev4_featpot"), str(source / "scripts")]
        import byc_cells as cells
        arms = [f"{arm}e{e}" for e in plan_value["checkpoints"]]
        for name, seed in omission_cells():
            for e in plan_value["checkpoints"]:
                out = run_dir / "cells" / f"{arm}e{e}" / f"without_{name}_s{seed}" / f"pred_{name}.tsv"
                out.parent.mkdir(parents=True, exist_ok=True)
                model = run_dir / "cells" / arm / f"without_{name}_s{seed}/ckpt/ckpt_epoch{e:03d}.bin"
                prediction = cells.predict_cell(model, Path(plan_value["root"]) / "wide/main/real" / f"{name}.parquet",
                                                out, SPEC)
                if not np.isfinite(prediction).all():
                    raise ValueError(f"{out}: nonfinite prediction")
        stage("e21-score")
        scores = {}
        for cells_dir, names in ((run_dir / "cells", arms), *((Path(d), [b]) for b, d in plan_value["baselines"].items())):
            result = subprocess.run([sys.executable, "-B", str(source / "scripts/rev4_featpot/byc_cells.py"), "score",
                                     "--arms", ",".join(names), "--control", "base-c", "--seeds", "0,1,2"],
                                    check=True, capture_output=True, text=True, env=dict(os.environ, BYC_CELLS=str(cells_dir)))
            scores.update(json.loads(result.stdout))
        paired = {}
        for candidate in arms:
            for baseline in plan_value["baselines"]:
                a, b = (scores[x]["per_dataset_per_seed_srocc"] for x in (candidate, baseline))
                delta = [float(np.mean([a[f][s] - b[f][s] for f in FOLDS])) for s in range(3)]
                paired[f"{candidate}-minus-{baseline}"] = dict(per_seed_delta=delta, mean=float(np.mean(delta)),
                    seed_unit_se=float(np.std(delta, ddof=1) / np.sqrt(3)), selection_gate=False,
                    per_dataset_delta={f: float(np.mean(np.asarray(a[f]) - np.asarray(b[f]))) for f in FOLDS})
        write(run_dir / "E21_OUT_OF_SAMPLE.json", finite_json(dict(scores=scores, paired=paired,
              note="E21 TRAIN datasets out of sample: each omission cell never trained on its dataset")), fresh=True)
        inventory = {str(p.relative_to(run_dir)): sha(p) for p in sorted(run_dir.rglob("*"))
                     if p.is_file() and p.name not in ("STATE.json", "QUEUE.json", "CURRENT.json")
                     and p.relative_to(run_dir).parts[0] != "source"}
        write(run_dir / "EVIDENCE_INVENTORY.json", inventory, fresh=True)
        write(run_dir / "COMPLETE.json", dict(plan_sha256=sha(path), evidence_inventory_sha256=sha(run_dir / "EVIDENCE_INVENTORY.json"),
              selector=None, production_qualified=False, objective_complete=False,
              assessment=plan_value["assessment"]), fresh=True)
        stage("omission-complete", next_action="read E21_OUT_OF_SAMPLE.json against Mlic and MlicFT16x8")
    except BaseException as error:
        stage("failed", failures=[*state["failures"], repr(error)])
        raise


# Stage C, full data (registered with stage A): full Mlic seeds 0-2 continue at the advanced budget on all of
# public_train; the advanced epoch's checkpoint is packed by the canonical deployment order (densify, f16, E33 output
# stage, TRAIN CID22 calibration), then scored in sample with Mlic and MlicFT16x8 packed models and mixture-tested.
FULL_SCHEMA = "aic-share-full-plan-v1"
PACK = ("import json, sys; from pathlib import Path; from v2_production_pack import pack_production; "
        "print(json.dumps(pack_production(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), e33_output_stage=True)))")


def full_plan(run, commit, trainer, meta_tool, point_run, arm, epoch, after_run):
    point = json.loads((Path(point_run) / "PLAN.json").read_text())
    if (point.get("schema") not in (FRONTIER_SCHEMA, RECOVERY_SCHEMA, XREF_SCHEMA, RECIPE_SCHEMA)
            or arm not in point["arms"] or epoch not in FRONTIER_CHECKPOINTS):
        raise ValueError("the advanced point must be a frontier, recovery, rehearsal or recipe arm at a studied epoch")
    population = point["arms"][arm]
    protocol, weight, _ = diagnostic_protocol(population)
    recipe, initial, recipe_root = recipe_point(point)
    fits = json.loads(Path(INPUTS["full_fits"][0]).read_text())
    baselines = {}
    for seed in SEEDS:
        if recipe:  # the recipe's own packed full model and its common 16x8 continuation
            own = initial[str(seed)]
            ft = [r for r in fits["continued"] if r["arm"] == f"{recipe}FT16x8" and r["seed"] == seed]
            if (len(ft) != 1 or sha(Path(own["directory"]) / "refit/production-f16.bin") != own["packed_sha256"]
                    or sha(Path(ft[0]["directory"]) / "refit/production-f16.bin") != ft[0]["packed_sha256"]):
                raise ValueError(f"packed {recipe} baselines differ")
            baselines[f"{recipe}|s{seed}"] = str(Path(own["directory"]) / "refit/production-f16.bin")
            baselines[f"{recipe}FT16x8|s{seed}"] = str(Path(ft[0]["directory"]) / "refit/production-f16.bin")
        ft = [r for r in fits["continued"] if r["arm"] == "MlicFT16x8" and r["seed"] == seed]
        if len(ft) != 1 or sha(Path(ft[0]["directory"]) / "refit/production-f16.bin") != ft[0]["packed_sha256"]:
            raise ValueError("packed MlicFT16x8 baseline differs")
        mlic = full_record(fits, "Mlic", seed)
        if sha(Path(mlic["directory"]) / "refit/production-f16.bin") != mlic["packed_sha256"]:
            raise ValueError("packed Mlic baseline differs")
        baselines[f"Mlic|s{seed}"] = str(Path(mlic["directory"]) / "refit/production-f16.bin")
        baselines[f"MlicFT16x8|s{seed}"] = str(Path(ft[0]["directory"]) / "refit/production-f16.bin")
    fold = json.loads(Path(INPUTS["fold_plan"][0]).read_text())
    root = recipe_root or Path(fold["common_root"])
    leg = json.loads((root / "wide/main/real/receipt.json").read_text())["legs"]["cid22"]["fit"]
    anchor = root / leg["rel"]
    if sha(anchor) != leg["sha256"]:
        raise ValueError("TRAIN CID22 calibration anchor differs")
    seven_fold_plan(run, commit, trainer, meta_tool, schema=FULL_SCHEMA, protocol=protocol,
                    registration=RECIPE_C_REGISTRATION if recipe else FRONTIER_REGISTRATION, arms={arm: population},
                    seeds=SEEDS, checkpoints=FRONTIER_CHECKPOINTS, initial=initial, root=recipe_root, extra=dict(
                        recipe=recipe,
                        aic_weights={arm: weight}, packed_epoch=epoch, anchor=dict(path=str(anchor), sha256=leg["sha256"]),
                        phases=two_phases(point, arm, population),
                        baselines={k: dict(path=v, sha256=sha(v)) for k, v in baselines.items()},
                        point=dict(run=str(point_run), arm=arm, epoch=epoch), after_run=str(after_run),
                        assessment="stage C full-data continuation of the advanced point; in-sample panels; no adoption"))


def full_run(path):
    import numpy as np
    import pandas as pd
    from concurrent.futures import ThreadPoolExecutor
    if not compute_lock(os.getpid())["held_by_worker_ancestor"]:
        raise ValueError("the full-data stage requires the shared compute lock")
    run_dir, output = path.parent, path.parent / "panels"
    plan_value = checked(path, FULL_SCHEMA)
    (arm, population), = plan_value["arms"].items()
    epoch = plan_value["packed_epoch"]
    require_clean_factors(os.environ)
    if (run_dir / "STATE.json").exists():
        raise ValueError("preserve previous diagnostic run state")
    state = dict(stage="fit-full", worker=process_snapshot(os.getpid()), plan_sha256=sha(path), completed=[],
                 failures=[], required=len(SEEDS), objective_complete=False, production_qualified=False)
    def stage(name, **details):
        state.update(stage=name, checked_utc=now(), **details)
        write(run_dir / "STATE.json", state)
    def save(name, value):
        write(output / name, finite_json(value), fresh=True)
    source = run_dir / "source/scripts"
    environment = dict(os.environ, REV4_V2_BIN_DIR=str(run_dir / "bin"), REV4_V2_ROOT=plan_value["root"],
                       ZEN_PANEL_BIN=str(run_dir / "bin/panel"), RAYON_NUM_THREADS="1", OMP_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1", ZENSIM_MAX_TIER="v3")
    def fit(seed):
        phases, init = [], None
        # A recovery point first continues AIC-only on all of public_train, then recovers from its start epoch.
        for phase_arm, phase_population, start_epoch in plan_value.get("phases", [[arm, population, None]]):
            if start_epoch is not None:
                init = Path(phases[-1]["directory"]) / "ckpt" / f"ckpt_epoch{start_epoch:03d}.bin"
            dest = run_dir / "full" / phase_arm / f"full_s{seed}"
            scratch = Path.home() / "tmp" / f"{run_dir.name}-{phase_arm}-full-s{seed}"
            scratch.mkdir(parents=True)
            dest.parent.mkdir(parents=True, exist_ok=True)
            env = dict(environment, TMPDIR=str(scratch))
            with (run_dir / "full" / phase_arm / f"full_s{seed}.log").open("x") as log:
                if subprocess.run(fit_command(dict(plan_value, arms={phase_arm: phase_population}), run_dir, phase_arm,
                                              seed, plan_value["initial"][str(seed)]["directory"], None, dest, init),
                                  stdout=log, stderr=subprocess.STDOUT, env=env).returncode:
                    raise RuntimeError(f"full {phase_arm} seed {seed} fit failed; preserved in its log")
            phases.append(fixed_outputs(dest, phase_population, f"{phase_arm}|full|s{seed}", plan_value["checkpoints"]))
        outputs = dict(phases[-1], phases=phases)
        if sha(plan_value["anchor"]["path"]) != plan_value["anchor"]["sha256"]:
            raise ValueError("TRAIN CID22 calibration anchor changed")
        pack = dest / f"pack_e{epoch:03d}"
        (pack / "refit").mkdir(parents=True)
        with (pack / "pack-wrapper.log").open("x") as log:
            result = subprocess.run([sys.executable, "-B", "-c", PACK, str(dest / "ckpt" / f"ckpt_epoch{epoch:03d}.bin"),
                                     str(pack), plan_value["anchor"]["path"], "--root", plan_value["root"]],
                                    stdout=subprocess.PIPE, stderr=log, text=True, env=dict(env, PYTHONPATH=os.pathsep.join(
                                        [str(source / "rev4_featpot"), str(source)])))
        if result.returncode:
            raise RuntimeError(f"full seed {seed} packing failed; preserved in its log")
        packed = json.loads(result.stdout.strip().splitlines()[-1])
        write(pack / "PACKED.json", dict(packed, source_checkpoint=str(dest / "ckpt" / f"ckpt_epoch{epoch:03d}.bin"),
              source_checkpoint_sha256=sha(dest / "ckpt" / f"ckpt_epoch{epoch:03d}.bin")), fresh=True)
        return dict(seed=seed, **outputs, packed=packed)
    try:
        stage("fit-full")
        with ThreadPoolExecutor(max_workers=3) as pool:
            for seed, future in [(s, pool.submit(fit, s)) for s in SEEDS]:
                try:
                    state["completed"].append(future.result())
                except BaseException as error:
                    state["failures"].append(dict(seed=seed, error=repr(error)))
                stage("fit-full")
        if state["failures"]:
            raise RuntimeError("full fit failures retained; inspect their complete logs")
        write(run_dir / "FITS_COMPLETE.json", dict(plan_sha256=sha(path), cells=state["completed"]), fresh=True)
        for name, record in plan_value["baselines"].items():
            if sha(record["path"]) != record["sha256"]:
                raise ValueError(f"packed baseline changed: {name}")
        stage("raw-panels")
        fold = json.loads(Path(plan_value["registered_inputs"]["fold_plan"]["path"]).read_text())
        os.environ.update(TMPDIR=str(Path.home() / "tmp"), REV4_V2_BIN_DIR=str(run_dir / "bin"), AIC_BIN=str(run_dir / "bin"),
                          BYC_ROOT=fold["common_root"], ZEN_PANEL_BIN=str(run_dir / "bin/panel"),
                          BYC_LABEL_LEDGER_DIR=str(output / "label-reads"), RAYON_NUM_THREADS="1", OMP_NUM_THREADS="1",
                          OPENBLAS_NUM_THREADS="1", ZENSIM_MAX_TIER="v3", ZENSIM_RING_FORM="raw", ZENSIM_PYRAMID="xyb")
        warm = Path(fold["warmstart_source"])
        sys.path[:0] = [str(warm / "scripts/rev4_featpot"), str(warm / "scripts/byconstruction"), str(warm / "scripts")]
        import byc_cells as cells
        from lib.zen_stats import panel_batch
        output.mkdir()
        _, populations = full_populations(fold, output, "registered-AIC-share-full-panels")
        models = {f"{arm}e{epoch}|s{s}": run_dir / "full" / arm / f"full_s{s}/pack_e{epoch:03d}/refit/production-f16.bin"
                  for s in SEEDS}
        models.update({name: Path(record["path"]) for name, record in plan_value["baselines"].items()})
        (output / "predictions").mkdir()
        jobs, rows = [], []
        for label, model in models.items():
            stage("predict", current_model=label)
            for name, population in populations.items():
                destination = output / "predictions" / f"{label}|{name}.tsv"
                cells.predict_cell(model, population["table"], destination, SPEC)
                frame = pd.read_csv(destination, sep="\t")
                prediction = frame.pred.to_numpy(dtype=float)
                if (list(frame.columns) != ["row_idx", "pred"] or len(prediction) != len(population["target"])
                        or not np.array_equal(frame.row_idx, np.arange(len(prediction))) or not np.isfinite(prediction).all()):
                    raise ValueError("ordered finite prediction coverage differs")
                jobs.append((f"{label}|{name}|in sample", prediction, population["target"]))
                rows.append(dict(model=label, population=name, scope="in sample", rows=len(prediction)))
        stage("per-model-panels")
        stats = panel_batch(jobs, stats="full")
        checked_statistics(jobs, stats)
        for row, value in zip(rows, stats, strict=True):
            row["statistics"] = value
        save("MODEL_PANELS.json", rows)
        by_model = {(r["model"], r["population"]): r["statistics"] for r in rows}
        paired = {}
        for baseline in sorted({name.split("|")[0] for name in plan_value["baselines"]}):
            for name in populations:
                paired[f"{arm}e{epoch}-minus-{baseline}|{name}"] = {m: seed_delta(
                    {s: by_model[f"{arm}e{epoch}|s{s}", name][m] for s in SEEDS},
                    {s: by_model[f"{baseline}|s{s}", name][m] for s in SEEDS}) for m in METRICS}
        save("PAIRED_SEEDS.json", dict(contrasts=paired, selection_gate=False, representation="packed f16, calibrated",
                                       note="every population is in sample for every full model"))
        stage("pixel-mixtures")
        pixel_mixtures(fold, warm, output, {label.replace("|", "_"): model for label, model in models.items()
                                            if label.startswith(arm) or label.endswith("|s0")}, save)
        inventory = {str(p.relative_to(run_dir)): sha(p) for p in sorted(run_dir.rglob("*"))
                     if p.is_file() and p.name not in ("STATE.json", "QUEUE.json", "CURRENT.json")
                     and p.relative_to(run_dir).parts[0] != "source"}
        write(run_dir / "EVIDENCE_INVENTORY.json", inventory, fresh=True)
        write(run_dir / "COMPLETE.json", dict(plan_sha256=sha(path), evidence_inventory_sha256=sha(run_dir / "EVIDENCE_INVENTORY.json"),
              selector=None, production_qualified=False, objective_complete=False,
              assessment=plan_value["assessment"]), fresh=True)
        stage("full-complete", next_action="near-identity and the label-free gates for the packed seed-0 model")
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


def fold_status(path):
    run_dir = path.parent
    state_path = run_dir / "STATE.json"
    state = json.loads((state_path if state_path.exists() else run_dir / "QUEUE.json").read_text())
    worker = state["worker"]
    live = bound_worker(worker["pid"], worker["start_ticks"])
    if not live and state["stage"] not in ("fold-diagnostic-complete", "run-returned") and not state["failures"]:
        state["failures"].append("recorded diagnostic process exited without completion")
    logged = []
    for log in sorted([*run_dir.glob("arms/*/s*_k*/train.log"), *run_dir.glob("cells/*/without_*_s*/train.log")]):
        done = [int(line.split()[1]) for line in log.read_text().splitlines() if line.startswith("  epoch ")]
        logged.append(max(done) if done else -1)
    state.update(checked_utc=now(), live=live, gates_finished=predecessor_finished(json.loads(path.read_text())),
                 fits_complete=len((lambda v: v.get("runs") or v.get("cells"))(json.loads((run_dir / "FITS_COMPLETE.json").read_text())))
                 if (run_dir / "FITS_COMPLETE.json").exists() else 0,
                 logged_epoch_range=[min(logged), max(logged)] if logged else None, logged_runs=len(logged),
                 compute_lock=compute_lock(worker["pid"] if live else None))
    write(run_dir / "CURRENT.json", state)
    print(json.dumps({k: state.get(k) for k in ("stage", "live", "gates_finished", "fits_complete", "logged_runs",
                                                 "logged_epoch_range", "failures", "compute_lock")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    for mode in ("plan", "fold-plan", "frontier-plan", "recovery-plan"):
        p = sub.add_parser(mode)
        p.add_argument("run", type=Path)
        p.add_argument("--source-commit", required=True)
        p.add_argument("--trainer", type=Path, required=True)
        p.add_argument("--meta-tool", type=Path, required=True)
    for mode in ("queue", "run", "status", "fold-queue", "fold-run", "fold-status", "omission-run", "full-run"):
        sub.add_parser(mode).add_argument("plan", type=Path)
    p = sub.add_parser("full-plan", help="stage C: full Mlic seeds 0-2 at the advanced point, packed at its epoch")
    p.add_argument("run", type=Path)
    p.add_argument("--source-commit", required=True)
    p.add_argument("--trainer", type=Path, required=True)
    p.add_argument("--meta-tool", type=Path, required=True)
    p.add_argument("--point", nargs=3, metavar=("RUN", "ARM", "EPOCH"), required=True)
    p.add_argument("--after-run", type=Path, required=True)
    p = sub.add_parser("omission-plan", help="stage C: E21 out of sample through the Mlic omission cells")
    p.add_argument("run", type=Path)
    p.add_argument("--source-commit", required=True)
    p.add_argument("--trainer", type=Path, required=True)
    p.add_argument("--meta-tool", type=Path, required=True)
    p.add_argument("--point", nargs=2, metavar=("RUN", "ARM"), required=True, help="the advancing run and arm")
    p.add_argument("--after-run", type=Path, required=True)
    p = sub.add_parser("recipe-plan", help="stage B: one recipe at the advanced budget")
    p.add_argument("run", type=Path)
    p.add_argument("--source-commit", required=True)
    p.add_argument("--trainer", type=Path, required=True)
    p.add_argument("--meta-tool", type=Path, required=True)
    p.add_argument("--recipe", choices=RECIPES, required=True)
    p.add_argument("--population", choices=sorted(RECIPE_ARM), nargs="+", required=True)
    p.add_argument("--after-run", type=Path, required=True, help="predecessor run; this one waits for it")
    p.add_argument("--compare", nargs=2, metavar=("RUN", "ARM"), action="append", required=True,
                   help="Mlic's point at the same budget, from stage A, A2 or D; once per population, in order")
    p.add_argument("--start", nargs=3, metavar=("RUN", "ARM", "EPOCH"),
                   help="recovery phase: continue from that run's fold checkpoints")
    p.add_argument("--no-control", action="store_true", help="the recipe's replay control already ran in its first phase")
    p = sub.add_parser("wise-plan", help="stage F: weight interpolation between Mlic and AIC-heavy fold models")
    p.add_argument("run", type=Path)
    p.add_argument("--source-commit", required=True)
    p.add_argument("--trainer", type=Path, required=True)
    p.add_argument("--meta-tool", type=Path, required=True, help="bake owner build with `interpolate`")
    p.add_argument("--after-run", type=Path, required=True)
    p = sub.add_parser("xref-plan", help="stage D: the within-dataset rehearsal frontier at x256 and x1024")
    p.add_argument("run", type=Path)
    p.add_argument("--source-commit", required=True)
    p.add_argument("--trainer", type=Path, required=True)
    p.add_argument("--meta-tool", type=Path, required=True)
    p.add_argument("--after-run", type=Path, required=True)
    p = sub.add_parser("seeds-plan", help="stage C: the advanced Mlic point for seeds 0-2 on the folds")
    p.add_argument("run", type=Path)
    p.add_argument("--source-commit", required=True)
    p.add_argument("--trainer", type=Path, required=True)
    p.add_argument("--meta-tool", type=Path, required=True)
    p.add_argument("--point", nargs=2, metavar=("RUN", "ARM"), required=True, help="the advancing run and arm")
    p.add_argument("--after-run", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "xref-plan":
        xref_plan(args.run, args.source_commit, args.trainer, args.meta_tool, args.after_run)
    elif args.mode == "wise-plan":
        wise_plan(args.run, args.source_commit, args.trainer, args.meta_tool, args.after_run)
    elif args.mode == "full-plan":
        full_plan(args.run, args.source_commit, args.trainer, args.meta_tool, Path(args.point[0]), args.point[1],
                  int(args.point[2]), args.after_run)
    elif args.mode == "omission-plan":
        omission_plan(args.run, args.source_commit, args.trainer, args.meta_tool, Path(args.point[0]), args.point[1],
                      args.after_run)
    elif args.mode == "seeds-plan":
        seeds_plan(args.run, args.source_commit, args.trainer, args.meta_tool, Path(args.point[0]), args.point[1],
                   args.after_run)
    elif args.mode == "recipe-plan":
        recipe_plan(args.run, args.source_commit, args.trainer, args.meta_tool, args.recipe, args.population,
                    args.after_run, [(Path(r), a) for r, a in args.compare], args.start, not args.no_control)
    elif args.mode in ("plan", "fold-plan", "frontier-plan", "recovery-plan"):
        globals()[args.mode.replace("-", "_")](args.run, args.source_commit, args.trainer, args.meta_tool)
    else:
        globals()[args.mode.replace("-", "_")](args.plan.resolve())
