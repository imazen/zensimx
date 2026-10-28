"""Complete in-sample panels for full AIC recipe fits and their serving bakes."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aic_adaptation import FOLDS, RECIPES, SPEC, STRENGTHS, bound_worker, compute_lock, process_snapshot, sha, write
from aic_adaptation_panel import finite_json
from aic_adaptation_full import full_recipes
from aic_adaptation import require_clean_factors
from aic_adaptation_diagnostics import checked_statistics, complete_mean, ladder_diagnostics

METRICS = ("srocc_signed", "krocc", "plcc", "plcc_raw", "or", "pwrc", "z_rmse", "mae")
REPRESENTATIONS = {"final-f32": "last.bin", "production-f16-calibrated": "production-f16.bin"}
LIVE_EXPOSED = Path("/var/tmp/byc-fx2-20261028/live/LIVE.json")


def live_rows(value):
    """Use only the previously exposed manifest, without reopening a LIVE table."""
    import math
    rows = value["rows"]
    if (len(rows) != 36 or value["rows_read"] != 36 or len({r["index"] for r in rows}) != 36
            or sum(r["case"] is True for r in rows) != 10
            or any(type(r["case"]) is not bool or not math.isfinite(r["human"]) for r in rows)):
        raise ValueError("the fixed 36-row, ten-case exposed LIVE scope differs")
    for row in rows:
        if row["case"] and not any(not c["case"] and c["ref"] == row["ref"] for c in rows):
            raise ValueError("exposed LIVE case lacks a same-source control")
    return [{k: r[k] for k in ("index", "case", "ref", "dist", "human")} for r in rows]


def live_plan(config, output):
    value = json.loads(config.read_text())
    exposed = json.loads(LIVE_EXPOSED.read_text())
    rows = live_rows(exposed)
    models, inputs = {}, [LIVE_EXPOSED, config, Path(value["binary"])]
    for name, path in value["models"].items():
        if not re.fullmatch(r"[a-z0-9]+", name):
            raise ValueError("safe distinct LIVE model names required")
        models[name] = dict(path=str(Path(path).resolve()), sha256=sha(path))
        inputs.append(Path(path))
    if not models:
        raise ValueError("LIVE diagnostic models required")
    inputs += [Path(r[k]) for r in rows for k in ("ref", "dist")]
    output.mkdir()
    write(output / "PLAN.json", dict(schema="fx2-exposed-live-v1", rows=rows, models=models,
          exposed=str(LIVE_EXPOSED), binary=str(Path(value["binary"]).resolve()), source_sha256=source_pins(),
          input_sha256={str(p.resolve()): sha(p) for p in inputs}, new_label_read=False,
          diagnostic_only=True, selection_gate=False), fresh=True)


def live_run(path, output):
    import math
    from concurrent.futures import ThreadPoolExecutor
    if not compute_lock(os.getpid())["held_by_worker_ancestor"]:
        raise ValueError("LIVE pixel scoring requires the shared compute lock")
    value = json.loads(path.read_text())
    if value["schema"] != "fx2-exposed-live-v1" or value["source_sha256"] != source_pins():
        raise ValueError("frozen LIVE diagnostic source differs")
    if any(sha(p) != digest for p, digest in value["input_sha256"].items()):
        raise ValueError("frozen LIVE diagnostic input changed")
    rows = live_rows(dict(rows=value["rows"], rows_read=len(value["rows"])))
    output.mkdir()
    env = dict(os.environ, ZENSIM_FORMULA_REV="7", ZENSIM_MAX_TIER="v3", ZENSIM_RING_FORM="raw",
               ZENSIM_PYRAMID="xyb", RAYON_NUM_THREADS="1", OMP_NUM_THREADS="1", TMPDIR=str(Path.home() / "tmp"))
    def score(item):
        name, model, row = item
        result = subprocess.run([value["binary"], "--bake", model["path"], "--bake-post", "raw",
            "--ref", row["ref"], "--dist", row["dist"]], capture_output=True, text=True, env=env)
        with (output / f"{name}-{row['index']}.log").open("x") as log:
            log.write(result.stdout + result.stderr)
        result.check_returncode()
        number = float(result.stdout.strip().splitlines()[-1])
        if not math.isfinite(number):
            raise ValueError("nonfinite exposed LIVE score")
        return number
    scores = {}
    with ThreadPoolExecutor(max_workers=12) as pool:
        for name, model in value["models"].items():
            scores[name] = list(pool.map(score, [(name, model, row) for row in rows]))
    results = [dict(row, **{name: score[i] for name, score in scores.items()}) for i, row in enumerate(rows)]
    inversions = {str(case["index"]): {name: sum(not c["case"] and c["ref"] == case["ref"]
        and c["human"] > case["human"] and case[name] > c[name] for c in results) for name in scores}
        for case in results if case["case"]}
    write(output / "LIVE.json", dict(rows=results, severe_above_less_damaged_controls=inversions,
          plan_sha256=sha(path), diagnostic_only=True, selection_gate=False), fresh=True)
    write(output / "EVIDENCE_INVENTORY.json", {p.name: sha(p) for p in sorted(output.iterdir())}, fresh=True)
    write(output / "COMPLETE.json", dict(plan_sha256=sha(path),
          evidence_inventory_sha256=sha(output / "EVIDENCE_INVENTORY.json")), fresh=True)


def seed_average(rows):
    import numpy as np
    if len(rows) != 3 or {r["seed"] for r in rows} != {0, 1, 2}:
        raise ValueError("three distinct fixed full-data seeds required")
    values = [np.asarray(r["values"], dtype=float) for r in sorted(rows, key=lambda r: r["seed"])]
    if any(v.ndim != 1 or v.shape != values[0].shape or not np.isfinite(v).all() for v in values):
        raise ValueError("complete aligned finite full-data predictions required")
    return np.mean(values, axis=0)


def seed_delta(candidate, baseline):
    import numpy as np
    if set(candidate) != {0, 1, 2} or set(baseline) != {0, 1, 2}:
        raise ValueError("three paired seed statistics required")
    values = [candidate[s] - baseline[s] for s in range(3)]
    summary = complete_mean(values)
    return dict(per_seed_delta=values, mean=summary["value"],
                seed_counts={k: v for k, v in summary.items() if k != "value"},
                seed_unit_se=float(np.std(values, ddof=1) / np.sqrt(3)) if summary["defined"] == 3 else None,
                selection_gate=False)


def finished(root, pin, recipes=RECIPES):
    if recipes not in (RECIPES, ("Mlic", "MlicConfidence")):
        raise ValueError("registered full panel recipes required")
    complete = root / "FULL_FITS_COMPLETE.json"
    if complete.exists():
        value = json.loads(complete.read_text())
        if (value["plan_sha256"] != pin or set(value["initial"]) != set(recipes)
                or any(len(v) != 3 for v in value["initial"].values()) or len(value["continued"]) != 6 * len(recipes)):
            raise ValueError("full-fit completion population differs")
        return True
    state = json.loads((root / "FULL_STATE.json").read_text())
    if state["failures"]:
        raise RuntimeError("preceding full fitting failed")
    worker = state["worker"]
    if not bound_worker(worker["pid"], worker["start_ticks"]):
        raise RuntimeError("full fitter exited without completion")
    return False


def source_pins():
    return {str(p): sha(p) for p in (Path(__file__), Path(__file__).with_name("aic_adaptation_panel.py"),
            Path(__file__).with_name("aic_adaptation_diagnostics.py"), Path(__file__).with_name("aic_adaptation.py"),
            Path(__file__).with_name("aic_adaptation_full.py"))}


def queue(path, output):
    root, pin, pins = path.parent, sha(path), source_pins()
    recipes = full_recipes(json.loads(path.read_text()))
    state = dict(stage="waiting-for-full-fits", worker=process_snapshot(os.getpid()), plan_sha256=pin,
                 source_sha256=pins, failures=[], output=str(output), checked_utc=datetime.now(timezone.utc).isoformat())
    write(root / "FULL_PANEL_QUEUE.json", state, fresh=True)
    try:
        while not finished(root, pin, recipes):
            time.sleep(30)
        if sha(path) != pin or any(sha(p) != digest for p, digest in pins.items()):
            raise ValueError("queued full panel changed")
        command = ["flock", str(Path.home() / "tmp/heavy.lock"), str(Path.home() / "work/claudehints/scripts/run-heavy"),
            "--mem", "24G", "--jobs", "4", "--", sys.executable, "-B", str(Path(__file__).resolve()), "run", str(path), str(output)]
        state.update(stage="queued-for-full-panels", command=command)
        write(root / "FULL_PANEL_QUEUE.json", state)
        with (root / "full-panel-launch.log").open("x") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    except BaseException as error:
        state.update(stage="failed", failures=[repr(error)], checked_utc=datetime.now(timezone.utc).isoformat())
        write(root / "FULL_PANEL_QUEUE.json", state)
        raise


def status(path, output):
    state_path = output / "STATE.json"
    value = json.loads((state_path if state_path.exists() else path.parent / "FULL_PANEL_QUEUE.json").read_text())
    worker = value["worker"]
    live = bound_worker(worker["pid"], worker["start_ticks"])
    if not live and value["stage"] != "full-panels-complete" and not value["failures"]:
        value["failures"].append("recorded full-panel process exited without completion")
    value.update(checked_utc=datetime.now(timezone.utc).isoformat(), compute_lock=compute_lock(worker["pid"] if live else None))
    write(path.parent / "CURRENT_FULL_PANEL.json", value)
    print(json.dumps({k: value.get(k) for k in ("stage", "completed_predictions", "required_predictions", "failures", "compute_lock")}))


def full_populations(fold, output, purpose):
    """The five ordered in-sample TRAIN reporting populations: D6 public_train plus the E21 sets.

    Callers have already put the fold's frozen warm-start source on sys.path.
    Returns (AIC pair metadata, {population: table/target/metadata}).
    """
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    import aic_panel as legacy
    from subjective_training_leg import log_dataset_read, read_train_panel
    from v2_human_role import D6_INPUT_PINS
    pairs_path, table = Path(fold["aic_pairs"]), Path(fold["aic_source_table"])
    if sha(pairs_path) != D6_INPUT_PINS["pairs_sha256"] or sha(table) != fold["aic_source_table_sha256"]:
        raise ValueError("D6 public_train extraction differs")
    log_dataset_read(output, pairs_path, 3920, ["key", "group", "codec", "quality", "jnd"],
                     purpose, table_sha256=sha(pairs_path))
    pairs = pd.read_csv(pairs_path, sep="\t")
    keys = pq.read_table(table, columns=["key", "ref_basename"])
    if (len(pairs) != 3920 or pairs.group.nunique() != 28 or pairs.key.duplicated().any()
            or keys["key"].to_pylist() != pairs.key.tolist() or keys["ref_basename"].to_pylist() != pairs.group.tolist()):
        raise ValueError("ordered full AIC population differs")
    populations = {"aic": dict(table=table, target=-pairs.jnd.to_numpy(dtype=float), metadata=pairs)}
    e21 = {d["name"]: d for d in read_train_panel(legacy.E34)["datasets"]}
    for name in FOLDS:
        d = e21[name]
        populations[name] = dict(table=Path(fold["common_root"]) / f"wide/main/real/{name}.parquet",
                                 target=np.asarray(d["target"]) * float(d["orientation"]),
                                 metadata=pd.DataFrame(dict(group=d["reference"], codec=d["codec"])))
    return pairs, populations


def pixel_mixtures(fold, source, output, models, save):
    """Local and global pixel-mixture checks of {label: bake} through the fold's frozen scorer.

    Callers have already put the fold's frozen warm-start source on sys.path.
    """
    import mixture_test
    from v2_common import dense_bake
    for name in ("mixture", "mixture_global"):
        original = Path(fold["initial_cells"]["Mlic"]).parents[1] / name
        dest = output / name
        dest.mkdir()
        (dest / "manifest.json").write_bytes((original / "manifest.json").read_bytes())
        (dest / "img").symlink_to((original / "img").resolve(), target_is_directory=True)
        images = {r[k] for r in json.loads((dest / "manifest.json").read_text())["rows"] for k in ("ref", "base", "more")}
        pins = {image: sha(dest / "img" / image) for image in sorted(images)}
        for label, bake in models.items():
            cache = dest / "models" / label
            cache.mkdir(parents=True)
            dense = dense_bake(bake, cache)
            with (dest / f"{label}.log").open("x") as log:
                subprocess.run([sys.executable, "-B", str(source / "scripts/byconstruction/mixture_test.py"), "score",
                    "--out", str(dest), "--bake", f"{label}={dense}", "--bin-dir", fold["bin"], "--jobs", "3"],
                    check=True, stdout=log, stderr=subprocess.STDOUT)
        if any(sha(dest / "img" / image) != pin for image, pin in pins.items()):
            raise ValueError("full-panel mixture image changed")
        save(f"{name}.json", dict(per_cell=mixture_test.report(dest), images_sha256=pins,
            scorer_sha256=sha(Path(fold["bin"]) / "score_pair_with_bake")))


def compression_mask(dataset, types):
    """The dashboard's existing compression cohort, without a second type definition."""
    import numpy as np
    from aic_dashboard_data import CODEC_TYPES
    values = np.asarray(types)
    return np.ones(len(values), bool) if CODEC_TYPES[dataset] is None else np.isin(values, CODEC_TYPES[dataset])


def e21_group(group):
    """Bind existing completed full/omission predictions to their actual model receipts."""
    from aic_adaptation import initial_cell
    directory, kind = Path(group["run"]), group["kind"]
    inputs, records = {}, []
    if kind not in ("full", "omission", "initial"):
        raise ValueError("E21 group must be full, omission or initial")
    if kind != "initial":
        plan_file, complete_file = directory / "PLAN.json", directory / "COMPLETE.json"
        plan, complete = json.loads(plan_file.read_text()), json.loads(complete_file.read_text())
        if complete["plan_sha256"] != sha(plan_file):
            raise ValueError("source fit completion differs from its plan")
        inventory = directory / "EVIDENCE_INVENTORY.json"
        if sha(inventory) != complete["evidence_inventory_sha256"]:
            raise ValueError("source fit inventory differs")
        source_inventory = json.loads(inventory.read_text())
        fits_file = directory / "FITS_COMPLETE.json"
        if sha(fits_file) != source_inventory["FITS_COMPLETE.json"]:
            raise ValueError("completed source fit records changed")
        fits = json.loads(fits_file.read_text())
        if fits["plan_sha256"] != sha(plan_file):
            raise ValueError("source fit records differ from the plan")
        inputs.update({str(p): sha(p) for p in (plan_file, complete_file, inventory, fits_file)})
        if kind == "full" and plan["schema"] != "aic-share-full-plan-v1":
            raise ValueError("full-data E21 group requires its full-data plan")
        if kind == "omission" and plan["schema"] != "aic-share-omission-plan-v1":
            raise ValueError("omission E21 group requires its omission plan")
    for seed in (0, 1, 2):
        for dataset in FOLDS:
            if kind == "initial":
                cell = directory / f"without_{dataset}_s{seed}"
                result = json.loads((cell / "result.json").read_text())
                row = initial_cell(cell, dataset, seed, spec=result["spec"])
                inputs.update(row["input_sha256"])
                model, prediction = cell / "refit/last.bin", cell / f"pred_{dataset}.tsv"
                scope = "out of sample"
            elif kind == "full":
                label = group["model"]
                if label == "candidate":
                    (arm, _), = plan["arms"].items()
                    epoch = plan["packed_epoch"]
                    label = f"{arm}e{epoch}"
                    rows = [r for r in fits["cells"] if r["seed"] == seed]
                    if len(rows) != 1:
                        raise ValueError("one completed full fit per seed required")
                    model = directory / "full" / arm / f"full_s{seed}/pack_e{epoch:03d}/refit/production-f16.bin"
                    expected = rows[0]["packed"]["packed_model_sha256"]
                else:
                    baseline = plan["baselines"][f"{label}|s{seed}"]
                    model, expected = Path(baseline["path"]), baseline["sha256"]
                if sha(model) != expected:
                    raise ValueError("packed full model changed")
                prediction = directory / "panels/predictions" / f"{label}|s{seed}|{dataset}.tsv"
                scope = "in sample"
            else:
                (arm, _), = plan["arms"].items()
                epoch = group["epoch"]
                if epoch not in plan["checkpoints"]:
                    raise ValueError("omission report must use a registered checkpoint")
                rows = [r for r in fits["cells"] if r["seed"] == seed and r["fold"] == dataset]
                if len(rows) != 1:
                    raise ValueError("one completed omission fit per dataset and seed required")
                model = directory / "cells" / arm / f"without_{dataset}_s{seed}/ckpt/ckpt_epoch{epoch:03d}.bin"
                if sha(model) != rows[0]["checkpoints"][str(epoch)]:
                    raise ValueError("omission checkpoint changed")
                prediction = directory / "cells" / f"{arm}e{epoch}" / f"without_{dataset}_s{seed}/pred_{dataset}.tsv"
                scope = "out of sample"
            if kind != "initial" and sha(prediction) != source_inventory[str(prediction.relative_to(directory))]:
                raise ValueError("completed source prediction changed")
            inputs.update({str(model): sha(model), str(prediction): sha(prediction)})
            records.append(dict(group=group["name"], seed=seed, dataset=dataset, scope=scope,
                                model=str(model), prediction=str(prediction)))
    return records, inputs


def e21_plan(config_path, output):
    """Freeze the E21-only report's exact model/prediction/metadata/native-panel inputs."""
    import aic_panel as legacy
    if output.exists():
        raise ValueError("preserve existing E21 report attempts")
    config = json.loads(config_path.read_text())
    groups = config["groups"]
    names = [g["name"] for g in groups]
    if (len(set(names)) != len(names) or not groups
            or any(not re.fullmatch(r"[A-Za-z0-9_-]+", n) for n in names)):
        raise ValueError("distinct nonempty report groups required")
    records, inputs = [], {str(config_path): sha(config_path)}
    for group in groups:
        rows, pins = e21_group(group)
        records += rows
        inputs.update(pins)
    for candidate, baseline in config["contrasts"]:
        if candidate not in names or baseline not in names:
            raise ValueError("contrast group absent")
        scopes = {r["scope"] for r in records if r["group"] in (candidate, baseline)}
        if len(scopes) != 1:
            raise ValueError("in-sample and out-of-sample groups cannot form a paired contrast")
    source = Path(__file__).resolve().parent
    files = [legacy.E34, legacy.DVIFM / "rows.parquet", legacy.DVIFM / "dvifm.tsv",
             source / "type_bias.py", source / "aic_dashboard_data.py", source / "aic_panel.py",
             Path(config["panel_binary"]), *map(Path, source_pins())]
    inputs.update({str(p): sha(p) for p in files})
    output.mkdir(parents=True)
    write(output / "PLAN.json", dict(schema="aic-e21-subset-report-plan-v1", records=records,
          contrasts=config["contrasts"], groups=groups, panel_binary=config["panel_binary"],
          metadata=str(legacy.E34), input_sha256=inputs, source_sha256=source_pins(),
          public_test_opened=False, eval_read=False), fresh=True)


def e21_run(path, output):
    """All-type/compression-only E21 panels through native zenstats, plus the existing type-bias owner."""
    import numpy as np
    import pandas as pd
    import aic_panel as legacy
    from subjective_training_leg import read_train_panel
    from lib.zen_stats import panel_batch
    if not compute_lock(os.getpid())["held_by_worker_ancestor"]:
        raise ValueError("E21 panels require the shared compute lock")
    value = json.loads(path.read_text())
    if value["schema"] != "aic-e21-subset-report-plan-v1" or value["source_sha256"] != source_pins():
        raise ValueError("E21 report owner changed")
    for name, pin in value["input_sha256"].items():
        if sha(name) != pin:
            raise ValueError(f"E21 report input changed: {name}")
    output.mkdir()
    os.environ.update(BYC_LABEL_LEDGER_DIR=str(output / "label-reads"), ZEN_PANEL_BIN=value["panel_binary"],
                      OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", RAYON_NUM_THREADS="1", TMPDIR=str(Path.home() / "tmp"))
    state = dict(stage="E21-subset-panels", worker=process_snapshot(os.getpid()), failures=[], plan_sha256=sha(path))
    write(output / "STATE.json", state, fresh=True)
    try:
        metadata = {d["name"]: d for d in read_train_panel(Path(value["metadata"]))["datasets"]}
        if set(metadata) != set(FOLDS):
            raise ValueError("E21 metadata population differs")
        _, types = legacy.dvifm_e21()
        jobs, rows = [], []
        aliases = output / "cells"
        for record in value["records"]:
            group, seed, dataset = record["group"], record["seed"], record["dataset"]
            d = metadata[dataset]
            if group in d["predictions"]:
                raise ValueError("report group collides with a cached metadata prediction")
            target = np.asarray(d["target"], float) * float(d["orientation"])
            frame = pd.read_csv(record["prediction"], sep="\t")
            prediction = frame.pred.to_numpy(float)
            if (list(frame.columns) != ["row_idx", "pred"] or len(prediction) != len(target)
                    or not np.array_equal(frame.row_idx, np.arange(len(target))) or not np.isfinite(prediction).all()
                    or not np.isfinite(target).all() or len(types[dataset]) != len(target)):
                raise ValueError("E21 prediction order/coverage differs")
            link = aliases / group / f"without_{dataset}_s{seed}" / f"pred_{dataset}.tsv"
            link.parent.mkdir(parents=True)
            link.symlink_to(Path(record["prediction"]).resolve())
            for subset, mask in (("all", np.ones(len(target), bool)), ("compression", compression_mask(dataset, types[dataset]))):
                if not mask.any():
                    raise ValueError("registered E21 subset is empty")
                jobs.append((f"{group}|s{seed}|{dataset}|{subset}", prediction[mask], target[mask]))
                rows.append(dict(group=group, seed=seed, dataset=dataset, subset=subset,
                                 scope=record["scope"], rows=int(mask.sum())))
        stats = panel_batch(jobs, stats="full")
        checked_statistics(jobs, stats)
        for row, result in zip(rows, stats, strict=True):
            row["statistics"] = result
        write(output / "MODEL_PANELS.json", finite_json(rows), fresh=True)
        by_model = {(r["group"], r["seed"], r["dataset"], r["subset"]): r["statistics"] for r in rows}
        paired = {}
        for candidate, baseline in value["contrasts"]:
            for dataset in FOLDS:
                for subset in ("all", "compression"):
                    paired[f"{candidate}-minus-{baseline}|{dataset}|{subset}"] = {m: seed_delta(
                        {s: by_model[candidate, s, dataset, subset][m] for s in (0, 1, 2)},
                        {s: by_model[baseline, s, dataset, subset][m] for s in (0, 1, 2)}) for m in METRICS}
        write(output / "PAIRED_SEEDS.json", finite_json(paired), fresh=True)
        with (output / "type-bias.log").open("x") as log:
            subprocess.run([sys.executable, "-B", str(Path(__file__).with_name("type_bias.py")),
                *[g["name"] for g in value["groups"]], "--out", str(output / "TYPE_BIAS.json")],
                env=dict(os.environ, BYC_CELLS=str(aliases)), stdout=log, stderr=subprocess.STDOUT, check=True)
        inventory = {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*"))
                     if p.is_file() and p.name != "STATE.json"}
        write(output / "EVIDENCE_INVENTORY.json", inventory, fresh=True)
        write(output / "COMPLETE.json", dict(plan_sha256=sha(path),
              evidence_inventory_sha256=sha(output / "EVIDENCE_INVENTORY.json"), eval_read=False), fresh=True)
        state.update(stage="E21-subset-panels-complete", checked_utc=datetime.now(timezone.utc).isoformat())
        write(output / "STATE.json", state)
    except BaseException as error:
        state.update(stage="failed", failures=[repr(error)])
        write(output / "STATE.json", state)
        raise


def run(path, output):
    import numpy as np
    import pandas as pd
    if not compute_lock(os.getpid())["held_by_worker_ancestor"]:
        raise ValueError("full panels require the shared compute lock")
    queued = json.loads((path.parent / "FULL_PANEL_QUEUE.json").read_text())
    if queued["plan_sha256"] != sha(path) or queued["source_sha256"] != source_pins():
        raise ValueError("full panel source or input identity changed")
    plan = json.loads(path.read_text())
    recipes = full_recipes(plan)
    require_clean_factors(os.environ)
    fold = json.loads(Path(plan["fold_plan"]).read_text())
    if sha(plan["fold_plan"]) != plan["fold_plan_sha256"] or not finished(path.parent, sha(path), recipes):
        raise ValueError("full-fit prerequisite changed")
    for name, pin in plan["input_sha256"].items():
        if sha(name) != pin:
            raise ValueError(f"full-panel input changed: {name}")
    source = Path(fold["warmstart_source"])
    os.environ.update(TMPDIR=str(Path.home() / "tmp"), REV4_V2_BIN_DIR=fold["bin"], AIC_BIN=fold["bin"],
                      BYC_ROOT=fold["common_root"], ZEN_PANEL_BIN=str(Path(fold["bin"]) / "panel"),
                      BYC_LABEL_LEDGER_DIR=str(output / "label-reads"), RAYON_NUM_THREADS="1", OMP_NUM_THREADS="1",
                      OPENBLAS_NUM_THREADS="1", ZENSIM_MAX_TIER="v3", ZENSIM_RING_FORM="raw", ZENSIM_PYRAMID="xyb")
    sys.path[:0] = [str(source / "scripts/rev4_featpot"), str(source / "scripts/byconstruction"), str(source / "scripts")]
    import byc_cells as cells
    import aic_panel as legacy
    from lib.zen_stats import logistic_fit, panel_batch, panel_batch_indexed
    output.mkdir()
    state = dict(stage="bind-full-evidence", worker=process_snapshot(os.getpid()), source_sha256=source_pins(),
                 completed_predictions=0, required_predictions=90 * len(recipes), failures=[], objective_complete=False)
    def stage(name, **details):
        state.update(stage=name, checked_utc=datetime.now(timezone.utc).isoformat(), **details)
        write(output / "STATE.json", state)
    def save(name, value):
        write(output / name, finite_json(value), fresh=True)
    try:
        stage("bind-full-evidence")
        fits = json.loads((path.parent / "FULL_FITS_COMPLETE.json").read_text())
        records = [dict(r, arm=recipe) for recipe, rows in fits["initial"].items() for r in rows] + fits["continued"]
        expected = {(arm, s) for recipe in recipes for arm in [recipe, *(f"{recipe}FT{w}x{e}" for w, e in STRENGTHS)] for s in range(3)}
        if len(records) != 9 * len(recipes) or {(r["arm"], r["seed"]) for r in records} != expected:
            raise ValueError("all full recipe/strength/seed cells required")
        for row in records:
            directory = Path(row["directory"])
            if (sha(directory / "refit/last.bin") != row["checkpoint_sha256"]
                    or sha(directory / "refit/production-f16.bin") != row["packed_sha256"]
                    or any(sha(p) != pin for p, pin in row["input_sha256"].items())):
                raise ValueError("completed full model evidence changed")
        pairs, populations = full_populations(fold, output, "registered-full-AIC-in-sample-panels")
        save("POPULATIONS.json", {k: dict(table=str(v["table"]), sha256=sha(v["table"]), rows=len(v["target"]),
             exposure="all sources used in continuation" if k == "aic" else "full fit, including original development rows")
             for k, v in populations.items()})
        (output / "predictions").mkdir()
        all_values, descriptions, jobs = {}, [], []
        for row in records:
            for representation, filename in REPRESENTATIONS.items():
                model = Path(row["directory"]) / "refit" / filename
                for name, population in populations.items():
                    label = f"{row['arm']}|s{row['seed']}|{representation}|{name}"
                    stage("predict-full-populations", current_model=label)
                    destination = output / "predictions" / f"{label}.tsv"
                    cells.predict_cell(model, population["table"], destination, SPEC)
                    frame = pd.read_csv(destination, sep="\t")
                    values = frame.pred.to_numpy(dtype=float)
                    target = population["target"]
                    if (list(frame.columns) != ["row_idx", "pred"] or len(values) != len(target)
                            or not np.array_equal(frame.row_idx, np.arange(len(target))) or not np.isfinite(values).all()):
                        raise ValueError("ordered finite full prediction coverage differs")
                    all_values.setdefault((row["arm"], representation, name), []).append(dict(seed=row["seed"], values=values))
                    jobs.append((label, values, target))
                    description = dict(arm=row["arm"], seed=row["seed"], representation=representation, population=name,
                        label=label, prediction_sha256=sha(destination), geometry=legacy.geometry(values, target),
                        sample="zero-shot initialization" if name == "aic" and row["arm"] in recipes else "in sample")
                    if name == "aic":
                        description["ladder"] = ladder_diagnostics(pairs, values, target, np.ones(len(target), bool), logistic_fit)
                    descriptions.append(description)
                    state["completed_predictions"] += 1
        stage("full-native-panels")
        stats = panel_batch(jobs, stats="full")
        checked_statistics(jobs, stats)
        for row, value in zip(descriptions, stats, strict=True):
            row["statistics"] = value
        save("CELL_PANELS.json", descriptions)
        aggregates = {key: seed_average(rows) for key, rows in all_values.items()}
        aggregate_jobs, aggregate_rows = [], []
        for (arm, representation, name), values in aggregates.items():
            population = populations[name]
            target, metadata = population["target"], population["metadata"]
            label = f"{arm}|{representation}|{name}"
            pd.DataFrame(dict(row_idx=np.arange(len(values)), pred=values, contributing_models=3)).to_csv(
                output / "predictions" / f"mean-{label}.tsv", sep="\t", index=False)
            groups = [("all", "ALL", np.arange(len(values)))]
            for category, columns in (("source", ["group"]), ("codec", ["codec"]), ("ladder", ["group", "codec"])):
                for key, index in metadata.groupby(columns, sort=True).indices.items():
                    groups.append((category, repr(key), np.asarray(index, dtype=int)))
            for category, group, index in groups:
                key = f"{label}|{category}|{group}"
                aggregate_jobs.append((key, values[index], target[index]))
                aggregate_rows.append(dict(arm=arm, representation=representation, population=name, category=category,
                    group=group, label=key, geometry=legacy.geometry(values[index], target[index]) if category in ("all", "codec") else None))
            save(f"bias-{label}.json", cells.codec_rank_bias(values, target, metadata.codec.to_numpy()))
        stage("full-source-codec-ladder-panels")
        stats = panel_batch(aggregate_jobs, stats="full")
        checked_statistics(aggregate_jobs, stats)
        for row, value in zip(aggregate_rows, stats, strict=True):
            row["statistics"] = value
        save("AGGREGATE_PANELS.json", dict(panels=aggregate_rows, aggregation="three fixed seeds' mean predictions",
              serving_model="fixed seed zero; the three-seed mean is a stability diagnostic, not the serving model"))
        contrasts = [(f"{recipe}FT{w}x{e}", recipe) for recipe in recipes for w, e in STRENGTHS]
        contrasts += [(f"{recipe}FT{w}x{e}", f"MlicFT{w}x{e}") for recipe in recipes[1:] for w, e in STRENGTHS]
        paired = {}
        for arm, control in contrasts:
            for representation in REPRESENTATIONS:
                for name in populations:
                    selected = [r for r in descriptions if r["representation"] == representation and r["population"] == name]
                    paired[f"{arm}-minus-{control}|{representation}|{name}"] = {
                        metric: seed_delta({r["seed"]: r["statistics"][metric] for r in selected if r["arm"] == arm},
                                           {r["seed"]: r["statistics"][metric] for r in selected if r["arm"] == control})
                        for metric in METRICS}
        save("PAIRED_SEEDS.json", paired)
        # This contrast includes quantization AND the existing TRAIN output dial.
        # It cannot identify a pure f16 arithmetic effect.
        deployment = {}
        for arm in {r["arm"] for r in records}:
            for name in populations:
                rows = [r for r in descriptions if r["arm"] == arm and r["population"] == name]
                deployment[f"{arm}|{name}"] = {metric: seed_delta(
                    {r["seed"]: r["statistics"][metric] for r in rows if r["representation"] == "production-f16-calibrated"},
                    {r["seed"]: r["statistics"][metric] for r in rows if r["representation"] == "final-f32"}) for metric in METRICS}
        save("DEPLOYMENT_EFFECT.json", dict(contrasts=deployment,
              interpretation="joint f16 packing and TRAIN output calibration; not isolated precision loss"))
        rng = np.random.default_rng(1)
        clusters = [np.asarray(index, int) for index in pairs.groupby("group", sort=True).indices.values()]
        draws = [np.concatenate([clusters[i] for i in rng.integers(0, 28, 28)]) for _ in range(2000)]
        intervals = {}
        point = {(r["arm"], r["representation"]): r["statistics"]["srocc_signed"] for r in aggregate_rows
                 if r["population"] == "aic" and r["category"] == "all"}
        for arm, control in contrasts:
            for representation in REPRESENTATIONS:
                label = f"{arm}-minus-{control}|{representation}"
                stage("full-AIC-source-intervals", current_model=label)
                bases = dict(target=populations["aic"]["target"], candidate=aggregates[arm, representation, "aic"],
                             baseline=aggregates[control, representation, "aic"])
                tasks = [(f"{i}-{name}", name, "target", index) for i, index in enumerate(draws) for name in ("baseline", "candidate")]
                values = panel_batch_indexed(bases, tasks, stats="srocc")
                checked_statistics([(label, index, index) for label, _, _, index in tasks], values)
                delta = np.asarray([values[i+1]["srocc_signed"] - values[i]["srocc_signed"] for i in range(0, len(values), 2)])
                good = delta[np.isfinite(delta)]
                intervals[label] = dict(metric="srocc_signed", aggregation="mean predictions over three fixed full-data seeds",
                    point_delta=point[arm, representation] - point[control, representation], resamples=2000, source_clusters=28,
                    seed=1, defined_resamples=len(good), undefined_resamples=2000-len(good), quantile_method="linear",
                    interval_95=np.quantile(good, [.025, .975], method="linear").tolist() if len(good) else None,
                    uncertainty="whole-source resampling conditional on fitted models; all AIC sources are in sample",
                    descriptive_only=True, selection_gate=False)
        save("SOURCE_INTERVALS.json", intervals)
        stage("full-model-mixtures")
        pixel_mixtures(fold, source, output, {f"{row['arm']}_s{row['seed']}_{representation}":
            Path(row["directory"]) / "refit" / filename for row in records
            for representation, filename in REPRESENTATIONS.items()}, save)
        inventory = {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*")) if p.is_file() and p.name != "STATE.json"}
        save("EVIDENCE_INVENTORY.json", dict(files_sha256=inventory, source_sha256=source_pins(),
             native_tools_sha256={p.name: sha(p) for p in Path(fold["bin"]).iterdir() if p.is_file()}))
        save("FULL_PANEL_COMPLETE.json", dict(plan_sha256=sha(path), fit_completion_sha256=sha(path.parent / "FULL_FITS_COMPLETE.json"),
             evidence_inventory_sha256=sha(output / "EVIDENCE_INVENTORY.json"),
             fold_panel=str(Path(plan["fold_panel"]) / "PANEL_COMPLETE.json"),
             fold_panel_sha256=sha(Path(plan["fold_panel"]) / "PANEL_COMPLETE.json"),
             selector=None, assessment_seed=0, production_qualified=False, objective_complete=False))
        stage("full-panels-complete", next_action="compare full tradeoffs; serving runtime and steering qualification remain")
    except BaseException as error:
        stage("failed", failures=[repr(error)])
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("queue", "run", "status", "e21-plan", "e21-run", "live-plan", "live-run"))
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    globals()[args.mode.replace("-", "_")](args.plan.resolve(), args.output.resolve())
