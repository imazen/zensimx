"""Matched AIC TRAIN continuation across frozen starting recipes.

The October 27 panel study has no SROCC selector. This owner binds inputs,
supervises fitting and prediction, and preserves the earlier studies unchanged.
Native work runs only under the canonical heavy lock and resource scope.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/rev4_featpot"))
from run_processes import bound_worker, compute_lock, process_snapshot, process_tree

FOLDS = ("kadid", "tid2013", "konfig", "cid22_a25")
RECIPES = ("Mlic", "MlicNeff", "MlicVote", "MlicOrder")
STRENGTHS = ((8, 4), (16, 8))
SPEC = "sel:3b7bd5ebe929@h32:H128:cv16:cf98:fx1:mono:thr:mp32"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value, *, fresh=False):
    if fresh:
        with path.open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
    else:
        temporary = path.with_suffix(path.suffix + ".new")
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)


def checked_plan(path):
    plan = json.loads(path.read_text())
    if (plan["schema"] != "aic-recipe-adaptation-plan-v1"
            or plan["recipes"] != list(RECIPES)
            or plan["selection"] != "full-panel-no-srocc-cutoff"
            or plan["target_role"] != "D6-public_train-only"
            or plan["coordinator_sha256"] != sha(__file__)):
        raise ValueError("registered panel-study contract differs")
    for name, pin in plan["input_sha256"].items():
        if sha(name) != pin:
            raise ValueError(f"frozen adaptation input changed: {name}")
    return plan


def admission_finished(root):
    state = json.loads((root / "ADMISSION_STATE.json").read_text())
    if state.get("failure") or state["stage"] == "failed":
        raise RuntimeError("preceding admission failed; inspect its preserved failure")
    if state["stage"] == "screen-complete-requires-membership-freeze":
        if sha(root / "screen/screen-summary.json") != state["screen_summary_sha256"]:
            raise ValueError("preceding admission completion changed")
        return True
    worker = state.get("worker", state["queue_process"])
    if not bound_worker(worker["pid"], worker["start_ticks"]):
        raise RuntimeError("preceding admission exited without completion")
    return False


def update(root, state, stage, **details):
    state.update(stage=stage, checked_utc=datetime.now(timezone.utc).isoformat(), **details)
    write(root / "RUN_STATE.json", state)


def queue(path):
    plan, root = checked_plan(path), path.parent
    state = dict(plan_sha256=sha(path), queue_process=process_snapshot(os.getpid()),
                 run_directory=str(root), completed=[], required=96, failures=[],
                 source_commit=plan["coordinator_commit"], binary_sha256=plan["trainer_sha256"],
                 objective_complete=False, next_action="wait for source screening, then bind all starting checkpoints")
    if (root / "RUN_STATE.json").exists():
        raise ValueError("preserve existing adaptation attempts")
    update(root, state, "waiting-for-admission")
    try:
        while not admission_finished(Path(plan["after_admission"])):
            time.sleep(30)
        checked_plan(path)
        cmd = ["flock", str(Path.home() / "tmp/heavy.lock"),
               str(Path.home() / "work/claudehints/scripts/run-heavy"), "--mem", "24G", "--jobs", "4", "--",
               sys.executable, "-B", str(Path(__file__).resolve()), "run", str(path)]
        update(root, state, "queued-for-fitting", command=cmd)
        with (root / "launch.log").open("x") as log:
            subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
    except BaseException as error:
        state = json.loads((root / "RUN_STATE.json").read_text())
        update(root, state, "failed", failures=[*state["failures"], repr(error)])
        raise


def initial_cell(directory, fold, seed):
    """Bind actual model bytes to their existing fit and prediction receipts."""
    result = json.loads((directory / "result.json").read_text())
    record = json.loads((directory / "byc_record.json").read_text())
    model, prediction = directory / "refit/last.bin", directory / f"pred_{fold}.tsv"
    if (result["training_only"] is not True or result["execution_contract"] != "registered-fit"
            or result["epochs"] != 120 or result["pairs_per_epoch"] != 50000
            or result["seed_index"] != seed or result["heldout"] != fold or result["spec"] != SPEC
            or result["head"] != "N" or result["research_extra_groups"] is not None
            or result["width"] != 1853 or result["kept_features"] != 410
            or result["selection"]["selected_epoch"] != 119
            or result["selected_bake_sha256"] != sha(model)
            or record["last_bin_sha256"] != sha(model) or record["pred_sha256"] != sha(prediction)
            or record["fold"] != fold or record["seed"] != seed):
        raise ValueError(f"starting checkpoint contract differs: {directory}")
    files = ("result.json", "byc_record.json", "DATASET_GROUP_SELECTION.json", "keep_features.txt",
             "refit/last.bin", f"pred_{fold}.tsv")
    return dict(directory=str(directory), fold=fold, seed=seed,
                input_sha256={str(directory / f): sha(directory / f) for f in files})


def bind_recipe_cell(recipe, row, study, selection, columns):
    directory = Path(row["directory"])
    matches = [r for r in selection["cells"] if (r["arm"], r["fold"], r["seed"]) == (recipe, row["fold"], row["seed"])]
    if len(matches) != 1 or matches[0]["model_sha256"] != sha(directory / "refit/last.bin"):
        raise ValueError("initial model does not belong to the declared recipe cell")
    original = study / "cells" / recipe / directory.name
    for name in ("result.json", "byc_record.json"):
        if selection["input_sha256"][str(original / name)] != sha(directory / name):
            raise ValueError("recipe cell receipt differs from its completed study")
    group = json.loads((directory / "DATASET_GROUP_SELECTION.json").read_text())
    result = json.loads((directory / "result.json").read_text())
    root = Path(group["root"])
    expected = selection["input_sha256"][str(study / recipe / "root/wide/frozen.json")]
    if (group["schema"] != "dataset-group-selection-v1" or group["root_sha256"] != expected
            or result["frozen_sha256"] != expected or sha(root / "wide/frozen.json") != expected
            or sha(root / "datasets/receipt.json") != group["receipt_sha256"]):
        raise ValueError("recipe root or dataset omission authority differs")
    receipt = json.loads((root / "datasets/receipt.json").read_text())
    all_collections = {r["collection"] for r in receipt["groups"]}
    retained, omitted = set(group["retained_collections"]), set(group["omitted_collections"])
    if retained & omitted or retained | omitted != all_collections:
        raise ValueError("recipe collection partition differs")
    if re.split(r"[,\s]+", (directory / "keep_features.txt").read_text().strip()) != re.split(r"[,\s]+", columns.strip()):
        raise ValueError("recipe feature binding differs")
    row["recipe_binding"] = dict(recipe=recipe, study=str(study),
        selection_sha256=sha(study / "TRAIN_SELECTION.json"), root_sha256=expected,
        dataset_receipt_sha256=group["receipt_sha256"])
    return row


def zero_checkpoint(directory):
    result = json.loads((directory / "result.json").read_text())
    model = Path(result["selected_bake"])
    if (result["epochs"] != 0 or result["selection"]["selected_epoch"] != -1
            or result["training_only"] is not True or result["spec"] != SPEC
            or not model.resolve().is_relative_to((directory / "refit").resolve())
            or sha(model) != result["selected_bake_sha256"]):
        raise ValueError("zero-step selected checkpoint receipt differs")
    return model


def require_prediction_parity(actual, reference):
    if actual.read_bytes() != reference.read_bytes():
        raise ValueError("zero-step predictions differ byte-for-byte")


def continuation_cell(directory, initial, fold, seed, weight, epochs, plan):
    result = json.loads((directory / "result.json").read_text())
    record = json.loads((directory / "byc_record.json").read_text())
    authority = json.loads((directory / "continuation-authority.json").read_text())
    continuation = authority["cells"][f"{fold}-s{seed}"]["continuation"]
    if (result["training_only"] is not True or result["epochs"] != epochs
            or result["pairs_per_epoch"] != 50000 or result["heldout"] != fold
            or result["seed_index"] != seed or result["spec"] != SPEC
            or result["frozen_sha256"] != plan["common_root_sha256"]
            or record["aic_fold"] != (4 * seed + FOLDS.index(fold)) % 7
            or record["trainer_sha256"] != plan["trainer_sha256"]
            or record["last_bin_sha256"] != sha(directory / "refit/last.bin")
            or result["selected_bake_sha256"] != record["last_bin_sha256"]
            or result["selection"]["selected_epoch"] != epochs - 1
            or result["selection"]["epoch_rule"] != "last"
            or record["pred_sha256"] != sha(directory / f"pred_{fold}.tsv")
            or record["aic_pred_sha256"] != sha(directory / f"pred_aic_k{record['aic_fold']}.tsv")
            or float(record["aic_weight"]) != weight
            or len(record["curve"]) != epochs
            or record["continuation"] != dict(initial_checkpoint=str(initial / "refit/last.bin"),
                                             epochs=epochs, lr=0.00001, lr_schedule="constant")):
        raise ValueError(f"continuation output differs: {directory}")
    # The native owner pins the initialization bytes, not only their pathname.
    if (authority["schema"] != "subjective-training-authority-v3"
            or continuation != dict(checkpoint=str(initial / "refit/last.bin"),
                checkpoint_sha256=sha(initial / "refit/last.bin"), epochs=epochs, lr=0.00001,
                lr_schedule="constant", optimizer_state="fresh")):
        raise ValueError("continuation initialization bytes differ")
    command = shlex.split((directory / "train.log").read_text().splitlines()[0])
    aic = [command[i + 1].split(":") for i, value in enumerate(command[:-1])
           if value == "--group" and command[i + 1].startswith("aic_public_train:")]
    k = record["aic_fold"]
    if (len(aic) != 1 or aic[0][1] != str(Path(plan["aic_folds"]) / f"aic_k{k}_fit.parquet")
            or float(aic[0][2]) != weight or aic[0][3:] != ["0", "rank"]):
        raise ValueError("continuation AIC mass differs")
    return dict(directory=str(directory), fold=fold, seed=seed, aic_fold=record["aic_fold"],
                epochs=epochs, weight=weight, checkpoint_sha256=record["last_bin_sha256"],
                result_sha256=sha(directory / "result.json"), record_sha256=sha(directory / "byc_record.json"),
                continuation_authority_sha256=sha(directory / "continuation-authority.json"),
                train_log_sha256=sha(directory / "train.log"))


def run(path):
    root = path.parent
    state = json.loads((root / "RUN_STATE.json").read_text())
    try:
        plan = checked_plan(path)
        if state["plan_sha256"] != sha(path) or not compute_lock(os.getpid())["held_by_worker_ancestor"]:
            raise ValueError("queued plan and shared compute lock required")
        state["worker"] = process_snapshot(os.getpid())
        update(root, state, "bind-starting-checkpoints")
        for study in plan["starting_studies"]:
            directory = Path(study)
            complete = json.loads((directory / "TRAIN_SCREEN_COMPLETE.json").read_text())
            if complete["selection_sha256"] != sha(directory / "TRAIN_SELECTION.json"):
                raise ValueError("starting study completion differs")
        inputs = {recipe: [initial_cell(Path(plan["initial_cells"][recipe]) / f"without_{fold}_s{seed}", fold, seed)
                          for seed in range(3) for fold in FOLDS] for recipe in RECIPES}
        for recipe in RECIPES:
            study = Path(plan["recipe_authority"][recipe])
            selection = json.loads((study / "TRAIN_SELECTION.json").read_text())
            if sha(study / "TRAIN_SELECTION.json") != json.loads((study / "TRAIN_SCREEN_COMPLETE.json").read_text())["selection_sha256"]:
                raise ValueError("recipe study completion differs")
            inputs[recipe] = [bind_recipe_cell(recipe, row, study, selection, Path(plan["columns"]).read_text())
                              for row in inputs[recipe]]
        write(root / "INITIAL_CHECKPOINTS.json", inputs, fresh=True)
        source = Path(plan["warmstart_source"])
        os.environ.update(BYC_ROOT=plan["common_root"], REV4_V2_ROOT=plan["common_root"],
            BYC_CELLS=str(root / "cells"), REV4_V2_BIN_DIR=plan["bin"], ZEN_PANEL_BIN=str(Path(plan["bin"]) / "panel"),
            BYC_COLS=plan["columns"], AIC_FOLDS=plan["aic_folds"], AIC_PAIRS_TRAIN=plan["aic_pairs"],
            AIC_PREDICT_FOLDS=plan["prediction_views"], BYC_SCRATCH=str(root / "scratch"),
            BYC_LABEL_LEDGER_DIR=str(root / "label-reads"), TMPDIR=str(Path.home() / "tmp"),
            OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", RAYON_NUM_THREADS="1", ZENSIM_MAX_TIER="v3",
            ZENSIM_RING_FORM="raw", ZENSIM_PYRAMID="xyb")
        sys.path[:0] = [str(source / "scripts/rev4_featpot"), str(source / "scripts")]
        import byc_cells as cells
        from subjective_training_leg import linked_subset_exclusions
        if cells.HERE.resolve() != source / "scripts/rev4_featpot":
            raise ValueError("warm-start owner resolved outside the frozen source")
        # Each different initial recipe must survive actual zero-step continuation.
        for recipe in RECIPES:
            update(root, state, "zero-step-control", current_recipe=recipe)
            original = Path(plan["initial_cells"][recipe]) / "without_kadid_s0"
            dest = root / "zero-step" / recipe
            dest.parent.mkdir(exist_ok=True)
            cmd = [sys.executable, "-B", str(cells.HERE / "v2_lodo_mlp.py"), "--root", plan["common_root"],
                "--spec", SPEC, "--columns", Path(plan["columns"]).read_text().strip(), "--head", "N",
                "--heldout", "kadid", "--seed-index", "0", "--strict-admission", "--data-role-decision",
                str(Path(plan["common_root"]) / "human_role_decision.json"), "--train-only", "--dest", str(dest),
                "--init-from", str(original / "refit/last.bin"), "--fine-tune-epochs", "0", "--fine-tune-lr", "0.00001",
                "--aic-public-train", str(Path(plan["aic_folds"]) / "aic_k0_fit.parquet"),
                "--research-extra-group", f"aic_public_train:{plan['aic_folds']}/aic_k0_fit.parquet:8.0:0:rank"]
            exclusions = linked_subset_exclusions(Path(plan["common_root"]), "kadid")
            if exclusions:
                cmd += ["--exclude-datasets", ",".join(exclusions)]
            with (dest.parent / f"{recipe}.log").open("x") as log:
                subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
            output = dest / "pred_kadid.tsv"
            selected = zero_checkpoint(dest)
            cells.predict_cell(selected, Path(plan["common_root"]) / "wide/main/real/kadid.parquet", output, SPEC)
            require_prediction_parity(output, original / "pred_kadid.tsv")
            write(dest / "CONTROL_VERIFIED.json", dict(byte_identical=True, rows=7869,
                  prediction_sha256=sha(output), initial_sha256=sha(original / "refit/last.bin"),
                  selected_checkpoint=str(selected), selected_sha256=sha(selected)), fresh=True)
        for recipe in RECIPES:
            for weight, epochs in STRENGTHS:
                label = f"{recipe}FT{weight}x{epochs}"
                cells.LABEL, cells.AIC_WEIGHT = label, float(weight)
                cells.FINE_TUNE = (Path(plan["initial_cells"][recipe]), epochs, 0.00001)
                update(root, state, "continuation-cells", current_recipe=recipe, current_arm=label,
                       next_action="complete all 96 matched cells, then fitted/held-out full panels")
                def fit(row):
                    fold, seed = row["fold"], row["seed"]
                    if any(sha(name) != pin for name, pin in row["input_sha256"].items()):
                        raise ValueError("starting checkpoint changed after freeze")
                    initial = Path(row["directory"])
                    if recipe == "Mlic":
                        directory = Path(plan["reuse_mlic"]) / "cells" / label / initial.name
                    else:
                        cells.run_cell("mono:thr:mp32", fold, seed)
                        directory = root / "cells" / label / initial.name
                    result = continuation_cell(directory, initial, fold, seed, weight, epochs, plan)
                    return dict(recipe=recipe, arm=label, reused=recipe == "Mlic", **result)
                with ThreadPoolExecutor(max_workers=3) as pool:
                    futures = {pool.submit(fit, row): row for row in inputs[recipe]}
                    for future in as_completed(futures):
                        try:
                            state["completed"].append(future.result())
                        except BaseException as error:
                            state["failures"].append(dict(arm=label, fold=futures[future]["fold"],
                                                          seed=futures[future]["seed"], error=repr(error)))
                        update(root, state, "continuation-cells")
                if state["failures"]:
                    raise RuntimeError("continuation failures retained; inspect their complete logs")
        write(root / "FOLD_FITS_COMPLETE.json", dict(cells=state["completed"],
              initial_checkpoints_sha256=sha(root / "INITIAL_CHECKPOINTS.json"),
              plan_sha256=sha(path), full_data_followthrough="pending", objective_complete=False), fresh=True)
        update(root, state, "fold-fits-complete-requires-panels", next_action="compute fitted/held-out panels, then full-data continuation")
    except BaseException as error:
        update(root, state, "failed", failures=[*state["failures"], repr(error)])
        raise


def status(path):
    root = path.parent
    if (root / "QUEUE_CANCELLED.json").exists():
        cancelled = json.loads((root / "QUEUE_CANCELLED.json").read_text())
        write(root / "CURRENT_RUN.json", dict(stage="cancelled-before-fitting", **cancelled))
        print(json.dumps(dict(stage="cancelled-before-fitting", reason=cancelled["reason"],
                              completed=cancelled["completed_new_cells"])))
        return
    state = json.loads((root / "RUN_STATE.json").read_text())
    worker = state.get("worker", state["queue_process"])
    live = bound_worker(worker["pid"], worker["start_ticks"])
    failure = state["failures"]
    if state["stage"] == "fold-fits-complete-requires-panels":
        complete = json.loads((root / "FOLD_FITS_COMPLETE.json").read_text())
        if len(complete["cells"]) != state["required"] or complete["plan_sha256"] != sha(path):
            failure = [*failure, "fold completion receipt differs"]
    elif not live and not failure:
        failure = ["recorded process exited without stage completion"]
    value = dict(state, failures=failure, live_processes=process_tree(worker["pid"]) if live else [],
                 compute_lock=compute_lock(worker["pid"] if live else None), checked_utc=datetime.now(timezone.utc).isoformat())
    write(root / "CURRENT_RUN.json", value)
    print(json.dumps(dict(stage=value["stage"], completed=len(value["completed"]), required=value["required"],
                         failure=failure, lock=value["compute_lock"], next_action=value["next_action"])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("queue", "run", "status"))
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    globals()[args.mode](args.plan.resolve())
