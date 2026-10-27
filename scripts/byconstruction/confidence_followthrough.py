"""Attached, serial followthrough of the preregistered confidence AIC comparison.

Freeze this owner and its input config before queueing. Completed omission
evidence supplies artifact identities, never a scalar eligibility decision.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aic_adaptation import bound_worker, process_snapshot, sha, write, require_clean_factors
from mlic_confidence_full import completed_omissions


def omission_ready(study):
    for name in ("FAILED.json", "LAUNCH_FAILED.json"):
        if (study / name).exists():
            raise RuntimeError(f"confidence prerequisite failed: {study / name}")
    complete = study / "TRAIN_SCREEN_COMPLETE.json"
    if complete.exists():
        receipt = json.loads(complete.read_text())
        selection = study / "TRAIN_SELECTION.json"
        if receipt["selection_sha256"] != sha(selection):
            raise ValueError("completed confidence omission evidence changed")
        completed_omissions(json.loads(selection.read_text()))
        return True
    state = study / "RUN_STATE.json"
    if state.exists():
        value = json.loads(state.read_text())
        if value.get("failure") or value.get("stage") == "failed":
            raise RuntimeError("confidence omission process failed")
        pid, ticks = value["pid"], value["start_ticks"]
    else:
        value = json.loads((study / "QUEUED.json").read_text())["queue_process"]
        pid, ticks = value["pid"], value["start_ticks"]
    if not bound_worker(pid, ticks):
        raise RuntimeError("confidence prerequisite exited without completion")
    return False


def fold_extension(template, study, source, commit, pins):
    if template["schema"] != "aic-recipe-adaptation-plan-v1":
        raise ValueError("original four-recipe fold registration required")
    plan = copy.deepcopy(template)
    plan.update(schema="aic-confidence-adaptation-plan-v1", recipes=["Mlic", "MlicConfidence"],
        coordinator_commit=commit, coordinator_sha256=pins[str(source / "aic_adaptation.py")],
        initial_cells={"Mlic":template["initial_cells"]["Mlic"],
                       "MlicConfidence":str(study / "cells/MlicConfidence")},
        starting_studies=[str(study)],
        recipe_authority={"Mlic":template["recipe_authority"]["Mlic"], "MlicConfidence":str(study)},
        required_new_fold_cells=24, required_reused_fold_cells=24,
        full_data_followthrough="serial supervisor: full confidence then common full continuation and panels",
        input_sha256={**template["input_sha256"], **pins})
    return plan


def full_extension(template, fold, fold_panel, initial, original, source, commit, pins):
    if template["schema"] != "aic-recipe-full-plan-v1":
        raise ValueError("original four-recipe full registration required")
    plan = copy.deepcopy(template)
    plan.update(schema="aic-confidence-full-plan-v1", recipes=["Mlic", "MlicConfidence"],
        coordinator_commit=commit, coordinator_sha256=pins[str(source / "aic_adaptation_full.py")],
        fold_plan=str(fold), fold_plan_sha256=pins[str(fold)], fold_panel=str(fold_panel),
        confidence_full=str(initial), reuse_mlic_full=str(original),
        required_new_initial_fits=0, required_new_continuations=6, required_reused_continuations=6,
        input_sha256={**template["input_sha256"], **pins})
    return plan


def run(config_path):
    config = json.loads(config_path.read_text())
    root = config_path.parent
    source = Path(__file__).resolve().parent
    if config["schema"] != "confidence-AIC-followthrough-v1":
        raise ValueError("registered confidence followthrough required")
    if config["source_files_sha256"].get(str(Path(__file__).resolve())) != sha(__file__):
        raise ValueError("frozen followthrough owner differs")
    def check_config():
        for name, pin in {**config["input_sha256"], **config["source_files_sha256"]}.items():
            if sha(name) != pin:
                raise ValueError(f"followthrough input changed: {name}")
    check_config()
    require_clean_factors(os.environ)
    study = Path(config["study"])
    if (root / "STATE.json").exists():
        raise ValueError("preserve previous followthrough attempt")
    state = dict(stage="waiting-for-confidence-omissions", worker=process_snapshot(os.getpid()),
                 config_sha256=sha(config_path), completed=[], failures=[], objective_complete=False)
    def record(stage, **details):
        state.update(stage=stage, checked_utc=datetime.now(timezone.utc).isoformat(), **details)
        write(root / "STATE.json", state)
    def pins(paths):
        return {str(p):sha(p) for p in paths}
    def execute(stage, argv):
        check_config()
        record(stage, command=list(map(str, argv)))
        with (root / f"{stage}.log").open("x") as log:
            subprocess.run(list(map(str, argv)), stdout=log, stderr=subprocess.STDOUT, check=True)
        state["completed"].append(stage)
        record(stage + "-complete")
    def invoke(name, *args):
        return [sys.executable, "-B", source / name, *args]
    record(state["stage"])
    try:
        while not omission_ready(study):
            time.sleep(30)
        check_config()
        selection = json.loads((study / "TRAIN_SELECTION.json").read_text())
        for name, pin in selection["input_sha256"].items():
            if sha(name) != pin:
                raise ValueError("confidence omission evidence changed before followthrough")
        source_receipt = json.loads((study / "SOURCE.json").read_text())
        queued = json.loads((study / "QUEUED.json").read_text())
        if (source_receipt["candidate"] != "MlicConfidence"
                or source_receipt["trainer_sha256"] != queued["binary_sha256"]
                or source_receipt["commit"] != queued["source_commit"]):
            raise ValueError("completed confidence study differs from queued identity")
        common_pins = {**config["source_files_sha256"], **pins([study / name for name in
            ("TRAIN_SELECTION.json", "TRAIN_SCREEN_COMPLETE.json", "SOURCE.json", "BASELINE.json",
             "bin/zensim_mlp_train", "source/scripts/rev4_featpot/v2_confirm_fit.py",
             "frozen_configuration/cols_c.txt", "MlicConfidence/root/wide/frozen.json")])}
        fold_root = root / "fold"
        fold_root.mkdir()
        fold_path = fold_root / "PLAN.json"
        template = json.loads(Path(config["base_fold_plan"]).read_text())
        write(fold_path, fold_extension(template, study, source, config["source_commit"], common_pins), fresh=True)
        execute("fold-fits", invoke("aic_adaptation.py", "queue", fold_path))
        execute("fold-panels", invoke("aic_adaptation_panel.py", "queue", fold_path, fold_root / "panels"))
        initial = root / "initial"
        initial.mkdir()
        registry = Path(config["full_registry"])
        initial_plan = dict(schema="mlic-confidence-full-plan-v1", study=str(study), registry=str(registry),
            runner_sha256=sha(source / "mlic_confidence_full.py"),
            input_sha256={**common_pins, **pins([registry])})
        write(initial / "PLAN.json", initial_plan, fresh=True)
        execute("full-confidence-fits", ["flock", Path.home() / "tmp/heavy.lock",
            Path.home() / "work/claudehints/scripts/run-heavy", "--mem", "24G", "--jobs", "3", "--",
            *invoke("mlic_confidence_full.py", initial / "PLAN.json")])
        full_root = root / "full"
        full_root.mkdir()
        original = Path(config["base_full_plan"]).parent
        template = json.loads(Path(config["base_full_plan"]).read_text())
        full_pins = {**common_pins, **pins([fold_path, initial / "PLAN.json", initial / "FULL_FITS_COMPLETE.json",
            original / "PLAN.json", original / "FULL_FITS_COMPLETE.json"])}
        full_path = full_root / "PLAN.json"
        write(full_path, full_extension(template, fold_path, fold_root / "panels", initial, original,
            source, config["source_commit"], full_pins), fresh=True)
        execute("full-continuations", invoke("aic_adaptation_full.py", "queue", full_path))
        execute("full-panels", invoke("aic_adaptation_full_panel.py", "queue", full_path, full_root / "panels"))
        record("confidence-AIC-panels-complete", next_action="review all recipe panels and remaining production qualification")
    except BaseException as error:
        record("failed", failures=[*state["failures"], repr(error)])
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    run(parser.parse_args().config.resolve())
