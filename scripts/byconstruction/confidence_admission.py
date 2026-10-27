"""Supervise a frozen native source screen behind registered TRAIN studies.

This binds committed zenpapers owners and projected identifiers. It opens no
quality responses and grants no fitting permission. Pixel work requires the
shared resource scope; outputs and failed attempts are retained.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/rev4_featpot"))
from run_processes import bound_worker, compute_lock, process_snapshot, process_tree


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".new")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def checked_plan(path):
    plan = json.loads(path.read_text())
    if plan["schema"] != "mlic-confidence-native-admission-plan-v1":
        raise ValueError("native admission plan schema differs")
    if plan["quality_responses_permitted"] is not False:
        raise ValueError("source admission cannot read quality responses")
    for name, pin in plan["input_sha256"].items():
        if sha(name) != pin:
            raise ValueError(f"native admission input changed: {name}")
    if sha(__file__) != plan["coordinator_sha256"]:
        raise ValueError("queued coordinator changed")
    return plan


def after_finished(study):
    failure = study / "LAUNCH_FAILED.json"
    if failure.exists():
        raise RuntimeError(f"preceding study failed; inspect {failure}")
    if (study / "QUEUE_EXIT.json").exists():
        exit_record = json.loads((study / "QUEUE_EXIT.json").read_text())
        complete = json.loads((study / "TRAIN_SCREEN_COMPLETE.json").read_text())
        if exit_record["returncode"] != 0 or complete["selection_sha256"] != sha(study / "TRAIN_SELECTION.json"):
            raise ValueError("preceding study completion does not match artifacts")
        return True
    process = json.loads((study / "QUEUED.json").read_text())["queue_process"]
    if not bound_worker(process["pid"], process["start_ticks"]):
        raise RuntimeError("preceding study queue exited without a completion receipt")
    return False


def record(root, state, stage, **extra):
    state.update(stage=stage, checked_utc=datetime.now(timezone.utc).isoformat(), **extra)
    write(root / "ADMISSION_STATE.json", state)


def queue(path):
    plan = checked_plan(path)
    root = path.parent
    state_path = root / "ADMISSION_STATE.json"
    if state_path.exists():
        raise ValueError("admission attempt already exists; preserve it")
    state = dict(plan_sha256=sha(path), source_commit=plan["zenpapers_commit"],
                 coordinator_sha256=plan["coordinator_sha256"], queue_process=process_snapshot(os.getpid()),
                 run_directory=str(root), required_stages=2, completed_stages=[],
                 failure=None, quality_responses_read=False, objective_complete=False,
                 next_action="wait for registered TRAIN study; replay native source admission")
    record(root, state, "waiting-for-training")
    try:
        while not after_finished(Path(plan["after_study"])):
            time.sleep(30)
        checked_plan(path)
        command = ["flock", str(Path.home() / "tmp/heavy.lock"),
                   str(Path.home() / "work/claudehints/scripts/run-heavy"),
                   "--mem", "24G", "--jobs", "4", "--",
                   sys.executable, "-B", str(Path(__file__).resolve()), "run", str(path)]
        record(root, state, "queued-for-native-screen", command=command)
        with (root / "launch.log").open("x") as log:
            subprocess.run(command, env=dict(os.environ, TMPDIR=str(Path.home() / "tmp"),
                OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1"),
                stdout=log, stderr=subprocess.STDOUT, check=True)
    except BaseException as error:
        state = json.loads(state_path.read_text())
        record(root, state, "failed", failure=repr(error))
        raise


def run(path):
    root = path.parent
    state = json.loads((root / "ADMISSION_STATE.json").read_text())
    try:
        if state["plan_sha256"] != sha(path):
            raise ValueError("queued plan changed")
        if not compute_lock(os.getpid())["held_by_worker_ancestor"]:
            raise ValueError("native screening requires the shared compute lock")
        plan = checked_plan(path)
        state["worker"] = process_snapshot(os.getpid())
        common = ["--inventory", plan["inventory"], "--out", plan["screen"],
                  "--thumbs", plan["thumbs"], "--packed-binary", plan["packed_binary"],
                  "--targets", plan["targets"], "--baseline-screen", plan["baseline_screen"]]
        for phase in ("decode", "screen"):
            command = [sys.executable, "-B", plan["screen_owner"], phase, *common]
            record(root, state, phase, command=command)
            subprocess.run(command, check=True)
            state["completed_stages"].append(phase)
            record(root, state, phase)
        screen = Path(plan["screen"])
        summary = json.loads((screen / "screen-summary.json").read_text())
        if summary["labels_read"] is not False or summary["native_decode_errors"] != 0:
            raise ValueError("native screen incomplete")
        for name, pin in summary["artifact_sha256"].items():
            if sha(screen / name) != pin:
                raise ValueError(f"native screen output changed: {name}")
        exported = {}
        for source, target in plan["metadata_exports"].items():
            target = Path(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            with Path(source).open("rb") as src, target.open("xb") as dest:
                shutil.copyfileobj(src, dest, length=1048576)
            if sha(source) != sha(target):
                raise ValueError("source-authority export changed")
            exported[str(target)] = sha(target)
        record(root, state, "screen-complete-requires-membership-freeze",
               screen_summary_sha256=sha(screen / "screen-summary.json"), metadata_exports=exported,
               next_action="review retained source matches; inherit roles; freeze and validate exact read scope")
    except BaseException as error:
        record(root, state, "failed", failure=repr(error))
        raise


def status(path):
    root = path.parent
    state = json.loads((root / "ADMISSION_STATE.json").read_text())
    worker = state.get("worker", state["queue_process"])
    live = bound_worker(worker["pid"], worker["start_ticks"])
    lock = compute_lock(worker["pid"] if live else None)
    stage, failure = state["stage"], state["failure"]
    if stage == "screen-complete-requires-membership-freeze":
        plan = json.loads(path.read_text())
        summary = Path(plan["screen"]) / "screen-summary.json"
        if not summary.is_file() or sha(summary) != state["screen_summary_sha256"]:
            failure = "screen completion artifact changed or missing"
    elif not live and not failure:
        failure = "recorded process is no longer live; no completion proof"
    result = dict(state, status="failed" if failure else stage, failure=failure,
                  live_processes=process_tree(worker["pid"]) if live else [], compute_lock=lock,
                  checked_utc=datetime.now(timezone.utc).isoformat())
    write(root / "CURRENT_RUN.json", result)
    print(json.dumps({k: result[k] for k in ("run_directory", "status", "completed_stages",
          "required_stages", "compute_lock", "failure", "checked_utc", "next_action")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("queue", "run", "status"))
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    globals()[args.mode](args.plan.resolve())
