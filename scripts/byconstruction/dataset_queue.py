"""Visible, supervised queue for a pinned independent TRAIN study.

The process remains attached to its caller. The shared lock and run-heavy
scope serialize all fitting/evaluation; no background daemon is created.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/"scripts/rev4_featpot"))
from run_processes import process_snapshot, compute_lock, bound_worker
from v2_common import sha


def panel_dependency(plan, pin):
    """Preserve AIC priority; require pinned completion or a verified live worker."""
    if sha(plan) != pin:
        raise ValueError("AIC full-panel dependency plan changed")
    root = Path(plan).parent
    complete = root / "panels/FULL_PANEL_COMPLETE.json"
    if complete.exists():
        result = json.loads(complete.read_text())
        inventory = root / "panels/EVIDENCE_INVENTORY.json"
        if (result["plan_sha256"] != pin or result["selector"] is not None
                or sha(inventory) != result["evidence_inventory_sha256"]):
            raise ValueError("AIC full-panel completion binding differs")
        return True
    path = root / "panels/STATE.json"
    state = json.loads((path if path.exists() else root / "FULL_PANEL_QUEUE.json").read_text())
    worker = state["worker"]
    if state.get("failures") or not bound_worker(worker["pid"], worker["start_ticks"]):
        raise RuntimeError("AIC full-panel dependency failed or exited; inspect preserved run")
    return False


def launch(study, commit, trainer, candidate="MlicOrder", after_full_panel=None):
    if candidate not in ("MlicOrder", "MlicConfidence"):
        raise ValueError("registered independent candidate required")
    if len(commit)!=40 or len(trainer)!=64 or any(c not in "0123456789abcdef" for c in commit+trainer):
        raise ValueError("exact source and trainer pins required")
    subprocess.run(["git","merge-base","--is-ancestor",commit,"origin/main"],cwd=REPO,check=True)
    build=json.loads((study/"trainer/BUILD.json").read_text())
    if build["binary_sha256"]!=trainer or sha(study/"trainer/zensim_mlp_train")!=trainer:
        raise ValueError("queued trainer identity differs")
    subprocess.run(["git","merge-base","--is-ancestor",build["source_commit"],commit],cwd=REPO,check=True)
    if (study/"RUN_STATE.json").exists() or (study/"SOURCE.json").exists():
        raise ValueError("queue requires a fresh study")
    dependency_pin = sha(after_full_panel) if after_full_panel is not None else None
    if candidate == "MlicConfidence" and after_full_panel is None:
        raise ValueError("confidence queue must preserve the current AIC full-panel priority")
    confidence_pin = None
    if candidate == "MlicConfidence":
        from confidence_weights import cell_binding
        registry = study / "policy/confidence-manifests.json"
        confidence_pin = sha(registry)
        for fold in ("kadid", "tid2013", "konfig", "cid22_a25", "pipal", "pairs", "mos7"):
            for seed in range(3):
                cell_binding(registry, confidence_pin, candidate, fold, seed)
    command=["flock",str(Path.home()/"tmp/heavy.lock"),
        str(Path.home()/"work/claudehints/scripts/run-heavy"),"--mem","36G","--jobs","12","--",
        "just", "dataset-confidence-chain" if candidate == "MlicConfidence" else "dataset-order-chain",
        str(study),commit,trainer]
    now=lambda:datetime.now(timezone.utc).isoformat()
    queue=dict(stage="queued",candidate=candidate,source_commit=commit,binary_sha256=trainer,
        queue_script_sha256=sha(__file__),queue_process=process_snapshot(os.getpid()),
        compute_lock=compute_lock(os.getpid()),queued_utc=now(),command=command,
        after_full_panel=str(after_full_panel) if after_full_panel else None,
        after_full_panel_sha256=dependency_pin,
        confidence_manifest_index_sha256=confidence_pin,
        completed_cells=0,required_cells=21,next_action="acquire shared lock; reproduce Mlic; admit candidate manifests; fit and screen")
    with (study/"QUEUED.json").open("x") as stream:json.dump(queue,stream,indent=2);stream.write("\n")
    try:
        if after_full_panel is not None:
            while not panel_dependency(after_full_panel, dependency_pin):
                time.sleep(30)
        environment=dict(os.environ,TMPDIR=str(Path.home()/"tmp"),JUST_TEMPDIR=str(Path.home()/"tmp"))
        with (study/"launch.log").open("x") as log:
            subprocess.run(command,cwd=REPO,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
    except BaseException as error:
        with (study/"LAUNCH_FAILED.json").open("x") as stream:
            json.dump(dict(error=repr(error),checked_utc=now(),queue_process=queue["queue_process"]),stream,indent=2)
        raise
    with (study/"QUEUE_EXIT.json").open("x") as stream:
        json.dump(dict(returncode=0,checked_utc=now(),objective_complete=False),stream,indent=2)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("study",type=Path);p.add_argument("source_commit");p.add_argument("trainer_sha256")
    p.add_argument("--candidate", choices=("MlicOrder", "MlicConfidence"), default="MlicOrder")
    p.add_argument("--after-full-panel", type=Path)
    a=p.parse_args();launch(a.study.resolve(),a.source_commit,a.trainer_sha256,a.candidate,a.after_full_panel)
