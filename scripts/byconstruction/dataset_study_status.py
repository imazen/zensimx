"""Resolve the active study population, live processes and queue before reporting.

No target table is opened. The canonical cell-progress reader owns cell checks.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/"scripts/rev4_featpot"))
from run_processes import bound_worker, compute_lock, process_tree

FOLDS = "kadid,tid2013,konfig,cid22_a25,pipal,pairs,mos7"


def population(study,state,candidate):
    if state["stage"] == "control":
        return study/"control-cells", "MlicControl", "kadid", "0"
    if state["stage"] == "cells":
        command = state["command"]
        return study/"cells", *(command[command.index(flag)+1] for flag in ("--label","--folds","--seeds"))
    return study/"cells", candidate, FOLDS, "0,1,2"


def status(study):
    state_path = study/"RUN_STATE.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        source = json.loads((study/"SOURCE.json").read_text())
        cells,arm,folds,seeds = population(study,state,source["candidate"])
        subprocess.run([sys.executable,"-B",str(REPO/"scripts/rev4_featpot/byc_cells.py"),"progress",
            "--arm",arm,"--folds",folds,"--seeds",seeds,"--run-state",str(state_path),
            "--output",str(study/"CURRENT_RUN.json")],env=dict(os.environ,BYC_CELLS=str(cells)),
            stdout=subprocess.PIPE,check=True,text=True)
        result = json.loads((study/"CURRENT_RUN.json").read_text())
        if not result["worker_alive"] and not result["failure"] and not (study/"TRAIN_SCREEN_COMPLETE.json").exists():
            result["status"] = "interrupted-before-screen-completion"
        result["objective_complete"] = False
    else:
        queue = json.loads((study/"QUEUED.json").read_text())
        q = queue["queue_process"]
        live = bound_worker(q["pid"],q["start_ticks"])
        failure_path = study/"LAUNCH_FAILED.json"
        failure = json.loads(failure_path.read_text()) if failure_path.exists() else None
        cancel_path = study/"QUEUE_CANCELLED.json"
        cancellation = json.loads(cancel_path.read_text()) if cancel_path.exists() else None
        result = dict(run_directory=str(study),source_commit=queue["source_commit"],
            binary_sha256=queue["binary_sha256"],declared_stage="queued",complete=0,requested=21,
            checked_utc=datetime.now(timezone.utc).isoformat(),worker_alive=False,queue_alive=bool(live),
            live_processes=process_tree(q["pid"]) if live else [],compute_lock=compute_lock(q["pid"] if live else None),
            status="failed" if failure else "queued" if live else "canceled-before-start" if cancellation else "queue-exited-without-starting",
            cancellation=cancellation,failure=failure,
            next_action="preserve canceled attempt; inspect replacement study" if cancellation and not live else queue["next_action"],
            objective_complete=False)
    path = study/"CURRENT_RUN.json"
    temporary = path.with_suffix(".json.new")
    temporary.write_text(json.dumps(result,indent=1)+"\n")
    temporary.replace(path)
    epochs=[c["latest_logged_epoch"] for c in result.get("cells",[]) if c["latest_logged_epoch"] is not None]
    print(json.dumps(dict(study=str(study),stage=result["declared_stage"],status=result["status"],
        complete=result["complete"],required=result["requested"],logged_epoch_range=[min(epochs),max(epochs)] if epochs else [],
        lock=result["compute_lock"],failure=result["failure"],checked_utc=result["checked_utc"])))
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("studies",nargs="+",type=Path)
    for study in parser.parse_args().studies: status(study.resolve())
