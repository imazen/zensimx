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

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/"scripts/rev4_featpot"))
from run_processes import process_snapshot, compute_lock
from v2_common import sha


def launch(study, commit, trainer):
    if len(commit)!=40 or len(trainer)!=64 or any(c not in "0123456789abcdef" for c in commit+trainer):
        raise ValueError("exact source and trainer pins required")
    subprocess.run(["git","merge-base","--is-ancestor",commit,"origin/main"],cwd=REPO,check=True)
    build=json.loads((study/"trainer/BUILD.json").read_text())
    if build["binary_sha256"]!=trainer or sha(study/"trainer/zensim_mlp_train")!=trainer:
        raise ValueError("queued trainer identity differs")
    subprocess.run(["git","merge-base","--is-ancestor",build["source_commit"],commit],cwd=REPO,check=True)
    if (study/"RUN_STATE.json").exists() or (study/"SOURCE.json").exists():
        raise ValueError("queue requires a fresh study")
    command=["flock",str(Path.home()/"tmp/heavy.lock"),
        str(Path.home()/"work/claudehints/scripts/run-heavy"),"--mem","36G","--jobs","12","--",
        "just","dataset-order-chain",str(study),commit,trainer]
    now=lambda:datetime.now(timezone.utc).isoformat()
    queue=dict(stage="queued",candidate="MlicOrder",source_commit=commit,binary_sha256=trainer,
        queue_script_sha256=sha(__file__),queue_process=process_snapshot(os.getpid()),
        compute_lock=compute_lock(os.getpid()),queued_utc=now(),command=command,
        completed_cells=0,required_cells=21,next_action="acquire shared lock; reproduce Mlic; match; fit and screen")
    with (study/"QUEUED.json").open("x") as stream:json.dump(queue,stream,indent=2);stream.write("\n")
    try:
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
    a=p.parse_args();launch(a.study.resolve(),a.source_commit,a.trainer_sha256)
