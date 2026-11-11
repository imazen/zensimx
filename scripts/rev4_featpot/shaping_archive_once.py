"""Archive each completed later-stage delivery once, retaining verified readback receipts."""
import argparse
import fcntl
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time
from shaping_completion import REPO, encoded, sha


def sweep(args, state):
    from shaping_execution import load_registration
    from shaping_archive_supervision import later_neural_bundles, later_control_bundles
    _, cells, _ = load_registration(args.study)
    allowed=None
    if getattr(args,"delivery_root",None):
        allowed=set()
        p=args.delivery_root/"continuous-collection/STATUS.json"
        if p.exists():allowed.update(json.loads(p.read_text()).get("job_ids",[]))
        p=args.delivery_root/"legacy-delivery/JOBS.json"
        if p.exists():allowed.update(r["cell"]["id"] for r in json.loads(p.read_text()))
    extras=set()
    if getattr(args,"extra_native_jobs",None):
        extras={row["cell"] for row in json.loads(args.extra_native_jobs.read_text())}
        if allowed is not None:allowed.update(extras)
    for result in sorted((args.study / "runs").glob("*/attempt-*/RESULT.json")):
        run = result.parent; cid = run.parent.name
        if (allowed is not None and cid not in allowed) or cid in state["complete"] or cid in state["failures"]:
            continue
        missing=state.get("waiting_inputs",{}).get(cid)
        if missing and not Path(missing).exists():
            continue
        value = json.loads(result.read_text()); cell = value["cell"]
        if cid not in extras and cell["stage"] not in ("outer","fresh_control_outer","fresh_control_outer_matched","human_only","source_curve","shuffle"):
            continue
        if value["exit_code"] != 0 or not (run / "WORKER_COLLECTION.json").exists():
            continue
        native = (run / "fit/RESULT.json").exists()
        panels = (run / "evaluation", args.study / "reexport_max_pairs" / cid / run.name / "final")
        if not any((p / "COMPLETE.json").exists() for p in panels):
            continue
        if cid not in extras and native and not any((run / n).exists() for n in ("EVALUATION_WORKER_IMPORT.json","DIRECT_LATER_EVALUATION.json")):
            continue
        try:
            if cid in extras:
                from shaping_archive_supervision import rescue_eligible, canonical_candidate
                if not rescue_eligible(args.study,run.parent,cells):continue
                if not canonical_candidate(args.study,run,cell):
                    raise ValueError("extra native delivery lacks qualified supplemental evidence")
                bundles=[("run",run.parent),("panels",args.study/"reexport_max_pairs"/cid),
                         ("bakes",args.study/"reexports"/cid)]
            else:
                bundles = (later_neural_bundles if native else later_control_bundles)(args.study,run.parent,cells)
            if not bundles:
                continue
            receipts = {}
            for kind, root in bundles:
                marker = root / "MIRROR_VERIFIED.json"
                if not marker.exists():
                    destination = args.destination / ("later-" + kind + "-" + cid)
                    log = args.output / (cid + "-" + kind + "-" + str(len(list(args.output.glob(cid + "-" + kind + "-*.log")))) + ".log")
                    with log.open("x") as stream:
                        subprocess.run(["python3","-B",str(REPO/"scripts/byconstruction/dataset_mirror.py"),
                            "--root",str(root),"--nfs","--cold-readback","--destination",str(destination)],
                            stdout=stream,stderr=subprocess.STDOUT,check=True)
                receipt = json.loads(marker.read_text())
                destination = Path(receipt["destination"])
                if not receipt["all_artifact_sha256_verified"] or sha(root/"SHA256SUMS")!=sha(destination/"SHA256SUMS"):
                    raise ValueError("archive readback receipt differs")
                receipts[str(root)] = dict(destination=str(destination),receipt_sha256=sha(marker),manifest_sha256=sha(root/"SHA256SUMS"))
            state["complete"][cid] = receipts
        except FileNotFoundError as error:
            state.setdefault("waiting_inputs",{})[cid]=str(error.filename)
        except Exception as error:
            logs=list(args.output.glob(cid+"-*.log"))
            transient=any("Stale file handle" in p.read_text() for p in logs)
            if transient:
                state.setdefault("storage_wait",{})[cid]=str(error)
            else:
                state["failures"][cid] = str(error)
        save(args,state)
    return state


def save(args,state):
    state.update(pid=os.getpid(),checked_utc=datetime.now(timezone.utc).isoformat())
    (args.output/"STATUS.json").write_bytes(encoded(state))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("study","output","destination"):parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--watch",action="store_true")
    parser.add_argument("--mount-config",type=Path)
    parser.add_argument("--extra-native-jobs",type=Path)
    parser.add_argument("--delivery-root",type=Path);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    path=args.output/"STATUS.json"
    state=json.loads(path.read_text()) if path.exists() else dict(stage="archiving_completed_deliveries",complete={},failures={})
    with (args.output/"OWNER.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            if args.mount_config:
                config=json.loads(args.mount_config.read_text())
                from shaping_archive_mount import ensure
                try:
                    ensure(Path(config["target"]),config["source"],Path(config["mount_root"]),config["subdirectory"])
                    state["storage_unavailable"]=None
                except OSError as error:
                    state["storage_unavailable"]=str(error);save(args,state)
                    if not args.watch:raise
                    time.sleep(60);continue
            sweep(args,state);save(args,state)
            if not args.watch:break
            time.sleep(60)
