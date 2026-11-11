"""Advance registered finishing analyses from saved artifacts, with no model polling."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from shaping_completion import encoded, sha


def ready_panel(study, cid):
    run=study/"runs"/cid/"attempt-001"
    return any((p/"COMPLETE.json").is_file() and not (p/"FAILED_GATE.json").exists() for p in
        (run/"evaluation",study/"reexport_max_pairs"/cid/run.name/"final"))


def job_complete(job):
    path=Path(job["complete_marker"])
    if not path.exists():return False
    record=json.loads(path.read_text())
    if any(record.get(k)!=v for k,v in job.get("completion",{}).items()):return False
    return all(isinstance(record.get(k),list) and len(record[k])==v for k,v in job.get("completion_lengths",{}).items())


def run(args):
    args.output.mkdir(parents=True,exist_ok=True)
    path=args.output/"STATUS.json"
    state=json.loads(path.read_text()) if path.exists() else dict(stage="waiting_for_registered_deliveries",jobs={})
    config=json.loads(args.config.read_text())
    with (args.output/"OWNER.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        children={}
        def save(stage,**values):
            state.update(stage=stage,pid=os.getpid(),checked_utc=datetime.now(timezone.utc).isoformat(),**values)
            path.write_bytes(encoded(state))
        while True:
            collection_path=args.dispatch/"continuous-collection/STATUS.json"
            legacy_path=args.dispatch/"legacy-delivery/STATUS.json"
            collection=json.loads(collection_path.read_text()) if collection_path.exists() else {}
            legacy=json.loads(legacy_path.read_text()) if legacy_path.exists() else {}
            alerts={}
            for name,directory in config["pipelines"].items():
                p=Path(directory)/"STATUS.json"
                if p.exists():
                    row=json.loads(p.read_text())
                    if row.get("stage") in ("failed_with_preserved_evidence","failed_preserved"):
                        alerts[name]=row
            if collection.get("collection_failures"):alerts["collection"]=collection["collection_failures"]
            if collection.get("failed_native_cells"):alerts["native_fits"]=collection["failed_native_cells"]
            if legacy.get("failures"):alerts["legacy_delivery"]=legacy["failures"]
            (args.output/"ALERTS.json").write_bytes(encoded(alerts))
            imported=collection.get("registered")==217 and collection.get("collected")==217 and len(legacy.get("completed",[]))==35
            ids=set(collection.get("job_ids",[])) | set(legacy.get("completed",[]))
            panels=sum(ready_panel(args.study,cid) for cid in ids)
            # A completed recovery can queue only the ten transferred fits for
            # final-only in-place evaluation; it never opens a tuning surface.
            for name,job in config["jobs"].items():
                prior=state["jobs"].get(name)
                if prior and prior["stage"] in ("complete","failed","submission_unresolved"):
                    continue
                if name in children:
                    process,log=children[name];code=process.poll()
                    if code is None:
                        continue
                    log.close();children.pop(name)
                    state["jobs"][name].update(stage="complete" if code==0 and job_complete(job) else "failed",exit_code=code)
                    continue
                if prior:
                    # The prior child may outlive a coordinator wrapper.
                    process=Path("/proc")/str(prior["pid"])/"cmdline"
                    if process.exists():
                        argv=[v.decode() for v in process.read_bytes().split(b"\0") if v]
                        script=next(v for v in job["command"] if v.endswith(".py"))
                        output=job["command"][job["command"].index("--output")+1]
                        if script in argv and output in argv:
                            continue
                    state["jobs"][name].update(stage="complete" if job_complete(job) else "submission_unresolved")
                    continue
                if job["gate"]=="legacy" and len(legacy.get("completed",[]))!=35:
                    continue
                if job["gate"]=="all_panels" and not (imported and len(ids)==252 and panels==252):
                    continue
                (args.output/(name+"-COMMAND.json")).write_bytes(encoded(job))
                log=(args.output/(name+".log")).open("x")
                child=subprocess.Popen(job["command"],stdout=log,stderr=subprocess.STDOUT)
                children[name]=(child,log)
                state["jobs"][name]=dict(stage="running",pid=child.pid,command_sha256=sha(args.output/(name+"-COMMAND.json")))
            failures={name:row for name,row in state["jobs"].items() if row["stage"] in ("failed","submission_unresolved")}
            if failures:alerts["finishing_jobs"]=failures
            archive_path=args.dispatch/"archive-once/STATUS.json"
            archived=json.loads(archive_path.read_text()) if archive_path.exists() else {}
            archived_ids=set(archived.get("complete",{}))
            if archived.get("failures"):alerts["archives"]=archived["failures"]
            (args.output/"ALERTS.json").write_bytes(encoded(alerts))
            complete=imported and len(ids)==252 and panels==252 and ids<=archived_ids and all(
                state["jobs"].get(name,{}).get("stage")=="complete" for name in config["jobs"])
            save("computation_and_archive_closure_ready" if complete else "needs_attention" if alerts else "advancing_registered_work",
                 delivered=collection.get("collected",0),required_new_deliveries=217,
                 recovered_legacy=len(legacy.get("completed",[])),required_legacy=35,
                 closed_final_panels=panels,expected_final_panels=252,archived_deliveries=len(ids&archived_ids),alerts=alerts)
            if complete:
                (args.output/"READY_FOR_SCIENTIFIC_SUMMARY.json").write_bytes(encoded(dict(
                    final_holdout_read=False,serving_model_promoted=False,
                    computational_closure=True,scientific_goal_completion_asserted=False,
                    registry_sha256=sha(args.config),source=__file__)))
                return
            time.sleep(60)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("study","dispatch","config","output"):parser.add_argument("--"+name,type=Path,required=True)
    run(parser.parse_args())
