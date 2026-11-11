"""Collect the finite resumed dispatch; each scientific fit retains its original owner."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time

ALLOWED = ("i134", "r3500", "r3800x", "r5600g")


def discover(study, root):
    jobs = {}
    for request in root.glob("**/jobs/*/REQUEST.json"):
        if request.relative_to(root).parts[0].startswith("source-"):
            continue
        q = json.loads(request.read_text())
        if q.get("schema") != "shaping-native-worker-v2" or q.get("execution_kind") not in ("control", "outer", "variant"):
            continue
        cid, host = q["cell"]["id"], q["host"]
        if host not in ALLOWED:
            raise ValueError("resumed dispatch contains excluded compute")
        prefix = "outer" if q["cell"]["stage"] in ("outer", "fresh_control_outer", "fresh_control_outer_matched") else "variant"
        assignment = study / "downstream_assignments" / f"{prefix}-{cid}-{host}.json"
        if not assignment.exists():
            continue  # Registration publishes the assignment after the request.
        a = json.loads(assignment.read_text())
        if "worker_request_path" in a and a["worker_request_path"] != str(request):
            continue
        if cid in jobs and jobs[cid]["request"] != str(request):
            raise ValueError("ambiguous resumed cell request")
        source = root / "source-f05" if q["source_root"] == "/home/lilith/work/zen/zensimx" else Path(q["source_root"])
        jobs[cid] = dict(cell=cid, host=host, request=str(request), assignment=str(assignment),
                         source=str(source), kind=q["execution_kind"], stage=q["cell"]["stage"])
    if len(jobs) > 217:
        raise ValueError("resumed dispatch exceeds its registered 214 core plus three matched controls")
    return jobs


def observe(host, jobs):
    code = "paths=" + repr({j["cell"]: j["request"] for j in jobs}) + "\n" + """
from pathlib import Path
import json
rows={}
for cid,name in paths.items():
 p=Path(name).with_name('STATUS.json')
 if p.exists():
  row=json.loads(p.read_text());rows[cid]=dict(stage=row['stage'],exit_code=row.get('exit_code'))
 else:rows[cid]=dict(stage='not_yet_started')
print(json.dumps(rows))
"""
    reply = subprocess.run(["ssh", host, "python3 -B -"], input=code, text=True,
                           capture_output=True, check=True, timeout=30)
    return json.loads(reply.stdout)


def run(args):
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo / "scripts/rev4_featpot"))
    from shaping_completion import encoded, sha
    args.output.mkdir(parents=True, exist_ok=args.resume)
    with (args.output / "OWNER.lock").open("a" if args.resume else "x") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = args.output / "STATUS.json"
        failures = json.loads(previous.read_text()).get("collection_failures", {}) if args.resume and previous.exists() else {}
        while True:
            jobs = discover(args.study, args.dispatch)
            observations, unavailable = {}, {}
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = {h: pool.submit(observe, h, [j for j in jobs.values() if j["host"] == h])
                           for h in ALLOWED}
                for host, future in futures.items():
                    try:
                        observations.update(future.result())
                    except Exception as error:
                        unavailable[host] = str(error)
            collected, trained, failed = [], [], []
            for cid, job in sorted(jobs.items()):
                root = args.study / "runs" / cid / "attempt-001"
                if (root / "WORKER_COLLECTION.json").exists():
                    collected.append(cid)
                    continue
                status = observations.get(cid, {})
                if status.get("stage") == "failed":
                    failed.append(cid)
                    continue
                if status.get("stage") != "trained" or cid in failures:
                    continue
                trained.append(cid)
                q = json.loads(Path(job["request"]).read_text())
                if job["kind"] == "control" and job["host"] != socket.gethostname() and job["stage"] not in ("outer",):
                    pipeline = Path(job["request"]).parents[3] / "STATUS.json"
                    state = json.loads(pipeline.read_text()) if pipeline.exists() else {}
                    if state.get("stage") not in ("failed_with_preserved_evidence", "all_variants_submitted_and_controls_collected"):
                        continue
                directory = args.output / cid
                directory.mkdir()
                def call(command):
                    with (directory / "collection.log").open("a") as log:
                        log.write(shlex.join(command) + "\n"); log.flush()
                        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
                try:
                    if job["host"] == socket.gethostname():
                        call(["python3", "-B", str(repo / "scripts/rev4_featpot/shaping_local_collection.py"),
                              "--study", str(args.study), "--request", job["request"],
                              "--assignment", job["assignment"], "--source", job["source"]])
                    elif job["kind"] != "control":
                        path = directory / "JOBS.json"; path.write_bytes(encoded([job]))
                        call(["python3", "-B", str(repo / "scripts/rev4_featpot/shaping_resume_collect.py"),
                              "--study", str(args.study), "--jobs", str(path), "--output", str(directory / "native")])
                    else:
                        source = Path(job["source"]); driver = source / "scripts/rev4_featpot/shaping_control_worker.py"
                        bundle = args.study.parent / "worker-bundles" / ("resume-control-" + cid + ".tar")
                        remote = ["env", "TMPDIR=/home/lilith/tmp", "PYTHONPATH=" + str(args.study.parent / "python-runtime"),
                                  "/home/lilith/work/claudehints/scripts/run-heavy", "--mem", "1G", "--jobs", "1", "--",
                                  "python3", "-B", str(driver), "pack", "--request", job["request"], "--bundle", str(bundle)]
                        call(["ssh", job["host"], shlex.join(remote)])
                        local = directory / bundle.name
                        call(["rsync", "-rlpt", "--ignore-existing", job["host"] + ":" + str(bundle),
                              job["host"] + ":" + str(bundle) + ".json", str(directory) + "/"])
                        pin = json.loads(Path(str(local) + ".json").read_text())["bundle_sha256"]
                        call(["python3", "-B", str(driver), "collect", "--request", job["request"],
                              "--bundle", str(local), "--bundle-sha256", pin])
                    collected.append(cid)
                except Exception as error:
                    failures[cid] = str(error)
                    (directory / "FAILURE.json").write_bytes(encoded(dict(error=str(error), training_replayed=False)))
            record = dict(pid=os.getpid(), stage="collecting_registered_dispatch",
                          checked_utc=datetime.now(timezone.utc).isoformat(), registered=len(jobs),
                          required=217, collected=len(collected), newly_trained_pending=trained,
                          failed_native_cells=failed, collection_failures={k:v for k,v in failures.items() if k not in collected}, unavailable_hosts=unavailable,
                          training_replayed=False, job_ids=sorted(jobs))
            (args.output / "STATUS.json").write_bytes(encoded(record))
            if len(jobs) == 217 and len(collected) == 217:
                record.update(stage="all_registered_dispatch_collected")
                (args.output / "STATUS.json").write_bytes(encoded(record))
                return
            if not args.watch:
                return
            time.sleep(60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "dispatch", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--resume", action="store_true", help="retain existing receipts and unresolved collection failures")
    run(parser.parse_args())
