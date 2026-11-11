"""Collect an explicit finite list of completed worker fits; never start training."""
import argparse
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


def run(study, jobs_path, output):
    jobs = json.loads(jobs_path.read_text())
    if not jobs or len({j["cell"] for j in jobs}) != len(jobs):
        raise ValueError("explicit distinct completed fit jobs required")
    if any(j["host"] not in ("i134", "r3500", "r3800x", "r5600g") for j in jobs):
        raise ValueError("current owner compute restriction")
    output.mkdir(parents=True, exist_ok=False)
    status = dict(pid=os.getpid(), stage="starting", completed=[], failed=None)

    def save(stage, **values):
        status.update(stage=stage, checked_utc=datetime.now(timezone.utc).isoformat(), **values)
        temporary = output / "STATUS.writing"
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(output / "STATUS.json")

    try:
        for job in jobs:
            cid, host = job["cell"], job["host"]
            destination = study / "runs" / cid / "attempt-001"
            if destination.exists():
                raise ValueError("preserve existing coordinator attempt: " + str(destination))
            source = Path(job["source"])
            driver = source / "scripts/rev4_featpot/shaping_worker_collect.py"
            bundle = study.parent / "worker-bundles" / ("resume-20261111-" + cid + ".tar")
            command = [str(Path.home() / "work/claudehints/scripts/run-heavy"),
                       "--mem", "1G", "--jobs", "1", "--", "python3", "-B", str(driver),
                       "pack", "--study", str(study), "--assignment", job["assignment"],
                       "--cell", cid, "--bundle", str(bundle)]
            save("packing_completed_fit", cell=cid, host=host, bundle=str(bundle))
            with (output / (cid + "-pack.log")).open("x") as log:
                subprocess.run(["ssh", host, "env TMPDIR=" + shlex.quote(str(Path.home() / "tmp"))
                                + " PYTHONPATH=" + shlex.quote(str(study.parent / "python-runtime"))
                                + " " + shlex.join(command)], stdout=log, stderr=subprocess.STDOUT, check=True)
            bundle.parent.mkdir(parents=True, exist_ok=True)
            save("transferring_completed_fit", cell=cid)
            subprocess.run(["rsync", "-rlpt", "--ignore-existing", host + ":" + str(bundle),
                            host + ":" + str(bundle) + ".json", str(bundle.parent) + "/"], check=True)
            sys.path.insert(0, str(source / "scripts/rev4_featpot"))
            owner = importlib.import_module("shaping_worker_collect")
            pin = json.loads(Path(str(bundle) + ".json").read_text())["bundle_sha256"]
            owner.verify_transfer(bundle, pin, Path(job["assignment"]))
            save("collecting_completed_fit", cell=cid)
            owner.collect(study, Path(job["assignment"]), bundle, pin)
            status["completed"].append(cid)
            save("collected_completed_fit", cell=cid)
        save("complete", cell=None)
    except Exception as error:
        save("failed_preserved", failed=str(error))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "jobs", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.study, args.jobs, args.output)
