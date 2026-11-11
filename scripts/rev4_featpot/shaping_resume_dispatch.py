"""Finish a partially registered outer dispatch without recreating requests or fits."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time


def run(args):
    sys.path.insert(0, str(args.source / "scripts/rev4_featpot"))
    from shaping_completion import encoded, sha
    from shaping_outer_pipeline import placements, HOST_SLOTS, collect_controls
    from shaping_placement import loads
    from shaping_execution import load_registration
    from shaping_outer_admission import resolve
    from shaping_outer_dispatch import selected_cells
    from shaping_rescue_worker import register
    from shaping_control_worker import register as control
    from shaping_rescue_host_launch import launch
    args.output.mkdir(parents=True, exist_ok=False)
    status = dict(pid=os.getpid(), outer=args.outer, training_replayed=False)
    def save(stage, **values):
        status.update(stage=stage, checked_utc=datetime.now(timezone.utc).isoformat(), **values)
        (args.output / "STATUS.json").write_bytes(encoded(status))
    try:
        _, cells, _ = load_registration(args.study)
        admission, _ = resolve(args.study, args.outer)
        selected = selected_cells(cells, admission)
        selected += [c for c in cells if c["stage"] == "outer"
                     and c["partition"] == f"outer-{args.outer}"
                     and c["family"] in ("extra_trees", "shaped_additive")]
        required = {c["id"]: c for c in selected}
        if len(required) != 19:
            raise ValueError("complete registered reduced outer scope required")
        jobs = {}
        for request in args.previous.glob("jobs/*/REQUEST.json"):
            q = json.loads(request.read_text())
            cid = q["cell"]["id"]
            assignment = args.study / "downstream_assignments" / ("outer-" + cid + "-" + q["host"] + ".json")
            a = json.loads(assignment.read_text())
            if (q["cell"] != required.get(cid) or a["cells"] != [cid] or a["host"] != q["host"]
                    or any(a[k] != q[k] for k in ("plan_sha256", "build_sha256"))):
                raise ValueError("preserve conflicting registered request")
            if "worker_request_sha256" in a and a["worker_request_sha256"] != sha(request):
                raise ValueError("preserve changed registered request")
            if q["host"] not in ("i134", "r3500", "r3800x", "r5600g"):
                raise ValueError("registered request has an excluded host")
            jobs[cid] = dict(cell=q["cell"], host=q["host"], request=str(request),
                             request_sha256=sha(request), assignment=str(assignment),
                             assignment_sha256=sha(assignment), source=str(args.previous_source))
        live, observation = loads(args.study, {h: HOST_SLOTS[h] for h in ("i134", "r3500", "r3800x", "r5600g")})
        (args.output / "PLACEMENT_OBSERVATION.json").write_bytes(encoded(observation))
        for cell, host in placements([c for c in selected if c["id"] not in jobs], args.outer, live):
            save("registering_remaining_outer_fits", cell=cell["id"], registered=len(jobs))
            directory = args.output / "jobs" / cell["id"]
            if cell["family"] in ("extra_trees", "shaped_additive"):
                control(args.study, cell["id"], args.binaries, host, directory)
            else:
                register(args.study, cell["id"], host, args.binaries, args.qualification, directory, execution_kind="outer")
            request = directory / "REQUEST.json"
            assignment = args.study / "downstream_assignments" / ("outer-" + cell["id"] + "-" + host + ".json")
            with assignment.open("xb") as stream:
                stream.write((directory / "ASSIGNMENT.json").read_bytes())
            jobs[cell["id"]] = dict(cell=cell, host=host, request=str(request),
                                   request_sha256=sha(request), assignment=str(assignment),
                                   assignment_sha256=sha(assignment), source=str(args.source))
            (args.output / "JOBS.json").write_bytes(encoded(list(jobs.values())))
        if set(jobs) != set(required):
            raise ValueError("outer registration coverage differs")
        for source in sorted({j["source"] for j in jobs.values()}):
            wave = args.output / ("wave-" + Path(source).name)
            wave.mkdir()
            subset = [j for j in jobs.values() if j["source"] == source]
            (wave / "REGISTRATION.json").write_bytes(encoded(dict(stage="registered", jobs=subset,
                source_driver_sha256=sha(Path(source) / "scripts/rev4_featpot/shaping_rescue_worker.py"))))
            for host in ("i134", "r3500", "r3800x", "r5600g"):
                if any(j["host"] == host for j in subset):
                    save("launching_remaining_outer_fits", host=host, source=source)
                    launch([wave], Path(source), host, wave / ("launch-" + host))
        save("all_outer_fits_submitted", required_models=len(jobs))
        while True:
            done, total = collect_controls(args, list(jobs.values()))
            save("collecting_outer_controls", collected_controls=done, required_controls=total)
            if len(done) == total:
                save("all_outer_fits_submitted_and_controls_collected", required_models=len(jobs))
                return
            time.sleep(60)
    except Exception as error:
        save("failed_with_preserved_evidence", error=str(error))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "previous-source", "study", "previous", "output", "binaries", "qualification"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--outer", type=int, choices=range(3), required=True)
    run(parser.parse_args())
