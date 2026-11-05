#!/usr/bin/env python3
"""One-shot live evidence using the repository's canonical process/lock owner."""
import datetime
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/rev4_featpot"))
from run_processes import bound_worker, compute_lock, process_snapshot, process_tree


def research_checks(root, stem="RESEARCH"):
    """Observe the check launcher separately from its stage-writer receipt."""
    path = root / f"{stem}_CHECKS.json"
    if not path.exists():
        return None
    state = json.loads(path.read_text())
    pid = state["launcher_pid"]
    identity_path = root / f"research-launcher-{pid}.json"
    if identity_path.exists():
        identity = json.loads(identity_path.read_text())
        launcher = bound_worker(pid, identity["start_ticks"])
    else:
        launcher = process_snapshot(pid)
        if launcher and launcher["state"] not in ("Z", "X"):
            try:
                matches = Path(f"/proc/{pid}/cwd").resolve() == Path(state["source"])
            except OSError:
                matches = False
            if matches:
                identity_path.write_text(json.dumps(launcher, indent=2) + "\n")
            else:
                launcher = None
        else:
            launcher = None
    processes = process_tree(pid) if launcher else []
    lock = compute_lock(pid)
    owns = bool(set(lock.get("holder_pids", [])) & {p["pid"] for p in processes})
    complete = set(state["completed"]) == set(state["required"])
    stage = state["stage"]
    if launcher and not complete and not state["failure"]:
        stage = ("between checks" if stage.endswith(" passed") else
                 stage.split()[0] + (" running" if owns else " queued"))
    elif not launcher and not complete and not state["failure"]:
        stage = "launcher stopped before completion"
    actual = hashlib.sha256((Path(state["source"]) / "zensim/src/feature_v2.rs").read_bytes()).hexdigest()
    value = {**state, "checked_utc": datetime.datetime.now(datetime.UTC).isoformat(),
             "observed_stage": stage, "launcher": launcher, "compute_lock": lock,
             "owns_compute_lock": owns, "source_hash_matches": actual == state["kernel_sha256"],
             "processes": processes}
    (root / f"{stem}_LIVE_CHECK.json").write_text(json.dumps(value, indent=2) + "\n")
    with (root / f"{stem}_SUPERVISION.jsonl").open("a") as stream:
        stream.write(json.dumps(value) + "\n")
    return {"stage": stage, "completed": len(state["completed"]),
            "required": len(state["required"]), "source_hash_matches": value["source_hash_matches"]}


def main():
    root = Path(sys.argv[1]).resolve()
    task_root = root
    if "--stage" in sys.argv[2:]:
        stage = sys.argv[sys.argv.index("--stage") + 1]
        assert stage in ("before", "after")
        root /= stage
    if "--build" in sys.argv[2:]:
        pid = int((root / "build.pid").read_text())
        process = process_snapshot(pid)
        assert process and any("heavy.lock" in arg for arg in process["argv"]), "build launcher no longer matches its PID"
        tree = process_tree(pid)
        compilers = [p for p in tree if p["argv"] and Path(p["argv"][0]).name in ("cargo", "rustc")]
        value = json.loads((root / "PROGRESS.json").read_text())
        value.update({"last_check": datetime.datetime.now(datetime.UTC).isoformat(),
                      "launcher": process, "compute_lock": compute_lock(pid),
                      "compiler_pids": [p["pid"] for p in compilers],
                      "stage": f"{root.name} building" if compilers else f"{root.name} awaiting heavy lock"})
        (root / "PROGRESS.json").write_text(json.dumps(value, indent=2) + "\n")
        with (root / "SUPERVISION.jsonl").open("a") as stream:
            stream.write(json.dumps(value) + "\n")
        print(json.dumps({"utc": value["last_check"], "stage": value["stage"],
                          "launcher_pid": pid, "compiler_pids": value["compiler_pids"],
                          "lock": value["compute_lock"],
                          "completed_cells": value["completed_cells"]}))
        return
    identity = json.loads((root / "BINARY.json").read_text())
    binary = (root / "bin/a-speed-peers").resolve()
    actual = hashlib.sha256(binary.read_bytes()).hexdigest()
    assert actual == identity["binary_sha256"], "frozen benchmark binary changed"
    collector_pid = int((root / "collector.pid").read_text())
    collector_identity = root / f"collector-{collector_pid}.json"
    if collector_identity.exists():
        recorded = json.loads(collector_identity.read_text())
        collector = bound_worker(collector_pid, recorded["start_ticks"])
    else:
        collector = process_snapshot(collector_pid)
        if collector and collector["argv"][-2:] == [str(root), str(binary)] and any(
            Path(arg).name == "run.sh" for arg in collector["argv"]
        ):
            collector_identity.write_text(json.dumps(collector, indent=2) + "\n")
        else:
            collector = None
    processes = process_tree(collector_pid) if collector else []
    workers = [p for p in processes if p["argv"] and p["argv"][0] == str(binary) and p["state"] not in ("Z", "X")]
    for worker in workers:
        assert Path(f'/proc/{worker["pid"]}/exe').resolve() == binary
        worker["compute_lock"] = compute_lock(worker["pid"])
    stage = (root / "PROGRESS.txt").read_text().strip()
    tokens = stage.split()
    log = root / "results" / f"{tokens[1]}_{tokens[2]}.log" if len(tokens) >= 3 else None
    lines = log.read_text().splitlines() if log and log.exists() else []
    completed_images = sum(" rounds × " in line for line in lines)
    parallel_gate = "--parallel-gate" in sys.argv[2:]
    lock = compute_lock(collector_pid)
    lock_holders = [process_snapshot(pid) for pid in lock.get("holder_pids", [])]
    value = {
        "checked_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "host": (root / "measurement-host.txt").read_text().strip(),
        "binary_sha256": actual,
        "collector": collector,
        "compute_lock": lock,
        "lock_holders": lock_holders,
        "workers": workers,
        "stage_record": stage,
        "current_completed_image_groups": completed_images,
        "current_gate_observations": sum("[zenbench gate] clean=" in line for line in lines),
        "current_foreign_compiler_traces": sum("foreign pid=" in line and any(f'name="{name}"' in line for name in ("rustc", "cargo", "cc1plus", "cmake")) for line in lines),
        "raw_latency_outputs_present": len(list((root / "results").glob("*t_*_latency.json"))),
        "raw_rss_outputs_present": len(list((root / "results").glob("*t_*_rss-*.time.txt"))),
        "required_latency_outputs": 25 if parallel_gate else 8,
        "required_rss_outputs": 0 if parallel_gate else 24,
        "score_outputs_present": len(list((root / "results").glob("*t_*_scores.log"))),
        "required_score_outputs": 25 if parallel_gate else 0,
        "raw_collection_complete_marker": (root / "COMPLETE").exists(),
        "admission": "Final admission is owned by report.py; output presence is not qualification.",
    }
    (root / "LIVE_CHECK.json").write_text(json.dumps(value, indent=2) + "\n")
    checks = research_checks(task_root) if parallel_gate else None
    goldens = research_checks(task_root, "RESEARCH_GOLDEN") if parallel_gate else None
    retry = research_checks(task_root, "RESEARCH_RETRY") if parallel_gate else None
    public = research_checks(task_root, "PUBLIC") if parallel_gate else None
    if parallel_gate and task_root != root:
        task = json.loads((task_root / "PROGRESS.json").read_text())
        # Retire build-era process fields; their observations remain in the
        # supervision history and must not masquerade as current identities.
        for key in ("launcher", "compiler_pids", "verified_pid", "lock_holder_pid"):
            task.pop(key, None)
        phases = task.get("phases", {})
        phases[root.name] = {"checked_utc": value["checked_utc"],
                             "stage": stage, "binary_sha256": actual,
                             "collector": collector, "workers": workers,
                             "compute_lock": lock,
                             "raw_timing_outputs": value["raw_latency_outputs_present"]}
        task.update({"last_check": value["checked_utc"], "stage": f"{root.name}: {stage}",
                     "phases": phases,
                     "source": str(task_root / "source/a-speed-runtime"),
                     "binary_sha256": actual,
                     "raw_timing_outputs": sum(len(list((task_root / phase / "results").glob("*t_*_latency.json"))) for phase in ("before", "after")),
                     "required_timing_cells": 50,
                     "verified_workers": workers, "collector": collector,
                     "compute_lock": lock, "lock_holders": lock_holders,
                     "research_checks": checks,
                     "research_goldens": goldens,
                     "research_retry": retry, "public_checks": public,
                     "admission": value["admission"]})
        (task_root / "PROGRESS.json").write_text(json.dumps(task, indent=2) + "\n")
    with (root / "SUPERVISION.jsonl").open("a") as stream:
        stream.write(json.dumps(value) + "\n")
    print(json.dumps({
        "utc": value["checked_utc"], "host": value["host"],
        "stage": stage, "images": completed_images,
        "gate_observations": value["current_gate_observations"],
        "compiler_traces": value["current_foreign_compiler_traces"],
        "latency_outputs": value["raw_latency_outputs_present"],
        "rss_outputs": value["raw_rss_outputs_present"],
        "workers": [{"pid": p["pid"], "start_ticks": p["start_ticks"],
                     "lock_holders": p["compute_lock"].get("holder_pids"),
                     "owns_lock": p["compute_lock"]["held_by_worker_ancestor"]} for p in workers],
        "binary_verified": True,
        "collector_alive": collector is not None,
        "collection_complete_marker": value["raw_collection_complete_marker"],
        "lock_holder_pids": lock.get("holder_pids"),
        "research_checks": checks,
        "research_goldens": goldens,
        "research_retry": retry, "public_checks": public,
    }))


if __name__ == "__main__":
    main()
