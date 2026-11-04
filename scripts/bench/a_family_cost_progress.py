#!/usr/bin/env python3
"""Resolve live study processes and artifact coverage; save a progress receipt."""
import argparse
import datetime
import json
import os
from pathlib import Path
import socket
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("study", type=Path)
    a = p.parse_args()
    root = a.study.resolve()
    processes = []
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            name = (proc / "comm").read_text().strip()
            if name not in ["a_family_cost", "perf"]:
                continue
            cmd = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode()
            if str(root) in cmd:
                processes.append(dict(pid=int(proc.name), name=name, command=cmd.strip()))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            # Processes may exit during enumeration. This is a live census,
            # never a missing-fixture skip or a completed-job assertion.
            continue
    lock_rows = json.loads(subprocess.check_output(
        ["lslocks", "-J", "-o", "PID,COMMAND,MODE,PATH,BLOCKER"], text=True))["locks"]
    lock_path = str(Path.home() / "tmp/heavy.lock")
    locks = [r for r in lock_rows if r.get("path") == lock_path]
    cells = []
    for attempt in [root, *sorted(root.glob("retry[0-9]*"))]:
        for threads in [1, 16]:
            for size in ["64", "256", "1024", "large"]:
                files = sorted((attempt / f"timing/{size}-{threads}t").glob(f"{size}_*.json"))
                if not files:
                    continue
                clean_counts = []
                acceptable = 0
                for f in files:
                    row = json.loads(f.read_text())
                    n = sum(s["gate_clean"] is True for s in row["comparisons"][0]["samples"])
                    clean_counts.append(n)
                    acceptable += n >= 16 and not row["unreliable"]
                cells.append(dict(attempt=str(attempt.relative_to(root)), size=size, threads=threads,
                                  images=len(files), acceptable_images=acceptable,
                                  clean_rounds=sum(clean_counts), min_clean_rounds=min(clean_counts)))
    profiles = []
    for size in ["64", "256", "1024", "large"]:
        log = root / f"logs/perf-{size}.log"
        text = log.read_text() if log.exists() else ""
        samples = root / f"perf/{size}/samples.txt"
        profiles.append(dict(size=size, capture_complete="run-heavy: done rc=0" in text,
                             parsed_samples_ready=samples.exists() and samples.stat().st_size > 0))
    result = dict(last_check=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  host=socket.gethostname(), study=str(root), processes=processes,
                  heavy_lock=locks, cells=cells, required_cells=8, profiles=profiles,
                  required_profiles=4)
    target = root / "PROGRESS.json"
    staging = target.with_suffix(".next")
    staging.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(staging, target)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
