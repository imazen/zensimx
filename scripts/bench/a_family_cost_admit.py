#!/usr/bin/env python3
"""Reject a shared-host cell if a foreign build/cloud/profiler process appears."""
import argparse
import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading

NAMES = {"rustc", "cargo", "aws", "perf"}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def descendant(pid, owner):
    while pid > 1:
        if pid == owner:
            return True
        try:
            pid = int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[1])
        except (FileNotFoundError, ProcessLookupError):
            return False
    return False


def census(owner):
    result = []
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            name = (proc / "comm").read_text().strip()
            if name in NAMES and not descendant(int(proc.name), owner):
                result.append(dict(pid=int(proc.name), name=name))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("receipt", type=Path)
    p.add_argument("--images", type=Path)
    p.add_argument("--size")
    p.add_argument("command", nargs=argparse.REMAINDER)
    a = p.parse_args()
    command = a.command[1:] if a.command[:1] == ["--"] else a.command
    assert command
    assert not a.receipt.exists(), "preserve the previous admission receipt"
    a.receipt.parent.mkdir(parents=True, exist_ok=True)
    owner = os.getpid()
    before = census(owner)
    result = dict(start=now(), before=before, during=[], after=[], command=command,
                  policy="reject any foreign rustc/cargo/aws/perf; no waits; 0.5s census plus zenbench trace")
    observed = {}
    windows = {}
    active = [None]
    case_ids = ([r["id"] for r in json.loads(a.images.read_text()) if r["size"] == a.size]
                if a.images else [])
    next_case = [0]
    done = threading.Event()

    def watch():
        while not done.wait(.5):
            for row in census(owner):
                observed.setdefault((row["pid"], row["name"]), dict(row, first_seen=now(), source="census"))
                case = active[0]
                if case is not None:
                    windows[case]["during"].setdefault(str(row["pid"]), dict(row, seen=now()))

    if before:
        result.update(status="not measured (contended)", end=now(), exit_code=None, after=census(owner))
        a.receipt.write_text(json.dumps(result, indent=2) + "\n")
        print("not measured (contended): foreign process present before cell", before, flush=True)
        return
    env = os.environ.copy()
    env["ZENBENCH_LAUNCHER_PIDS"] = str(owner)
    env["ZENBENCH_GATE_TRACE"] = "1"
    worker = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    thread = threading.Thread(target=watch)
    thread.start()
    for line in worker.stdout:
        print(line, end="", flush=True)
        if case_ids and line.startswith("[zenbench] timer resolution:"):
            case = case_ids[next_case[0]]
            next_case[0] += 1
            windows[case] = dict(start=now(), before=census(owner), during={}, after=[])
            active[0] = case
        m = re.search(r'foreign pid=(\d+) name="(rustc|cargo|aws|perf)"', line)
        if m:
            key = (int(m[1]), m[2])
            observed.setdefault(key, dict(pid=key[0], name=key[1], first_seen=now(), source="zenbench"))
            if active[0] is not None:
                windows[active[0]]["during"].setdefault(str(key[0]), dict(pid=key[0], name=key[1], seen=now()))
        finished = re.fullmatch(r"completed (\S+) (\d+)t\s*", line)
        if finished and case_ids:
            case = finished[1]
            assert active[0] == case
            window = windows[case]
            window.update(after=census(owner), end=now())
            window["status"] = "accepted" if not window["before"] and not window["during"] and not window["after"] else "not measured (contended)"
            active[0] = None
    rc = worker.wait()
    done.set()
    thread.join()
    result.update(during=list(observed.values()), after=census(owner), end=now(), exit_code=rc)
    result["status"] = "accepted" if not observed and not result["after"] and rc == 0 else "not measured (contended)"
    if case_ids:
        assert len(windows) == len(case_ids) or rc != 0
        for window in windows.values():
            window["status"] = "accepted" if not window["before"] and not window["during"] and not window["after"] else "not measured (contended)"
        result["cells"] = windows
        result["status"] = "completed with image admission" if rc == 0 else "failed"
    if rc != 0:
        result["status"] = "failed"
    a.receipt.write_text(json.dumps(result, indent=2) + "\n")
    print("CELL ADMISSION:", result["status"], flush=True)
    if rc:
        raise SystemExit(rc if rc > 0 else 128 - rc)


if __name__ == "__main__":
    main()
