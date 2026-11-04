#!/usr/bin/env python3
"""Complete primary + at most two retry passes; never wait for foreign jobs."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess

CELLS = [(size, threads) for size in ["64", "256", "1024", "large"] for threads in [1, 16]]
CRITICAL = re.compile(r'foreign pid=\d+ name="(rustc|cargo|aws|perf)"')


def cell_status(root, size, threads, epoch):
    folder = root / f"timing/{size}-{threads}t"
    receipt = folder / "admission.json"
    image_admission = {}
    if receipt.exists():
        admission = json.loads(receipt.read_text())
        if "cells" in admission:
            image_admission = {key: value["status"] == "accepted" for key, value in admission["cells"].items()}
            mode = "before/during/after process census and zenbench trace"
        elif admission["status"] == "accepted":
            mode = "whole-cohort process census and zenbench trace"
        else:
            return {}, admission["status"]
    else:
        log = root / f"logs/bench-{size}-{threads}t.log"
        text = log.read_text() if log.exists() else ""
        if "run-heavy: done rc=0" not in text:
            return {}, "not measured (contended or incomplete)"
        for chunk in text.split("[zenbench] timer resolution:")[1:]:
            finished = re.search(r"completed (\S+) (\d+)t", chunk)
            if finished:
                image_admission[finished[1]] = CRITICAL.search(chunk) is None
        mode = "legacy per-image zenbench trace; direct census unrecorded"
    files = sorted(folder.glob(f"{size}_*.json"))
    if len(files) != 20:
        return {}, "not measured (incomplete)"
    valid = {}
    for f in files:
        row = json.loads(f.read_text())
        if row["timestamp"] < epoch or row["unreliable"] or not image_admission.get(f.stem, not image_admission):
            continue
        samples = row["comparisons"][0]["samples"]
        assert len(samples) == 32
        assert all(s["iterations"] == 1 and s["gate_clean"] is not None for s in samples)
        clean = sum(s["gate_clean"] for s in samples)
        valid[f.stem] = dict(path=str(f), clean=clean, admission_mode=mode)
    return valid, mode


def profile_valid(root, size):
    folder = root / f"perf/{size}"
    receipt = folder / "admission.json"
    script = folder / "script-receipt.log"
    return (receipt.exists() and json.loads(receipt.read_text())["status"] == "accepted"
            and script.exists() and "run-heavy: done rc=0" in script.read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("study", type=Path)
    a = p.parse_args()
    root = a.study.resolve()
    epoch = (root / "QUIET_EPOCH.txt").read_text().strip()
    runner = root / "a_family_cost.sh"
    attempts = [root, root / "retry01", root / "retry02"]
    for index, attempt in enumerate(attempts):
        attempt.mkdir(exist_ok=True)
        if attempt != root:
            for name in ["binaries", "build-source", "pairs", "input", "production-f16.bin", "inputs-A0.tsv"]:
                path = attempt / name
                if not path.exists():
                    path.symlink_to(root / name)
        print(f"PASS {index} of primary + two retries", flush=True)
        for size, threads in CELLS:
            valid = {}
            for existing in attempts[:index + 1]:
                values, _ = cell_status(existing, size, threads, epoch)
                for key, value in values.items():
                    valid[key] = valid.get(key, 0) + value["clean"]
            folder = attempt / f"timing/{size}-{threads}t"
            if sum(n >= 16 for n in valid.values()) == 20:
                print(size, threads, "20/20 admitted; no repeat", flush=True)
            elif folder.exists():
                print(size, threads, "existing attempt retained; admitted", sum(n >= 16 for n in valid.values()), "/20", flush=True)
            else:
                env = os.environ.copy()
                env["A_COST_CPUS"] = "2" if threads == 1 else "0-15"
                subprocess.run(["bash", str(runner), "bench", str(attempt), str(threads), size], env=env, check=True)
        for size, repeats in [("64", "1500"), ("256", "400"), ("1024", "30"), ("large", "3")]:
            if any(profile_valid(existing, size) for existing in attempts[:index + 1]):
                print(size, "profile admitted; no repeat", flush=True)
                continue
            folder = attempt / f"perf/{size}"
            if folder.exists():
                print(size, "existing profile attempt retained", flush=True)
                continue
            env = os.environ.copy()
            env["A_COST_CPUS"] = "2"
            subprocess.run(["bash", str(runner), "profile", str(attempt), size, repeats], env=env, check=True)
    print("BOUNDED PASSES COMPLETE; missing observations remain not measured (contended)", flush=True)


if __name__ == "__main__":
    main()
