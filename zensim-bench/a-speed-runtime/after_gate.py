#!/usr/bin/env python3
"""Start the after matrix only after complete baseline admission and build."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from report import parallel_gate


def main():
    root = Path(sys.argv[1]).resolve()
    receipt = root / "AFTER_CONTROLLER.json"
    assert not receipt.exists(), receipt
    state = {"launcher_pid": os.getpid(), "stage": "waiting for baseline and gated binary",
             "failure": None}

    def save():
        state["checked_utc"] = datetime.datetime.now(datetime.UTC).isoformat()
        receipt.write_text(json.dumps(state, indent=2) + "\n")

    save()
    try:
        while not ((root / "before/COMPLETE").exists() and (root / "after/BINARY.json").exists()):
            time.sleep(10)
        state["stage"] = "admitting baseline"
        save()
        baseline = parallel_gate(root, "before")
        assert baseline["parallel_min_pixels"] == 36_864, baseline["crossovers"]
        path = root / "SUMMARY.before.json"
        assert not path.exists(), path
        path.write_text(json.dumps(baseline, indent=2) + "\n")
        state["stage"] = "after matrix collecting"
        save()
        env = dict(os.environ, A_SPEED_PARALLEL_GATE="1", BENCH_CPUS="0-15", BENCH_MODES="latency scores")
        common = "64 96 128 160 192 256 320 384 512 640 768 1024"
        for threads, sizes in (("16 4", common), ("16", "4096")):
            env.update(BENCH_THREADS=threads, BENCH_SIZES=sizes)
            with (root / "after-collector.log").open("a") as log:
                subprocess.run(["bash", str(root / "source/a-speed-runtime/run.sh"),
                                str(root / "after"), str(root / "after/bin/a-speed-peers")],
                               env=env, stdout=log, stderr=log, check=True)
        state["stage"] = "admitting full matrix"
        save()
        result = parallel_gate(root, "after")
        path = root / "SUMMARY.after.json"
        assert not path.exists(), path
        path.write_text(json.dumps(result, indent=2) + "\n")
        state.update(stage="full matrix admitted", cells=len(result["rows"]),
                     score_comparisons=result["score_comparisons"], score_differences=result["score_differences"])
        save()
    except BaseException as error:
        state["failure"] = repr(error)
        save()
        raise


if __name__ == "__main__":
    main()
