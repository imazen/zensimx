"""Supervise existing label-free gate owners for the certified TRAIN winner.

Caller holds heavy.lock/run-heavy. Completion means measurements exist, not
that qualification passed. No EVAL labels, timing benchmark or recipe changes.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import traceback

from aic_finetune_status import status


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".new")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "bin-dir", "instruments", "corpus-root"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    study = args.study.resolve()
    output = study / "gates"
    output.mkdir()  # Refuse a previous attempt, including partial evidence.
    owner = Path(__file__).resolve().parents[1] / "byconstruction/gates.py"
    def state(stage):
        write(study / "QUALIFY_RUN_STATE.json", dict(stage=stage, pid=os.getpid(),
            utc=datetime.now(timezone.utc).isoformat(), runner_sha256=sha(Path(__file__))))
    stop = threading.Event()
    def progress():
        while not stop.wait(15):
            write(study / "PROGRESS.json", status(study))
    monitor = threading.Thread(target=progress, daemon=True)
    try:
        state("label-free-gate-preflight")
        write(study / "PROGRESS.json", status(study))
        monitor.start()
        full = json.loads((study / "FULL_COMPLETE.json").read_text())
        if len(full["seeds"]) != 3 or {r["seed"] for r in full["seeds"]} != {0, 1, 2}:
            raise ValueError("three certified full-data seeds required")
        for row in full["seeds"]:
            if sha(Path(row["bake"])) != row["sha256"] or not row["certificate"]["certified"] or not row["certificate"]["monotone"]:
                raise ValueError("full-data bake/certificate changed")
        pins = {}
        for line in (args.bin_dir / "SHA256SUMS").read_text().splitlines():
            digest, name = line.split()
            if sha(args.bin_dir / name) != digest:
                raise ValueError("gate tool differs from its existing manifest")
            pins[name] = digest
        model = next(row["bake"] for row in full["seeds"] if row["seed"] == 0)
        name = full["arm"].lower()
        command = [sys.executable, "-B", str(owner), "--name", name, "--model", model,
            "--out", str(output), "--bin-dir", str(args.bin_dir), "--instruments", str(args.instruments),
            "--corpus-root", str(args.corpus_root), "--formula-rev", "7"]
        write(output / "INPUTS.json", dict(command=command, tools=pins,
            full_complete_sha256=sha(study / "FULL_COMPLETE.json"), owner_sha256=sha(owner),
            model_sha256=sha(Path(model)), seed=0, eval="unread"))
        for step in ("nearid", "verdict", "steer", "identity"):
            state("label-free-" + step)
            with (output / (step + ".log")).open("x") as log:
                subprocess.run(command + ["--steps", step], check=True, stdout=log, stderr=subprocess.STDOUT)
            write(output / (step + ".complete.json"), dict(step=step, exit_code=0))
        for tool, digest in pins.items():
            if sha(args.bin_dir / tool) != digest:
                raise ValueError("gate tool changed during measurement")
        with (output / "SUMMARY.json").open("x") as stream:
            subprocess.run([sys.executable, "-B", str(owner), "--summary", "--out", str(output), "--name", name],
                stdout=stream, check=True)
        write(output / "COMPLETE.json", dict(qualification="requires assessment", eval="unread",
            summary_sha256=sha(output / "SUMMARY.json"), inputs_sha256=sha(output / "INPUTS.json")))
        state("label-free-gates-complete")
        return 0
    except BaseException as error:
        write(output / "FAILED.json", dict(error=str(error), traceback=traceback.format_exc(),
            utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid()))
        state("label-free-gates-failed")
        raise
    finally:
        stop.set()
        if monitor.is_alive():
            monitor.join()
        write(study / "PROGRESS.json", status(study))


if __name__ == "__main__":
    raise SystemExit(main())
