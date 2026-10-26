"""Read guarded continuation receipts and epoch markers; never read targets."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import re


def epoch(path):
    if not path.is_file():
        return None
    values = [int(x) for x in re.findall(r"subjective-rank epoch=(\d+)", path.read_text())]
    return max(values) if values else None


def status(study):
    state = json.loads((study / "RUN_STATE.json").read_text()) if (study / "RUN_STATE.json").is_file() else {"stage": "queued"}
    full_state = study / "FULL_RUN_STATE.json"
    if full_state.is_file():
        state.update(json.loads(full_state.read_text()))
    qualify_state = study / "QUALIFY_RUN_STATE.json"
    if qualify_state.is_file():
        state.update(json.loads(qualify_state.read_text()))
    eval_state = study / "EVAL_RUN_STATE.json"
    if eval_state.is_file():
        state.update(json.loads(eval_state.read_text()))
    result = {key: value for key, value in state.items() if key != "command"}
    result["checked_utc"] = datetime.now(timezone.utc).isoformat()
    result["study"] = str(study.resolve())
    queue = study / "QUEUE_PID"
    result["queue_pid"] = int(queue.read_text()) if queue.is_file() else None
    pid = state.get("pid")
    result["worker_alive"] = False
    if isinstance(pid, int) and pid > 0:
        try:
            command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            result["worker_alive"] = str(study.resolve()).encode() in command
        except OSError:
            pass
    result["binary_frozen"] = (study / "BIN_FROZEN.json").is_file()
    for name in ("ZERO_STEP", "DEFAULT_PARITY"):
        path = study / f"{name}.json"
        if path.is_file():
            record = json.loads(path.read_text())
            result[name] = {key: record[key] for key in ("byte_identical", "rows", "prediction_sha256")}
    result["default_control_latest_epoch"] = epoch(study / "cells/MlicDefaultControl/without_kadid_s0/train.log")
    result["arms"] = {}
    for arm, epochs in (("MlicFT8x4", 4), ("MlicFT16x8", 8)):
        cells = [p for p in (study / "cells" / arm).glob("without_*_s*") if p.is_dir()]
        complete = sum((cell / "byc_record.json").is_file() for cell in cells)
        active = {cell.name: epoch(cell / "train.log") for cell in cells if not (cell / "byc_record.json").is_file()}
        result["arms"][arm] = dict(complete=complete, requested=12, epochs=epochs, active=active)
    for name in ("FAILED.json", "LAUNCH_FAILED.json"):
        path = study / name
        if path.is_file():
            result[name] = json.loads(path.read_text())
    failed = any(name in result for name in ("FAILED.json", "LAUNCH_FAILED.json"))
    completed = sum(arm["complete"] for arm in result["arms"].values())
    result["completed_cells"] = completed
    result["required_cells"] = 24
    result["status"] = ("failed" if failed else "train-evaluation-complete" if
        (study / "EVALUATION_COMPLETE.json").is_file() and completed == 24 else
        "running" if result["worker_alive"] else "training-complete" if
        (study / "TRAIN_COMPLETE.json").is_file() and completed == 24 else "not-running")
    if full_state.is_file():
        cells = [p for p in (study / "full").glob("*/full_s*") if p.is_dir()]
        failures = list((study / "full").glob("*/FULL_FAILED.json"))
        if (study / "FULL_LAUNCH_FAILED.json").is_file():
            failures.append(study / "FULL_LAUNCH_FAILED.json")
        result["full"] = dict(complete=sum((p / "FULL_CELL_VERIFIED.json").is_file() for p in cells),
            required=3, active={str(p.relative_to(study / "full")): epoch(p / "train.log")
                for p in cells if not (p / "FULL_CELL_VERIFIED.json").is_file()},
            failures={str(p.relative_to(study)): json.loads(p.read_text()) for p in failures})
        result["status"] = ("failed" if failed or failures else "full-data-certified" if
            (study / "FULL_COMPLETE.json").is_file() and result["full"]["complete"] == 3 else
            "no-finalist" if (study / "NO_FINALIST.json").is_file() else
            "running" if result["worker_alive"] else "not-running")
    if qualify_state.is_file():
        gates = study / "gates"
        required = {"nearid": "nearid-s0/*.GATES.json", "standard": "verdict-s0-*/standard/gaddr.json",
            "ladder": "verdict-s0-*/ladder/gaddr.json", "steer": "steer135/*/result.json", "identity": "identity-s0.jsonl"}
        result["gates"] = dict(complete=[name for name, pattern in required.items() if list(gates.glob(pattern)) and
            (gates / (("verdict" if name in ("standard", "ladder") else name) + ".complete.json")).is_file()],
            required=list(required), failure=json.loads((gates / "FAILED.json").read_text()) if (gates / "FAILED.json").is_file() else None)
        result["status"] = ("failed" if failed or result["gates"]["failure"] else
            "label-free-gates-complete" if (gates / "COMPLETE.json").is_file() and
            len(result["gates"]["complete"]) == len(required) else "running" if result["worker_alive"] else "not-running")
    if eval_state.is_file():
        assessment = Path(state.get("assessment_directory", study / "assessment"))
        result["assessment"] = dict(complete=[name for name in ("main", "bapps") if
            (assessment / name / "COMPLETE.json").is_file()], required=["main", "bapps"],
            failure=json.loads((assessment / "FAILED.json").read_text()) if (assessment / "FAILED.json").is_file() else None)
        result["status"] = ("failed" if failed or result["assessment"]["failure"] else
            "coordinator-EVAL-complete" if (assessment / "COMPLETE.json").is_file() and
            len(result["assessment"]["complete"]) == 2 else "running" if result["worker_alive"] else "not-running")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--output", type=Path, help="atomically update a durable progress record")
    args = parser.parse_args()
    value = status(args.study)
    text = json.dumps(value, indent=2) + "\n"
    if args.output:
        temporary = args.output.with_suffix(args.output.suffix + ".new")
        temporary.write_text(text)
        temporary.replace(args.output)
    print(text, end="")
