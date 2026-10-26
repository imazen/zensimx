"""Read guarded continuation receipts and epoch markers; never read targets."""

import argparse
import json
from pathlib import Path
import re


def epoch(path):
    if not path.is_file():
        return None
    values = [int(x) for x in re.findall(r"subjective-rank epoch=(\d+)", path.read_text())]
    return max(values) if values else None


def status(study):
    state = json.loads((study / "RUN_STATE.json").read_text()) if (study / "RUN_STATE.json").is_file() else {"stage": "queued"}
    result = {key: value for key, value in state.items() if key != "command"}
    result["queue_pid"] = int((study / "QUEUE_PID").read_text())
    result["binary_frozen"] = (study / "BIN_FROZEN.json").is_file()
    for name in ("ZERO_STEP", "DEFAULT_PARITY"):
        path = study / f"{name}.json"
        if path.is_file():
            record = json.loads(path.read_text())
            result[name] = {key: record[key] for key in ("byte_identical", "rows", "prediction_sha256")}
    result["default_control_latest_epoch"] = epoch(study / "cells/MlicDefaultControl/without_kadid_s0/train.log")
    result["arms"] = {}
    for arm, epochs in (("MlicFT8x4", 4), ("MlicFT16x8", 8)):
        cells = list((study / "cells" / arm).glob("without_*_s*"))
        complete = sum((cell / "byc_record.json").is_file() for cell in cells)
        active = {cell.name: epoch(cell / "train.log") for cell in cells if not (cell / "byc_record.json").is_file()}
        result["arms"][arm] = dict(complete=complete, requested=12, epochs=epochs, active=active)
    for name in ("FAILED.json", "LAUNCH_FAILED.json"):
        path = study / name
        if path.is_file():
            result[name] = json.loads(path.read_text())
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    print(json.dumps(status(parser.parse_args().study), indent=2))
