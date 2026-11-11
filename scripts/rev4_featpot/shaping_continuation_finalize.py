"""Finalize successful full-state continuations without executing a trainer."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import socket
import sys


def finalize(study, capture, native):
    from shaping_completion import encoded, sha
    from shaping_state_probe import read_state
    cap = json.loads(capture.read_text())
    done = json.loads((native / "RESULT.json").read_text())
    cell = done["cell"]
    cid = cell["id"]
    if (done["stage"] != "native_complete_pending_scientific_collection"
            or done["exit_code"] != 0 or done["completed_epochs_replayed"] != 0
            or done["execution_host"] != socket.gethostname()
            or done["execution_host"] not in ("i134", "r3500", "r3800x", "r5600g")
            or done["source_capture_sha256"] != sha(capture)
            or not cap["exact_state_checksum_verified"]):
        raise ValueError("successful owned exact-state continuation required")
    selected = cap["selected"]
    generation, state, pin, raw = read_state(Path(selected["captured"]))
    if (pin != selected["sha256"] or pin != done["snapshot_sha256"]
            or state["epoch"] != selected["epoch"] or state["epoch"] != done["source_epoch"]
            or state["pair"] != 0 or done["source_pair"] != 0):
        raise ValueError("epoch-boundary exact-state source differs")
    root = study / "runs" / cid / "attempt-001"
    phase = root / "fit"
    original = capture.parent / "original-attempt"
    initial = json.loads((root / "RUN.json").read_text())
    phase_initial = json.loads((phase / "RUN.json").read_text())
    if (sha(root / "RUN.json") != sha(original / "RUN.json")
            or sha(phase / "RUN.json") != sha(original / "fit/RUN.json")
            or phase_initial["parent_run_sha256"] != sha(root / "RUN.json")
            or initial["cell"] != cell or phase_initial["cell"] != cell
            or done["build_sha256"] != initial["build_sha256"]
            or done["build_sha256"] != phase_initial["build_sha256"]
            or initial["execution_host"] != done["source_host"]):
        raise ValueError("preserve original run/phase provenance")
    command = list(cap["native"]["command"])
    if command != phase_initial["command"][phase_initial["command"].index("--") + 1:]:
        raise ValueError("captured native command differs")
    for option in ("--out", "--dump-checkpoints-dir"):
        command[command.index(option) + 1] = done["command"][done["command"].index(option) + 1]
    if command != done["command"]:
        raise ValueError("continuation changed a scientific argument")
    for name, digest in phase_initial["manifest_sha256"].items():
        if sha(name) != digest:
            raise ValueError("continuation optimizer/head/shaping manifest drift")
    epochs = initial["epochs"]
    if epochs not in (120, 240, 480) or (root / "pretraining").exists():
        raise ValueError("one registered native continuation phase required")
    expected = set(range(epochs))
    receipts = {}
    for line in (native / "train.log").read_text().splitlines():
        if line.startswith("  checkpoint dump: "):
            row = json.loads(line.removeprefix("  checkpoint dump: "))
            e = row["epoch"]
            if e in receipts and receipts[e] != row:
                raise ValueError("conflicting continuation checkpoint receipts")
            receipts[e] = row
    if set(receipts) != expected:
        raise ValueError("complete native checkpoint receipt coverage required")
    for line in state["checkpoint_receipts"]:
        row = json.loads(line.removeprefix("  checkpoint dump: "))
        after = receipts[row["epoch"]]
        if any(row[k] != after[k] for k in ("epoch", "bytes", "sha256")):
            raise ValueError("continued checkpoint prefix differs from optimizer snapshot")
    if len(state["checkpoint_receipts"]) != state["epoch"]:
        raise ValueError("complete snapshot prefix required")
    def epoch_rows(text):
        rows = {}
        for line in text.splitlines():
            match = re.match(r"^\s*epoch\s+(\d+)\s+\|", line)
            if match:
                e = int(match[1])
                if e in rows and rows[e] != line:
                    raise ValueError("conflicting epoch diagnostics")
                rows[e] = line
        return rows
    old_log = (original / "fit/train.log").read_text()
    new_log = (native / "train.log").read_text()
    old_epochs, new_epochs = epoch_rows(old_log), epoch_rows(new_log)
    if (set(old_epochs) != set(range(state["epoch"]))
            or set(new_epochs) != set(range(state["epoch"], epochs))):
        raise ValueError("continuation epoch coverage is incomplete or replayed")
    target = phase / "checkpoints"
    target.mkdir(exist_ok=True)
    for e in sorted(expected):
        source = native / "checkpoints" / f"ckpt_epoch{e:03}.bin"
        row = receipts[e]
        if source.stat().st_size != row["bytes"] or sha(source) != row["sha256"]:
            raise ValueError("native continued checkpoint drift")
        dest = target / source.name
        if dest.exists():
            if sha(dest) != row["sha256"]:
                raise ValueError("preserve conflicting original checkpoint")
        else:
            with source.open("rb") as src, dest.open("xb") as dst:
                shutil.copyfileobj(src, dst, 1024 * 1024)
                dst.flush()
                os.fsync(dst.fileno())
    if {p.name for p in target.glob("ckpt_epoch*.bin")} != {f"ckpt_epoch{e:03}.bin" for e in expected}:
        raise ValueError("canonical checkpoint coverage differs")
    evidence = root / "continuation-evidence"
    evidence.mkdir(exist_ok=True)
    def retain(source, dest):
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if sha(dest) != sha(source):
                raise ValueError("preserve different continuation evidence")
        else:
            with source.open("rb") as src, dest.open("xb") as dst:
                shutil.copyfileobj(src, dst, 1024 * 1024)
    for source, name in ((capture, "CAPTURE.json"), (native / "RESULT.json", "NATIVE_RESULT.json"),
                         (native / "train.log", "native-train.log"),
                         (original / "fit/train.log", "original-train.log")):
        retain(source, evidence / name)
    retain(Path(selected["captured"]), evidence / state["identity"] / "source.snap")
    proof = dict(schema="exact-state-continuation-finalization-v1", cell=cell,
                 original_run_sha256=sha(root / "RUN.json"), original_phase_sha256=sha(phase / "RUN.json"),
                 initial_execution_host=initial["execution_host"], resumed_execution_host=done["execution_host"],
                 snapshot_sha256=pin, resumed_at_epoch=state["epoch"], completed_epochs_replayed=0,
                 checkpoint_sha256={f"ckpt_epoch{e:03}.bin": receipts[e]["sha256"] for e in sorted(expected)},
                 evidence_sha256={str(p.relative_to(root)): sha(p) for p in evidence.rglob("*") if p.is_file()},
                 finalized_utc=datetime.now(timezone.utc).isoformat(), driver_sha256=sha(__file__),
                 log_processing="Identical repeated native output lines collapsed; both raw logs retained.")
    proof_path = root / "CONTINUATION.json"
    if proof_path.exists():
        raise ValueError("preserve completed or partially published continuation receipt")
    canonical = []
    seen = set()
    for text in (old_log, new_log):
        for line in text.splitlines():
            if line.startswith("  checkpoint dump: "):
                continue
            if line in seen:
                continue
            seen.add(line)
            canonical.append(line)
    for e in sorted(expected):
        row = dict(receipts[e], path=str(target / f"ckpt_epoch{e:03}.bin"))
        canonical.append("  checkpoint dump: " + json.dumps(row, separators=(",", ":")))
    log = phase / "train.log"
    if log.exists():
        log.rename(phase / "train.before-continuation.log")
    with log.open("x") as stream:
        stream.write("\n".join(canonical) + "\n")
    for name in ("best.bin", "best.bin.spec.json"):
        source = native / name
        dest = phase / name
        if dest.exists():
            dest.rename(phase / (name + ".before-continuation"))
        retain(source, dest)
    with proof_path.open("xb") as stream:
        stream.write(encoded(proof))
    finished = datetime.fromtimestamp((native / "RESULT.json").stat().st_mtime, timezone.utc).isoformat()
    phase_result = dict(phase_initial, stage="trained", exit_code=0, finished_utc=finished,
                        bake_sha256=receipts[epochs - 1]["sha256"], continuation_sha256=sha(proof_path))
    controller = dict(initial, stage="trained", exit_code=0, finished_utc=finished,
                      bake_sha256=phase_result["bake_sha256"], phases=[phase_result],
                      continuation_sha256=sha(proof_path))
    for path, value in ((phase / "RESULT.json", phase_result), (root / "RESULT.json", controller)):
        with path.open("xb") as stream:
            stream.write(encoded(value))
    return dict(cell=cid, initial_execution_host=initial["execution_host"],
                resumed_execution_host=done["execution_host"], completed_epochs_replayed=0,
                continuation_sha256=sha(proof_path), epochs=epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "capture", "native"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(finalize(args.study, args.capture, args.native)))
