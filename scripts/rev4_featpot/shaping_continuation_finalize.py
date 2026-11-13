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


def publish(root, phase, initial, phase_initial, proof_path, native, final_pin, phase_only=False):
    from shaping_completion import encoded, sha
    finished = datetime.fromtimestamp((native / "RESULT.json").stat().st_mtime, timezone.utc).isoformat()
    phase_result = dict(phase_initial, stage="trained", exit_code=0, finished_utc=finished,
                        bake_sha256=final_pin, continuation_sha256=sha(proof_path))
    controller = dict(initial, stage="trained", exit_code=0, finished_utc=finished,
                      bake_sha256=final_pin, phases=[phase_result], continuation_sha256=sha(proof_path))
    records = ((phase / "RESULT.json", phase_result, phase_initial),)
    if not phase_only:records += ((root / "RESULT.json", controller, initial),)
    for path, value, original in records:
        if not path.exists():
            continue
        old = json.loads(path.read_text())
        if old == value:
            continue
        if (old.get("stage") != "failed" or old.get("exit_code") == 0
                or any(old.get(k) != v for k, v in original.items() if k != "stage")):
            raise ValueError("preserve conflicting phase/controller result")
    for path, value, original in records:
        if path.exists():
            if json.loads(path.read_text()) == value:
                continue
            retained = path.with_name("RESULT.interrupted.json")
            if retained.exists():
                raise ValueError("preserve existing interrupted result receipt")
            path.rename(retained)
        with path.open("xb") as stream:
            stream.write(encoded(value))


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
    phase_name=cap.get("recovery_phase","fit")
    if phase_name not in ("fit","pretraining"):raise ValueError("registered recovery phase required")
    phase_only=phase_name=="pretraining"
    phase = root / phase_name
    original = capture.parent / "original-attempt"
    initial = json.loads((root / "RUN.json").read_text())
    if phase_only:
        schedule=initial.get("schedule",initial.get("resolution",{}).get("schedule",{}))
        if schedule.get("phase_epochs")!=[120,120] or (root/"fit").exists():
            raise ValueError("registered unfinished two-phase controller required")
    phase_initial = json.loads((phase / "RUN.json").read_text())
    if (sha(root / "RUN.json") != sha(original / "RUN.json")
            or sha(phase / "RUN.json") != sha(original / phase_name / "RUN.json")
            or phase_initial["parent_run_sha256"] != sha(root / "RUN.json")
            or initial["cell"] != cell or phase_initial["cell"] != cell
            or done["build_sha256"] != initial["build_sha256"]
            or done["build_sha256"] != phase_initial["build_sha256"]
            or initial["execution_host"] != done["source_host"]):
        raise ValueError("preserve original run/phase provenance")
    evidence_root=phase if phase_only else root
    proof_path = evidence_root / "CONTINUATION.json"
    if proof_path.exists():
        proof = json.loads(proof_path.read_text())
        if (proof["original_run_sha256"] != sha(root / "RUN.json")
                or proof["original_phase_sha256"] != sha(phase / "RUN.json")
                or proof["snapshot_sha256"] != done["snapshot_sha256"]
                or proof["completed_epochs_replayed"] not in (0, 1)):
            raise ValueError("partial continuation publication identity differs")
        for name, digest in proof["checkpoint_sha256"].items():
            if sha(phase / "checkpoints" / name) != digest:
                raise ValueError("preserve changed finalized checkpoint")
        for name, digest in proof["evidence_sha256"].items():
            if sha(root / name) != digest:
                raise ValueError("preserve changed continuation evidence")
        publish(root,phase,initial,phase_initial,proof_path,native,
                proof["checkpoint_sha256"][f"ckpt_epoch{phase_initial['epochs']-1:03}.bin"],phase_only)
        return dict(cell=cid,completed_epochs_replayed=proof["completed_epochs_replayed"],publication_reconciled=True,
                    continuation_sha256=sha(proof_path),epochs=initial["epochs"])
    for directory, original_identity in ((root, initial), (phase, phase_initial)):
        path = directory / "RESULT.json"
        if path.exists():
            old = json.loads(path.read_text())
            if (old.get("stage") != "failed" or old.get("exit_code") == 0
                    or any(old.get(k) != v for k,v in original_identity.items() if k != "stage")):
                raise ValueError("preserve conflicting interrupted result")
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
    epochs = phase_initial["epochs"]
    if epochs not in (120, 240, 480) or ((root / "pretraining").exists() and not phase_only):
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
    old_log = (original / phase_name / "train.log").read_text()
    new_log = (native / "train.log").read_text()
    old_epochs, new_epochs = epoch_rows(old_log), epoch_rows(new_log)
    prefix_epochs = set(range(state["epoch"]))
    tail = set(old_epochs) - prefix_epochs
    if (not prefix_epochs <= set(old_epochs) or tail not in (set(), {state["epoch"]})
            or set(new_epochs) != set(range(state["epoch"], epochs))):
        raise ValueError("continuation epoch coverage is incomplete or replayed")
    for e in tail:
        checkpoint = original / phase_name / "checkpoints" / f"ckpt_epoch{e:03}.bin"
        if (sha(checkpoint) != receipts[e]["sha256"]
                or checkpoint.stat().st_size != receipts[e]["bytes"]):
            raise ValueError("recomputed uncaptured tail checkpoint differs")

    postprocessed = f"Stamped {epochs} checkpoint dump(s) with complete training metadata" in new_log
    if postprocessed:
        binary = Path(done["command"][0])
        build = binary.parent / "BUILD.json"
        metadata = json.loads(build.read_text())
        if (sha(build) != done["build_sha256"]
                or metadata["binary_sha256"][binary.name] != sha(binary)):
            raise ValueError("native checkpoint postprocessor binary drift")
    for line in state["checkpoint_receipts"]:
        row = json.loads(line.removeprefix("  checkpoint dump: "))
        prefix = original / phase_name / "checkpoints" / f"ckpt_epoch{row['epoch']:03}.bin"
        if prefix.stat().st_size != row["bytes"] or sha(prefix) != row["sha256"]:
            raise ValueError("original optimizer snapshot checkpoint prefix drift")
    target = phase / "checkpoints"
    target.mkdir(exist_ok=True)
    final_pins = {}
    for e in sorted(expected):
        source = native / "checkpoints" / f"ckpt_epoch{e:03}.bin"
        row = receipts[e]
        final_pin = sha(source)
        if not postprocessed and (source.stat().st_size != row["bytes"] or final_pin != row["sha256"]):
            raise ValueError("native continued checkpoint drift")
        final_pins[e] = dict(row, bytes=source.stat().st_size, sha256=final_pin)
        dest = target / source.name
        if dest.exists() and sha(dest) != final_pin:
            # Native completion stamps metadata and applies the frozen
            # derived-input compaction, including to the relocated prefix.
            # Preserve the raw optimizer-snapshot prefix before replacing it.
            if e not in (prefix_epochs | tail) or sha(dest) != row["sha256"]:
                raise ValueError("preserve conflicting original checkpoint")
            raw_prefix = evidence_root / "continuation-evidence/raw-prefix" / dest.name
            raw_prefix.parent.mkdir(parents=True, exist_ok=True)
            if raw_prefix.exists():
                raise ValueError("preserve existing raw-prefix checkpoint")
            dest.rename(raw_prefix)
        if not dest.exists():
            if postprocessed and e in prefix_epochs:
                raw_source=original/phase_name/"checkpoints"/source.name
                if sha(raw_source)!=final_pin:
                    raw_prefix=evidence_root/"continuation-evidence/raw-prefix"/source.name
                    raw_prefix.parent.mkdir(parents=True,exist_ok=True)
                    if raw_prefix.exists():
                        if (raw_prefix.stat().st_size != raw_source.stat().st_size
                                or sha(raw_prefix) != sha(raw_source)):
                            raise ValueError("preserve different predecessor raw checkpoint")
                    else:
                        shutil.copy2(raw_source,raw_prefix)
            with source.open("rb") as src, dest.open("xb") as dst:
                shutil.copyfileobj(src, dst, 1024 * 1024)
                dst.flush()
                os.fsync(dst.fileno())
    if {p.name for p in target.glob("ckpt_epoch*.bin")} != {f"ckpt_epoch{e:03}.bin" for e in expected}:
        raise ValueError("canonical checkpoint coverage differs")
    evidence = evidence_root / "continuation-evidence"
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
                         (original / phase_name / "train.log", "original-train.log")):
        retain(source, evidence / name)
    retain(Path(selected["captured"]), evidence / state["identity"] / "source.snap")
    proof = dict(schema="exact-state-continuation-finalization-v1", cell=cell,
                 recovery_phase=phase_name,controller_complete=not phase_only,
                 original_run_sha256=sha(root / "RUN.json"), original_phase_sha256=sha(phase / "RUN.json"),
                 initial_execution_host=initial["execution_host"], resumed_execution_host=done["execution_host"],
                 snapshot_sha256=pin, resumed_at_epoch=state["epoch"], completed_epochs_replayed=len(tail),
                 committed_snapshot_epochs_replayed=0, recomputed_uncaptured_tail_epochs=sorted(tail),
                 uncaptured_tail_raw_sha256={str(e):receipts[e]["sha256"] for e in sorted(tail)},
                 checkpoint_sha256={f"ckpt_epoch{e:03}.bin": final_pins[e]["sha256"] for e in sorted(expected)},
                 native_checkpoint_postprocessing=postprocessed,
                 evidence_sha256={str(p.relative_to(root)): sha(p) for p in evidence.rglob("*") if p.is_file()},
                 finalized_utc=datetime.now(timezone.utc).isoformat(), driver_sha256=sha(__file__),
                 log_processing="Identical repeated native output lines collapsed; both raw logs retained.")
    evidence_root=phase if phase_only else root
    proof_path = evidence_root / "CONTINUATION.json"
    if proof_path.exists():
        raise ValueError("preserve completed or partially published continuation receipt")
    canonical = []
    seen = set()
    for is_original, text in ((True, old_log), (False, new_log)):
        for line in text.splitlines():
            if is_original and tail:
                match = re.match(r"^\s*epoch\s+(\d+)\s+\|",line)
                diagnostic = line.startswith(("TRAINING_DIAGNOSTICS ", "FIRST_WEIGHT_DIAGNOSTICS "))
                if match and int(match[1]) in tail:
                    continue
                if diagnostic and json.loads(line.split(" ",1)[1]).get("epoch") in tail:
                    continue
            if line.startswith("  checkpoint dump: "):
                continue
            if line in seen:
                continue
            seen.add(line)
            canonical.append(line)
    for e in sorted(expected):
        row = dict(final_pins[e], path=str(target / f"ckpt_epoch{e:03}.bin"))
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
    publish(root,phase,initial,phase_initial,proof_path,native,final_pins[epochs-1]["sha256"],phase_only)
    return dict(cell=cid, initial_execution_host=initial["execution_host"],
                resumed_execution_host=done["execution_host"], completed_epochs_replayed=len(tail),
                continuation_sha256=sha(proof_path), epochs=epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "capture", "native"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(finalize(args.study, args.capture, args.native)))
