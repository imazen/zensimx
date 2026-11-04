#!/usr/bin/env python3
"""Reclaim this instrument's inactive Cargo caches, retaining release controls.

Run under the shared heavy-work lock and run-heavy. Fixtures, reports, source
snapshots and frozen benchmark binaries live outside these two Cargo targets.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/rev4_featpot"))
from run_processes import compute_lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("label", help="unique preservation receipt directory")
    args = parser.parse_args()
    root = args.root.resolve()
    label = args.label
    assert label and Path(label).name == label and label not in (".", "..")
    assert compute_lock(os.getpid())["held_by_worker_ancestor"], "take the shared compute lock"
    targets = [root / name for name in ("target", "target-tests")]
    for target in targets:
        assert target.is_dir() and target.resolve() == target, target
        assert (target / "CACHEDIR.TAG").read_text().startswith(
            "Signature: 8a477f597d28d172789f06886806bc55"
        ), (target, "not a Cargo cache")
    prefixes = tuple(str(p) + "/" for p in targets)
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            assert not any(p in (proc / "maps").read_text() for p in prefixes), (
                proc.name, "cache is mapped by a live process"
            )
            for fd in (proc / "fd").iterdir():
                assert not os.readlink(fd).startswith(prefixes), (proc.name, "cache file is open")
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue  # System processes and exits during the snapshot.
    preserved = root / "preserved-cargo" / label
    assert not preserved.exists(), "never overwrite a previous preservation receipt"
    preserved.mkdir(parents=True)
    copies = []
    candidates = [*(targets[0] / "release").glob("*"),
                  *(targets[0] / "release/examples").glob("*")]
    for source in candidates:
        if not source.is_file():
            continue
        with source.open("rb") as stream:
            if stream.read(4) != b"\x7fELF":
                continue
        destination = preserved / source.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        assert hashlib.sha256(destination.read_bytes()).hexdigest() == digest
        copies.append({"source": str(source.relative_to(root)),
                       "preserved": str(destination.relative_to(root)), "sha256": digest})
    receipt = {"utc": datetime.datetime.now(datetime.UTC).isoformat(),
               "targets": [str(p) for p in targets], "preserved": copies,
               "free_bytes_before": shutil.disk_usage(root).free}
    (preserved / "PRESERVED.json").write_text(json.dumps(receipt, indent=2) + "\n")
    for target in targets:
        shutil.rmtree(target)
    receipt["free_bytes_after"] = shutil.disk_usage(root).free
    (preserved / "RECLAIMED.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
