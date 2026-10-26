"""Replay one frozen Adam engineering control with fresh outputs and perf counters.

The original trainer argv, data authority and seed streams remain the owners of
the recipe. This wrapper changes only the executable and output locations. It
retains every checkpoint and records original/actual argv and executable hashes.
Caller supplies resource limits, CPU affinity and the appropriate perf events.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--argv-json", type=Path, required=True)
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--events", required=True)
    args = parser.parse_args()
    original = json.loads(args.argv_json.read_text())
    command = original.copy()
    command[0] = str(args.trainer)
    # Refuse anything except the registered full control budget and seed.
    for flag, expected in [("--epochs", "120"), ("--pairs-per-epoch", "50000"),
                           ("--init-seed", "1101"), ("--sample-seed", "101")]:
        if command[command.index(flag) + 1] != expected:
            raise ValueError(f"frozen control differs: {flag}")
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "refit").mkdir()
    (args.out / "ckpt").mkdir()
    command[command.index("--out") + 1] = str(args.out / "refit/best.bin")
    command[command.index("--dump-checkpoints-dir") + 1] = str(args.out / "ckpt")
    # Log the pre-existing authority before the trainer reads any target table.
    authority = Path(command[command.index("--subjective-authority") + 1])
    expected = command[command.index("--subjective-authority-sha256") + 1]
    if digest(authority) != expected:
        raise ValueError("frozen dataset authority changed")
    env = dict(os.environ, ZENSIM_MAX_TIER="v3", RAYON_NUM_THREADS="1",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    record = {"schema": "adam-denormal-control-v1", "original_argv": original,
              "actual_argv": command, "trainer_sha256": digest(args.trainer),
              "authority_sha256": expected, "purpose": "engineering replay; no study replacement",
              "environment": {key: env[key] for key in ("ZENSIM_MAX_TIER", "RAYON_NUM_THREADS",
                              "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}
    (args.out / "INVOCATION.json").write_text(json.dumps(record, indent=2) + "\n")
    measured = ["perf", "stat", "-o", str(args.out / "perf.stat"), "-e", args.events,
                "--", "/usr/bin/time", "-v", "-o", str(args.out / "time.txt"), *command]
    with (args.out / "train.log").open("w") as log:
        result = subprocess.run(measured, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"trainer rc={result.returncode}; see {args.out / 'train.log'}")
    final = args.out / "ckpt/ckpt_epoch119.bin"
    shutil.copyfile(final, args.out / "refit/last.bin")
    record["refit_sha256"] = {p.name: digest(p) for p in sorted((args.out / "refit").iterdir())}
    record["returncode"] = result.returncode
    (args.out / "COMPLETE.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
