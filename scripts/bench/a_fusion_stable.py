#!/usr/bin/env python3
"""Preserve and compare the explicitly ignored legacy-pool failure, never waive it."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess

POOLS = ("masked_sum", "masked_fourth", "masked_square", "iw_sum", "iw_fourth", "iw_square",
         "mask_only_sum", "mask_only_fourth_xor_square_bits", "iw_only_sum")
SIZES = (1, 15, 16, 17, 31, 33, 64, 129, 512)
TEST = "ssim_form::tests::stable_moments_are_exact_across_simd_tiers"


def differences(path):
    lines = path.read_text().splitlines()
    left = [line.strip()[6:] for line in lines if line.strip().startswith("left: ")]
    right = [line.strip()[7:] for line in lines if line.strip().startswith("right: ")]
    assert len(left) == len(right) == 1, "expected exactly the preserved assertion failure"
    a, b = ast.literal_eval(left[0]), ast.literal_eval(right[0])
    result = []

    def walk(x, y, key=()):
        if isinstance(x, (list, tuple)):
            assert len(x) == len(y)
            for i, (u, v) in enumerate(zip(x, y)):
                walk(u, v, key + (i,))
        elif x != y:
            row = dict(index=list(key), left_bits=x, right_bits=y)
            if key[0] == 1:
                row.update(component="legacy_weighted_pool", elements=SIZES[key[1]], pool=POOLS[key[2]])
            else:
                row.update(component="stable_pixel_output")
            result.append(row)

    walk(a, b)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("before", "after", "sde", "out"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    assert os.uname().nodename == "i270", "run native gates on the authorized host"
    args.out.mkdir(parents=True, exist_ok=False)
    receipt = dict(test=TEST, ignored_marker_unchanged=True, runs=[], equal=True)
    for revision in (1, 7):
        for mode in ("native", "sde"):
            rows = []
            for arm in ("before", "after"):
                binary = getattr(args, arm).resolve()
                command = ([str(args.sde), "-spr", "--"] if mode == "sde" else [])
                command += [str(binary), "--exact", TEST, "--ignored", "--nocapture", "--test-threads=1"]
                log = args.out / f"rev{revision}-{mode}-{arm}.log"
                with log.open("x") as stream:
                    run = subprocess.run(command, env=dict(os.environ, ZENSIM_FORMULA_REV=str(revision)),
                                         stdout=stream, stderr=subprocess.STDOUT)
                assert run.returncode == 101, f"unexpected outcome {revision}/{mode}/{arm}: {run.returncode}"
                values = differences(log)
                row = dict(revision=revision, mode=mode, arm=arm, returncode=run.returncode,
                           environment={"ZENSIM_FORMULA_REV": str(revision)}, command=command,
                           binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                           log=str(log), log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
                           differences=values)
                receipt["runs"].append(row)
                rows.append(values)
            count = 41 if mode == "native" else 40
            equal = len(rows[0]) == count and rows[0] == rows[1]
            receipt["equal"] &= equal
            print(f"Rev{revision} {mode}: before={len(rows[0])} after={len(rows[1])} complete_lists_equal={equal}", flush=True)
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    assert receipt["equal"], "fusion changed the baseline's complete failure list; see receipt"


if __name__ == "__main__":
    main()
