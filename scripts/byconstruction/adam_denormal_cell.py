"""Replay one frozen Adam engineering control with fresh outputs and perf counters.

The original trainer argv, data authority and seed streams remain the owners of
the recipe. This wrapper changes only the executable and output locations. It
retains every checkpoint and records original/actual argv and executable hashes.
Caller supplies resource limits, CPU affinity and the appropriate perf events.
"""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_changes(before, after, path=""):
    """Record exact semantic provenance changes using JSON Pointer paths."""
    if isinstance(before, dict) and isinstance(after, dict):
        changes = []
        for key in sorted(before.keys() | after.keys()):
            pointer = path + "/" + key.replace("~", "~0").replace("/", "~1")
            if key not in before or key not in after:
                changes.append({"path": pointer, "reference_present": key in before,
                                "candidate_present": key in after,
                                "reference": before.get(key), "candidate": after.get(key)})
            else:
                changes.extend(json_changes(before[key], after[key], pointer))
        return changes
    if isinstance(before, list) and isinstance(after, list) and len(before) == len(after):
        return [change for i, (x, y) in enumerate(zip(before, after))
                for change in json_changes(x, y, f"{path}/{i}")]
    if type(before) is not type(after) or before != after:
        return [{"path": path, "reference": before, "candidate": after}]
    return []


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
    (args.out / "tmp").mkdir()
    command[command.index("--out") + 1] = str(args.out / "refit/best.bin")
    command[command.index("--dump-checkpoints-dir") + 1] = str(args.out / "ckpt")
    # Log the pre-existing authority before the trainer reads any target table.
    authority = Path(command[command.index("--subjective-authority") + 1])
    expected = command[command.index("--subjective-authority-sha256") + 1]
    if digest(authority) != expected:
        raise ValueError("frozen dataset authority changed")
    env = dict(os.environ, TMPDIR=str(args.out / "tmp"), ZENSIM_MAX_TIER="v3", RAYON_NUM_THREADS="1",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    record = {"schema": "adam-denormal-control-v1", "original_argv": original,
              "actual_argv": command, "trainer_sha256": digest(args.trainer),
              "authority_sha256": expected, "purpose": "engineering replay; no study replacement",
              "environment": {key: env[key] for key in ("TMPDIR", "ZENSIM_MAX_TIER", "RAYON_NUM_THREADS",
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


def compare():
    parser = argparse.ArgumentParser(description="Compare the frozen control, native model bytes and packed bakes")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--tools", type=Path, required=True)
    parser.add_argument("--probe", type=Path, help="native section-splice comparison tool")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(sys.argv[2:])
    args.out.mkdir(parents=True, exist_ok=False)
    os.environ["REV4_V2_ROOT"] = str(args.root)
    os.environ["REV4_V2_BIN_DIR"] = str(args.tools)
    os.environ.update(ZENSIM_MAX_TIER="v3", RAYON_NUM_THREADS="1", OMP_NUM_THREADS="1",
                      OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rev4_featpot"))
    from v2_common import dense_bake
    from v2_production_pack import pack_production

    tool = args.tools / "bake_dial_refit"
    results = {"schema": "adam-denormal-byte-comparison-v1", "tool_sha256": digest(tool),
               "reference": str(args.reference), "candidate": str(args.candidate), "comparisons": {},
               "environment": {key: os.environ[key] for key in ("ZENSIM_MAX_TIER", "RAYON_NUM_THREADS",
                               "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}
    if args.probe:
        results["splice_probe_sha256"] = digest(args.probe)

    def run(command, log):
        with log.open("w") as stream:
            subprocess.run([str(x) for x in command], stdout=stream, stderr=subprocess.STDOUT, check=True)

    def compare_bytes(name, first, second):
        a, b = first.read_bytes(), second.read_bytes()
        different = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
        offsets = args.out / f"{name}.byte-differences.tsv"
        with offsets.open("w") as stream:
            stream.write("offset_zero_based\treference_hex\tcandidate_hex\n")
            for i in different:
                stream.write(f"{i}\t{a[i]:02x}\t{b[i]:02x}\n")
            for i in range(min(len(a), len(b)), max(len(a), len(b))):
                stream.write(f"{i}\t{a[i]:02x}\tEOF\n" if i < len(a) else f"{i}\tEOF\t{b[i]:02x}\n")
        native = subprocess.run(["cmp", "-s", str(first), str(second)])
        if native.returncode not in (0, 1):
            raise RuntimeError("native cmp failed")
        if (a == b) != (native.returncode == 0):
            raise RuntimeError("byte comparison disagrees with native cmp")
        results["comparisons"][name] = {
            "equal": a == b, "reference_bytes": len(a), "candidate_bytes": len(b),
            "different_common_offsets": len(different), "unmatched_tail_bytes": abs(len(a) - len(b)),
            "reference_sha256": digest(first), "candidate_sha256": digest(second),
            "all_different_bytes": str(offsets), "difference_listing_sha256": digest(offsets),
        }

    table = args.root / "wide/main/real/kadid.parquet"
    receipt = json.loads((args.root / "wide/main/real/receipt.json").read_text())
    anchor_record = receipt["legs"]["cid22"]["fit"]
    anchor = args.root / anchor_record["rel"]
    if digest(anchor) != anchor_record["sha256"]:
        raise ValueError("frozen TRAIN packing anchor changed")
    (args.out / "LABEL_READS.json").write_text(json.dumps({
        "purpose": "same frozen control engineering comparison; no candidate selection",
        "TRAIN_packing": {"path": str(anchor), "sha256": digest(anchor), "column": "human_score"},
        "prediction_table": {"path": str(table), "sha256": digest(table), "operation": "native prediction and byte comparison; no target used for selection"},
    }, indent=2) + "\n")
    predictions, packed, normalized, dense_models = {}, {}, {}, {}
    for label, cell in [("reference", args.reference), ("candidate", args.candidate)]:
        dest = args.out / label
        (dest / "refit").mkdir(parents=True)
        dense = dense_bake(cell / "refit/last.bin", dest)
        dense_models[label] = dense
        pred = dest / "pred_kadid.tsv"
        run([tool, "predict", "--bake", dense, "--corpus", table, "--score-units", "--out", pred], dest / "predict.log")
        predictions[label] = pred
        record = pack_production(cell / "refit/last.bin", dest, anchor, e33_output_stage=True)
        packed[label] = Path(record["packed_model"])
        (dest / "PACKED.json").write_text(json.dumps(record, indent=2) + "\n")
        for name, bake in [("best", cell / "refit/best.bin"), ("last", cell / "refit/last.bin"),
                           ("dense", dense), ("packed", packed[label])]:
            target = dest / f"{name}.without-repro.bin"
            run([tool, "strip", "--in", bake, "--out", target, "--key", "zentrain.repro"], dest / f"{name}.strip.log")
            normalized[label, name] = target
            if args.probe:
                spliced = dest / f"{name}.constant-repro.bin"
                run([args.probe, "--normalize-repro", bake, spliced], dest / f"{name}.splice.log")
                normalized[label, f"splice_{name}"] = spliced
                repro = dest / f"{name}.repro.json"
                run([args.probe, "--repro", bake, repro], dest / f"{name}.repro.log")
                normalized[label, f"repro_{name}"] = repro

    compare_bytes("reference_prediction_replay", args.reference / "pred_kadid.tsv", predictions["reference"])
    compare_bytes("predictions", args.reference / "pred_kadid.tsv", predictions["candidate"])
    with (args.reference / "pred_kadid.tsv").open() as first, predictions["candidate"].open() as second:
        a = list(csv.DictReader(first, delimiter="\t"))
        b = list(csv.DictReader(second, delimiter="\t"))
    if len(a) != len(b) or any(x["row_idx"] != y["row_idx"] for x, y in zip(a, b)):
        raise ValueError("prediction row identities differ")
    changes = []
    for x, y in zip(a, b):
        before, after = float(x["pred"]), float(y["pred"])
        if not math.isfinite(before) or not math.isfinite(after):
            raise ValueError("nonfinite prediction")
        if x["pred"] != y["pred"]:
            changes.append({"row_idx": x["row_idx"], "reference": x["pred"], "candidate": y["pred"],
                            "absolute_change": abs(before - after),
                            "f64_bits_differ": struct.pack(">d", before) != struct.pack(">d", after)})
    values = args.out / "prediction-value-differences.tsv"
    with values.open("w") as stream:
        writer = csv.DictWriter(stream, delimiter="\t", fieldnames=["row_idx", "reference", "candidate", "absolute_change", "f64_bits_differ"])
        writer.writeheader()
        writer.writerows(changes)
    results["prediction_values"] = {"rows": len(a), "different_text_rows": len(changes),
                                    "different_f64_rows": sum(x["f64_bits_differ"] for x in changes),
                                    "max_absolute_change": max((x["absolute_change"] for x in changes), default=0.0),
                                    "all_changes": str(values), "sha256": digest(values)}
    for name in ("best", "last"):
        compare_bytes(f"raw_{name}", args.reference / f"refit/{name}.bin", args.candidate / f"refit/{name}.bin")
    compare_bytes("raw_dense", dense_models["reference"], dense_models["candidate"])
    compare_bytes("raw_packed", packed["reference"], packed["candidate"])
    results["refit_inventory"] = {}
    for label, cell in [("reference", args.reference), ("candidate", args.candidate)]:
        results["refit_inventory"][label] = {p.name: digest(p) for p in sorted((cell / "refit").iterdir()) if p.is_file()}
    first_spec = args.reference / "refit/best.bin.spec.json"
    second_spec = args.candidate / "refit/best.bin.spec.json"
    compare_bytes("refit_best_spec", first_spec, second_spec)
    results["spec_value_differences"] = json_changes(json.loads(first_spec.read_text()), json.loads(second_spec.read_text()))
    for name in ("best", "last", "dense", "packed"):
        compare_bytes(f"without_repro_{name}", normalized["reference", name], normalized["candidate", name])
        if args.probe:
            compare_bytes(f"constant_repro_splice_{name}", normalized["reference", f"splice_{name}"], normalized["candidate", f"splice_{name}"])
            changes = json_changes(json.loads(normalized["reference", f"repro_{name}"].read_text()), json.loads(normalized["candidate", f"repro_{name}"].read_text()))
            listing = args.out / f"{name}.repro-value-differences.json"
            listing.write_text(json.dumps(changes, indent=2) + "\n")
            results[f"{name}_repro_value_differences"] = {"count": len(changes), "all_changes": str(listing), "sha256": digest(listing)}
    (args.out / "COMPARISON.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


def status():
    parser = argparse.ArgumentParser(description="Read one engineering replay's existing progress files")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(sys.argv[2:])
    log = args.out / "train.log"
    text = log.read_text() if log.is_file() else ""
    epochs = re.findall(r"^subjective-rank epoch=(\d+)", text, re.MULTILINE)
    wrapper = args.out.parent / "results" / f"{args.out.name}-wrapper.log"
    errors = wrapper.read_text() if wrapper.is_file() else ""
    print(json.dumps({"cell": args.out.name, "started": log.is_file(),
                      "last_logged_epoch": int(epochs[-1]) if epochs else None,
                      "complete": (args.out / "COMPLETE.json").is_file(),
                      "failed": "Traceback" in errors or "RuntimeError" in errors}))


def docker_control():
    parser = argparse.ArgumentParser(description="Run one frozen-image control with recorded resource limits")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--phase", choices=("before", "after"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(sys.argv[2:])
    plan = json.loads(args.plan.read_text())
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", args.output):
        raise ValueError("output must be one fresh directory name")
    if not re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", plan["image"]):
        raise ValueError("control image must be pinned by digest")
    if int(plan["memory_bytes"]) <= 0 or int(plan["cpu"]) < 0:
        raise ValueError("invalid resource limits")
    command = ["docker", "run", "--rm", "--name", "adam-denormal-" + args.output,
               "--cap-add", "PERFMON", "--security-opt", "seccomp=unconfined",
               "--cpuset-cpus", str(plan["cpu"]), "--cpu-shares", "256",
               "--memory", str(plan["memory_bytes"]), "--memory-swap", str(plan["memory_bytes"])]
    for source, target in plan["mounts"].items():
        if not Path(source).exists():
            raise FileNotFoundError(source)
        command.extend(["-v", source + ":" + target + ("" if target == "/adam" else ":ro")])
    command.extend(["--entrypoint", "/usr/bin/nice", plan["image"], "-n19", "ionice", "-c3",
                    "python3", "/adam/input/" + plan.get("launch_script", "launch-dev.py"),
                    args.phase, args.output])
    if args.dry_run:
        print(json.dumps(command, indent=2))
        return
    tests = plan.get("native_tests", {}).get(args.phase)
    if tests is not None:
        tests = Path(tests).resolve(strict=True)
        record = args.plan.parent / "results" / f"{args.output}.native-test-command.json"
        with record.open("x") as stream:
            stream.write(json.dumps({"argv": [str(tests), "--test-threads=1"],
                                     "sha256": digest(tests)}, indent=2) + "\n")
        env = dict(os.environ)
        env.pop("ZENSIM_MAX_TIER", None)
        log = args.plan.parent / "results" / f"{args.output}.native-tests.log"
        with log.open("x") as stream:
            subprocess.run([str(tests), "--test-threads=1"], stdout=stream,
                           stderr=subprocess.STDOUT, env=env, check=True)
    record = args.plan.parent / "results" / f"{args.output}.docker-command.json"
    with record.open("x") as stream:
        stream.write(json.dumps(command, indent=2) + "\n")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        compare()
    elif len(sys.argv) > 1 and sys.argv[1] == "status":
        status()
    elif len(sys.argv) > 1 and sys.argv[1] == "docker":
        docker_control()
    else:
        main()
