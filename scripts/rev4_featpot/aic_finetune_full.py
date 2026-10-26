"""Continue and certify the three full-data seeds of the TRAIN-selected arm.

Caller owns heavy.lock/run-heavy. Selection is recomputed from the completed
TRAIN analysis. Registry publication and the coordinator's gates remain separate.
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

HERE = Path(__file__).resolve().parent
STRENGTHS = {"MlicFT8x4": (4, 8.0), "MlicFT16x8": (8, 16.0)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def selected_arm(directory):
    e21 = json.loads((directory / "E21.json").read_text())
    aic = json.loads((directory / "AIC_OOF.json").read_text())
    mixtures = json.loads((directory / "mixture_global.json").read_text())
    qualified = []
    for arm, (epochs, weight) in STRENGTHS.items():
        mix = mixtures[arm]
        if mix["cells"] != 12 or len(mix["per_cell"]) != 12 or len(aic[arm]["per_cell"]) != 12:
            raise ValueError("selection requires all twelve cells per arm")
        if (all(value["inversions"] == 0 for value in mix["per_cell"].values())
                and e21[arm]["delta"] >= -2 * e21[arm]["seed_unit_se"]):
            qualified.append((aic[arm]["mean_cell_global_gain"], -epochs, -weight, arm))
    winner = max(qualified)[3] if qualified else None
    receipt = json.loads((directory / "TRAIN_SELECTION.json").read_text())
    if receipt["winner"] != winner:
        raise ValueError("selection receipt differs from the frozen TRAIN rule")
    return winner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--folds", required=True, type=Path)
    parser.add_argument("--jobs", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    study = args.study.resolve()
    frozen = json.loads((study / "BIN_FROZEN.json").read_text())
    packet = json.loads(args.packet.read_text())
    if sha(study / "TRAIN_COMPLETE.json") != json.loads(args.selection.read_text())["training_complete_sha256"]:
        raise ValueError("selection is not bound to this completed training inventory")
    if sha(args.folds / "_MANIFEST.json") != packet["fold_manifest_sha256"]:
        raise ValueError("full-data AIC inventory differs")
    winner = selected_arm(args.selection.parent)
    if winner is None:
        write(study / "NO_FINALIST.json", dict(winner=None, selection_sha256=sha(args.selection)))
        print("No arm qualifies under the frozen rule; no full-data fit or registry addition.", flush=True)
        return 0
    epochs, weight = STRENGTHS[winner]
    inventory = json.loads((study / "INITIAL_CHECKPOINTS_VERIFIED.json").read_text())
    initial = {row["seed"]: row for row in inventory["checkpoints"] if row["kind"] == "full"}
    if set(initial) != {0, 1, 2}:
        raise ValueError("three receipt-bound full-data initial checkpoints required")
    output = study / "full" / winner
    output.mkdir(parents=True)
    os.environ.update(REV4_V2_ROOT=packet["root"], REV4_V2_BIN_DIR=str(study / "bin"),
        ZEN_PANEL_BIN=str(study / "bin/panel"), RAYON_NUM_THREADS="1", OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1", ZENSIM_MAX_TIER="v3")

    def tools_unchanged():
        if any(sha(study / "bin" / name) != pin for name, pin in frozen["binaries"].items()):
            raise ValueError("frozen experiment tool changed")

    def fit(seed):
        tools_unchanged()
        source = Path(initial[seed]["path"])
        if sha(source) != initial[seed]["sha256"]:
            raise ValueError("full-data initial checkpoint changed")
        parent = source.parent.parent
        original = json.loads((parent / "result.json").read_text())
        if (sha(parent / "result.json") != initial[seed]["record_sha256"]
                or original["output_stage"] != "e33-identity-knot-tail2"):
            raise ValueError("original full-data fit/packing receipt differs")
        dest = output / f"full_s{seed}"
        columns = ",".join((parent / "keep_features.txt").read_text().split())
        scratch = Path(os.environ["TMPDIR"]) / winner / f"full_s{seed}"
        scratch.mkdir(parents=True)
        command = [sys.executable, "-B", str(HERE / "v2_confirm_fit.py"), "--root", packet["root"],
            "--spec", original["spec"], "--head", "N", "--columns", columns, "--seed-index", str(seed),
            "--strict-admission", "--data-role-decision", str(Path(packet["root"]) / "human_role_decision.json"),
            "--dest", str(dest), "--train-only", "--pack-production", "--e33-output-stage",
            "--init-from", str(source), "--fine-tune-epochs", str(epochs), "--fine-tune-lr", "0.00001",
            "--aic-public-train", str(args.folds / "aic_full_fit.parquet"), "--aic-weight", str(weight)]
        print(json.dumps(dict(stage="full-data-fit", arm=winner, seed=seed, epochs=epochs)), flush=True)
        with (output / f"full_s{seed}.wrapper.log").open("x") as stream:
            stream.write(json.dumps(command) + "\n")
            stream.flush()
            subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT,
                           env=dict(os.environ, TMPDIR=str(scratch)))
        result = json.loads((dest / "result.json").read_text())
        bake = dest / "refit/production-f16.bin"
        continuation = result["continuation"]
        if (result["epochs"] != epochs or result["pairs_per_epoch"] != 50000
                or result["selection"]["selected_epoch"] != epochs - 1
                or result["init_seed"] != original["init_seed"]
                or result["sample_seed"] != original["sample_seed"]
                or result["output_stage"] != original["output_stage"]
                or result["TRAIN_calibration_table_sha256"] != original["TRAIN_calibration_table_sha256"]
                or continuation["initial_checkpoint_sha256"] != initial[seed]["sha256"]
                or continuation["lr_schedule"] != "constant" or continuation["lr"] != 0.00001
                or continuation["optimizer_state"] != "fresh"
                or result["packed_model_sha256"] != sha(bake)):
            raise ValueError("completed full-data recipe/seed/packing inventory differs")
        tools_unchanged()
        certificate = subprocess.run([str(study / "bin/bake_dial_refit"), "certify-monotone", "--bake", str(bake)],
            text=True, capture_output=True, check=True)
        cert = json.loads(certificate.stdout)
        if not cert["certified"] or not cert["monotone"]:
            raise ValueError("packed full-data model is not certified monotone")
        record = dict(arm=winner, seed=seed, bake=str(bake), sha256=sha(bake), certificate=cert,
            initial_checkpoint_sha256=initial[seed]["sha256"], selection_sha256=sha(args.selection),
            trainer_sha256=frozen["binaries"]["zensim_mlp_train"],
            trainer_source_commit=frozen["source_commit"], driver_sha256=sha(HERE / "v2_confirm_fit.py"),
            runner_sha256=sha(Path(__file__)), result_sha256=sha(dest / "result.json"))
        write(dest / "FULL_CELL_VERIFIED.json", record)
        print(json.dumps(dict(stage="full-data-certified", arm=winner, seed=seed, sha256=record["sha256"])), flush=True)
        return record

    results, failures = [], []
    with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
        pending = {pool.submit(fit, seed): seed for seed in range(3)}
        for future in concurrent.futures.as_completed(pending):
            try:
                results.append(future.result())
            except Exception as error:
                failures.append(dict(seed=pending[future], error=str(error), traceback=traceback.format_exc()))
    if failures:
        write(output / "FULL_FAILED.json", dict(failures=failures, verified_seeds=results, exit_code=1))
        return 1
    write(study / "FULL_COMPLETE.json", dict(arm=winner, seeds=sorted(results, key=lambda row: row["seed"]),
        selection_sha256=sha(args.selection), registry="pending", coordinator_gates="pending", eval="unread"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
