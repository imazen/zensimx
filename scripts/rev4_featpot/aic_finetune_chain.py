"""Guarded local continuation of the preregistered Mlic AIC TRAIN arms.

Called by byc_cells under the shared heavy lock. Every failed stage exits nonzero;
only byte-exact actual-trainer controls permit the binary freeze and arm launch.
All paths and private input pins come from the caller's private launch packet.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".new")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def packet(prior, output):
    """Pin a fresh retry after committing source; preserve every earlier packet."""
    repo = Path(__file__).resolve().parents[2]
    if subprocess.check_output(["jj", "diff", "--summary"], cwd=repo, text=True).strip():
        raise ValueError("commit source before creating the launch packet")
    commit = subprocess.check_output(["jj", "log", "--no-graph", "-r", "@-", "-T", "commit_id"], cwd=repo, text=True).strip()
    files = subprocess.check_output(["jj", "file", "list", "-r", commit], cwd=repo, text=True).splitlines()
    value = json.loads(Path(prior).read_text())
    value["source_commit"] = commit
    value["source_files"] = {name: digest(repo / name) for name in files
        if (name.endswith((".rs", ".toml", ".lock", ".py", ".json", ".txt", ".sh")) or name == "justfile")
        and (repo / name).is_file()}
    value["prior_packet"] = str(Path(prior).resolve())
    with Path(output).open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
    print(json.dumps(dict(source_commit=commit, source_files=len(value["source_files"]), packet=str(output))))


def run(args, cells):
    study = Path(args.study).resolve()
    packet = json.loads(Path(args.packet).read_text())
    repo = Path(__file__).resolve().parents[2]
    stage = "source-admission"

    def state(name, **details):
        nonlocal stage
        stage = name
        value = dict(stage=name, utc=datetime.now(timezone.utc).isoformat(),
                     pid=os.getpid(), source_commit=packet["source_commit"], **details)
        write_json(study / "RUN_STATE.json", value)
        print(json.dumps(value), flush=True)

    def command(name, argv, *, env=None):
        state(name, command=[str(x) for x in argv])
        log = study / f"{name}.log"
        with log.open("x") as stream:
            stream.write(json.dumps([str(x) for x in argv]) + "\n")
            stream.flush()
            subprocess.run([str(x) for x in argv], cwd=repo, env=env,
                           stdout=stream, stderr=subprocess.STDOUT, check=True)

    def compare(name, actual, reference):
        if actual.read_bytes() != reference.read_bytes():
            raise ValueError(f"{name}: predictions differ byte for byte: {actual}, {reference}")
        value = dict(control=name, byte_identical=True, prediction_sha256=digest(actual),
                     rows=len(actual.read_text().splitlines()) - 1,
                     actual=str(actual), reference=str(reference))
        write_json(study / f"{name}.json", value)
        print(json.dumps(value), flush=True)
        return value

    try:
        state(stage)
        for relative, pin in packet["source_files"].items():
            if digest(repo / relative) != pin:
                raise ValueError(f"queued source changed: {relative}")
        if cells.ROOT.resolve() != Path(packet["root"]).resolve():
            raise ValueError("frozen root differs from launch packet")
        if cells.CELLS.resolve() != study / "cells":
            raise ValueError("cell output root differs from launch packet")
        bindir = Path(os.environ["REV4_V2_BIN_DIR"])
        if bindir.resolve() != study / "bin" or bindir.exists():
            raise ValueError("new binary directory must be fresh and in this study")
        oldbin = Path(packet["frozen_bin"])
        for name, pin in packet["frozen_binaries"].items():
            if digest(oldbin / name) != pin:
                raise ValueError(f"frozen binary changed: {name}")
        command("checks", ["just", "aic-finetune-checks"])
        command("python-admission", ["just", "aic-finetune-admission-tests"])
        command("subjective-admission", [sys.executable, "-B", "-m", "unittest",
                "scripts.tests.test_subjective_training_leg", "scripts.tests.test_dataset_legs"],
                env=dict(os.environ, PYTHONPATH=str(repo / "scripts/tests")))
        command("script-lint", ["just", "lint-scripts"])
        command("api-doc-check", ["just", "api-doc-check"])
        command("trainer-lib-tests", ["cargo", "test", "--offline", "--locked", "-p",
                "zensim-validate", "--lib", "--", "--test-threads=1"])
        command("trainer-cli-tests", ["cargo", "test", "--offline", "--locked", "-p",
                "zensim-validate", "--bin", "zensim_mlp_train", "--", "--test-threads=1"])
        command("fold-readback", ["just", "aic-finetune-fold-verify", os.environ["AIC_FOLDS"]])
        command("trainer-build", ["cargo", "build", "--offline", "--locked", "--release",
                "-p", "zensim-validate", "--bin", "zensim_mlp_train"])
        bindir.mkdir()
        for name in packet["frozen_binaries"]:
            source = (Path(os.environ["CARGO_TARGET_DIR"]) / "release" / name
                      if name == "zensim_mlp_train" else oldbin / name)
            shutil.copy2(source, bindir / name)
        write_json(study / "BIN_UNVERIFIED.json", dict(
            source_commit=packet["source_commit"], trainer_sha256=digest(bindir / "zensim_mlp_train")))
        # Native execution matches the frozen cells: identical dispatch and one thread.
        os.environ.update(RAYON_NUM_THREADS="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                          ZENSIM_MAX_TIER="v3")
        original = Path(packet["initial_cells"]) / "without_kadid_s0"
        zero = study / "controls" / "zero-step"
        zero.parent.mkdir()
        base = [sys.executable, "-B", cells.HERE / "v2_lodo_mlp.py", "--root", cells.ROOT,
                "--spec", cells.spec_of("mono:thr:mp32"), "--columns", cells.COLS.read_text().strip(),
                "--head", "N", "--heldout", "kadid", "--seed-index", "0", "--strict-admission",
                "--data-role-decision", cells.ROOT / "human_role_decision.json", "--train-only"]
        command("zero-step-cli", [*base, "--dest", zero, "--init-from", original / "refit/last.bin",
                "--fine-tune-epochs", "0", "--fine-tune-lr", "0.00001",
                "--aic-public-train", Path(os.environ["AIC_FOLDS"]) / "aic_k0_fit.parquet",
                "--research-extra-group", f"aic_public_train:{os.environ['AIC_FOLDS']}/aic_k0_fit.parquet:8.0:0:rank"])
        state("zero-step-prediction")
        selected = Path(json.loads((zero / "result.json").read_text())["selected_bake"])
        output = zero / "pred_kadid.tsv"
        cells.predict_cell(selected, cells.ROOT / "wide/main/real/kadid.parquet", output,
                           cells.spec_of("mono:thr:mp32"))
        zero_result = compare("ZERO_STEP", output, original / "pred_kadid.tsv")
        state("default-parity-training", frozen_trainer_sha256=packet["frozen_binaries"]["zensim_mlp_train"])
        cells.LABEL = "MlicDefaultControl"
        cells.AIC_WEIGHT = None
        cells.FINE_TUNE = None
        default_record = cells.run_cell("mono:thr:mp32", "kadid", 0)
        default = cells.CELLS / cells.LABEL / "without_kadid_s0"
        parity_result = compare("DEFAULT_PARITY", default / "pred_kadid.tsv", original / "pred_kadid.tsv")
        if default_record["last_bin_sha256"] != digest(default / "refit/last.bin"):
            raise ValueError("default control checkpoint changed")
        # Freeze only after both actual-trainer controls and all admission/tests pass.
        pins = {name: digest(bindir / name) for name in packet["frozen_binaries"]}
        frozen = dict(source_commit=packet["source_commit"], source_files=packet["source_files"],
                      frozen_trainer_sha256=packet["frozen_binaries"]["zensim_mlp_train"],
                      binaries=pins, zero_step=zero_result, default_parity=parity_result,
                      base_includes_adam=packet["base_includes_adam"])
        write_json(study / "BIN_FROZEN.json", frozen)
        write_json(study / "PREFLIGHT_PASSED.json", frozen)
        for path in bindir.iterdir():
            path.chmod(0o555)
        bindir.chmod(0o555)
        for label, weight, epochs in (("MlicFT8x4", 8.0, 4), ("MlicFT16x8", 16.0, 8)):
            if any(digest(bindir / name) != pin for name, pin in pins.items()):
                raise ValueError("frozen experiment tool changed")
            state("registered-arm", arm=label, epochs=epochs, aic_weight=weight,
                  trainer_sha256=pins["zensim_mlp_train"])
            arm = argparse.Namespace(label=label, aic_weight=weight, fine_tune_epochs=epochs,
                fine_tune_lr=0.00001, init_from_cells=Path(packet["initial_cells"]),
                base_spec=None, cols=None, folds=",".join(cells.FOLDS), seeds="0,1,2",
                jobs=args.jobs, arm="mono:thr:mp32")
            if cells.cmd_run(arm):
                raise RuntimeError(f"{label}: at least one registered cell failed")
            for seed in range(3):
                for fold in cells.FOLDS:
                    dest = cells.CELLS / label / f"without_{fold}_s{seed}"
                    record = json.loads((dest / "byc_record.json").read_text())
                    if (record["continuation"]["epochs"] != epochs or record["aic_fold"] != cells.aic_fold(fold, seed)
                            or record["trainer_sha256"] != pins["zensim_mlp_train"]
                            or record["last_bin_sha256"] != digest(dest / "refit/last.bin")
                            or len(record["curve"]) != epochs):
                        raise ValueError(f"completed cell inventory differs: {dest}")
        state("training-complete", registered_cells=24)
        write_json(study / "TRAIN_COMPLETE.json", json.loads((study / "RUN_STATE.json").read_text()))
        return 0
    except BaseException as error:
        failure = dict(stage=stage, error=str(error), traceback=traceback.format_exc(),
                       utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid(), exit_code=1)
        write_json(study / "FAILED.json", failure)
        print(json.dumps(failure), flush=True)
        return 1


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description="Create a content-pinned guarded continuation retry packet")
    cli.add_argument("prior")
    cli.add_argument("output")
    options = cli.parse_args()
    packet(options.prior, options.output)
