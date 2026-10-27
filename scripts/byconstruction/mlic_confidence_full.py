"""Three fixed full confidence fits after completed omission evidence, without scalar selection.

The caller supplies a hash-bound plan after the independent omission study
completes. This owner never starts AIC continuation or promotes a model.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aic_adaptation_full import check_fit, fit_command
from aic_adaptation import compute_lock, process_snapshot, sha, write
from confidence_weights import verify_receipt

FOLDS = ("kadid", "tid2013", "konfig", "cid22_a25", "pipal", "pairs", "mos7")
PREFIX = "ZENSIM_RESEARCH_CONFIDENCE_"


def full_binding(registry, pin, seed):
    if seed not in (0, 1, 2) or sha(registry) != pin:
        raise ValueError("fixed full seed and exact registry required")
    index = json.loads(Path(registry).read_text())
    if (index["schema"] != "mlic-full-contrast-manifests-v1"
            or set(index["cells"]) != {f"full-s{s}" for s in range(3)}):
        raise ValueError("separate complete full confidence registry required")
    binding = index["cells"][f"full-s{seed}"]
    if sha(binding["path"]) != binding["sha256"]:
        raise ValueError("full confidence manifest changed")
    value = json.loads(Path(binding["path"]).read_text())
    if (value["candidate"], value["fold"], value["seed_index"], value["n_features"]) != (
            "MlicConfidence", "full", seed, 2263):
        raise ValueError("full confidence manifest identity differs")
    return binding


def child_environment(parent, binding=None, destination=None):
    if any(k.startswith(PREFIX) for k in parent):
        raise ValueError("inherited confidence factors are forbidden")
    env = dict(parent)
    if binding is not None:
        if destination is None:
            raise ValueError("fresh native receipt destination required")
        receipt = Path(destination) / "CONFIDENCE_NORMALIZERS.json"
        if receipt.exists():
            raise ValueError("preserve existing native confidence receipt")
        env.update(ZENSIM_RESEARCH_CONFIDENCE_MANIFEST=binding["path"],
                   ZENSIM_RESEARCH_CONFIDENCE_SHA256=binding["sha256"],
                   ZENSIM_RESEARCH_CONFIDENCE_RECEIPT=str(receipt))
    return env


def completed_omissions(selection):
    if (selection["schema"] != "mlic-contrast-confidence-TRAIN-selection-v1"
            or selection["rule_commit"] != "d37a01340fa9"
            or selection["eval_labels_used"] is not False):
        raise ValueError("completed independent confidence TRAIN evidence required")
    expected = {(arm, fold, seed) for arm in ("Mlic", "MlicConfidence")
                for fold in FOLDS for seed in range(3)}
    cells = selection["cells"]
    if len(cells) != len(expected) or {(c["arm"], c["fold"], c["seed"]) for c in cells} != expected:
        raise ValueError("complete paired omission inventory required")
    # The immutable historical finalist verdict is retained, not an AIC gate.


def checked_plan(path):
    plan = json.loads(path.read_text())
    if plan["schema"] != "mlic-confidence-full-plan-v1" or plan["runner_sha256"] != sha(__file__):
        raise ValueError("pinned full confidence runner required")
    study = Path(plan["study"])
    required = [study / "TRAIN_SELECTION.json", study / "TRAIN_SCREEN_COMPLETE.json",
                study / "SOURCE.json", study / "bin/zensim_mlp_train",
                study / "source/scripts/rev4_featpot/v2_confirm_fit.py",
                study / "frozen_configuration/cols_c.txt", Path(plan["registry"]),
                study / "MlicConfidence/root/wide/frozen.json"]
    required += [Path(__file__).with_name(name) for name in
                 ("aic_adaptation_full.py", "aic_adaptation.py", "confidence_weights.py")]
    required += [Path(__file__).resolve().parents[1] / "rev4_featpot/run_processes.py"]
    if any(str(p) not in plan["input_sha256"] for p in required):
        raise ValueError("all full confidence prerequisite identities required")
    for name, pin in plan["input_sha256"].items():
        if sha(name) != pin:
            raise ValueError(f"full confidence input changed: {name}")
    selection = json.loads(required[0].read_text())
    completed_omissions(selection)
    complete = json.loads(required[1].read_text())
    if complete["selection_sha256"] != sha(required[0]):
        raise ValueError("omission completion binding changed")
    for name, pin in selection["input_sha256"].items():
        if sha(name) != pin:
            raise ValueError(f"omission evidence changed: {name}")
    source = json.loads(required[2].read_text())
    if source["candidate"] != "MlicConfidence" or source["trainer_sha256"] != sha(required[3]):
        raise ValueError("independently validated confidence trainer required")
    for seed in range(3):
        full_binding(plan["registry"], plan["input_sha256"][plan["registry"]], seed)
    return plan


def run(path):
    if not compute_lock(os.getpid())["held_by_worker_ancestor"]:
        raise ValueError("full confidence fitting requires shared heavy lock")
    plan = checked_plan(path)
    output, study = path.parent, Path(plan["study"])
    source, binary = study / "source", study / "bin"
    root = study / "MlicConfidence/root"
    columns = (study / "frozen_configuration/cols_c.txt").read_text().strip()
    state = dict(stage="full-confidence-fits", worker=process_snapshot(os.getpid()),
                 plan_sha256=sha(path), completed=[], required=3, failures=[], objective_complete=False)
    def record():
        state["checked_utc"] = datetime.now(timezone.utc).isoformat()
        write(output / "RUN_STATE.json", state)
    if (output / "RUN_STATE.json").exists():
        raise ValueError("preserve previous full confidence attempt")
    record()
    parent = child_environment(dict(os.environ, REV4_V2_BIN_DIR=str(binary),
        ZEN_PANEL_BIN=str(binary / "panel"), ZENSIM_MAX_TIER="v3", RAYON_NUM_THREADS="1",
        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", TMPDIR=str(Path.home() / "tmp")))
    def fit(seed):
        dest = output / f"full_s{seed}"
        binding = full_binding(plan["registry"], plan["input_sha256"][plan["registry"]], seed)
        with (output / f"full_s{seed}.log").open("x") as log:
            subprocess.run(fit_command(source, root, dest, columns, seed), cwd=source,
                env=child_environment(parent, binding, dest), stdout=log, stderr=subprocess.STDOUT, check=True)
            result = check_fit(dest, seed, sha(root / "wide/frozen.json"))
            result.update(confidence_manifest_sha256=binding["sha256"],
                          confidence_receipt_sha256=verify_receipt(dest / "CONFIDENCE_NORMALIZERS.json", binding))
            subprocess.run([str(binary / "bake_dial_refit"), "certify-monotone", "--bake",
                str(dest / "refit/production-f16.bin")], env=parent, stdout=log, stderr=subprocess.STDOUT, check=True)
        return result
    with ThreadPoolExecutor(max_workers=3) as pool:
        pending = {pool.submit(fit, seed): seed for seed in range(3)}
        for future in as_completed(pending):
            try:
                state["completed"].append(future.result())
            except BaseException as error:
                state["failures"].append(dict(seed=pending[future], error=repr(error)))
            record()
    if state["failures"]:
        state["stage"] = "failed"
        record()
        raise RuntimeError("full confidence failures preserved")
    checked_plan(path)
    state["completed"].sort(key=lambda r: r["seed"])
    write(output / "FULL_FITS_COMPLETE.json", dict(models=state["completed"], plan_sha256=sha(path),
        assessment_seed=0, selector=None, production_qualified=False), fresh=True)
    state["stage"] = "full-fits-complete-requires-common-AIC-continuation"
    record()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    run(parser.parse_args().plan.resolve())
