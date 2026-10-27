"""All registered full-data recipes and both AIC strengths, without scalar selection.

Original recipe fits precede common AIC continuation. No EVAL read or serving
promotion is performed. Existing experiments and their decisions are immutable.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
import re
from pathlib import Path
import shlex
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aic_adaptation import RECIPES, SPEC, STRENGTHS, bound_worker, compute_lock, process_snapshot, process_tree, sha, write
from aic_adaptation import require_prediction_parity, zero_checkpoint, require_clean_factors


def fit_command(source, root, output, columns, seed, *, initial=None, strength=None, aic=None, matching=None):
    if seed not in (0, 1, 2):
        raise ValueError("three fixed full-data seeds required")
    command = [sys.executable, "-B", str(source / "scripts/rev4_featpot/v2_confirm_fit.py"),
        "--root", str(root), "--spec", SPEC, "--columns", columns, "--head", "N", "--seed-index", str(seed),
        "--strict-admission", "--data-role-decision", str(root / "human_role_decision.json"),
        "--dest", str(output), "--train-only", "--pack-production", "--e33-output-stage"]
    if initial is not None:
        if strength not in STRENGTHS or aic is None or matching is not None:
            raise ValueError("common registered continuation must not carry starting-recipe factors")
        weight, epochs = strength
        command += ["--init-from", str(initial), "--fine-tune-epochs", str(epochs), "--fine-tune-lr", "0.00001",
                    "--aic-public-train", str(aic), "--aic-weight", str(weight)]
    elif strength is not None or aic is not None:
        raise ValueError("continuation requires an initial model")
    if matching is not None:
        command += ["--matched-codec-full", "--matched-codec-manifest", matching["path"],
                    "--matched-codec-sha256", matching["sha256"]]
    return command


def check_fit(directory, seed, root_pin, *, initial=None, strength=None):
    result = json.loads((directory / "result.json").read_text())
    epochs = 120 if initial is None else strength[1]
    # Same fixed seed streams as the canonical full-fit owner.
    seeds = ((1101, 101), (1103, 100000101), (1107, 200000101))
    if (result["schema"] != "rev5-qualified-training-cell-v1" or result["training_only"] is not True
            or result["execution_contract"] != "registered-fit" or result["seed_index"] != seed
            or result["spec"] != SPEC or result["head"] != "N" or result["epochs"] != epochs
            or result["pairs_per_epoch"] != 50000 or result["selection"]["epoch_rule"] != "last"
            or result["selection"]["selected_epoch"] != epochs - 1
            or (result["init_seed"], result["sample_seed"]) != seeds[seed]
            or result["frozen_sha256"] != root_pin or result["width"] != 1853 or result["kept_features"] != 410
            or result["output_stage"] != "e33-identity-knot-tail2"
            or result["selected_bake_sha256"] != sha(directory / "refit/last.bin")
            or result["packed_model_sha256"] != sha(directory / "refit/production-f16.bin")):
        raise ValueError("full-data fit recipe, final checkpoint or packing differs")
    selection = json.loads((directory / "DATASET_GROUP_SELECTION.json").read_text())
    if selection["root_sha256"] != root_pin or selection["requested_exclusions"] or selection["omitted_collections"]:
        raise ValueError("full-data fit retained an omission")
    authority_file = directory / ("continuation-authority.json" if initial is not None else "dataset-authority.json")
    authority = json.loads(authority_file.read_text())
    cell_name = f"full-s{seed}"
    if set(authority["cells"]) != {cell_name} or authority["cells"][cell_name]["route"] != "full":
        raise ValueError("full-data authority population differs")
    if initial is not None:
        expected = dict(initial_checkpoint=str(initial), initial_checkpoint_sha256=sha(initial),
                        epochs=epochs, lr=.00001, lr_schedule="constant", optimizer_state="fresh")
        if result.get("continuation") != expected:
            raise ValueError("full-data continuation initialization differs")
        bound = authority["cells"][cell_name].get("continuation", {})
        if (authority["schema"] != "subjective-training-authority-v3"
                or bound != dict(checkpoint=str(initial.resolve()), checkpoint_sha256=sha(initial), epochs=epochs,
                                 lr=.00001, lr_schedule="constant", optimizer_state="fresh")):
            raise ValueError("native full-data continuation authority differs")
        command = shlex.split((directory / "train.log").read_text().splitlines()[0])
        extras = [command[i+1].split(":") for i, t in enumerate(command[:-1])
                  if t == "--group" and command[i+1].startswith("aic_public_train:")]
        if (len(extras) != 1 or Path(extras[0][1]).name != "aic_full_fit.parquet"
                or float(extras[0][2]) != strength[0] or extras[0][3:] != ["0", "rank"]):
            raise ValueError("actual full continuation AIC mass or population differs")
    return dict(directory=str(directory), seed=seed, checkpoint_sha256=sha(directory / "refit/last.bin"),
                packed_sha256=sha(directory / "refit/production-f16.bin"),
                input_sha256={str(directory / name): sha(directory / name) for name in
                    ("result.json", "DATASET_GROUP_SELECTION.json", "keep_features.txt", "train.log", authority_file.name)})


def require_full_control_parity(original, replay, tool, output):
    """Compare all model bytes except run-specific reproduction metadata.

    This uses the existing E33 native strip owner; original artifacts remain
    untouched and complete result/admission records must match apart from their
    output paths and model hashes, which are independently bound by check_fit.
    """
    original, replay, output = Path(original), Path(replay), Path(output)
    a, b = [json.loads((p / "result.json").read_text()) for p in (original, replay)]
    transport = {"selected_bake", "selected_bake_sha256", "dense_model_sha256",
                 "packed_model", "packed_model_sha256"}
    if {k: v for k, v in a.items() if k not in transport} != {k: v for k, v in b.items() if k not in transport}:
        raise ValueError("full control recipe/admission/result fields changed")
    output.mkdir(exist_ok=False)
    rows = []
    for name in ("last.bin", "production-f16.bin"):
        copies = []
        for label, cell in (("original", original), ("replay", replay)):
            dest = output / f"{label}-{name}"
            with (output / f"{label}-{name}.log").open("x") as log:
                subprocess.run([str(tool), "strip", "--in", str(cell / "refit" / name),
                                "--out", str(dest), "--key", "zentrain.repro"],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            copies.append(dest)
        if copies[0].read_bytes() != copies[1].read_bytes():
            raise ValueError(f"full control numerical/serving bytes changed: {name}")
        rows.append(dict(model=name, nonrepro_sha256=sha(copies[0]),
                         original_sha256=sha(original / "refit" / name),
                         replay_sha256=sha(replay / "refit" / name)))
    return dict(comparison="exact native model bytes excluding zentrain.repro only",
                original_artifacts_unchanged=True, result_fields_equal=True,
                native_owner_sha256=sha(tool), models=rows)


def panel_finished(plan):
    output = Path(plan["fold_panel"])
    complete = output / "PANEL_COMPLETE.json"
    if complete.exists():
        result = json.loads(complete.read_text())
        if result["plan_sha256"] != plan["fold_plan_sha256"] or result["selector"] is not None:
            raise ValueError("preceding panel registration differs")
        for name, pin in result["outputs_sha256"].items():
            if sha(output / name) != pin:
                raise ValueError("preceding panel output changed")
        return True
    path = output / "PANEL_STATE.json"
    state = json.loads((path if path.exists() else output.parent / "PANEL_QUEUE.json").read_text())
    if state.get("failure") or state.get("failures"):
        raise RuntimeError("preceding panel failed; inspect preserved evidence")
    worker = state["worker"]
    if not bound_worker(worker["pid"], worker["start_ticks"]):
        raise RuntimeError("preceding panel process exited without completion")
    return False


def full_recipes(plan):
    expected = {"aic-recipe-full-plan-v1": RECIPES,
                "aic-confidence-full-plan-v1": ("Mlic", "MlicConfidence")}.get(plan.get("schema"))
    if expected is None or plan.get("recipes") != list(expected):
        raise ValueError("registered full recipe inventory differs")
    return expected


def unique_seed(rows, seed):
    if len(rows) != 3 or {r["seed"] for r in rows} != {0, 1, 2}:
        raise ValueError("all three distinct fixed seeds required")
    return next(r for r in rows if r["seed"] == seed)


def confidence_initial(plan, seed, root_pin):
    from mlic_confidence_full import full_binding
    from confidence_weights import verify_receipt
    root = Path(plan["confidence_full"])
    complete = json.loads((root / "FULL_FITS_COMPLETE.json").read_text())
    if complete["plan_sha256"] != sha(root / "PLAN.json") or complete["selector"] is not None:
        raise ValueError("confidence full-fit completion differs")
    prior_plan = json.loads((root / "PLAN.json").read_text())
    for name, pin in prior_plan["input_sha256"].items():
        if sha(name) != pin:
            raise ValueError("confidence full-fit prerequisite changed")
    row = unique_seed(complete["models"], seed)
    dest = Path(row["directory"])
    current = check_fit(dest, seed, root_pin)
    if any(current[k] != row[k] for k in current):
        raise ValueError("completed confidence model or recipe changed")
    binding = full_binding(prior_plan["registry"], prior_plan["input_sha256"][prior_plan["registry"]], seed)
    receipt = dest / "CONFIDENCE_NORMALIZERS.json"
    if (row["confidence_manifest_sha256"] != binding["sha256"]
            or row["confidence_receipt_sha256"] != verify_receipt(receipt, binding)):
        raise ValueError("full confidence native receipt changed")
    current["input_sha256"].update({binding["path"]:binding["sha256"], str(receipt):sha(receipt)})
    return dict(current, recipe="MlicConfidence", reused=True)


def reused_mlic(plan, seed, strength):
    root = Path(plan["reuse_mlic_full"])
    complete = json.loads((root / "FULL_FITS_COMPLETE.json").read_text())
    if complete["plan_sha256"] != sha(root / "PLAN.json") or complete["selector"] is not None:
        raise ValueError("original Mlic continuation completion differs")
    weight, epochs = strength
    arm = f"MlicFT{weight}x{epochs}"
    rows = [r for r in complete["continued"] if r["arm"] == arm]
    row = unique_seed(rows, seed)
    if row["recipe"] != "Mlic" or (row["weight"], row["epochs"]) != strength:
        raise ValueError("reused Mlic recipe/strength changed")
    for name, pin in row["input_sha256"].items():
        if sha(name) != pin:
            raise ValueError("reused Mlic continuation evidence changed")
    return Path(row["directory"])


def checked_plan(path):
    plan = json.loads(path.read_text())
    full_recipes(plan)
    if (plan["selection"] != "full-panel-no-srocc-cutoff"
            or plan["coordinator_sha256"] != sha(__file__)
            or sha(plan["fold_plan"]) != plan["fold_plan_sha256"]):
        raise ValueError("full-data study registration differs")
    for name, pin in plan["input_sha256"].items():
        if sha(name) != pin:
            raise ValueError(f"full-data input changed: {name}")
    if plan["schema"] == "aic-confidence-full-plan-v1":
        required = [Path(plan[key]) / name for key in ("confidence_full", "reuse_mlic_full")
                    for name in ("PLAN.json", "FULL_FITS_COMPLETE.json")]
        required += [Path(__file__).with_name(name) for name in
                     ("mlic_confidence_full.py", "confidence_weights.py", "aic_adaptation.py")]
        if any(str(p) not in plan["input_sha256"] for p in required):
            raise ValueError("full confidence extension requires pinned completed initial fits and Mlic controls")
    if "replay_control" in plan:
        control = Path(plan["replay_control"])
        required = ["result.json", "refit/last.bin", "refit/production-f16.bin",
                    "DATASET_GROUP_SELECTION.json", "dataset-authority.json", "keep_features.txt", "train.log"]
        if any(str(control / name) not in plan["input_sha256"] for name in required):
            raise ValueError("reused full control must bind all artifacts and authority")
    return plan


def record(root, state, stage, **changes):
    state.update(stage=stage, checked_utc=datetime.now(timezone.utc).isoformat(), **changes)
    write(root / "FULL_STATE.json", state)


def queue(path):
    plan, root = checked_plan(path), path.parent
    state = dict(stage="waiting-for-fold-panel", plan_sha256=sha(path), worker=process_snapshot(os.getpid()),
                 completed=[], failures=[], required_initial=3 * len(full_recipes(plan)), required_continued=6 * len(full_recipes(plan)), objective_complete=False,
                 source_commit=plan["coordinator_commit"], next_action="complete every registered full-data recipe")
    write(root / "FULL_STATE.json", state, fresh=True)
    try:
        while not panel_finished(plan):
            time.sleep(30)
        checked_plan(path)
        cmd = ["flock", str(Path.home() / "tmp/heavy.lock"), str(Path.home() / "work/claudehints/scripts/run-heavy"),
               "--mem", "24G", "--jobs", "6", "--", sys.executable, "-B", str(Path(__file__).resolve()), "run", str(path)]
        record(root, state, "queued-full-data", command=cmd)
        with (root / "launch.log").open("x") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    except BaseException as error:
        current = json.loads((root / "FULL_STATE.json").read_text())
        record(root, current, "failed", failures=[*current["failures"], repr(error)])
        raise


def active_training(processes):
    """Report logged epochs only for trainers present in the verified live tree."""
    rows = []
    for process in processes:
        argv = process["argv"]
        if not argv or Path(argv[0]).name != "zensim_mlp_train":
            continue
        identity = {key: process[key] for key in ("pid", "start_ticks", "state", "cpu_ticks")}
        try:
            directory = Path(argv[argv.index("--out") + 1]).parent.parent
            epochs = int(argv[argv.index("--epochs") + 1])
        except (ValueError, IndexError):
            rows.append(dict(**identity, error="cannot bind live trainer output and budget"))
            continue
        path = directory / "train.log"
        logged = []
        if path.exists():
            text = path.read_text()
            logged = [int(v) for v in re.findall(r"subjective-rank epoch=(\d+)", text)]
            logged += [int(v) for v in re.findall(r"^\s*epoch\s+(\d+)\s*\|", text, re.MULTILINE)]
        rows.append(dict(**identity, directory=str(directory), epochs=epochs,
                         latest_logged_epoch=max(logged) if logged else None,
                         log_present=path.exists(), completion_inferred=False))
    return rows


def status(path):
    state = json.loads((path.parent / "FULL_STATE.json").read_text())
    worker = state["worker"]
    live = bound_worker(worker["pid"], worker["start_ticks"])
    if not live and state["stage"] != "full-fits-complete" and not state["failures"]:
        state["failures"].append("recorded worker exited without completion")
    state.update(checked_utc=datetime.now(timezone.utc).isoformat(),
                 live_processes=process_tree(worker["pid"]) if live else [], compute_lock=compute_lock(worker["pid"] if live else None))
    state["active_training"] = active_training(state["live_processes"])
    write(path.parent / "CURRENT_FULL.json", state)
    print(json.dumps({k: state[k] for k in ("stage", "completed", "failures", "compute_lock", "active_training")}))


def run(path):
    plan, output = checked_plan(path), path.parent
    recipes = full_recipes(plan)
    require_clean_factors(os.environ)
    if not compute_lock(os.getpid())["held_by_worker_ancestor"]:
        raise ValueError("full-data work requires the canonical compute lock")
    fold_plan = json.loads(Path(plan["fold_plan"]).read_text())
    state = json.loads((output / "FULL_STATE.json").read_text())
    state["worker"] = process_snapshot(os.getpid())
    columns = Path(fold_plan["columns"]).read_text().strip()
    common = Path(fold_plan["common_root"])
    base_environment = dict(os.environ, TMPDIR=str(Path.home() / "tmp"), RAYON_NUM_THREADS="1",
                            OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", ZENSIM_MAX_TIER="v3")
    def execute(label, command, binary):
        env = dict(base_environment, REV4_V2_BIN_DIR=str(binary), ZEN_PANEL_BIN=str(binary / "panel"))
        with (output / f"{label}.log").open("x") as log:
            log.write(json.dumps(command) + "\n")
            log.flush()
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    try:
        original = {}
        for seed in range(3):
            directory = Path(plan["original_full"]) / f"full_s{seed}"
            original[seed] = check_fit(directory, seed, fold_plan["common_root_sha256"])
        order_source, order_bin = Path(plan["order_source"]), Path(plan["order_bin"])
        record(output, state, "full-control", binary_sha256=sha(order_bin / "zensim_mlp_train"))
        control = Path(plan["replay_control"]) if "replay_control" in plan else output / "control"
        if "replay_control" not in plan:
            execute("control", fit_command(order_source, common, control, columns, 0), order_bin)
        control_result = check_fit(control, 0, fold_plan["common_root_sha256"])
        parity = require_full_control_parity(Path(plan["original_full"]) / "full_s0", control,
                                           order_bin / "bake_dial_refit", output / "control-parity")
        write(output / "CONTROL_VERIFIED.json", dict(nonrepro_byte_identical=True, parity=parity,
              original=original[0], replay=control_result), fresh=True)
        matching = None
        if "MlicOrder" in recipes:
            record(output, state, "full-matching")
            execute("matching", [sys.executable, "-B", str(order_source / "scripts/rev4_featpot/matched_codec.py"),
                                  str(output / "matching"), "--full"], order_bin)
            matching = json.loads((output / "matching/MANIFESTS.json").read_text())
            if matching["schema"] != "mlic-full-codec-cells-v1" or set(matching["cells"]) != {"0", "1", "2"}:
                raise ValueError("three full-model matching schedules required")
        initial = dict(Mlic=[dict(r, recipe="Mlic", reused=True) for r in original.values()])
        def base_fit(recipe, seed):
            if recipe == "MlicConfidence":
                return confidence_initial(plan, seed, fold_plan["common_root_sha256"])
            study = Path(fold_plan["recipe_authority"][recipe])
            source = order_source if recipe == "MlicOrder" else study / "source"
            binary = order_bin if recipe == "MlicOrder" else study / "bin"
            root = common if recipe == "MlicOrder" else study / recipe / "root"
            dest = output / "initial" / recipe / f"full_s{seed}"
            schedule = matching["cells"][str(seed)] if recipe == "MlicOrder" else None
            if schedule and sha(schedule["path"]) != schedule["sha256"]:
                raise ValueError("full matching schedule changed")
            execute(f"{recipe}-s{seed}", fit_command(source, root, dest, columns, seed, matching=schedule), binary)
            row = check_fit(dest, seed, sha(root / "wide/frozen.json"))
            return dict(row, recipe=recipe, trainer_sha256=sha(binary / "zensim_mlp_train"))
        record(output, state, "initial-full-fits")
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = {pool.submit(base_fit, recipe, seed): (recipe, seed) for recipe in recipes[1:] for seed in range(3)}
            for future in as_completed(futures):
                try:
                    row = future.result()
                    initial.setdefault(row["recipe"], []).append(row)
                    state["completed"].append(dict(stage="initial", recipe=row["recipe"], seed=row["seed"]))
                except BaseException as error:
                    state["failures"].append(dict(cell=futures[future], error=repr(error)))
                record(output, state, "initial-full-fits")
        if state["failures"]:
            raise RuntimeError("full starting recipe failures preserved")
        for rows in initial.values():
            rows.sort(key=lambda r: r["seed"])
        write(output / "INITIAL_FULL.json", initial, fresh=True)
        # Use the existing zero-epoch continuation interface only as an identity
        # control. No held-out interpretation is attached to these full models.
        record(output, state, "full-initialization-zero-step-controls")
        source = Path(fold_plan["warmstart_source"])
        os.environ.update(base_environment, REV4_V2_BIN_DIR=fold_plan["bin"], BYC_ROOT=str(common),
                          ZEN_PANEL_BIN=str(Path(fold_plan["bin"]) / "panel"))
        sys.path[:0] = [str(source / "scripts/rev4_featpot"), str(source / "scripts")]
        import byc_cells as cells
        from subjective_training_leg import linked_subset_exclusions
        if cells.HERE.resolve() != source / "scripts/rev4_featpot":
            raise ValueError("full zero-step owner resolved outside frozen source")
        for recipe in recipes:
            start = Path(initial[recipe][0]["directory"]) / "refit/last.bin"
            dest = output / "zero-step" / recipe
            command = [sys.executable, "-B", str(source / "scripts/rev4_featpot/v2_lodo_mlp.py"),
                "--root", str(common), "--spec", SPEC, "--columns", columns, "--head", "N", "--heldout", "kadid",
                "--seed-index", "0", "--strict-admission", "--data-role-decision", str(common / "human_role_decision.json"),
                "--train-only", "--dest", str(dest), "--init-from", str(start), "--fine-tune-epochs", "0",
                "--fine-tune-lr", "0.00001", "--aic-public-train", f"{fold_plan['aic_folds']}/aic_k0_fit.parquet",
                "--research-extra-group", f"aic_public_train:{fold_plan['aic_folds']}/aic_k0_fit.parquet:8.0:0:rank"]
            exclusions = linked_subset_exclusions(common, "kadid")
            if exclusions:
                command += ["--exclude-datasets", ",".join(exclusions)]
            execute(f"zero-{recipe}", command, Path(fold_plan["bin"]))
            table = Path(fold_plan["aic_source_table"])
            cells.predict_cell(start, table, dest / "initial.tsv", SPEC)
            selected = zero_checkpoint(dest)
            cells.predict_cell(selected, table, dest / "zero.tsv", SPEC)
            require_prediction_parity(dest / "zero.tsv", dest / "initial.tsv")
            write(dest / "CONTROL_VERIFIED.json", dict(byte_identical=True, rows=3920,
                  initial_sha256=sha(start), prediction_sha256=sha(dest / "zero.tsv"),
                  purpose="initialization identity only; no held-out quality interpretation"), fresh=True)
        checked_plan(path)
        record(output, state, "common-full-continuation")
        def continuation(recipe, seed, strength):
            weight, epochs = strength
            arm = f"{recipe}FT{weight}x{epochs}"
            start = Path(initial[recipe][seed]["directory"]) / "refit/last.bin"
            if sha(start) != initial[recipe][seed]["checkpoint_sha256"]:
                raise ValueError("full initial checkpoint changed")
            reused = recipe == "Mlic" and ("reuse_mlic_full" in plan or strength == (16, 8))
            if recipe == "Mlic" and "reuse_mlic_full" in plan:
                dest = reused_mlic(plan, seed, strength)
            else:
                dest = (Path(fold_plan["reuse_mlic"]) / "full" / arm / f"full_s{seed}") if reused else output / "continued" / arm / f"full_s{seed}"
            if not reused:
                execute(f"{arm}-s{seed}", fit_command(Path(fold_plan["warmstart_source"]), common, dest, columns, seed,
                    initial=start, strength=strength, aic=Path(fold_plan["aic_folds"]) / "aic_full_fit.parquet"), Path(fold_plan["bin"]))
            row = check_fit(dest, seed, fold_plan["common_root_sha256"], initial=start, strength=strength)
            return dict(row, recipe=recipe, arm=arm, weight=weight, epochs=epochs, reused=reused)
        continued = []
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = {pool.submit(continuation, recipe, seed, strength): (recipe, seed, strength)
                       for recipe in recipes for seed in range(3) for strength in STRENGTHS}
            for future in as_completed(futures):
                try:
                    row = future.result()
                    continued.append(row)
                    state["completed"].append(dict(stage="continued", recipe=row["arm"], seed=row["seed"]))
                except BaseException as error:
                    state["failures"].append(dict(cell=futures[future], error=repr(error)))
                record(output, state, "common-full-continuation")
        if state["failures"]:
            raise RuntimeError("full continuation failures preserved")
        checked_plan(path)
        write(output / "FULL_FITS_COMPLETE.json", dict(initial=initial, continued=continued,
              plan_sha256=sha(path), fold_panel_sha256=sha(Path(plan["fold_panel"]) / "PANEL_COMPLETE.json"),
              selector=None, assessment_seed=0, production_qualified=False, objective_complete=False), fresh=True)
        record(output, state, "full-fits-complete", next_action="full-data panels and production qualification; no automatic selection")
    except BaseException as error:
        record(output, state, "failed", failures=[*state["failures"], repr(error)])
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("queue", "run", "status"))
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    globals()[args.mode](args.plan.resolve())
