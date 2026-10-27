"""MlicOrder metadata and frozen-model predictions; Rust owns pair matching.

Registration: benchmarks/byconstruction_2026-10-22.md, 2026-10-27.
No human target comparison selects a pair. Preparation runs in the serialized
heavy-work slot; all source roots and matching models remain immutable.
"""
import argparse
import json
import os
from pathlib import Path

from subjective_training_leg import DATASET_FOLDS, document, log_dataset_read
from v2_common import sha
from stimulus_identity import PAIRS_SHA, promoted_stimuli

GROUP = "human_codec_strata"
OLD = Path("/var/tmp/byc-mixes-20261025")
ROOT_SHA = "a7fd4af0c2bb5a8981be636e5c1cb1a51db57496b7c60d7bf6da01d71dd16952"
FOLDS = ("kadid", "tid2013", "konfig", "cid22_a25", "pipal", "pairs", "mos7")
POLICY = "bidirectional-nearest-frozen-Mlic;quarter-accepted-reference;affine-alternate-cycles"
FULL_REGISTRATION = "c6dcbfdd301063cb1283446d9c244abc90b505db"
FULL_ROOT = Path("/var/tmp/byc-finalists-20261026/Mlic")
FULL_RECORDS = (
    "352a9fabacd012cc6307a396141bb6f52e91825f7aa625ff117b63ae58810964",
    "a7bd8e86dcb8897377302c3079d6f95a7f5d4b446fee0af00f0cc220eb7290cc",
    "6895e40f0fd6beb4d49af8f621c43c3550013cbfa03ddc12a3b368bc5f1f5424",
)


def add_args(parser):
    parser.add_argument("--matched-codec-manifest", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--matched-codec-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--matched-codec-full", action="store_true", help=argparse.SUPPRESS)


def attach(args, groups, root, init_seed, sample_seed, recipe):
    path, pin = args.matched_codec_manifest, args.matched_codec_sha256
    full = getattr(args, "matched_codec_full", False)
    if path is None and pin is None:
        if full:
            raise ValueError("full matched codecs require their explicit manifest")
        return groups, recipe
    if path is None or pin is None or not (args.strict_admission and args.train_only):
        raise ValueError("matched codecs require pinned strict TRAIN admission")
    d = document(path, pin)
    fold = getattr(args, "heldout", None)
    if full:
        if fold is not None or args.exclude_datasets:
            raise ValueError("full matching cannot use an omission route")
        fold = "full"
        if d.get("registration") != FULL_REGISTRATION:
            raise ValueError("full matched-codec registration differs")
    elif fold is None:
        excluded = set(filter(None, args.exclude_datasets.split(",")))
        choices = [f for f, collections in DATASET_FOLDS.items() if excluded == set(collections)]
        if len(choices) != 1:
            raise ValueError("matched-codec confirmation route requires an exact registered omission")
        fold = choices[0]
    if (d.get("schema") != ("mlic-matched-codecs-full-v1" if full else "mlic-matched-codecs-v1") or d.get("role") != "TRAIN"
            or d.get("candidate") != "MlicOrder" or d.get("policy") != POLICY
            or fold not in (("full",) if full else FOLDS[1:]) or d.get("fold") != fold or d.get("seed_index") != args.seed_index
            or (d.get("init_seed"), d.get("sample_seed")) != (init_seed, sample_seed)
            or d.get("original_root_sha256") != ROOT_SHA or sha(root/"wide/frozen.json") != ROOT_SHA):
        raise ValueError("matched-codec cell, seed, policy or original root differs")
    human = [g for g in groups if g[0] == "human"]
    if len(human) != 1 or human[0][3:] != (0, "withinref,rank"):
        raise ValueError("matched codecs require the unchanged fitting human group")
    metadata = document(str(human[0][1])+".manifest.json")
    if metadata["table_sha256"] != d["table_sha256"] or metadata["keys_sha256"] != d["keys_sha256"]:
        raise ValueError("matched-codec table/key binding differs")
    return [(GROUP, *g[1:]) if g[0] == "human" else g for g in groups], {
        **recipe, "matched_codec": dict(path=str(path), sha256=pin)}


def command_flags(recipe, *, strict_admission, head):
    binding = (recipe or {}).get("matched_codec")
    if binding is None:
        return []
    if not strict_admission or head != "N":
        raise ValueError("matched codecs require strict plain CPU training")
    return ["--matched-codec-manifest", binding["path"], "--matched-codec-sha256", binding["sha256"],
            "--no-sample-coverage"]


def cell_binding(registry_path, pin, arm, fold, seed):
    registry = document(registry_path, pin)
    if (arm != "MlicOrder" or registry.get("schema") != "mlic-codec-cells-v1"
            or registry.get("original_root_sha256") != ROOT_SHA
            or set(registry.get("cells", {})) != {f"{f}_s{s}" for f in FOLDS for s in range(3)}):
        raise ValueError("matched-codec registry population differs")
    cell = registry["cells"][f"{fold}_s{seed}"]
    if fold == "kadid":
        if cell != dict(inactive="KADID omitted; unchanged base sampler"):
            raise ValueError("KADID omission must have no matching schedule")
        return None
    d = document(cell["path"], cell["sha256"])
    if d.get("fold") != fold or d.get("seed_index") != seed or d.get("policy") != POLICY:
        raise ValueError("matched-codec manifest identity differs")
    return cell


def full_binding(seed):
    """Exact already-completed full Mlic fit; no omission checkpoint substitution."""
    from v2_common import confirm_seeds
    if seed not in (0, 1, 2):
        raise ValueError("three registered full-data seeds required")
    directory = FULL_ROOT / f"full_s{seed}"
    result = document(directory / "result.json", FULL_RECORDS[seed])
    authority = document(directory / "dataset-authority.json")
    selection = document(directory / "DATASET_GROUP_SELECTION.json")
    if (result["training_only"] is not True or result["execution_contract"] != "registered-fit"
            or result["epochs"] != 120 or result["pairs_per_epoch"] != 50000
            or result["selection"]["epoch_rule"] != "last" or result["selection"]["selected_epoch"] != 119
            or result["seed_index"] != seed or (result["init_seed"], result["sample_seed"]) != confirm_seeds(seed)
            or result["frozen_sha256"] != ROOT_SHA or result["output_stage"] != "e33-identity-knot-tail2"
            or result["selected_bake"] != str(directory / "refit/last.bin")
            or sha(directory / "refit/last.bin") != result["selected_bake_sha256"]
            or selection["root_sha256"] != ROOT_SHA or selection["requested_exclusions"] != []
            or selection["omitted_collections"] != [] or set(authority["cells"]) != {f"full-s{seed}"}):
        raise ValueError("original full Mlic identity differs")
    cell = authority["cells"][f"full-s{seed}"]
    if (cell["route"] != "full" or cell["seed_index"] != seed
            or (cell["init_seed"], cell["sample_seed"]) != confirm_seeds(seed)):
        raise ValueError("original full Mlic authority differs")
    return directory, result, cell


def prepare_full(output):
    from run_processes import compute_lock
    from v2_human_role import preflight_recipe
    from v2_lodo_mlp import predict, strict_training_groups
    if not compute_lock(os.getpid())["held_by_worker_ancestor"]:
        raise ValueError("full matching predictions require the shared compute lock")
    root = OLD / "Mlic/root"
    if sha(root / "wide/frozen.json") != ROOT_SHA:
        raise ValueError("original full matching root changed")
    preflight_recipe(root, root / "human_role_decision.json")
    receipt = document(root / "wide/main/real/receipt.json")
    fit = receipt["legs"]["human_all"]["fit"]
    table = root / fit["rel"]
    _, _, inputs = promoted_stimuli(("kadid_train", "kadid_select"))
    paths = [p for p, pin in inputs.items() if pin == PAIRS_SHA]
    if len(paths) != 1:
        raise ValueError("full matching stimulus source is ambiguous")
    banks = {}
    for member in ("kadid_train", "kadid_select"):
        path = Path("/var/tmp/rev4-featbank/bank") / member / "keys.parquet"
        banks[member] = dict(keys_path=str(path), keys_sha256=sha(path))
    originals = [full_binding(seed) for seed in range(3)]
    output.mkdir()
    manifests = {}
    for seed, (directory, result, cell) in enumerate(originals):
        human = [g for g in cell["groups"] if g["name"] == "human"]
        if len(human) != 1:
            raise ValueError("full matching requires the original human group")
        dest = output / f"full_s{seed}"
        dest.mkdir()
        log_dataset_read(dest, table, fit["rows"], ["ref_basename", "human_score", "f[registered read set]"],
                         "frozen-full-model-TRAIN-matching-prediction", table_sha256=fit["sha256"],
                         labels_used_for_matching=False)
        strict_training_groups([("human", table, human[0]["train_weight"], 0, "withinref,rank")],
                               root / "human_role_decision.json")
        model = directory / "refit/last.bin"
        prediction = dest / "predictions.tsv"
        predict(model, table, prediction)
        d = dict(schema="mlic-matched-codecs-full-v1", registration=FULL_REGISTRATION, role="TRAIN",
            candidate="MlicOrder", policy=POLICY, fold="full", seed_index=seed,
            init_seed=cell["init_seed"], sample_seed=cell["sample_seed"],
            original_root=str(root), original_root_sha256=ROOT_SHA,
            table_sha256=fit["sha256"], keys_sha256=fit["keys_sha256"], codec_banks=banks,
            stimulus_source=dict(path=paths[0], sha256=PAIRS_SHA), metadata_source_sha256=inputs,
            original_cell_record_sha256=FULL_RECORDS[seed],
            original_authority_sha256=sha(directory / "dataset-authority.json"),
            matching_model=str(model), matching_model_sha256=result["selected_bake_sha256"],
            predictions=str(prediction), predictions_sha256=sha(prediction))
        path = dest / "manifest.json"
        path.write_text(json.dumps(d, indent=2) + "\n")
        manifests[str(seed)] = dict(path=str(path), sha256=sha(path))
    path = output / "MANIFESTS.json"
    path.write_text(json.dumps(dict(schema="mlic-full-codec-cells-v1", cells=manifests,
                                   registration=FULL_REGISTRATION), indent=2) + "\n")
    print(json.dumps(dict(path=str(path), sha256=sha(path))))


def prepare(study):
    from run_processes import compute_lock
    from v2_human_role import preflight_recipe
    from v2_lodo_mlp import predict, strict_training_groups
    if not compute_lock(os.getpid())["held_by_worker_ancestor"]:
        raise ValueError("matching predictions require the shared compute lock")
    root = OLD/"Mlic/root"
    if sha(root/"wide/frozen.json") != ROOT_SHA:
        raise ValueError("original Mlic root changed")
    preflight_recipe(root, root/"human_role_decision.json")
    receipt = document(root/"wide/main/real/receipt.json")
    output = study/"matching"
    output.mkdir()  # Never overwrite partial preparation.
    _, _, source_inputs = promoted_stimuli(("kadid_train", "kadid_select"))
    source_paths = [p for p, pin in source_inputs.items() if pin == PAIRS_SHA]
    if len(source_paths) != 1:
        raise ValueError("matched codec stimulus source is ambiguous")
    banks = {}
    for member in ("kadid_train", "kadid_select"):
        path = Path("/var/tmp/rev4-featbank/bank")/member/"keys.parquet"
        banks[member] = dict(keys_path=str(path), keys_sha256=sha(path))
    registry = dict(schema="mlic-codec-cells-v1", original_root_sha256=ROOT_SHA, cells={})
    for seed in range(3):
        for fold in FOLDS:
            key = f"{fold}_s{seed}"
            if fold == "kadid":
                registry["cells"][key] = dict(inactive="KADID omitted; unchanged base sampler")
                continue
            source = OLD/"cells/Mlic"/f"without_{key}"
            record = document(source/"byc_record.json")
            if record["fold"] != fold or record["seed"] != seed or record["tokens"] != "mono:thr:mp32":
                raise ValueError("original matching cell identity differs")
            authority = document(source/"dataset-authority.json")
            if len(authority["cells"]) != 1:
                raise ValueError("original matching authority is ambiguous")
            cell = next(iter(authority["cells"].values()))
            human = [g for g in cell["groups"] if g["name"] == "human"]
            if len(human) != 1:
                raise ValueError("original matching authority lacks fitting human group")
            leg = f"human_without_{fold}" if fold in ("tid2013", "konfig", "cid22_a25") else "human_all"
            fit = receipt["legs"][leg]["fit"]
            table = root/fit["rel"]
            dest = output/key
            dest.mkdir()
            log_dataset_read(dest, table, fit["rows"], ["ref_basename", "human_score", "f[registered read set]"],
                "frozen-model-TRAIN-matching-prediction", table_sha256=fit["sha256"], labels_used_for_matching=False)
            strict_training_groups([("human",table,human[0]["train_weight"],0,"withinref,rank")],
                                   root/"human_role_decision.json")
            model = source/"refit/last.bin"
            if sha(model) != record["last_bin_sha256"]:
                raise ValueError("original matching model changed")
            prediction = dest/"predictions.tsv"
            predict(model, table, prediction)
            d = dict(schema="mlic-matched-codecs-v1", role="TRAIN", candidate="MlicOrder", policy=POLICY,
                fold=fold, seed_index=seed, init_seed=cell["init_seed"], sample_seed=cell["sample_seed"],
                original_root=str(root), original_root_sha256=ROOT_SHA,
                table_sha256=fit["sha256"], keys_sha256=fit["keys_sha256"], codec_banks=banks,
                stimulus_source=dict(path=source_paths[0],sha256=PAIRS_SHA),
                metadata_source_sha256=source_inputs,
                original_cell_record_sha256=sha(source/"byc_record.json"),
                original_authority_sha256=sha(source/"dataset-authority.json"),
                matching_model=str(model), matching_model_sha256=sha(model),
                predictions=str(prediction), predictions_sha256=sha(prediction))
            path = dest/"manifest.json"
            path.write_text(json.dumps(d,indent=2)+"\n")
            registry["cells"][key] = dict(path=str(path), sha256=sha(path))
    path = output/"MANIFESTS.json"
    path.write_text(json.dumps(registry,indent=2)+"\n")
    print(json.dumps(dict(path=str(path),sha256=sha(path))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study",type=Path)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    (prepare_full if args.full else prepare)(args.study)
