"""Freeze omission source identities from Mlic's original identifier receipts.

The full E21 assessment keys, not just the smaller fitting subset, define
each human omission. No feature, target or uncertainty column is interpreted.
This prepares identities for the later independent source-family join; it
does not itself clear donors or authorize response reads.
"""
import argparse
import json
from pathlib import Path
import sys

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rev4_featpot"))
from subjective_training_leg import DATASET_FOLDS, document, sha
from bank_pins import PINNED_MANIFEST_SHA

# Exact frozen root; kept separate from its transport/archive location.
ROOT_SHA = "a7fd4af0c2bb5a8981be636e5c1cb1a51db57496b7c60d7bf6da01d71dd16952"
MEMBERS = {
    "kadid": {"kadid_train", "kadid_select"},
    "tid2013": {"tid2013"},
    "konfig": {"konfig_train", "konfig_val"},
    "cid22_a25": {"cid22_a25"},
}
KEYS = {"pair_key", "source_row_id", "ref_basename", "member_set"}
DATASET_KEYS = {"row", "observation", "source", "reference", "family", "role", "tier"}


def identity(row):
    if set(row) != KEYS or type(row["source_row_id"]) is not int or row["source_row_id"] < 0:
        raise ValueError("original observation-key schema required")
    return tuple(row[name] for name in sorted(KEYS))


def human_scope(fold, all_fit, retained_fit, assessment, banks):
    """Preserve the complete omitted assessment population, including non-fit references."""
    members = MEMBERS[fold]
    if not all_fit or not assessment:
        raise ValueError("nonempty frozen human populations required")
    for rows in (all_fit, retained_fit, assessment):
        if len({identity(r) for r in rows}) != len(rows):
            raise ValueError("duplicate observation identity")
    expected = [identity(r) for r in all_fit if r["member_set"] not in members]
    if [identity(r) for r in retained_fit] != expected:
        raise ValueError("omission fitting keys differ from the exact original filter")
    if any(r["member_set"] not in members for r in assessment):
        raise ValueError("assessment contains a different omitted collection")
    assessed = {identity(r) for r in assessment}
    removed = [r for r in all_fit if r["member_set"] in members]
    if not removed or not all(identity(r) in assessed for r in removed):
        raise ValueError("omitted fitting observations are not in the full assessment")
    references = {}
    for row in assessment:
        source = banks[row["member_set"]][row["pair_key"]]
        name = row["ref_basename"]
        if source["ref_group"] != name:
            raise ValueError("bank reference differs from frozen reference identity")
        entry = references.setdefault(name, dict(reference_group=name, source_paths=set(), members=set(), pixels=set()))
        entry["source_paths"].add(source["ref_path"])
        entry["members"].add(row["member_set"])
        pixel = (source["ref_pixels_sha256"], source["width"], source["height"])
        if len(pixel[0]) != 64 or any(type(v) is not int or v <= 0 for v in pixel[1:]):
            raise ValueError("native reference pixel identity or dimensions differ")
        entry["pixels"].add(pixel)
    return dict(omitted_members=sorted(members), removed_fit_rows=len(removed),
        retained_fit_rows=len(retained_fit), full_assessment_rows=len(assessment),
        fitting_reference_count=len({r["ref_basename"] for r in removed}),
        full_assessment_reference_count=len(references),
        reference_identities=[dict(reference_group=k, source_paths=sorted(v["source_paths"]),
            members=sorted(v["members"]), pixels=[dict(sha256=h, width=w, height=t) for h, w, t in sorted(v["pixels"])])
            for k, v in sorted(references.items())])


def dataset_families(collection, rows, expected_rows):
    if len(rows) != expected_rows or not rows:
        raise ValueError("dataset fitting population incomplete")
    families = set()
    for i, row in enumerate(rows):
        if (set(row) != DATASET_KEYS or row["role"] != "fit" or row["row"] != i
                or type(row["family"]) is not int or row["family"] < 0
                or not row["observation"].startswith(collection + ":")):
            raise ValueError("original dataset observation/family identity differs")
        families.add(row["family"])
    return sorted(families)


def prepare(root, bank, output):
    frozen = document(root / "wide/frozen.json", ROOT_SHA)
    wide_path = root / "wide/main/real/receipt.json"
    wide = document(wide_path, frozen["wide_receipts"]["main/real"])
    inputs = {str(root / "wide/frozen.json"): ROOT_SHA, str(wide_path): sha(wide_path)}
    banks = {}
    for member in sorted(set().union(*MEMBERS.values())):
        mp = bank / member / "_MANIFEST.json"
        manifest = document(mp, PINNED_MANIFEST_SHA[member])
        kp = bank / member / "keys.parquet"
        pin = manifest["files"]["keys.parquet"]["sha256"]
        if sha(kp) != pin:
            raise ValueError("original promoted reference keys changed")
        inputs[str(mp)] = PINNED_MANIFEST_SHA[member]
        inputs[str(kp)] = pin
        rows = pq.read_table(kp, columns=["pair_key", "ref_group", "ref_path", "ref_pixels_sha256", "width", "height"]).to_pylist()
        banks[member] = {r["pair_key"]: r for r in rows}
        if len(banks[member]) != len(rows):
            raise ValueError("ambiguous promoted pixel key")

    def keys(description):
        path = root / description["rel"]
        declaration = Path(str(path) + ".manifest.json")
        d = document(declaration, description["manifest_sha256"])
        key_path = path.with_suffix(".keys.parquet")
        if (d["table_sha256"] != description["sha256"]
                or d["keys_sha256"] != description["keys_sha256"]
                or sha(key_path) != description["keys_sha256"]):
            raise ValueError("frozen metadata/key binding differs")
        inputs[str(declaration)] = description["manifest_sha256"]
        inputs[str(key_path)] = description["keys_sha256"]
        table = pq.read_table(key_path)
        if set(table.schema.names) != KEYS or table.num_rows != description["rows"]:
            raise ValueError("original key schema or population differs")
        return table.to_pylist()

    all_fit = keys(wide["legs"]["human_all"]["fit"])
    folds = {}
    for fold in MEMBERS:
        retained = keys(wide["legs"][f"human_without_{fold}"]["fit"])
        assessment = keys(wide["legs"][fold]["full"])
        human = human_scope(fold, all_fit, retained, assessment, banks)
        if human["full_assessment_reference_count"] != wide["legs"][fold]["full"]["references"]:
            raise ValueError("full assessment reference coverage differs")
        folds[fold] = dict(human=human, prior_dataset_family_ids=[], omitted_collections=[])
    packet_path = root / "datasets/receipt.json"
    packet = document(packet_path, frozen["auxiliary_files"]["datasets/receipt.json"])
    inputs[str(packet_path)] = sha(packet_path)
    groups = {}
    for group in packet["groups"]:
        collection = group["collection"]
        if collection in groups:
            raise ValueError("ambiguous original dataset group")
        key_path = root / "datasets" / Path(group["keys_path"]).name
        declaration = root / "datasets" / (Path(group["path"]).name + ".manifest.json")
        d = document(declaration, group["manifest_sha256"])
        if d["native_role"] != "TRAIN" or group["train_weight"] <= 0:
            raise ValueError("original active TRAIN group required")
        rows = document(key_path, group["keys_sha256"])
        inputs[str(key_path)] = group["keys_sha256"]
        inputs[str(declaration)] = group["manifest_sha256"]
        groups[collection] = dict(prior_family_ids=dataset_families(collection, rows, group["rows"]),
            fitting_rows=group["rows"])
    for fold, collections in DATASET_FOLDS.items():
        present = sorted(set(collections) & groups.keys())
        families = sorted({family for c in present for family in groups[c]["prior_family_ids"]})
        if not present or not families:
            raise ValueError("original omission has no bound fitting families")
        folds[fold] = dict(human=None, prior_dataset_family_ids=families,
            omitted_collections=list(collections), present_original_collections=present,
            absent_original_collections=sorted(set(collections) - groups.keys()))
    folds["full"] = dict(human=None, prior_dataset_family_ids=[], omitted_collections=[])
    value = dict(schema="mlic-confidence-omission-identities-v1", original_root_sha256=ROOT_SHA,
        input_sha256=inputs, producer_sha256=sha(__file__), response_values_parsed=False,
        uncertainty_values_parsed=False, feature_values_parsed=False, source_family_join_complete=False,
        response_reads_authorized=False, dataset_groups=groups, folds=folds,
        next_action="join every reference and prior-family alias to independently validated native source families")
    output.mkdir()
    path = output / "OMISSION_IDENTITIES.json"
    path.write_text(json.dumps(value, indent=1) + "\n")
    print(json.dumps(dict(path=str(path), sha256=sha(path),
        human_reference_counts={k: v["human"]["full_assessment_reference_count"]
            for k, v in folds.items() if v["human"]},
        dataset_family_counts={k: len(v["prior_dataset_family_ids"]) for k, v in folds.items()})))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("bank", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.root, args.bank, args.output)
