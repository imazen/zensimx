"""Bind admitted confidence maps to original Mlic TRAIN rows without changing targets."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import struct

import pyarrow.parquet as pq

ROOT = "/var/tmp/byc-mixes-20261025/Mlic/root"
ARCHIVE = Path("/mnt/tower/output/zensim-datamixes-archive-2026-10-25/Mlic/root")
MAP_PIN = "aefbc62b2aceaed2b648e1d8bd346d2bd5c10dc0fff974fa4287896b6f0b14fb"
LINK_PIN = "486aeae1fcc68dcce5ceb56440ca714179700ac55e32c0086d631bbb34924a2d"
FAMILY_PIN = "657a3f3dae4b3814d8ef534ffbce0839c213450a1f1f231849b384505416519c"
AUTHORITY_PIN = "e8a148ae9a0e957187a989e6a7c47774dab0bc23ef00c05f5f0704eb053ab1ad"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=1, allow_nan=False)
        stream.write("\n")


def bind(scores, references, selected, maps):
    """Native loader order: f64 targets, dense references in first-seen order."""
    if len(scores) != len(references):
        raise ValueError("score/reference row count differs")
    if any(not math.isfinite(v) for v in scores):
        raise ValueError("nonfinite original training target")
    ids, dense = {}, []
    for name in references:
        dense.append(ids.setdefault(name, len(ids)))
    points = [None] * len(scores)
    for row in selected:
        i = row["table_row"]
        native = row["native"]
        key = row["collection"], native["item"], native["reference"]
        if not 0 <= i < len(scores) or points[i] is not None or key not in maps:
            raise ValueError("ambiguous or missing admitted row")
        if "reference_group" in row and references[i] != row["reference_group"]:
            raise ValueError("human reference row order differs")
        value = maps[key]
        if len(value) != 2 or any(not math.isfinite(v) or v < 0 for v in value):
            raise ValueError("invalid admitted teacher distance or SE")
        points[i] = value
    for reference in set(dense):
        covered = [points[i] is not None for i, r in enumerate(dense) if r == reference]
        if any(covered) and not all(covered):
            raise ValueError(
                "partially covered reference cannot receive confidence weights"
            )
    digest = hashlib.sha256()
    for value, ref in zip(scores, dense):
        digest.update(struct.pack("<dI", value, ref))
    return dict(
        rows=len(scores),
        scores_refs_sha256=digest.hexdigest(),
        points=points,
        covered_rows=sum(p is not None for p in points),
        references=len(ids),
    )


def prepare(base, output):
    output.mkdir()
    inputs = {}

    def document(path, pin):
        if sha(path) != pin:
            raise ValueError("pinned confidence input changed: " + str(path))
        inputs[str(path)] = pin
        return json.loads(Path(path).read_text())

    maps_index = document(base / "RATING_MAPS_COMPLETE.json", MAP_PIN)
    link_index = document(base / "LINKING_FITS_COMPLETE.json", LINK_PIN)
    families = document(base / "rating-families-reviewed-run01.json", FAMILY_PIN)
    authorities = document(base / "original-cell-authorities/INDEX.json", AUTHORITY_PIN)
    population = document(
        base / "original-rating-populations.json",
        "e8b80de3006829f87ad8593b819b06d844612419f5eab7b49f2d0c8e379ff078",
    )
    for fold, records in authorities.items():
        for index in (maps_index, link_index):
            entry = index["folds"][fold]
            result = document(Path(entry["result"]), entry["sha256"])
            if result["status"] != "COMPLETE":
                raise ValueError(
                    "complete native calibration required before target reads"
                )
        for record in records:
            document(Path(record["path"]), record["sha256"])
    # Freeze every exact TRAIN table/column before interpreting targets.
    needed = {}
    for fold in authorities:
        for row in families["rating_scopes"][fold]["rows"]:
            name = row["table"]
            original = population["tables"][name]
            path = ARCHIVE / Path(original["table"]).relative_to(ROOT)
            needed[str(path)] = original["table_sha256"]
    for path, pin in needed.items():
        if sha(path) != pin:
            raise ValueError("original TRAIN table changed")
        inputs[path] = pin
    write(
        output / "READ_EVENT.json",
        dict(
            purpose="registered-contrast-normalizers",
            role="TRAIN",
            started_utc=datetime.now(timezone.utc).isoformat(),
            inputs=inputs,
            target_tables=needed,
            columns=["human_score", "ref_basename"],
            target_scale=1,
            feature_values_read=False,
            source_sha256=sha(__file__),
        ),
    )
    tables = {}
    for name, description in population["tables"].items():
        path = ARCHIVE / Path(description["table"]).relative_to(ROOT)
        if str(path) not in needed:
            continue
        table = pq.read_table(path, columns=["human_score", "ref_basename"])
        if (
            table.column("human_score").null_count
            or table.column("ref_basename").null_count
        ):
            raise ValueError("null original TRAIN target or reference")
        tables[name] = table.to_pydict()
    manifests = {}
    for fold, records in authorities.items():
        mr = maps_index["folds"][fold]
        mapped = document(Path(mr["result"]), mr["sha256"])
        lr = link_index["folds"][fold]
        linking = document(Path(lr["result"]), lr["sha256"])
        if mapped["status"] != "COMPLETE" or linking["status"] != "COMPLETE":
            raise ValueError("complete native calibration required")
        lookup = {}
        for ds, result in mapped["maps"].items():
            if not result["input_valid"]:
                raise ValueError("native teacher map not valid")
            for item in result["items"]:
                key = ds, item["item"], item["ref"]
                if key in lookup:
                    raise ValueError("duplicate calibrated native item")
                lookup[key] = [item["x"], item["se_jod"]]
        selections = {}
        for row in families["rating_scopes"][fold]["rows"]:
            group = "human" if row["table"].startswith("human_") else row["table"]
            selections.setdefault(group, []).append(row)
        bindings = {}
        for group, selection in selections.items():
            names = {row["table"] for row in selection}
            if len(names) != 1:
                raise ValueError("one original fitting table per group required")
            values = tables[next(iter(names))]
            bindings[group] = bind(
                values["human_score"], values["ref_basename"], selection, lookup
            )
        for record in records:
            authority = document(Path(record["path"]), record["sha256"])
            cell = authority["cells"][record["cell"]]
            if (cell["epochs"], cell["pairs_per_epoch"], cell["selected_epoch"]) != (
                120,
                50000,
                119,
            ):
                raise ValueError("original fixed training budget differs")
            groups = []
            for g in cell["groups"]:
                group = dict(
                    name=g["name"],
                    train_weight=g["train_weight"],
                    validation_weight=g["val_weight"],
                    mode=g["mode"],
                )
                if g["name"] in bindings:
                    if (
                        g["mode"] != "withinref,rank"
                        or g["val_weight"] != 0
                        or g["train_weight"] <= 0
                    ):
                        raise ValueError(
                            "confidence applies only to registered TRAIN rank groups"
                        )
                    group["binding"] = bindings[g["name"]]
                groups.append(group)
            if {g["name"] for g in groups if "binding" in g} != set(bindings):
                raise ValueError(
                    "original cell groups omit an admitted confidence population"
                )
            path = output / (record["cell"] + ".json")
            write(
                path,
                dict(
                    schema="mlic-contrast-confidence-v1",
                    candidate="MlicConfidence",
                    root_sha256=population["original_root_sha256"],
                    fold=fold,
                    seed_index=cell["seed_index"],
                    init_seed=cell["init_seed"],
                    sample_seed=cell["sample_seed"],
                    epochs=120,
                    pairs_per_epoch=50000,
                    selected_epoch=119,
                    n_features=len(cell["feature_ids"]),
                    hidden=128,
                    sigma_L=linking["sigma_L"],
                    item_floor=0.05,
                    authority_sha256=record["sha256"],
                    input_sha256=dict(inputs),
                    groups=groups,
                    tie_policy="existing-native-owner",
                    pair_law="within-ref-u64-modulo",
                    absolute_loss_changed=False,
                    group_masses_changed=False,
                ),
            )
            manifests[record["cell"]] = dict(path=str(path), sha256=sha(path))
    write(
        output / "MANIFESTS.json",
        dict(
            schema="mlic-contrast-manifests-v1",
            cells=manifests,
            source_sha256=sha(__file__),
            read_event_sha256=sha(output / "READ_EVENT.json"),
            native_normalizers_computed=False,
            candidate_fits_started=False,
        ),
    )
    print(
        json.dumps(
            dict(cells=len(manifests), index_sha256=sha(output / "MANIFESTS.json"))
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("base", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    prepare(a.base, a.output)
