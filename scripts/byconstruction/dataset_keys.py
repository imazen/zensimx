#!/usr/bin/env python3
"""Prepare label-free assessment keys from accepted frozen memberships.

No subjective table is read. Extraction belongs to rev5_bank.py extract via
its instrument-manifest route. Mapping rows retain every FROZEN membership,
including explicit skips; A/B studies map each side to its own feature pair.
"""

from pathlib import Path
import argparse
import ast
import collections
import hashlib
import json
import re
import time
import sys

import pyarrow as pa
import pyarrow.parquet as pq

EXTRACTOR_SHA = "ed4957417339779dd8ed25fe59a6166326ba9db86d5c1530b1e1c788c9d614d8"
COLUMNS = [
    "dataset_id",
    "input_table",
    "row_index",
    "item_path",
    "reference_path",
    "a_path",
    "b_path",
    "native_subset",
    "role",
    "family_id",
    "membership_state",
]
MAP_SCHEMA = pa.schema(
    [
        ("dataset_id", pa.string()),
        ("input_table", pa.string()),
        ("membership_row_index", pa.int64()),
        ("role", pa.string()),
        ("family_id", pa.int64()),
        ("side", pa.string()),
        ("pair_key", pa.string()),
        ("skip_reason", pa.string()),
    ]
)
KEY_SCHEMA = pa.schema(
    [
        ("row_id", pa.int64()),
        ("pair_key", pa.string()),
        ("ref_path", pa.string()),
        ("dist_path", pa.string()),
        ("origin", pa.string()),
    ]
)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def safe_rel(value):
    p = Path(value)
    if not value or p.is_absolute() or ".." in p.parts:
        raise ValueError("a nonempty native relative image path is required")
    if p.suffix.lower() not in {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tif",
        ".tiff",
        ".ppm",
        ".pgm",
    }:
        raise ValueError("native image path required")
    return p


def image_pairs(dataset, row, spcd_mode):
    """Return declared stimulus pairs, never manufacture a pristine reference."""
    ref = row["reference_path"]
    if dataset == "diffiqa":
        assert not ref
        return (
            [],
            "No pristine/full reference; a is an imperfect input, not a substitute pristine image",
        )
    if dataset == "spcd":
        if spcd_mode == "pending":
            raise ValueError(
                "SPCD native peer-pair interpretation awaits the coordinator"
            )
        if spcd_mode == "skip":
            return (
                [],
                "Native colour-difference peer pair supplies no pristine reference",
            )
        assert not ref and row["a_path"] and row["b_path"]
        return [("native_pair", row["a_path"], row["b_path"])], None
    if not ref:
        return [], "No native full-reference image supplied"
    if dataset == "colourlab-chroma-masking":
        # The adapter's public metadata order is Cr[0:480], Cb[480:960].
        # item/ref both name the mask; rated target images are shipped separately.
        i = row["row_index"]
        assert 0 <= i < 960
        channel, prefix = ("Cr", "cr") if i < 480 else ("Cb", "cb")
        number = i % 480 + 1
        assert Path(ref).name.lower() == f"t{number:03d}.bmp"
        target = str(
            Path(ref).parents[1]
            / f"masks_and_{channel}_targets"
            / f"{prefix}{number:03d}.bmp"
        )
        return [(channel, ref, target)], None
    if row["item_path"]:
        return [("item", ref, row["item_path"])], None
    if row["a_path"] and row["b_path"]:
        return [("a", ref, row["a_path"]), ("b", ref, row["b_path"])], None
    return [], "No distorted image path supplied for native reference"


def prepare(args):
    started = time.monotonic()
    summary = json.loads(
        (args.admission / "frozen-membership-summary.json").read_text()
    )
    dataset = args.collection
    assert dataset in summary
    token = dataset.replace("-", "_")
    assert re.fullmatch("[a-z0-9_]+", token)
    source = args.admission / dataset
    for line in (source / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert sha(source / name) == digest, (dataset, name)
    membership = source / "membership.parquet"
    table = pq.ParquetFile(membership)
    assert set(COLUMNS) <= set(table.schema_arrow.names)
    assert not any(
        name in table.schema_arrow.names
        for name in ["score", "human_score", "target", "MOS", "DMOS"]
    )
    assert sha(args.extractor) == EXTRACTOR_SHA
    owner = Path(__file__).resolve().parents[1] / "rev4_featpot/rev5_bank.py"
    tree = ast.parse(owner.read_text())
    admitted_columns = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "ASSESSMENT_KEY_COLUMNS"
            for t in node.targets
        )
    )
    assert set(KEY_SCHEMA.names) <= admitted_columns
    out = args.out / "inputs" / dataset
    out.mkdir(parents=True, exist_ok=False)
    content = args.library / dataset / "content"
    if args.content_root is not None:
        assert args.content_root.is_absolute() and args.content_root.is_relative_to(args.library)
        content = args.content_root
    if dataset == "bapps":
        assert args.bapps_root.is_absolute()
        content = args.bapps_root
    pairs = {}
    by_role = collections.Counter()
    mapped_roles = collections.Counter()
    skipped_roles = collections.Counter()
    reasons = collections.Counter()
    mapping_rows = []
    rows = 0
    side_counts = collections.Counter()
    mapping = out / "membership-to-pairs.parquet"
    with pq.ParquetWriter(mapping, MAP_SCHEMA, compression="zstd") as writer:
        for batch in table.iter_batches(
            batch_size=1024, columns=COLUMNS, use_threads=False
        ):
            for row in batch.to_pylist():
                if row["membership_state"] != "FROZEN":
                    continue
                assert row["dataset_id"] == dataset
                rows += 1
                role = row["role"]
                by_role[role] += 1
                edges, reason = image_pairs(dataset, row, args.spcd_mode)
                base = {
                    "dataset_id": dataset,
                    "input_table": row["input_table"],
                    "membership_row_index": row["row_index"],
                    "role": role,
                    "family_id": row["family_id"],
                }
                if reason:
                    skipped_roles[role] += 1
                    reasons[reason] += 1
                    mapping_rows.append(
                        base | {"side": "none", "pair_key": None, "skip_reason": reason}
                    )
                else:
                    mapped_roles[role] += 1
                    for side, native_ref, native_dist in edges:
                        reference = str(content / safe_rel(native_ref))
                        distorted = str(content / safe_rel(native_dist))
                        pair_key = (
                            token
                            + ":"
                            + hashlib.sha256(
                                json.dumps(
                                    [dataset, native_ref, native_dist],
                                    separators=(",", ":"),
                                ).encode()
                            ).hexdigest()
                        )
                        origin = str(row["family_id"])
                        if pair_key in pairs:
                            assert pairs[pair_key][1:] == (reference, distorted, origin)
                        else:
                            pairs[pair_key] = (len(pairs), reference, distorted, origin)
                        mapping_rows.append(
                            base
                            | {"side": side, "pair_key": pair_key, "skip_reason": None}
                        )
                        side_counts[role] += 1
                if len(mapping_rows) >= 4096:
                    writer.write_table(
                        pa.Table.from_pylist(mapping_rows, schema=MAP_SCHEMA)
                    )
                    mapping_rows.clear()
            print(
                f"{dataset} membership rows {rows}; distinct pairs {len(pairs)}",
                flush=True,
            )
        if mapping_rows:
            writer.write_table(pa.Table.from_pylist(mapping_rows, schema=MAP_SCHEMA))
    expected = {
        key.removesuffix(" FROZEN"): value
        for key, value in summary[dataset]["frozen_counts"].items()
        if key.endswith(" FROZEN")
    }
    assert dict(by_role) == expected, (by_role, expected)
    keys = out / "keys.parquet"
    chunk = []
    with pq.ParquetWriter(keys, KEY_SCHEMA, compression="zstd") as writer:
        for pair_key, (index, ref, dist, origin) in pairs.items():
            chunk.append(
                {
                    "row_id": index,
                    "pair_key": pair_key,
                    "ref_path": ref,
                    "dist_path": dist,
                    "origin": origin,
                }
            )
            if len(chunk) == 4096:
                writer.write_table(pa.Table.from_pylist(chunk, schema=KEY_SCHEMA))
                chunk.clear()
        if chunk:
            writer.write_table(pa.Table.from_pylist(chunk, schema=KEY_SCHEMA))
    receipt = {
        "build_commit": args.build_commit,
        "dataset_id": dataset,
        "collection_token": token,
        "membership_sha256": sha(membership),
        "membership_receipts_sha256": sha(source / "SHA256SUMS"),
        "frozen_membership_rows": rows,
        "membership_rows_by_role": by_role,
        "mapped_membership_rows_by_role": mapped_roles,
        "pair_occurrences_by_role": side_counts,
        "skipped_membership_rows_by_role": skipped_roles,
        "skip_reasons": reasons,
        "distinct_pairs": len(pairs),
        "keys_sha256": sha(keys),
        "mapping_sha256": sha(mapping),
        "label_values_read": False,
        "membership_columns_read": COLUMNS,
        "key_preparer_sha256": sha(__file__),
        "bank_owner_sha256": sha(owner),
        "spcd_mode": args.spcd_mode if dataset == "spcd" else None,
        "elapsed_seconds": time.monotonic() - started,
        "state": "KEYS_ONLY; pixel extraction and feature verification pending",
    }
    write_json(out / "PREPARATION.json", receipt)
    if pairs:
        write_json(
            out / "instrument.json",
            {
                "schema": "rev5-assessment-keys-v1",
                "labels_read": False,
                "rows": len(pairs),
                "keys": {"path": str(keys), "sha256": sha(keys)},
                "extractor": {"path": str(args.extractor), "sha256": EXTRACTOR_SHA},
                "immutable_roots": [str(content), str(out)],
                "instrument": {
                    "name": token,
                    "source": {
                        "membership_sha256": receipt["membership_sha256"],
                        "preparation_sha256": sha(out / "PREPARATION.json"),
                        "bank_owner_sha256": sha(owner),
                    },
                    "selection": "all FROZEN roles; distinct native image pairs; no label selection",
                },
                "exposure_freeze": {
                    "state": "features-only",
                    "labels_read": False,
                    "required_before_labels": [
                        "coordinator approval of the leg proposal",
                        "TRAIN-only source and native protocol binding",
                    ],
                    "scope": "No training, selection or assessment label values read",
                },
            },
        )
    (out / "SHA256SUMS").write_text(
        "".join(
            sha(p) + "  " + p.name + "\n" for p in sorted(out.iterdir()) if p.is_file()
        )
    )
    print(json.dumps(receipt), flush=True)


def hdr_source_partition(sources, reserved):
    """Nearest 70/30 reference count, keeping related frames and reservations whole.

    Ties use a fixed hash of family IDs, never labels or metric results.
    Returned TRAIN roles are prospective until native source screening passes.
    """
    import itertools

    families = collections.defaultdict(list)
    for source, family in sources.items():
        families[family].append(source)
    if not set(reserved) <= set(sources):
        raise ValueError("unknown reserved HDR source")
    excluded = {sources[s] for s in reserved}
    eligible = sorted(set(families) - excluded)
    candidates = []
    for n in range(1, len(eligible) + 1):
        for train in itertools.combinations(eligible, n):
            count = sum(len(families[f]) for f in train)
            if count == len(sources):
                continue
            tie = hashlib.sha256(("hdr-source-split-v1\n" + "\n".join(train)).encode()).hexdigest()
            candidates.append((abs(10 * count - 7 * len(sources)), tie, train))
    if not candidates:
        raise ValueError("no source-disjoint nonempty TRAIN/EVAL partition")
    train = set(min(candidates)[2])
    return {s: "TRAIN" if f in train else "EVAL" for s, f in sorted(sources.items())}


def freeze_hdr_membership(args):
    """Read only explicit release metadata; never enumerate or open target files."""
    import csv

    names = ("rousselot-4kdtb", "itm-color-graded-hdr-images")
    rows = []
    pins = {}
    summaries = {}
    for name in names:
        root = args.library / name
        inventory = root / "SHA256SUMS"
        pins[str(inventory)] = sha(inventory)
        files = {}
        for line in inventory.read_text().splitlines():
            digest, rel = line.split("  ", 1)
            if len(digest) != 64 or rel in files or Path(rel).is_absolute() or ".." in Path(rel).parts:
                raise ValueError("invalid or duplicate inventory path")
            files[rel] = digest
        sources = {}
        refs = {}
        reserved = {}
        if name == "rousselot-4kdtb":
            prefix = "content/Subjective_test_4Kdtb/Original_Images/"
            for rel in files:
                if rel.startswith(prefix) and rel.endswith("_RGB_444.hdr"):
                    source = Path(rel).name.removesuffix("_RGB_444.hdr")
                    family = source.split("_", 1)[0]
                    if family not in {"Bike", "Regatta"}:
                        raise ValueError("unreviewed 4Kdtb source family")
                    sources[source] = "4kdtb:" + family
                    refs[source] = rel
            assert len(sources) == 8 and collections.Counter(sources.values()) == {"4kdtb:Bike": 4, "4kdtb:Regatta": 4}
            pairs = []
            for rel in files:
                if "/Distorted_Images/" in rel and rel.endswith(".hdr"):
                    source = Path(rel).parent.name
                    if source not in sources:
                        raise ValueError("distortion without reference")
                    pairs.append((source, rel))
            assert len(pairs) == 96
        else:
            source_csv = root / "labels/images.csv"
            pins[str(source_csv)] = sha(source_csv)
            with source_csv.open(newline="") as stream:
                for row in csv.DictReader(stream):
                    source = Path(row["HDR"]).stem
                    if source in sources or Path(row["HDR"]).name != row["HDR"]:
                        raise ValueError("duplicate/invalid Ghent source")
                    family = "ghent:" + source
                    if source in {"00409", "00008880"}:
                        family = "tears-of-steel"
                    if source in {"066", "195"}:
                        family = "si-hdr:" + source
                        reserved[source] = "SI-HDR evaluation source alias"
                    if source == "showgirl_02_302300":
                        family = "stuttgart:showgirl"
                        reserved[source] = "conservative HDdtb Showgirl family reservation"
                    sources[source] = family
                    refs[source] = "content/hdr/" + row["HDR"]
            assert len(sources) == 12
            pairs = [(Path(rel).stem, rel) for rel in files
                     if rel.startswith("content/hdr_itmo/") and Path(rel).stem in sources and rel.endswith(".exr")]
            assert len(pairs) == 108
        counts = collections.Counter(s for s, _ in pairs)
        assert set(counts) == set(sources) and set(counts.values()) == {12 if name == names[0] else 9}
        proposed_roles = hdr_source_partition(sources, reserved)
        # Owner fallback: Ghent has no established EXR-value-to-nits contract.
        # Preserve its proposed partition for exposure; grant no new TRAIN use.
        roles = proposed_roles if name == names[0] else {s: "EVAL" for s in sources}
        summaries[name] = {"sources": sources, "roles": roles, "proposed_roles": proposed_roles, "reserved": reserved,
                           "source_counts": dict(collections.Counter(roles.values())),
                           "pair_counts": dict(collections.Counter(roles[s] for s, _ in pairs))}
        for source, dist in sorted(pairs):
            ref = refs[source]
            rows.append(dict(dataset_id=name, source_id=source, family_id=sources[source],
                             role=roles[source], ref_path=str(root / ref), dist_path=str(root / dist),
                             ref_sha256=files[ref], dist_sha256=files[dist],
                             admission_state="SOURCE_SCREEN_AND_LIGHT_CONTRACT_PENDING"))
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "memberships.json", rows)
    write_json(args.out / "_MANIFEST.json", dict(
        schema="hdr-human-source-membership-v1", build_commit=args.build_commit,
        rule="hdr-source-split-v1; minimize abs(10*n_train-7*n_total); hashed family-set tie break",
        owner_quote="for hdr, we probs need human training data, as well as eval sets",
        owner_override_collections=list(names), metadata_sha256=pins,
        memberships_sha256=sha(args.out / "memberships.json"), sources=summaries,
        producer_sha256=sha(__file__), labels_read=False, pixels_read=False,
        training_permitted=False,
        remaining_gates=["native source-family screen and retained-match review",
                         "verified photometric/primaries contract", "registered feature extraction"],
        exposure="Historically evaluated; new EVAL membership is not an untouched test"))
    print(json.dumps(summaries, indent=2))


def hdr_membership_main():
    ap = argparse.ArgumentParser(description="Freeze label-blind HDR source roles; admission remains gated")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--build-commit", required=True)
    ap.add_argument("--library", type=Path, default=Path("/mnt/tower/input/iqa/datasets"))
    freeze_hdr_membership(ap.parse_args(sys.argv[2:]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("collection")
    ap.add_argument("--admission", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--extractor", type=Path, required=True)
    ap.add_argument("--build-commit", required=True)
    ap.add_argument(
        "--library", type=Path, default=Path("/mnt/tower/input/iqa/datasets")
    )
    ap.add_argument("--content-root", type=Path, help="explicit admitted parent content root")
    ap.add_argument(
        "--bapps-root",
        type=Path,
        default=Path("/mnt/tower/scratch/dataset-features-rev7-20261025/bapps"),
    )
    ap.add_argument(
        "--spcd-mode", choices=["pending", "skip", "direct"], default="pending"
    )
    prepare(ap.parse_args())


if __name__ == "__main__":
    if sys.argv[1:2] == ["hdr-membership"]:
        hdr_membership_main()
    else:
        main()
