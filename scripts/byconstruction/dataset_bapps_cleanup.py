#!/usr/bin/env python3
"""Verify BAPPS original-byte bindings, then remove only owned unpack scratch.

The phase brief explicitly authorizes removing this scratch after extraction.
Run under the shared heavy lock and resource wrapper. Retained library files,
archives, feature outputs and unpack/hash receipts are never removed.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import pyarrow.parquet as pq

from dataset_features import BUILD_COMMIT, EXTRACTOR_SHA


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    args = ap.parse_args()
    unpack_path = args.root / "bapps-unpack/UNPACK_VERIFIED.json"
    unpack = json.loads(unpack_path.read_text())
    scratch = Path(unpack["scratch"])
    assert scratch.is_relative_to("/mnt/tower/scratch") or scratch.is_relative_to(
        "/var/tmp"
    )
    assert scratch.name == "bapps"
    marker = scratch / ".phase2-owned-scratch"
    keys_path = args.root / "inputs/bapps/keys.parquet"
    assert marker.read_text().strip() == str(keys_path)
    assert sha(keys_path) == unpack["keys_sha256"]
    png_receipt = args.root / "bapps-unpack/png-sha256.tsv"
    assert sha(png_receipt) == unpack["png_receipt_sha256"]
    with png_receipt.open() as stream:
        original = {
            row["relative_path"]: row["source_sha256"]
            for row in csv.DictReader(stream, delimiter="\t")
        }
    assert len(original) == unpack["png_count"]
    bank = args.root / "bank/bapps"
    manifest = json.loads((bank / "_MANIFEST.json").read_text())
    assert (
        manifest["build_commit"] == BUILD_COMMIT
        and manifest["binary_sha256"] == EXTRACTOR_SHA
    )
    assert sha(bank / "keys.parquet") == manifest["keys_sha256"]
    columns = [
        "ref_path",
        "dist_path",
        "reference_file_sha256",
        "distorted_file_sha256",
    ]
    keys = pq.read_table(bank / "keys.parquet", columns=columns)
    assert keys.num_rows == manifest["rows"]
    seen, bindings = set(), 0
    for row in keys.to_pylist():
        for path_field, hash_field in [
            ("ref_path", "reference_file_sha256"),
            ("dist_path", "distorted_file_sha256"),
        ]:
            rel = Path(row[path_field]).relative_to(scratch).as_posix()
            assert original[rel] == row[hash_field], rel
            seen.add(rel)
            bindings += 1
    assert seen == original.keys(), (
        "every unpacked PNG must be bound to native extraction"
    )
    result = {
        "schema": "bapps-original-byte-binding-v1",
        "distinct_original_pngs": len(seen),
        "native_file_hash_bindings": bindings,
        "unpack_receipt_sha256": sha(unpack_path),
        "bank_manifest_sha256": sha(bank / "_MANIFEST.json"),
        "cleanup_owner_sha256": sha(Path(__file__)),
        "scratch_removed": str(scratch),
        "original_library_untouched": True,
        "response_payloads_opened": False,
    }
    failed = scratch.with_name(scratch.name + ".failed-gzip-assumption.bak")
    if failed.exists():
        assert sorted(p.name for p in failed.iterdir()) == [marker.name]
        assert (failed / marker.name).read_text().strip() == str(keys_path)
    with (args.root / "BAPPS_SOURCE_BINDING.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    shutil.rmtree(scratch)
    if failed.exists():
        shutil.rmtree(failed)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
