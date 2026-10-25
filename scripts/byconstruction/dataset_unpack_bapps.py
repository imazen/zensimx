#!/usr/bin/env python3
"""Unpack only frozen BAPPS PNGs from retained library archives; never labels.

Run under the shared heavy lock and resource wrapper. Original PNG bytes are
verified against the supplied library index. Scratch is caller-owned and may
be removed only after feature extraction and preservation of this receipt.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import tarfile

import pyarrow.parquet as pq

ARCHIVES = {
    "train": (
        "twoafc_train.tar.gz",
        "b4f5c772c7cd88ac44bcb4ff0741745d434c05f0094f2bfb262ddcd0055bf393",
    ),
    "val": (
        "twoafc_val.tar.gz",
        "41486671b06de923e7ba277c4797d4d9d86d609863ba35ecdb576133f8b9d859",
    ),
}


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keys", type=Path, required=True)
    ap.add_argument("--library", type=Path, required=True)
    ap.add_argument("--scratch", type=Path, required=True)
    ap.add_argument("--receipt-dir", type=Path, required=True)
    args = ap.parse_args()
    keys = pq.read_table(args.keys, columns=["ref_path", "dist_path"])
    wanted = {
        str(Path(value).relative_to(args.scratch))
        for col in keys.columns
        for value in col.to_pylist()
    }
    for rel in wanted:
        p = Path(rel)
        assert not p.is_absolute() and ".." not in p.parts
        assert len(p.parts) == 4 and p.parts[0] in ARCHIVES
        assert p.parts[2] in {"ref", "p0", "p1"} and p.suffix == ".png"
    index = args.library / "datasets/bapps/content_index.tsv"
    expected = {}
    with index.open() as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if row["rel"] in wanted:
                expected[row["rel"]] = row["source_sha256"]
    assert expected.keys() == wanted, "library PNG identity coverage"
    args.scratch.mkdir(parents=True, exist_ok=False)
    (args.scratch / ".phase2-owned-scratch").write_text(str(args.keys) + "\n")
    args.receipt_dir.mkdir(parents=True, exist_ok=False)
    archives = []
    found = set()
    with (args.receipt_dir / "png-sha256.tsv").open("x") as receipt:
        receipt.write("relative_path\tsource_sha256\tbytes\n")
        for split, (filename, expected_sha) in ARCHIVES.items():
            archive = (
                args.library / "raw/huggingface/AaronCIH__IQA_dataset/BAPPS" / filename
            )
            actual_sha = sha(archive)
            assert actual_sha == expected_sha, "retained original archive changed"
            archives.append({"path": str(archive), "sha256": actual_sha})
            # The retained files are plain tar despite their .tar.gz names.
            # Detect the container by bytes rather than trusting the suffix.
            with tarfile.open(archive, "r|*") as source:
                for member in source:
                    parts = Path(member.name).parts
                    if not member.isfile() or len(parts) < 4:
                        continue
                    rel = "/".join(parts[-4:])
                    if rel not in wanted:
                        continue
                    assert rel.startswith(split + "/") and rel not in found
                    assert 0 < member.size <= 1024 * 1024
                    with source.extractfile(member) as stream:
                        data = stream.read()
                    actual = hashlib.sha256(data).hexdigest()
                    assert actual == expected[rel], rel
                    dest = args.scratch / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with dest.open("xb") as stream:
                        stream.write(data)
                    receipt.write(f"{rel}\t{actual}\t{len(data)}\n")
                    found.add(rel)
                    if len(found) % 10000 == 0:
                        print(
                            f"verified original PNGs {len(found)}/{len(wanted)}",
                            flush=True,
                        )
            print(f"finished retained {split} archive: {len(found)} PNGs", flush=True)
    assert found == wanted, f"missing {len(wanted - found)} required PNGs"
    result = {
        "schema": "bapps-original-png-scratch-v1",
        "keys_sha256": sha(args.keys),
        "library_index_sha256": sha(index),
        "archives": archives,
        "png_count": len(found),
        "png_receipt_sha256": sha(args.receipt_dir / "png-sha256.tsv"),
        "scratch": str(args.scratch),
        "labels_read": False,
        "third_party_pixel_conversion": False,
    }
    (args.receipt_dir / "UNPACK_VERIFIED.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
