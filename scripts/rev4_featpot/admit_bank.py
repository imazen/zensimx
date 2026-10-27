"""Join the promoted Rev3 bank for POTENTIAL diagnostics, keyed by pair_key.

POTENTIAL — ceiling, not a model score. This is an input adapter only: it does
not fit, score or calculate statistics. It may read labels only for the nine
D1/D2 sets explicitly pinned below, and only from their bank labels files.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from bank_pins import PINNED_MANIFEST_SHA


PROMOTED_BANK = Path("/var/tmp/rev4-featbank/bank")
SNAPSHOT_BANK = Path("/var/tmp/rev4-featpot/bank_snapshot")
SNAPSHOT_RECEIPT_SHA = "1c61f154e3f8f263e2c7e2f3ac125f98a9d94653cf616518087bd00048e899fc"
OUT = Path("/var/tmp/rev4-featpot/admitted")
FEATURE_SET_ID = "basic+peaks+masked+iw+v2+append+append2@w944/ceiling_rev3#b782e349"

EXPECTED_TARGET = {
    "kadid_train": "human_score",
    "tid2013": "human_score",
    "konfig_train": "human_score",
    "konjnd_bpg_train": "ssim2_oracle",
    "konjnd_bpg_val": "ssim2_oracle",
    "cid22_a25": "human_score",
    "aic3": "human_score",
    "kadid_select": "human_score",
    "konfig_val": "human_score",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# Part B may extend the promoted bank manifests during this long baseline grid.
# Preserve the original byte-pinned D1/D2 input if that happens. The snapshot
# contains only the nine admitted sets and was independently file-hash audited.
def promoted_manifests_pinned() -> bool:
    for name, want in PINNED_MANIFEST_SHA.items():
        path = PROMOTED_BANK / name / "_MANIFEST.json"
        if not path.is_file() or sha256(path) != want:
            return False
    return True


if promoted_manifests_pinned():
    BANK = PROMOTED_BANK
elif ((SNAPSHOT_BANK / "SNAPSHOT_RECEIPT.json").is_file()
      and sha256(SNAPSHOT_BANK / "SNAPSHOT_RECEIPT.json") == SNAPSHOT_RECEIPT_SHA):
    BANK = SNAPSHOT_BANK
else:
    raise ValueError("promoted bank changed and admitted bank snapshot is not pinned")


def load_pinned_set(name: str) -> dict:
    directory = BANK / name
    manifest_path = directory / "_MANIFEST.json"
    actual = sha256(manifest_path)
    expected = PINNED_MANIFEST_SHA[name]
    if actual != expected:
        raise ValueError(f"{name}: manifest SHA mismatch {actual} != {expected}")
    manifest = json.loads(manifest_path.read_text())
    if manifest["set"] != name or manifest["feature_set_id"] != FEATURE_SET_ID:
        raise ValueError(f"{name}: feature-set identity mismatch")
    if manifest["formula_revision"] != "3" or manifest["dtype"] != "f32":
        raise ValueError(f"{name}: incompatible arithmetic or dtype")
    files = manifest["files"]
    label_files = [filename for filename in files if filename.startswith("labels__")]
    feature_files = [filename for filename in files if filename.startswith("features__")]
    if len(label_files) != 1 or len(feature_files) != 1:
        raise ValueError(f"{name}: expected exactly one label and feature file")
    for filename in ("keys.parquet", feature_files[0], label_files[0]):
        got = sha256(directory / filename)
        want = files[filename]["sha256"]
        if got != want:
            raise ValueError(f"{name}/{filename}: SHA mismatch {got} != {want}")
    return manifest


def admit(name: str) -> dict:
    manifest = load_pinned_set(name)
    directory = BANK / name
    files = manifest["files"]
    feature_file = next(filename for filename in files if filename.startswith("features__"))
    label_file = next(filename for filename in files if filename.startswith("labels__"))
    target = EXPECTED_TARGET[name]

    # Only these three exact bank files are opened. In particular, no pairs/
    # or raw/ copy of any held-out target can enter this adapter.
    keys = pq.read_table(
        directory / "keys.parquet",
        columns=["pair_key", "ref_group", "codec", "knob", "pixels_identical"],
    )
    features = pq.read_table(directory / feature_file)
    labels = pq.read_table(directory / label_file)
    if target not in labels.column_names:
        raise ValueError(f"{name}: missing expected target {target}")
    if keys.num_rows != manifest["unique_pair_keys"] or features.num_rows != keys.num_rows:
        raise ValueError(f"{name}: key/feature row count mismatch")
    if labels.num_rows != manifest["row_count"]:
        raise ValueError(f"{name}: label/stimulus row count mismatch")
    ids = manifest["populated_feature_ids"]
    zeros = manifest["structural_zero_feature_ids"]
    if sorted(ids + zeros) != list(range(944)) or len(ids) != 905 or len(zeros) != 39:
        raise ValueError(f"{name}: feature-ID partition mismatch")
    if features.column_names != ["pair_key"] + [f"f{i}" for i in ids]:
        raise ValueError(f"{name}: sidecar schema mismatch")

    k = keys.to_pandas()
    f = features.to_pandas()
    y = labels.select(["pair_key", "source_row_id", target]).to_pandas()
    if k.pair_key.duplicated().any() or f.pair_key.duplicated().any():
        raise ValueError(f"{name}: duplicate key/feature pair_key")
    if set(k.pair_key) != set(f.pair_key) or set(k.pair_key) != set(y.pair_key):
        raise ValueError(f"{name}: pair_key set mismatch")
    joined = y.merge(k, on="pair_key", how="left", sort=False, validate="many_to_one")  # joinsafety-ok: pair_key-keyed feature/label join with validate= and full-coverage asserts, not a metric-table join
    joined = joined.merge(f, on="pair_key", how="left", sort=False, validate="many_to_one")  # joinsafety-ok: pair_key-keyed feature/label join with validate= and full-coverage asserts, not a metric-table join
    if len(joined) != manifest["row_count"] or joined.ref_group.isna().any():
        raise ValueError(f"{name}: join coverage mismatch")
    if joined[target].isna().any() or not np.isfinite(joined[target]).all():
        raise ValueError(f"{name}: missing/nonfinite target")
    for feature_id in zeros:
        joined[f"f{feature_id}"] = np.float32(0.0)
    if not np.isfinite(joined[[f"f{i}" for i in ids]].to_numpy()).all():
        raise ValueError(f"{name}: nonfinite features")
    joined = joined.rename(columns={"ref_group": "ref_basename"})
    order = [
        "pair_key",
        "source_row_id",
        "ref_basename",
        "codec",
        "knob",
        "pixels_identical",
    ] + [
        f"f{i}" for i in range(944)
    ]
    joined = joined[order]
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"POT_{name}_rev3_944.parquet"
    pq.write_table(pa.Table.from_pandas(joined, preserve_index=False), dest, compression="zstd")
    result = {
        "set": name,
        "label": "POTENTIAL — ceiling, not a model score",
        "rows": len(joined),
        "unique_pair_keys": joined.pair_key.nunique(),
        "references": joined.ref_basename.nunique(),
        "target": target,
        "target_scale": manifest["label_scale"],
        "manifest_sha256": PINNED_MANIFEST_SHA[name],
        "output": str(dest),
        "output_sha256": sha256(dest),
    }
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sets", nargs="+", choices=sorted(PINNED_MANIFEST_SHA))
    args = parser.parse_args()
    for name in args.sets:
        admit(name)


if __name__ == "__main__":
    main()
