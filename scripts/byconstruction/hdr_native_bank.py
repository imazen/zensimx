#!/usr/bin/env python3
"""Assemble a native revision-7 HDR feature bank from zenfleet shard ledgers.

Inputs are a plan directory (views.spec.json, membership-feature-map.json, _MANIFEST.json)
and one or more shard directories (ledger.parquet + blobs/). Every ledger row must be
`done`, every CAS object must hash to its name, every referenced sidecar must exist, and
every planned view must be covered exactly once (first-wins on repeated identical work;
a conflicting repeat refuses). Output: keys.parquet (membership metadata, no quality
target), features.parquet (row_id + f0..f{W-1} float64, absent slots NaN) and a
_MANIFEST.json carrying build_commit and the input hashes. No label is read.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

WIDTH = 1825
IDS = list(range(228)) + list(range(372, 720))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assemble(plan, shards, out, build_commit, foreign_ok=False):
    plan, out = Path(plan), Path(out)
    spec = json.loads((plan / "views.spec.json").read_text())
    pm = json.loads((plan / "_MANIFEST.json").read_text())
    if pm["spec_sha256"] != sha(plan / "views.spec.json") or pm["mapping_sha256"] != sha(plan / "membership-feature-map.json"):
        raise ValueError("plan files differ from their manifest")
    mapping = json.loads((plan / "membership-feature-map.json").read_text())
    by_knobs = {v["cell"]["knob_tuple_json"]: i for i, v in enumerate(spec["views"])}
    blobs, results = {}, {}
    for shard in map(Path, shards):
        for p in (shard / "blobs").rglob("*"):
            if p.is_file() and ".staging" not in p.parts:
                if sha(p) != p.name:
                    raise ValueError(f"CAS object does not hash to its name: {p}")
                blobs[p.name] = p
        for row in pq.read_table(shard / "ledger.parquet").to_pylist():
            if row["status"] != "done":
                raise ValueError(f"non-done ledger row in {shard}: {row['status']}")
            vi = by_knobs.get(row["knob_tuple_json"])
            if vi is None:
                if foreign_ok:
                    continue
                raise ValueError("ledger row not in plan")
            if vi in results and results[vi] != row["output_sha"]:
                raise ValueError(f"conflicting outputs for view {vi}")
            results[vi] = row["output_sha"]
    if sorted(results) != list(range(len(spec["views"]))):
        raise ValueError("planned views not covered exactly")
    tables, producer = {}, None
    for vi, out_sha in results.items():
        res = json.loads(blobs[out_sha].read_bytes())
        controls = json.loads(spec["views"][vi]["cell"]["knob_tuple_json"])
        if res["controls"] != controls or res["labels_read"] is not False:
            raise ValueError("result does not bind the planned controls")
        feat = blobs[res["sidecars"]["features.tsv"]]
        man = json.loads(blobs[res["sidecars"]["features.manifest.json"]].read_bytes())
        if man["research"]["zensim_build_commit"] != build_commit or man["formula_revision"] != "Rev7":
            raise ValueError("native producer identity differs")
        if producer is None:
            producer = man["research"]
        elif producer != man["research"]:
            raise ValueError("research provenance changed between views")
        t = pacsv.read_csv(feat, parse_options=pacsv.ParseOptions(delimiter="\t"),
                           convert_options=pacsv.ConvertOptions(column_types={"q": pa.string(), "dist_basename": pa.string(),
                                                                              **{f"f{i}": pa.float64() for i in range(WIDTH)}}))
        if t.num_rows != len(controls["candidates"]):
            raise ValueError("feature row count differs from candidates")
        for i in IDS:
            if not all(math.isfinite(x) for x in t[f"f{i}"].to_pylist()):
                raise ValueError("nonfinite measured feature")
        tables[vi] = t
    mapping.sort(key=lambda m: (m.get("source_index", 0), m["view_index"], m["feature_row"]))
    rows = []
    for rid, m in enumerate(mapping):
        t = tables[m["view_index"]]
        rows.append((rid, t.slice(m["feature_row"], 1)))
    feats = pa.concat_tables([t.select([f"f{i}" for i in range(WIDTH)]) for _, t in rows])
    feats = feats.add_column(0, "row_id", pa.array([r for r, _ in rows], pa.int64()))
    keys = pa.Table.from_pylist([dict(row_id=rid, **{k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in m.items() if k != "metadata"},
                                      **({"metadata_json": json.dumps(m["metadata"], sort_keys=True)} if "metadata" in m else {}))
                                 for rid, m in enumerate(mapping)])
    out.mkdir(parents=True, exist_ok=False)
    pq.write_table(keys, out / "keys.parquet", compression="zstd")
    pq.write_table(feats, out / "features.parquet", compression="zstd")
    manifest = dict(schema="hdr-native-rev7-bank-v1", build_commit=build_commit, plan_manifest=pm, plan_sha256=sha(plan / "_MANIFEST.json"),
                    rows=len(mapping), views=len(spec["views"]), feature_width=WIDTH, measured_ids=IDS, absent_slots="NaN",
                    keys_sha256=sha(out / "keys.parquet"), features_parquet_sha256=sha(out / "features.parquet"),
                    research=producer, cas_objects_verified=len(blobs), labels_read=False,
                    assembler_sha256=sha(__file__))
    (out / "_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--build-commit", required=True)
    ap.add_argument("--foreign-ok", action="store_true", help="shards also hold rows of another plan; skip them (every row of this plan must still be done)")
    a = ap.parse_args()
    m = assemble(a.plan, a.shards, a.out, a.build_commit, a.foreign_ok)
    print(json.dumps({k: m[k] for k in ("rows", "views", "keys_sha256", "features_parquet_sha256", "cas_objects_verified")}))
