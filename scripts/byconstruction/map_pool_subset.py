#!/usr/bin/env python3
"""A `map-pool-v1` holding a subset of another pool's views (E36 revised plan R3: the map term on low-quality views).

Keeps the parent's views whose encoder quality setting (the third field of the view key, `<codec>-<size>-<q>-<sha8>`)
lies in [--q-min, --q-max] and whose key is not listed in --exclude-keys (one key per line; R4 drops the views whose
new target is flat), in the parent's order, and rewrites the integrand and target offsets. Feature rows, integrands
and targets are copied byte for byte; the manifest binds the new files and names the parent.

    map_pool_subset.py PARENT_DIR OUT_DIR --q-max 60 [--q-min 0] [--exclude-keys FILE]
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parent", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--q-min", type=int, default=0)
    ap.add_argument("--q-max", type=int, default=100)
    ap.add_argument("--exclude-keys", type=Path)
    ap.add_argument("--registration", default="revised plan R3")
    a = ap.parse_args()
    pm = json.loads((a.parent / "manifest.json").read_text())
    for part in ("features", "views", "integrands", "targets"):
        if sha(pm[part]["path"]) != pm[part]["sha256"]:
            raise SystemExit(f"parent {part} changed since its manifest")
    views = json.loads(Path(pm["views"]["path"]).read_text())
    side = json.loads(Path(pm["provenance"]["views_meta"]["path"]).read_text())
    if [v["key"] for v in views] != [s["key"] for s in side]:
        raise SystemExit("views and views_meta disagree")
    drop = set(a.exclude_keys.read_text().split()) if a.exclude_keys else set()
    if drop - {v["key"] for v in views}:
        raise SystemExit(f"--exclude-keys names views the parent lacks: {sorted(drop - {v['key'] for v in views})[:5]}")
    keep = [i for i, v in enumerate(views)
            if a.q_min <= int(v["key"].split("-")[2]) <= a.q_max and v["key"] not in drop]
    a.out.mkdir(parents=True)
    integ = np.fromfile(pm["integrands"]["path"], "<f4")
    targ = np.fromfile(pm["targets"]["path"], "<f4")
    new_views, oi, ot = [], 0, 0
    with open(a.out / "integrands.f32", "wb") as fi, open(a.out / "targets.f32", "wb") as ft:
        for i in keep:
            v = views[i]
            n_i = len(v["positions"]) * v["blocks"]
            fi.write(integ[v["integrand_offset"]:v["integrand_offset"] + n_i].tobytes())
            ft.write(targ[v["target_offset"]:v["target_offset"] + 2 * v["blocks"]].tobytes())
            new_views.append(dict(v, integrand_offset=oi, target_offset=ot))
            oi += n_i
            ot += 2 * v["blocks"]
    t = pq.read_table(pm["features"]["path"])
    if t.num_rows != len(views) or t["row_idx"].to_pylist() != list(range(len(views))):
        raise SystemExit("parent feature rows are not one per view in order")
    sub = t.take(keep)
    sub = sub.set_column(0, "row_idx", __import__("pyarrow").array(np.arange(len(keep)), type=sub.schema.field(0).type))
    pq.write_table(sub, a.out / "features.parquet", compression="zstd")
    (a.out / "views.json").write_text(json.dumps(new_views) + "\n")
    (a.out / "views_meta.json").write_text(json.dumps([side[i] for i in keep]) + "\n")
    bound = lambda n: dict(path=str((a.out / n).resolve()), sha256=sha(a.out / n))  # noqa: E731
    m = dict(schema="map-pool-v1", features=bound("features.parquet"), views=bound("views.json"),
             integrands=bound("integrands.f32"), targets=bound("targets.f32"), feature_ids=pm["feature_ids"],
             view_count=len(keep),
             provenance=dict(pm["provenance"], views_meta=bound("views_meta.json"),
                             subset_of=dict(manifest=str((a.parent / "manifest.json").resolve()),
                                            manifest_sha256=sha(a.parent / "manifest.json"), views=len(views)),
                             subset_rule=f"encoder quality setting in [{a.q_min}, {a.q_max}] (view key field 3)"
                             + (f"; {len(drop)} views excluded by key ({a.exclude_keys.resolve()})" if drop else ""),
                             registration=f"zensimx benchmarks/e36_subset_test_2026-11-01.md ({a.registration})"))
    (a.out / "manifest.json").write_text(json.dumps(m, indent=1) + "\n")
    print(json.dumps(dict(views=len(keep), of=len(views), integrand_floats=oi, target_floats=ot)))


if __name__ == "__main__":
    main()
