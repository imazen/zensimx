#!/usr/bin/env python3
"""A `map-pool-v1` with another spatial target on the same views (E36 revised plan R4: DVIFM's map as the target).

Keeps the parent pool's feature rows, views and integrands (the manifest binds the parent's files, unchanged) and
writes new targets: per view, the new block sums and their within-view rank z-scores, in the parent's view order and
offsets. With --consensus the target is the parent's rank z-score and the new one averaged per block, then ranked and
scaled again (the stored "sums" half then holds that average). The sums come from one little-endian f32 file per view,
`SUMS/<key>.<suffix>.f32`, on the parent's block grid (dvifmish `examples/block_sums.rs` writes `<key>.dvifm_sum.f32`).

    map_pool_retarget.py PARENT_DIR OUT_DIR --sums DIR --meta SUMS.jsonl [--suffix dvifm_sum] [--consensus]

SUMS.jsonl is the producer's per-view JSON lines (`key`, `blocks`, `block`, `preset`); every view must have one whose
grid matches the parent's. A view whose new sums are all equal carries no rank and is refused.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rank_z(v):
    r = rankdata(v)
    return (r - r.mean()) / max(r.std(), 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parent", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--sums", type=Path, required=True)
    ap.add_argument("--meta", type=Path, required=True)
    ap.add_argument("--suffix", default="dvifm_sum")
    ap.add_argument("--consensus", action="store_true")
    a = ap.parse_args()
    pm = json.loads((a.parent / "manifest.json").read_text())
    for part in ("features", "views", "integrands", "targets"):
        if sha(pm[part]["path"]) != pm[part]["sha256"]:
            raise SystemExit(f"parent {part} changed since its manifest")
    views = json.loads(Path(pm["views"]["path"]).read_text())
    side = json.loads(Path(pm["provenance"]["views_meta"]["path"]).read_text())
    if [v["key"] for v in views] != [s["key"] for s in side]:
        raise SystemExit("views and views_meta disagree")
    meta = {m["key"]: m for m in map(json.loads, a.meta.read_text().splitlines())}
    presets = sorted({m["preset"] for m in meta.values()})
    if len(presets) != 1:
        raise SystemExit(f"sums from more than one preset: {presets}")
    parent_t = np.fromfile(pm["targets"]["path"], "<f4")
    a.out.mkdir(parents=True)
    out_t, off = [], 0
    for v, s in zip(views, side):
        k, nb = v["key"], v["blocks"]
        m = meta.get(k)
        if m is None or m["blocks"] != [s["blocks_x"], s["blocks_y"]] or m["block"] != s["block"]:
            raise SystemExit(f"{k}: no sums on the parent's grid")
        if v["target_offset"] != off:
            raise SystemExit(f"{k}: parent target offsets are not contiguous")
        new = np.fromfile(a.sums / f"{k}.{a.suffix}.f32", "<f4").astype(np.float64)
        if new.size != nb or not np.isfinite(new).all():
            raise SystemExit(f"{k}: {new.size} sums for {nb} blocks, or non-finite")
        if np.ptp(new) == 0:
            raise SystemExit(f"{k}: every block has the same sum; no rank to fit")
        if a.consensus:
            mean = (parent_t[off + nb:off + 2 * nb].astype(np.float64) + rank_z(new)) / 2
            out_t += [mean, rank_z(mean)]
        else:
            out_t += [new, rank_z(new)]
        off += 2 * nb
    if off != parent_t.size:
        raise SystemExit("parent targets hold more values than its views")
    np.concatenate(out_t).astype("<f4").tofile(a.out / "targets.f32")
    what = ("per view: the mean of the parent's rank z-score and the new sums' rank z-score, then its rank z-score"
            if a.consensus else "per view: the new block sums, then their within-view rank z-scores")
    m = dict(schema="map-pool-v1", features=pm["features"], views=pm["views"], integrands=pm["integrands"],
             targets=dict(path=str((a.out / "targets.f32").resolve()), sha256=sha(a.out / "targets.f32")),
             feature_ids=pm["feature_ids"], view_count=pm["view_count"],
             provenance=dict(pm["provenance"], targets=what,
                             retarget_of=dict(manifest=str((a.parent / "manifest.json").resolve()),
                                              manifest_sha256=sha(a.parent / "manifest.json")),
                             new_target=dict(sums=str(a.sums.resolve()), suffix=a.suffix, preset=presets[0],
                                             meta=str(a.meta.resolve()), meta_sha256=sha(a.meta)),
                             registration="zensimx benchmarks/e36_subset_test_2026-11-01.md (revised plan R4)"))
    (a.out / "manifest.json").write_text(json.dumps(m, indent=1) + "\n")
    print(json.dumps(dict(views=len(views), target_floats=off, preset=presets[0], consensus=a.consensus)))


if __name__ == "__main__":
    main()
