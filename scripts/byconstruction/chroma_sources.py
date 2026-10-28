#!/usr/bin/env python3
"""Freeze pilot sources from Mlic TRAIN keys and canonical K500 representatives.

Pillow inspects source headers only. Source bytes are copied verbatim; imazen
tools on the worker own every derivative. Missing LFS data is fetched from the
representative's canonical URL and must match the repository's LFS SHA256.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
from urllib.parse import unquote, urlparse

import pyarrow.parquet as pq
from PIL import Image


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def group(content_class):
    n = int(content_class[:4])
    if n == 2400:
        return "texture"
    if n in [7000, 8000, 8100]:
        return "screen_plot"
    if n in [2200, 9000, 9094, 9226]:
        return "illustration_render"
    if n in [5000, 5200, 5300, 6000, 6600, 6800]:
        return "lineart_scan"
    return "photo"


def select(a):
    a.root.mkdir(parents=True, exist_ok=True)
    rawdir = a.root / "raw"
    rawdir.mkdir(exist_ok=True)
    spec = {"schema": "chroma-pilot-sources-v1", "build_commit": a.build_commit,
            "binary_sha256": {p.name: sha(p) for p in (a.root / "bin").iterdir()},
            "sources": [], "controls": [], "pins": {}, "source_exclusions": []}

    def include(path, identity, kind, cls, source_class, reference_group, url=None, expected=None):
        name = identity + ".png"
        dest = rawdir / name
        if not dest.exists():
            if url:
                print("download", identity, url, flush=True)
                with urllib.request.urlopen(url, timeout=90) as stream, dest.open("xb") as out:
                    shutil.copyfileobj(stream, out, length=1024 * 1024)
            else:
                shutil.copyfile(path, dest)
        digest = sha(dest)
        if expected and digest != expected:
            raise ValueError(f"source bytes differ from canonical LFS identity: {identity}")
        with Image.open(dest) as image:
            if image.mode not in ["RGB", "L"] or image.format != "PNG":
                raise ValueError(f"{identity}: pilot requires opaque RGB8/gray8 PNG, got {image.mode}")
            w, h = image.size
        source = {"id": identity, "type": kind, "class_group": cls, "content_class": source_class,
                  "reference_group": reference_group, "split": "train", "source_path": str(path),
                  "url": url, "source_sha256": digest, "raw": name, "width": w, "height": h}
        spec["sources"].append(source)
        return source

    # All forty candidates were fixed before scoring; their visual classes are
    # supplied explicitly, not inferred from teacher or feature values.
    candidates = json.loads(a.safe_candidates.read_text())
    assignments = {"lineart_scan": [0, 1, 9, 12, 15, 18, 25, 35, 38],
                   "illustration_render": [6, 21, 28, 39],
                   "texture": [24, 26, 27, 29, 30, 33, 36]}
    class_of = {i: c for c, ids in assignments.items() for i in ids}
    safe_keys = pq.read_table(a.bank / "safesyn/keys.parquet").to_pylist()
    safe_labels = {r["pair_key"]: r["ssim2_oracle"] for r in
                   pq.read_table(a.bank / "safesyn/labels__ssim2_oracle.parquet").to_pylist()}
    for leg in ["safesyn", "cid22"]:
        keypath = a.mlic_root / "wide/main/real" / f"{leg}_fit.keys.parquet"
        spec["pins"][str(keypath)] = sha(keypath)
    fit_safe = set(pq.read_table(a.mlic_root / "wide/main/real/safesyn_fit.keys.parquet",
                                 columns=["ref_basename"]).column(0).to_pylist())
    first_by_group = {}
    for r in safe_keys:
        first_by_group.setdefault(r["ref_group"], r)
    for i, (ref, path) in enumerate(candidates):
        if ref not in fit_safe:
            raise ValueError("SafeSyn source outside Mlic fitting partition")
        source = include(Path(path), f"safe{i:02}", "safesyn", class_of.get(i, "photo"),
                         "safesyn_visual_class", ref)
        if i in [0, 4, 6, 15]:
            r = first_by_group[ref]
            dist = Path(r["dist_path"])
            dist_name = f"control{i:02}" + dist.suffix
            shutil.copyfile(dist, rawdir / dist_name)
            spec["controls"].append({"raw": source["raw"], "dist_raw": dist_name,
                "pair_key": r["pair_key"], "expected_ssim2": safe_labels[r["pair_key"]],
                "reference_pixels_sha256": r["ref_pixels_sha256"],
                "distorted_pixels_sha256": r["dist_pixels_sha256"], "dist_sha256": sha(dist)})

    cid_fit = set(pq.read_table(a.mlic_root / "wide/main/real/cid22_fit.keys.parquet",
                                columns=["ref_basename"]).column(0).to_pylist())
    cid = {r["ref_group"]: r["ref_path"] for r in
           pq.read_table(a.bank / "cid22_train/keys.parquet", columns=["ref_group", "ref_path"]).to_pylist()
           if r["ref_group"] in cid_fit}
    ordered = sorted(cid)
    for i in range(8):
        ref = ordered[i * len(ordered) // 8]
        include(Path(cid[ref]), f"cid{i:02}", "cid22", "photo", "cid22_train", ref)

    splitpath = a.imazen / "manifests/split_map.tsv"
    representatives = a.imazen / "manifests/imazen26_representatives_K500_even_2026-06-18.tsv"
    for p in [splitpath, representatives, a.safe_candidates]:
        spec["pins"][str(p)] = sha(p)
    split = {r["id"]: r for r in csv.DictReader(splitpath.open(), delimiter="\t")}
    buckets = defaultdict(lambda: defaultdict(list))
    seen = set()
    for r in csv.DictReader(representatives.open(), delimiter="\t"):
        rel = unquote(urlparse(r["url"]).path).split("/imazen-26-png-v3/")[-1]
        identity = Path(rel).name[:4]
        if identity in seen:
            continue
        seen.add(identity)
        if split[identity]["split"] != "train":
            continue
        if int(identity) % 10 not in [0, 2, 4, 6, 8]:
            raise ValueError("canonical split disagrees with last-digit rule")
        if split[identity]["content_class"] != r["content_class"]:
            raise ValueError("representative class differs from canonical split map")
        buckets[group(r["content_class"])][r["content_class"]].append((r, rel, identity))
    for cls, classes in sorted(buckets.items()):
        # Round-robin classes so small classes are present before filling a group.
        selected = []
        for index in range(8):
            for content_class in sorted(classes):
                candidates_in_class = classes[content_class]
                if index < len(candidates_in_class):
                    selected.append(candidates_in_class[index])
        for r, rel, identity in selected[:8]:
            path = a.imazen / "png-v3" / rel
            initial = path.read_bytes() if path.stat().st_size < 1024 else b""
            expected = None
            url = None
            if initial.startswith(b"version https://git-lfs.github.com/spec/v1"):
                expected = initial.decode().split("oid sha256:")[1].split()[0]
                obj = a.imazen / ".git/lfs/objects" / expected[:2] / expected[2:4] / expected
                if obj.exists():
                    path = obj
                else:
                    url = r["url"]
            include(path, f"im{identity}", "imazen26", cls, r["content_class"],
                    f"imazen26:{identity}", url, expected)
    (a.root / "SOURCES.json").write_text(json.dumps(spec, indent=2) + "\n")
    print("Frozen", len(spec["sources"]), "sources", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--mlic-root", required=True, type=Path)
    p.add_argument("--bank", required=True, type=Path)
    p.add_argument("--imazen", required=True, type=Path)
    p.add_argument("--safe-candidates", required=True, type=Path)
    p.add_argument("--build-commit", required=True)
    select(p.parse_args())
