#!/usr/bin/env python3
"""Local chroma pilot transport and registered ladder admission.

Source selection and frozen teacher controls are supplied in SOURCES.json.
Run on one idle worker inside run-heavy; this is not a fleet launcher.
Existing imazen preparation, sweep and frozen teacher executables own pixels
and scores. The old teacher's incidental features are never training inputs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def rows(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def admission(scores):
    """Stop on adjacent nonfall; retain >=1-point steps from last kept rung."""
    if not scores or not all(math.isfinite(s) for s in scores):
        raise ValueError("empty/nonfinite teacher ladder")
    kept = [0]
    stop = None
    for i in range(1, len(scores)):
        if scores[i] >= scores[i - 1]:
            stop = i
            break
        if scores[kept[-1]] - scores[i] >= 1.0:
            kept.append(i)
    return {"kept_indices": kept if len(kept) >= 3 else [],
            "candidate_indices": kept, "stop_index": stop,
            "drop_reason": "fewer than three kept rungs" if len(kept) < 3 else None}


def grids():
    chroma = [0, .5, .70710678, 1, 1.41421356, 2, 2.82842712, 4,
              5.65685425, 8, 11.3137085, 16, 22.627417, 32, 45.254834, 64]
    common = {"aq_enabled": [False], "deringing": [False], "trellis": [False]}
    result = {}
    for family, sub in [("ycc444", "444"), ("ycc420", "420")]:
        result[family] = {**common, "subsampling": [sub], "plane_tables": [
            {"m": [luma, c, c]} for luma in [0, .05] for c in chroma]}
    result["xyb"] = {**common, "xyb": [True], "xyb_subsampling": ["bquarter"],
                     "chroma_distance_scales": [[s, s] for s in
                         [.1, .14142136, .2, .28284271, .4, .56568542,
                          .8, 1, 1.13137085, 1.6, 2.2627417, 3.2, 4, 5]]}
    return result


def command(root, label, argv):
    print(label, " ".join(map(str, argv)), flush=True)
    start = time.monotonic()
    with (root / "logs" / f"{label}.log").open("x") as log:
        with subprocess.Popen(list(map(str, argv)), stdout=log, stderr=subprocess.STDOUT) as proc:
            first = False
            while proc.poll() is None:
                if label in grids() and not first:
                    p = next((root / "enc").glob("*.jpg"), None)
                    if p is not None:
                        save(root / f"{label}.first-artifact.json", {
                            "path": str(p), "bytes": p.stat().st_size, "sha256": sha(p)})
                        first = True
                try:
                    proc.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
            if proc.returncode:
                raise subprocess.CalledProcessError(proc.returncode, argv)
    elapsed = time.monotonic() - start
    print(label, "completed", elapsed, "seconds", flush=True)
    return elapsed


def teacher(root, label, pairs):
    return command(root, label, [root / "bin/teacher", "--corpus", "pairs-tsv", "--path", pairs,
        "--out", root / f"{label}.csv", "--full-944", "--audit-jsonl", root / f"{label}.jsonl",
        "--audit-ssim2"])


def cas(root, family):
    dest = root / "artifacts/zenjpeg"
    dest.mkdir(parents=True, exist_ok=True)
    result = []
    for r in rows(root / f"{family}.tsv"):
        p = root / "enc" / r["encoded_filename"]
        digest = sha(p)
        blob = dest / f"{digest}.jpg"
        if blob.exists():
            if sha(blob) != digest:
                raise ValueError("CAS collision")
        else:
            os.link(p, blob)
        r["encode_sha256"] = digest
        r["artifact"] = str(blob)
        result.append(r)
    save(root / f"{family}.cas.json", result)
    return len(result)


def run(root):
    spec = json.loads((root / "SOURCES.json").read_text())
    for name, digest in spec["binary_sha256"].items():
        if sha(root / "bin" / name) != digest:
            raise ValueError(f"binary identity: {name}")
    for d in ["src", "logs", "enc", "dist", "scratch"]:
        (root / d).mkdir(exist_ok=True)
    os.environ.update(TMPDIR=str(root / "scratch"), ZENSIM_FORMULA_REV="3", ZENSIM_ROOT_FORM="sqrt")
    progress = {"stage": "teacher-replay", "pid": os.getpid(), "root": str(root),
                "build_commit": spec["build_commit"], "required_sources": len(spec["sources"])}
    save(root / "PROGRESS.json", progress)
    controls = root / "controls.tsv"
    with controls.open("x") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["ref_path", "dist_path"])
        for c in spec["controls"]:
            writer.writerow([root / "raw" / c["raw"], root / "raw" / c["dist_raw"]])
    teacher(root, "control", controls)
    audit = [json.loads(s) for s in (root / "control.jsonl").read_text().splitlines()]
    by_pair = {(r["reference"], r["distorted"]): r["peer_ssim2"]["score"] for r in audit}
    actual = [by_pair[(str(root / "raw" / c["raw"]), str(root / "raw" / c["dist_raw"]))]
              for c in spec["controls"]]
    expected = [c["expected_ssim2"] for c in spec["controls"]]
    if actual != expected:
        save(root / "CONTROL.json", {"pass": False, "actual": actual, "expected": expected})
        raise ValueError("frozen SafeSyn teacher replay differs")
    save(root / "CONTROL.json", {"pass": True, "actual": actual, "expected": expected})
    progress["stage"] = "prepare"
    save(root / "PROGRESS.json", progress)
    prepared = []
    for source in spec["sources"]:
        raw = root / "raw" / source["raw"]
        if sha(raw) != source["source_sha256"]:
            raise ValueError("source identity")
        for size in [256, 512]:
            if max(source["width"], source["height"]) < size:
                continue  # Predeclared no-upscale size rule, recorded in PREPARED.json.
            dest = root / "src" / f'{source["id"]}_m{size}.png'
            if max(source["width"], source["height"]) == size:
                shutil.copyfile(raw, dest)
            else:
                command(root, dest.stem, [root / "bin/prepare", "resize", "--in", raw,
                                         "--out", dest, "--max", str(size)])
            prepared.append({**source, "max_side": size, "path": str(dest), "sha256": sha(dest)})
    save(root / "PREPARED.json", prepared)
    progress["prepared_views"] = len(prepared)
    elapsed = {}
    for family, grid in grids().items():
        save(root / f"{family}.grid.json", grid)
        progress["stage"] = family
        save(root / "PROGRESS.json", progress)
        elapsed[family] = command(root, family, [root / "bin/zenmetrics", "sweep", "--codec", "zenjpeg",
            "--sources", root / "src", "--q-grid", "95", "--knob-grid", json.dumps(grid),
            "--metric", "ssim2", "--use-legacy-scheduler", "--output", root / f"{family}.tsv",
            "--encoded-out-dir", root / "enc", "--distorted-out-dir", root / "dist",
            "--pairs-tsv", root / f"{family}.pairs.tsv", "--jobs", "8"])
        n = cas(root, family)
        wanted = len(prepared) * (32 if family.startswith("ycc") else 14)
        if n != wanted:
            raise ValueError(f"{family}: {n} of {wanted} encoded cells")
        progress[f"{family}_cells"] = n
        save(root / "PROGRESS.json", progress)
    pairfile = root / "teacher-pairs.tsv"
    with pairfile.open("x") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["ref_path", "dist_path"])
        for family in grids():
            for r in rows(root / f"{family}.pairs.tsv"):
                writer.writerow([r["ref_path"], r["dist_path"]])
    progress["stage"] = "frozen-teacher"
    save(root / "PROGRESS.json", progress)
    elapsed["frozen_teacher"] = teacher(root, "teacher", pairfile)
    save(root / "TIMING.json", elapsed)
    progress["stage"] = "encoded-and-scored"
    save(root / "PROGRESS.json", progress)


def analyze(root):
    prepared = {r["path"]: r for r in json.loads((root / "PREPARED.json").read_text())}
    # Audit ordering is parallel; bind the actual path pair, never its row position.
    audit = [json.loads(s) for s in (root / "teacher.jsonl").read_text().splitlines()]
    scores = {(r["reference"], r["distorted"]): r["peer_ssim2"]["score"] for r in audit}
    ladders = {}
    for family in grids():
        pairs = {tuple(r[k] for k in ["image_path", "q", "knob_tuple_json"]): r
                 for r in rows(root / f"{family}.pairs.tsv")}
        for r in json.loads((root / f"{family}.cas.json").read_text()):
            pair = pairs[tuple(r[k] for k in ["image_path", "q", "knob_tuple_json"])]
            r["teacher_ssim2"] = scores[(pair["ref_path"], pair["dist_path"])]
            knobs = json.loads(r["knob_tuple_json"])
            luma, rung = (knobs["plane_tables"]["m"][:2] if family != "xyb"
                          else ("q95", knobs["chroma_distance_scales"][0]))
            source = prepared[r["image_path"]]
            key = (source["id"], source["max_side"], family, str(luma))
            ladders.setdefault(key, []).append({**r, "rung": rung, "source": source})
    result = []
    for key, cells in sorted(ladders.items()):
        cells.sort(key=lambda c: c["rung"])
        result.append({"key": key, "admission": admission([c["teacher_ssim2"] for c in cells]), "cells": cells})
    save(root / "LADDERS.json", result)
    print("ladders", len(result), "admitted", sum(bool(r["admission"]["kept_indices"]) for r in result))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["run", "analyze"])
    p.add_argument("root", type=Path)
    a = p.parse_args()
    {"run": run, "analyze": analyze}[a.action](a.root.resolve())


if __name__ == "__main__":
    main()
