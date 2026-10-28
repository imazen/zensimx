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
import statistics
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
    existing = set((root / "enc").glob("*.jpg")) if label in grids() else set()
    with (root / "logs" / f"{label}.log").open("x") as log:
        with subprocess.Popen(list(map(str, argv)), stdout=log, stderr=subprocess.STDOUT) as proc:
            first = False
            while proc.poll() is None:
                if label in grids() and not first:
                    p = next((p for p in (root / "enc").glob("*.jpg") if p not in existing
                              and (root / "dist" / f"{p.stem}.png").is_file()), None)
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
        writer.writerow(["ref_path", "dist_path", "human_score"])
        for c in spec["controls"]:
            writer.writerow([root / "raw" / c["raw"], root / "raw" / c["dist_raw"], 0])
    teacher(root, "control", controls)
    audit = [json.loads(s) for s in (root / "control.jsonl").read_text().splitlines()]
    by_pair = {(r["reference"], r["distorted"]): r["peer_ssim2"]["score"] for r in audit}
    actual = [by_pair[(str(root / "raw" / c["raw"]), str(root / "raw" / c["dist_raw"]))]
              for c in spec["controls"]]
    expected = [c["expected_ssim2"] for c in spec["controls"]]
    by_pair_record = {(r["reference"], r["distorted"]): r for r in audit}
    for c in spec["controls"]:
        r = by_pair_record[(str(root / "raw" / c["raw"]), str(root / "raw" / c["dist_raw"]))]
        for key in ["reference_pixels_sha256", "distorted_pixels_sha256"]:
            if r[key] != c[key]:
                raise ValueError(f"frozen control pixel identity differs: {key}")
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
        writer.writerow(["ref_path", "dist_path", "human_score"])
        for family in grids():
            for r in rows(root / f"{family}.pairs.tsv"):
                writer.writerow([r["ref_path"], r["dist_path"], 0])
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
    by_pair = {(r["reference"], r["distorted"]): r for r in audit}
    if len(by_pair) != len(audit):
        raise ValueError("duplicate frozen teacher pair")
    ladders = {}
    for family in grids():
        pairs = {tuple(r[k] for k in ["image_path", "q", "knob_tuple_json"]): r
                 for r in rows(root / f"{family}.pairs.tsv")}
        for r in json.loads((root / f"{family}.cas.json").read_text()):
            pair = pairs[tuple(r[k] for k in ["image_path", "q", "knob_tuple_json"])]
            record = by_pair[(pair["ref_path"], pair["dist_path"])]
            r["teacher_ssim2"] = record["peer_ssim2"]["score"]
            r["width"], r["height"] = record["width"], record["height"]
            if r["width"] * r["height"] >= 1_000_000:
                raise ValueError("pilot raster exceeds owner's sub-megapixel limit")
            r["bpp"] = 8 * int(r["encoded_bytes"]) / (r["width"] * r["height"])
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


def inspect(root):
    """Independent read-only JPEG/PNG oracle; never writes training pixels."""
    from PIL import Image

    ladders = json.loads((root / "LADDERS.json").read_text())
    audits = [json.loads(s) for s in (root / "teacher.jsonl").read_text().splitlines()]
    by_pair = {(r["reference"], r["distorted"]): r for r in audits}
    tested = 0
    unique = set()
    for ladder in ladders:
        family = ladder["key"][2]
        pairs = {r["knob_tuple_json"]: r for r in rows(root / f"{family}.pairs.tsv")
                 if r["image_path"] == ladder["cells"][0]["image_path"]}
        fixed_y = None
        for cell in ladder["cells"]:
            encoded = root / "enc" / cell["encoded_filename"]
            digest = sha(encoded)
            if digest != cell["encode_sha256"] or sha(root / "artifacts/zenjpeg" / f"{digest}.jpg") != digest:
                raise ValueError("persisted encode identity differs")
            unique.add(digest)
            with Image.open(encoded) as image:
                image.load()
                if image.size != (cell["width"], cell["height"]):
                    raise ValueError("independent JPEG dimensions differ")
                # zenjpeg XYB writes X/Y/B in component order; Y is the middle
                # component. Ordinary YCbCr writes Y first.
                component = 1 if family == "xyb" else 0
                table = image.quantization[image.layer[component][3]]
                if family != "xyb" and ladder["key"][3] == "0" and table != [1] * 64:
                    raise ValueError("all-ones Y DQT differs")
                if fixed_y is None:
                    fixed_y = table
                elif table != fixed_y:
                    raise ValueError("luma quantization varies within chroma ladder")
            pair = pairs[cell["knob_tuple_json"]]
            record = by_pair[(pair["ref_path"], pair["dist_path"])]
            with Image.open(pair["dist_path"]) as image:
                pixels = image.convert("RGB").tobytes()
            if hashlib.sha256(pixels).hexdigest() != record["distorted_pixels_sha256"]:
                raise ValueError("independent PNG decode differs from frozen teacher pixels")
            tested += 1
    receipt = {"pass": True, "cells": tested, "unique_encoded_sha256": len(unique),
               "oracle": f"Pillow {Image.__version__}",
               "checks": ["all JPEGs independently decoded", "all encode and CAS hashes match",
                          "all decoded PNG RGB hashes equal teacher input", "fixed Y DQT within every ladder",
                          "all-ones Y DQT exact", "JPEG dimensions equal frozen audit"]}
    save(root / "ORACLE.json", receipt)
    print(json.dumps(receipt))


def report(root, destination):
    ladders = json.loads((root / "LADDERS.json").read_text())
    spec = json.loads((root / "SOURCES.json").read_text())
    timing = json.loads((root / "TIMING.json").read_text())
    if (json.loads((root / "PROGRESS.json").read_text())["stage"] != "encoded-and-scored"
            or not json.loads((root / "CONTROL.json").read_text())["pass"]
            or not json.loads((root / "ORACLE.json").read_text())["pass"]):
        raise ValueError("report requires completed generation, exact replay and independent oracle")
    if len(spec["sources"]) != 81 or len(ladders) != 810:
        raise ValueError("this Stage A report is bound to the registered run04 population")
    families = [("ycc444", "0"), ("ycc444", "0.05"), ("ycc420", "0"),
                ("ycc420", "0.05"), ("xyb", "q95")]
    summary = []
    lines = ["# Stage A chroma pilot", "",
        "MISSING: broader texture coverage (one unique imazen-26 texture source in this pilot), "
        "SafeSyn screen/plot examples, validation-bucket pilot diagnostics, and crop-specific imazen-26 pilot "
        "measurements. Stages B–D have not started. This pilot tests TRAIN-only Mitchell downscales and existing "
        "SafeSyn tiles; it does not qualify a new model or a training leg.", "",
        "Stage A generation, exact teacher replay, frozen-build scoring and registered ladder admission are complete. "
        "Stop here for coordinator review before Stage B.", "",
        "## Run and controls", "",
        f"Worker: i270 only. Run: `{root}` on i270 and dev. Generation source commit: `{spec['build_commit']}`. "
        "The coordinator authorized three family sweeps sequentially inside one run-heavy job. "
        "No other run's workers were touched.", "",
        "All four stored SafeSyn teacher scores and all eight decoded pixel hashes replay exactly. "
        "The exact preserved teacher executable scores the persisted decoded PNG pairs; the sweep's own "
        "SSIMULACRA2 column is diagnostic only. Incidental Rev3 feature CSVs from that executable are "
        "not Stage C features and must never enter the new training leg.", "",
        "| Executable | SHA256 |", "|---|---|"]
    lines += [f"| {name} | `{digest}` |" for name, digest in spec["binary_sha256"].items()]
    lines += ["", "The sweep executable is from `ghcr.io/imazen/zenfleet-worker:exec-cvvdp-0a61830d`; "
        "its bytes, not an inferred current checkout, identify the encoder build. "
        "Sources are byte-pinned in SOURCES.json, with canonical imazen-26 LFS hashes checked against the tower "
        "mirror. All sources inherit TRAIN roles. CID22 sources are oracle TRAIN references, never the 49 human references.", "",
        "## Source coverage", "", "| Type | Group | Unique sources |", "|---|---|---:|"]
    counts = {}
    for source in spec["sources"]:
        key = (source["type"], source["class_group"])
        counts[key] = counts.get(key, 0) + 1
    lines += [f"| {kind} | {group} | {count} |" for (kind, group), count in sorted(counts.items())]
    lines += ["", "81 sources, two views each (maximum long edge 256/512), 162 views total. "
        "The largest actual raster is 512×512 = 262,144 pixels; every generated reference and encode is below 1 MP. "
        "All 21 imazen-26 content classes are represented. Four texture representative rows in K500_even resolve "
        "to only two unique sources; im2402 (8736×11648) exceeded the frozen preparation decoder's input pixel "
        "limit and was excluded before scoring. Its failed run and original plan are preserved. "
        "No input guard or established test expectation was relaxed.", "",
        "Early attempts are preserved: root-level attempt stopped on the missing audit target placeholder; "
        "pilot-run02 passed the teacher replay then stopped on the copied preparation executable's mode; "
        "pilot-run03 passed replay and stopped at im2402. pilot-run04 is the complete scientific pilot.", "",
        "## Admission and luma choice", "",
        "Rungs are ordered by increasing chroma multiplier. Keep the first rung, then retain only scores at least "
        "1.0 below the last kept score. Stop at the first adjacent tie or increase, even when that score is below "
        "an earlier kept score. Drop the whole ladder if fewer than three rungs remain. No late desaturation "
        "recovery or rung after a stop is admitted. Counts below are source×size ladders; the two YCbCr luma "
        "settings are separate pilot arms.", "",
        "| Family | Fixed luma | Ladders kept/total | Kept rungs | Median base bpp | Median all-rung bpp |",
        "|---|---|---:|---:|---:|---:|"]
    for family, luma in families:
        subset = [r for r in ladders if r["key"][2:] == [family, luma]]
        n = sum(bool(r["admission"]["kept_indices"]) for r in subset)
        nr = sum(len(r["admission"]["kept_indices"]) for r in subset)
        bpp = statistics.median(c["bpp"] for r in subset for c in r["cells"])
        base_bpp = statistics.median(r["cells"][0]["bpp"] for r in subset)
        summary.append({"family": family, "luma": luma, "total": len(subset), "kept": n,
                        "kept_rungs": nr, "median_base_bpp": base_bpp, "median_bpp": bpp})
        lines.append(f"| {family} | {luma} | {n}/{len(subset)} | {nr} | {base_bpp:.6f} | {bpp:.6f} |")
    lines += ["", "Recommend all-ones Y (multiplier 0) for the full YCbCr grid. The fine multiplier 0.05 "
        "saves bytes but adds a luma-quality confound without recovering a single additional ladder in this pilot. "
        "AQ, deringing and trellis were explicitly disabled in all families; chroma was the only changing knob "
        "within each ladder. Fixed quantization tables do not imply identical RGB-derived luma after chroma "
        "reconstruction/gamut clipping.", "", "## Rung scores by family", "",
        "These are per-rung medians over all 162 views, including dropped ladders. They are descriptive "
        "population summaries, not an admitted synthetic ladder. Exact per-cell doubles are in LADDERS.json.", ""]
    for family, luma in families:
        subset = [r for r in ladders if r["key"][2:] == [family, luma]]
        lines += [f"### {family}, luma {luma}", "", "| Chroma rung | Median teacher score |", "|---:|---:|"]
        for i, cell in enumerate(subset[0]["cells"]):
            median = statistics.median(r["cells"][i]["teacher_ssim2"] for r in subset)
            lines.append(f"| {cell['rung']:.8g} | {median:.6f} |")
        lines.append("")
    xyb = [r for r in ladders if r["key"][2] == "xyb"]
    drops = [r["cells"][0]["teacher_ssim2"] - r["cells"][-1]["teacher_ssim2"] for r in xyb]
    lines += ["## XYB range", "",
        f"At fixed generic quality 95 with B-quarter subsampling, 0.1→5.0 lowers the teacher score on all "
        f"{len(drops)} views. Endpoint drops: minimum {min(drops):.6f}, median {statistics.median(drops):.6f}, "
        f"maximum {max(drops):.6f} points; {sum(d >= 1 for d in drops)}/{len(drops)} drop at least 1 point and "
        f"{sum(d >= 5 for d in drops)}/{len(drops)} drop at least 5. The allowed range reaches clear damage; "
        "no range extension is warranted by this pilot. Early ties/small reversals still cause many XYB ladders "
        "to fail the registered rule; endpoint separation never overrides admission.", "",
        "## Coverage after admission", "",
        "The recommended all-ones arms and XYB are shown. Classifications are metadata/visual groups, not "
        "teacher-derived labels. Every class with a majority of dropped ladders is marked explicitly.", "",
        "| Content class | ycc444 | ycc420 | xyb |", "|---|---:|---:|---:|"]
    classes = sorted({r["cells"][0]["source"]["content_class"] for r in ladders})
    for cls in classes:
        values = []
        for family in ["ycc444", "ycc420", "xyb"]:
            subset = [r for r in ladders if r["key"][2] == family and r["key"][3] in ["0", "q95"]
                      and r["cells"][0]["source"]["content_class"] == cls]
            kept = sum(bool(r["admission"]["kept_indices"]) for r in subset)
            values.append(f"{kept}/{len(subset)}" + (" **majority dropped**" if kept * 2 < len(subset) else ""))
        lines.append("| " + " | ".join([cls, *values]) + " |")
    lines += ["", "## Timing", "",
        "One worker, eight threads, one pass; these are observed pilot costs, not a controlled runtime "
        "benchmark or a forecast. Existing per-cell encoder/decode timers are preserved in the TSVs. "
        "The frozen teacher also extracts unused Rev3 features, so its wall average includes that overhead.", "",
        "| Stage | Cells | Wall seconds | Wall ms/cell | Median encode ms | Median decode ms |",
        "|---|---:|---:|---:|---:|---:|"]
    total = 0
    for family in grids():
        cells = [c for r in ladders if r["key"][2] == family for c in r["cells"]]
        n = len(cells)
        total += n
        lines.append(f"| {family} | {n} | {timing[family]:.6f} | {1000 * timing[family] / n:.6f} | "
                     f"{statistics.median(float(c['encode_ms']) for c in cells):.6f} | "
                     f"{statistics.median(float(c['decode_ms']) for c in cells):.6f} |")
    lines += [f"| frozen teacher + incidental features | {total} | {timing['frozen_teacher']:.6f} | "
              f"{1000 * timing['frozen_teacher'] / total:.6f} | — | — |", "",
        "Resource wrapper: `run-heavy: done rc=0 152s | peak-RSS 0.50GiB | min-avail 25963MiB | peak-load 7.07`. "
        "16G hard cgroup limit, eight-thread cap. Zero encode/decode/sweep-score failures; frozen teacher "
        "completed every pair with zero failures.", "",
        "## Proposed full grid — not launched", "",
        "Propose the 481 distinct base-name SafeSyn 512-square tiles in the Mlic fitting partition "
        "(one named tile per available base; independent-family dedup still required), all 201 CID22 oracle TRAIN references with their original "
        "fit/development/calibration family partitions preserved, and the 386 unique TRAIN origins in "
        "K500_even: 1,068 candidate sources. A separate diagnostic proposal uses the 128 unique validate "
        "origins in ordinary K500; none enter training. These counts are metadata censuses, not new "
        "admission or independent-family qualification. A complete source-role/family audit must precede declaration.", "",
        "Size axis, maximum long edge: **32, 40, 48, 64, 80, 96, 128, 160, 192, 256, 320, 384, 512, 640, "
        "768, 960**. Use both whole-image Mitchell fits and centered square crops at each feasible size; "
        "never upscale and require every actual raster to have fewer than 1,000,000 pixels. "
        "Skip infeasible geometries explicitly in the manifest, deduplicate identical pixel views, and "
        "keep original-family groupings across every derivative. The source decoder limit must be addressed "
        "explicitly before admitting im2402 or any other oversized original.", "",
        "Use the pilot's 16 YCbCr rungs (0 baseline plus 0.5–64 in half-octave steps), all-ones Y, "
        "both 4:4:4 and 4:2:0, and the 14 existing XYB rungs at q95, with the same admission rule. "
        "That is 46 encodes/view. Nominal ceilings before no-upscale/geometry/dedup admission: "
        "1,068 × 16 × 2 × 46 = **1,572,096 TRAIN-source cells**; "
        "128 × 16 × 2 × 46 = **188,416 validate diagnostic cells**; **1,760,512 total**. "
        "These are proposed matrix upper bounds, not generated/admitted counts or runtime/storage estimates. "
        "Stage B requires the coordinator's go, exact manifests and resource admission; it must use zenfleet. "
        "Butteraugli/CVVDP diagnostic scoring, pinned Rev7 extraction and model work remain later stages.", "",
        "## Complete individual ladders", "",
        "Scores follow the rung vectors above. Kept values list chroma multiplier:score. A dropped ladder "
        "contributes no rungs, including its otherwise admissible prefix. The stop rung and score are shown "
        "so ties/reversals and near-grey cases remain visible. Rounded text is for reading; LADDERS.json "
        "preserves all teacher doubles and per-encode SHA256 links.", "",
        "| Source | Size | Family | Luma | All rung scores | Kept rungs | Stop / drop reason |",
        "|---|---:|---|---|---|---|---|"]
    for r in ladders:
        a = r["admission"]
        kept = "; ".join(f"{r['cells'][i]['rung']:.5g}:{r['cells'][i]['teacher_ssim2']:.6f}" for i in a["kept_indices"]) or "none"
        scores = "; ".join(f"{c['teacher_ssim2']:.6f}" for c in r["cells"])
        stop = "end of grid" if a["stop_index"] is None else f"nonfall at {r['cells'][a['stop_index']]['rung']:.6g}"
        if a["drop_reason"]:
            stop += "; DROP: " + a["drop_reason"]
        source, size, family, luma = r["key"]
        lines.append(f"| {source} | {size} | {family} | {luma} | {scores} | {kept} | {stop} |")
    lines += ["", "## Artifact identity and disposition", "",
        "SOURCES.json, PREPARED.json, CONTROL.json, TIMING.json, LADDERS.json, sweep TSVs, pair TSVs, "
        "frozen teacher audit/CSV and first-artifact receipts live in the run directory. JPEG content-addressed "
        "bytes are under `artifacts/zenjpeg/<sha256>.jpg`, with original encode names retained as hard links. "
        "The final coordinator report adds independent JPEG inspection, mirror readbacks, final source/analysis "
        "commit pins and workspace disposition below. No AIC-2026 statistic was read or written by this pilot.", ""]
    save(root / "SUMMARY.json", summary)
    destination.write_text("\n".join(lines))
    print("Wrote", destination)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["run", "analyze", "inspect", "report"])
    p.add_argument("root", type=Path)
    p.add_argument("--report", type=Path)
    a = p.parse_args()
    root = a.root.resolve()
    try:
        if a.action == "report":
            if a.report is None:
                p.error("report requires --report PATH")
            report(root, a.report)
        else:
            {"run": run, "analyze": analyze, "inspect": inspect}[a.action](root)
    except Exception as error:
        if a.action == "run":
            progress = json.loads((root / "PROGRESS.json").read_text()) if (root / "PROGRESS.json").exists() else {}
            progress.update(failed_stage=progress.get("stage"), stage="FAILED", error=str(error))
            save(root / "PROGRESS.json", progress)
        raise


if __name__ == "__main__":
    main()
