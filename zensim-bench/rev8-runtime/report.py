#!/usr/bin/env python3
"""Admit complete Rev8 cells from raw captures and render the paired matrix."""
import argparse
import json
import math
from pathlib import Path
import re
import statistics

ARMS = ("zensim-a-rev7", "rev8-alg-s4", "rev8-alg-s5", "rev8-hybrid-s5")
PLACEMENTS = (("dev", "v4x"), ("dev", "v3"), ("i265", "v3"))
SIZES = ((64, 64), (256, 256), (840, 944), (1024, 1024), (4000, 3000))
THREADS = (1, 16)
IMAGES = 20
ROUNDS = 32


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def quantile(samples, p):
    ordered = sorted(samples)
    position = p * (len(ordered) - 1)
    lo = math.floor(position)
    hi = math.ceil(position)
    return ordered[lo] + (position - lo) * (ordered[hi] - ordered[lo])


def receipt(log, tier, threads, geometry):
    data = log.read_text()
    name = {"v3": "X64V3Token", "v4x": "X64V4xToken"}[tier]
    require(f"tier={name} threads={threads} geometry={geometry}" in data,
            f"incorrect tier, pool or geometry in {log.name}")
    match = re.search(r"^run-heavy: done rc=0 .*peak-RSS[^\n]*$", data, re.MULTILINE)
    require(match is not None, f"successful run-heavy receipt missing in {log.name}")
    return data, match.group()


def latency(root, stem, tier, threads, geometry):
    log, summary = receipt(root / f"{stem}_latency.log", tier, threads, geometry)
    require(not re.search(r'foreign pid=\d+ name="(cc1plus|cc1|rustc|clippy-driver|cargo|cmake|ninja|zensim_mlp_trai)"', log),
            "foreign compilation or training during the capture")
    run = json.loads((root / f"{stem}_latency.json").read_text())
    require(not run["unreliable"], "zenbench declined capture")
    groups = {g["group_name"]: g for g in run["comparisons"]}
    expected = [f"{geometry}/{i:02d}" for i in range(IMAGES)]
    require(len(run["comparisons"]) == IMAGES and sorted(groups) == expected, "wrong fixture groups")
    pooled = [[] for _ in ARMS]
    excluded = 0
    for name in expected:
        group = groups[name]
        require([b["name"] for b in group["benchmarks"]] == list(ARMS), f"wrong arms at {name}")
        samples = group["samples"]
        clean = [r for r in samples if r["gate_clean"] is True and r["iterations"] == 1][:ROUNDS]
        require(len(clean) == ROUNDS, f"{name} has {len(clean)} clean single-call rounds")
        excluded += sum(r["gate_clean"] is not True for r in samples)
        for round_ in clean:
            values = round_["elapsed_ns"]
            require(sorted(round_["execution_order"]) == list(range(len(ARMS)))
                    and len(values) == len(ARMS) and all(v > 0 for v in values),
                    f"invalid paired round at {name}")
            for arm, v in zip(pooled, values):
                arm.append(v / 1e6)
    return {
        arm: {"p50_ms": statistics.median(samples), "p95_ms": quantile(samples, .95)}
        for arm, samples in zip(ARMS, pooled)
    }, excluded, summary


def rss(root, stem, arm, tier, threads, geometry):
    log, summary = receipt(root / f"{stem}_{arm}.log", tier, threads, geometry)
    scores = re.findall(r"^score (\d{2}) ([0-9a-f]{16})$", log, re.MULTILINE)
    require([i for i, _ in scores] == [f"{i:02d}" for i in range(IMAGES)], "wrong RSS fixture scores")
    time = (root / f"{stem}_{arm}.time.txt").read_text()
    match = re.search(r"Maximum resident set size \(kbytes\): (\d+)", time)
    require(match is not None and "Exit status: 0" in time, "invalid GNU time receipt")
    return {"rss_kib": int(match[1]), "rss_run_heavy": summary, "score_bits": [v for _, v in scores]}


def search(roots, run):
    errors = []
    for root in roots:
        try:
            return run(root), root.name
        except (OSError, ValueError, KeyError) as error:
            errors.append(f"{root.name}: {error}")
    raise ValueError(" | ".join(errors) or "no capture roots")


def summarize(artifacts, digest):
    cells, missing = [], []
    for host, tier in PLACEMENTS:
        roots = [
            p for p in sorted(artifacts.glob(f"{host}-{tier}*"), reverse=True)
            if (p / "collector.tsv").is_file()
            and (p / "collector.tsv").read_text().split()[:2] == [host, tier]
            and (p / "binary.sha256").read_text().split()[0] == digest
        ]
        for threads in THREADS:
            for width, height in SIZES:
                geometry = f"{width}x{height}"
                stem = f"{threads}t_{geometry}"
                cell = {"host": host, "tier": tier, "threads": threads, "geometry": geometry,
                        "samples_per_arm": IMAGES * ROUNDS, "arms": {arm: {} for arm in ARMS}}
                try:
                    result, origin = search(roots, lambda p: latency(p, stem, tier, threads, geometry))
                    arms, excluded, receipt_ = result
                    cell.update(excluded_rounds=excluded, latency_run_heavy=receipt_, latency_capture=origin)
                    baseline = arms[ARMS[0]]["p50_ms"]
                    for arm, data in arms.items():
                        median = data["p50_ms"]
                        cell["arms"][arm].update(data, mpx_s=width * height / (median * 1000), speedup=baseline / median)
                except ValueError as error:
                    missing.append((host, tier, stem, "latency", str(error)))

                for arm in ARMS:
                    try:
                        data, origin = search(roots, lambda p: rss(p, stem, arm, tier, threads, geometry))
                        cell["arms"][arm].update(data, rss_capture=origin)
                    except ValueError as error:
                        missing.append((host, tier, stem, arm, str(error)))
                cells.append(cell)
    by_geometry = {}
    for cell in cells:
        key = (cell["host"], cell["tier"], cell["geometry"])
        by_geometry.setdefault(key, {})[cell["threads"]] = cell["arms"]
    thread_score_checks = []
    for key, counts in by_geometry.items():
        for arm in ARMS:
            a = counts[1][arm].get("score_bits")
            b = counts[16][arm].get("score_bits")
            if a is not None and b is not None:
                thread_score_checks.append({
                    "host": key[0], "tier": key[1], "geometry": key[2], "arm": arm, "pairs": len(a),
                    "different": [i for i, (x, y) in enumerate(zip(a, b)) if x != y],
                })
    return {"runtime_sha256": digest, "cells": cells, "missing": missing,
            "thread_score_checks": thread_score_checks}


def matrix(data):
    lines = [
        "| Host / tier | Threads | Geometry | Rev7 | Alg S4 | Alg S5 | Hybrid S5 |",
        "|---|---:|---|---|---|---|---|",
    ]
    for cell in data["cells"]:
        line = [f'{cell["host"]} / {cell["tier"]}', cell["threads"], cell["geometry"]]
        for arm in ARMS:
            d = cell["arms"][arm]
            timing = (f'{d["p50_ms"]:.3f}/{d["p95_ms"]:.3f} · {d["mpx_s"]:.2f} · {d["speedup"]:.3f}×'
                      if "p50_ms" in d else "—")
            lines_number = f'{d["rss_kib"]:,}' if "rss_kib" in d else "—"
            line.append(f"{timing} · {lines_number}")
        lines.append("| " + " | ".join(map(str, line)) + " |")
    return "\n".join(lines) + "\n"



def descriptive_cell(root, stem, tier, threads, geometry):
    log, run_heavy = receipt(root / f"{stem}_descriptive.log", tier, threads, geometry)
    fixtures = re.findall(r"^fixture\t(\d{2})\t([0-9a-f]{16})\t([0-9a-f]{16})$", log, re.MULTILINE)
    require(len(fixtures) == 1, "expected one resident fixture")
    samples = re.findall(r"^sample\t(\d)\t(\d)\t([^\t]+)\t(\d+)\t([0-9a-f]{16})$", log, re.MULTILINE)
    expected = [(str(round_), str(slot), arm)
                for round_ in range(1, 4)
                for slot, arm in enumerate(ARMS if round_ % 2 else reversed(ARMS), start=1)]
    require([sample[:3] for sample in samples] == expected, "invalid round count or alternating arm order")
    per_arm = {}
    for arm in ARMS:
        ns = [int(row[3]) for row in samples if row[2] == arm]
        scores = {row[4] for row in samples if row[2] == arm}
        require(all(x > 0 for x in ns) and len(scores) == 1, "invalid timing or nonrepeatable score")
        per_arm[arm] = {
            "median_ms": statistics.median(ns) / 1e6, "min_ms": min(ns) / 1e6,
            "max_ms": max(ns) / 1e6, "ns": ns, "score_bits": scores.pop(),
        }
    return fixtures[0], per_arm, run_heavy


def bounded(artifacts, clean_digest, digest):
    prior = summarize(artifacts, clean_digest)
    clean = [cell for cell in prior["cells"] if "latency_capture" in cell]
    desc, missing = [], []
    fixtures = {}
    for host, tier in PLACEMENTS:
        roots = [
            p for p in sorted(artifacts.glob(f"{host}-{tier}*"), reverse=True)
            if (p / "collector.tsv").is_file()
            and (p / "collector.tsv").read_text().split()[:2] == [host, tier]
            and (p / "binary.sha256").read_text().split()[0] == digest
        ]
        for threads in THREADS:
            for geometry in ("1024x1024", "4000x3000"):
                stem = f"{threads}t_{geometry}"
                ident = {"host": host, "tier": tier, "threads": threads, "geometry": geometry}
                try:
                    result, origin = search(roots, lambda p: descriptive_cell(p, stem, tier, threads, geometry))
                    fixture, arms, timing_receipt = result
                    require(fixtures.setdefault(geometry, fixture) == fixture, "different resident fixture bytes")
                    for item in arms.values():
                        item["speedup"] = arms[ARMS[0]]["median_ms"] / item["median_ms"]
                        width, height = map(int, geometry.split("x"))
                        item["mpx_s"] = width * height / (1000 * item["median_ms"])
                    desc.append({**ident, "capture": origin, "fixture": fixture, "arms": arms,
                                 "run_heavy": timing_receipt})
                except ValueError as error:
                    missing.append({**ident, "mode": "descriptive", "reason": str(error)})

                if host == "i265":
                    prior_cell = next(c for c in prior["cells"]
                                      if all(c[k] == v for k, v in ident.items()))
                    for arm in ARMS:
                        if "rss_kib" in prior_cell["arms"][arm]:
                            continue
                        try:
                            data, origin = search(roots, lambda p: rss(p, stem, arm, tier, threads, geometry))
                            prior_cell["arms"][arm].update(data, rss_capture=origin, rss_binary_sha256=digest)
                        except ValueError as error:
                            missing.append({**ident, "mode": arm, "reason": str(error)})

    keys = {(c["host"], c["tier"], c["geometry"]) for c in desc}
    for host, tier, geometry in sorted(keys):
        pair = {c["threads"]: c for c in desc if (c["host"], c["tier"], c["geometry"]) == (host, tier, geometry)}
        if len(pair) == 2:
            for arm in ARMS:
                a = pair[1]["arms"][arm]["score_bits"]
                b = pair[16]["arms"][arm]["score_bits"]
                require(a == b, f"{host}/{tier}/{geometry}/{arm} differed between 1T and 16T")
    return {
        "clean_binary_sha256": clean_digest, "bounded_binary_sha256": digest, "fixtures": fixtures,
        "descriptive": desc, "accepted_zenbench": clean, "rss_and_original_cells": prior["cells"],
        "dev_rss_thread_score_checks": prior["thread_score_checks"], "missing": missing,
    }


def bounded_matrix(data):
    lines = [
        "| Host / tier | Threads | Geometry | Fixture | Rev7 | Alg S4 | Alg S5 | Hybrid S5 |",
        "|---|---:|---|---:|---|---|---|---|",
    ]
    for c in data["descriptive"]:
        line = [f'{c["host"]} / {c["tier"]}', c["threads"], c["geometry"], c["fixture"][0]]
        for arm in ARMS:
            item = c["arms"][arm]
            line.append(f'{item["median_ms"]:.3f} [{item["min_ms"]:.3f}, {item["max_ms"]:.3f}]'
                        f' · {item["mpx_s"]:.2f} · {item["speedup"]:.3f}×')
        lines.append("| " + " | ".join(map(str, line)) + " |")
    return "\n".join(lines) + "\n"

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("runtime_sha256")
    parser.add_argument("output", type=Path)
    parser.add_argument("--partial", action="store_true")
    parser.add_argument("--bounded-sha256")
    args = parser.parse_args()
    if args.bounded_sha256:
        data = bounded(args.artifacts, args.runtime_sha256, args.bounded_sha256)
        view = bounded_matrix(data)
    else:
        data = summarize(args.artifacts, args.runtime_sha256)
        view = matrix(data)
    args.output.with_suffix(".json").write_text(json.dumps(data, indent=2) + "\n")
    args.output.with_suffix(".md").write_text(view)
    if data["missing"] and not args.partial:
        raise SystemExit(f'{len(data["missing"])} missing or rejected capture cells')


if __name__ == "__main__":
    main()
