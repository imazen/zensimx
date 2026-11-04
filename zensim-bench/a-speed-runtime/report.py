#!/usr/bin/env python3
"""Summarize retained zenbench paired rounds; refuse incomplete evidence."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

ARMS = ("zensim_a", "fast_ssim2", "margarine")
SIZES = (64, 256, 1024, 4096)


def quantile(values, p):
    values = sorted(values)
    position = (len(values) - 1) * p
    low = math.floor(position)
    high = math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def fit(points):
    xs, ys = zip(*points)
    mx, my = statistics.mean(xs), statistics.mean(ys)
    beta = sum((x - mx) * (y - my) for x, y in points) / sum((x - mx) ** 2 for x in xs)
    alpha = my - beta * mx
    residual = sum((y - alpha - beta * x) ** 2 for x, y in points)
    total = sum((y - my) ** 2 for y in ys)
    return {"alpha_ms": alpha, "beta_ms_per_MP": beta * 1e6, "r_squared": 1 - residual / total}


def summarize(root):
    assert (root / "COMPLETE").exists(), "collector has not completed"
    fixture_rows = json.loads((root / "fixtures/pairs.json").read_text())
    pairs = {r["id"]: r for r in fixture_rows}
    assert len(pairs) == 80
    rows, fits = [], []
    for threads in (1, 16):
        points = {arm: [] for arm in ARMS}
        for size in SIZES:
            path = root / f"results/{threads}t_{size}_latency.json"
            run = json.loads(path.read_text())
            assert not run["unreliable"], path
            assert len(run["comparisons"]) == 20, path
            pooled = {arm: [] for arm in ARMS}
            excluded = 0
            pixels = []
            for group in run["comparisons"]:
                _, pixel_text, pair_id = group["group_name"].split("/")
                pair = pairs[pair_id]
                assert pair["size"] == size
                pixel_count = pair["width"] * pair["height"]
                assert int(pixel_text) == pixel_count
                pixels.append(pixel_count)
                names = [b["name"] for b in group["benchmarks"]]
                assert names == list(ARMS), names
                samples = group["samples"]
                assert all(r["iterations"] == 1 for r in samples)
                clean = [r for r in samples if r["gate_clean"] is True][:32]
                assert len(clean) == 32, group["group_name"]
                excluded += sum(r["gate_clean"] is not True for r in samples)
                for arm_index, arm in enumerate(ARMS):
                    values = [r["elapsed_ns"][arm_index] / 1e6 for r in clean]
                    assert all(math.isfinite(v) and v > 0 for v in values)
                    assert all(sorted(r["execution_order"]) == [0, 1, 2] for r in clean)
                    pooled[arm].extend(values)
                    points[arm].append((pixel_count, statistics.median(values)))
            row = {"host": "i270", "tier": "v3/AVX2", "threads": threads, "size": size,
                   "images": 20, "samples_per_arm": 640, "flagged_rounds": excluded,
                   "min_pixels": min(pixels), "max_pixels": max(pixels), "mean_pixels": statistics.mean(pixels)}
            for arm in ARMS:
                row[arm] = {"p50_ms": quantile(pooled[arm], .5), "p95_ms": quantile(pooled[arm], .95)}
                stem = root / f"results/{threads}t_{size}_rss-{arm}"
                log = stem.with_suffix(".log").read_text()
                line = next(line for line in log.splitlines() if line.startswith("run-heavy: done"))
                assert "rc=0" in line, line
                time_text = stem.with_suffix(".time.txt").read_text()
                rss = int(re.search(r"Maximum resident set size \(kbytes\): (\d+)", time_text)[1])
                row[arm].update({"rss_kib": rss, "run_heavy": line})
            row["a_over_fast"] = row["zensim_a"]["p50_ms"] / row["fast_ssim2"]["p50_ms"]
            row["a_over_margarine"] = row["zensim_a"]["p50_ms"] / row["margarine"]["p50_ms"]
            rows.append(row)
        for arm in ARMS:
            assert len(points[arm]) == 80
            fits.append({"host": "i270", "threads": threads, "arm": arm, **fit(points[arm])})
    for row in rows:
        single = next(r for r in rows if r["threads"] == 1 and r["size"] == row["size"])
        for arm in ARMS:
            row[arm]["speedup_16t"] = single[arm]["p50_ms"] / next(r for r in rows if r["threads"] == 16 and r["size"] == row["size"])[arm]["p50_ms"]
    return {"rows": rows, "fits": fits}


def render(root, result):
    lines = ["# Public A versus fast-ssim2 and margarine — 2026-11-04", "",
             "Measured on **i270**, Intel Core Ultra 7 270K Plus, runtime v3/AVX2. "
             "Affinity is CPUs 0–15 for every arm and both thread counts; this is a mix of P and E cores, with no CCD boundary. "
             "These observations do not measure Zen 5.", "",
             "Each cell uses 20 real images, zenjpeg q40 YCbCr 4:2:0 reconstructions, "
             "100 ms warmup per arm/image, randomized interleaved single-call zenbench rounds. "
             "The first 32 clean rounds per image are retained with shared indices across arms (640 calls per arm/cell); "
             "all raw rounds remain archived. p50/p95 are type-7 quantiles of those individual calls, pooled equally across images. "
             "No native CPU compilation flags. Both builds enable fast-ssim2's rayon feature; "
             "RAYON_NUM_THREADS is 1 or 16, and public A uses with_parallel(false/true). "
             "All three arms score the same resident RGB8 sRGB bytes in process; decode, resize, JPEG encoding and I/O are excluded. "
             "The unchanged private margarine kernel is included directly; no CLI or process-start cost enters latency. "
             "Margarine uses simd-row-malta, 128 rows/512 columns, primary max pooling; lower means less distortion.", "",
             "Small cells mix ten CID22 TRAIN and ten CLIC TRAIN sources; 1024² uses twenty CLIC-1024 TRAIN images. "
             "The large cell uses twenty canonical imazen-26 PNG-v3 SDR renders across three folders, "
             "native when the long edge is below 4096, otherwise Mitchell-downscaled to long edge 4096, never enlarged. "
             "The smaller cells use a central square crop and zenresize Mitchell. This is runtime evidence, not a quality comparison.", "",
             "| Host | Threads | Size | A p50/p95 ms | fast-ssim2 p50/p95 ms | margarine p50/p95 ms | A/fast | A/margarine |",
             "|---|---:|---|---:|---:|---:|---:|---:|"]
    for row in result["rows"]:
        vals = [f'{row[a]["p50_ms"]:.4f}/{row[a]["p95_ms"]:.4f}' for a in ARMS]
        lines.append(f'| i270 | {row["threads"]} | {row["size"]} class | '+" | ".join(vals)+f' | {row["a_over_fast"]:.3f} | {row["a_over_margarine"]:.3f} |')
    lines += ["", "Ratios below one favor A. The large-cell actual pixel count ranges from "
              f'{result["rows"][3]["min_pixels"]:,} to {result["rows"][3]["max_pixels"]:,}; '
              f'mean {result["rows"][3]["mean_pixels"]:,.1f}.', "",
              "| Host | Size | A 16-thread speedup | fast-ssim2 speedup | margarine speedup |", "|---|---|---:|---:|---:|"]
    for row in result["rows"][:4]:
        lines.append(f'| i270 | {row["size"]} class | '+" | ".join(f'{row[a]["speedup_16t"]:.3f}×' for a in ARMS)+" |")
    lines += ["", "OLS fits use the 80 per-image medians against actual pixels, separately for each arm/thread count: "
              "`time_ms = alpha_ms + beta_ms_per_MP * pixels / 1,000,000`. Intercepts are unconstrained regression terms, "
              "not measured process-start costs. Content changes at the largest size, so these are descriptive fits rather than causal scaling laws.", "",
              "| Host | Threads | Arm | α ms | β ms/MP | R² |", "|---|---:|---|---:|---:|---:|"]
    for fit_row in result["fits"]:
        lines.append(f'| i270 | {fit_row["threads"]} | {fit_row["arm"]} | {fit_row["alpha_ms"]:.5f} | {fit_row["beta_ms_per_MP"]:.5f} | {fit_row["r_squared"]:.5f} |')
    lines += ["", "RSS uses a fresh process for each arm/thread/size and scores all twenty pairs with one pair resident at a time. "
              "Values include inputs, the runtime, allocator retention and scoring scratch. GNU time supplies exact KiB; "
              "run-heavy's rounded total-process line is retained below.", "",
              "| Host | Threads | Size | A peak KiB | fast-ssim2 peak KiB | margarine peak KiB |", "|---|---:|---|---:|---:|---:|"]
    for row in result["rows"]:
        lines.append(f'| i270 | {row["threads"]} | {row["size"]} class | '+" | ".join(str(row[a]["rss_kib"]) for a in ARMS)+" |")
    lines += ["", "```text"]
    for row in result["rows"]:
        for arm in ARMS:
            lines.append(f'i270 {row["threads"]}t {row["size"]} {arm}: {row[arm]["run_heavy"]}')
    lines += ["```", "", "Source/build identities and exact reproduction commands are recorded in the adjacent metadata file. "
              "Raw outputs, fixture bytes and hashes are retained outside git in the adjacent artifact pointer.", "",
              "Canonical large-cell source bytes (paths relative to `imazen-26-png-v3/`):", "",
              "| File | SHA-256 |", "|---|---|"]
    for row in json.loads((root / "source-provenance.json").read_text()):
        if "/png-v3/" in row["source"]:
            relative = "png-v3/" + row["source"].split("/png-v3/", 1)[1]
            lines.append(f'| `{relative}` | `{row["sha256"]}` |')
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = summarize(args.root)
    args.output.write_text(render(args.root, result))
    summary = args.root / "SUMMARY.json"
    summary.write_text(json.dumps(result, indent=2) + "\n")
    print(hashlib.sha256(summary.read_bytes()).hexdigest(), summary)


if __name__ == "__main__":
    main()
