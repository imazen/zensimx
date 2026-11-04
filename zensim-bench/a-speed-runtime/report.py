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


def size_label(size):
    return f"{size}²" if size != 4096 else "large (12 MP class)"


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


def command_startup(root):
    rows = []
    for threads in (1, 16):
        path = root / f"startup-{threads}t.json"
        log = path.with_suffix(".log").read_text()
        assert not re.findall(r'foreign pid=\d+ name="(cc1plus|cc1|rustc|cargo|cmake|ninja|zensim_mlp_trai)"', log)
        run = json.loads(path.read_text())
        assert not run["unreliable"]
        assert len(run["comparisons"]) == 1
        group = run["comparisons"][0]
        assert group["group_name"] == "margarine/command-help-startup"
        assert [b["name"] for b in group["benchmarks"]] == ["margarine_help"]
        assert all(s["iterations"] == 1 for s in group["samples"])
        clean = [s for s in group["samples"] if s["gate_clean"] is True][:32]
        assert len(clean) == 32
        assert all(s["execution_order"] == [0] for s in clean)
        values = [s["elapsed_ns"][0] / 1e6 for s in clean]
        assert all(math.isfinite(v) and v > 0 for v in values)
        rows.append({"host": "i270", "threads_environment": threads, "calls": 32,
                     "p50_ms": quantile(values, .5), "p95_ms": quantile(values, .95)})
    return rows


def summarize(root):
    assert (root / "measurement-host.txt").read_text().strip() == "i270", "this report requires the recorded i270 measurement host"
    assert (root / "COMPLETE").exists(), "collector has not completed"
    fixture_rows = json.loads((root / "fixtures/pairs.json").read_text())
    pairs = {r["id"]: r for r in fixture_rows}
    assert len(pairs) == 80
    rows, fits, rss_scores = [], [], {}
    for threads in (1, 16):
        points = {arm: [] for arm in ARMS}
        for size in SIZES:
            path = root / f"results/{threads}t_{size}_latency.json"
            latency_log = path.with_suffix(".log").read_text()
            forbidden = re.findall(r'foreign pid=\d+ name="(cc1plus|cc1|rustc|cargo|cmake|ninja|zensim_mlp_trai)"', latency_log)
            assert not forbidden, (path, "foreign build/training observed", forbidden)
            run = json.loads(path.read_text())
            assert not run["unreliable"], path
            assert len(run["comparisons"]) == 20, path
            pooled = {arm: [] for arm in ARMS}
            excluded = 0
            pixels = []
            seen = set()
            for group in run["comparisons"]:
                _, pixel_text, pair_id = group["group_name"].split("/")
                pair = pairs[pair_id]
                assert pair_id not in seen, (path, "duplicate image", pair_id)
                seen.add(pair_id)
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
            assert seen == {r["id"] for r in fixture_rows if r["size"] == size}
            row = {"host": "i270", "tier": "v3/AVX2", "threads": threads, "size": size,
                   "images": 20, "samples_per_arm": 640, "flagged_rounds": excluded,
                   "min_pixels": min(pixels), "max_pixels": max(pixels), "mean_pixels": statistics.mean(pixels)}
            for arm in ARMS:
                row[arm] = {"p50_ms": quantile(pooled[arm], .5), "p95_ms": quantile(pooled[arm], .95)}
                stem = root / f"results/{threads}t_{size}_rss-{arm}"
                log = stem.with_suffix(".log").read_text()
                score_rows = [m for line in log.splitlines()
                              if (m := re.fullmatch(rf"(\d+_\d+)\t{arm}\t([-+0-9.eE]+)", line))]
                assert len(score_rows) == len(seen), (stem, "RSS image coverage differs")
                scores = {m[1]: m[2] for m in score_rows}
                assert set(scores) == seen, (stem, "RSS image coverage differs")
                assert all(math.isfinite(float(value)) for value in scores.values())
                rss_scores[(threads, size, arm)] = scores
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
    score_diagnostics = []
    for size in SIZES:
        for arm in ARMS:
            one, sixteen = rss_scores[(1, size, arm)], rss_scores[(16, size, arm)]
            mismatches = [{"id": key, "one_thread": one[key], "sixteen_threads": sixteen[key]}
                          for key in one if one[key] != sixteen[key]]
            score_diagnostics.append({"size": size, "arm": arm, "pairs": 20,
                                      "printed_score_mismatches": mismatches})
    return {"rows": rows, "fits": fits, "printed_score_thread_diagnostics": score_diagnostics}


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
        lines.append(f'| i270 | {row["threads"]} | {size_label(row["size"])} | '+" | ".join(vals)+f' | {row["a_over_fast"]:.3f} | {row["a_over_margarine"]:.3f} |')
    conclusions = []
    for threads in (1, 16):
        subset = [r for r in result["rows"] if r["threads"] == threads]
        for peer, label in (("fast_ssim2", "fast-ssim2"), ("margarine", "margarine")):
            faster = [size_label(r["size"]) for r in subset if r["zensim_a"]["p50_ms"] < r[peer]["p50_ms"]]
            slower = [size_label(r["size"]) for r in subset if r["zensim_a"]["p50_ms"] > r[peer]["p50_ms"]]
            conclusions.append(f'On i270 at {threads} thread{"s" if threads != 1 else ""}, A has lower p50 than {label} at {", ".join(faster) or "no sizes"}'
                               + (f' and higher p50 at {", ".join(slower)}.' if slower else '.'))
    for threads in (1, 16):
        subset = [r for r in result["rows"] if r["threads"] == threads]
        first, last = subset[0], subset[-1]
        conclusions.append(f'At {threads} thread{"s" if threads != 1 else ""}, A/fast changes from {first["a_over_fast"]:.3f} at 64² to {last["a_over_fast"]:.3f} at large, '
                           f'and A/margarine from {first["a_over_margarine"]:.3f} to {last["a_over_margarine"]:.3f}.')
    lines += ["", " ".join(conclusions)]
    lines += ["", "Ratios below one favor A. The large-cell actual pixel count ranges from "
              f'{result["rows"][3]["min_pixels"]:,} to {result["rows"][3]["max_pixels"]:,}; '
              f'mean {result["rows"][3]["mean_pixels"]:,.1f}.', "",
              "| Host | Size | A 16-thread speedup | fast-ssim2 speedup | margarine speedup |", "|---|---|---:|---:|---:|"]
    for row in result["rows"][:4]:
        lines.append(f'| i270 | {size_label(row["size"])} | '+" | ".join(f'{row[a]["speedup_16t"]:.3f}×' for a in ARMS)+" |")
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
        lines.append(f'| i270 | {row["threads"]} | {size_label(row["size"])} | '+" | ".join(str(row[a]["rss_kib"]) for a in ARMS)+" |")
    lines += ["", "```text"]
    for row in result["rows"]:
        for arm in ARMS:
            lines.append(f'i270 {row["threads"]}t {row["size"]} {arm}: {row[arm]["run_heavy"]}')
    mismatches = sum(len(d["printed_score_mismatches"]) for d in result["printed_score_thread_diagnostics"])
    lines += ["```", "", f"The isolated RSS calls also supply 240 per-pair, per-arm comparisons of printed scores between thread counts; {mismatches} differ. "
              "This diagnostic compares printed values, not a separate numerical qualification study.", "",
              "Source/build identities and exact reproduction commands are recorded in the "
              "[metadata](a_speed_vs_fastssim2_margarine_2026-11-04.meta.md). "
              "Raw outputs, fixture bytes and hashes are retained outside git in the "
              "[artifact pointer](a_speed_vs_fastssim2_margarine_2026-11-04.artifacts.md).", "",
              ]
    if "command_startup" in result:
        lines += ["The original margarine CLI is also built unchanged against the frozen dependency graph. "
                  "Separate zenbench measurements time a fresh `margarine --help` process, including spawn, executable loading, argument handling and exit. "
                  "This no-input baseline does not initialize image decoding or the scoring Rayon pool, and is not subtracted from resident-pixel timings. "
                  "It does not measure end-to-end CLI scoring latency. Warmup is 100 ms; the first 32 clean single-call rounds are retained.", "",
                  "| Host | RAYON_NUM_THREADS environment | Command startup p50/p95 ms |", "|---|---:|---:|"]
        for row in result["command_startup"]:
            lines.append(f'| i270 | {row["threads_environment"]} | {row["p50_ms"]:.4f}/{row["p95_ms"]:.4f} |')
        lines.append("")
    lines += ["Canonical large-cell source bytes (paths relative to `imazen-26-png-v3/`):", "",
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
    parser.add_argument("--command-startup", action="store_true", help="require both separate command-startup runs")
    parser.add_argument("--parallel-gate", choices=("before", "after"), help="validate the A serial/parallel matrix")
    args = parser.parse_args()
    if args.parallel_gate:
        result = parallel_gate(args.root, args.parallel_gate)
        args.output.write_text(render_parallel_gate(result))
        (args.root / "SUMMARY.json").write_text(json.dumps(result, indent=2) + "\n")
        return
    result = summarize(args.root)
    if args.command_startup:
        result["command_startup"] = command_startup(args.root)
    args.output.write_text(render(args.root, result))
    summary = args.root / "SUMMARY.json"
    summary.write_text(json.dumps(result, indent=2) + "\n")
    print(hashlib.sha256(summary.read_bytes()).hexdigest(), summary)


GATE_SIZES = (64, 96, 128, 160, 192, 256, 320, 384, 512, 640, 768, 1024)


def parallel_gate(root, phase):
    """Retain one shared first-32-clean index set for the two arms."""
    fixtures = json.loads((root / "fixtures/pairs.json").read_text())
    pairs = {r["id"]: r for r in fixtures}
    assert len(pairs) == len(fixtures), "duplicate fixture IDs"
    rows = []
    for stage in (("before",) if phase == "before" else ("before", "after")):
        run_root = root / stage
        assert (run_root / "measurement-host.txt").read_text().strip() == "i270"
        for threads in (16, 4):
            for size in (*GATE_SIZES, 4096) if threads == 16 else GATE_SIZES:
                expected = {r["id"] for r in fixtures if r["size"] == size}
                assert len(expected) == 20
                path = run_root / f"results/{threads}t_{size}_latency.json"
                log = path.with_suffix(".log").read_text()
                assert "run-heavy: done rc=0" in log, (path, "runner failed")
                assert not re.findall(r'foreign pid=\d+ name="(cc1plus|cc1|rustc|cargo|cmake|ninja|make|zensim_mlp_trai)"', log), (path, "foreign build/training")
                run = json.loads(path.read_text())
                assert not run["unreliable"]
                assert len(run["comparisons"]) == 20
                seen, pooled, excluded = set(), [[], []], 0
                per_image = [[], []]
                for group in run["comparisons"]:
                    label, pixels, pair_id = group["group_name"].split("/")
                    assert pair_id in expected and pair_id not in seen
                    seen.add(pair_id)
                    pair = pairs[pair_id]
                    assert int(label) == size and int(pixels) == pair["width"] * pair["height"]
                    assert [b["name"] for b in group["benchmarks"]] == ["serial", "parallel"]
                    samples = group["samples"]
                    assert all(s["iterations"] == 1 for s in samples)
                    clean = [s for s in samples if s["gate_clean"] is True][:32]
                    assert len(clean) == 32, (path, pair_id, "insufficient clean rounds")
                    excluded += sum(s["gate_clean"] is not True for s in samples)
                    for i in (0, 1):
                        values = [s["elapsed_ns"][i] / 1e6 for s in clean]
                        assert all(math.isfinite(v) and v > 0 for v in values)
                        assert all(sorted(s["execution_order"]) == [0, 1] for s in clean)
                        pooled[i].extend(values)
                        per_image[i].append((int(pixels), statistics.median(values)))
                assert seen == expected
                score_log = run_root / f"results/{threads}t_{size}_scores.log"
                score_text = score_log.read_text()
                assert "run-heavy: done rc=0" in score_text
                scores = {}
                for line in score_text.splitlines():
                    fields = line.split("\t")
                    if len(fields) == 5 and fields[0] in expected:
                        pair_id, serial, parallel, serial_bits, parallel_bits = fields
                        assert pair_id not in scores
                        assert serial == parallel and serial_bits == parallel_bits, (pair_id, "score moved")
                        scores[pair_id] = serial_bits
                assert set(scores) == expected, (score_log, "incomplete scores")
                row = {"stage": stage, "threads": threads, "size": size, "images": 20,
                       "samples_per_arm": 640, "excluded_rounds": excluded, "score_bits": scores,
                       "runner": next(line for line in log.splitlines() if line.startswith("run-heavy: done"))}
                for i, arm in enumerate(("serial", "parallel")):
                    row[arm] = {"p50_ms": quantile(pooled[i], .5), "p95_ms": quantile(pooled[i], .95)}
                row["speedup"] = row["serial"]["p50_ms"] / row["parallel"]["p50_ms"]
                rows.append(row)
    before = {(r["threads"], r["size"]): r for r in rows if r["stage"] == "before"}
    for row in rows:
        assert row["score_bits"] == before[(row["threads"], row["size"])]["score_bits"], "before/after scores moved"
    for size in GATE_SIZES:
        assert before[(4, size)]["score_bits"] == before[(16, size)]["score_bits"], "thread-count scores moved"
    crossovers = {str(t): next((s for s in GATE_SIZES if before[(t, s)]["speedup"] > 1), None) for t in (16, 4)}
    assert all(crossovers.values()), "no measured crossover"
    gate_side = max(crossovers.values())
    return {"rows": rows, "crossovers": crossovers, "parallel_min_pixels": gate_side ** 2,
            "score_comparisons": len(rows) * 20, "score_differences": 0}


def render_parallel_gate(result):
    lines = ["# A parallel extraction gate — 2026-11-04", "",
             "i270, Intel Core Ultra 7 270K Plus, v3/AVX2, CPUs 0–15. "
             "Each cell pools 640 single-call timings per arm: 20 images, first 32 clean paired zenbench rounds, "
             "100 ms warmup per arm/image. Type-7 p50/p95, milliseconds. No target-cpu=native.", "",
             "Sizes through 512² use ten CID22 TRAIN + ten CLIC TRAIN sources; larger square cells use "
             "twenty CLIC TRAIN sources because the CID22 fixtures are 512². Mitchell downscale, never enlarged; "
             "zenjpeg q40 4:2:0 reconstruction. The 12 MP cell reuses the original twenty PNG-v3 fixtures.", "",
             "| Size | 16T serial p50/p95 | 16T parallel p50/p95 | Serial/parallel | 4T serial p50/p95 | 4T parallel p50/p95 | Serial/parallel |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    index = {(r["stage"], r["threads"], r["size"]): r for r in result["rows"]}
    def timing(row, arm):
        return f'{row[arm]["p50_ms"]:.4f}/{row[arm]["p95_ms"]:.4f}'
    for size in GATE_SIZES:
        a, b = (index[("before", t, size)] for t in (16, 4))
        lines.append(f'| {size}² | {timing(a, "serial")} | {timing(a, "parallel")} | {a["speedup"]:.3f}× | {timing(b, "serial")} | {timing(b, "parallel")} | {b["speedup"]:.3f}× |')
    has_after = ("after", 16, 64) in index
    lines += ["", f'Crossover: 16T {result["crossovers"]["16"]}²; 4T {result["crossovers"]["4"]}². '
              f'Gate: **{result["parallel_min_pixels"]:,} pixels**, the higher crossover.', "",
              f'Printed and bitwise serial/parallel scores: **{result["score_differences"]} differences in {result["score_comparisons"]} comparisons**. '
              + ('Before/after and pool-size score bits also match.' if has_after else 'Pool-size score bits also match; after measurements are pending.'), ""]
    if has_after:
        lines += ["| Threads | Size | Serial before/after p50 ms | Parallel before/after p50 ms | Parallel before/after p95 ms |", "|---:|---|---:|---:|---:|"]
        for threads in (16, 4):
            for size in (*GATE_SIZES, 4096) if threads == 16 else GATE_SIZES:
                a, b = (index[(stage, threads, size)] for stage in ("before", "after"))
                lines.append(f'| {threads} | {size_label(size)} | {a["serial"]["p50_ms"]:.4f}/{b["serial"]["p50_ms"]:.4f} | {a["parallel"]["p50_ms"]:.4f}/{b["parallel"]["p50_ms"]:.4f} | {a["parallel"]["p95_ms"]:.4f}/{b["parallel"]["p95_ms"]:.4f} |')
    lines += ["", "Runner resource observations (whole-process peaks, not a separate memory comparison):", "", "```text"]
    lines += [f'{r["stage"]} {r["threads"]}T {r["size"]}: {r["runner"]}' for r in result["rows"]]
    return "\n".join(lines + ["```", ""])


if __name__ == "__main__":
    main()
