#!/usr/bin/env python3
"""Attribute perf self samples to explicit shared extraction-kernel buckets.

Input: `perf script --no-inline -F period,ip,sym` from full BakeScorer runs.
Only a sample's first (executing) symbol contributes self cost. Callers select
the extraction denominator, never add inclusive cost to a kernel bucket.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re


def bucket(symbol):
    if ("zensim::fused::" in symbol or "zensim::feature_v2::V1BasicSums" in symbol
            or "zensim::feature_v2::fold_v1_" in symbol):
        return "basic_fused"
    if "dense_block_kernel" in symbol or "zensim::feature_v2::OnlineMoments" in symbol:
        return "v2_dense"
    if any(s in symbol for s in ["gradient_block_kernel", "gradient_border_pixel",
                                  "gradient_interior_elem_canon", "gradient_pools_finish"]):
        return "v2_gradient"
    if "zensim::blur::" in symbol:
        return "shared_blur_pyramid"
    if "zensim::color::" in symbol or "rows_to_linear" in symbol or "rows_to_xyb" in symbol:
        return "shared_color"
    if any(s in symbol for s in ["memset", "memmove", "memcpy", "malloc", "calloc",
                                  "_int_free", "__libc_free", "_mid_memalign", "_int_memalign"]):
        return "shared_memory"
    if "blockiness" in symbol:
        return "retained_blockiness"
    return "other_extraction"


def summarize(path):
    all_period = extraction_period = sample_count = extraction_samples = 0
    weights = Counter()
    symbols = defaultdict(Counter)
    excluded_symbols = Counter()
    for sample in path.read_text().strip().split("\n\n"):
        lines = sample.strip().splitlines()
        assert re.fullmatch(r"\d+", lines[0].strip()), lines[0]
        period = int(lines[0])
        frames = []
        for line in lines[1:]:
            m = re.fullmatch(r"\s*[0-9a-f]+\s+(.+)", line)
            assert m, line
            frames.append(m[1])
        assert frames
        sample_count += 1
        all_period += period
        executing = frames[0]
        if not any("zensim::fold_engine::compute_fold_backed_stoppable" in f for f in frames):
            excluded_symbols[executing] += period
            continue
        extraction_samples += 1
        extraction_period += period
        key = bucket(executing)
        weights[key] += period
        symbols[key][executing] += period
    assert extraction_period > 0
    assert sum(weights.values()) == extraction_period
    return dict(samples=sample_count, extraction_samples=extraction_samples,
                all_period=all_period, extraction_period=extraction_period,
                extraction_share_of_whole_percent=100 * extraction_period / all_period,
                buckets={key: dict(period=value, self_percent_of_extraction=100 * value / extraction_period,
                                   symbols=dict(symbols[key].most_common()))
                         for key, value in weights.most_common()},
                outside_extraction_symbols=dict(excluded_symbols.most_common()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("study", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--host", choices=["i270", "r5900xt"], required=True)
    a = p.parse_args()
    selection = json.loads((a.study / "selection.json").read_text())
    states = {row["size"]: row for row in selection["profiles"]}
    results = {size: summarize(a.study / f"perf/{size}/samples.txt") if states[size]["source"] else None
               for size in ["64", "256", "1024", "large"]}
    a.out.write_text(json.dumps(dict(host=a.host, tier="v3", sizes=results), indent=2) + "\n")
    print(f"{a.host} / v3; period-weighted self samples; extraction denominator selected by call chain.")
    print("| Shared kernel bucket | 64² | 256² | 1024² | Large |")
    print("|---|---:|---:|---:|---:|")
    for key in ["basic_fused", "v2_dense", "v2_gradient", "shared_blur_pyramid", "shared_color",
                "shared_memory", "retained_blockiness", "other_extraction"]:
        vals = [f'{v["buckets"].get(key, {}).get("self_percent_of_extraction", 0):.3f}%' if v else "not measured (contended)"
                for v in results.values()]
        print("| " + key + " | " + " | ".join(vals) + " |")
    for size, row in results.items():
        if row is None:
            print(f'{size}: not measured (contended)')
            continue
        print(f'{size}: {row["samples"]} samples, {row["extraction_samples"]} under extraction; '
              f'{row["extraction_share_of_whole_percent"]:.3f}% of whole-call sampled cycles')


if __name__ == "__main__":
    main()
