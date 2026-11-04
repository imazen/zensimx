#!/usr/bin/env python3
"""Summarize retained paired single-call zenbench samples and pixel-cost fits."""
import argparse
import json
import math
from pathlib import Path
import random
import statistics


def quantile(values, q):
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def fit(points):
    xs, ys = zip(*points)
    mx, my = statistics.mean(xs), statistics.mean(ys)
    slope = sum((x - mx) * (y - my) for x, y in points) / sum((x - mx) ** 2 for x in xs)
    intercept = my - slope * mx
    residual = sum((y - intercept - slope * x) ** 2 for x, y in points)
    total = sum((y - my) ** 2 for y in ys)
    return dict(alpha_ms=intercept, beta_ms_per_MP=slope,
                r_squared=1 - residual / total, images=len(points))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("study", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--host", choices=["i270", "r5900xt"], required=True)
    a = p.parse_args()
    pairs = {r["id"]: r for r in json.loads((a.study / "pairs/pairs.json").read_text())}
    selection = json.loads((a.study / "selection.json").read_text())
    cells, fit_rows = [], []
    for threads in [1, 16]:
        points = {name: [] for name in ["extract492", "extract336", "compute_A"]}
        for size in ["64", "256", "1024", "large"]:
            root = a.study / f"timing/{size}-{threads}t"
            expected = {key for key, row in pairs.items() if row["size"] == size}
            outside = a.host == "i270" and size in ["64", "256"]
            files = {f.stem: f for f in root.glob("*.json") if f.stem in pairs}
            assert files.keys() <= expected, (root, len(files), len(expected))
            all_samples = {name: [] for name in points}
            deltas = []
            retained, excluded = 0, 0
            for case_id, path in sorted(files.items()):
                result = json.loads(path.read_text())
                assert not result["unreliable"], (case_id, "zenbench unreliable")
                assert len(result["comparisons"]) == 1
                group = result["comparisons"][0]
                names = [b["name"] for b in group["benchmarks"]]
                assert set(names) == set(points)
                assert len(group["samples"]) == 16
                samples = {name: [] for name in names}
                for row in group["samples"]:
                    assert row["iterations"] == 1, "p95 needs single-call samples"
                    assert sorted(row["execution_order"]) == list(range(3))
                    assert row["gate_clean"] is not None, "gates must be enabled"
                    if not row["gate_clean"]:
                        excluded += 1
                        continue
                    retained += 1
                    for name, ns in zip(names, row["elapsed_ns"]):
                        samples[name].append(ns / 1e6)
                assert len(samples[names[0]]) >= 16, (case_id, "insufficient clean paired rounds")
                medians = {name: statistics.median(values) for name, values in samples.items()}
                deltas.append(medians["extract492"] - medians["extract336"])
                pixels = pairs[case_id]["width"] * pairs[case_id]["height"]
                for name, values in samples.items():
                    all_samples[name].extend(values)
                    points[name].append((pixels / 1e6, medians[name]))
            rng = random.Random(20261104)
            bootstrap = ([statistics.mean(rng.choices(deltas, k=len(deltas))) for _ in range(10000)]
                         if len(deltas) >= 2 else [])
            cells.append(dict(size=size, threads=threads, images=len(files),
                              state=("not collected on this host" if outside else "measured" if len(files) == 20
                                     else "partial (contended)" if files else "not measured (contended)"),
                              retained_rounds=retained, excluded_rounds=excluded,
                              pixel_min=min(pairs[k]["width"] * pairs[k]["height"] for k in expected),
                              pixel_max=max(pairs[k]["width"] * pairs[k]["height"] for k in expected),
                              arms={name: dict(p50_ms=quantile(vals, .5), p95_ms=quantile(vals, .95)) if vals else None
                                    for name, vals in all_samples.items()},
                              paired_median_saving_ms_mean=statistics.mean(deltas) if deltas else None,
                              paired_median_saving_ms_ci95=([quantile(bootstrap, .025), quantile(bootstrap, .975)] if bootstrap else None)))
        fit_rows.extend(dict(threads=threads, arm=name, **fit(data)) if len(set(x for x,_ in data)) >= 2
                        else dict(threads=threads, arm=name, alpha_ms=None, beta_ms_per_MP=None,
                                  r_squared=None, images=len(data)) for name, data in points.items())
    a.out.write_text(json.dumps(dict(host=a.host, cells=cells, fits=fit_rows,
                                    missing_images=selection["missing"]), indent=2) + "\n")
    print(f"{a.host} / AVX2-v3. Milliseconds: p50 / p95 of retained individual calls.")
    if selection.get("provisional"):
        print("PROVISIONAL VALIDATION SNAPSHOT; the bounded run is still active.")
    print("| Size | Threads | Full 492 | Reduced 336 | Complete A scorer | Images | Calls/arm |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for row in cells:
        vals = [f'{row["arms"][n]["p50_ms"]:.4f} / {row["arms"][n]["p95_ms"]:.4f}' if row["arms"][n] else row["state"]
                for n in ["extract492", "extract336", "compute_A"]]
        print(f'| {row["size"]} | {row["threads"]} | ' + " | ".join(vals)
              + f' | {row["images"]}/20 | {row["retained_rounds"]} |')
    print("\nOLS on admitted image medians; x = measured MP, y = ms. Missing images remain explicit.")
    print("| Threads | Arm | α ms | β ms/MP | R² | Images |")
    print("|---:|---|---:|---:|---:|---:|")
    for row in fit_rows:
        if row["alpha_ms"] is None:
            print(f'| {row["threads"]} | {row["arm"]} | not measured (contended) | not measured (contended) | — | {row["images"]}/80 |')
            continue
        print(f'| {row["threads"]} | {row["arm"]} | {row["alpha_ms"]:.5f} | '
              f'{row["beta_ms_per_MP"]:.5f} | {row["r_squared"]:.5f} | {row["images"]}/80 |')
    print("\nFull minus reduced: mean of admitted within-image median differences; image-cluster bootstrap 95% CI.")
    for row in cells:
        if row["paired_median_saving_ms_ci95"] is None:
            print(f'{row["size"]}/{row["threads"]}t: {row["state"]}; CI unavailable')
            continue
        lo, hi = row["paired_median_saving_ms_ci95"]
        print(f'{row["size"]}/{row["threads"]}t: {row["paired_median_saving_ms_mean"]:.6f} ms [{lo:.6f}, {hi:.6f}]')
    outside_count = sum(r["state"] == "not collected on this host" for r in selection["missing"])
    print(f'\nNot measured (contended) on this host: {len(selection["missing"]) - outside_count}/160; '
          f'not collected on this host: {outside_count}/160')


if __name__ == "__main__":
    main()
