#!/usr/bin/env python3
"""Independently verify retained source rounds, percentiles and least-squares fits.

Run on the measurement host under the same heavy-work wrapper after collection.
NumPy is an independent numerical cross-check, never an input pixel producer.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("view", type=Path)
    parser.add_argument("summary", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--raw-root", type=Path, help="relocated archive root for original remote paths")
    args = parser.parse_args()
    selection = json.loads((args.view / "selection.json").read_text())
    summary = json.loads(args.summary.read_text())
    pairs = {r["id"]: r for r in json.loads((args.view / "pairs/pairs.json").read_text())}
    assert len(selection["chosen"]) + len(selection["missing"]) == 160
    keys = [(r["cell"], r["case"]) for r in selection["chosen"] + selection["missing"]]
    assert len(set(keys)) == 160
    points = defaultdict(list)
    values = defaultdict(list)
    source_cache = {}
    request_folders = set()
    for selected in selection["chosen"]:
        cell, case = selected["cell"], selected["case"]
        derived = json.loads((args.view / f"timing/{cell}/{case}.json").read_text())
        group = derived["comparisons"][0]
        assert len(group["samples"]) == len(selected["selected"]) == 16
        for sample, origin in zip(group["samples"], selected["selected"]):
            source = Path(origin["source"])
            if args.raw_root:
                source = args.raw_root / source.relative_to("/home/lilith/tmp/zensim-cost")
            if str(source) not in source_cache:
                raw = source.read_bytes()
                source_cache[str(source)] = (hashlib.sha256(raw).hexdigest(), json.loads(raw))
                request_folders.add(source.parent)
            digest, raw = source_cache[str(source)]
            assert digest == origin["sha256"]
            assert not raw["unreliable"] and raw["timestamp"] >= selection["quiet_epoch"]
            assert raw["comparisons"][0]["samples"][origin["source_round"]] == sample
            assert sample["gate_clean"] and sample["iterations"] == 1
            assert sorted(sample["execution_order"]) == [0, 1, 2]
        array = np.array([r["elapsed_ns"] for r in group["samples"]], dtype=np.float64) / 1e6
        names = [r["name"] for r in group["benchmarks"]]
        size, thread_text = cell.split("-")
        threads = int(thread_text[:-1])
        pixels_mp = pairs[case]["width"] * pairs[case]["height"] / 1e6
        for index, name in enumerate(names):
            values[(size, threads, name)].extend(array[:, index])
            points[(threads, name)].append((pixels_mp, np.median(array[:, index])))
    for folder in request_folders:
        request = json.loads((folder / "requests.json").read_text())
        full, reduced = set(request["full_ids"]), set(request["reduced_ids"])
        assert len(full) == 504 and len(reduced) == 348
        assert reduced <= full and full - reduced == set(request["omitted_damage_ids"])
        assert len(request["omitted_damage_ids"]) == 156 and request["layout_width"] == 720
        assert (folder / "full-manifest.json").read_bytes() == (folder / "reduced-manifest.json").read_bytes()
    for cell in summary["cells"]:
        for name, reported in cell["arms"].items():
            observed = values[(cell["size"], cell["threads"], name)]
            if reported is None:
                assert not observed
                continue
            independent = np.percentile(observed, [50, 95], method="linear")
            np.testing.assert_allclose(independent, [reported["p50_ms"], reported["p95_ms"]], rtol=1e-12)
    for fit in summary["fits"]:
        rows = points[(fit["threads"], fit["arm"])]
        assert len(rows) == fit["images"]
        if fit["alpha_ms"] is None:
            assert len({x for x, _ in rows}) < 2
            continue
        xy = np.array(rows)
        design = np.column_stack([np.ones(len(rows)), xy[:, 0]])
        solution, _, rank, _ = np.linalg.lstsq(design, xy[:, 1], rcond=None)
        assert rank == 2
        np.testing.assert_allclose(solution, [fit["alpha_ms"], fit["beta_ms_per_MP"]], rtol=1e-10, atol=1e-10)
    receipt = dict(numpy=np.__version__, provisional=selection["provisional"],
                   images_verified=len(selection["chosen"]), source_files_verified=len(source_cache),
                   paired_rounds_verified=16 * len(selection["chosen"]),
                   percentile_cells_verified=len(summary["cells"]), fit_rows_checked=len(summary["fits"]),
                   numeric_fits_verified=sum(row["alpha_ms"] is not None for row in summary["fits"]),
                   identical_producer_manifests_verified=len(request_folders),
                   status="passed")
    args.out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
