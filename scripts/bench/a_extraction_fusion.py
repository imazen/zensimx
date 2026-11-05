#!/usr/bin/env python3
"""Report the extraction fusion study's admitted native observations."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import datetime
import os
import subprocess
import struct

from a_family_cost_perf import summarize
from a_fusion_artifacts import parsed_path


def percentile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lo = int(position)
    return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (position - lo)


def phase1(root):
    result = {}
    for size in ("256", "1024", "large"):
        record, stat = (root / "phase1" / f"{size}-{mode}" for mode in ("record", "stat"))
        for directory in (record, stat):
            assert json.loads((directory / "admission.json").read_text())["status"] == "accepted"
        counters = {}
        with (stat / "counters.csv").open() as stream:
            for row in csv.reader(line for line in stream if not line.startswith("#")):
                if not row or not row[2].startswith("cpu_core/"):
                    continue
                assert not row[0].startswith("<"), row
                assert float(row[4]) >= 99, "multiplexed counter cannot qualify"
                counters[row[2].split("/")[1]] = int(row[0]) / 200
        result[size] = dict(attribution=summarize(record / "samples.txt"),
                            counters_per_image=counters,
                            instructions_per_cycle=counters["instructions"] / counters["cycles"],
                            L1d_miss_fraction=counters["L1-dcache-load-misses"] / counters["L1-dcache-loads"],
                            LLC_miss_fraction=counters["LLC-load-misses"] / counters["LLC-loads"])
    return dict(host="i270", cpu="Core Ultra 7 270K Plus", tier="v3", cpu_affinity="2 (P core)",
                images_per_size=20, repeats=10, sizes=result)


def timing(root):
    admission = json.loads((root / "admission.json").read_text())
    assert admission["status"] == "accepted", "whole-cell contention cannot qualify timing"
    rows = []
    path = root / "inner.json"
    doc = json.loads(path.read_text())
    workers = root / "workers.json"
    ready = json.loads(workers.read_text())
    for index, image in enumerate(ready["before"]["records"]):
        assert doc["accepted"] and not doc["unreliable"]
        selected = doc["selected"]
        assert len(selected) == 32
        assert selected == [i for i, flag in enumerate(doc["gate_clean"]) if flag is True][:32]
        samples = {arm: [doc[f"{arm}_ns"][i][index] / 1e6 for i in selected] for arm in ("before", "after")}
        rows.append(dict(image=image["pair"]["id"],
                         **{arm: dict(p50_ms=statistics.median(values), p95_ms=percentile(values, .95))
                            for arm, values in samples.items()}))
    assert len(rows) == 20, "twenty complete image pairs required"
    return dict(images=rows, workers_sha256=hashlib.sha256(workers.read_bytes()).hexdigest(),
                aggregate="median of twenty per-image percentiles",
                **{arm: {key: statistics.median(row[arm][key] for row in rows)
                         for key in ("p50_ms", "p95_ms")} for arm in ("before", "after")})


def progress(root):
    processes = []
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode()
            name = (proc / "comm").read_text().strip()
            executable = command.split(" ", 1)[0]
            if str(root) not in command or not (
                    name in ("cargo", "rustc", "a_family_cost", "perf", "flock", "sde64")
                    or executable.startswith(str(root) + "/")):
                continue
            stat = (proc / "stat").read_text().rsplit(")", 1)[1].split()
            processes.append(dict(pid=int(proc.name), name=name, command=command,
                                  state=stat[0], ppid=int(stat[1]), start_ticks=int(stat[19])))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue  # A process exiting during a status census is not a fixture skip.
    lock = str(Path.home() / "tmp/heavy.lock")
    locks = json.loads(subprocess.check_output(["lslocks", "-J", "-o", "PID,COMMAND,MODE,PATH,BLOCKER"], text=True))["locks"]
    captures = [p for p in (root / "phase1").glob("*/admission.json")
                if json.loads(p.read_text())["status"] == "accepted"]
    cell_files = list(root.glob("**/inner.json"))
    cells = {str(p.parent.relative_to(root)): json.loads(p.read_text())["accepted"] for p in cell_files}
    parsed_cells = {str(p.parent.relative_to(root)): str(parsed_path(p.parent, "summary.json"))
                    for p in cell_files if parsed_path(p.parent, "summary.json").is_file()}
    identity = {str(p.relative_to(root)): sum(1 for _ in p.open())
                for p in root.glob("*/identity.jsonl")}
    stages = {str(p.relative_to(root)): p.read_text().splitlines() for p in root.glob("*/stages.tsv")}
    failure_invariance = {str(p.relative_to(root)): json.loads(p.read_text())["equal"]
                          for p in root.glob("*/stable-diff*/receipt.json")}
    return dict(last_check=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                stage=(root / "PROGRESS.txt").read_text().splitlines()[-1],
                source_parent=(root / "BASE_COMMIT.txt").read_text().strip(),
                current_export_identity=(root / "SOURCE_COMMIT.txt").read_text().strip(),
                orchestration_receipts={str(p.relative_to(root)):p.read_text().splitlines()
                                        for p in root.glob("*RECEIPTS.tsv")},
                processes=processes, heavy_lock=[row for row in locks if row.get("path") == lock],
                phase1_captures_completed=len(captures), phase1_captures_required=9,
                completed_image_timing_cells=cells, parsed_image_timing_cells=parsed_cells,
                identity_lines=identity, stage_outcomes=stages,
                complete_failure_list_invariance=failure_invariance,
                candidate_grid="20 images x 3 sizes x 2 thread counts x 32 clean paired rounds",
                next_action="inspect active job and its saved log; finish exact candidates, then three isolated era experiments")



def test_coverage(root):
    """Keep every executed baseline test when avoiding non-test example links."""
    import re
    def cases(paths):
        found={};target=None
        for path in paths:
            for line in path.read_text().splitlines():
                text=line.strip()
                if text.startswith("Running "):
                    target=text[8:].split(" (",1)[0]
                elif text.startswith("Doc-tests "):
                    target=text
                match=re.fullmatch(r"test (.+) \.\.\. (ok|ignored|FAILED)(?:,.*)?",text)
                if match:
                    name=match[1]
                    key=(target,name)
                    assert key not in found,(path,key)
                    found[key]=match[2]
        assert found
        return found
    baseline=root.parent/"baseline-registry-fix.log"
    files=[root/"logs/test.log",root/"logs/doc.log"]
    old,new=cases([baseline]),cases(files)
    missing=[key for key,status in old.items() if new.get(key)!=status]
    assert not missing,missing[:16]
    return dict(baseline_test_cases=len(old),candidate_test_cases=len(new),
                every_baseline_case_and_status_preserved=True,
                added_cases=len(new.keys()-old.keys()),
                files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [baseline]+files})


def pixel_bits(root):
    expected = root / "expected"
    actual = root / "actual"
    old = json.loads((expected / "steer.json").read_text())["rows"]
    new = json.loads((actual / "steer.json").read_text())["rows"]
    assert len(old) == len(new) == 135
    for a, b in zip(old, new):
        assert a["key"] == b["key"]
        for key in ("served_base_bits", "served_repair_bits"):
            assert a[key] == b[key], (a["key"], key, "score bits differ")

    def identity(path):
        records = [json.loads(line) for line in path.read_text().splitlines()]
        values = {(row["permutation"], row["reference"]): row["candidate_score_bits"][0]
                  for row in records if not row.get("summary")}
        assert len(values) == 744, "all 62 sources and all 12 executable/emulated permutations required"
        return values

    assert identity(expected / "identity.jsonl") == identity(actual / "identity.jsonl")

    def nearid(path):
        doc = json.loads(path.read_text())
        assert doc["rows"] == 648
        scores = {(ref["reference"], ladder, rung): struct.pack("<d", score)
                  for ref in doc["references"] for ladder, values in ref["ladders"].items()
                  for rung, score in zip(values["rungs"], values["scores"])}
        assert len(scores) == 768
        return scores

    assert nearid(expected / "nearid.json") == nearid(actual / "nearid.json")
    return dict(steer_cases=135, steer_base_and_repair_bits_equal=True,
                identity_score_bits=744, nearid_rows=648, nearid_ladder_score_bits=768,
                all_equal=True,
                files={str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                       for directory in (expected, actual) for path in sorted(directory.iterdir())})


def drift(root):
    """Compare exact served f64 scores, without rounding through printed decimals."""
    def decode(bits):
        return struct.unpack("<d", struct.pack("<Q", bits))[0]

    def scores(directory):
        result = {}
        steer = json.loads((directory / "steer.json").read_text())["rows"]
        assert len(steer) == 135
        result["steer_base"] = {i: decode(row["served_base_bits"]) for i, row in enumerate(steer)}
        result["steer_repairs"] = {(case, i): decode(bits) for case, row in enumerate(steer)
                                   for i, bits in enumerate(row["served_repair_bits"])}
        identity = [json.loads(line) for line in (directory / "identity.jsonl").read_text().splitlines()]
        result["identity"] = {(row["permutation"], row["reference"]): decode(row["candidate_score_bits"][0])
                              for row in identity if not row.get("summary")}
        assert len(result["identity"]) == 744
        near = json.loads((directory / "nearid.json").read_text())
        assert near["rows"] == 648
        result["nearid"] = {(ref["reference"], name, rung): score for ref in near["references"]
                            for name, ladder in ref["ladders"].items()
                            for rung, score in zip(ladder["rungs"], ladder["scores"])}
        assert len(result["nearid"]) == 768
        rows = [json.loads(line) for line in (directory / "eval.jsonl").read_text().splitlines()]
        result["eval"] = {row["pair_key"]: decode(row["score_bits"]) for row in rows}
        assert len(result["eval"]) == 4678
        return result

    old, new = scores(root / "expected"), scores(root / "actual")
    result = {}
    for family, before in old.items():
        after = new[family]
        assert before.keys() == after.keys(), family
        differences = [abs(before[key] - after[key]) for key in before]
        assert all(__import__("math").isfinite(value) for value in differences)
        result[family] = dict(scores=len(differences), changed=sum(value != 0 for value in differences),
                              max_absolute=max(differences), mean_absolute=statistics.fmean(differences))
    return dict(families=result, score_units="served 0–100", files={
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for name in ("expected", "actual") for path in sorted((root / name).iterdir())})


def pixel_features(root):
    expected,actual = root/"expected",root/"actual"
    identity_receipt=identity_bits(root)
    old,new=[json.loads((directory/"steer.json").read_text())["rows"] for directory in (expected,actual)]
    assert len(old)==len(new)==135
    repairs=0
    for a,b in zip(old,new):
        for key in ("key","served_base_bits","served_repair_bits","served_base_features_sha256","served_repair_features_sha256"):
            assert a[key]==b[key],(a["key"],key)
        repairs+=len(a["served_repair_features_sha256"])
    def nearid(path):
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        result={(row["reference"],row["ladder"],row["rung"]):
                (row["reference_pixels_sha256"],row["dist_pixels_sha256"],row["consumed_feature_bits"],row["served_bits"],
                 row["full_feature_count"],row["full_feature_bits_sha256"])
                for row in rows}
        assert len(result)==648
        return result
    assert nearid(expected/"nearid.jsonl")==nearid(actual/"nearid.jsonl")
    return dict(identity_sources=62,identity_cells=identity_receipt["cells"],identity_features_and_scores_equal=True,
                steer_cases=135,steer_repairs=repairs,steer_full_feature_bits_equal=True,
                nearid_rows=648,nearid_features_and_scores_equal=True,
                files={str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest()
                       for d in (expected,actual) for path in d.iterdir()})


def identity_bits(root):
    files=[root/name/"identity.jsonl" for name in ("expected","actual")]
    def load(path):
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        rows=[row for row in rows if not row.get("summary")]
        result={(row["permutation"],row["reference"]):
                (row["pixels_sha256"],row["consumed_ids"],row["consumed_feature_bits"],row["candidate_score_bits"],
                 row["full_feature_count"],row["full_feature_bits_sha256"])
                for row in rows}
        assert len(rows)==len(result)==744
        assert len({key[0] for key in result})==12 and len({key[1] for key in result})==62
        assert all(len(row[1])==len(row[2])==504 and row[4]>=504 and len(row[5])==64 for row in result.values())
        return result
    old,new=[load(path) for path in files]
    assert old==new, "changed identity input/feature/score bits at the same permutation"
    return dict(cells=744,permutations=12,sources=62,features_and_scores_equal=True,
                files={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})


def eval_bits(root):
    files = [root / name / "eval.jsonl" for name in ("expected", "actual")]
    def load(path):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        keyed = {row["pair_key"]: row for row in rows}
        assert len(rows) == len(keyed) == 4678
        assert all(len(row["consumed_ids"]) == len(row["consumed_feature_bits"]) == 504
                   and row["full_feature_count"]>=504 and len(row["full_feature_bits_sha256"])==64 for row in rows)
        return keyed
    old,new = [load(path) for path in files]
    assert old.keys() == new.keys()
    differences = [key for key in old if old[key] != new[key]]
    assert not differences, f"changed EVAL feature/score/input bits: {differences[:8]}"
    return dict(rows=4678, consumed_slots_per_pair=504, features_and_scores_equal=True,
                identical_frozen_pixels=True,
                files={str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in files})


def eval_baseline(root):
    import pyarrow.parquet as pq

    frozen = root / "fixtures/eval-frozen-pred-mixF3-s0.tsv"
    pixels = root / "before3/eval.jsonl"
    with frozen.open() as stream:
        expected = list(csv.DictReader(stream, delimiter="\t"))
    actual = [json.loads(line) for line in pixels.read_text().splitlines()]
    assert len(expected) == len(actual) == 4678
    feature_keys = pq.read_table(root / "fixtures/eval-frozen-features.parquet",
                                 columns=["ref_basename"]).column("ref_basename").to_pylist()
    assert len(feature_keys) == len(set(feature_keys)) == 4678
    assert {row["pair_key"] for row in actual} == set(feature_keys)
    keyed = {}
    for index, (key, row) in enumerate(zip(feature_keys, expected)):
        assert int(row["row_idx"]) == index
        keyed[key] = (index, row)
    different = []
    for index, b in enumerate(actual):
        feature_index, a = keyed[b["pair_key"]]
        bits = struct.unpack("<Q", struct.pack("<d", float(a["pred"])))[0]
        if bits != b["score_bits"]:
            different.append(dict(pixel_row=index, feature_row=feature_index, pair_key=b["pair_key"],
                                  feature_prediction_bits=bits, pixel_prediction_bits=b["score_bits"]))
    return dict(rows=4678, join="frozen ref_basename to pixel pair_key", equal=not different, differences=different,
                files={str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in (frozen, pixels, root / "fixtures/eval-frozen-features.parquet")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("phase1", "timing", "progress", "pixel-bits", "drift", "eval-baseline", "eval-bits", "pixel-features", "identity-bits", "test-coverage"))
    parser.add_argument("root", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    result = {"phase1": phase1, "timing": timing, "progress": progress,
              "pixel-bits": pixel_bits, "drift": drift, "eval-baseline": eval_baseline, "eval-bits": eval_bits, "pixel-features": pixel_features, "identity-bits": identity_bits, "test-coverage": test_coverage}[args.stage](args.root)
    if args.stage == "progress":
        temporary = args.out.with_suffix(".next")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        os.replace(temporary, args.out)
        with args.out.with_suffix(".jsonl").open("a") as stream:
            stream.write(json.dumps(result) + "\n")
        print(json.dumps(result))
        return
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    if args.stage == "phase1":
        for size, row in result["sizes"].items():
            print(size, json.dumps({k: v for k, v in row.items() if k != "attribution"}))
            print("buckets", {k: round(v["self_percent_of_extraction"], 3)
                              for k, v in row["attribution"]["buckets"].items()})
    elif args.stage == "timing":
        print(json.dumps({arm: result[arm] for arm in ("before", "after")}))
    elif args.stage == "eval-baseline":
        print(json.dumps(dict(rows=result["rows"], equal=result["equal"], differences=len(result["differences"]))))
        assert result["equal"], "pixel replay differs from the frozen feature prediction; see receipt"
    else:
        print(json.dumps({key: value for key, value in result.items() if key != "files"}))


if __name__ == "__main__":
    main()
