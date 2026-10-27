#!/usr/bin/env python3
"""Summarize serve_custom_bake --prodqual evidence, without reading labels."""
import argparse
import json
from pathlib import Path


def summarize(native, wasm):
    assert native["schema"] == 2, "cache-feature audit requires schema 2"
    if wasm is not None:
        assert wasm["schema"] == 2
    results = []
    for seed in native["seeds"]:
        permutations = seed["permutations"]
        baseline = permutations[0]["rows"]
        rows = [row for permutation in permutations for row in permutation["rows"]]
        identity = [row for row in baseline if row["dropped_bits"] == 0]
        ladders = []
        for width, height in native["geometries"]:
            ladder = [r for r in baseline if (r["width"], r["height"]) == (width, height)]
            ladders.append({"width": width, "height": height,
                            "scores": [r["pixel_score"] for r in ladder],
                            "nonincreasing": all(a["pixel_score"] >= b["pixel_score"]
                                                 for a, b in zip(ladder, ladder[1:]))})
        keys = ("pixel_bits", "cached_bits", "feature_bits", "identity_aware_bits", "read_bits",
                "feature_mismatches", "cached_feature_mismatches", "finite", "density_cells")
        tier_differences = sum(any(row[k] != base[k] for k in keys)
                               for permutation in permutations[1:]
                               for row, base in zip(permutation["rows"], baseline, strict=True))
        wasm_differences = None
        wasm_rows = []
        if wasm is not None:
            other = wasm["seeds"][seed["seed"]]
            assert other["declared_reads"] == seed["declared_reads"]
            wasm_rows = [r for p in other["permutations"] for r in p["rows"]]
            wasm_differences = sum(any(row[k] != base[k] for k in keys)
                                   for permutation in other["permutations"]
                                   for row, base in zip(permutation["rows"], baseline, strict=True))
        results.append({
            "seed": seed["seed"], "caller_width": seed["caller_width"],
            "declared_reads": seed["declared_reads"], "pairs_per_permutation": len(baseline),
            "permutations": [p["label"] for p in permutations],
            "finite": all(r["finite"] for r in rows),
            "pixel_cache_mismatches": sum(r["pixel_bits"] != r["cached_bits"] for r in rows),
            "pixel_identity_aware_feature_mismatches": sum(r["pixel_bits"] != r["identity_aware_bits"] for r in rows),
            "consumed_feature_mismatches": sum(len(r["feature_mismatches"]) for r in rows),
            "cached_feature_mismatches": sum(len(r["cached_feature_mismatches"]) for r in rows),
            "native_tier_row_mismatches": tier_differences,
            "wasm_row_mismatches": wasm_differences,
            "wasm_finite": all(r["finite"] for r in wasm_rows) if wasm is not None else None,
            "wasm_consumed_feature_mismatches": sum(len(r["feature_mismatches"]) for r in wasm_rows) if wasm is not None else None,
            "wasm_cached_feature_mismatches": sum(len(r["cached_feature_mismatches"]) for r in wasm_rows) if wasm is not None else None,
            "pixel_identity_exact_100": all(r["pixel_score"] == 100.0 for r in identity),
            "feature_identity_scores": [r["feature_score"] for r in identity],
            "feature_identity_band_97_5_to_100": all(97.5 <= r["feature_score"] <= 100.0 for r in identity),
            "no_distortion_above_identity": all(r["pixel_score"] <= 100.0 for r in baseline),
            "ladders_report_only": ladders,
        })
    scope = "three independent packed seeds; label-free synthetic SDR only"
    if native.get("formula_revision") == 7:
        scope = f"{len(results)} explicitly named Rev7 models; label-free synthetic SDR only"
    return {"schema": 2, "scope": scope,
            "geometries": native["geometries"], "dropped_bits": native["dropped_bits"], "seeds": results}


def prepare_steerfix(root):
    """Freeze the existing engineering roster, never a corpus/label reader."""
    import hashlib
    prior = Path("/mnt/v/output/zensim/prodqual-b-2026-10-07")
    adjudicate = Path("/mnt/v/output/zensim/adjudicate-2026-10-07")
    sha = lambda data: hashlib.sha256(data).hexdigest()
    source = prior / "STEERING_PREREAD.json"
    roster = json.loads(source.read_text())
    assert len(roster["cases"]) == 135 and roster["rules"] == {"m2": .99, "m3f": .7}
    failures = json.loads((adjudicate / "STEERING.json").read_text())
    keys = {r["case"] for r in failures if r["model"] == "seed0"}
    assert len(keys) == 7
    dest = root / "inputs"
    dest.mkdir()
    models = {}
    pins = json.loads((adjudicate / "FINAL_PROVENANCE.json").read_text())["models"]
    for name in ("seed0.bin", "dense.bin"):
        blob = (adjudicate / "models" / name).read_bytes()
        assert sha(blob) == pins[name]
        (dest / name).write_bytes(blob)
        models[name] = (str(dest / name), pins[name])
    copied = {}
    base = []
    for row in roster["cases"]:
        paths = []
        for field, pin in (("ref_path", "reference_file_sha256"),
                           ("dist_path", "distorted_file_sha256")):
            path = Path(row[field])
            assert path.parent.parent == Path("/var/tmp/shippath7/delivery")
            digest = row[pin]
            target = dest / (digest + ".png")
            if digest not in copied:
                blob = path.read_bytes()
                assert sha(blob) == digest
                target.write_bytes(blob)
                copied[digest] = str(target)
            paths.append(str(target))
        base.append({"key": row["case"], "reference": paths[0], "distorted": paths[1],
                     "reference_sha256": row["reference_file_sha256"],
                     "distorted_sha256": row["distorted_file_sha256"],
                     "reference_pixels_sha256": row["reference_pixels_sha256"],
                     "distorted_pixels_sha256": row["distorted_pixels_sha256"],
                     "block": row["block"]})
    selected = [r for r in base if r["key"] in keys]
    assert len(selected) == 7
    diagnostic = []
    for row in selected:
        for objective in ("served", "pre-floor", "smooth-floor"):
            model, pin = models["seed0.bin"]
            diagnostic.append(dict(row, model=model, model_sha256=pin, objective=objective))
        model, pin = models["dense.bin"]
        diagnostic.append(dict(row, model=model, model_sha256=pin, objective="served"))
    model, pin = models["seed0.bin"]
    full = [dict(r, model=model, model_sha256=pin, objective="served") for r in base]
    for name, rows, engine in (("diagnostic-before", diagnostic, False),
                               ("engine", diagnostic, True),
                               ("full-before", full, False), ("full", full, True)):
        packet = {"cases": rows, "engine": engine, "floor_recovery": name == "full",
                  "output": str(root / (name + ".json"))}
        (root / (name + "-packet.json")).write_text(json.dumps(packet, indent=2) + "\n")
    receipt = {"engineering_roster_sha256": sha(source.read_bytes()), "models": models,
               "input_file_hashes": copied, "cases": roster["cases"], "labels_read": False,
               "case_exclusions": [], "bars": roster["rules"]}
    (root / "INPUT_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"STEERFIX: admitted 135 cases, seven failures, {len(copied)} pinned files")


def report_steerfix(root):
    """Join native engineering receipts; never score pixels or read labels."""
    import csv
    import hashlib
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    read = lambda name: json.loads((root / name).read_text())
    admitted = read("INPUT_RECEIPT.json")
    diagnostic = read("diagnostic-before.json")["rows"]
    replay = read("engine.json")["rows"]
    before = read("full-before.json")["rows"]
    after = read("full.json")["rows"]
    # Four mild-JPEG names occur in both owner/b32 and jpeg8/b8 panels.
    # Retain both; the frozen admission key includes the block size.
    roster = [(r["case"], r["block"]) for r in admitted["cases"]]
    assert len(roster) == 135 and len(set(roster)) == 135
    row_key = lambda r: (r["key"], r["blocks"][0]["bounds"][2])
    assert [row_key(r) for r in before] == roster == [row_key(r) for r in after]
    assert len(diagnostic) == len(replay) == 28
    assert admitted["bars"] == {"m2": .99, "m3f": .7}
    packed = admitted["models"]["seed0.bin"][1]
    dense = admitted["models"]["dense.bin"][1]
    index = lambda rows: {(r["key"], r["model"], r["objective"]): r for r in rows}
    original, exact = index(diagnostic), index(replay)
    assert original.keys() == exact.keys()
    score_checks = 0
    for a, b in zip(before, after, strict=True):
        assert a["model"] == b["model"] == packed
        assert a["served_base_bits"] == b["served_base_bits"]
        assert a["served_repair_bits"] == b["served_repair_bits"]
        score_checks += 1 + len(a["served_repair_bits"])
    for key in original:
        assert original[key]["served_base_bits"] == exact[key]["served_base_bits"]
        assert original[key]["served_repair_bits"] == exact[key]["served_repair_bits"]
    assert all(r["sensitivity_independent_probe_exact"] for r in before + after + diagnostic + replay)
    disagreements = sum(r["engine_feature_disagreements"] for r in replay)
    comparisons = sum(r["engine_feature_comparisons"] for r in replay)
    full_disagreements = sum(r["engine_feature_disagreements"] for r in after)
    assert disagreements == full_disagreements == 0
    assert all(r["engine_feature_comparisons"] > 0 for r in replay + after)
    verdicts = []
    for key in dict.fromkeys(r["key"] for r in diagnostic):
        served = original[key, packed, "Served"]
        raw = original[key, packed, "PreFloor"]
        smooth = original[key, packed, "SmoothFloor"]
        neighbour = exact[key, packed, "PreFloor"]
        dense_old, dense_new = original[key, dense, "Served"], exact[key, dense, "Served"]
        fixed = next(r for r in after if r["key"] == key)
        floor = (served["base_score"] != raw["base_score"]
                 and all(b["score_delta"] == 0.0 for b in served["blocks"])
                 and any(b["score_delta"] != 0.0 for b in raw["blocks"]))
        frozen_map = raw["m3f"] < .7 <= neighbour["m3f"]
        code = floor or frozen_map
        model = not neighbour["pass"]
        verdict = "BOTH" if code and model else "CODE" if code else "MODEL"
        if floor:
            assert fixed["sensitivity_objective"] == "PreFloor"
            assert [b["refinement_gain"] for b in fixed["blocks"]] == [b["refinement_gain"] for b in neighbour["blocks"]]
            assert [b["linearized_gain"] for b in fixed["blocks"]] == [b["linearized_gain"] for b in neighbour["blocks"]]
        verdicts.append({"case": key, "verdict": verdict, "floor_signal_loss": floor,
                         "frozen_map_loss": frozen_map,
                         "served": [served["m2"], served["m3f"]],
                         "pre_floor": [raw["m2"], raw["m3f"]],
                         "smooth_floor": [smooth["m2"], smooth["m3f"]],
                         "pre_floor_replay": [neighbour["m2"], neighbour["m3f"]],
                         "dense_before": [dense_old["m2"], dense_old["m3f"]],
                         "dense_replay": [dense_new["m2"], dense_new["m3f"]],
                         "fixed_served": [fixed["m2"], fixed["m3f"]],
                         "feature_comparisons": neighbour["engine_feature_comparisons"],
                         "feature_disagreements": neighbour["engine_feature_disagreements"]})
    before_failures = [r["key"] for r in before if not r["pass"]]
    after_failures = [r["key"] for r in after if not r["pass"]]
    assert before_failures == after_failures, "new G-STEER failure requires investigation"
    artifacts = ["INPUT_RECEIPT.json", "diagnostic-before.json", "engine.json", "full-before.json", "full.json",
                 "provenance/instrument-before.json", "provenance/instrument-after.json"]
    summary = {"base": "462f7fe5", "labels_read": False, "case_exclusions": [],
               "bars": admitted["bars"], "verdicts": verdicts,
               "score_bit_checks": score_checks, "score_bit_mismatches": 0,
               "seven_case_feature_comparisons": comparisons, "seven_case_feature_disagreements": disagreements,
               "full_feature_comparisons": sum(r["engine_feature_comparisons"] for r in after),
               "full_feature_disagreements": full_disagreements,
               "g_steer_before_pass": len(before) - len(before_failures),
               "g_steer_after_pass": len(after) - len(after_failures),
               "remaining_g_steer_failures": after_failures, "model_qualified": False,
               "raw_sha256": {name: sha(root / name) for name in artifacts}}
    with (root / "SUMMARY.json").open("x") as out:
        json.dump(summary, out, indent=2)
        out.write("\n")
    with Path("benchmarks/steerfix_cases_2026-10-09.csv").open("x") as out:
        writer = csv.writer(out)
        writer.writerow(["case", "block", "before_m2", "before_m3f", "after_m2", "after_m3f", "pass", "sensitivity_objective",
                         "engine_feature_comparisons", "engine_feature_disagreements"])
        for a, b in zip(before, after, strict=True):
            writer.writerow([*row_key(a), a["m2"], a["m3f"], b["m2"], b["m3f"], b["pass"], b["sensitivity_objective"],
                             b["engine_feature_comparisons"], b["engine_feature_disagreements"]])
    with Path("benchmarks/steerfix_cases_2026-10-09.csv.meta").open("x") as out:
        json.dump({"runtime_commit": read("provenance/instrument-after.json")["runtime_commit"],
                   "base": "462f7fe5", "host": "dev", "formula_revision": 5,
                   "grid": "all 135 original engineering cases; original block sizes; bin 1; serial scorer",
                   "bars": admitted["bars"], "raw_root": str(root),
                   "command": "just --justfile benchmarks/steerfix.just steerfix-packet <frozen-binary> <full-packet> <0-before-or-1-after>",
                   "roster_sha256": admitted["engineering_roster_sha256"],
                   "models": admitted["models"], "summary_sha256": sha(root / "SUMMARY.json")}, out, indent=2)
        out.write("\n")
    print(json.dumps(summary, indent=2))


def nearid_register(root):
    """Admit coordinator-selected TRAIN roles; project metadata, never labels/features."""
    import csv
    import hashlib
    import subprocess
    import pyarrow.parquet as pq

    def digest(path):
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    bank = Path("/var/tmp/rev4-featbank/bank/kadid_train")
    picker = Path("/mnt/v/output/canonical-picker-2026-06-27/zenjpeg_lossy")
    corpus = Path("/mnt/v/output/clean-picker-corpus-2026-06-26")
    folders = Path("/mnt/v/output/imazen-26-features/imazen26_split_evenodd.tsv")
    manifest = json.loads((bank / "_MANIFEST.json").read_text())
    assert manifest["role"] == "train"
    assert digest(bank / "keys.parquet") == manifest["files"]["keys.parquet"]["sha256"]
    columns = ["ref_group", "ref_path", "ref_pixels_sha256", "width", "height"]
    keys = pq.read_table(bank / "keys.parquet", columns=columns).to_pylist()
    assert len(keys) == manifest["unique_pair_keys"] == 4880
    refs = {}
    for row in keys:
        if row["ref_group"] in refs:
            assert refs[row["ref_group"]] == row
        refs[row["ref_group"]] = row
    assert len(refs) == 40
    photos = sorted(refs.values(), key=lambda r: hashlib.sha256(r["ref_group"].encode()).hexdigest())[:8]
    pm = json.loads((picker / "_MANIFEST.json").read_text())
    assert digest(picker / "train.parquet") == pm["splits"]["train"]["sha256"]
    columns = ["split", "origin_id", "variant_name", "ref_filename", "source_r2_url",
               "content_image_sha", "content_class", "content_source", "width", "height"]
    rows = pq.read_table(picker / "train.parquet", columns=columns).to_pylist()
    assert len(rows) == pm["splits"]["train"]["rows"]
    assert all(r["split"] == "train" and r["origin_id"][-1] in "02468" for r in rows)
    origins = {r["origin_id"] for r in rows}
    with folders.open() as stream:
        fm = {r["stem"]: r for r in csv.DictReader(stream, delimiter="\t") if r["stem"] in origins}
    assert len(fm) == len(origins) == 212
    selected = [dict(r, content_class="photo", role="TRAIN", file_sha256=digest(Path(r["ref_path"])),
                     admission="KADID D1 TRAIN bank (40 references)") for r in photos]
    rules = {"screen": "8100-lilith-web-screenshots", "line_art": "7000-lilith-plots"}
    for category, folder in rules.items():
        eligible = {r["origin_id"] for r in rows if fm[r["origin_id"]]["content_class"] == folder
                    and fm[r["origin_id"]]["split"] == fm[r["origin_id"]]["manifest_split"] == "train"}
        chosen = sorted(eligible, key=lambda origin: hashlib.sha256(origin.encode()).hexdigest())[:8]
        assert len(chosen) == 8
        for origin in chosen:
            variants = {r["ref_filename"]: r for r in rows if r["origin_id"] == origin
                        and max(r["width"], r["height"]) == 512}
            assert variants, origin
            row = variants[sorted(variants)[0]]
            assert row["source_r2_url"] == "s3://codec-corpus/clean-picker-corpus-2026-06-26/" + row["ref_filename"]
            expected = digest(corpus / row["ref_filename"])
            selected.append(dict(row, ref_group="picker:" + row["variant_name"],
                                 ref_path=str(root / "inputs" / row["ref_filename"]),
                                 content_class=category, recorded_content_class=row["content_class"],
                                 corpus_folder=folder, role="TRAIN", file_sha256=expected,
                                 hash_basis="pre-download SHA256 of existing canonical rendition mirror",
                                 admission="canonical-picker zenjpeg_lossy/train; even origin; both folder manifest roles TRAIN"))
    packet = {"schema": "nearid-preread-v2", "labels_read": False, "features_reused": False,
              "selection": "KADID first 8 by SHA256(group); picker first 8 distinct origins by SHA256(origin) in each stated folder; existing 512-long-edge rendition, no resizing; unknown table class uses folder rule",
              "folder_rules": rules, "sources": selected,
              "metadata_pins": {str(p): digest(p) for p in (bank / "keys.parquet", bank / "_MANIFEST.json",
                                                          picker / "train.parquet", picker / "_MANIFEST.json", folders)},
              "columns_opened": {"kadid": ["ref_group", "ref_path", "ref_pixels_sha256", "width", "height"], "picker": columns}}
    with (root / "PREREAD_SOURCE_POLICY.json").open("x") as stream:
        json.dump(packet, stream, indent=2)
        stream.write("\n")
    (root / "inputs").mkdir(exist_ok=True)
    for row in selected[8:]:
        destination = Path(row["ref_path"])
        assert not destination.exists()
        subprocess.run(["bash", "-lc", '. "$HOME/.config/cloudflare/r2-env.sh"; '
                        'export AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" AWS_REGION=auto; '
                        'exec s5cmd --endpoint-url "https://$R2_ACCOUNT_ID.r2.cloudflarestorage.com" cp "$1" "$2"',
                        "nearid-fetch", row["source_r2_url"], str(destination)], check=True)
        assert digest(destination) == row["file_sha256"], "fetched rendition hash"
    with (root / "PREREAD_FINAL.json").open("x") as stream:
        json.dump(packet, stream, indent=2)
        stream.write("\n")
    print("Frozen 24 TRAIN references; 16 recorded R2 URLs fetched and SHA256-verified; labels/features unread")


def nearid_reference_report(model, reference, values):
    """One model/reference NEARID panel: highest nonidentical score, one-pixel rungs and the six ladders."""
    assert len(values) == 27
    assert len({(v["ladder"], v["rung"]) for v in values}) == 27
    anchor = next(v for v in values if v["ladder"] == "identity")
    assert anchor["identical"] and anchor["served_score"] == 100
    changed = [v for v in values if not v["identical"]]
    assert changed
    highest = max(changed, key=lambda v: v["served_score"])
    ladders = {}
    for ladder in ("one_pixel_1", "one_pixel_-1", "fraction", "noise", "zenjpeg444", "gaussian"):
        sequence = [anchor] + [v for v in values if v["ladder"] == ladder]
        reversals = [{"from": a["rung"], "to": b["rung"],
                      "increase": b["served_score"] - a["served_score"]}
                     for a, b in zip(sequence, sequence[1:])
                     if b["served_score"] > a["served_score"]]
        crossings = {}
        for threshold in (99, 98, 95, 90):
            below = [v for v in sequence if v["served_score"] < threshold]
            transitions = [{"from": a["rung"], "to": b["rung"],
                            "direction": "below" if b["served_score"] < threshold else "above"}
                           for a, b in zip(sequence, sequence[1:])
                           if (a["served_score"] < threshold) != (b["served_score"] < threshold)]
            crossings[str(threshold)] = {
                "first_below_rung": below[0]["rung"] if below else None,
                "all_transitions": transitions}
        ladders[ladder] = {"rungs": [v["rung"] for v in sequence],
                           "scores": [v["served_score"] for v in sequence],
                           "changed_pixels": [v["changed_pixels"] for v in sequence],
                           "nonincreasing": not reversals, "reversals": reversals,
                           "crossings": crossings}
    return {"model": model, "reference": reference, "class": anchor["class"],
            "highest_nonidentical_score": highest["served_score"],
            "gap_to_100": 100 - highest["served_score"],
            "highest_rung": {k: highest[k] for k in ("ladder", "rung", "changed_pixels")},
            "one_pixel_scores": [v["served_score"] for v in values
                                 if v["ladder"].startswith("one_pixel")],
            "ladders": ladders}


def nearid_summary(root):
    """Report observed discrete crossings and reversals; never interpolate."""
    import csv
    import math
    from collections import defaultdict

    rows = [json.loads(line) for model in ("seed0", "B", "A")
            for line in (root / f"{model}.jsonl").read_text().splitlines()]
    assert len(rows) == 3 * 24 * 27
    groups = defaultdict(list)
    for row in rows:
        assert math.isfinite(row["served_score"])
        groups[(row["model"], row["reference"])].append(row)
    assert len(groups) == 72
    reports = [nearid_reference_report(model, reference, values)
               for (model, reference), values in groups.items()]
    aggregates = {}
    for model in ("seed0", "B", "A"):
        vals = [r for r in reports if r["model"] == model]
        allrows = [r for r in rows if r["model"] == model]
        aggregates[model] = {
            "references": len(vals), "rows": len(allrows),
            "highest_nonidentical_score": max(r["highest_nonidentical_score"] for r in vals),
            "gap_to_100_range": [min(r["gap_to_100"] for r in vals), max(r["gap_to_100"] for r in vals)],
            "one_pixel_score_range": [min(s for r in vals for s in r["one_pixel_scores"]),
                                      max(s for r in vals for s in r["one_pixel_scores"])],
            "monotone_ladders": sum(v["nonincreasing"] for r in vals for v in r["ladders"].values()),
            "total_ladders": len(vals) * 6,
            "gaussian_identical_rungs": sum(r["identical"] for r in allrows if r["ladder"] == "gaussian"),
            "class_gaps": {c: [min(r["gap_to_100"] for r in vals if r["class"] == c),
                                max(r["gap_to_100"] for r in vals if r["class"] == c)]
                           for c in ("photo", "screen", "line_art")}}
    identity = [r for r in rows if r["model"] == "seed0" and r["ladder"] == "identity"]
    diagnostics = {k: [min(r[k] for r in identity), max(r[k] for r in identity)]
                   for k in ("raw_model_score", "precalibration_score", "neutral_fragility_score")}
    result = {"schema": "nearid-summary-v1", "rows": len(rows),
              "crossing_rule": "strictly below threshold; observed rungs only; all recrossings retained",
              "monotonicity_rule": "exact nonincrease of served f64 values, identity anchor included",
              "aggregate": aggregates, "seed0_identity_diagnostics": diagnostics,
              "references": reports}
    with (root / "SUMMARY.json").open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    columns = ["model", "reference", "class", "width", "height", "ladder", "rung",
               "changed_pixels", "identical", "served_score", "raw_model_score",
               "precalibration_score", "neutral_fragility_score", "reference_pixels_sha256",
               "dist_pixels_sha256", "encoded_sha256"]
    with (root / "scores.tsv").open("x") as f:
        writer = csv.DictWriter(f, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# NEARID measured near-identity scores", "",
             "24 TRAIN RGB8 references: 8 KADID photos, 8 canonical-picker screen/UI and 8 graphics.",
             "Separate processes: seed-0 Rev5, named B Rev1, named A Rev1. Labels unread.", ""]
    for model, aggregate in aggregates.items():
        lines += [f"{model}: highest nonidentical score {aggregate['highest_nonidentical_score']:.17g}; "
                  f"gap range {aggregate['gap_to_100_range']}; one-pixel range {aggregate['one_pixel_score_range']}; "
                  f"nonincreasing ladders {aggregate['monotone_ladders']}/{aggregate['total_ladders']}.", ""]
    lines += [f"Seed-0 raw identity diagnostic ranges: {diagnostics}.", "",
              "Raw model score means complete feature-only inference including the frozen output spline; "
              "precalibration_score additionally removes that spline. Counterfactual zeros only the ten "
              "reference-only PJND_FRAGILITY inputs and is diagnostic, never a serving candidate.", "",
              "Byte-identical RGB8 blur rungs remain identity anchors and are excluded from nonidentical maxima. "
              "Crossings are observed first-below rungs; SUMMARY.json retains every recrossing. "
              "Content classes and reference selection use pixels/metadata only, before any scoring. "
              "This finite discrete panel cannot establish mathematical continuity or human perceptual accuracy."]
    (root / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    nearid_plot(root, rows)
    print(json.dumps({"aggregate": aggregates, "seed0_identity_diagnostics": diagnostics}, indent=2))


def nearid_candidate(root, label):
    """E33 section 9.2 N1-N3 for one candidate's NEARID rows, with every reversal and crossing retained."""
    import math
    from collections import defaultdict

    rows = [json.loads(line) for line in (root / f"candidate-{label}.jsonl").read_text().splitlines()]
    assert len(rows) == 24 * 27
    groups = defaultdict(list)
    for row in rows:
        assert math.isfinite(row["served_score"])
        groups[row["reference"]].append(row)
    assert len(groups) == 24
    reports = [nearid_reference_report(f"candidate-{label}", reference, values)
               for reference, values in groups.items()]
    n1 = [r["reference"] for r in reports if min(r["one_pixel_scores"]) < 99.0]
    n2 = [r["reference"] for r in reports if r["highest_nonidentical_score"] < 99.0]
    monotone = sum(v["nonincreasing"] for r in reports for v in r["ladders"].values())
    return {"schema": "e33-nearid-gates-v1", "model": f"candidate-{label}", "rows": len(rows),
            "N1": {"rule": "both one-pixel rungs >= 99.0 on all 24 references", "pass": not n1, "failing": n1,
                   "one_pixel_score_range": [min(s for r in reports for s in r["one_pixel_scores"]),
                                             max(s for r in reports for s in r["one_pixel_scores"])]},
            "N2": {"rule": "highest nonidentical served score >= 99.0 on every reference", "pass": not n2,
                   "failing": n2, "highest_nonidentical_range": [min(r["highest_nonidentical_score"] for r in reports),
                                                                 max(r["highest_nonidentical_score"] for r in reports)]},
            "N3": {"rule": "nonincreasing ladders >= 122 of 144", "pass": monotone >= 122,
                   "monotone_ladders": monotone, "total_ladders": 6 * len(reports)},
            "references": reports}


def nearid_verify(root):
    """Check artifact replay and independently decode the retained JPEGs."""
    import hashlib
    from PIL import Image

    packet = json.loads((root / "PREREAD_FINAL.json").read_text())
    records = {model: [json.loads(line) for line in (root / f"{model}.jsonl").read_text().splitlines()]
               for model in ("seed0", "B", "A")}
    ids = json.loads((root / "seed0.metadata.json").read_text())["consumed_feature_ids"]
    assert len(ids) == 420
    assert all(row["consumed_feature_mismatches"] == 0
               and len(row["consumed_feature_bits"]) == 420 for row in records["seed0"])
    for other in ("B", "A"):
        assert len(records[other]) == len(records["seed0"]) == 648
        for a, b in zip(records["seed0"], records[other], strict=True):
            for key in ("reference", "class", "ladder", "rung", "changed_pixels", "identical",
                        "reference_pixels_sha256", "dist_pixels_sha256", "encoded_sha256"):
                assert a[key] == b[key], (other, key, a["reference"])
    for source in packet["sources"]:
        path = root / f"source_{source['file_sha256']}.png"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == source["file_sha256"]
    import re
    external_sources = {source["ref_path"] for source in packet["sources"]
                        if not source["ref_path"].startswith(str(root) + "/")}
    data_reads = set()
    for model in ("seed0", "B", "A"):
        for line in (root / f"{model}.openat.log").read_text().splitlines():
            match = re.search(r'openat\([^,]+, "([^"]+)".*O_RDONLY', line)
            if not match:
                continue
            path = match.group(1)
            assert "imazen-26" not in path and "_sealed" not in path
            if path.startswith(("/mnt/v/", "/var/tmp/")):
                assert path.startswith(str(root) + "/") or path in external_sources, path
                data_reads.add(path)
    assert external_sources <= data_reads
    pixel_shas = set()
    for row in records["seed0"]:
        digest = row["dist_pixels_sha256"]
        if digest not in pixel_shas:
            data = (root / f"{digest}.rgb").read_bytes()
            assert hashlib.sha256(data).hexdigest() == digest
            assert len(data) == row["width"] * row["height"] * 3
            pixel_shas.add(digest)
    images = []
    for row in records["seed0"]:
        if row["ladder"] != "zenjpeg444":
            continue
        name = hashlib.sha256(row["reference"].encode()).hexdigest()
        path = root / f"{name}_q{row['rung']}.jpg"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["encoded_sha256"]
        with Image.open(path) as image:
            assert image.format == "JPEG"
            assert image.size == (row["width"], row["height"])
            assert len(image.layer) == 3 and all(layer[1:3] == (1, 1) for layer in image.layer)
            image.load()
            decoded = image.convert("RGB").tobytes()
            assert len(decoded) == row["width"] * row["height"] * 3
            images.append({"sha256": row["encoded_sha256"], "reference": row["reference"],
                           "quality": row["rung"], "width": image.width, "height": image.height})
    assert len(images) == 168
    result = {"schema": "nearid-replay-verification-v1", "pass": True,
              "pairs_identical_across_processes": 648, "raw_pixel_artifacts": len(pixel_shas),
              "source_png_hashes": len(packet["sources"]), "process_data_reads": sorted(data_reads),
              "no_unadmitted_data_reads": True,
              "jpeg_oracle": "Pillow/libjpeg read only: decode, dimensions and 4:4:4 sampling",
              "jpeg_images": images}
    with (root / "REPLAY_VERIFIED.json").open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(f"Replay PASS: 648 identical pixel pairs across processes; {len(pixel_shas)} raw buffers; 168 independent JPEG decodes")


def nearid_plot(root, rows):
    """Standalone scientific plot; all measured points remain in exported TSV."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from collections import defaultdict

    colors = {"photo": "#0072B2", "screen": "#D55E00", "line_art": "#009E73"}
    ladders = ("one_pixel", "fraction", "noise", "zenjpeg444", "gaussian")
    fig, axes = plt.subplots(3, 5, figsize=(20, 10), sharey=True)
    for i, model in enumerate(("seed0", "B", "A")):
        for j, ladder in enumerate(ladders):
            ax = axes[i, j]
            grouped = defaultdict(list)
            for row in rows:
                if row["model"] == model and (row["ladder"] == ladder or
                                              (ladder == "one_pixel" and row["ladder"].startswith("one_pixel"))):
                    grouped[row["reference"]].append(row)
            for values in grouped.values():
                if ladder == "one_pixel":
                    for value in values:
                        sign = int(value["ladder"].rsplit("_", 1)[1])
                        ax.plot([0, sign], [100, value["served_score"]], ".-", alpha=.55,
                                color=colors[value["class"]], linewidth=.7)
                    continue
                ax.plot([float(v["rung"]) for v in values],
                        [v["served_score"] for v in values], ".-", alpha=.55,
                        color=colors[values[0]["class"]], linewidth=.7)
            if ladder == "one_pixel":
                ax.set_xticks([-1, 0, 1])
            if ladder == "fraction":
                ax.set_xscale("log")
            if ladder == "zenjpeg444":
                ax.invert_xaxis()
            for threshold in (99, 98, 95, 90):
                ax.axhline(threshold, color="#666666", linestyle=":", linewidth=.6)
            ax.set_xlabel(ladder)
            ax.set_title(model if j == 0 else "")
            ax.grid(alpha=.15)
            if j == 0:
                ax.set_ylabel("served score")
    handles = [plt.Line2D([], [], color=color, label=label) for label, color in colors.items()]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .965), ncol=3)
    fig.suptitle("Near-identity TRAIN panel: scores at observed distortion rungs", y=.995)
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(root / "curves.svg")
    fig.savefig(root / "curves.png", dpi=150)
    plt.close(fig)


def main():
    import sys
    if len(sys.argv) == 3 and sys.argv[1] == "--nearid-verify":
        nearid_verify(Path(sys.argv[2]))
        return
    if len(sys.argv) == 3 and sys.argv[1] == "--nearid-register":
        nearid_register(Path(sys.argv[2]))
        return
    if len(sys.argv) == 4 and sys.argv[1] == "--nearid-candidate":
        root = Path(sys.argv[2])
        result = nearid_candidate(root, sys.argv[3])
        with (root / f"candidate-{sys.argv[3]}.GATES.json").open("x") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
        print(json.dumps({k: v for k, v in result.items() if k != "references"}, indent=2))
        return
    if len(sys.argv) == 3 and sys.argv[1] == "--nearid-summary":
        nearid_summary(Path(sys.argv[2]))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path)
    parser.add_argument("--steerfix-prepare", type=Path)
    parser.add_argument("--steerfix-report", type=Path)
    parser.add_argument("--wasm", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.steerfix_report is not None:
        assert args.native is None and args.output is None and args.wasm is None and args.steerfix_prepare is None
        report_steerfix(args.steerfix_report)
        return
    if args.steerfix_prepare is not None:
        assert args.native is None and args.output is None and args.wasm is None
        prepare_steerfix(args.steerfix_prepare)
        return
    if args.native is None or args.output is None:
        parser.error("--native and --output are required for summary")
    native = json.loads(args.native.read_text())
    wasm = json.loads(args.wasm.read_text()) if args.wasm else None
    if wasm is not None:
        assert native["geometries"] == wasm["geometries"]
        assert native["dropped_bits"] == wasm["dropped_bits"]
        assert len(native["seeds"]) == len(wasm["seeds"])
    summary = summarize(native, wasm)
    with args.output.open("x") as out:
        json.dump(summary, out, indent=2)
        out.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
