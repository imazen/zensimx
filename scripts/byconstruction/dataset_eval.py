#!/usr/bin/env python3
"""The provided-library datasets' held-out EVAL splits: today's models and peers (benchmarks/byconstruction_2026-10-22.md,
"The new datasets' EVAL splits").

  dataset_eval.py mix-eval --study DIR --predict BINARY   frozen TRAIN-selected finalists, copied EVAL + feature-only BAPPS
  dataset_eval.py prepare --out DIR      EVAL rows, their labels (logged), feature table and judge pairs
  dataset_eval.py score --out DIR --model NAME=BAKE ... --zm ZENMETRICS
  dataset_eval.py report --out DIR
  dataset_eval.py paired --out DIR --dest NEW --a A --vs B[,C]   paired A − B through `panel evaluate`
  dataset_eval.py scatter --out DIR      the dashboard's scatter views (logistic fit, rank map, matched) per MOS set

Inputs: the frozen memberships and revision-7 feature banks of the 2026-10-25 admission
(/mnt/v/zensim/dataset-features-rev7-20261025/{inputs,bank}/<collection>/) and each collection's label table in the
tower IQA library (/mnt/tower/input/iqa/datasets/<id>/). Only rows whose frozen role is EVAL are read; every label read
is listed in DIR/LABEL_READS.json. MOS tables (`score`, `higher_is_better`, `ref`) are oriented to higher = better and
scored by Spearman correlation over all rows and within each reference; pairwise tables (`p_a`: the probability that A
is closer to the reference) by the two-alternative agreement `p_a·[A ahead] + (1 − p_a)·[B ahead] + ½·[tie]`. Each
statistic carries a 95% bootstrap interval over references (2,000 resamples, seed 1).
"""
import argparse
import csv
import json
import math
import hashlib
import shutil
import statistics as st
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.zen_stats import panel_batch, reference_srocc_interval  # noqa: E402

FEAT = Path("/mnt/v/zensim/dataset-features-rev7-20261025")
LIB = Path("/mnt/tower/input/iqa/datasets")
# Collections with FROZEN EVAL rows and features. Left out: LocVisVC (its markings are spatial maps, not a scalar
# target) and Colourlab chroma masking (each row pairs a Cb and a Cr stimulus: a masking-threshold design, not a rating).
COLLECTIONS = ["cid-iq", "cid-oa", "colourlab-rating-variability-2022", "fg-iqa", "iqa-odi", "irccyn-ivc", "live-md",
               "mdid", "nits-iqa", "pieapp", "vcl-fer"]


def token(c):
    return c.replace("-", "_")


def cmd_prepare(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    reads, rows, pair_keys = {}, [], set()
    collections = a.collections.split(",") if a.collections else COLLECTIONS
    if len(set(collections)) != len(collections) or not set(collections) <= set(COLLECTIONS+["bapps"]):
        raise ValueError("explicit supported EVAL collections required")
    for c in collections:
        m = pq.read_table(FEAT / "inputs" / c / "membership-to-pairs.parquet").to_pylist()
        ev = [r for r in m if r["role"] == "EVAL" and r["skip_reason"] is None]
        if not ev:
            continue
        table = {r["input_table"] for r in ev}
        assert len(table) == 1, (c, table)
        path = LIB / c / table.pop()
        want = {r["membership_row_index"] for r in ev}
        labels = {}
        with open(path) as f:
            header = next(csv.reader([next(f)],delimiter="\t"))
            reads[c] = {"table":str(path),"eval_rows":len(want),"rows":sorted(want),
                        "columns":header,"state":"requested"}
            (out/"LABEL_READS.json").write_text(json.dumps({"role":"EVAL","label_values_read_for":reads},indent=1)+"\n")
            for i, line in enumerate(f):
                if i in want:
                    values = next(csv.reader([line],delimiter="\t"))
                    if len(values) != len(header):
                        raise ValueError("one normalized TSV record per line required")
                    labels[i] = dict(zip(header,values))
        assert set(labels) == want, (c, len(want - set(labels)))
        # The frozen role governs; the library's own split column is recorded, not trusted.
        native = defaultdict(int)
        for r in labels.values():
            native[r.get("split", "")] += 1
        reads[c].update(native_split_values=dict(native),state="read")
        by_row = defaultdict(dict)
        for r in ev:
            by_row[r["membership_row_index"]][r["side"]] = r["pair_key"]
            pair_keys.add(r["pair_key"])
        for i, sides in sorted(by_row.items()):
            lab = labels[i]
            if "p_a" in lab:
                rows.append({"collection": c, "row": i, "kind": "pair", "group": lab["ref"], "a": sides["a"],
                             "b": sides["b"], "target": float(lab["p_a"])})
            else:
                hib = lab["higher_is_better"].strip().lower() in ("true", "1", "yes")
                v = float(lab["score"])
                rows.append({"collection": c, "row": i, "kind": "mos", "group": lab["ref"], "a": sides["item"],
                             "b": None, "target": v if hib else -v})
    (out / "LABEL_READS.json").write_text(json.dumps({"role": "EVAL", "label_values_read_for": reads,
                                                      "rows": len(rows)}, indent=1) + "\n")
    with open(out / "eval_rows.json", "w") as f:
        json.dump(rows, f)
    # One feature table over the needed pairs (bank order kept per collection), and the judge pairs.
    parts, judge = [], []
    for c in collections:
        bank = FEAT / "bank" / token(c)
        if not bank.exists():
            continue
        t = pq.read_table(bank / "features.parquet")
        keep = [k in pair_keys for k in t.column("pair_key").to_pylist()]
        t = t.filter(pa.array(keep)).drop(["row_id"]).rename_columns(
            ["ref_basename" if n == "pair_key" else n for n in t.drop(["row_id"]).column_names])
        parts.append(t)
        k = pq.read_table(bank / "keys.parquet").to_pylist()
        judge += [(r["ref_path"], r["dist_path"], r["pair_key"]) for r in k if r["pair_key"] in pair_keys]
    feats = pa.concat_tables(parts)
    # The production predictor requires a target column; it holds no label here.
    feats = feats.append_column("human_score", pa.array(np.zeros(feats.num_rows)))
    pq.write_table(feats, out / "features.parquet", compression="zstd")
    with open(out / "judge_pairs.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["ref_path", "dist_path", "pair_key"])
        w.writerows(judge)
    print(f"{len(rows)} EVAL rows over {len(reads)} collections; {feats.num_rows} feature rows; {len(judge)} judge pairs")


def cmd_judge_png(a):
    """Lossless PNG copies of the BMP judge inputs (the pinned judge build has no BMP decoder). Each BMP is decoded,
    its RGB8 bytes' SHA-256 must equal the pixel hash our extractor recorded for that image, and only then is it
    written as PNG; the judge pairs are rewritten to the copies. JPEG inputs stay as they are: the judge decodes them
    with its own decoder, which this step does not check."""
    import hashlib
    from PIL import Image
    out = Path(a.out)
    pngdir = out / "judge_png"
    pngdir.mkdir(exist_ok=False)
    recorded = {}
    for c in COLLECTIONS+["bapps"]:
        bank = FEAT / "bank" / token(c)
        if (bank / "keys.parquet").exists():
            for r in pq.read_table(bank / "keys.parquet").to_pylist():
                recorded[r["ref_path"]] = r["reference_pixels_sha256"]
                recorded[r["dist_path"]] = r["distorted_pixels_sha256"]
    rows = list(csv.DictReader(open(out / "judge_pairs.tsv"), delimiter="\t"))
    converted, kept = {}, defaultdict(int)
    for r in rows:
        for col in ("ref_path", "dist_path"):
            path = r[col]
            if not path.lower().endswith(".bmp"):
                kept[Path(path).suffix.lower()] += 1
                continue
            if path not in converted:
                rgb = Image.open(path).convert("RGB")
                digest = hashlib.sha256(rgb.tobytes()).hexdigest()
                if digest != recorded[path]:
                    raise SystemExit(f"{path}: decoded pixels differ from the extractor's ({digest} != {recorded[path]})")
                dest = pngdir / f"{digest}.png"
                if not dest.exists():
                    rgb.save(dest, format="PNG")
                converted[path] = str(dest)
            r[col] = converted[path]
    (out / "judge_pairs.bmp.tsv").write_text((out / "judge_pairs.tsv").read_text())
    with open(out / "judge_pairs.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    (out / "JUDGE_PNG.json").write_text(json.dumps({"bmp_converted": len(converted), "pixel_hash_verified": True,
                                                    "kept_as_is_by_extension": dict(kept)}, indent=1) + "\n")
    print(f"{len(converted)} BMP images verified and written as PNG; kept as is: {dict(kept)}")


def cmd_score(a):
    out = Path(a.out)
    feats = out / "features.parquet"
    for spec in a.model:
        name, bake = spec.split("=", 1)
        subprocess.run([a.predict, "predict", "--bake", bake, "--corpus", str(feats), "--score-units",
                        "--out", str(out / f"pred_{name}.tsv")], check=True)
    for metric in a.judge:
        subprocess.run(["nice", "-n19", a.zm, "batch", "--metric", metric, "--pairs", str(out / "judge_pairs.tsv"),
                        "--output", str(out / f"judge_{metric}.tsv")], check=True)


def cmd_bapps_keys(a):
    """Rebind EVAL judge paths for the existing original-PNG unpack owner."""
    out, scratch = Path(a.out), Path(a.scratch)
    original = out/"judge_pairs.original-paths.bak.tsv"
    if original.exists() or (out/"bapps-unpack-keys.parquet").exists():
        raise ValueError("BAPPS judge rebind destination already exists")
    with open(out/"judge_pairs.tsv") as stream:
        rows = list(csv.DictReader(stream,delimiter="\t"))
    rebound = []
    for r in rows:
        r = dict(r)
        for c in ("ref_path","dist_path"):
            parts = Path(r[c]).parts[-4:]
            if len(parts)!=4 or parts[0]!="val" or parts[2] not in ("ref","p0","p1") or not parts[-1].endswith(".png"):
                raise ValueError("BAPPS EVAL requires its original native VAL PNG paths")
            r[c] = str(scratch.joinpath(*parts))
        rebound.append(r)
    keys = pa.table({c:[r[c] for r in rebound] for c in ("ref_path","dist_path")})
    pq.write_table(keys,out/"bapps-unpack-keys.parquet",compression="zstd")
    (out/"judge_pairs.tsv").rename(original)
    with open(out/"judge_pairs.tsv","x",newline="") as f:
        writer = csv.DictWriter(f,fieldnames=list(rebound[0]),delimiter="\t")
        writer.writeheader()
        writer.writerows(rebound)
    print(f"{len(rebound)} BAPPS EVAL judge pairs rebound to original-PNG scratch; no target opened")


def load_scores(out):
    keys = pq.read_table(out / "features.parquet", columns=["ref_basename"]).column("ref_basename").to_pylist()
    scores = {}
    for p in sorted(out.glob("pred_*.tsv")):
        with open(p) as f:
            v = [float(r["pred"]) for r in csv.DictReader(f, delimiter="\t")]
        assert len(v) == len(keys), p
        scores[p.stem[5:]] = dict(zip(keys, v))
    # Judge rows carry their pair key. Several rows can share one distorted image (LIVE-MD rates 18 images twice,
    # once in each part), so a path never identifies a row.
    with open(out / "judge_pairs.tsv") as f:
        judge_paths = {r["pair_key"]: (r["ref_path"], r["dist_path"]) for r in csv.DictReader(f, delimiter="\t")}
    for p in sorted(out.glob("judge_*.tsv")):
        if p.name.startswith("judge_pairs"):
            continue
        with open(p) as f:
            rd = csv.DictReader(f, delimiter="\t")
            if "pair_key" not in rd.fieldnames:
                raise ValueError(f"{p.name}: judge output without pair_key")
            cols = [c for c in rd.fieldnames if c not in ("ref_path", "dist_path", "pair_key")]
            prefer = ("butteraugli_pnorm3", "ssim2", "cvvdp_cpu_imazen_v", "zensim")
            col = next((c for k in prefer for c in cols if c.startswith(k)), cols[0])
            sign = -1.0 if "butteraugli" in p.stem else 1.0
            s = {}
            for r in rd:
                if judge_paths.get(r["pair_key"]) != (r["ref_path"], r["dist_path"]):
                    raise ValueError(f"{p.name}: row {r['pair_key']} does not match judge_pairs.tsv")
                if r[col] not in ("", "nan"):
                    s[r["pair_key"]] = sign * float(r[col])
        scores[p.stem[6:]] = s
    return scores


def spearman(x, y):
    return panel_batch([("rank", x, y)], stats="srocc")[0]["srocc_signed"]


def mos_interval(rows, scores, resamples=2000):
    """Canonical midrank Spearman and whole-reference bootstrap, with no tie breaking by row order."""
    points = [r for r in rows if r["a"] in scores]
    groups = {g:i for i,g in enumerate(sorted({r["group"] for r in points}))}
    result = reference_srocc_interval([scores[r["a"]] for r in points], [r["target"] for r in points],
                                     [groups[r["group"]] for r in points], resamples=resamples, seed=1)
    return result, len(points)


def pair_agreement(rows, s, resamples=2000, seed=1):
    """Forced-choice agreement from the canonical owner (zenstats::forced_choice through `panel --pairwise`) with a
    whole-reference bootstrap interval. A pair's soft probability p_a that A is closer to the reference becomes two
    weighted responses (B more distorted, weight p_a; A more distorted, weight 1 - p_a), so each pair weighs 1 and the
    owner's weighted agreement is the mean over pairs of p_a·[A ahead] + (1 - p_a)·[B ahead] + ½·[tie]."""
    from lib.zen_stats import _find_panel_bin
    pairs = [r for r in rows if r["a"] in s and r["b"] in s]
    resp = [(r["group"], s[r["a"]], s[r["b"]], c, w) for r in pairs
            for c, w in (("right", r["target"]), ("left", 1 - r["target"])) if w > 0]
    if not resp:
        return math.nan, math.nan, math.nan, 0
    tsv = "group\ts_left\ts_right\tchoice\tweight\n" + "".join(f"{g}\t{a!r}\t{b!r}\t{c}\t{w!r}\n" for g, a, b, c, w in resp)
    out = subprocess.run([_find_panel_bin(), "--pairwise", "-"], input=tsv, capture_output=True, text=True,
                         check=True).stdout.strip().splitlines()
    head, row = out[0].split("\t"), out[1].split("\t")
    acc = float(row[head.index("acc_response")])
    # The interval resamples whole references over per-reference sums of the same agreement, checked against the owner.
    gi = {g: i for i, g in enumerate(dict.fromkeys(x[0] for x in resp))}
    g = np.array([gi[x[0]] for x in resp])
    a, b, w = (np.array([x[k] for x in resp], np.float64) for k in (1, 2, 4))
    right = np.array([x[3] == "right" for x in resp])
    agree = np.where(a == b, 0.5, np.where(right, a > b, a < b).astype(np.float64))
    num, den = np.bincount(g, w * agree, len(gi)), np.bincount(g, w, len(gi))
    if abs(num.sum() / den.sum() - acc) > 1e-8:
        raise ValueError(f"pairwise agreement {num.sum() / den.sum()} disagrees with the owner's {acc}")
    rng = np.random.default_rng(seed)
    boots = np.empty(resamples)
    for i in range(resamples):
        k = np.bincount(rng.integers(0, len(gi), len(gi)), minlength=len(gi))
        boots[i] = (k @ num) / (k @ den)
    lo, hi = np.quantile(boots, [0.025, 0.975])  # Hyndman-Fan type 7
    return acc, float(lo), float(hi), len(pairs)


def within(rows, s):
    by = defaultdict(list)
    for r in rows:
        if r["a"] in s:
            by[r["group"]].append((s[r["a"]], r["target"]))
    jobs = [(str(i),[p[0] for p in g],[p[1] for p in g]) for i,g in enumerate(by.values()) if len(g) >= 3]
    vals = [r["srocc_signed"] for r in panel_batch(jobs,stats="srocc")] if jobs else []
    vals = [v for v in vals if not math.isnan(v)]
    return st.mean(vals) if vals else float("nan")


def read_eval_rows(out, purpose):
    path = out / "eval_rows.json"
    if (out / "EVAL_ASSESSMENT.json").exists():
        ledger = out / "EVAL_DERIVED_LABEL_READS.json"
        value = json.loads(ledger.read_text()) if ledger.exists() else dict(role="EVAL", reads=[])
        value["reads"].append(dict(table=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                   rows="all", columns=["target", "group"], purpose=purpose,
                                   state="authorized-before-read"))
        ledger.write_text(json.dumps(value,indent=1)+"\n")
    return json.loads(path.read_text())


def cmd_mix_eval(a):
    """Assess frozen seed-0 finalists once, reusing frozen Mdata predictions."""
    study = Path(a.study)
    frozen_path = study / "FINALISTS.json"
    frozen = json.loads(frozen_path.read_text())
    selection_path = study / "TRAIN_SELECTION.json"
    if (frozen["schema"] != "dataset-mix-finalists-v1" or frozen["selection_uses_EVAL"]
            or frozen["TRAIN_selection_sha256"] != hashlib.sha256(selection_path.read_bytes()).hexdigest()):
        raise ValueError("frozen TRAIN-only finalist decision required")
    for model in frozen["models"].values():
        if hashlib.sha256(Path(model["path"]).read_bytes()).hexdigest() != model["sha256"]:
            raise ValueError("frozen finalist model changed")
    for directory, source, mdata_source in (
        ("eval", Path(a.source), Path("/var/tmp/datalegs-20261025/eval")),
        ("eval-bapps", Path("/var/tmp/datalegs-20261025/eval-bapps"), Path("/var/tmp/datalegs-20261025/eval-bapps")),
    ):
        out = study / directory
        out.mkdir()
        copies = {}
        # Scoring uses the top-level tables and receipts. No image is unpacked
        # or rescored by peers; the corrected peer TSVs are copied unchanged.
        for path in sorted(source.iterdir()):
            if path.is_file() and path.suffix.lower() not in (".png", ".jpg", ".bmp", ".tif"):
                shutil.copyfile(path, out/path.name)
                copies[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        for name in ("features.parquet", "eval_rows.json"):
            if hashlib.sha256((out/name).read_bytes()).digest() != hashlib.sha256((mdata_source/name).read_bytes()).digest():
                raise ValueError("Mdata baseline and EVAL copy have different inputs")
        baseline = out / "pred_Mdata.tsv"
        expected = hashlib.sha256((mdata_source/baseline.name).read_bytes()).hexdigest()
        if baseline.exists() and hashlib.sha256(baseline.read_bytes()).hexdigest() != expected:
            raise ValueError("copied Mdata predictions differ from the frozen assessment")
        if not baseline.exists():
            shutil.copyfile(mdata_source/baseline.name, baseline)
        prior = out / "copied-source-reports"
        prior.mkdir()
        for name in ("report.json", "report.md"):
            if (out/name).exists():
                (out/name).rename(prior/name)
        receipt = dict(schema="dataset-mix-EVAL-assessment-v1", source=str(source), copied_file_sha256=copies,
                       finalists_sha256=hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
                       models=frozen["models"], Mdata_prediction_sha256=expected,
                       Mdata_assessment_reused=True, peer_images_unpacked=False)
        (out/"EVAL_ASSESSMENT.json").write_text(json.dumps(receipt,indent=1)+"\n")
        new = []
        for name, model in frozen["models"].items():
            if name != "Mdata":
                if (out/f"pred_{name}.tsv").exists():
                    raise ValueError("a finalist already has stored predictions; preserve this assessment")
                new.append(f"{name}={model['path']}")
        if new:
            cmd_score(argparse.Namespace(out=str(out), model=new, predict=a.predict, judge=[]))
        cmd_report(argparse.Namespace(out=str(out)))
        if directory == "eval":
            names = set(load_scores(out))
            for name in frozen["models"]:
                peers = [n for n in ("Mdata", "M7", "B7", "ssim2", "butteraugli", "cvvdp") if n != name and n in names]
                cmd_paired(argparse.Namespace(out=str(out),dest=str(out/f"paired-{name}"),a=name,vs=peers,resamples=2000))
    return 0


def cmd_report(a):
    out = Path(a.out)
    rows = read_eval_rows(out, "dataset-eval-report")
    scores = load_scores(out)
    names = sorted(scores)
    lines = ["| Collection | Rows | Statistic | " + " | ".join(names) + " |", "|---|---|---|" + "---|" * len(names)]
    result = {}
    for c in sorted({r["collection"] for r in rows}):
        rr = [r for r in rows if r["collection"] == c]
        if not rr:
            continue
        kind = rr[0]["kind"]
        cells, cells_w = [], []
        for n in names:
            method = None
            if kind == "mos":
                interval, used = mos_interval(rr,scores[n])
                v,lo,hi = (interval["signed"][k] for k in ("estimate","lo","hi"))
                method = interval["srocc_ci_method"]
            else:
                v, lo, hi, used = pair_agreement(rr, scores[n])
            cells.append(f"{v:.3f} [{lo:.3f}, {hi:.3f}]")
            result.setdefault(c, {})[n] = {"value": v, "ci95": [lo, hi], "n": used}
            if method is not None:
                result[c][n]["srocc_ci_method"] = method
            if kind == "mos":
                w = within(rr, scores[n])
                cells_w.append(f"{w:.3f}")
                result[c][n]["within_reference"] = w
        stat = "2AFC agreement" if kind == "pair" else "SROCC"
        lines.append(f"| {c} | {len(rr)} | {stat} | " + " | ".join(cells) + " |")
        if kind == "mos":
            lines.append("| | | within-reference SROCC | " + " | ".join(cells_w) + " |")
    (out / "report.json").write_text(json.dumps(result, indent=1) + "\n")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def cmd_paired(a):
    """Paired A-minus-B differences on each MOS collection, through the canonical `panel evaluate` owner.

    Writes one table per collection (id, oriented target, reference_id, one column per stored score) and a suite
    with a comparison of A against each B; the owner resamples whole references for both arms on the same draw.
    Pairwise collections stay with `report`: their soft-probability agreement has no panel comparison.
    """
    import hashlib
    from lib.zen_stats import _find_panel_bin
    out, dest = Path(a.out), Path(a.dest)
    rows = read_eval_rows(out, "dataset-eval-paired")
    scores = load_scores(out)
    names = sorted(scores)
    for n in [a.a, *a.vs]:
        if n not in scores:
            raise ValueError(f"no stored scores named {n}; have {names}")
    dest.mkdir(parents=True)
    suite = {"schema": "metric-evaluation-v1", "datasets": [], "comparisons": [],
             "metrics": [{"id": n, "direction": "higher", "implementation": f"stored scores {n} from {out}",
                          "input_contract": "dataset_eval.py score: models in score units, peers oriented higher "
                                            "is better (butteraugli negated)",
                          "source": {"kind": "column", "column": n}} for n in names]}
    for c in sorted({r["collection"] for r in rows if r["kind"] == "mos"}):
        rr = [r for r in rows if r["collection"] == c]
        missing = {n: sum(r["a"] not in scores[n] for r in rr) for n in names}
        if any(missing.values()):
            raise ValueError(f"{c}: rows without a stored score {missing}")
        path = dest / f"{c}.tsv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f, delimiter="\t", lineterminator="\n")
            w.writerow(["id", "target", "reference_id", *names])
            for r in rr:
                w.writerow([f"{c}:{r['row']}", repr(r["target"]), r["group"], *(repr(scores[n][r["a"]]) for n in names)])
        suite["datasets"].append({"id": c, "path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                  "role": "eval", "target_kind": "human",
                                  "columns": {"id": "id", "target": "target", "reference_id": "reference_id",
                                              **{n: n for n in names}}})
        suite["comparisons"] += [{"dataset": c, "a": a.a, "b": b, "bootstrap_resamples": a.resamples, "seed": 1}
                                 for b in a.vs]
    (dest / "suite.json").write_text(json.dumps(suite, indent=1) + "\n")
    subprocess.run([_find_panel_bin(), "evaluate", "--manifest", str(dest / "suite.json"),
                    "--output", str(dest / "report")], check=True)
    render_paired(dest)


SCATTER_KEYS = {"B7": "b7", "M7": "m7", "Mdata": "mdata", "Mlic": "mlic", "ssim2": "ssim2", "cvvdp": "cvvdp",
                "butteraugli": "butter", "MlicXref256e59": "xref256", "MlicAICOnlye29": "aiconly"}


def cmd_scatter(a):
    """Scatter views of every stored score on each MOS collection's own label scale, as the dashboard draws the E21
    sets: the panel's logistic fit (f_), zenstats' rank map (r_) and mean/SD matching (a_), with zenstats geometry and
    the logistic-fit error tails per collection and model. Writes eval_scatter.json (rows, no image names) and
    eval_scatter_stats.json."""
    import pandas as pd
    from aic_panel import geometry  # the E21 views' owner
    out = Path(a.out)
    rows = read_eval_rows(out, "dataset-eval-scatter")
    scores = load_scores(out)
    names = [n for n in sorted(scores) if n in SCATTER_KEYS]
    frames, stats = [], []
    for c in sorted({r["collection"] for r in rows if r["kind"] == "mos"}):
        rr = [r for r in rows if r["collection"] == c]
        y = np.array([r["target"] for r in rr], np.float64)
        frame = {"ds": c, "label": np.round(y, 4)}
        for n in names:
            missing = sum(r["a"] not in scores[n] for r in rr)
            if missing:
                raise ValueError(f"{c}: {missing} rows without a {n} score")
            x = np.array([scores[n][r["a"]] for r in rr], np.float64)
            st, arr = geometry(x, y, arrays=True)
            pn = panel_batch([(n, x, y)], stats="full")[0]
            k = SCATTER_KEYS[n]
            frame[f"f_{k}"] = np.round(arr["fit"], 4)
            # zenstats leaves the rank geometry unmeasured on small sets: that view stays empty, never invented.
            frame[f"r_{k}"] = np.round(arr["rank"], 4) if arr["rank"] is not None else [None] * len(y)
            frame[f"a_{k}"] = np.round(y.mean() + (x - x.mean()) * (y.std() / x.std()), 4)
            stats.append({"collection": c, "key": k, "model": n, "n": int(pn["n"]), "srocc": pn["srocc_signed"],
                          "plcc": pn["plcc"], "krocc": pn["krocc"], "pwrc": pn["pwrc"], "z_rmse": pn["z_rmse"],
                          "or": pn["or"], **st})
        frames.append(pd.DataFrame(frame))
    df = pd.concat(frames, ignore_index=True)
    df.sample(frac=1, random_state=0).reset_index(drop=True).to_json(out / "eval_scatter.json", orient="records")
    (out / "eval_scatter_stats.json").write_text(json.dumps(stats, indent=1) + "\n")
    print(f"{len(df)} rows over {len(frames)} collections; models {names}")


def render_paired(dest):
    report = json.loads((Path(dest) / "report" / "report.json").read_text())
    lines = ["| Collection | A − B | ΔSROCC [95%] | two-sided p (Δ = 0) | references |", "|---|---|---|---|---|"]
    for cmp in report["comparisons"]:
        spec, rep = cmp["spec"], cmp["report"]
        if rep.get("state") != "measured":
            lines.append(f"| {spec['dataset']} | {spec['a']} − {spec['b']} | {rep.get('state')}: {rep.get('reason')} | | |")
            continue
        m = rep["measurements"]
        d = m["delta"][m["delta_order"].index("srocc")]
        lines.append(f"| {spec['dataset']} | {spec['a']} − {spec['b']} | {d['estimate']:+.4f} [{d['lo']:+.4f}, "
                     f"{d['hi']:+.4f}] | {d['p_zero']:.3f} | {m['bootstrap']['references']} |")
    lines += ["", "Whole references resampled for both arms on each draw (zenstats cluster bootstrap). With fewer than "
                  "about 30 references the percentile interval under-covers: read these as descriptive."]
    (Path(dest) / "paired.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mix-eval")
    m.add_argument("--study", required=True)
    m.add_argument("--source", default="/var/tmp/byc-dataset-eval-2026-10-25-v2")
    m.add_argument("--predict", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--out", required=True)
    p.add_argument("--collections", help="explicit comma-separated subset; bapps is also supported")
    s = sub.add_parser("score")
    s.add_argument("--out", required=True)
    s.add_argument("--model", action="append", default=[], help="NAME=BAKE")
    s.add_argument("--predict", default="/var/tmp/byconstruction-bin/be72a3dfb039/bake_dial_refit")
    s.add_argument("--zm", default="/var/tmp/byconstruction-bin/zenmetrics-judge-7a0f8394/zenmetrics")
    s.add_argument("--judge", action="append", default=[])
    r = sub.add_parser("report")
    r.add_argument("--out", required=True)
    v = sub.add_parser("scatter")
    v.add_argument("--out", required=True)
    q = sub.add_parser("paired")
    q.add_argument("--out", required=True)
    q.add_argument("--dest", required=True, help="new directory for the tables, suite and evaluator report")
    q.add_argument("--a", required=True)
    q.add_argument("--vs", required=True, type=lambda s: s.split(","), help="comma-separated B arms")
    q.add_argument("--resamples", type=int, default=2000)
    j = sub.add_parser("judge-png")
    j.add_argument("--out", required=True)
    b = sub.add_parser("bapps-keys")
    b.add_argument("--out", required=True)
    b.add_argument("--scratch", required=True)
    a = ap.parse_args()
    return {"prepare": cmd_prepare, "score": cmd_score, "report": cmd_report, "paired": cmd_paired, "scatter": cmd_scatter,
            "judge-png": cmd_judge_png, "bapps-keys":cmd_bapps_keys, "mix-eval":cmd_mix_eval}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
