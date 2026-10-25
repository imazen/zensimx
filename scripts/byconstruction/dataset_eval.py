#!/usr/bin/env python3
"""The provided-library datasets' held-out EVAL splits: today's models and peers (benchmarks/byconstruction_2026-10-22.md,
"The new datasets' EVAL splits").

  dataset_eval.py prepare --out DIR      EVAL rows, their labels (logged), feature table and judge pairs
  dataset_eval.py score --out DIR --model NAME=BAKE ... --zm ZENMETRICS
  dataset_eval.py report --out DIR

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
import random
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


def statistic(rows, s, kind):
    if kind == "pair":
        vals = []
        for r in rows:
            if r["a"] in s and r["b"] in s:
                d = s[r["a"]] - s[r["b"]]
                vals.append(r["target"] if d > 0 else (1 - r["target"]) if d < 0 else 0.5)
        return st.mean(vals) if vals else float("nan"), len(vals)
    pts = [(s[r["a"]], r["target"]) for r in rows if r["a"] in s]
    return spearman([p[0] for p in pts], [p[1] for p in pts]), len(pts)


def within(rows, s):
    by = defaultdict(list)
    for r in rows:
        if r["a"] in s:
            by[r["group"]].append((s[r["a"]], r["target"]))
    jobs = [(str(i),[p[0] for p in g],[p[1] for p in g]) for i,g in enumerate(by.values()) if len(g) >= 3]
    vals = [r["srocc_signed"] for r in panel_batch(jobs,stats="srocc")] if jobs else []
    vals = [v for v in vals if not math.isnan(v)]
    return st.mean(vals) if vals else float("nan")


def cmd_report(a):
    out = Path(a.out)
    rows = json.loads((out / "eval_rows.json").read_text())
    scores = load_scores(out)
    names = sorted(scores)
    rng = random.Random(1)
    lines = ["| Collection | Rows | Statistic | " + " | ".join(names) + " |", "|---|---|---|" + "---|" * len(names)]
    result = {}
    for c in sorted({r["collection"] for r in rows}):
        rr = [r for r in rows if r["collection"] == c]
        if not rr:
            continue
        kind = rr[0]["kind"]
        groups = sorted({r["group"] for r in rr})
        cells, cells_w = [], []
        for n in names:
            method = None
            if kind == "mos":
                interval, used = mos_interval(rr,scores[n])
                v,lo,hi = (interval["signed"][k] for k in ("estimate","lo","hi"))
                method = interval["srocc_ci_method"]
            else:
                v, used = statistic(rr, scores[n], kind)
                # Pairwise agreement retains the original reference bootstrap.
                by_group = {g:[r for r in rr if r["group"] == g] for g in groups}
                boots = []
                for _ in range(2000):
                    pick = [groups[rng.randrange(len(groups))] for _ in groups]
                    sample = [r for g in pick for r in by_group[g]]
                    boots.append(statistic(sample, scores[n], kind)[0])
                boots = sorted(b for b in boots if not math.isnan(b))
                lo, hi = (boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]) if boots else (math.nan,) * 2
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
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
    j = sub.add_parser("judge-png")
    j.add_argument("--out", required=True)
    b = sub.add_parser("bapps-keys")
    b.add_argument("--out", required=True)
    b.add_argument("--scratch", required=True)
    a = ap.parse_args()
    return {"prepare": cmd_prepare, "score": cmd_score, "report": cmd_report, "judge-png": cmd_judge_png,
            "bapps-keys":cmd_bapps_keys}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
