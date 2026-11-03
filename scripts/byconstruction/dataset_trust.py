#!/usr/bin/env python3
"""Dataset trust screen (E36 method review M2): which compression-centric human sets are reliable, and how well do
SSIMULACRA2, butteraugli, CVVDP and DVIFM 0.3 agree with them? No zensim model is scored here.

Exposure: `docs/DATA_SPLITS.md`, "Exposure registration — 2026-11-03: dataset trust screen". Record:
`benchmarks/e36_subset_test_2026-11-01.md`, "M2 registration".

  dataset_trust.py plan --out DIR                 rows, labels (ledgered in DIR/LABEL_READS.json), colour metadata scan
  dataset_trust.py decode --out DIR --tool BIN    BMP/PNM inputs -> RGB8 PNG (zenbitmaps, `trust_decode`)
  dataset_trust.py score --out DIR --zm ZM --threads N     pinned judges + frozen DVIFM 0.3 runtime
  dataset_trust.py report --out DIR               trust table (DIR/trust.json, DIR/trust.md)

Images resolve under the tower IQA library (`/mnt/tower/input/iqa/datasets/<id>/content/`). MOS-type labels are
oriented to higher = better; a pair's `p_a` is the share of votes saying A is closer to the reference. Statistics come
from the canonical owners (`panel` through scripts/lib/zen_stats.py, and dataset_eval.py's bootstrap helpers): SROCC
with a whole-reference 95% interval, within-reference SROCC, PLCC after the panel's 4PL fit, KROCC, and for pair sets
the two-alternative agreement with its interval and the vote ceiling mean(max(p_a, 1 - p_a)).
"""
import argparse
import csv
import hashlib
import json
import math
import statistics as st
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.zen_stats import panel_batch  # noqa: E402
import dataset_eval as de  # noqa: E402

LIB = Path("/mnt/tower/input/iqa/datasets")
# id -> (table, kind, row filter). Registered 2026-11-03; changing this list needs a new DATA_SPLITS entry.
DATASETS = {
    "nncd-iqa": ("labels.tsv", "mos", None),
    "mmspg-jpeg-xr": ("labels.tsv", "mos", None),
    "midd-cap": ("labels.tsv", "mos", None),
    "midd-gar": ("labels.tsv", "mos", None),
    "clic-2021-perceptual": ("pairs.tsv", "pair", ("split", "valid")),
    "mucped23": ("pairs.tsv", "pair", None),
    "fidelity-lic-vcip2023": ("pairs.tsv", "pair", None),
}
# Library tables whose answer orientation is wrong, corrected here (the owner's fix belongs in zenpapers'
# `scripts/iqa_library/organize.py`). mucped23: the authors' compute_elo.ipynb counts a win for method X when the
# OTHER method is named in `answerValue`, so `answerValue` names the distortion judged farther from the original; the
# library counts it as the closer one. Checked 2026-11-03: `answerValue == A` agrees with the authors' published Elo
# order in 3,609 of 13,488 answers (26.8%).
INVERTED_ANSWERS = {"mucped23"}
DVIFM_RUNTIME = Path("/mnt/v/output/zensim/jnd-reconstruction-2026-10-20-dvifm03/dvifm-runtime")
JUDGES = ("ssim2", "butteraugli", "cvvdp")
# Metric column read from each judge output, and its direction (+1: higher = better quality).
COLUMNS = {"ssim2": ("ssim2", 1), "butteraugli_max": ("butteraugli_max", -1),
           "butteraugli_pnorm3": ("butteraugli_pnorm3", -1), "cvvdp": ("cvvdp", 1), "dvifm": ("jnd", -1)}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_table(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


# ----------------------------------------------------------------------------------------------- colour metadata
def colour_tags(path):
    """Container-level colour signalling, read from the bytes (no pixel decode): PNG iCCP/sRGB/gAMA/cHRM/cICP chunks,
    JPEG APP2 ICC_PROFILE segments, WebP ICCP chunks, BMP info-header size and V4/V5 colour-space type."""
    b = Path(path).read_bytes()
    tags = []
    if b[:8] == b"\x89PNG\r\n\x1a\n":
        i = 8
        while i + 8 <= len(b):
            n = int.from_bytes(b[i:i + 4], "big")
            t = b[i + 4:i + 8].decode("latin-1")
            if t in ("iCCP", "sRGB", "gAMA", "cHRM", "cICP", "mDCV", "cLLI"):
                tags.append(t)
            if t == "IEND":
                break
            i += 12 + n
    elif b[:2] == b"\xff\xd8":
        i = 2
        while i + 4 <= len(b) and b[i] == 0xFF:
            m = b[i + 1]
            if m in (0xD9, 0xDA):
                break
            n = int.from_bytes(b[i + 2:i + 4], "big")
            if m == 0xE2 and b[i + 4:i + 16] == b"ICC_PROFILE\x00":
                tags.append("ICC")
            i += 2 + n
    elif b[:4] == b"RIFF" and b[8:12] == b"WEBP":
        i = 12
        while i + 8 <= len(b):
            t, n = b[i:i + 4].decode("latin-1"), int.from_bytes(b[i + 4:i + 8], "little")
            if t == "ICCP":
                tags.append("ICCP")
            i += 8 + n + (n & 1)
    elif b[:2] == b"BM":
        hs = int.from_bytes(b[14:18], "little")
        if hs >= 108:
            cs = b[14 + 56:14 + 60]
            tags.append(f"BMPV{4 if hs < 124 else 5}:{cs[::-1].decode('latin-1', 'replace')}")
    elif b[:1] == b"P" and b[1:2] in b"1234567":
        pass
    else:
        tags.append("unknown-container")
    return tags


def icc_description(path):
    """The `desc` of a PNG's embedded ICC profile (iCCP), or None; read from the bytes, nothing decoded."""
    import zlib
    b = Path(path).read_bytes()
    if b[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    i = 8
    while i + 8 <= len(b):
        n = int.from_bytes(b[i:i + 4], "big")
        if b[i + 4:i + 8] == b"iCCP":
            d = b[i + 8:i + 8 + n]
            name = d.split(b"\0", 1)[0]
            prof = zlib.decompress(d[len(name) + 2:])
            for k in range(int.from_bytes(prof[128:132], "big")):
                e = prof[132 + 12 * k:144 + 12 * k]
                if e[:4] == b"desc":
                    off, ln = int.from_bytes(e[4:8], "big"), int.from_bytes(e[8:12], "big")
                    body = prof[off:off + ln]
                    if body[:4] == b"desc":
                        return body[12:12 + int.from_bytes(body[8:12], "big")].rstrip(b"\0").decode("latin-1")
                    if body[:4] == b"mluc":
                        so, sl = int.from_bytes(body[20:24], "big"), int.from_bytes(body[24:28], "big")
                        return body[so:so + sl].decode("utf-16-be")
            return name.decode("latin-1")
        if b[i + 4:i + 8] == b"IEND":
            return None
        i += 12 + n
    return None


# ---------------------------------------------------------------------------------------------------------- plan
def cmd_plan(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    rows, reads, merged = [], {}, {}
    for ds, (table, kind, filt) in DATASETS.items():
        path = LIB / ds / table
        content = LIB / ds / "content"
        reads[ds] = dict(table=str(path), sha256=sha(path), kind=kind, filter=filt, state="requested")
        (out / "LABEL_READS.json").write_text(json.dumps(reads, indent=1) + "\n")
        with open(path, newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            kept = 0
            for i, r in enumerate(reader):
                if filt is not None and r[filt[0]] != filt[1]:
                    continue  # rows outside the registered split: no label column is used
                kept += 1
                if kind == "mos":
                    hib = r["higher_is_better"].strip().lower() in ("1", "true", "yes")
                    score = float(r["score"])
                    rows.append(dict(ds=ds, row=i, kind=kind, group=r["ref"], ref=str(content / r["ref"]),
                                     dist=str(content / r["item"]), target=score if hib else -score,
                                     distortion=r["distortion"], level=r["level"],
                                     std=float(r["std"]) if r["std"] else None,
                                     n=float(r["n_ratings"]) if r["n_ratings"] else None,
                                     scale=[r["scale_min"], r["scale_max"]], score_kind=r["score_kind"],
                                     is_reference=r["item"] == r["ref"]))
                else:
                    # Vote counts; p_a is filled from them where the table leaves it empty. Rows asking the same
                    # question (either order) merge into one pair with summed votes.
                    na, nb, nt = (float(r[c] or 0) for c in ("n_a", "n_b", "n_tie"))
                    a_, b_ = r["a"], r["b"]
                    if a_ > b_:
                        a_, b_, na, nb = b_, a_, nb, na
                    k = (ds, r["ref"], a_, b_)
                    if k not in merged:
                        merged[k] = dict(ds=ds, row=i, kind=kind, group=r["ref"], ref=str(content / r["ref"]),
                                         a=str(content / a_), b=str(content / b_), n_a=0.0, n_b=0.0, n_tie=0.0,
                                         subset=r.get("subset", ""), source_rows=0)
                    m = merged[k]
                    m["n_a"] += na; m["n_b"] += nb; m["n_tie"] += nt; m["source_rows"] += 1
        reads[ds].update(state="read", rows=kept)
    for m in merged.values():
        m["n"] = m["n_a"] + m["n_b"] + m["n_tie"]
        if m["n"] > 0:
            m["target"] = (m["n_a"] + 0.5 * m["n_tie"]) / m["n"]
            rows.append(m)
    for ds in DATASETS:
        reads[ds]["pairs_after_merge"] = sum(r["ds"] == ds for r in rows)
    (out / "LABEL_READS.json").write_text(json.dumps(reads, indent=1) + "\n")
    (out / "rows.json").write_text(json.dumps(rows) + "\n")
    pairs = {}
    for r in rows:
        for d in ([r["dist"]] if r["kind"] == "mos" else [r["a"], r["b"]]):
            pairs[(r["ref"], d)] = None
    missing = sorted({p for rd in pairs for p in rd if not Path(p).is_file()})
    if missing:
        raise SystemExit(f"{len(missing)} image paths missing, e.g. {missing[:3]}")
    images = sorted({p for rd in pairs for p in rd})
    colour = {p: colour_tags(p) for p in images}
    (out / "colour_tags.json").write_text(json.dumps(colour) + "\n")
    with open(out / "judge_pairs.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["ref_path", "dist_path", "pair_key"])
        for k, (r, d) in enumerate(pairs):
            w.writerow([r, d, f"p{k:06d}"])
    summary = defaultdict(lambda: defaultdict(int))
    for p, t in colour.items():
        ds = Path(p).relative_to(LIB).parts[0]
        for tag in (t or ["none"]):
            summary[ds][tag] += 1
    print(json.dumps(dict(rows={ds: sum(r["ds"] == ds for r in rows) for ds in DATASETS}, judge_pairs=len(pairs),
                          images=len(images), colour={k: dict(v) for k, v in summary.items()}), indent=1))


# -------------------------------------------------------------------------------------------------------- decode
def cmd_decode(a):
    """BMP and PNM inputs (the pinned judge build reads PNG, JPEG and WebP only) go through zenbitmaps; the judge
    pairs are rewritten to the PNG copies, the original list kept beside them."""
    out = Path(a.out)
    rows = read_table(out / "judge_pairs.tsv")
    raw = sorted({p for r in rows for p in (r["ref_path"], r["dist_path"])
                  if Path(p).read_bytes()[:2] == b"BM" or Path(p).read_bytes()[:1] == b"P"})
    (out / "decode_list.txt").write_text("".join(p + "\n" for p in raw))
    with open(out / "decode_map.tsv", "w") as f:
        subprocess.run([a.tool, str(out / "decode_list.txt"), str(out / "png")], stdout=f, check=True)
    m = {r["input"]: r["png"] for r in read_table(out / "decode_map.tsv")}
    if set(m) != set(raw):
        raise SystemExit("decode map does not cover the BMP/PNM inputs")
    (out / "judge_pairs.original.tsv").write_text((out / "judge_pairs.tsv").read_text())
    with open(out / "judge_pairs.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["ref_path", "dist_path", "pair_key"])
        for r in rows:
            w.writerow([m.get(r["ref_path"], r["ref_path"]), m.get(r["dist_path"], r["dist_path"]), r["pair_key"]])
    print(json.dumps(dict(decoded=len(raw), tool=a.tool, tool_sha256=sha(a.tool))))


# --------------------------------------------------------------------------------------------------------- score
def cmd_score(a):
    out = Path(a.out)
    for metric in JUDGES:
        dest = out / f"judge_{metric}.tsv"
        if not dest.exists():
            subprocess.run(["nice", "-n19", a.zm, "batch", "--metric", metric, "--pairs", str(out / "judge_pairs.tsv"),
                            "--output", str(dest)], check=True)
    dest = out / "judge_dvifm.tsv"
    if not dest.exists():
        frozen = json.loads((DVIFM_RUNTIME / "FROZEN.json").read_text())
        binary = DVIFM_RUNTIME / "dvifmish"
        if sha(binary) != frozen["binary_sha256"]:
            raise SystemExit("not the frozen DVIFM runtime")
        rows = read_table(out / "judge_pairs.tsv")
        # The frozen runtime reads no WebP: those inputs go through zenwebp (`trust_decode`) for DVIFM alone.
        webp = sorted({p for r in rows for p in (r["ref_path"], r["dist_path"])
                       if Path(p).read_bytes()[:12][8:12] == b"WEBP"})
        m = {}
        if webp:
            if not a.tool:
                raise SystemExit(f"{len(webp)} WebP inputs need --tool for DVIFM")
            (out / "decode_list_webp.txt").write_text("".join(p + "\n" for p in webp))
            with open(out / "decode_map_webp.tsv", "w") as f:
                subprocess.run([a.tool, str(out / "decode_list_webp.txt"), str(out / "png")], stdout=f, check=True)
            m = {r["input"]: r["png"] for r in read_table(out / "decode_map_webp.tsv")}
        src = out / "dvifm_pairs.tsv"
        with open(src, "w", newline="") as f:
            w = csv.writer(f, delimiter="\t", lineterminator="\n")
            w.writerow(["ref_path", "dist_path"])
            w.writerows([m.get(r["ref_path"], r["ref_path"]), m.get(r["dist_path"], r["dist_path"])] for r in rows)
        subprocess.run(["nice", "-n19", str(binary), "batch", "--pairs", str(src), "--out", str(dest), "--preset",
                        frozen["preset"], "--threads", str(a.threads)], check=True)
    print(json.dumps({p.name: sha(p) for p in sorted(out.glob("judge_*.tsv"))}, indent=1))


# -------------------------------------------------------------------------------------------------------- report
def load_metrics(out):
    """{metric: {(original ref, original dist): quality-oriented score}} over the judge pairs."""
    keys = {r["pair_key"]: (r["ref_path"], r["dist_path"]) for r in read_table(out / "judge_pairs.original.tsv")} \
        if (out / "judge_pairs.original.tsv").exists() else \
        {r["pair_key"]: (r["ref_path"], r["dist_path"]) for r in read_table(out / "judge_pairs.tsv")}
    order = list(keys.values())
    metrics = {}
    for metric in JUDGES:
        t = read_table(out / f"judge_{metric}.tsv")
        for name, (col, sign) in COLUMNS.items():
            hit = [c for c in (t[0] if t else {}) if c == col or c.startswith(col + "_")]
            if hit:
                metrics[name] = {keys[r["pair_key"]]: sign * float(r[hit[0]]) for r in t}
    t = read_table(out / "judge_dvifm.tsv")
    dv = [(r["ref_path"], r["dist_path"]) for r in read_table(out / "dvifm_pairs.tsv")]
    if len(dv) != len(order) or len(t) != len(order) or any(
            (r["ref_path"], r["dist_path"]) != dv[int(r["row"])] for r in t):
        raise SystemExit("DVIFM rows do not match its pair list")
    metrics["dvifm"] = {order[int(r["row"])]: -float(r["jnd"]) for r in t}
    for name, v in metrics.items():
        if len(v) != len(order) or not all(math.isfinite(x) for x in v.values()):
            raise SystemExit(f"{name}: missing or non-finite scores")
    return metrics


def label_quality_mos(rr):
    q = {}
    y = np.array([r["target"] for r in rr])
    se = [r["std"] / math.sqrt(r["n"]) for r in rr if r["std"] is not None and r["n"]]
    if len(se) == len(rr):
        q["reliability"] = float(1 - np.mean(np.square(se)) / np.var(y))  # 1 - mean SE^2 / var(MOS)
        q["median_n"] = float(np.median([r["n"] for r in rr]))
    # Within (reference, distortion) ladders: the share of adjacent numeric levels whose ratings move against the
    # ladder's own majority direction.
    groups = defaultdict(list)
    for r in rr:
        try:
            groups[(r["group"], r["distortion"])].append((float(r["level"]), r["target"]))
        except ValueError:
            pass
    adj = inv = 0
    for g in groups.values():
        g.sort()
        d = [b[1] - a[1] for a, b in zip(g, g[1:]) if b[0] != a[0]]
        if len(d) < 1:
            continue
        sign = np.sign(sum(np.sign(d)))
        if sign == 0:
            continue
        adj += len(d)
        inv += sum(1 for x in d if np.sign(x) == -sign)
    if adj:
        q["ladder_steps"] = adj
        q["ladder_inversion_share"] = inv / adj
    return q


def cmd_report(a):
    out = Path(a.out)
    rows = json.loads((out / "rows.json").read_text())
    for r in rows:
        if r["ds"] in INVERTED_ANSWERS:
            r["n_a"], r["n_b"] = r["n_b"], r["n_a"]
            r["target"] = (r["n_a"] + 0.5 * r["n_tie"]) / r["n"]
    metrics = load_metrics(out)
    names = list(metrics)
    report = {}
    for ds in DATASETS:
        rr = [r for r in rows if r["ds"] == ds]
        kind = rr[0]["kind"]
        rep = dict(kind=kind, rows=len(rr), references=len({r["group"] for r in rr}), metrics={})
        if kind == "mos":
            refs = [r for r in rr if r["is_reference"]]
            rr = [r for r in rr if not r["is_reference"]]
            rep.update(rows_scored=len(rr), reference_rows_excluded=len(refs), labels=label_quality_mos(rr),
                       score_kind=rr[0]["score_kind"])
            y = [r["target"] for r in rr]
            for m in names:
                s = {f"{r['row']}": metrics[m][(r["ref"], r["dist"])] for r in rr}
                rows_k = [dict(r, a=f"{r['row']}") for r in rr]
                (ci, n) = de.mos_interval(rows_k, s)
                x = [s[f"{r['row']}"] for r in rr]
                pn = panel_batch([(m, x, y)], stats="full")[0]
                rep["metrics"][m] = dict(srocc=ci["signed"]["estimate"], ci=[ci["signed"]["lo"], ci["signed"]["hi"]],
                                         within=de.within(rows_k, s),
                                         plcc=pn["plcc"], krocc=pn["krocc"], n=n)
            items = [(r["ref"], r["dist"]) for r in rr]
        else:
            # Observer agreement: each vote of a pair with at least two votes is predicted by the majority of the
            # pair's other votes (a tie among them earns half; a tie vote counts half for each side).
            agree = total = 0.0
            for r in rr:
                if r["n"] < 2:
                    continue
                for side, cnt in (("a", r["n_a"]), ("b", r["n_b"]), ("t", r["n_tie"])):
                    if cnt == 0:
                        continue
                    oa = r["n_a"] - (side == "a") + 0.5 * (r["n_tie"] - (side == "t"))
                    ob = r["n_b"] - (side == "b") + 0.5 * (r["n_tie"] - (side == "t"))
                    pred = 0.5 if oa == ob else float(oa > ob)
                    vote = 1.0 if side == "a" else 0.0 if side == "b" else 0.5
                    agree += cnt * (pred * vote + (1 - pred) * (1 - vote))
                    total += cnt
            n = np.array([r["n"] for r in rr])
            rep["labels"] = dict(observer_agreement=agree / total if total else None,
                                 pairs_with_2plus_votes=int((n >= 2).sum()), median_votes=float(np.median(n)),
                                 tie_vote_share=float(sum(r["n_tie"] for r in rr) / n.sum()))
            for m in names:
                s, rows_k = {}, []
                for k, r in enumerate(rr):
                    s[f"{k}a"] = metrics[m][(r["ref"], r["a"])]
                    s[f"{k}b"] = metrics[m][(r["ref"], r["b"])]
                    rows_k.append(dict(r, a=f"{k}a", b=f"{k}b"))
                acc, lo, hi, n = de.pair_agreement(rows_k, s)
                rep["metrics"][m] = dict(agreement=acc, ci=[lo, hi], n=n)
            items = sorted({(r["ref"], d) for r in rr for d in (r["a"], r["b"])})
        # Colour sensitivity: rows whose reference or distorted image embeds a non-sRGB ICC profile (the judges compare
        # raw values), dropped and re-scored.
        colour = json.loads((out / "colour_tags.json").read_text())
        flagged = {p for p, t in colour.items() if "iCCP" in t and "sRGB" not in (icc_description(p) or "")}
        sides = (lambda r: (r["ref"], r["dist"])) if kind == "mos" else (lambda r: (r["ref"], r["a"], r["b"]))
        clean = [r for r in rr if not set(sides(r)) & flagged]
        if len(clean) < len(rr):
            sens = {}
            for m in names:
                if kind == "mos":
                    s = {f"{r['row']}": metrics[m][(r["ref"], r["dist"])] for r in clean}
                    ci, n = de.mos_interval([dict(r, a=f"{r['row']}") for r in clean], s)
                    sens[m] = dict(srocc=ci["signed"]["estimate"], n=n)
                else:
                    s, rows_k = {}, []
                    for k, r in enumerate(clean):
                        s[f"{k}a"], s[f"{k}b"] = metrics[m][(r["ref"], r["a"])], metrics[m][(r["ref"], r["b"])]
                        rows_k.append(dict(r, a=f"{k}a", b=f"{k}b"))
                    acc, lo, hi, n = de.pair_agreement(rows_k, s)
                    sens[m] = dict(agreement=acc, ci=[lo, hi], n=n)
            rep["without_non_srgb_icc"] = dict(rows=len(clean), dropped=len(rr) - len(clean),
                                               profiles=sorted({icc_description(p) for p in flagged
                                                                if Path(p).is_relative_to(LIB / ds)}), metrics=sens)
        # How much the metrics agree with each other on this set's images (Spearman, unique distorted images).
        vec = {m: [metrics[m][i] for i in items] for m in names}
        jobs = [(f"{m1}|{m2}", vec[m1], vec[m2]) for i, m1 in enumerate(names) for m2 in names[i + 1:]]
        rep["metric_agreement"] = {j[0]: r["srocc_signed"] for j, r in zip(jobs, panel_batch(jobs, stats="srocc"))}
        report[ds] = rep
    colour = json.loads((out / "colour_tags.json").read_text())
    for ds in DATASETS:
        tags = defaultdict(int)
        for pth, t in colour.items():
            if Path(pth).is_relative_to(LIB / ds):
                for tag in (t or ["none"]):
                    tags[tag] += 1
        report[ds]["colour_tags"] = dict(tags)
    (out / "trust.json").write_text(json.dumps(report, indent=1) + "\n")
    lines = ["| set | kind | rows / refs | label quality | " + " | ".join(names) + " | metric-metric SROCC (median) |",
             "|---|---|---|---|" + "---|" * len(names) + "---|"]
    for ds, rep in report.items():
        if rep["kind"] == "mos":
            lq = rep["labels"]
            lab = ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in lq.items())
            cells = [f"{v['srocc']:.3f} [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}] w {v['within']:.3f}"
                     for v in rep["metrics"].values()]
        else:
            lq = rep["labels"]
            oa = lq["observer_agreement"]
            lab = (f"observer agreement {oa:.3f} over {lq['pairs_with_2plus_votes']} pairs" if oa is not None else
                   "single votes") + f", median votes {lq['median_votes']:.0f}, ties {lq['tie_vote_share']:.2f}"
            cells = [f"{v['agreement']:.3f} [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}]" for v in rep["metrics"].values()]
        mm = st.median(rep["metric_agreement"].values())
        lines.append(f"| {ds} | {rep['kind']} | {rep['rows']} / {rep['references']} | {lab} | " + " | ".join(cells)
                     + f" | {mm:.3f} |")
    (out / "trust.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--out", required=True)
    d = sub.add_parser("decode")
    d.add_argument("--out", required=True)
    d.add_argument("--tool", required=True)
    s = sub.add_parser("score")
    s.add_argument("--out", required=True)
    s.add_argument("--zm", default="/var/tmp/byconstruction-bin/zenmetrics-judge-7a0f8394/zenmetrics")
    s.add_argument("--threads", type=int, default=16)
    s.add_argument("--tool", help="trust_decode binary, for WebP inputs DVIFM cannot read")
    r = sub.add_parser("report")
    r.add_argument("--out", required=True)
    a = ap.parse_args()
    return {"plan": cmd_plan, "decode": cmd_decode, "score": cmd_score, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
