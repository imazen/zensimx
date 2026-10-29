#!/usr/bin/env python3
"""G-CHROMA: rank agreement with the SSIMULACRA2 teacher inside each held-out chroma ladder.

Registered (before any model was scored) in benchmarks/byconstruction_2026-10-22.md, "Profile F ... registered
chroma-ladder gate". Population: the diagnostic rows (roles dev, calibration; never fitted) of the stage-C bank, grouped
into ladders by (view, family) with at least 3 kept rungs. A packed bake scores each rung's stored f64 feature row
(`predict_features_with_bake --f64-wire --production`, identity layout); per ladder the statistic is the Spearman rho
between the model score and the teacher over the rungs.

  chroma_ladder_gate.py score  --bank BANK --predict BIN --bake NAME=PATH [--bake ...] --out DIR
  chroma_ladder_gate.py report --out DIR --compare F=E [--compare Fcodec=ECodec]
"""
import argparse
import json
import struct
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

FAMILIES = ["ycc444", "ycc420", "xyb-capped", "xyb-extended", "band-ac"]
HELD_OUT = ("dev", "calibration")
MEDIAN_BAR, SHARE_BAR, SHARE_RHO = 0.90, 0.80, 0.9


def spearman(a, b):
    ra = np.argsort(np.argsort(a, kind="stable"), kind="stable").astype(float)
    rb = np.argsort(np.argsort(b, kind="stable"), kind="stable").astype(float)
    # average ranks for ties
    for x, r in ((a, ra), (b, rb)):
        order = np.argsort(x, kind="stable")
        xs = np.asarray(x)[order]
        i = 0
        while i < len(xs):
            j = i
            while j + 1 < len(xs) and xs[j + 1] == xs[i]:
                j += 1
            if j > i:
                r[order[i:j + 1]] = (i + j) / 2.0
            i = j + 1
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def held_out_rows(bank, roles_wanted=HELD_OUT):
    """(keys table, f64 matrix) of the held-out rows of the wanted roles, in bank order."""
    keys = pq.read_table(Path(bank) / "keys.parquet")
    roles = keys["mlic_role"].to_pylist()
    take = [i for i, r in enumerate(roles) if r in roles_wanted]
    features = pq.ParquetFile(Path(bank) / "features.parquet")
    names = [f"f{i}" for i in range(1825)]
    rows, offset, wanted = [], 0, set(take)
    for batch in features.iter_batches(batch_size=2000, columns=names):
        n = batch.num_rows
        local = [i - offset for i in range(offset, offset + n) if i in wanted]
        if local:
            table = pa.Table.from_batches([batch]).take(pa.array(local))
            rows.append(np.column_stack([table[c].to_numpy() for c in names]).astype("<f8"))
        offset += n
    return keys.take(pa.array(take)), np.vstack(rows)


def score_rows(predict, bake, matrix):
    with tempfile.NamedTemporaryFile(dir=Path.home() / "tmp", suffix=".bin") as f:
        f.write(struct.pack("<II", matrix.shape[1], matrix.shape[0]) + matrix.tobytes())
        f.flush()
        out = subprocess.run([predict, "--bake", bake, "--features-file", f.name, "--f64-wire", "--production"],
                             capture_output=True, text=True, check=True).stdout.split()
    if len(out) != len(matrix):
        raise ValueError(f"{bake}: {len(out)} scores for {len(matrix)} rows")
    return np.array([float(v) for v in out])


def cmd_score(a):
    out = Path(a.out)
    out.mkdir(parents=True)
    keys, matrix = held_out_rows(a.bank, tuple(a.roles.split(",")))
    pq.write_table(keys.select(["pair_key", "view_id", "family", "rung", "content_class", "mlic_role", "teacher_ssim2"]),
                   out / "rows.parquet")
    index = {}
    for spec in a.bake:
        name, path = spec.split("=", 1)
        scores = score_rows(a.predict, path, matrix)
        np.save(out / f"scores_{name}.npy", scores)
        index[name] = path
    (out / "SCORED.json").write_text(json.dumps(dict(rows=len(matrix), bakes=index), indent=1) + "\n")


def ladders(out):
    rows = pq.read_table(Path(out) / "rows.parquet").to_pylist()
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[(r["view_id"], r["family"])].append((int(r["rung"]), i))
    return rows, {k: sorted(v) for k, v in groups.items() if len(v) >= 3}


def per_ladder_rho(out, name):
    rows, groups = ladders(out)
    scores = np.load(Path(out) / f"scores_{name}.npy")
    result = {}
    for key, members in groups.items():
        idx = [i for _, i in members]
        result[key] = (spearman(scores[idx], np.array([float(rows[i]["teacher_ssim2"]) for i in idx])), rows[idx[0]]["content_class"])
    return result


def summarize(rho):
    by_family = defaultdict(list)
    by_class = defaultdict(list)
    for (view, family), (value, cls) in rho.items():
        if np.isfinite(value):
            by_family[family].append(value)
            by_class[(family, cls)].append(value)
    fam = {f: dict(ladders=len(v), median_rho=float(np.median(v)), share_rho_ge_0_9=float(np.mean(np.array(v) >= SHARE_RHO)),
                   passes=bool(np.median(v) >= MEDIAN_BAR and np.mean(np.array(v) >= SHARE_RHO) >= SHARE_BAR))
           for f, v in by_family.items()}
    cls = {f"{f}|{c}": dict(ladders=len(v), median_rho=float(np.median(v)), share_rho_ge_0_9=float(np.mean(np.array(v) >= SHARE_RHO)))
           for (f, c), v in sorted(by_class.items()) if len(v) >= 10}
    return dict(per_family=fam, per_class=cls, passes=all(fam.get(f, {}).get("passes") for f in FAMILIES))


def paired(out, new, old, boot=2000, seed=0):
    a, b = per_ladder_rho(out, new), per_ladder_rho(out, old)
    keys = [k for k in a if k in b and np.isfinite(a[k][0]) and np.isfinite(b[k][0])]
    rng = np.random.default_rng(seed)
    views = sorted({k[0] for k in keys})
    by_view = defaultdict(list)
    for k in keys:
        by_view[k[0]].append(k)

    def mean_diff(selected_views, family=None):
        d = [a[k][0] - b[k][0] for v in selected_views for k in by_view[v] if family is None or k[1] == family]
        return float(np.mean(d)) if d else float("nan")

    def interval(family=None):
        draws = [mean_diff(rng.choice(views, size=len(views)), family) for _ in range(boot)]
        draws = [d for d in draws if np.isfinite(d)]
        return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]

    result = {f: dict(mean_diff=mean_diff(views, f), ci95=interval(f)) for f in FAMILIES if any(k[1] == f for k in keys)}
    pooled = dict(mean_diff=mean_diff(views), ci95=interval())
    chroma_aware = bool(pooled["ci95"][0] > 0 and all(v["ci95"][1] >= 0 for v in result.values()))
    return dict(per_family=result, pooled=pooled, chroma_aware=chroma_aware, ladders=len(keys), views=len(views))


LEVEL_DRAWS, LEVEL_SEED = 10000, 20261029


def level_gate(out, candidate, baseline):
    """per family pooled MAE of candidate and baseline, and the ladder-bootstrap interval of mean(|c-t| - |b-t|)."""
    rows, groups = ladders(out)
    t = np.array([float(r["teacher_ssim2"]) for r in rows])
    c, b = np.load(Path(out) / f"scores_{candidate}.npy"), np.load(Path(out) / f"scores_{baseline}.npy")
    ec, eb = np.abs(c - t), np.abs(b - t)
    by_family = defaultdict(list)
    for (view, family), members in groups.items():
        by_family[family].append(np.array([i for _, i in members]))
    rng = np.random.default_rng(LEVEL_SEED)
    result = {}
    for family in FAMILIES:
        ladders_f = by_family.get(family)
        if not ladders_f:
            continue
        d_sum = np.array([(ec[ix] - eb[ix]).sum() for ix in ladders_f])
        n = np.array([len(ix) for ix in ladders_f])
        draws = rng.integers(0, len(ladders_f), size=(LEVEL_DRAWS, len(ladders_f)))
        boot = d_sum[draws].sum(axis=1) / n[draws].sum(axis=1)
        allrows = np.concatenate(ladders_f)
        lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
        mc, mb = float(ec[allrows].mean()), float(eb[allrows].mean())
        result[family] = dict(ladders=len(ladders_f), rows=int(len(allrows)), mae_candidate=mc, mae_baseline=mb,
                              bias_candidate=float((c - t)[allrows].mean()), bias_baseline=float((b - t)[allrows].mean()),
                              mean_diff=mc - mb, ci95=[lo, hi], passes=bool(mc < mb and hi < 0))
    return dict(per_family=result, passes=all(result.get(f, {}).get("passes") for f in FAMILIES))


def cmd_level(a):
    out = Path(a.out)
    report = dict(registration="benchmarks/byconstruction_2026-10-22.md, G-CHROMA-L (registered 2026-10-29 after round-1 results were seen)",
                  baseline=a.baseline, draws=LEVEL_DRAWS, seed=LEVEL_SEED, candidates={}, also={})
    for cand in a.candidate:
        entry = level_gate(out, cand, a.baseline)
        entry["rank_non_regression"] = summarize(per_ladder_rho(out, cand))
        report["candidates"][cand] = entry
    for other in a.also:
        for cand in a.candidate:
            report["also"][f"{cand}-vs-{other}"] = level_gate(out, cand, other)["per_family"]
    (out / "G_CHROMA_L.json").write_text(json.dumps(report, indent=1) + "\n")
    for cand, e in report["candidates"].items():
        print(cand, "L-PASS" if e["passes"] else "L-FAIL", "rank-ok" if e["rank_non_regression"]["passes"] else "rank-FAIL",
              {f: (round(v["mae_candidate"], 2), round(v["mae_baseline"], 2), round(v["ci95"][1], 2)) for f, v in e["per_family"].items()})


def cmd_report(a):
    out = Path(a.out)
    names = json.loads((out / "SCORED.json").read_text())["bakes"]
    report = dict(registration="benchmarks/byconstruction_2026-10-22.md, Profile F registered chroma-ladder gate",
                  bars=dict(median_rho=MEDIAN_BAR, share_rho_ge_0_9=SHARE_BAR), models={n: summarize(per_ladder_rho(out, n)) for n in names})
    report["comparisons"] = {c: paired(out, *c.split("=")) for c in a.compare}
    (out / "G_CHROMA.json").write_text(json.dumps(report, indent=1) + "\n")
    for n, s in report["models"].items():
        print(n, "PASS" if s["passes"] else "FAIL", {f: round(v["median_rho"], 3) for f, v in s["per_family"].items()})
    for c, v in report["comparisons"].items():
        print(c, "chroma_aware" if v["chroma_aware"] else "not shown", round(v["pooled"]["mean_diff"], 4), v["pooled"]["ci95"])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--bank", required=True)
    s.add_argument("--predict", required=True)
    s.add_argument("--bake", action="append", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--roles", default="dev,calibration", help="selection reads only dev; the final gate reads calibration once")
    s.set_defaults(fn=cmd_score)
    lv = sub.add_parser("level", help="G-CHROMA-L (registered 2026-10-29 after round-1 results were seen)")
    lv.add_argument("--out", required=True)
    lv.add_argument("--baseline", default="E")
    lv.add_argument("--candidate", action="append", required=True)
    lv.add_argument("--also", action="append", default=[], help="reported alongside, not decisive")
    lv.set_defaults(fn=cmd_level)
    r = sub.add_parser("report")
    r.add_argument("--out", required=True)
    r.add_argument("--compare", action="append", default=[])
    r.set_defaults(fn=cmd_report)
    a = p.parse_args()
    a.fn(a)
