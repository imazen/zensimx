#!/usr/bin/env python3
"""UPIQ-380 HDR front-end arms side by side (registration: benchmarks/e36_subset_test_2026-11-01.md,
"HDR front-end experiment", 2026-11-04). No fit, no selection: reads hdr_panel_upiq_all.py outputs.

  hdr_frontend_compare.py --keys <bank>/keys.parquet --jod upiq_cid_jod.csv \
      --arm pu21=<panel dir> --arm cube-root:100=<panel dir> ... --baseline pu21 \
      --models F_s0,F_s1,F_s2,F3_s0,F3_s1,F3_s2,E --rule-models F_s0,F_s1,F_s2 --out <dir>

Per arm x model: pooled SROCC, per-study SROCC (narwaria, korshunov), mean within-reference SROCC.
Paired bootstrap over the 30 references (resampled with replacement, 2,000 draws): arm minus the baseline
front end for the same model, pooled / per-study / within-reference. The registered reading rule is
evaluated for --rule-models: an arm counts as better only if, for every rule model, its pooled and both
per-study deltas are non-negative and the pooled interval lies above zero.
"""
import argparse, csv, json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq


def rank(a):
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    r = np.empty(len(a))
    s = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and s[j + 1] == s[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x, y):
    rx, ry = rank(x), rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def read_pred(panel_dir, model):
    lines = (Path(panel_dir) / f"pred_upiq_all_{model}.tsv").read_text().splitlines()[1:]
    return np.array([float(l.split("\t")[1]) for l in lines if l])


def stats(pred, y, study, ref):
    out = dict(pooled=spearman(pred, y))
    for s in ("narwaria", "korshunov"):
        m = study == s
        out[s] = spearman(pred[m], y[m])
    within = []
    for r in sorted(set(ref)):
        m = ref == r
        rho = spearman(pred[m], y[m])
        if rho is not None:
            within.append(rho)
    out["within_ref"] = float(np.mean(within)) if within else None
    return out


def paired_bootstrap(pa_, pb, y, study, ref, boot, seed):
    refs = sorted(set(ref))
    by = {r: np.flatnonzero(ref == r) for r in refs}
    wa = {r: spearman(pa_[by[r]], y[by[r]]) for r in refs}
    wb = {r: spearman(pb[by[r]], y[by[r]]) for r in refs}
    rng = np.random.default_rng(seed)
    deltas = {k: [] for k in ("pooled", "narwaria", "korshunov", "within_ref")}
    for _ in range(boot):
        pick = rng.choice(refs, len(refs), replace=True)
        idx = np.concatenate([by[r] for r in pick])
        for k in ("pooled", "narwaria", "korshunov"):
            m = np.ones(len(idx), bool) if k == "pooled" else (study[idx] == k)
            a, b = spearman(pa_[idx][m], y[idx][m]), spearman(pb[idx][m], y[idx][m])
            if a is not None and b is not None:
                deltas[k].append(a - b)
        wa_s = [wa[r] for r in pick if wa[r] is not None]
        wb_s = [wb[r] for r in pick if wb[r] is not None]
        if wa_s and wb_s:
            deltas["within_ref"].append(float(np.mean(wa_s)) - float(np.mean(wb_s)))
    return {k: (dict(mean=float(np.mean(v)), lo=float(np.percentile(v, 2.5)), hi=float(np.percentile(v, 97.5)), n=len(v)) if v else None)
            for k, v in deltas.items()}


def fmt(x):
    return "n/a" if x is None else f"{x:.3f}"


def fmt_d(d):
    return "n/a" if d is None else f"{d['mean']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", required=True)
    ap.add_argument("--jod", required=True)
    ap.add_argument("--arm", action="append", required=True, help="NAME=PANEL_DIR")
    ap.add_argument("--baseline", default="pu21")
    ap.add_argument("--models", required=True)
    ap.add_argument("--rule-models", default="")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    keys = pq.read_table(a.keys).to_pylist()
    jod = {r[0]: float(r[1]) for r in csv.reader(open(a.jod))}
    y = np.array([100.0 + 10.0 * jod[k["condition_id"]] for k in keys])
    study = np.array([k["dataset"] for k in keys])
    ref = np.array([f"{k['dataset']}:{k['content']:02d}" for k in keys])
    arms = dict(m.split("=", 1) for m in a.arm)
    models = a.models.split(",")
    rule_models = [m for m in a.rule_models.split(",") if m]
    preds = {arm: {m: read_pred(d, m) for m in models} for arm, d in arms.items()}
    for arm in arms:
        for m in models:
            if len(preds[arm][m]) != len(y):
                raise SystemExit(f"{arm}/{m}: {len(preds[arm][m])} predictions for {len(y)} rows")
    result = {"arms": {}, "deltas_vs_baseline": {}, "baseline": a.baseline, "rows": len(y), "references": len(set(ref))}
    for arm in arms:
        result["arms"][arm] = {m: stats(preds[arm][m], y, study, ref) for m in models}
    for arm in arms:
        if arm == a.baseline:
            continue
        result["deltas_vs_baseline"][arm] = {m: paired_bootstrap(preds[arm][m], preds[a.baseline][m], y, study, ref, a.boot, a.seed) for m in models}
    verdict = {}
    for arm, per in result["deltas_vs_baseline"].items():
        ok = bool(rule_models)
        for m in rule_models:
            d = per[m]
            ok &= all(d[k] is not None and d[k]["mean"] >= 0 for k in ("pooled", "narwaria", "korshunov")) and d["pooled"]["lo"] > 0
        verdict[arm] = "better than the baseline under the registered rule" if ok else "does not beat the baseline under the registered rule"
    result["verdict"] = verdict
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "frontend_compare.json").write_text(json.dumps(result, indent=1) + "\n")
    lines = [f"# UPIQ-380 HDR front-end arms ({len(y)} rows, {len(set(ref))} references; baseline `{a.baseline}`)", "",
             "| arm | model | pooled | Narwaria | Korshunov | within-reference |", "|---|---|---:|---:|---:|---:|"]
    for arm in arms:
        for m in models:
            s = result["arms"][arm][m]
            lines.append(f"| {arm} | {m} | {fmt(s['pooled'])} | {fmt(s['narwaria'])} | {fmt(s['korshunov'])} | {fmt(s['within_ref'])} |")
    lines += ["", f"Paired bootstrap over the references ({a.boot} draws, seed {a.seed}): arm minus `{a.baseline}`, same model.", "",
              "| arm | model | pooled | Narwaria | Korshunov | within-reference |", "|---|---|---:|---:|---:|---:|"]
    for arm, per in result["deltas_vs_baseline"].items():
        for m in models:
            d = per[m]
            lines.append(f"| {arm} | {m} | {fmt_d(d['pooled'])} | {fmt_d(d['narwaria'])} | {fmt_d(d['korshunov'])} | {fmt_d(d['within_ref'])} |")
    if rule_models:
        lines += ["", f"Registered reading rule over {', '.join(rule_models)}:"] + [f"- `{arm}`: {v}" for arm, v in verdict.items()]
    (a.out / "frontend_compare.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
