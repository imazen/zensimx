#!/usr/bin/env python3
"""HDR evidence panel for packed revision-7 models: native-bank rows forwarded through byc_predict (no re-extraction).

Populations (registration: docs/HDR_TRAINING_PATHS_2026-10-29.md, HDR_P2P3_EXECUTION_2026-10-29.md):
  4kdtb_eval      Regatta EVAL half (48): signed pooled SROCC / raw PLCC against MOS, within-source Spearman
  ladder_eval     consumer-EVAL HDR chroma-ladder rungs (3,245): per-ladder rank agreement with natural SSIMULACRA2,
                  inversion/tie counts (report; pooled score offsets across ladders are never mixed)
  hdrteach_val    registered HDRTEACH VAL (3,900): within-reference Spearman and pooled SROCC vs 10*q_jod (report-only)
  upiq_dev        UPIQ-380 development rows (50): report-only
Uncertainty: whole-family resampling (families = sequences / origins / scenes); undefined correlations stay None.
No fit, no selection: this only reads banks and bakes. HDR-VDP-3 labels: UPIQ is not independent evidence for models trained with them.
"""
import argparse, csv, json, subprocess, sys
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "rev4_featpot"))
from ring_root import bank_transport  # noqa: E402
from hdr_devfold_score import rank, spearman  # noqa: E402

BANKS = Path("/var/tmp/byc-hdr-20261029/banks")
LIB = Path("/mnt/tower/output/zensim/byc-chroma-20261028/hdr-full")


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def populations(width):
    out = {}
    # 4Kdtb EVAL
    keys = pq.read_table(BANKS / "4kdtb-r1-bank/keys.parquet").to_pylist()
    feats = pq.read_table(BANKS / "4kdtb-r1-bank/features.parquet")
    lab = {r["item"]: float(r["score"]) for r in csv.DictReader(open("/mnt/tower/input/iqa/datasets/rousselot-4kdtb/labels.tsv"), delimiter="\t")}
    idx, ref, y, fam = [], [], [], []
    for i, k in enumerate(keys):
        m = json.loads(k["metadata_json"])
        if m["role"] == "EVAL":
            idx.append(i); ref.append("4kdtb:" + m["source_id"]); fam.append(m["family_id"])
            y.append(lab[m["dist_path"].split("/content/")[1]])
    out["4kdtb_eval"] = (feats, idx, ref, y, fam)
    # ladders EVAL
    keys = pq.read_table(BANKS / "ladders-r1-bank/keys.parquet").to_pylist()
    feats = pq.read_table(BANKS / "ladders-r1-bank/features.parquet")
    cells = {(c["view_id"], c["encoded_sha256"]): c for c in pq.read_table(LIB / "admission-r1/CELLS.parquet",
             columns=["view_id", "encoded_sha256", "quality", "natural_ssim2"]).to_pylist()}
    idx, ref, y, fam = [], [], [], []
    for i, k in enumerate(keys):
        if k["consumer_role"] == "eval":
            c = cells[(k["view_id"], k["encoded_sha256"])]
            idx.append(i); ref.append(f"ladder:{k['view_id']}|q{c['quality']}"); y.append(float(c["natural_ssim2"])); fam.append(k["source"])
    out["ladder_eval"] = (feats, idx, ref, y, fam)
    # teacher VAL
    keys = pq.read_table(BANKS / "teacher-val-bank/keys.parquet").to_pylist()
    feats = pq.read_table(BANKS / "teacher-val-bank/features.parquet")
    t = {r["row_id"]: r for r in pq.read_table("/mnt/v/output/zensim/hdrteach-2026-10-04/hdrteach_val.parquet",
         columns=["row_id", "ref_sha256", "source_family", "hdrvdp3_q_jod"]).to_pylist()}
    idx, ref, y, fam = [], [], [], []
    for i, k in enumerate(keys):
        r = t[json.loads(k["metadata_json"])["row_id"]]
        idx.append(i); ref.append("hdrteach:" + r["ref_sha256"][:16]); y.append(10.0 * float(r["hdrvdp3_q_jod"])); fam.append(str(r["source_family"]))
    out["hdrteach_val"] = (feats, idx, ref, y, fam)
    # UPIQ-380 development rows
    keys = pq.read_table(BANKS / "upiq-r2-bank/keys.parquet").to_pylist()
    feats = pq.read_table(BANKS / "upiq-r2-bank/features.parquet")
    jod = {r[0]: float(r[1]) for r in csv.reader(open("/mnt/v/output/zenmetrics/upiq-pu/upiq_cid_jod.csv"))}
    idx, ref, y, fam = [], [], [], []
    for i, k in enumerate(keys):
        if k["split"] == "development":
            idx.append(i); ref.append(f"upiq:{k['dataset']}:{k['content']:02d}"); y.append(100.0 + 10.0 * jod[k["condition_id"]]); fam.append(ref[-1])
    out["upiq_dev"] = (feats, idx, ref, y, fam)
    return out


def metrics(pred, y, ref, fam, boot=2000, seed=0):
    pred, y = np.asarray(pred, float), np.asarray(y, float)
    groups = {}
    for i, r in enumerate(ref):
        groups.setdefault(r, []).append(i)

    def stat(sel):
        rho = [spearman(pred[g], y[g]) for g in groups.values() if len(g) >= 2 and (sel is None or fam_of[g[0]] in sel)]
        rho = [r for r in rho if r is not None]
        return float(np.mean(rho)) if rho else None
    fam_of = list(fam)
    inv = ties = pairs = 0
    for g in groups.values():
        for a in range(len(g)):
            for b in range(a + 1, len(g)):
                if y[g[a]] == y[g[b]]:
                    continue
                pairs += 1
                d = (pred[g[a]] - pred[g[b]]) * (y[g[a]] - y[g[b]])
                inv += d < 0; ties += pred[g[a]] == pred[g[b]]
    return dict(rows=len(y), groups=len(groups), families=len(set(fam)), pooled_srocc=spearman(pred, y), pooled_plcc=pearson(pred, y),
                within_group_spearman_mean=stat(None), pairs=pairs, inversions=int(inv), prediction_ties=int(ties))


def family_bootstrap(preds, y, ref, fam, base, name, boot=2000, seed=0):
    """Whole-family resampling of the paired change in mean within-group Spearman vs the baseline model."""
    fams = sorted(set(fam)); rng = np.random.default_rng(seed)
    if len(fams) < 3:
        return None
    by = {f: [i for i, x in enumerate(fam) if x == f] for f in fams}
    def wg(p, sel):
        groups = {}
        for f in sel:
            for i in by[f]:
                groups.setdefault(ref[i], []).append(i)
        r = [spearman(np.asarray(p)[g], np.asarray(y)[g]) for g in groups.values() if len(g) >= 2]
        r = [x for x in r if x is not None]
        return float(np.mean(r)) if r else None
    ds = []
    for _ in range(boot):
        sel = list(rng.choice(fams, len(fams), replace=True))
        a, b = wg(preds[name], sel), wg(preds[base], sel)
        if a is not None and b is not None:
            ds.append(a - b)
    return dict(n=len(ds), delta_mean=float(np.mean(ds)), ci95=[float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))]) if ds else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="NAME=BAKE:TOKEN (fx1|fx2); the first is the baseline")
    ap.add_argument("--byc-predict", required=True)
    ap.add_argument("--width-root", default="/var/tmp/byc-mixes-20261025/Mlic/root")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    from v2_common import derived_declaration
    width = json.loads((Path(a.width_root) / "wide/main/real/receipt.json").read_text())["width"]
    a.out.mkdir(parents=True, exist_ok=True)
    pops = populations(width)
    models = {m.split("=")[0]: m.split("=")[1].rsplit(":", 1) for m in a.models}
    result, base = {}, a.models[0].split("=")[0]
    for pname, (feats, idx, ref, y, fam) in pops.items():
        tab = bank_transport(feats.take(pa.array(idx)).select([f"f{i}" for i in range(1825)]), ref, y, width)
        table = a.out / f"{pname}.parquet"
        pq.write_table(tab, table, compression="zstd")
        preds = {}
        for name, (bake, token) in models.items():
            pred = a.out / f"pred_{pname}_{name}.tsv"
            subprocess.run([a.byc_predict, "--bake", bake, "--decl", str(derived_declaration(token)[0]), "--corpus", str(table), "--out", str(pred)], check=True)
            preds[name] = np.array([float(l.split("\t")[1]) for l in pred.read_text().splitlines()[1:] if l])
        result[pname] = {n: dict(metrics(p, y, ref, fam), **({} if n == base else dict(delta_within_group_vs_base=family_bootstrap(preds, y, ref, fam, base, n))))
                         for n, p in preds.items()}
    (a.out / "panel.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({p: {n: (m["pooled_srocc"], m["within_group_spearman_mean"]) for n, m in v.items()} for p, v in result.items()}, indent=0))


if __name__ == "__main__":
    main()
