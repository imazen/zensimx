#!/usr/bin/env python3
"""Build the hdrgrid944 cvvdp-mix training leg (944-regime features, same recipe).

A NEW-LINEAGE HDR leg (multi-arm hdrgrid corpus, 30-point q grid, LAN-store fleet
scores) — distinct from the appendix-Q `hdr944-leg` (2026-07-03 zenjxl pairs,
944-regime features). This leg is **944-regime** (Folded720Append2; the hdrfeat944-zenjxl arm's
feature run, exec-zensim944hdr images); label it so and NEVER column-mix with
372/720/924 tables. Same 24,750-row cvvdp∩ssim2 target population as the 372
leg (the overlap is all-zenjxl by tier design), same target mix, same split.

Recipe (appendix-Q target, criterion-2 discipline):
  * source: /mnt/v/output/hdrgrid-2026-08-06/harvest-2026-08-26/{scores,features}.parquet
    (writeback_scores.py two-stage; zenmetrics e461f96d; manifest runs.harvest).
  * rows: cells where BOTH `ssim2_gpu` and `cvvdp_cpu_imazen_v0_1_0` are non-null —
    the score waves' size-tier overlap (24,750 cells, all zenjxl by tier design).
  * target: human_score = 0.5*clip01(ssim2/100) + 0.5*clip01((JOD-6)/4)  (Appendix Q).
  * join: scores x features on the exact 5-field ID (image_path, codec, q,
    knob_tuple_json, encode_sha); duplicate keys are a hard error (join_safety spirit).
  * split: zenmetrics/scripts/picker/origin_split.py::split_of on ref_basename
    (THE split owner; per-rendition digit rule). train+val files written; test
    digits held out entirely (counted in the manifest, not written).
  * schema: ref_basename, human_score, score_cvvdp (JOD), score_ssim2, zensim_score,
    q, knob_tuple_json, encode_sha, codec, f0..f943  (old-leg convention + identity).
  * gate: run check_target_orientation.py --corpus hdrgrid944 on both files BEFORE
    any training use; verdicts land in the leg _MANIFEST.json.

usage: build_hdrgrid372_leg.py [--harvest DIR] [--out DIR]
"""
import argparse, hashlib, json, os, sys, datetime, subprocess
import pyarrow as pa, pyarrow.parquet as pq

sys.path.insert(0, os.path.expanduser("~/work/zen/zenmetrics/scripts/picker"))
from origin_split import split_of  # THE split owner — never a seeded shuffle

def clip01(x): return 0.0 if x < 0 else (1.0 if x > 1 else x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harvest", default="/mnt/v/output/hdrgrid-2026-08-06/harvest-2026-08-26")
    ap.add_argument("--features", default="/mnt/v/output/hdrgrid-2026-08-06/harvest-feat944-2026-08-26/zenjxl/features_folded720append2.parquet")
    ap.add_argument("--out", default="/mnt/v/output/zensim/hdrgrid944-leg")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    ID = ["image_path", "codec", "q", "knob_tuple_json", "encode_sha"]
    s = pq.read_table(f"{a.harvest}/scores.parquet").to_pydict()
    f = pq.read_table(a.features)
    fw = sum(1 for c in f.column_names if c.startswith("feat_"))
    fd = f.to_pydict()

    feat_by_key = {}
    for i in range(len(fd["image_path"])):
        k = tuple(fd[c][i] for c in ID)
        if k in feat_by_key:
            raise SystemExit(f"duplicate feature key {k}")
        feat_by_key[k] = [fd[f"feat_{j}"][i] for j in range(fw)]

    rows = []
    seen = set()
    for i in range(len(s["image_path"])):
        jod = s["cvvdp_cpu_imazen_v0_1_0"][i]
        s2 = s["ssim2_gpu"][i]
        if jod is None or s2 is None:
            continue
        k = tuple(s[c][i] for c in ID)
        if k in seen:
            raise SystemExit(f"duplicate score key {k}")
        seen.add(k)
        feats = feat_by_key.get(k)
        if feats is None:
            raise SystemExit(f"score row without feature row: {k}")
        hs = 0.5 * clip01(s2 / 100.0) + 0.5 * clip01((jod - 6.0) / 4.0)
        rows.append({
            "ref_basename": s["image_path"][i], "human_score": hs,
            "score_cvvdp": jod, "score_ssim2": s2, "zensim_score": s["zensim_score"][i],
            "q": s["q"][i], "knob_tuple_json": s["knob_tuple_json"][i],
            "encode_sha": s["encode_sha"][i], "codec": s["codec"][i],
            "split": split_of(s["image_path"][i]) or "none", "_feats": feats,
        })
    print(f"mix rows: {len(rows)} (feature width {fw})", flush=True)

    base_cols = ["ref_basename", "human_score", "score_cvvdp", "score_ssim2",
                 "zensim_score", "q", "knob_tuple_json", "encode_sha", "codec"]
    out_files = {}
    counts = {}
    for split, fname in (("train", "hdrgrid944_v3mix_traindigits_2026-08-26.parquet"),
                         ("val", "hdrgrid944_v3mix_valdigits_2026-08-26.parquet")):
        sel = [r for r in rows if r["split"] == split]
        counts[split] = len(sel)
        cols = {c: [r[c] for r in sel] for c in base_cols}
        for j in range(fw):
            cols[f"f{j}"] = [r["_feats"][j] for r in sel]
        path = os.path.join(a.out, fname)
        pq.write_table(pa.table(cols), path, compression="zstd")
        h = hashlib.sha256(open(path, "rb").read()).hexdigest()
        out_files[split] = {"path": fname, "rows": len(sel), "sha256": h}
        print(f"WROTE {path}: {len(sel)} rows sha256={h[:16]}…", flush=True)
    counts["test_heldout"] = sum(1 for r in rows if r["split"] == "test")
    counts["unsplittable"] = sum(1 for r in rows if r["split"] == "none")

    zen_commit = subprocess.run(["git", "-C", os.path.expanduser("~/work/zen/zensimx"),
                                 "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True).stdout.strip()
    man = {
        "leg": "hdrgrid944-v3mix (hdrgrid harvest 2026-08-26 + hdrfeat944 zenjxl features, cvvdp-mix target)",
        "date": datetime.date.today().isoformat(),
        "build_commit_zensim": zen_commit,
        "build_commit_zenmetrics_writeback": "e461f96d",
        "front_end": "Folded720Append2 944 HDR route (hdrfeat944-zenjxl arm). "
                     "NEW-REGIME NOTE: 944-regime leg — never column-mix with 372/720/924 tables.",
        "target": "human_score = 0.5*clip01(ssim2_gpu/100) + 0.5*clip01((cvvdp_JOD-6)/4) (Appendix Q formula)",
        "sources": {
            "harvest": a.harvest,
            "harvest_scores_sha256": hashlib.sha256(open(f"{a.harvest}/scores.parquet", "rb").read()).hexdigest(),
            "harvest_features_sha256": hashlib.sha256(open(f"{a.harvest}/features.parquet", "rb").read()).hexdigest(),
            "estate_manifest": "/mnt/v/output/hdrgrid-2026-08-06/_MANIFEST.json (runs.harvest)",
        },
        "files": out_files,
        "counts": counts,
        "coverage_note": "all-zenjxl by score-wave size-tier design (cvvdp∩ssim2 overlap tier)",
        "split_rule": "origin_split.split_of on ref_basename (per-rendition digit rule; test digits held out, not written)",
        "regime_purity": "own dataset; NEVER column-mix with SDR canonicals or 944/924 HDR legs",
        "target_orientation": {"gate": "check_target_orientation.py --corpus hdrgrid372 (in-table, Appendix-Q class)",
                               "status": "PENDING — run before ANY training use"},
    }
    with open(os.path.join(a.out, "_MANIFEST.json"), "w") as fh:
        json.dump(man, fh, indent=1)
    print(f"WROTE {a.out}/_MANIFEST.json", flush=True)

if __name__ == "__main__":
    main()
