#!/usr/bin/env python3
"""dial_grid_924col builder: foldapp-extract the re-encoded dial-grid pairs and
join onto the ORIGINAL 4,817-row identity (image_id, codec, q, codec_param,
param_kind) from dial_grid_720col_2026-07-22. Gate: 4817/4817 matched."""
import csv
import json
import os
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq

D = os.path.expanduser("~/tmp/dial924")
# v2_ab_extract from THIS repo (env override for a custom build); the
# original as-run used a since-deleted sibling worktree — same source.
EXTRACT = os.environ.get("ZM924_BIN") or os.path.expanduser(
    "~/work/zen/zensimx/target/release/examples/v2_ab_extract"
)
OLD = "/mnt/v/output/zensim/v2-eval-720-2026-07-22/dial_grid_720col_2026-07-22.parquet"
OUT = "/mnt/v/output/zensim/v2-eval-924-2026-07-27/dial_grid_924col_2026-07-28.parquet"
FAM = {"zenjpeg": "jpeg", "zenwebp": "webp", "zenavif": "avif", "zenjxl": "jxl"}

feat_by_key = {}
for tag in ("jpeg", "webp", "avif", "jxl"):
    pairs = f"{D}/pairs_{tag}.tsv"
    rows = list(csv.DictReader(open(pairs), delimiter="\t"))
    drv = f"{D}/drv_{tag}.tsv"
    keys = []
    with open(drv, "w") as f:
        f.write("ref_path\tdist_path\thuman_score\n")
        for i, r in enumerate(rows):
            img = os.path.basename(r["image_path"]).rsplit(".", 1)[0]
            fam = FAM[r["codec"]]
            if fam == "jxl":
                d = json.loads(r["knob_tuple_json"]).get("distance")
                key = (img, fam, round(float(d), 3))
            else:
                key = (img, fam, round(float(r["q"]), 3))
            keys.append(key)
            f.write(f"{r['ref_path']}\t{r['dist_path']}\t{i}\n")
    csvp = f"{D}/feat_{tag}.csv"
    env = dict(os.environ, ZENSIM_AB_MODE="foldapp")
    r = subprocess.run(["nice", "-n19", EXTRACT, drv, csvp], env=env,
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"extract {tag} failed: {r.stderr[-300:]}")
    with open(csvp) as f:
        rd = csv.reader(f)
        hdr = next(rd)
        fi = [hdr.index(f"f{i}") for i in range(924)]
        hs = hdr.index("human_score")
        n = 0
        for row in rd:
            k = keys[int(float(row[hs]))]
            feat_by_key[k] = [float(row[j]) for j in fi]
            n += 1
    print(f"{tag}: {len(rows)} pairs -> {n} feature rows", flush=True)

old = pq.read_table(OLD, columns=["image_id", "codec", "q", "codec_param", "param_kind"])
ident = list(zip(*[old.column(c).to_pylist()
                   for c in ("image_id", "codec", "q", "codec_param", "param_kind")]))
missing, out_feats = [], []
for img, codec, q, cp, pk in ident:
    key = (img, codec, round(float(cp), 3)) if pk == "distance" else (img, codec, round(float(q), 3))
    f = feat_by_key.get(key)
    if f is None:
        missing.append(key)
        out_feats.append(None)
    else:
        out_feats.append(f)
print(f"identity rows {len(ident)}; missing {len(missing)}; "
      f"extracted-not-in-identity {len(feat_by_key) - (len(ident) - len(missing))}")
if missing:
    print("first missing:", missing[:6])
    sys.exit(9)
cols = {c: old.column(c) for c in ("image_id", "codec", "q", "codec_param", "param_kind")}
for i in range(924):
    cols[f"f{i}"] = pa.array([r[i] for r in out_feats], pa.float64())
os.makedirs(os.path.dirname(OUT), exist_ok=True)
pq.write_table(pa.table(cols), OUT, compression="zstd")
print(f"G-DIAL: 4817/4817 matched -> {OUT}")
