#!/usr/bin/env python3
"""3-px edge crop vs full frame on the held-out chroma ladders (G-CHROMA population, 23,690 rows / 2,120 ladders).

Per ladder: extract each rung's exact decoded PNG from its EVIF, crop 3 px from every edge of reference and distorted,
score full and cropped with E, ECodec (score_pair_with_bake), the frozen SSIMULACRA2 0.8.2 teacher and butteraugli
(zenmetrics-diagnostics compare). Full-frame scores are controls against the stored teacher bits and G-CHROMA E scores.
Writes one JSON line per row to OUT/rows.jsonl; resumable (skips ladders already written).
  chroma_crop3.py OUT [--jobs N] [--limit-ladders K]
"""
import argparse, json, os, shutil, subprocess, sys, tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
from PIL import Image

HERE = Path(__file__).resolve().parent
BIN = Path(os.environ.get("CROP3_BIN", str(Path.home() / "tmp/crop3/bin")))  # evif, chroma-frozen-teacher, zenmetrics-diagnostics from the chroma archive b/binaries
SPWB = "/var/tmp/zensim-bin-10d087cd/score_pair_with_bake"
CAS = Path("/mnt/tower/output/zensim/byc-chroma-20261028/b/full/cas")
BAKES = {"E": "/mnt/v/output/zensim/mlic-aic-fulldata-xrefaic256e59-20261027-run01/full/XrefAIC256/full_s0/pack_e059/refit/production-f16.bin",
         "ECodec": "/mnt/v/output/zensim/mlic-aic-fulldata-aiconlye29-20261027-run01/full/AICOnly/full_s0/pack_e029/refit/production-f16.bin"}
CROP = 3
ENV = dict(os.environ, RAYON_NUM_THREADS="1", OMP_NUM_THREADS="1")


def run(args, **kw):
    return subprocess.run(args, check=True, capture_output=True, text=True, env=ENV, **kw).stdout


def crop(src, dst):
    im = Image.open(src).convert("RGB")
    w, h = im.size
    im.crop((CROP, CROP, w - CROP, h - CROP)).save(dst)


def butter(ref, dists):
    out = run([str(BIN / "zenmetrics-diagnostics"), "compare", "--reference", str(ref), *sum((["--variant", str(d)] for d in dists), []),
               "--metric", "butteraugli"])
    vals, cur = {}, None
    for line in out.splitlines():
        if " vs " in line and line.rstrip().endswith(":"):
            cur = line.rsplit(" vs ", 1)[1].rstrip()[:-1]
            vals[cur] = {}
        elif cur and line.strip().startswith("butteraugli_"):
            k, v = line.split()
            vals[cur][k] = float(v)
    return [vals[str(d)] for d in dists]


def ladder(job):
    rows, ref_path = job
    tmp = Path(tempfile.mkdtemp(dir=os.environ.get("CROP3_SCRATCH")))
    try:
        refc = tmp / "ref_crop.png"
        crop(ref_path, refc)
        full, cropped = [], []
        for r in rows:
            d = r["evif_sha256"]
            run([str(BIN / "evif"), "extract", str(CAS / d[:2] / d[2:4] / d), str(tmp), "--format", "png"])
            p = tmp / f"{d}.png"
            c = tmp / f"{d}_crop.png"
            crop(p, c)
            full.append(p); cropped.append(c)
        out = []
        pairs = [(ref_path, p) for p in full] + [(refc, c) for c in cropped]
        teach = run([str(BIN / "chroma-frozen-teacher")], input="".join(json.dumps(dict(reference=str(a), distorted=str(b))) + "\n" for a, b in pairs))
        teach = [json.loads(l) for l in teach.splitlines() if l.strip()]
        bf, bc = butter(ref_path, full), butter(refc, cropped)
        for i, r in enumerate(rows):
            rec = dict(pair_key=r["pair_key"], view_id=r["view_id"], family=r["family"], rung=r["rung"], width=r["width"], height=r["height"],
                       content_class=r["content_class"], mlic_role=r["mlic_role"], teacher_stored_bits=r["teacher_bits"])
            for cond, (a, b), t, bu in (("full", pairs[i], teach[i], bf[i]), ("crop", pairs[len(rows) + i], teach[len(rows) + i], bc[i])):
                rec[f"teacher_{cond}"] = t["value"]; rec[f"teacher_{cond}_bits"] = t["value_bits"]
                rec[f"butter_max_{cond}"] = bu["butteraugli_max"]; rec[f"butter_p3_{cond}"] = bu["butteraugli_pnorm3"]
                for name, bake in BAKES.items():
                    rec[f"{name}_{cond}"] = float(run([SPWB, "--bake", bake, "--ref", str(a), "--dist", str(b)]).strip())
            out.append(rec)
        return out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--limit-ladders", type=int)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    keys = pq.read_table("/var/tmp/byc-fchroma-20261029/c/merged-bank/keys.parquet").to_pandas()
    gate = pq.read_table("/var/tmp/byc-fchroma-20261029/gchroma/rows.parquet").to_pandas()
    held = keys[keys.pair_key.isin(set(gate.pair_key))]
    assert len(held) == len(gate) == 23690
    done = set()
    rows_path = a.out / "rows.jsonl"
    if rows_path.exists():
        done = {json.loads(l)["pair_key"] for l in rows_path.read_text().splitlines() if l.strip()}
    jobs = []
    for (view, fam), g in held.groupby(["view_id", "family"], sort=True):
        g = g.sort_values("rung")
        if set(g.pair_key) <= done:
            continue
        jobs.append((g.to_dict("records"), g.image_path.iloc[0]))
    if a.limit_ladders:
        jobs = jobs[: a.limit_ladders]
    print(f"{len(jobs)} ladders to score", flush=True)
    with open(rows_path, "a") as f, ProcessPoolExecutor(a.jobs) as ex:
        futs = [ex.submit(ladder, j) for j in jobs]
        for n, fut in enumerate(as_completed(futs), 1):
            for rec in fut.result():
                f.write(json.dumps(rec) + "\n")
            f.flush()
            if n % 50 == 0:
                print(f"{n}/{len(jobs)} ladders", flush=True)


if __name__ == "__main__":
    main()
