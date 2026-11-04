#!/usr/bin/env python3
"""Re-extract an HDR feature plan locally through a chosen front end, in the fleet's shard layout, then assemble the bank.

Registration: benchmarks/e36_subset_test_2026-11-01.md, "HC1" (2026-11-04). The fleet produced the 2026-10-29 banks by
running `hdr944_extract` once per plan view (one reference, its candidates) and publishing a result object with
`features.tsv` + `features.manifest.json` sidecars into a CAS shard with a ledger. This script does the same work on one
host with a new extractor build and `--front-end`, writes a NEW plan directory whose cells carry the new producer pins
and the front end (so `hdr_native_bank.py` binds results to the right controls), and assembles the bank with the
unchanged assembler. No label is read; the plan's membership map is copied verbatim.

    hdr_frontend_reextract.py --plan OLD --out-plan NEW --shard SHARD --bank BANK --extractor BIN --build-commit SHA \
        --front-end cube-root:203 [--jobs 16] [--only-views 0,5,9] [--no-bank]
"""
import argparse, hashlib, json, os, shutil, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
IDS = ",".join(str(i) for i in list(range(228)) + list(range(372, 720)))


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_file(p):
    return sha_bytes(Path(p).read_bytes())


def put_blob(blobs, data: bytes):
    h = sha_bytes(data)
    p = blobs / h
    if not p.exists():
        tmp = blobs / ".staging" / f"{h}.{os.getpid()}"
        tmp.parent.mkdir(exist_ok=True)
        tmp.write_bytes(data)
        os.replace(tmp, p)
    return h


def run_view(vi, view, a, new_knobs, env, shard, blobs):
    cell = view["cell"]
    knobs = json.loads(cell["knob_tuple_json"])
    work = shard / "work" / f"v{vi:05d}"
    work.mkdir(parents=True, exist_ok=True)
    pairs = work / "pairs.tsv"
    with open(pairs, "w") as f:
        f.write("q\tref_path\tdist_path\n")
        for cp in knobs["candidate_paths"]:
            f.write(f"{cell['q']}\t{cell['image_path']}\t{cp}\n")
    out = work / "features.tsv"
    if out.exists():
        out.unlink()
    man_path = work / "features.manifest.json"
    if man_path.exists():
        man_path.unlink()
    cmd = [a.extractor, "--pairs", str(pairs), "--enc-root", str(work), "--ref-root", str(work), "--absolute-pairs",
           "--out", str(out), "--input-contract", knobs["input_contract"], "--era-label", "rev7_rowlanes",
           "--requested-ids", IDS, "--threads", str(a.threads_per_view), "--front-end", a.front_end]
    if any(cp.endswith(".png") for cp in knobs["candidate_paths"]):
        cmd.append("--pq-png-distortions")
    t0 = time.time()
    log = work / "extract.log"
    with open(log, "w") as lf:
        rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env).returncode
    if rc != 0:
        return vi, None, f"rc={rc}: {log}"
    man = json.loads(man_path.read_text())
    if man.get("hdr_front_end") != a.front_end or man["formula_revision"] != "Rev7" or man["research"]["zensim_build_commit"] != a.build_commit:
        return vi, None, f"manifest does not bind the requested producer/front end: {man_path}"
    feat_bytes = out.read_bytes()
    n_rows = feat_bytes.count(b"\n") - 1
    if n_rows != len(knobs["candidates"]):
        return vi, None, f"{n_rows} rows for {len(knobs['candidates'])} candidates"
    feat_sha = put_blob(blobs, feat_bytes)
    man_sha = put_blob(blobs, man_path.read_bytes())
    log_sha = put_blob(blobs, log.read_bytes())
    result = dict(schema="hdr-native-rev7-features-v1", image_path=cell["image_path"], knob_tuple_json=new_knobs[vi],
                  controls=json.loads(new_knobs[vi]), labels_read=False, rows=n_rows,
                  feature_set_id=man["research"]["feature_set_id"], measured_ids=[int(x) for x in IDS.split(",")],
                  feature_sha256=feat_sha, hdr_front_end=a.front_end,
                  sidecars={"features.tsv": feat_sha, "features.manifest.json": man_sha, "extract.log": log_sha})
    res_sha = put_blob(blobs, (json.dumps(result, sort_keys=True) + "\n").encode())
    return vi, res_sha, f"{n_rows} rows in {time.time() - t0:.0f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out-plan", required=True)
    ap.add_argument("--shard", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--extractor", required=True)
    ap.add_argument("--build-commit", required=True)
    ap.add_argument("--front-end", required=True, help="pu21 | cube-root:<nits>")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--threads-per-view", type=int, default=1)
    ap.add_argument("--only-views", default="", help="comma-separated view indices (a control run); the bank is skipped")
    ap.add_argument("--no-bank", action="store_true")
    a = ap.parse_args()
    plan, out_plan, shard = Path(a.plan), Path(a.out_plan), Path(a.shard)
    spec = json.loads((plan / "views.spec.json").read_text())
    old_pm = json.loads((plan / "_MANIFEST.json").read_text())
    extractor_sha = sha_file(a.extractor)
    this_sha = sha_file(__file__)
    new_views, new_knobs = [], {}
    for vi, v in enumerate(spec["views"]):
        knobs = json.loads(v["cell"]["knob_tuple_json"])
        knobs.update(extractor_build_commit=a.build_commit, extractor_sha256=extractor_sha, executor_sha256=this_sha,
                     hdr_front_end=a.front_end)
        kt = json.dumps(knobs, sort_keys=True, separators=(",", ":"))
        new_knobs[vi] = kt
        nv = dict(v); nv["cell"] = dict(v["cell"], knob_tuple_json=kt)
        new_views.append(nv)
    only = [int(x) for x in a.only_views.split(",") if x]
    todo = only or list(range(len(spec["views"])))
    shard.mkdir(parents=True, exist_ok=True)
    blobs = shard / "blobs"; blobs.mkdir(exist_ok=True)
    env = dict(os.environ, ZENSIM_FORMULA_REV="7", RAYON_NUM_THREADS=str(a.threads_per_view))
    ledger, failures = [], []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        futs = [ex.submit(run_view, vi, spec["views"][vi], a, new_knobs, env, shard, blobs) for vi in todo]
        for k, fut in enumerate(futs):
            vi, res_sha, msg = fut.result()
            if res_sha is None:
                failures.append((vi, msg)); print(f"FAILED view {vi}: {msg}", file=sys.stderr, flush=True)
                continue
            c = spec["views"][vi]["cell"]
            ledger.append(dict(job_id=sha_bytes(f"hc1:{a.front_end}:{new_knobs[vi]}".encode()), image_path=c["image_path"], codec=c["codec"],
                               q=str(c["q"]), knob_tuple_json=new_knobs[vi], output_sha=res_sha, status="done", error_class="None",
                               attempts="1", ts=str(int(time.time())), worker=os.uname().nodename, provider="local",
                               kind_json=json.dumps(spec["views"][vi].get("kind", {"kind": "bake", "view": "hdr-native-rev7-features-v1"}))))
            if (k + 1) % 25 == 0 or k + 1 == len(futs):
                print(f"{k + 1}/{len(futs)} views, {time.time() - t0:.0f}s ({msg})", file=sys.stderr, flush=True)
    pq.write_table(pa.Table.from_pylist(ledger), shard / "ledger.parquet")
    if failures:
        print(f"{len(failures)} views failed; no bank", file=sys.stderr); sys.exit(1)
    if only or a.no_bank:
        print(json.dumps(dict(views=len(ledger), shard=str(shard)))); return
    out_plan.mkdir(parents=True, exist_ok=True)
    (out_plan / "views.spec.json").write_text(json.dumps(dict(spec, views=new_views), indent=1) + "\n")
    shutil.copyfile(plan / "membership-feature-map.json", out_plan / "membership-feature-map.json")
    pm = dict(old_pm, derived_from_plan=str(plan), derived_from_plan_manifest_sha256=sha_file(plan / "_MANIFEST.json"),
              spec_sha256=sha_file(out_plan / "views.spec.json"), mapping_sha256=sha_file(out_plan / "membership-feature-map.json"),
              hdr_front_end=a.front_end, extractor_sha256=extractor_sha, extractor_build_commit=a.build_commit, reextract_executor_sha256=this_sha)
    (out_plan / "_MANIFEST.json").write_text(json.dumps(pm, indent=1) + "\n")
    cmd = [sys.executable, str(HERE / "hdr_native_bank.py"), "--plan", str(out_plan), "--shards", str(shard), "--out", a.bank,
           "--build-commit", a.build_commit]
    print(" ".join(cmd), file=sys.stderr)
    subprocess.run(cmd, check=True)
    man = json.loads((Path(a.bank) / "_MANIFEST.json").read_text())
    man["hdr_front_end"] = a.front_end
    (Path(a.bank) / "_MANIFEST.json").write_text(json.dumps(man, indent=2) + "\n")
    print(json.dumps(dict(views=len(ledger), bank=a.bank, rows=man["rows"], front_end=a.front_end)))


if __name__ == "__main__":
    main()
