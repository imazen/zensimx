#!/usr/bin/env python3
"""Assemble the UPIQ-380 native revision-7 bank from the verified published CAS objects (no recomputation, no labels)."""
import hashlib, json, math, sys
from pathlib import Path
import pyarrow as pa, pyarrow.csv as pacsv, pyarrow.parquet as pq

IDS = list(range(228)) + list(range(372, 720))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(run, admission, out, build_commit):
    run, out = Path(run), Path(out)
    blobs = {}
    for p in (run / "blobs").rglob("*"):
        if p.is_file() and ".staging" not in p.parts:
            if sha(p) != p.name:
                raise ValueError(f"CAS object does not hash to its name: {p}")
            blobs[p.name] = p
    led = pq.read_table(run / "ledger.parquet").to_pylist()
    if len(led) != 1 or led[0]["status"] != "done":
        raise ValueError("UPIQ ledger must hold exactly one done row")
    res = json.loads(blobs[led[0]["output_sha"]].read_bytes())
    if res["labels_read"] is not False or res["rows"] != 380 or res["controls"]["extractor_build_commit"] != build_commit:
        raise ValueError("UPIQ result contract differs")
    feat = blobs[res["sidecars"]["features.tsv"]]
    man = json.loads(blobs[res["sidecars"]["features.tsv.manifest.json"]].read_bytes())
    if man["formula_revision"] != 7 or man["research"]["zensim_build_commit"] != build_commit:
        raise ValueError("UPIQ producer identity differs")
    t = pacsv.read_csv(feat, parse_options=pacsv.ParseOptions(delimiter="\t"),
                       convert_options=pacsv.ConvertOptions(column_types={"condition_id": pa.string(), **{f"f{i}": pa.float64() for i in range(1825)}}))
    a = json.loads(Path(admission).read_text())
    if t["condition_id"].to_pylist() != [r["condition_id"] for r in a["rows"]]:
        raise ValueError("UPIQ condition order differs from the admission")
    for i in IDS:
        if not all(math.isfinite(x) for x in t[f"f{i}"].to_pylist()):
            raise ValueError("nonfinite measured feature")
    out.mkdir(parents=True, exist_ok=False)
    pq.write_table(pa.Table.from_pylist([dict(row_id=i, **{k: r[k] for k in ("condition_id", "dataset", "content", "split", "pair_key")}) for i, r in enumerate(a["rows"])]), out / "keys.parquet", compression="zstd")
    pq.write_table(t.drop(["condition_id"]).add_column(0, "row_id", pa.array(range(380), pa.int64())), out / "features.parquet", compression="zstd")
    (out / "_MANIFEST.json").write_text(json.dumps(dict(schema="hdr-native-rev7-bank-v1", build_commit=build_commit, rows=380,
        feature_width=1825, measured_ids=IDS, absent_slots="NaN", admission_sha256=sha(admission), keys_sha256=sha(out / "keys.parquet"),
        features_parquet_sha256=sha(out / "features.parquet"), research=man["research"], cas_objects_verified=len(blobs), labels_read=False,
        assembler_sha256=sha(__file__)), indent=2) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:5])
