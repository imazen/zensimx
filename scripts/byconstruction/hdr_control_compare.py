#!/usr/bin/env python3
"""HC1 control: a `--front-end pu21` re-extraction of a few plan views must reproduce the stored 2026-10-29 bank rows
bit for bit (the front-end refactor is inert for PU21).

    hdr_control_compare.py --plan PLAN --bank BANK --shard CONTROL_SHARD [--out report.json]

The shard is what hdr_frontend_reextract.py writes with --only-views: for every ledger row, the result object's
features.tsv rows are matched to the bank through the plan's membership map (view_index, feature_row -> row_id) and
compared value by value on the measured ids (parsed float64, exact equality; NaN equals NaN on absent slots).
"""
import argparse, glob, json, math, sys
from pathlib import Path
import pyarrow.parquet as pq

IDS = list(range(228)) + list(range(372, 720))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--shard", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    plan, bank, shard = Path(a.plan), Path(a.bank), Path(a.shard)
    spec = json.loads((plan / "views.spec.json").read_text())
    by_knobs = {v["cell"]["knob_tuple_json"]: i for i, v in enumerate(spec["views"])}
    mapping = json.loads((plan / "membership-feature-map.json").read_text())
    mapping.sort(key=lambda m: (m.get("source_index", 0), m["view_index"], m["feature_row"]))
    row_of = {(m["view_index"], m["feature_row"]): rid for rid, m in enumerate(mapping)}
    feats = pq.read_table(bank / "features.parquet")
    cols = {i: feats[f"f{i}"].to_pylist() for i in IDS}
    blobs = {Path(p).name: Path(p) for p in glob.glob(str(shard / "blobs" / "**"), recursive=True) if Path(p).is_file()}
    report = dict(plan=str(plan), bank=str(bank), views=0, rows=0, mismatches=0, detail=[])
    for led in pq.read_table(shard / "ledger.parquet").to_pylist():
        # the control shard's cells carry new producer pins; match the view by reference + candidates instead
        k = json.loads(led["knob_tuple_json"])
        vi = next(i for i, v in enumerate(spec["views"])
                  if json.loads(v["cell"]["knob_tuple_json"])["reference_sha256"] == k["reference_sha256"]
                  and json.loads(v["cell"]["knob_tuple_json"])["candidates"] == k["candidates"])
        res = json.loads(blobs[led["output_sha"]].read_bytes())
        lines = blobs[res["sidecars"]["features.tsv"]].read_text().splitlines()
        header = lines[0].split("\t"); fcol = {int(h[1:]): j for j, h in enumerate(header) if h.startswith("f") and h[1:].isdigit()}
        report["views"] += 1
        for fr, line in enumerate(lines[1:]):
            vals = line.split("\t")
            rid = row_of[(vi, fr)]
            report["rows"] += 1
            bad = []
            for i in IDS:
                new = float(vals[fcol[i]]) if vals[fcol[i]] not in ("", "NaN", "nan") else math.nan
                old = cols[i][rid]
                if not ((math.isnan(new) and (old is None or math.isnan(old))) or new == old):
                    bad.append((i, old, new))
            if bad:
                report["mismatches"] += 1
                report["detail"].append(dict(view=vi, feature_row=fr, row_id=rid, first=bad[:3], count=len(bad)))
    report["verdict"] = "bit-identical" if report["mismatches"] == 0 and report["rows"] > 0 else "DIFFERS"
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in ("plan", "views", "rows", "mismatches", "verdict")}))
    sys.exit(0 if report["verdict"] == "bit-identical" else 1)


if __name__ == "__main__":
    main()
