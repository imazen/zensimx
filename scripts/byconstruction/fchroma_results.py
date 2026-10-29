#!/usr/bin/env python3
"""Collect profile F's panel numbers and gate verdicts into one JSON for the coordinator's plots.

Reads finished artifacts only (nothing is recomputed) and writes the JSON to a path outside the repository: it holds
AIC-2026 public_train statistics, which are not committed. Missing artifacts are recorded as "not_run", never dropped.

  fchroma_results.py --out ~/tmp/handoff/f-chroma/results.json
"""
import argparse
import hashlib
import json
from pathlib import Path

V = Path("/mnt/v/output/zensim")
S = Path("/var/tmp/byc-fchroma-20261029")
ST = S / "stage2"
BAKES = Path.home() / "tmp/fchroma-bakes"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.is_file() else "not_run"


def panel(path, labels):
    """per model (seed 0 and the three-seed mean) per dataset: srocc/plcc/krocc/n."""
    rows = load(path)
    if rows == "not_run":
        return rows
    keep = ("n", "srocc", "plcc", "krocc", "srocc_signed", "plcc_raw", "pwrc", "mae")
    out = {}
    for label in labels:
        per = {}
        for r in rows:
            m, s = r["model"].split("|")
            if m == label:
                per.setdefault(r["population"], {})[s] = {k: r["statistics"].get(k) for k in keep}
        out[label] = {pop: dict(seed0=v.get("s0"), mean_of_seeds={k: (sum(v[s][k] for s in v) / len(v) if all(v[s][k] is not None for s in v) else None)
                                                                for k in keep if k != "n"}, seeds=sorted(v))
                      for pop, v in per.items()}
    return out


def gate_block(summary_path, label):
    d = load(summary_path)
    if d == "not_run":
        return d
    d = d.get(label, d)
    return {g: {k: v for k, v in d[g].items() if k.startswith("C") or k == "contract"} for g in ("standard", "ladder") if g in d} | {
        "gsteer": d.get("gsteer"), "nearid": d.get("nearid")}


def gsteer(result_path):
    r = load(result_path)
    if r == "not_run":
        return r
    rows = r["rows"]
    return dict(passed=sum(x["pass"] for x in rows), total=len(rows), failing=[x["key"] for x in rows if not x["pass"]],
                engine_feature_disagreements=sum(x["engine_feature_disagreements"] for x in rows))


def identity(path):
    p = Path(path)
    if not p.is_file():
        return "not_run"
    lines = [json.loads(x) for x in p.read_text().splitlines()]
    rows = [x for x in lines if "reference" in x]
    perms = {x["permutation"] for x in rows}
    failing = [x["reference"] + "|" + x["permutation"] for x in rows
               if x["candidate_score_bits"] != [x["pin_bits"]] and x["candidate_score_bits"] != x["pin_bits"]
               or x["nonzero_direct"] or x["bad_fragility"] or not all(x["candidate_read_set_bits_equal_first_permutation"])]
    return dict(rows=len(rows), sources=len({x["reference"] for x in rows}), permutations=len(perms), failing_rows=len(failing),
                first_failures=failing[:5])


def nearid(path, arm=None):
    d = load(path)
    return d if d == "not_run" else [r for r in d["results"] if arm is None or r["arm"] == arm]


def e21(path):
    d = load(path)
    return d if d == "not_run" else {a: dict(per_dataset_srocc=v["per_dataset_srocc"], per_dataset_per_seed_srocc=v["per_dataset_per_seed_srocc"],
                                              mean_of_four=sum(v["per_dataset_srocc"].values()) / 4) for a, v in d.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    g = ST / "gates"
    pg = V / "mlic-candidate-gates-20261028-run03/out"
    models = {
        "E": dict(role="curated general (Mlic x256+rehearsal, e59), seed 0", path=str(BAKES / "E.bin"), sha256=sha(BAKES / "E.bin")),
        "ECodec": dict(role="curated codec (AIC-only, e29), seed 0", path=str(BAKES / "ECodec.bin"), sha256=sha(BAKES / "ECodec.bin")),
        "F": dict(role="fx2 + chroma teacher leg, general continuation e59, seed 0",
                  path=str(ST / "fx2-full-general-e59-repack/full/XrefAIC256/full_s0/pack_e059/refit/production-f16.bin"), sha256=sha(BAKES / "F.bin")),
        "Fcodec": dict(role="fx2 + chroma teacher leg, codec continuation e29 (no-prune pack), seed 0",
                       path=str(ST / "fx2-full-codec-e29-repack/full/AICOnly/full_s0/pack_e029/refit/production-f16.bin"), sha256=sha(BAKES / "Fcodec.bin")),
    }
    out = dict(
        schema="fchroma-results-v1", note="private: contains AIC-2026 public_train statistics; do not commit",
        models=models,
        in_sample_full_data_panels=dict(
            E=panel(V / "mlic-aic-fulldata-xrefaic256e59-20261027-run01/panels/MODEL_PANELS.json", ["XrefAIC256e59", "Mlic"]),
            ECodec=panel(V / "mlic-aic-fulldata-aiconlye29-20261027-run01/panels/MODEL_PANELS.json", ["AICOnlye29", "Mlic"]),
            F=panel(ST / "fx2-full-general-e59-repack/panels/MODEL_PANELS.json", ["XrefAIC256e59", "Initial", "Mlic"]),
            Fcodec=panel(ST / "fx2-full-codec-e29-repack/panels/MODEL_PANELS.json", ["AICOnlye29", "Initial", "Mlic"]),
            scope="all populations are in sample for full-data models; aic = AIC-2026 public_train (local only)"),
        e21_out_of_sample_diagnostic=dict(
            note="Mlic recipe cells (12 per arm): Mlic = fx1 on the Mlic root (stored), Mlicfx2 = fx2 + chroma leg on the chroma root",
            cells=e21(S / "p1fx2/scorecard/e21.json")),
        gates=dict(
            F=dict(contracts=gate_block(g / "fgeneral.summary.json", "fgeneral") if (g / "fgeneral.summary.json").is_file() else "not_run",
                   gsteer=gsteer(g / "fgeneral/steer135/fgeneral/result.json"),
                   nearid=nearid(ST / "nearid/COMPLETE.json", "FGeneral"),
                   identity_i265_8_permutations=identity(g / "fgeneral-id/identity-s0.jsonl"),
                   identity_dev_12_permutations=identity(g / "fgeneral-id12/identity-s0.jsonl")),
            Fcodec=dict(contracts=gate_block(g / "fcodec.summary.json", "fcodec") if (g / "fcodec.summary.json").is_file() else "not_run",
                        gsteer=gsteer(g / "fcodec/steer135/fcodec/result.json"),
                        nearid=nearid(ST / "nearid/COMPLETE.json", "FCodec"),
                        identity_i265_8_permutations=identity(g / "fcodec-id/identity-s0.jsonl"),
                        identity_dev_12_permutations=identity(g / "fcodec-id12/identity-s0.jsonl")),
            E=dict(contracts=gate_block(pg / "xrefaic256e59.summary.json", "xrefaic256e59"), gsteer=gsteer(pg / "xrefaic256e59/steer135/xrefaic256e59/result.json"),
                   nearid=nearid(Path("/mnt/tower/output/zensim") / "mlic-nearid-xrefaic256e59-20261028-run01/COMPLETE.json")),
            ECodec=dict(contracts=gate_block(pg / "aiconlye29.summary.json", "aiconlye29"), gsteer=gsteer(pg / "aiconlye29/steer135/aiconlye29/result.json"),
                        nearid=nearid(Path("/mnt/tower/output/zensim") / "mlic-nearid-aiconlye29-20261028-run01/COMPLETE.json"))),
        g_chroma=load(S / "gchroma/G_CHROMA.json"),
        g_chroma_secondary_error_vs_teacher=load(S / "gchroma/G_CHROMA_secondary_error.json"),
        continuation_path_control=load(ST / "fx1-reproduction/CONTROL.json") if False else dict(path=str(ST / "fx1-reproduction/CONTROL.json"),
            pass_=load(ST / "fx1-reproduction/CONTROL.json").get("pass") if (ST / "fx1-reproduction/CONTROL.json").is_file() else "not_run"),
        host_controls={h: load(S / "p1fx2" / f"HOSTCTL_{h}.json") for h in ("dev",)} | {"note": "i270, r3800x, i134, r5600g, i265 each reproduced the stored Mlic cell byte for byte; receipts on the hosts (HOSTCTL_<host>.json)"},
    )
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
