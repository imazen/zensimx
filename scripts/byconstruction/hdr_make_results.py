#!/usr/bin/env python3
"""Assemble ~/tmp/handoff/f-hdr/results.json from the run artifacts (no number is typed by hand)."""
import json, hashlib, os, subprocess
from pathlib import Path
D = Path("/var/tmp/byc-hdr-20261029"); OUT = Path.home() / "tmp/handoff/f-hdr/results.json"
def J(p):
    try: return json.loads(Path(p).read_text())
    except Exception: return None
def sha(p):
    try: return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    except Exception: return None
MLIC_E = "/mnt/v/output/zensim/mlic-aic-fulldata-xrefaic256e59-20261027-run01/full/XrefAIC256/full_s0/pack_e059/refit/production-f16.bin"
MLIC_EC = "/mnt/v/output/zensim/mlic-aic-fulldata-aiconlye29-20261027-run01/full/AICOnly/full_s0/pack_e029/refit/production-f16.bin"
models = {
 "E": dict(bake=MLIC_E, role="general baseline (finalized)"),
 "ECodec": dict(bake=MLIC_EC, role="codec baseline (finalized)"),
 "P2base": dict(bake=f"{D}/p2-s05/full/Mhdr2s05/full_s0/refit/production-f16.bin", role="P2 HDR, Mlic recipe seed 0, scale 0.5 (not continued)", gates="p2base"),
 "P2gen": dict(bake=f"{D}/cont/p2-full-xref59/full/XrefAIC256/full_s0/pack_e059/refit/production-f16.bin", role="P2 general: x256+rehearsal e59 continuation", gates="p2gen"),
 "P2codec": dict(bake=f"{D}/cont/p2-full-aic29/full/AICOnly/full_s0/pack_e029/refit/production-f16.bin", role="P2 codec: AIC-only e29 continuation", gates="p2codec"),
 "P3gen": dict(bake=f"{D}/cont/p3b-full-xref59/full/XrefAIC256/full_s0/pack_e059/refit/production-f16.bin", role="P3 general: x256+rehearsal e59 continuation (6-worker plan p3b)", gates="p3gen"),
 "P3codec": dict(bake=f"{D}/cont/p3b-full-aic29-repack/full/AICOnly/full_s0/pack_e029/refit/production-f16.bin", role="P3 codec: AIC-only e29 continuation, no-prune repack (derived-input bake)", gates="p3codec"),
 "P3base": dict(bake=f"{D}/p3-s05/full/Mhdr3s05/full_s0/refit/production-f16.bin", role="P3 (fx2+chroma root + HDR legs), Mlic recipe seed 0, scale 0.5 (not continued)", gates="p3base"),
}
panel = J(D / "panel-all/panel.json")
panel3 = J(D / "panel-p3/panel.json")
for _p in (panel3 or {}):
    for _n in ("P3gen", "P3codec"):
        panel.setdefault(_p, {})[_n] = panel3[_p][_n]
out = dict(generated_utc=subprocess.run(["date", "-u", "+%FT%TZ"], capture_output=True, text=True).stdout.strip(), models={}, selection={}, e21={}, controls={}, incomplete=[])
for n, m in models.items():
    g = m.pop("gates", None)
    e = dict(m, sha256=sha(m["bake"]))
    if g:
        s = (J(D / f"gates/{g}.summary.json") or {}).get(g, {})
        nid = J(D / f"gates/{g}-nearid/summary.json")
        try:
            nid = json.loads(subprocess.run(["python3", "-B", str(Path.home() / "work/zen/zensimx--fhdr/scripts/byconstruction/gates.py"), "--summary", "--out", str(D / f"gates/{g}-nearid"), "--name", g + "n"], capture_output=True, text=True).stdout)[g + "n"]["nearid"]
        except Exception: nid = None
        idl = D / f"gates/{g}/identity-s0.jsonl"
        rows = [json.loads(l) for l in idl.read_text().splitlines() if l.strip()] if idl.exists() else []
        e["gates"] = dict(C1_C6_standard=s.get("standard", {}).get("contract"), C1_C6_ladder=s.get("ladder", {}).get("contract"), G_STEER=s.get("gsteer"),
                          G_STEER_baseline_required="135/135", N1_N2_N3=nid, identity_rows=len(rows), identity_note="62 sources x 12 permutations (e33-identity log)")
    e["hdr_panel"] = {p: panel[p][n] for p in panel if panel and n in panel[p]} if panel else None
    out["models"][n] = e
for p in ("p2", "p3"):
    out["selection"][p] = J(D / f"SELECTION_{p}.json")
for tag in ("p2-s05", "p3-s05", "p2-s10"):
    d = J(D / tag / "scorecard/e21.json")
    if d:
        k = [x for x in d if x != "Mlic"][0]
        out["e21"][tag] = dict(arm=k, paired_with_Mlic=d[k].get("paired_with"), arm_per_dataset_srocc=d[k].get("per_dataset_srocc"), Mlic_per_dataset_srocc=d["Mlic"].get("per_dataset_srocc"),
                               bar="paired mean >= -0.002, each population >= -0.005 (E26 as-good rule)")
for tag in ("p2-s05", "p3-s05"):
    for f in ("mixture", "mixture_global"):
        v = J(D / tag / f"scorecard/{f}.json")
        if v: out["e21"].setdefault(tag + "_" + f, v)
import collections
for pth in ("p2", "p3"):
    f = D / f"cont/{pth}-e21-report/PAIRED_SEEDS.json"
    d = J(f)
    if d:
        agg = collections.defaultdict(dict)
        for k, v in d.items():
            c, ds, sc = k.split("|")
            agg[c][f"{ds}/{sc}"] = {m: dict(mean=v[m]["mean"], seed_unit_se=v[m]["seed_unit_se"]) for m in ("srocc_signed", "plcc_raw", "krocc") if m in v}
        out["e21_continuation_" + pth] = dict(note="paired seed deltas vs the finalized fx1 counterpart; *_full = in-sample full-data seeds, *_oos = omission (out-of-sample) cells; bar: each population >= -0.005", contrasts=agg)
out["controls"] = dict(trainer_sha256=sha(D / "trainer/zensim_mlp_train"), dev_control=J(D / "ctl-v5/CONTROL.json"), p2_s10_v1_control=J(D / "p2-s10/CONTROL.json"),
                       continuation_repro=dict(pass_=(J(D / "cont/repro/CONTROL.json") or {}).get("pass"), checkpoints=len((J(D / "cont/repro/CONTROL.json") or {}).get("comparisons", {}))))
out["incomplete"] = ["P3 pixel mixtures for the base cells (score_pair_with_bake cannot read fx2 research bakes); P2 base mixtures are in e21[p2-s05_mixture*]", "G-CHROMA, HDR-VDC/AVT stored Rev7 features, UPIQ outside the 380 (SDR images), Ghent108 (no EXR-to-nits convention)",
  "G-CHROMA for P3 (definition owned by lane fchroma)"]
OUT.write_text(json.dumps(out, indent=1) + "\n"); print(OUT, OUT.stat().st_size)
