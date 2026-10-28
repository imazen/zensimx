#!/usr/bin/env python3
"""W12-U lodestar battery verdict (frozen protocol, balance_campaign W12-U).
Computes per cell: paired CIs (cid22 vs incumbent AND vs A; validate-hfnl vs A),
LF bots (c22 band<=0.7 vs 0.697; tid<=0.5 vs 0.847 + paired CI per the audit),
LF mono + G-GRAN v1 (dial curves), G-OUT v2 (shelled to the owner), M3a.
Absent axes print as NOT MEASURED, never fail."""
import json, csv, subprocess, struct, sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib"))
from zen_stats import panel_batch_indexed, panel

WD = os.path.expanduser(os.environ.get("W12U_WD", "~/tmp/w12ubat"))
FE = "/mnt/v/output/zensim/reports/fulleval"
OUT = "/mnt/v/output/zensim/bakes/sdr-pure-2026-08-28"
# cells override: a JSON {tag: {"bake":..., "m3a":..., "fe_stem":...}} via $W12U_CELLS
if os.environ.get("W12U_CELLS"):
    _cfg = json.load(open(os.environ["W12U_CELLS"]))
    CELLS = {t: e["bake"] for t, e in _cfg.items()}
    FE_STEM = {t: e["fe_stem"] for t, e in _cfg.items() if e.get("fe_stem")}
    M3A = {t: e["m3a"] for t, e in _cfg.items()}
_DEFAULT_CELLS = {
    "lstar4021_final": f"{OUT}/LSTAR_s4021_packed.bin",
    "lstar4021_e080":  f"{OUT}/LSTAR_s4021_ckpts/ckpt_epoch080_s4021_packed.bin",
    "lstar4022_final": f"{OUT}/LSTAR_s4022_packed.bin",
    "lstar4022_e070":  f"{OUT}/LSTAR_s4022_ckpts/ckpt_epoch070_s4022_packed.bin",
    "lstar4022_e080":  f"{OUT}/LSTAR_s4022_ckpts/ckpt_epoch080_s4022_packed.bin",
    "lstar4023_e070":  f"{OUT}/LSTAR_s4023_ckpts/ckpt_epoch070_s4023_packed.bin",
}
if not os.environ.get("W12U_CELLS"):
    CELLS = _DEFAULT_CELLS
    FE_STEM = {"lstar4021_final": "LSTAR_s4021_packed", "lstar4022_final": "LSTAR_s4022_packed"}
if not os.environ.get("W12U_CELLS"):
    M3A = {"lstar4021_final": 0.838767, "lstar4021_e080": 0.854230, "lstar4022_final": 0.821222,
           "lstar4022_e070": 0.855970, "lstar4022_e080": 0.867652, "lstar4023_e070": 0.817556}
REACH = {"avif": 96.2, "jpeg": 94.4, "jxl": 96.6, "webp": 91.9}

def pp(tag, ax):
    h, p = [], []
    for r in csv.DictReader(open(f"{WD}/pp_{ax}_{tag}.tsv"), delimiter="\t"):
        h.append(float(r["human"])); p.append(float(r["pred"]))
    return np.array(h), np.array(p)

def paired_ci(h, pa, pb, sub=None):
    m = np.ones(len(h), bool) if sub is None else sub
    n = int(m.sum()); rng = np.random.default_rng(11)
    idxs = [rng.integers(0, n, n).tolist() for _ in range(5000)]
    bases = {"h": h[m].tolist(), "a": pa[m].tolist(), "b": pb[m].tolist()}
    jobs = []
    for i, ix in enumerate(idxs):
        jobs += [(f"a{i}", "a", "h", ix), (f"b{i}", "b", "h", ix)]
    res = {r["label"]: r["srocc"] for r in panel_batch_indexed(bases, jobs, stats="srocc")}
    d = np.array([res[f"a{i}"] - res[f"b{i}"] for i in range(5000)])
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

def spline_range(bake):
    o = json.loads(subprocess.run(["/home/lilith/work/zen/zenanalyze/target/release/zenpredict",
        "inspect", bake], capture_output=True, text=True).stdout)
    for m in o["metadata"]:
        if m["key"] == "zentrain.output_calibration_spline":
            b = bytes.fromhex(m["value_hex"]); n = struct.unpack("<I", b[:4])[0]
            pts = [struct.unpack("<ff", b[4+8*i:12+8*i]) for i in range(n)]
            return pts[0][1], pts[-1][1]
    return None

def ggran(fe):
    o = json.load(open(fe)); fails = []
    for c, pts in (o["dial"]["curves"] or {}).items():
        if c not in REACH: continue
        hf = sorted([p for p in pts if p[0] >= 88]); p50 = [p[2] for p in hf]
        if len(p50) < 3: continue
        d = np.diff(p50); mono = float((d >= -0.05).mean())
        if p50[-1] - p50[0] < 8: fails.append(f"{c}:span({p50[-1]-p50[0]:.1f})")
        if p50[-1] < REACH[c] - 1: fails.append(f"{c}:top({p50[-1]-REACH[c]+1:+.2f})")
        if mono < 0.93: fails.append(f"{c}:mono({mono:.2f})")
        lf = sorted([p for p in pts if p[0] < 88]); lp = [p[2] for p in lf]
        if len(lp) > 2 and float((np.diff(lp) >= -0.05).mean()) < 1.0:
            fails.append(f"{c}:LFmono")
    return fails

# references
hc_i, pc_i = pp("incumbent", "cid22"); hc_a, pc_a = pp("A", "cid22")
hv_a, pv_a = pp("A", "hfnlproxy"); ht_i, pt_i = pp("incumbent", "tid")
assert np.allclose(hc_i, hc_a)

# G-OUT: one shelled run over all cells with ranges + peers
ranges = []
for tag, bake in CELLS.items():
    r = spline_range(bake)
    nm = FE_STEM.get(tag, tag)
    if r: ranges += ["--range", f"{nm}={r[0]:.2f}:{r[1]:.2f}"]
gout_cmd = ["/home/lilith/.venvs/pytools/bin/python", "scripts/v_next/outlier_gate.py",
            *ranges]
for p in ("peer_ssim2", "peer_butteraugli", "peer_cvvdp", "peer_iwssim"):
    gout_cmd += ["--peer", f"{FE}/{p}.fulleval.json"]
gout_cmd += [f"{FE}/{FE_STEM.get(t, t)}.fulleval.json" for t in CELLS]
go = subprocess.run(gout_cmd, capture_output=True, text=True, cwd="/home/lilith/work/zen/zensimx")
print("=== G-OUT v2 (owner output; full copy at ~/tmp/w12ubat/gout_full.txt) ===")
open(os.path.expanduser("~/tmp/w12ubat/gout_full.txt"), "w").write(go.stdout)
print("\n".join(l for l in go.stdout.splitlines() if "=>" in l or "kadid" in l))
if go.returncode not in (0, 1): print("G-OUT stderr:", go.stderr[-500:])

print("=== FROZEN BATTERY ===")
print(f"{'cell':<17}{'m3a':>7}{'cid22 vs inc':>22}{'cid22 vs A':>22}{'vhfnl vs A':>22}{'c22bot':>7}{'tidbot':>7}  G-GRANv1")
for tag in CELLS:
    hc, pc = pp(tag, "cid22"); hv, pv = pp(tag, "hfnlproxy"); ht, pt = pp(tag, "tid")
    assert np.allclose(hc, hc_i) and np.allclose(hv, hv_a) and np.allclose(ht, ht_i)
    ci_inc = paired_ci(hc, pc, pc_i); ci_a = paired_ci(hc, pc, pc_a)
    vh_a = paired_ci(hv, pv, pv_a)
    c22bot = panel(pc[hc <= 0.7].tolist(), hc[hc <= 0.7].tolist())["srocc"]
    tidbot = panel(pt[ht <= 0.5].tolist(), ht[ht <= 0.5].tolist())["srocc"]
    gg = ggran(f"{FE}/{FE_STEM.get(tag, tag)}.fulleval.json")
    fmt = lambda c: f"{c[0]:+.4f}[{c[1]:+.4f},{c[2]:+.4f}]"
    print(f"{tag:<17}{M3A[tag]:>7.4f} {fmt(ci_inc):>21} {fmt(ci_a):>21} {fmt(vh_a):>21}"
          f"{c22bot:>7.3f}{tidbot:>7.3f}  {';'.join(gg) or 'PASS'}")
print("\nbars: c22bot>=0.697 tidbot>=0.847; CI clause = not wholly below 0")
