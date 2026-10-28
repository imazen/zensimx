#!/usr/bin/env python3
"""THE bake metrics-sidecar owner (2026-07-18). Produces a complete, traceable
`<bake>.metrics.json` so bake factors are PERSISTED + keyed, never recomputed into
scratch (the "amnesiac with a sieve" this replaces).

Rigor contract — every sidecar records, keyed on the bake's sha256:
  - provenance : the bake's `.spec.json` (train_corpora / recipe) embedded verbatim
  - eval.panel : the full Mohammadi panel per corpus, FROM bake_verdict --json (the
                 stats owner — never re-implemented here; no-duplication rule)
  - eval.dial  : dial monotonicity + range, FROM bake_verdict --json
  - eval.ood_max : max |raw| on the kadis_negrich severe tail (stability; NOT a stat)
  - eval.corruption_gate : corr<q20 rate on the corr-lq corruption grid (a gate count)
  - eval.diffmap_basic_fraction : |w|/scale mass on the additive basic block (f0..155)
        — the closed-loop-diffmap proxy (1.0 = exact-gradient-capable, basic-only)
  - inputs : sha256 + row count of every eval parquet used (traceability)
  - tool   : this script's git commit + ISO timestamp

Usage:  python3 scripts/v_next/emit_bake_metrics.py <bake.bin> [<bake2.bin> ...]
        python3 scripts/v_next/emit_bake_metrics.py --all <dir>   # every *.bin in dir
Writes <bake>.metrics.json next to each bake. Idempotent (re-run to refresh).

Follow-up (task): migrate ood_max / corruption_gate / diffmap_basic_fraction INTO
bake_verdict --json so there is a single Rust owner and this becomes a thin driver.
"""
import subprocess, sys, json, hashlib, struct, os, glob
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

REPO = Path.home() / "work/zen/zensimx"
BV = str(REPO / "target/release/bake_verdict")
PREDICT = str(REPO / "target/release/predict_features_with_bake")
ZP = str(Path.home() / "work/zen/zenanalyze/target/release/zenpredict")
CORRLQ = Path("/mnt/v/output/zensim/corr-lq")
OOD_PARQUET = CORRLQ / "kadis_negrich_gate.parquet"
CORRUPTION_PARQUET = CORRLQ / "corruption_gate.parquet"
DIAL_GRID = "/mnt/v/output/zensim/eval_panels_2026-05-29/dial_grid_372col_2026-05-29_quarantined_v2.parquet"
CORPORA = "cid22,imazen26,nonphoto,konjnd,aic3,aic4,hf_nearlossless,csiq,pipal,live"
SIDECAR_SCHEMA = "zensim.bake_metrics.v1"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tool_commit():
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip()[:12]
    except Exception:
        return "unknown"


def iso_now():
    return subprocess.run(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], capture_output=True, text=True).stdout.strip()


def score_raw(bake, parquet):
    """max|raw| + the raw vector — reuses the runtime scorer, no stat re-impl."""
    t = pq.read_table(str(parquet))
    n = t.num_rows
    feats = np.stack([np.asarray(t[f"f{i}"], dtype=np.float32) for i in range(372)], axis=1)
    fp = f"/home/lilith/tmp/_mx_{os.getpid()}.bin"
    open(fp, "wb").write(struct.pack("<II", 372, n) + feats.tobytes())
    out = subprocess.run([PREDICT, "--bake", bake, "--bake-post", "raw", "--features-file", fp],
                         capture_output=True, text=True)
    Path(fp).unlink(missing_ok=True)
    return np.array([float(x) for x in out.stdout.split()]), t


def corruption_gate(bake):
    sc, t = score_raw(bake, CORRUPTION_PARQUET)
    refs = [str(x) for x in t["ref_basename"].to_pylist()]
    hs = np.asarray(t["human_score"], dtype=float)
    from collections import defaultdict
    grp = defaultdict(dict)
    for k, (r, h) in enumerate(zip(refs, hs)):
        grp[r]["c" if h < 0.05 else ("q" if h > 0.5 else "m")] = k
    ok = tot = 0
    for r, d in grp.items():
        if "c" in d and "q" in d:
            tot += 1
            ok += sc[d["c"]] < sc[d["q"]]
    return round(ok / max(tot, 1), 4), tot


def closed_loop_properties(bake):
    """The TWO DISTINCT closed-loop-diffmap properties (do NOT conflate them):

      additive        — every layer is `identity`, i.e. score = Σ wᵢ·featureᵢ. Only then is
                        the diffmap the EXACT per-feature spatial gradient (a fixed per-block
                        weight map). An MLP with a LeakyReLU hidden layer is NOT additive: its
                        gradient is input-dependent, so its diffmap is not a fixed per-block map.
      basic_input_only — the model reads only the spatializable basic block f0..155 (per-scale
                        per-channel SSIM/edge/detail/mse/hf). Its diffmap then CANNOT depend on
                        the murky peak/max/masked/IW features (f156..371). This is a real
                        closed-loop advantage, but it is INDEPENDENT of additivity.
      diffmap_basic_fraction — |w|/scale mass on f0..155. DEFINED ONLY for additive bakes
                        (an MLP has no single per-feature weight, so this is None for it).

    Prior bug (fixed 2026-07-18): this returned 1.0 whenever n_inputs≤156, i.e. it reported a
    156-input LeakyReLU MLP (the `Ebothg` winner) as additive/exact-gradient. It is not."""
    r = subprocess.run([ZP, "inspect", bake, "--weights"], capture_output=True, text=True)
    try:
        d = json.loads(r.stdout)
    except Exception:
        return {"additive": None, "basic_input_only": None, "diffmap_basic_fraction": None}
    layers = d.get("layers", [])
    additive = bool(layers) and all(l.get("activation") == "identity" for l in layers)
    n_inputs = d.get("n_inputs") or 999
    basic_input_only = n_inputs <= 156
    frac = None
    if additive and len(layers) == 1:  # truly-linear single map (B, linear-projection bakes)
        lyr = layers[0]
        w = next((np.array(lyr[k], dtype=float) for k in ("weights", "weight", "values")
                  if isinstance(lyr.get(k), list)), None)
        sc = np.array(d.get("scaler_scale", []), dtype=float)
        if w is not None:
            n = w.size
            if n <= 156:
                frac = 1.0
            elif n == 372 and sc.size == 372:
                s = np.abs(w) / np.where(sc != 0, sc, 1e9)
                frac = round(float(s[:156].sum() / s.sum()), 4) if s.sum() > 0 else None
    return {"additive": additive, "basic_input_only": basic_input_only,
            "diffmap_basic_fraction": frac}


def emit(bake):
    bake = str(bake)
    tmp = f"/home/lilith/tmp/_bv_{os.getpid()}.json"
    subprocess.run([BV, "--bake", bake, "--corpora", CORPORA, "--dial-grid", DIAL_GRID, "--json", tmp,
                    "--output", "/dev/null"], capture_output=True, text=True)
    try:
        vj = json.load(open(tmp))
    except Exception as e:
        print(f"  {Path(bake).name}: bake_verdict --json FAILED ({e})"); return None
    Path(tmp).unlink(missing_ok=True)
    ood, _ = score_raw(bake, OOD_PARQUET)
    corr, corr_n = corruption_gate(bake)
    spec_p = Path(bake + ".spec.json")
    spec = json.load(open(spec_p)) if spec_p.exists() else None
    sidecar = {
        "schema": SIDECAR_SCHEMA,
        "bake": bake,
        "bake_sha256": vj["bake_sha256"],
        "n_inputs": vj["n_inputs"],
        "provenance": spec,  # verbatim .spec.json (recipe/train_corpora) or null
        "provenance_present": spec is not None,
        "eval": {
            "panel": vj["corpora"],       # full Mohammadi panel per corpus (from bake_verdict)
            "dial": vj["dial"],           # monotonicity/range (from bake_verdict)
            "ood_max_abs_raw": round(float(np.max(np.abs(ood))), 2),
            "corruption_gate_q20": corr,
            "corruption_n": corr_n,
            "closed_loop": closed_loop_properties(bake),
        },
        "inputs": {
            "dial_grid": {"path": DIAL_GRID, "sha256": sha256_file(DIAL_GRID)[:16]},
            "ood_grid": {"path": str(OOD_PARQUET), "sha256": sha256_file(OOD_PARQUET)[:16]},
            "corruption_grid": {"path": str(CORRUPTION_PARQUET), "sha256": sha256_file(CORRUPTION_PARQUET)[:16]},
        },
        "tool": {"emitter": "scripts/v_next/emit_bake_metrics.py", "commit": tool_commit(), "timestamp": iso_now()},
    }
    out = Path(bake + ".metrics.json")
    out.write_text(json.dumps(sidecar, indent=2))
    cid = next((c["srocc"] for c in vj["corpora"] if c["display"] == "CID22"), None)
    cl = sidecar["eval"]["closed_loop"]
    print(f"  ✓ {Path(bake).name}: CID22 {cid:.4f} dial {vj['dial']['monotonicity']:.3f} "
          f"OOD {sidecar['eval']['ood_max_abs_raw']:.0f} corr {corr*100:.0f}% "
          f"additive={cl['additive']} basic-input={cl['basic_input_only']} prov {spec is not None}")
    return out


def main():
    args = sys.argv[1:]
    if args and args[0] == "--all":
        bakes = sorted(glob.glob(str(Path(args[1]) / "*.bin")))
    else:
        bakes = args
    if not bakes:
        print(__doc__); sys.exit(2)
    print(f"emitting metrics sidecars for {len(bakes)} bakes ...")
    for b in bakes:
        try:
            emit(b)
        except Exception as e:
            print(f"  ✗ {Path(b).name}: {e}")


if __name__ == "__main__":
    main()
