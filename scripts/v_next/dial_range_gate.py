#!/usr/bin/env python3
"""dial_range_gate — the peer-anchored dial gate evaluator (G-GRAN v2
candidate, built 2026-08-28 on user directive: "do all of this, and also
establish addressability expectations at the ends of codec qualities -
also consider distance for jxl, not its quality mapping unless that spline
is truly fair. some codecs are integer qualities and some having meaningful
floating point and we should support both well").

Supersedes webp_ceiling_audit.py (same map-inversion construction,
generalized to both ends + zone entry + gaps + attainability).

Design (each bar carries a PROVENANCE tag; model-derived bars are banned):
- EFFECTIVE LADDER: adjacent grid knobs whose encodes are identical for
  >50% of images (f0 fingerprint) merge into one effective step. jpeg
  q99..100 and q0..10 collapse; webp top collapses to ~1q classes; avif
  only 99.9=100; jxl (float distance) collapses nothing. Ties/gaps/mono
  are computed on the EFFECTIVE ladder only (forced ties on duplicate
  encodes are correct behavior, not defects).
- JXL runs natively on DISTANCE. The q=100-4d display map is MEASURED
  unfair (at display q88 jxl truth is ssim2 75.2 vs jpeg 84.7) and is not
  used for any zone or gate.
- ZONES in peer-truth space [peer-derived]: HF zone entry = lowest
  effective step with peer p50 ssim2 >= 83 (the strictest entry point of
  the legacy q>=88 zones); top = max effective step.
- END + ENTRY CALIBRATION (two-sided) [peer+noise-derived]: per codec,
  per anchor in {bottom, hf_entry, top}: honest = map^-1(peer p50 truth)
  under the bake's own optimal-class monotone translation (loop_proxy.qmap
  fit on all cells); gate |actual - honest| <= tol. Two-sided: catches
  under-reporting AND stretch. tol_top = tol_entry = 1.0 (2x the measured
  cross-map spread ~0.5); tol_bottom derived in-run from the cross-map
  spread at the bottom anchor and printed.
- GAPS [goal-derived]: per image, max |emission step| between adjacent
  effective HF-zone steps. Integer-quantum codecs (jpeg/webp/avif): gate
  p90(per-image max gap) <= 4 (a target midway in a gap g has best-case
  error g/2; the ratified loop tolerance is +/-2). jxl: continuous knob -
  grid gaps are sampling artifacts; reported as diagnostic only (the
  attainability proxy is jxl's gate).
- MONO [convention, unchanged]: p50-curve monotone fraction over effective
  HF steps >= 0.93 (slack -0.05). Tie-rate (|d| <= 0.05) diagnostic.
- ATTAINABILITY [goal-derived, the unifying gate]: seeded-secant proxy
  (loop_proxy.secant_ladder, k=3) per image per integer target across the
  honest range [ceil(honest_entry)..floor(honest_top)]; gate median
  |achieved - target| <= 2. Also reported in translated ssim2 units
  (gaming resistance). NOTE: the proxy quantizes to grid cells, which
  UNDER-estimates jxl's true (continuous-knob) attainability - stated.
"""
import numpy as np, os, sys, json, csv, math
import loop_proxy as lp

Z_HF = 83.0          # peer-derived HF-zone entry truth (ssim2)
TOL_TOP = 1.0        # noise-derived: 2x cross-map spread at top (~0.5 measured)
GAP_BAR = 4.0        # goal-derived: 2x the ratified +/-2 loop tolerance
MONO_BAR = 0.93      # convention (unchanged from G-GRAN v1)
ATTAIN_BAR = 2.0     # goal-derived: the ratified loop tolerance
INT_CODECS = {"jpeg", "webp", "avif"}   # effective integer-quantum knobs

BAKES = {
    "incumbent_s4003": "/mnt/v/output/zensim/bakes/sota944/bakes/W10L9_s4003_packed.bin",
    "A_PH_s4004":      "/mnt/v/output/zensim/bakes/sdr-pure-2026-08-28/W10L9PH_s4004_packed.bin",
    "B_e060":          "/mnt/v/output/zensim/bakes/htraj-2026-08-28/ckpt_epoch060_packed_stamped.bin",
    "w11_s4014_e050":  "/mnt/v/output/zensim/bakes/sdr-pure-2026-08-28/W11J_s4014_ckpts/ckpt_epoch050_s4014_packed.bin",
    "w11_s4012_e080":  "/mnt/v/output/zensim/bakes/sdr-pure-2026-08-28/W11J_s4012_ckpts/ckpt_epoch080_s4012_packed.bin",
    "w11_s4014_final": "/mnt/v/output/zensim/bakes/sdr-pure-2026-08-28/W11J_s4014_ckpts/W11J_s4014_s4014_packed.bin",
}

def main():
    for n, p in BAKES.items():
        assert os.path.exists(p), f"missing bake {n}: {p}"
    X, imgs, codecs, prm = lp.load_grid()
    ss = lp.load_peer("dialgrid_ssim2_gpu.tsv", "ssim2_gpu")
    bcol = [c for c in csv.DictReader(open(f"{lp.REFM}/dialgrid_butteraugli_gpu.tsv"),
            delimiter="\t").fieldnames if "max" in c or "butter" in c][0]
    bt = lp.load_peer("dialgrid_butteraugli_gpu.tsv", bcol, neg=True)
    keys = [(im, "zen" + c, round(float(p), 4)) for im, c, p in zip(imgs, codecs, prm)]
    sv = np.array([ss[k] for k in keys])
    bv = np.array([-bt[k] for k in keys])
    f0 = X[:, 0]

    # ---- effective ladders (dup-merge along each codec's grid ladder) ----
    eff = {}    # codec -> list of classes, each = list of knob values (ascending)
    for c in sorted(set(codecs.tolist())):
        m = codecs == c
        ladder = sorted(set(np.round(prm[m], 4).tolist()))
        classes = [[ladder[0]]]
        for a, b in zip(ladder[:-1], ladder[1:]):
            ia = {im: v for im, v in zip(imgs[m & (np.round(prm, 4) == a)], f0[m & (np.round(prm, 4) == a)])}
            ib = {im: v for im, v in zip(imgs[m & (np.round(prm, 4) == b)], f0[m & (np.round(prm, 4) == b)])}
            common = set(ia) & set(ib)
            dup = common and sum(1 for im in common if ia[im] == ib[im]) > 0.5 * len(common)
            (classes[-1].append(b) if dup else classes.append([b]))
        eff[c] = classes

    def cls_mask(c, cls):
        m = codecs == c
        return m & np.isin(np.round(prm, 4), cls)

    # ---- codec anchors + expectations (bake-independent truth) ----
    anchors, expect = {}, {}
    for c, classes in eff.items():
        p50 = [float(np.median(sv[cls_mask(c, cl)])) for cl in classes]
        entry_i = next(i for i, v in enumerate(p50) if v >= Z_HF)
        anchors[c] = {"bottom": 0, "hf_entry": entry_i, "top": len(classes) - 1}
        expect[c] = {
            "knob_kind": "float-distance (native; q display map measured unfair)" if c == "jxl"
                         else "integer-quantum q (effective ladder)",
            "n_grid_steps": sum(len(cl) for cl in classes),
            "n_effective_steps": len(classes),
            "anchors": {a: {
                "knob": classes[i][-1] if a == "top" else classes[i][0],
                "knobs_in_class": classes[i],
                "ssim2_p50": p50[i],
                "butter_p50": float(np.median(bv[cls_mask(c, classes[i])])),
            } for a, i in anchors[c].items()},
            "hf_zone_truth_span": p50[-1] - p50[entry_i],
        }

    # ---- per-bake evaluation ----
    results = {}
    honest_bottoms = {}
    for name, path in BAKES.items():
        pr = lp.forward(path, X)
        m = lp.qmap(pr, sv, imgs)
        gx = np.linspace(pr.min() - 10, pr.max() + 20, 6001)
        gy = np.asarray(m(gx))
        inv = lambda y: float(gx[min(np.searchsorted(gy, y), len(gx) - 1)])
        R = {}
        for c, classes in eff.items():
            an = {}
            for a, i in anchors[c].items():
                truth = expect[c]["anchors"][a]["ssim2_p50"]
                honest = inv(truth)
                actual = float(np.median(pr[cls_mask(c, classes[i])]))
                an[a] = {"honest": honest, "actual": actual, "delta": actual - honest}
                if a == "bottom":
                    honest_bottoms.setdefault(c, []).append(honest)
            # per-(image, effective step) emissions across the HF zone
            zi = range(anchors[c]["hf_entry"], len(classes))
            emis = {}
            for i in zi:
                mm = cls_mask(c, classes[i])
                for im, v in zip(imgs[mm], pr[mm]):
                    emis.setdefault(im, {})[i] = float(v)
            gaps = [max(abs(d[i + 1] - d[i]) for i in list(d)[:-1]) if len(d) > 1 else 0.0
                    for d in ({im: {i: e[i] for i in sorted(e)} for im, e in emis.items()}).values()]
            p50c = [float(np.median([e[i] for e in emis.values() if i in e])) for i in zi]
            dd = np.diff(p50c)
            mono = float((dd >= -0.05).mean()) if len(dd) else 1.0
            ties = float((np.abs(dd) <= 0.05).mean()) if len(dd) else 0.0
            # attainability proxy over the honest range
            errs, terrs = [], []
            lo_t = math.ceil(an["hf_entry"]["honest"]) + 1
            hi_t = math.floor(an["top"]["honest"])
            cm = codecs == c
            for im in sorted(set(imgs[cm].tolist())):
                mi = cm & (imgs == im)
                order = np.argsort(prm[mi])
                qs, scores = prm[mi][order], pr[mi][order]
                for t in range(lo_t, hi_t + 1, 2):
                    gi = lp.secant_ladder(qs, scores, float(t), 3, lp.SEEDS[c])
                    got = float(scores[gi])   # secant_ladder returns the cell INDEX
                    errs.append(abs(got - t))
                    terrs.append(abs(float(m(got)) - float(m(float(t)))))
            R[c] = {"anchors": an,
                    "gap_p90": float(np.percentile(gaps, 90)) if gaps else 0.0,
                    "mono": mono, "tie_rate": ties,
                    "attain_med_native": float(np.median(errs)) if errs else float("nan"),
                    "attain_med_ssim2": float(np.median(terrs)) if terrs else float("nan"),
                    "n_targets": len(errs)}
        results[name] = R

    tol_bottom = max(3.0, 2.0 * max(float(np.ptp(v)) for v in honest_bottoms.values()))
    # ---- report ----
    print("=== ADDRESSABILITY EXPECTATIONS (bake-independent, peer truth) ===")
    for c, e in expect.items():
        print(f"\n{c} [{e['knob_kind']}]  grid {e['n_grid_steps']} -> effective {e['n_effective_steps']} steps; "
              f"HF-zone truth span {e['hf_zone_truth_span']:.2f} ssim2")
        for a in ("bottom", "hf_entry", "top"):
            d = e["anchors"][a]
            merged = f" (class of {len(d['knobs_in_class'])})" if len(d["knobs_in_class"]) > 1 else ""
            print(f"  {a:9} knob {d['knob']:9.3f}{merged}: ssim2 {d['ssim2_p50']:7.2f}  butter {d['butter_p50']:7.3f}")
    print(f"\ntol_top/entry = {TOL_TOP} [noise-derived]; tol_bottom = {tol_bottom:.2f} "
          f"[noise-derived: 2x cross-map honest-bottom spread]; gap bar {GAP_BAR} [goal-derived]; "
          f"attain bar {ATTAIN_BAR} [goal-derived]; mono {MONO_BAR} [convention]")

    print("\n=== PER-BAKE GATE (two-sided calibration + gaps + mono + attainability) ===")
    for name, R in results.items():
        fails = []
        print(f"\n{name}")
        for c, r in R.items():
            a = r["anchors"]
            for an_name, tol in (("top", TOL_TOP), ("hf_entry", TOL_TOP), ("bottom", tol_bottom)):
                # cap-aware: a bounded [0,100] metric cannot emit above the cap,
                # so the un-reachable portion of `honest` is excused, not failed
                # (the residual under-report inside the scale still gates).
                eff_tol = tol + max(0.0, a[an_name]["honest"] - 100.0) if an_name == "top" else tol
                if abs(a[an_name]["delta"]) > eff_tol:
                    fails.append(f"{c}:{an_name}({a[an_name]['delta']:+.2f})")
            if c in INT_CODECS and r["gap_p90"] > GAP_BAR:
                fails.append(f"{c}:gap({r['gap_p90']:.1f})")
            if r["mono"] < MONO_BAR:
                fails.append(f"{c}:mono({r['mono']:.2f})")
            if r["attain_med_native"] > ATTAIN_BAR:
                fails.append(f"{c}:attain({r['attain_med_native']:.2f})")
            cap_note = f" (cap-excused {max(0.0, a['top']['honest'] - 100.0):.2f})" if a["top"]["honest"] > 100.0 else ""
            gap_note = f"gap_p90 {r['gap_p90']:5.2f}" + ("" if c in INT_CODECS else " (diag: float knob)") + cap_note
            print(f"  {c:5} top {a['top']['delta']:+5.2f} entry {a['hf_entry']['delta']:+5.2f} "
                  f"bottom {a['bottom']['delta']:+6.2f} | {gap_note} mono {r['mono']:.3f} "
                  f"ties {r['tie_rate']:.2f} | attain med {r['attain_med_native']:.2f} "
                  f"(ssim2-units {r['attain_med_ssim2']:.2f}, n={r['n_targets']})")
        print(f"  => {'PASS' if not fails else 'FAIL: ' + ', '.join(fails)}")

    out = {"z_hf": Z_HF, "tol_top": TOL_TOP, "tol_bottom": tol_bottom, "gap_bar": GAP_BAR,
           "attain_bar": ATTAIN_BAR, "mono_bar": MONO_BAR, "expectations": expect,
           "results": results, "bakes": BAKES,
           "provenance": {"top/entry tol": "noise-derived (2x cross-map spread)",
                          "bottom tol": "noise-derived (in-run)",
                          "gap/attain": "goal-derived (ratified +/-2 loop tolerance)",
                          "zones": "peer-derived (ssim2 >= 83 entry)",
                          "mono": "convention"}}
    op = "/home/lilith/work/zen/zensimx/benchmarks/dial_addressability_2026-08-28.json"
    json.dump(out, open(op, "w"), indent=1)
    print(f"\nwrote {op}")

if __name__ == "__main__":
    main()
