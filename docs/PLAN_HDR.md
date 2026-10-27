# PLAN_HDR — HDR training + dial-continuity alignment (drafted 2026-07-03)

**Current execution, October 27:** this is a historical recipe record. Follow
[the current HDR/SDR plan](PLAN_HDR_SDR_ALIGNMENT.md#october-27-continuation-and-precision-accounting)
and [active goal](../SESSION-RESUME.md#current-work--october-27-2026).
The July “ship pick” below does not adopt a model for the current work.
Seamless native-precision HDR remains queued after stable candidate adoption;
better-than-PU21 quality and maintained speed both require measurement.

> **STATUS 2026-07-04 — v3 pipeline COMPLETE; ship pick = linear anchored2.**
> Steps 1-5 done (corpus 16.9k cells PU-linear, dedup, splits; w5/w6/w7 MLP
> fans trained; consistency measured ≤0.92pt SDR-anchored). Ship candidate is
> the DETERMINISTIC linear `lp_hdr-lasso0.001-shaped-anchored2` (UPIQ 0.7313 —
> best zensim-family ever; knots to the 92.8 data ceiling), optional +0.041
> residual head validated. MLP HDR heads remain collapse-prone (43.75%) —
> gate-protected only. Gates status: consistency PASS (anchored transfer);
> UPIQ > A (0.7313 vs 0.6933) but < cvvdp 0.758; dial-grid HDR panel still
> to run on an HDR-native grid. Provenance:
> `benchmarks/provenance_best_results_2026-07-04.md`.


Goal: an HDR zensim head (`PreviewV0_5Hdr` slot) trained with the proven
t1dro strategy stack, aligned for dial continuity with the same machinery
that gates the SDR candidate (PCHIP dial spline + densified HDR dial grid +
G1/G3 gates). Everything below is reproducible-by-construction: pinned
inputs, manifests via make_manifest, validate_parquet contracts, DATA_SPLITS
registration in the same commit as first use.

## Ground state (surveyed 2026-07-03)

- **UPIQ baseline reproduced**: 0.694 (task #1 done) — the legacy HDR anchor.
- **AIC-HDR2025: UNOBTAINABLE (user ruling 2026-08-05) — data was never
  released; authors unresponsive; STOP live-checking the repo.** T0 = UPIQ +
  the paper's published numbers, permanently. (Historical: README-only since
  2026-07-03; the ordered-probit loader-swap plan is moot.)
- **HDR codec corpus: FAR further along than the task list said.** The
  76-source rendition grid EXISTS (1,140 PQ-PNGs at
  `/mnt/v/output/imazen-26-hdr-grid-2026-06-14`; user: imazen-26 HDR subset
  + scales/crops). The June encode-half is DONE on R2
  (`datagen-2026-06-23-hdr`: 7,980 zenjxl cells, variants.tar, omni with
  inline ssim2) and cvvdp is scored (7,980 rows local). REMAINING data work
  (launched 2026-07-03): the zensim-372-feature + 4-metric score pass
  (datagen_score_hdr.sh, local GPU) and the near-lossless densification
  top-up (QG 90-100, launched) for intuitive dial curves.
- **Infra already supports HDR end-to-end**: `hetzner_cpu_sweep.sh HDR=1`
  (zenjxl --hdr over PQ-PNG sources, scores cvvdp+butteraugli+ssim2+zensim on
  CPU; needs SWEEP_BIN_OVERRIDE with the `hdr` feature); zenmetrics
  `score/batch --hdr` (EXR/HEIC tmap/UltraHDR → PU path, PR #19); zensim
  PU-XYB feature path.

## Known gap (2026-07-03): CPU metrics lack an HDR PU-shell dispatch

`score-pairs --hdr` on CPU-only builds works for **zensim (CLI-side, +372
features), cvvdp, and iwssim** (verified: hdrhq1 box shards carry their
parquets) — only **ssim2 and dssim** lack a CPU HDR dispatch (error: "metric
'ssim2' is not enabled in this build"; both route through the GPU umbrella
only). Until the wiring fix
lands in zenmetrics, the fleet scores zensim+features and the 4 GPU-only
metrics run on the local 5070 (documented norm exception). The first-pair gate
catches this class on every box.

## Steps (each lands with its artifact; ~4 sessions of work)

1. **Corpus generation (data).** Convert the 76 HDR refs to PQ-PNG
   (hdr-corpus-convert pipeline), stage to R2, run the HDR=1 sweep on cx
   boxes across the q-grid — DENSIFIED where the dial lives (step-1 near-
   lossless + fractional q + jxl-distance), per the sweep-discipline rules
   (q5-q60 density == q60-q100). Persist encodes + all metric variants
   (mandatory). Output: `hdr_sweep_<date>` omni + variants tars.
2. **Features + splits (data).** Extract PU-XYB 372-feature parquets for all
   (ref, encode) pairs; LSD origin splits; register in DATA_SPLITS (T2 train
   + a held-out val group — the kb25/s7 lesson); validate_parquet contracts
   declared in the manifests.
3. **Anchors (data).** (a) Reconstruct AIC-HDR2025 JND scale (reuse the
   SDR25 script; verify response semantics via traps/design agreement as we
   did for KonFiG/SDR25). Register as T0 — never train. (b) UPIQ stays the
   legacy check.
4. **HDR dial grid (instrument).** Densified q-sweep feature grid over HDR
   sources (the SDR dial grid recipe with PU features) → the G1/G3 dial
   panel + the continuity measurement (median adjacent step, flat %,
   backwards % per zone — the 2026-07-03 dial-continuity methodology).
5. **Train (model).** `hdr_t1dro_s{17,7,31}` manifests via make_manifest:
   t1dro stack (ema=0.9, hard_pair 0.5@0.05 with the delta dial per
   deployment, strat=10, dro_eta=0.5) over {hdr groups + the SDR groups for
   shared structure}, held-out HDR val group for selection, dial spline fit
   on an HDR multiband anchor. QAT per the standard packing.
6. **Gates (pre-registered, before looking).** ADDED 2026-07-03 (user
   directive: "once optimal, no regression on SDR including when hdr contains
   only sdr"):
   - **G-HDR-SDR-CONSISTENCY**: SDR-range content fed through the HDR path
     (PU21 shell on nits) must match the SDR path (sRGB u8) — p95 |Δscore|
     ≤ 2 points and pairwise rank preserved on an SDR-range consistency set.
     **MEASURED 2026-07-03** (zenmetrics-api tests/hdr_sdr_consistency.rs):
     the fixed-10k-peak PuRescale seam reaches 6.4pt at heavy distortion
     (rank+identity fine); **SDR-anchored rescale (peak=max(content_peak,
     203)) closes it to ≤0.92pt** — the continuous no-router fix. Adopting it
     is a FEATURE-REGIME change (all HDR features re-extract at the anchored
     transfer → B v2 retrain); B v1 stands on the fixed-anchor regime, which
     is internally consistent.
   - **G-HDR-SDR-PANEL**: the shipped HDR-capable bake regresses NOTHING on
     the standard SDR rank+dial panels vs the SDR ship.
   Original gates: AIC-HDR2025 SROCC ≥ published
   metric baselines from the QoMEX'25 paper; UPIQ ≥ 0.694 baseline; HDR dial
   G1 ≥ 0.95, mono ≥ 0.93, JND-zone median step ≥ 0.5pt, backwards ≤ 3%;
   NO regression on the SDR panel for the SDR ship.
7. **Reproduction kit.** `scripts/reproduce_hdr.sh` mirroring
   reproduce_t1dro51.sh (pinned commit + R2 inputs + sha verification).

## Continuity alignment specifics

The dial spline is fit on the PROJECTED+QUANTIZED net (standard QAT-native
packing); the HDR anchor parquet must span the dial range with per-row
target_score (multiband). Continuity is then MEASURED, not assumed: the HDR
dial grid gives per-zone adjacent-step distributions; the corpus-ridge
lesson (2026-07-03) says eval scatter clustering is sample structure — the
gate is the densified-grid response, and training data must be densified
where the dial needs resolution (step 1 above bakes that in).
