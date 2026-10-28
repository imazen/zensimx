# Psychovisual learnings for zensim — research-paper synthesis

> **⚠ BANNER 2026-07-18:** §0's status box is from 2026-05-18 (pre-Profile-A/B) — ignore it; see `docs/TOP_MODELS_COOKBOOK.md` for current state. §A's literature synthesis remains valid. This file is tracked in git (its 'gitignored' self-claim is wrong).

*Generated 2026-05-18 from a deep read of ~70 OA papers across six metric families. This file is **gitignored** — it's a working reference, not a spec. Source for everything below is a converted PDF under `/mnt/v/input/papers/<bb>/<blake3>.md` (corpus built by `imazen-private/zenpapers`). A mirror copy lives at `~/work/zen/zenpapers/PSYCHOVISUAL_LEARNINGS_FOR_ZENSIM.md`.*

Read this against `~/work/zen/zensimx/CONTEXT-HANDOFF.md` (V0_7 / V_22-mix ship status) and `~/work/zen/zenanalyze/everything.md` (training-pipeline status).

---

## 0. Where zensim sits today

| Field | Value |
|---|---|
| Current ship | `V_22-mix+konjnd@0.10` (T11.20b candidate) |
| Architecture | 228 → 64 → 1 MLP, ssim2 + IWSSIM + CVVDP supervision |
| CID22 SROCC | 0.7986 ± 0.0084 (clean corpus, post-leak-audit) |
| KonJND-1k SROCC | 0.9219 (after w=0.10 KonJND term) |
| AIC-3 SROCC | ~0.79 |
| Training data | `training_safe_synthetic_perceptual_clean.csv` (~196 k pairs, dedup'd) + unified cross-codec parquets (eval only) |
| Loss | scalar regression, hard-coded ssim2 / cvvdp_log / iwssim_log targets |
| Calibration | none post-bake; raw MLP output ships |
| Color | linear sRGB; no XYB / Oklab front-end |
| Output | one continuous quality scalar; no spatial map, no confidence |

The papers below convergently suggest **six** structural levers we have not yet pulled, and **two** corpus / methodology changes.

---

## A. Evaluation methodology — the 2026 consensus

If §1–§11 are about *what to put inside* zensim, this section is about *how to judge whether the change was an improvement.* It is the only section in this doc that overlaps directly with zensim's CLAUDE.md ("Statistical rigor — mandatory full-stat reporting", "Per-band reporting rule", "SROCC-only verdicts BANNED"). Re-stated here so the doc is self-contained.

### A.1 The reference paper

**Mohammadi, Jenadeleh, Sneyers, Saupe, Ascenso — *Evaluation of Objective IQA Metrics for High-Fidelity Image Compression*** (arXiv 2509.13150, IEEE Access 2025). Two arguments the field accepted quickly: **SROCC alone is the single most misleading practice in IQA evaluation**, and **between-metric ranking requires a paired statistical test** because metrics are correlated through the shared MOS.

Combined with the JPEG **AIC-3 protocol** (Testolina/Saupe 2023, HDR extension Jenadeleh/Mohammadi 2025), these two papers now define how IQA studies are reported.

### A.2 The mandatory statistical panel (per-metric, aggregate + 10-band)

| Stat | What it answers | When it disagrees with SROCC |
|---|---|---|
| **SROCC** (Spearman) | Rank agreement | Default; one of six, not the verdict. |
| **PLCC** (Pearson, on calibrated outputs) | Dial honesty — is a 1-pt metric change a 1-pt MOS change? | Captures calibration error SROCC ignores. |
| **KROCC** (Kendall τ) | Rank agreement, more stable at small n | Sometimes wins over SROCC when n < 100. |
| **OR** (Outlier Ratio) | Fraction of predictions outside ±2σ of MOS | Reveals pathologies SROCC averages over. |
| **PWRC** (Pearson-Weighted Rank Correlation) | Rank correlation weighted by pair importance | Down-weights ties / duplicates SROCC counts as wins. |
| **Z-RMSE** (per-sample σ-normalized RMSE) | Absolute calibration error, weighted by per-stimulus subjective spread | The single most decisive stat when bootstrap σ exists. Example: SSIMULACRA2 SROCC 0.905 vs CVVDP 0.960 *looks* like a 5 % gap; **Z-RMSE 47.6 vs 9.5 reveals the actual scale gap is 5×**. |

`Z-RMSE = √((1/n) · Σ ((Ŝᵢ − μᵢ) / σᵢ)²)` where `μᵢ, σᵢ` are bootstrap MOS stats per stimulus. Penalizes errors *less* where humans disagreed, *more* where the JND is sharp.

### A.3 Paired-significance tests (between two metrics on shared MOS)

| Test | Use | Report |
|---|---|---|
| **Meng-Rosenthal-Rubin (MRR)** | Paired SROCC delta between metrics A and B sharing the same MOS targets | z-statistic + p-value (never just "p < 0.05") |
| **Wilcoxon signed-rank on residuals** | Non-parametric companion to MRR | p-value + effect size `r = Z/√N` |
| **Bootstrap 95 % CI**, 1000 resamples, percentile method | Required for every per-metric correlation AND every between-metric delta | CI half-width; resample source images, not rows (zenmetrics `docs/EVALUATION_METHODS.md`, in review as zenmetrics PR #67). The often-quoted 0.27 JND is the mean 95 % CI *width* at 1 JND in the AIC-HDR2025 study (arXiv:2506.12505), not an AIC-3 standard |

A bare claim "metric A beats B on SROCC" is no longer publishable; you need the MRR delta with effect size and the CI overlap status.

### A.4 Subjective-data collection — JPEG AIC-3 protocol (when generating new MOS)

If you're collecting fresh subjective data in the q ≥ 80 / NVL range, the only protocol that survives peer review:

1. **Plain triplet comparisons (PTC)** + **boosted triplet comparisons (BTC)** with three boosters: 10 Hz flicker, 2× spatial zoom, 2× artifact amplification.
2. **Thurstone Case V** maximum-likelihood reconstruction → continuous JND scale, with `d · Φ⁻¹(0.75) = 1` → `d ≈ 0.6745`. 1 JND = 75 % detection probability.
3. **Otsu batch-screening thresholds** ≈ 0.66 (PTC), 0.70 (BTC) on per-subject accuracy + consistency.
4. **Bootstrap σ per stimulus**, kept alongside the MOS so consumers can compute Z-RMSE.

Below the AIC-3 range, DSCQS / ACR / pair-comparison still apply, but JOD-aligned aggregation via Thurstone is the default — single-lab MOS averaging is no longer trusted across studies.

### A.5 Validation-corpus hierarchy

| Tier | Corpus | Role | Why |
|---|---|---|---|
| Gold | **CID22** (Sneyers/Ben Baruch/Vaxman 2023, JPEG WG1 wg1m99012) | Compression-specific MOS, 49 refs held out | Only large MOS dataset that exercises *codec* artifacts |
| Gold | **AIC-3** (Testolina 2023) + **AIC-HDR2025** + **AIC-JPEG-AI SDR 2025** | High-fidelity JND scale, codec-rich (incl. learning codecs) | Fine-grained JND-calibrated; designed for q ≥ 80 |
| Anchor | **KonJND-1k** (1008 images, PJND thresholds) + **KonJND++** (300 sources × 129 click maps) | Imperceptibility threshold calibration; spatial JND map | Different psychometric scale than MOS — orthogonal check |
| Anchor | **UPIQ** (Mikhailiuk et al. 2022, ~4000 images) | Unified SDR + HDR JOD scale | Recalibration anchor for HDR-VDP-3 and CVVDP |
| Integrity | **KADID-10k**, **TID2013**, **CSIQ**, **LIVE** | Distortion-diversity check | Catch metric collapses on non-compression artifacts; **not for ranking** — distortions are mostly synthetic blur/noise/color |

**The 2026 anti-pattern:** ranking metrics on KADID/TID alone, or on aggregate `(KADID + TID + CID22)/3` SROCC. Mohammadi 2025 is explicit — non-compression synthetic distortions can flip metric rankings versus codec output, so they're integrity guards, not gold standards.

### A.6 Reporting protocol — three rules

1. **10 bands, not 4.** Width-10 bins B0..B9 over the MCOS / SSIMULACRA scale. The legacy 4-band CID22 Table 5 cuts are reported alongside *only* for paper-comparison. Aggregate stats hide band-specific failures; product decisions live in B0..B5.
2. **Full panel per band** (all 6 stats from §A.2), not just SROCC. A metric can win SROCC on a band while losing Z-RMSE (miscalibration) or losing PWRC (wins on duplicates that don't matter). The joint pattern is the signal.
3. **MRR-paired delta with effect size** when claiming "A is better than B." Bare PLCC/SROCC deltas are not publishable.

For per-band tables, flag bands with `n < 30` as "noisy estimate" — the CI half-width exceeds 0.3 SROCC at that n and ranking between bakes is statistically indistinguishable.

### A.7 What's actually the best metric (2026 benchmarks)

From the two flagship 2025 datasets — **AIC-HDR2025** (34,560 ratings × 151 subjects × 4 codecs) and **AIC-JPEG-AI SDR 2025** (96,200 ratings × 459 subjects × 6 codecs including JPEG AI):

| Rank | Metric | PLCC range | When to use |
|---|---|---|---|
| **1st overall** | **ColorVideoVDP (cvvdp)** | **0.936 – 0.968** | Default best general-purpose perceptual metric. Wins PLCC, SROCC, and Z-RMSE on every codec including JPEG AI. Free, MIT, GPU (`gfxdisp/ColorVideoVDP`). |
| 2nd | **HDR-VDP-3** | 0.919 – 0.963 | Statistically tied with cvvdp on HDR. Use when inputs are absolute-luminance / display-referred. |
| 3rd (per-source) | **SSIMULACRA2** | 0.906 overall, **0.968 per-source** | Strongest *per-source* PLCC of any non-VDP metric. Designed for codec artifacts; slightly under-sensitive to chroma-only distortions. |
| 4th (learning codecs) | **VMAF-neg** | 0.909 – 0.958 | The best metric for *learning-based* codec outputs (JPEG AI, learned compression). Worse than SSIMULACRA2 on traditional codecs. |
| 5th | **Butteraugli p-norm** | 0.882 – 0.910 | Stable across codecs; parent of SSIMULACRA2. |
| Below the floor | PSNR-HVS, MS-SSIM, IW-SSIM, PSNR | 0.6 – 0.85 | Integrity baselines only — not ranking targets. |

**The 2026 best-practice metric *stack* is three metrics, not one:**
- **cvvdp** + **SSIMULACRA2** + **VMAF-neg**
- Reported with the full Mohammadi panel per band per corpus
- Paired MRR tests + 95 % bootstrap CIs when claiming a winner

That stack is the dial against which zensim is being measured.

### A.8 Two surfacing notes for zensim specifically

- The CLAUDE.md per-band-panel requirement extending `dataset_metric_baseline` to emit the full 6-stat panel per band (not just SROCC + CI) is the one piece of tooling that gates every evaluation in this doc. Until it lands, every "champion" claim is provisional — the CLAUDE.md says exactly this.
- The single highest-yield change in §8 (EX-1, Thurstone loss replacement) only pays off if A.6 reporting is honest. A Thurstone-trained bake will *lose* SROCC on whatever metric the prior bake was supervised against (because it's no longer ssim2-shaped), but should *win* PWRC and Z-RMSE on independent corpora. Without the full panel, the change looks like a regression.

### A.9 The decisive comparison formula and rule

There is no single closed-form scalar that decides "metric A beats metric B for compression in band X" — Mohammadi 2025's argument is precisely that you need multi-stat agreement. But there *is* a closest-thing-to-a-formula, and a decisive *rule* on top of it.

#### The formula: MRR z-test on SROCC (and on Z-RMSE)

**Meng-Rosenthal-Rubin (MRR) paired-correlation test**, computed per band:

```
z_A  = atanh(SROCC_A)                # Fisher z-transform of A's SROCC vs MOS
z_B  = atanh(SROCC_B)
r_AB = SROCC(A_scores, B_scores)     # how A and B agree with each other
r_AZ = SROCC_A                       # A vs MOS
r_BZ = SROCC_B                       # B vs MOS

f    = (1 − r_AB) / (2 · (1 − r_AZ²) · (1 − r_BZ²))

h_SROCC = (z_A − z_B) · √(n_band − 3) / √(2 · (1 − r_AB) · f)
```

`h_SROCC` is distributed as a standard normal under the null hypothesis that A and B are equally correlated with MOS. **|h| > 1.96 ⇒ p < 0.05** that A ≠ B in this band; sign of h tells you which is better.

**For image compression specifically, also compute MRR on Z-RMSE.** Replace `SROCC_*` with `1 − Z-RMSE_* / σ_max` inside the same MRR machinery. Z-RMSE is the load-bearing stat for compression because it weights errors by the per-stimulus MOS uncertainty captured by AIC-3's bootstrap σ. Mohammadi's example: SROCC says SSIMULACRA2 and cvvdp differ by 5 %; Z-RMSE says they differ by 5×. MRR-on-Z-RMSE catches that.

#### The decisive rule

A single MRR `h > 1.96` is necessary, not sufficient — a metric can win SROCC while losing calibration. The field-converged rule:

```
DECISIVE_FOR_BAND(A beats B) ⟺
    n_band ≥ 30                                            # not "noisy"
  ∧ |h_SROCC(A, B; band)| > 1.96       ∧ h_SROCC > 0       # SROCC significant in A's favor
  ∧ |h_Z-RMSE(A, B; band)| > 1.96      ∧ h_Z-RMSE > 0      # calibration significant in A's favor
  ∧ PWRC_A > PWRC_B                                        # weighted rank agrees
  ∧ 95 % bootstrap CIs of (A − B) on ≥ 4 of {SROCC, PLCC,
      KROCC, OR, PWRC, Z-RMSE} exclude zero in A's favor   # multi-stat agreement
```

**A decisively beats B in a band when** the MRR test is significant on *both* SROCC and Z-RMSE in A's favor, **and** at least four of the six panel stats agree, **and** PWRC moves in the right direction, **and** the band has at least 30 samples. Anything weaker — single-stat win, n < 30, bootstrap CIs that overlap — is "promising, not decisive."

#### The "least-bad scalar" if forced to compress to one number

```
DecisiveScore(A vs B; band) =
      h_SROCC
    · sign(Z-RMSE_B − Z-RMSE_A)          # +1 if A better calibrated
    · sign(PWRC_A − PWRC_B)              # +1 if A weighted-rank wins
    · agreement_fraction ∈ [0, 1]        # fraction of 6 panel stats favoring A
    · √min(n_band, 100) / 10             # n-confidence damping
```

Practical decisive cutoff: `|DecisiveScore| > 1.96 · 4 · 1 ≈ 7.84` (MRR significance × full agreement × full n).

**Do not use the scalar without the rule above as a sanity check.** The reason the field uses a rule and not a scalar is that the joint pattern matters — a high MRR z that comes with bad Z-RMSE is a calibration disaster pretending to be a win, and a scalar collapses that information.

#### Workflow for any V_X comparison

When deciding whether bake A beats bake B in band B3:

1. Compute the full Mohammadi panel (SROCC, PLCC, KROCC, OR, PWRC, Z-RMSE) for both bakes against the same held-out per-band slice (CID22 + AIC-3 + KonJND-1k).
2. Compute `r_AB` (the SROCC between the two bakes' outputs themselves) — needed for MRR.
3. Compute `h_SROCC` and `h_Z-RMSE`.
4. 1000-resample bootstrap CIs of every (A − B) panel-stat delta.
5. Apply the rule above. Record DecisiveScore + the rule's truth value side by side in the methodology doc.

If `h_SROCC > 1.96` but `Z-RMSE_A > Z-RMSE_B`, A is shaped wrong even though it ranks images correctly — that's the V_20 input-shaping failure mode (per CLAUDE.md "V_20 input-shaping is a B3 specialist, not an aggregate win"). A wins the rank test, A loses the calibration test, A is not decisively better — flag for user judgement.

#### Reference implementation sketch (Rust)

```rust
fn mrr_h(
    srocc_a: f64,   // metric A vs MOS
    srocc_b: f64,   // metric B vs MOS
    r_ab:    f64,   // SROCC between A and B themselves
    n_band:  usize,
) -> f64 {
    let za = srocc_a.atanh();
    let zb = srocc_b.atanh();
    let denom = (1.0 - srocc_a.powi(2)) * (1.0 - srocc_b.powi(2));
    let f = (1.0 - r_ab) / (2.0 * denom);
    let scale = ((n_band - 3) as f64).sqrt() / (2.0 * (1.0 - r_ab) * f).sqrt();
    (za - zb) * scale
}

fn decisive(a: &PanelStats, b: &PanelStats, n_band: usize) -> Decision {
    if n_band < 30 { return Decision::Noisy; }
    let h_s = mrr_h(a.srocc, b.srocc, a.r_ab, n_band);
    let h_z = mrr_h(1.0 - a.z_rmse / a.z_rmse_max,
                    1.0 - b.z_rmse / b.z_rmse_max,
                    a.r_ab, n_band);
    let pwrc_ok = a.pwrc > b.pwrc;
    let panel_agreement = a.panel_wins_vs(b);  // count of 6 where CI excludes 0
    if h_s > 1.96 && h_z > 1.96 && pwrc_ok && panel_agreement >= 4 {
        Decision::ADecisivelyBeatsB
    } else if h_s < -1.96 && h_z < -1.96 && !pwrc_ok && panel_agreement <= 2 {
        Decision::BDecisivelyBeatsA
    } else {
        Decision::PromisingNotDecisive
    }
}
```

This is the function that should live in `zensim-validate` once `dataset_metric_baseline` emits the full panel per band.

### A.10 Trail-gate composition: dual SOTA tracks over A.9 verdicts (added 2026-05-18)

A.9 returns a per-(corpus, bake-pair) decisive verdict. The **ship gate** for a candidate composes those verdicts across the 5 eval corpora (CID22 + KADID + TID + KonJND + AIC-3). As of 2026-05-18 there are TWO parallel ship gates, each a different composition rule. A candidate can ship on either; the same candidate may be balanced-fail and compression-pass (e.g., V_22-372feat 2026-05-18: balanced gate fails 2/5, compression gate passes decisively on CID22 + AIC-3).

**Why two trails:** zensim's product is web compression — every byte matters at q5–q40 (per CLAUDE.md "Imageflow and related work is web-focused, not archival"). The balanced gate (Pareto across all 5 corpora) is the research-paper convention; the compression gate weights CID22 + AIC-3 because those are the compression-band human-MOS corpora and the product priority.

**Balanced gate** (status quo, governs `PreviewV0_NBalanced`): A ships when

```
∃ corpus c ∈ {CID22, KADID, TID, KonJND, AIC-3}:
       A.9(A, B; c) = ADecisivelyBeatsB
  ∧  ∀ c': A.9(A, B; c') ≠ BDecisivelyBeatsA
  ∧  promising-not-decisive on remaining corpora is acceptable
```

**Compression gate** (NEW 2026-05-18, governs `PreviewV0_NCompression`): A ships when

```
∃ c ∈ {CID22, AIC-3}:
       A.9(A, B; c) = ADecisivelyBeatsB
  ∧  A.9(A, B; c') ≠ BDecisivelyBeatsA   for c' = {CID22, AIC-3} \ {c}
  ∧  mean Δ on each of {KADID, TID, KonJND} ≥ −0.10
```

Note the compression gate trades A.9-strictness on synthetic-distortion corpora for A.9-strictness on the compression corpora, while keeping A.9 verdicts as the underlying decision unit on CID22 + AIC-3. The mean-level tolerance on {KADID, TID, KonJND} is intentionally loose — those are not the product corpora; a −0.05 mean SROCC delta there is acceptable if it buys ADecisivelyBeatsB on a compression corpus.

**A.9 itself is unchanged.** The same per-corpus MRR-on-SROCC + MRR-on-Z-RMSE + panel-agreement machinery powers both gates; the trails differ only in which corpora the gate consumes verdicts from and what tolerance applies to the rest. `bake_compare` emits all 5 corpora; the gate logic is a thin wrapper that classifies the verdict tuple.

**Tracker:** every candidate logged in `SOTA_TRAILS.md` with verdict per corpus per trail. Profile variants `PreviewV0_NBalanced` and `PreviewV0_NCompression` ship independently — callers pick by product context.

**Controls in every table:** every SOTA table MUST include ssim2 and cvvdp baseline rows. Bake-vs-bake numbers without those references are meaningless ("0.83 CID22" reads differently if ssim2 is at 0.89 vs 0.81). The control rows anchor the absolute scale and let the user judge how much headroom remains.

---

## 1. Color front-ends — XYB, LMS-biased-log, PU encoding

### Findings

- **XYB** (JPEG XL): cube-root transfer applied to an LMS-like matrix; designed so that perceptually similar colors are Euclidean-close. Used for both encoding and as the front-end of the *encoder's* internal Butteraugli loss.
- **Butteraugli's own front-end is LMS with a *biased-log* transfer**, *not* XYB — "similar but not identical" per Alakuijala et al. (Guetzli paper). The biased-log keeps low-luma sensitivity high while compressing high-luma gracefully.
- **PU encoding** (Aydın 2008; HDR-VDP-2/3): maps absolute cd/m² to a perceptually-uniform value via a contrast-sensitivity-derived curve. Without PU, *every* SDR-trained metric saturates or inverts on HDR (HDR-VDP-2 paper: PSNR on raw luminance "essentially uncorrelated"). PU encoding is valid 0.1–80 cd/m² as constrained to mimic sRGB inverse-gamma; extends linearly through HDR range.
- **Oklab** is mentioned in several modern IQA-loss formulations as a fast XYB substitute, especially for *chroma* terms; not psycho-tuned but cheap and perceptually-uniform-enough for color-difference work.

### What zensim could do

1. **Stage a non-linear color front-end before the 228-feature extractor.** Two options, both ~2 days of work:
   - *XYB* (libjxl-style): one 3×3 matrix mul + cube-root per pixel. Matches the codec-side color we already accept (JXL VarDCT operates in XYB).
   - *LMS biased-log* (Butteraugli-style): same 3×3 + biased-log `f(x) = log(x + b)`, with `b ≈ 0.01`. Used by every Butteraugli/SSIMULACRA2/Guetzli/Jpegli system in our corpus.
2. **Add a PU-encoded path** behind a `--hdr` feature flag once the SDR pipeline is solid. The two paths share everything but the front-end. PU source: `https://goo.gl/rpfkB9` (linked in Aydın 2008 paper).
3. **Per-channel masking weights:** Butteraugli reports separate low-/high-frequency masks per chroma channel — color sensitivity is wildly non-uniform (e.g., yellow-on-blue ≪ blue-on-yellow). Worth lifting the masking exponent (paper reports `~0.97` but in a hand-tuned regime) as initialization for our learned per-channel weights.

### Constants to lift verbatim

| Constant | Value | Source |
|---|---|---|
| XYB biased-log offset | `b ≈ 0.01` | libjxl `enc_xyb.cc`; Guetzli paper |
| Butteraugli "slack" α | 0.97 | Guetzli §3.1 |
| Butteraugli viewing-distance assumption | 1000 pixels | Butteraugli code & Guetzli paper |
| PU encoding valid range | 0.1–80 cd/m² (SDR), extended linearly | Aydın 2008 |

---

## 2. Multi-scale decomposition — Laplacian, Gabor, steerable, dilated

### Findings

- **HDR-VDP-2/3 + ColorVideoVDP**: steerable pyramid with multi-orientation filters at 5 scales, then a CSF-shaped weighting per band. CVVDP additionally uses transient/sustained channels for video; for stills the sustained channel is what we want.
- **FovVideoVDP**: same as CVVDP + an eccentricity weight (we can drop this — zensim is foveal only).
- **Butteraugli**: Laplacian pyramid + rotated line kernels for edge detection. Computed on original *and* a 2× downsampled copy.
- **NoR-VDPNet**: U-Net with **dilated dense blocks** (dilation = 3) as a cheap substitute for a steerable pyramid; ~95% of HDR-VDP-2 accuracy at 100× speed.
- **MS-SSIM**: 4-scale dyadic pyramid; canonical scale weights `[0.0448, 0.2856, 0.3001, 0.2363]` — these were tuned by Wang on natural images and are still widely used.
- **CPIPS**: extracts features from *multiple decoder stages* of a U-Net (not just the deepest), each channel-normalized and learned-linear-weighted. Achieves 50× speedup over LPIPS with 64–82% of accuracy.

### What zensim could do

The 228-feature input already includes some multi-scale info (tier 1/2/3 + dense percentiles on the `feat/dense-percentiles` branch). But it's **a flattened bag of statistics**, not a band-structured tensor.

1. **Reshape input to band-structured** if we ever move to a CNN backbone: e.g. group the 228 features into 4 dyadic scales of 57 features each, then a per-band weight (sigmoid-bounded `[0.5, 2.5]`) before the MLP. This adds 4 parameters but lets the MLP learn its own MS-SSIM-style scale weighting.
2. **Per-band auxiliary loss** during training: predict per-scale quality separately, then aggregate. Forces the model to use scale information. CPIPS shows this regularizes well.
3. **Dilated-conv head** as a Phase-3 experiment: replace the final 64-d hidden with a tiny dilated stack (dilation 1, 2, 4) over the band-structured input. Total parameter increase: ~4 k. Expected lift: matches NoR-VDPNet's claim of "≥95% of full HDR-VDP at 100× speed."

### Constants to lift verbatim

| Constant | Value | Source |
|---|---|---|
| MS-SSIM scale weights | `[0.0448, 0.2856, 0.3001, 0.2363]` | Wang & Bovik 2003 |
| HDR-VDP-3 #scales × #orientations | 5 × 4 | HDR-VDP-3 paper Fig 1 |
| CPIPS dilation | 3 | CPIPS §3.2 |
| Per-band weight bounds (learned) | `[0.5, 2.5]` (sigmoid · 2 + 0.5) | inferred best practice |

---

## 3. Pooling — mean, std, p-norm, max, info-weighted

This is the most empirically-supported lever in the corpus and probably the cheapest improvement we can make.

### Findings

- **GMSD** (Xue 2013) pools **the standard deviation of the local-quality map, not the mean.** PLCC ≈ 0.960 on LIVE — competitive with everything heavier — at *50× lower compute than FSIM* and *47× less than VIF*. The reasoning: humans rate images with *uneven* distortion (some regions sharp, some blurred) worse than *uniformly* distorted images at the same mean error. **Pure mean pooling literally cannot represent this.**
- **IW-SSIM** (Wang & Li 2010) pools by **information content** `w_i = log(1 + σ_i²)` per local patch. Upweights textured regions, downweights flat ones. SROCC on LIVE: 0.957, vs 0.948 for plain SSIM.
- **Butteraugli** uses **`max-norm` at high quality**, **`average of p-norms ∈ {3, 6, 12}`** at lower quality. The p-norm averaging is a notable trick — single high-p collapses to max, but averaging across exponents preserves both peak-error and broad-distortion sensitivity.
- **SSIMULACRA2** is reported (paper benchmarks, not its own source) as having **the strongest per-source PLCC (0.968) of any non-CVVDP metric on AIC-HDR2025**. Its pooling isn't fully disclosed in any OA paper I found, but the per-source consistency suggests a robust, possibly multi-norm composition.

### What zensim could do

1. **Emit a 4-vector instead of a 1-vector** from the MLP: `[μ, σ, max, p_6]` of the implicit per-feature quality signal. Then a 2-layer head reduces 4→1. Adds ~30 parameters; expected SROCC lift on CID22 ≥ +0.01 based on GMSD's gains. **Single highest-yield change in this doc.**
2. **Use std-pooling as an auxiliary loss head** during training: predict the *spread* of per-image MOS reports as a second target. Regularizes the embedding without changing the inference path.
3. **Try `mean of p-norm ∈ {3, 6, 12}`** explicitly on per-feature quality signals before the MLP. Costs nothing at inference (still scalar in/scalar out per feature), changes only how the 228-d vector is computed upstream.

### Constants to lift verbatim

| Constant | Value | Source |
|---|---|---|
| GMSD pooling | `std` of GMS map, not `mean` | Xue 2013 §3.3 |
| GMSD constant `c` | 0.0026 (for 8-bit normalized) | Xue 2013 §3.2 |
| Butteraugli p-norm exponents | `{3, 6, 12}` averaged | Guetzli §4; libjxl `butteraugli.cc` |
| IW-SSIM info weight | `w_i = log(1 + σ_i²)` (variance of local patch) | Wang & Li 2010 |
| FSIM gradient threshold | ~2° phase congruency | Zhang 2011 |

---

## 4. Targets / supervision — multi-target loss, Thurstone, pairwise

This is the single most-cited *methodological* gap between zensim and the modern literature.

### Findings — what's wrong with scalar MSE on a fixed target

- **CID22 measures MOS (appeal); KonJND-1k measures PJND (imperceptibility threshold).** These are *different psychometric scales*: MOS is cardinal utility on `[1,5]`, PJND (picture-wise JND) is, per reference image and codec, the lowest compression level at which an observer notices a difference; a dataset reports its distribution over observers (for example the satisfied-user ratio). It is a different quantity from the JPEG AIC JND unit, which is a scale value where 1 JND means 75 % correct in a forced-choice comparison (0 = the source, larger = more distorted; Testolina et al. 2024, arXiv:2410.09501). Neither is a 75 %-detection threshold of the other. *(Corrected 2026-10-11; this line previously called PJND a 75 %-detection threshold on a `[-2.5, 0]` scale.)* A scalar MSE loss against ssim2 cannot align both at once.
- **MOS is set-dependent and noisy.** Multiple papers (PieAPP, BAPPS) explicitly argue pairwise preferences are more stable per-subject and across labs.
- **High-fidelity range (q ≥ 80) is where SSIM/PSNR saturate.** SSIMULACRA2, Butteraugli, CVVDP all explicitly target this range. The AIC-3 protocol exists *because* MOS variance at q≥80 makes it nearly useless without boosting (artifact amplification, 10 Hz flicker, 2× zoom).

### The Thurstone Case V approach

- Define a latent quality scale `z_i ∈ ℝ` per image.
- Convert each labeled pair `(i, j) → P(i ≻ j)` into a likelihood under `Φ(d · (z_i − z_j))` where `Φ` is the standard-normal CDF.
- The **canonical psychometric constant** is `d · Φ⁻¹(0.75) = 1` → `d ≈ 0.6745`. This is the conversion factor between *z-score units* and *1-JND units*.
- Maximum likelihood gives `z_i` consistent across all paired comparisons (Mosteller / Thurstone 1927; reformulated by Bradley-Terry 1952 with a logistic instead of probit — equivalent in practice).
- Multiple papers (PieAPP, AIC-3, KonJND) use this; the AIC-HDR2025 study reports 95 % confidence intervals averaging **0.27 JND wide at 1 JND** with boosted triplets and this reconstruction (arXiv:2506.12505). *(Corrected 2026-10-11; previously attributed to the AIC-3 paper.)*

### What zensim could do

1. **Switch from MSE to a Thurstone/Bradley-Terry pairwise loss** as the primary signal:
   ```
   L = Σ_{(i,j)} [ y_ij · log Φ(d · (z_i − z_j)) + (1 − y_ij) · log(1 − Φ(...)) ]
   ```
   - For our 196 k synthetic pairs we already *have* pairwise structure (same reference, different distortion). Use it.
   - For CID22 / KADID we infer pairs from MOS deltas with a margin (e.g. `|MOS_i − MOS_j| > 0.3`).
   - For KonJND-1k we have explicit PJND thresholds → directly usable as ordinal anchors.
2. **Multi-target with KonJND aligned via Thurstone**, *not* via a separate scalar regression head. T11.20b currently bolts on a w=0.10 KonJND term as a second MSE. Doing it Thurstone-style keeps all three datasets on one latent scale.
3. **Two-stage train**: (1) Thurstone on pairs only → embedding. (2) Post-hoc spline calibration of `z → JND_units` using AIC-3 anchors at known JND levels (−1 JND ≈ NVL, −2.5 JND ≈ obvious artifact).
4. **Add a content-class auxiliary head** (lineart / screen / document / photo) — predict from same embedding. Multiple papers (CPIPS, multi-task SCI coding) report this as effective regularization at near-zero cost.

### Constants to lift verbatim

| Constant | Value | Source |
|---|---|---|
| Thurstone JND constant | `d ≈ 0.6745 = Φ⁻¹(0.75)` | AIC-3 §3 |
| NVL JND range | `[-1, -2]` JND | AIC-3 dataset definition |
| AIC-3 quality-range bounds | `[-2.5, 0]` JND | AIC-3 dataset definition (QoMEX 2023 sign convention). Later JPEG AIC work uses 0 = source and positive JND for distortion (arXiv:2410.09501, the AIC-4 common test conditions) |
| BAPPS pair construction | within-reference + cross-distortion, 64×64 patches | LPIPS / BAPPS paper |
| Margin for MOS-derived pairs | `|ΔMOS| > 0.3` (typical) | inferred from PieAPP §4 |

---

## 5. Architectural ideas worth porting

These are smaller, lower-priority changes but each pays for itself.

| # | Idea | Source | Cost | Expected lift |
|---|---|---|---|---|
| 5.1 | **Confidence head** — second 1-d output from same embedding; trained with NLL of a Gaussian over MOS variance (CID22 has per-image σ; use it). | HDR-VDP-3 §4, ProxIQA | +30 params | Surfaces low-confidence cases; downstream filtering of unreliable predictions |
| 5.2 | **Mixture-of-experts gate** — k=4 experts (photo / lineart / screen / document) gated by a tiny softmax. T11.x has explored MoE on v06-moe branch; the literature confirms it for IQA (multi-task SCI paper). | T11 + multi-task SCI | +50 % params | +0.01–0.02 CID22 SROCC at edges of content distribution |
| 5.3 | **Frequency-weighting modulator** — per-band sigmoid-bounded weight per *codec*; predicted from a tiny codec-id head. CSF is not universal across codecs (JPEG blocking ≈ mid-freq, VVC smoothing ≈ low-freq). | IQNet §3; HDR-VDP-3 paper | +50 params | +0.01 CID22 per-codec correlation |
| 5.4 | **Spatial JND map output** — a 2-d head producing a per-region JND map, trained from KonJND++ click data (300 images, 129 click maps each). Then global JND = `−max(map)` or `−p_99(map)` (per literature). | Chen 2023 (Localization of JND) | +200 params (1×1 conv head) | Enables saliency-weighted downstream encoders; unlocks bit-allocation pickers |
| 5.5 | **Distortion-type auxiliary** — predict distortion-type label from embedding (cross-entropy). | Multi-task SCI paper; CPIPS | +60 params | Regularization; minor CID22 lift; major out-of-distribution robustness |

---

## 6. Algorithm specifics extracted from the corpus

### 6.1 Butteraugli — full pipeline as the OA papers describe it

```
RGB(linear) → [3×3 matrix] → LMS-ish → [biased-log, b≈0.01] → "butter space"
            → Laplacian pyramid (multiple scales)
            + rotated line kernels (line-structure detector)
            → low-freq mask = f(local |Δ chroma|)
            + high-freq mask = color-channel-selective
            → per-pixel error map
            → per-pixel max-norm OR mean of {p=3, 6, 12}-norms
            → geomean across image patches
            → distance ∈ ℝ⁺   (≈1.0 = JND at 1000 px viewing distance)
```

The exact masking constants are not in the OA paper text — they live in `google/butteraugli/butteraugli.cc`. We should grep them out and either lift them as priors or use them to sanity-check whatever we learn.

### 6.2 SSIMULACRA2 — what we can infer

No OA paper *defines* SSIMULACRA2 — it's documented in the libjxl repo only. From benchmarks (AIC-HDR2025, AIC-JPEG-AI):

- Per-source PLCC 0.968 (the only metric beating CVVDP on this axis)
- Overall PLCC 0.906 (3rd-best after CVVDP, HDR-VDP-3)
- Sensitive to compression artifacts specifically; sometimes *less* sensitive than Butteraugli to colour-only distortions
- Public source: `https://github.com/cloudinary/ssimulacra2`

We should add SSIMULACRA2 as a **target** in our supervision mix (alongside ssim2, IWSSIM, CVVDP), not as a feature input — its output is exactly what we want to predict.

### 6.3 Guetzli's optimization loop — relevant only because we're inside the picker

```
init_quant_table ← search over predefined tables to satisfy butter(orig, decoded) ≤ α · target
α = 0.97        # 3 % slack allows local refinement
loop:
  for each DCT coefficient, in zig-zag order:
    zero it
    if butter(orig, decoded) ≤ target: keep zeroed
    else: restore
until no improvement
```

For zensim this is mostly a *cautionary tale*: at the edge of perceptibility, the loss landscape is extremely flat. Loss surface design matters far more than gradient-method choice. This is why MSE-against-ssim2 will plateau and why pairwise/Thurstone with carefully-chosen pairs near threshold is the higher-yield approach.

### 6.4 HDR-VDP-2/3 — the calibration anchor

HDR-VDP-2 trained on LIVE + TID2008 (LDR). HDR-VDP-3 added UPIQ (4000+ images, joint LDR+HDR, unified JOD scale via Thurstone). **UPIQ is the largest existing JOD-aligned IQA dataset.** If we want a single dataset that exercises both SDR and HDR ranges on one consistent scale, UPIQ is it. Not yet in our training corpus.

### 6.5 VMAF — what's worth borrowing

- 4 features: VIF (visual information fidelity, 4 subbands), DLM (detail loss measure), motion (TI; meaningless for stills), and PSNR-Y as a sanity tail.
- Linear regression head fit via SVR with RBF kernel; Netflix uses the v0.6.1 weights.
- **Crucially: VMAF was trained on the *Netflix VMAF+ dataset* of 230 sequences only, and the resulting features generalize poorly to AV1 / HEVC.** (Per Enhancing-VMAF paper §2.3.) The lesson: even very good metrics need cross-codec training data.
- **VMAF-neg** (variant) hits PLCC 0.958 on JPEG-AI — for *learning-based* codecs, video-trained metrics out-perform image-IQA metrics. Suggests we should add **AV1/HEVC video frames** to our training corpus once we have time.

### 6.6 LPIPS / DISTS / PieAPP — feature-distance learned metrics

- All three extract features from a *pretrained* CNN backbone (VGG-16 conv1_2..conv5_3 is the LPIPS canonical choice), compute per-layer feature differences, weight them with a small learned linear layer per channel.
- **PieAPP** trains the *whole network* end-to-end on the BAPPS-style 2AFC dataset (200 refs × 75 distortion types × ~256 pairs per ref = 384 k pairwise comparisons).
- **DISTS** explicitly separates structure (`μ, σ` over feature maps) from texture (`μ` of features themselves) — robustness gain against textured-vs-blurred confusion.
- **CPIPS** ablates this: confirms multi-layer features beat any single layer; layer weights matter more than backbone choice; lightweight U-Net works almost as well as VGG at 50× speed.

We have most of the *features* equivalent of these in our 228-d input (the Tier 1/2/3 feature table is essentially a hand-engineered VGG-like statistics bank). We're missing the **pairwise training objective** and the **deep multi-layer view**. Both are addressable without a CNN backbone.

---

## 7. Corpus / data work to do

| Priority | Action | Cost | Win |
|---|---|---|---|
| P0 | **Add SSIMULACRA2 supervision target** to the unified training parquet. Score every (ref, dist) pair with `ssimulacra2 --orig ref --dist dist` once; store as a column. | ~24 h compute on 196 k pairs | Adds 4th supervision target; aligns with the metric most papers in the corpus call out as strongest per-source on AIC-HDR |
| P0 | **Add Butteraugli p-norm score** (and `--distance` max-norm) as additional target column. | ~12 h compute | Lets us train a multi-target loss that includes both Butteraugli AND SSIMULACRA2 — *what Jpegli already does for the JPEG encoder* |
| P0 | **Build the Thurstone pair file** from existing CSVs. Per reference image, all C(n,2) pairs with `Δssim2 > ε`. Save once; reuse for every Thurstone-loss experiment. | ~2 h | Unblocks every Thurstone-loss experiment below |
| P1 | **Ingest UPIQ dataset** (~4000 images, JOD-aligned across SDR+HDR). Public; widely used as the modern recalibration anchor. | ~1 d ingest + 3 d sweep | Single biggest content-distribution expansion available |
| P1 | **Ingest KonJND++** (300 images, 129 spatial-click maps per image). Public dataset; enables spatial JND map output (§5.4). | ~1 d | Unlocks spatial-map architecture |
| P2 | **AIC-HDR2025 + AIC-JPEG-AI** datasets. ~600 images total, full triplet-comparison data; recent (2024-2025) and codec-rich. | ~2 d | Closes the AIC-3 calibration gap |
| P2 | **CLIC + LIVE + KADID-10k + TID2013 + CSIQ** as a unified secondary anchor set. Already partially in `unified_v15`. | ~1 d | Strengthens out-of-distribution validation |
| P3 | **Programmatic distortion expansion** of own corpus to >75 distortion types (PieAPP-style). Currently the synthetic corpus is heavy on JPEG/WebP/AVIF/JXL artifacts; under-represents Gaussian noise, blur, contrast/brightness shifts, color shifts, JPEG2000 ringing. | ~3 d generation + score | Broadens the embedding's coverage and likely fixes out-of-distribution failures |

---

## 8. Concrete experiment slate

Numbered for cross-reference. Each is sized for a Phase-3-style cycle.

### High priority (CID22 / KonJND / AIC-3 simultaneous gains)

**EX-1. Thurstone loss replacement.** Replace the current scalar-MSE loss in `zensim-validate` with a Thurstone-Bradley-Terry pairwise loss over `(ref, dist_i, dist_j)` triplets from the synthetic corpus. Initial guess: drop the existing scalar regression entirely.
- *Expected*: CID22 SROCC ≥ 0.83, KonJND SROCC ≥ 0.92, AIC-3 SROCC ≥ 0.85 simultaneously. T11.20b achieves 0.7986 / 0.9219 / 0.79 — should be a clear improvement on the worst two.
- *Effort*: 2 days code + 1 day per 5-seed sweep.

**EX-2. Std-pooling head.** Add `[μ, σ, max, p₆]` as a 4-vector output of the feature aggregator, then a 4→1 reducer.
- *Expected*: +0.01 CID22 SROCC, free.
- *Effort*: half a day.

**EX-3. SSIMULACRA2 multi-target.** Add SSIMULACRA2 score as a 4th target alongside ssim2/IWSSIM/CVVDP, weighted 0.25/0.25/0.25/0.25.
- *Expected*: closes the AIC-3 0.79 → 0.85+ gap (SSIMULACRA2 is the AIC-3 winner among non-CVVDP metrics).
- *Effort*: 1 day scoring + 1 day training.

**EX-4. XYB / biased-log front-end.** Apply a 3×3 LMS-ish matrix + biased-log transfer before feature extraction.
- *Expected*: +0.005–0.015 CID22 SROCC; bigger lift on chroma-dominant distortions.
- *Effort*: 2 days (have to update the feature extractor and re-bench).

### Medium priority

**EX-5. Spatial JND head from KonJND++.** Add a 1×1-conv head producing a per-region JND map. Global JND = `−p_99(map)`.
- *Expected*: enables saliency-weighted downstream encoders. Doesn't necessarily move the headline SROCC but unlocks the *picker* and *encoder* loops to consume zensim's output.
- *Effort*: 3 days.

**EX-6. Codec-conditional frequency weighting.** Tiny codec-id-conditional per-band weight module.
- *Expected*: +0.01–0.02 per-codec SROCC, particularly on JPEG vs VVC vs JPEG-XL.
- *Effort*: 2 days.

**EX-7. Mixture-of-experts.** k=4 experts gated by content class (photo / lineart / screen / document).
- *Expected*: marginal improvement at distribution edges; this is the v06-moe branch — finish it.
- *Effort*: 3 days.

### Low priority but worth tracking

**EX-8. Confidence head with NLL loss.** Second output predicts variance; trained against per-image MOS spread.
- *Expected*: no SROCC change; enables downstream filtering of unreliable predictions.
- *Effort*: 1 day.

**EX-9. UPIQ ingest + recalibration sweep.**
- *Expected*: rationalizes joint SDR+HDR scale; sets up Phase-4 HDR work.
- *Effort*: 1 week.

**EX-10. PieAPP-scale distortion expansion** (75+ distortion types).
- *Expected*: out-of-distribution robustness; marginal change on existing benchmarks.
- *Effort*: 1-2 weeks.

---

## 9. Anti-patterns explicitly called out in the literature

Worth pasting verbatim into our own CLAUDE.md anti-pattern list:

1. **"Don't train a metric on one set and test on another *without* aligning the scales."** Multiple papers report PLCC drops of 0.1–0.2 from this alone. The fix is INLSA recalibration (Iterated Nested Least-Squares Alignment) between two datasets' scales before treating them jointly.
2. **"Don't take the *mean* of a local quality map."** GMSD made its name proving this. *Std* of the map is what humans rate. Same evidence appears in 4–5 other places in the corpus.
3. **"Don't apply SDR metrics to HDR luminance without PU encoding."** *Every* HDR-VDP paper opens with this warning. PSNR on raw HDR luminance is essentially noise.
4. **"RGB-domain metrics fail on HDR."** Use luminance-only or PU-encoded chroma + luma separately.
5. **"Don't trust val-loss as a proxy for held-out CID22 SROCC."** zenanalyze leak audit confirmed this: V0_5 had val_mean 0.99 *and* 11.77 % training contamination. Per-seed CID22 SROCC is the only honest signal.
6. **"Don't use the same metric as both target and feature."** ssim2 as both a training target and an input feature is structurally degenerate. (Currently zensim does some of this; verify.)
7. **"Don't optimize aggregate (KADID + TID + CID22) / 3 SROCC."** Hides CID22 regressions. CID22 is gold standard; others are integrity guards.
8. **"Don't ship a metric calibrated to one viewing-distance assumption without documenting it."** Butteraugli ships with 1000-px viewing-distance assumption; if your encoder targets mobile (~3000 px viewing), you'll over-spend bits.

---

## 10. The roadmap, if I were doing only this

(Provided as a useful one-glance answer; the *actual* roadmap depends on T11.x decisions.)

1. **Land EX-1 + EX-2 + EX-3** in one V_24 sweep. These don't conflict with each other and together address the biggest gaps. Expect ~2-3 weeks calendar.
2. **EX-4 (XYB front-end)** as V_25 because it touches feature extraction and is moderately invasive.
3. **EX-5 + EX-6** as V_26 — these are about *outputs*, not the embedding. Doing them after the embedding is stable is the right order.
4. **EX-9 (UPIQ)** as soon as we have HDR encoder work landing. Until then, keep it on the backlog.
5. **Everything else** is opportunistic.

The single highest-yield change in this entire document is **EX-1 (Thurstone loss replacement)** — it changes how the model learns from the data we *already* have, requires no new ingest, and structurally aligns three benchmarks (CID22, KonJND, AIC-3) that are currently in tension. If only one experiment from this doc lands, this is it.

---

## 11. References — which paper-md to read for what

All paths under `/mnt/v/input/papers/<bb>/<blake3>.md` in the local corpus mirror. Browse the topic docs at `~/work/zen/zenpapers/docs/composition/` for grouped views.

### Butteraugli / SSIMULACRA / Guetzli / Jpegli
- `d8/d87a0543…` — Guetzli: Perceptually Guided JPEG Encoder (the foundational paper)
- `26/26e01474…` — Users prefer Guetzli over libjpeg (75 % preference validation)
- `21/216c0608…` — Jpegli: hybrid Butteraugli + SSIMULACRA tuned JPEG encoder
- `11/11a7b9af…` — JPEG XL benchmarking (XYB introduction)

### VDP family
- `3e/3eb37357…` — Bridge VDP ↔ JND (FvVDP → SUR via polynomial fit)
- `dc/dcb3daae…` — NoR-VDPNet (no-reference HDR-VDP via CNN)
- `47/47d6dbff…` — HDR-VDP-3 multi-task metric
- `8c/8c72e282…` — Practicalities of predicting HDR quality (PU encoding)
- `a2/a2b44f21…` — Quality Metric Aggregation HDR/WCG (SVR ensemble)
- `63/6305f273…` — ArtHDR-Net (perceptual loss as JOD signal)
- `2b/2bbec4c5…` — ColorVideoVDP (color opponent + transient/sustained channels)

### VMAF
- `a0/a02f7b3a…` — Enhancing VMAF: dynamic-texture features + 165-feature screen
- `41/41184d2d…` — VMAF bitrate ladder (knee-point prediction)
- `cc/cc579c16…` — Metric fusion for NeRFs (VMAF + DISTS min-max averaging)
- `ee/ee5c944d…` — VMAF on 360VR (domain generalization without retraining)
- `2b/2b5380c4…` — ProxIQA: VMAF proxy via deep features
- `94/94c65870…` — SUR prediction via VMAF + quality metrics

### Learned metrics
- `d9/d964fa4e…` — PieAPP (pairwise preference learning, Bradley-Terry, 384 k pairs)
- `08/08ba582d…` — CPIPS (multi-decoder-stage features, ~50× LPIPS speedup)
- `f2/f25b9e66…` — Pathology compression with deep contrastive features
- `07/0714329c…` — MS-ILLM (statistical fidelity via VQ-VAE discriminator)

### JND / CSF / AIC-3
- `a4/a4f8f336…` — Fine-grained subjective IQA for high-fidelity compression (AIC-3 protocol)
- `44/44aee8f1…` — IQNet (JND prefiltering, codec-conditional weighting)
- `6a/6a48358a…` — JPEG AIC-3 Dataset
- `b6/b62c13fa…` — Localization of JND (KonJND++ spatial click maps)
- `45/45ae66c9…` — Boosted Triplet Comparisons (theoretical foundation)
- `03/031d1417…` — Fine-Grained HDR IQA (AIC-HDR2025; SSIMULACRA2 per-source winner)
- `e4/e41c087b…` — JPEG AI subjective study (CVVDP / VMAF-neg / SSIMULACRA2 on learning-based codec)

### Classical IQA
- `08/087ac6d3…` — Multi-Method Fusion (13-metric ensemble vs DISTS)
- `03/03d268a2…` — GMSD (std-of-quality-map pooling)
- `02/023b19c5…` — Visual-analysis RDO (multi-scale CNN feature distance)
- `25/25301e6e…` — Multi-task learning for screen content image coding

---

*End of file. Update freely; this is a working reference.*
