# Frozen seed-0 failure adjudication

Production remains D1 seed 0 alone, SHA-256
`f803b74c4252952f337abdc0234c2930839d45dddc32ae9b8b5296d6c840f400`.
No model, calibration, threshold or public API changed; no push or protected
payload read. All inference uses the existing native Rust owners. This is a
failure diagnosis, not a complete release qualification.

## C5: real raw-feature model failure; stale explanation

All 38 cached rows replay their original scores. All 15,960 consumed input
positions match canonical native extraction after the declared f32 storage
round trip. The ten nonzero reference-only positions in each row differ at
f64 precision because of that storage rounding; the resulting complete scores
remain exactly equal. The serving extraction and research extraction agree on
all 420 native input bits for each Rev5 model. A one-bit f422 negative control
is detected by the input audit despite score rounding.

The vector is not zero: consumed IDs 422,480,509,538,567,596,625,654,683,712
are reference-only PJND_FRAGILITY values. The registry declares them
ReferenceOnly; Rev5 deliberately computes them on identical pairs. For seed0,
the artificial zero row scores 99.8088634090379, whereas real identity rows
score 92.21993638332302–97.49362681388259. The zero-vector annotation in the
addressability report is false for this packet; changing it cannot repair C5.

| Model / explicit extraction revision | Inside [97.5,100] | Raw identity minimum | Maximum |
|---|---:|---:|---:|
| D1 packed seed0 / Rev5 | 0/38 | 92.21993638332302 | 97.49362681388259 |
| D1 packed seed1 / Rev5 | 0/38 | 94.27860248581597 | 97.29189157847993 |
| D1 packed seed2 / Rev5 | 0/38 | 93.96345024289823 | 97.2005848691835 |
| Research by_v2fy full-s0 / Rev5 | 1/38 | 89.675048828125 | 98.03453826904297 |
| D1 dense seed0 / Rev5 | 0/38 | 91.45294189453125 | 97.3927001953125 |
| Serving B canonical raw features / Rev1 | 0/38 | 96.0899119319179 | 96.23576709083017 |

Serving B runs separately at Rev1 on the same 38 pixel references. Its
historical identity shortcut returns a zero placeholder feature vector before
extraction; scoring that placeholder gives the constant 96.24115978721524.
The canonical non-shortcut extractor retains Rev1 numerical identity residues
and gives the range above. Both raw-feature interpretations fail the unchanged
band. Neither is a Rev5-cache/Rev1-model comparison. Every model's native pixel
identity and independently identity-aware cached score is exactly 100.

Conclusion: the failure predates the final f16 export and is present in the
research model and incumbent raw-feature path. The frozen raw C5 contract
requires a new trained/calibrated candidate with real reference-dependent
identity constraints, followed by the affected and final gates. Proven pixel
identity can use the existing identity-aware API, but that product route does
not satisfy or waive raw C5. Dropping reference-only inputs, substituting zeros
or widening the band is not a harness correction.

## C2: matching absent slots were misclassified

The original owner subtracts all 720 feature slots in `feat_eq`. Rev5 tables
mark uncomputed slots NaN, so NaN minus NaN fails the existing 1e-5 comparison.
This charges feature-equivalent codec-saturated rungs to the flat/clamp bucket.
The correction preserves the finite epsilon and requires matching missingness.
NaN-versus-finite, unequal lengths and infinities remain unequal.

The native owner overlay was built and its two focused regression tests passed.
Its complete-score predictions are compared with the original owner, with
all models and inputs unchanged. An independent classification mirror agrees
with the owner before correction; the corrected native reports supply the
verdict below.

| Grid | Adjacent rungs | Original flat | Corrected flat | Corrected proportion | <=0.05 |
|---|---:|---:|---:|---:|---|
| Standard | 4318 | 329 | 28 | 0.006484483557202408 | pass |
| Ladder | 9411 | 343 | 343 | 0.03644671129529274 | pass |

The remaining 28 standard flat rungs occur at the packed spline's lower
floor, -16.90630340576172. There are 52 floor cells in standard and 407 in
ladder. Dense has no spline: its original 301 standard ties become zero after
the same missing-slot correction, and its ladder has zero flat rungs. Packing
also adds TRAIN calibration, so dense-versus-packed score differences cannot
be attributed solely to f16 rounding. The uncalibrated packed outputs and
JPEG repair probes retain variation that the calibrated floor removes.

Conclusion: apply the small owner patch and rerun the unchanged frozen model.
That fixes the C2 verdict; it leaves C5 and steering failed. Per-rung evidence
includes exact image/codec/q keys, original and corrected buckets, consumed
feature equality, complete scores and uncalibrated outputs.

## G-STEER: real scorer/map failures; f16 is not the common cause

All seven packed failures reproduce exactly, including original block records,
score deltas, input pixel hashes, refinement coverage and M2/M3f values.
Both dense and packed use the same already admitted pixels and block sizes.

| Case | Packed M2 / M3f | Dense M2 / M3f | Explanation |
|---|---|---|---|
| broad-206-b32 | .969793621013133 / .972045028142589 | .9727954971857409 / .971294559099437 | M2 miss persists; finite first-order prediction does not rank these finite repairs adequately |
| broad-206-b64 | .9440559440559375 / .7972027972027916 | .9440559440559375 / .7972027972027916 | M2 miss identical in dense and packed |
| I01_10_05, block8 | 0 / 0 | .9999816261414052 / .6531795586562122 | packed floor erases score deltas; dense still misses M3f |
| I21_10_05, block8 | 0 / 0 | .9999849134002947 / .6520734843428108 | packed floor erases score deltas; dense still misses M3f |
| I41_10_05, block8 | 0 / 0 | .9999658146623001 / .76909931421085 | packed floor causes this observed failure; dense clears both bars |
| I61_10_05, block8 | 0 / 0 | .9999139979372235 / .8396604833899269 | packed floor causes this observed failure; dense clears both bars |
| dog-256-20 | .9872252747252745 / .9679945054945054 | .9874999999999998 / .9689102564102563 | M2 miss persists |

For each heavy JPEG pair, the packed base and every registered single-block
repair have the same floor score. Rank correlations return zero on these
constant score-change vectors; removing these cases would waive a failed
contract. Geometry-selected first/middle/last-block native probes preserve
varying f16 network outputs beneath the floor. For I01, packed raw output
changes from -69.7903823852539 to -69.70488739013672 for the first repair, while
both complete scores remain -16.90630340576172. Dense corresponding outputs
are -69.80496215820312 and -69.71944427490234. This is calibration clipping,
not f16 collapsing the network response.

Serving B's declared read set includes masked/IW features f228–f371. The
prepared steering owner explicitly refuses that family because it has no
spatial refinement integrand. Scalar/pixel scoring support cannot establish
spatial support. The original 135 refusals are retained; no numeric B map
baseline is invented.

Conclusion: this is not a cache or statistical harness fix. Recalibration is a
new candidate, even with unchanged network weights. The remaining dense M2
misses and two dense JPEG M3f misses show that recalibration alone is not an
established solution. Options are a new candidate satisfying the existing
contracts or an improved native steering implementation with complete parity,
coverage and runtime qualification. Choosing either requires the owner;
neither is implemented by this analysis. Removing degenerate cases or changing
M2/M3f bars is not an option taken here.

## Coverage interpretation

The original steering registration defines owner12 as four moderate JPEG
pairs times three individual research seeds, each at block32. PRODQUAL-B's
frozen composition is a single seed, hence four owner cases. Those are four
of four input geometries for seed0, not a claimed 12-case three-seed result.
The D1 control rerun passes 12/12 owner cases across seeds0,1,2, with seed0
exactly reproducing its original four cases. Minimum M2 is
0.9993285950682256 and minimum M3f is 0.9501095907947129. These are individual
seed controls, not an ensemble or an expansion of the shipping composition.

The Rev5 JPEG owner enumerates I01/I21/I41/I61 at levels03/05 and block8:
eight pairs with 8x8 blocks. The gate map's command also says `--block 8`.
It does not register eight different JPEG block geometries or 64 cases.
The older JPEG research diagnostic additionally tested block16; its historical
results are not inherited by the frozen model. Seed0 JPEG coverage is eight
of eight registered pairs, with four failures; broad96 and full27 M3 fixtures
remain the separate geometry populations.

## Evidence and scope

Base: `dcf7e3552b06e1bc0b36520f945d77d62d60295b`. Dense was reconstructed
from the hash-bound selected last checkpoint through the existing densify
owner. Its SHA is exactly the declared
`e6995b1073de6022c31e012a104a50a1710488d4e520ee7fd016c649d924c4c4`;
512 densify probe predictions are bit-identical. No calibration table or human
opinion target was opened. The existing corpus-disabled verdict command uses
the registered empty-root manifest and feature-set admission, not defaults.

Evidence: `/mnt/v/output/zensim/adjudicate-2026-10-07/`; mirror:
`/mnt/tower/output/zensim/adjudicate-2026-10-07/`. Exact verification and archive
receipts are `FINAL_VERIFICATION.json`, `_MANIFEST.json`,
`MIRROR_VERIFICATION.json` and `CARGO_CLEANUP.json`. They are required before
the final completion marker. Input path mappings retain original paths and
bind archived engineering packets/pixels by SHA256. Original reports and
intermediate failure logs remain preserved. No R2 copy is claimed.

The earlier coordinator amendment allowed run-heavy work, then e28-sol
requested another hold. The lane stopped and ended its waiting turn without
polling. The 2026-10-08 10:20Z coordinator release explicitly ends the hold;
every subsequent heavy command uses its shared flock wrapper. No adjudication
training process exists. The supplied blocking PIDs belonged to the separate
V40R2 lane and were not signaled. SPEEDQ's completed timing report separately
fails its release criterion; no speed pass is inferred here.

Validation: native owner comparator regression tests pass; scoped Rust fmt
and the audit example's warnings-denied clippy pass. Full library tests and
remote CI are not claimed. The twelve-case owner rerun's run-heavy receipt is
`rc=0 22s | peak-RSS 0.06GiB | min-avail 37779MiB | peak-load 0.76`.
This process receipt is not a per-size scorer RSS or speed qualification.

The proposed Known Bugs update is a small review patch because the existing
CLAUDE.md exceeds 30KB; committing that full file requires user approval under
the shared AGENTS rule. The native harness correction is supplied as a reviewable
overlay patch, with the canonical owner restored byte-for-byte. Rust source
below 600KB is size-exempt under the revised shared instructions; integration
is left to the coordinator because this brief requests analysis and no pushes.
