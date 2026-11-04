# R12: non-negative linear heads against adopted A (2026-11-04)

The original nine fits and their registered assessments are complete. The separately
registered L0/L4 addition is in progress. No candidate is selected or adopted here.

Registration: [E36 R12 record](e36_subset_test_2026-11-01.md#r12-registration-2026-11-04-before-any-fit-or-new-read-linear-formulas-over-as-inputs),
commit `9088eacf`; exposure: [DATA_SPLITS R12](../docs/DATA_SPLITS.md#exposure-registration--2026-11-04-r12-linear-heads-against-the-adopted-mix-a).
The development-unit correction was committed as `284ccfb7` before L3.

L1 permits all 984 inputs. L2 fixes all but the top eight audit families to zero:
144 damage features and their 144 fragility products remain available. Both direct
and product coefficients are non-negative. Three random seed indices per arm.
L3 continues each L1 seed through the mix's AIC-2026 public_train ×256 recipe with
rehearsal and fixed epoch 59. No public_test or confirmation-set read is authorized.
No AIC-2026 result statistics are recorded here.

All base fits use the mix's exact admitted root, group targets, weights and sampling
through `v2_confirm_fit.py` / `v2_lodo_mlp.py`: TID2013 + MDID MOS, BAPPS, SafeSyn,
CID22 TRAIN teacher, coverage and chroma-teacher legs. Only the head architecture
and L2's fixed zero-weight mask differ. The existing N + mono path with H1 and zero
hidden bias gives an identity ReLU on non-negative damage inputs: raw = 100 minus
a non-negative weighted sum. The common output stage and TRAIN-fitted PCHIP dial
remain nonlinear. Adam optimizes the existing loss; these are three fitted solutions,
not a certified global linear optimum. L1/L2 stop at epoch 119, L3 at epoch 59.
Canonical densify, f16 quantization with protected last layer, no input pruning,
then CID22 TRAIN calibration with E33 identity knot/tail factor 2.

The historical dataset authority pins H128. R12 has an explicit H1 authorization
in the Python/native admission owners (`b901766a`), with the registration identity
and sparse-mask digest bound before optimization. All old population, seed, budget,
continuation and data-role checks remain. Seven Python admission tests and twelve
native admission tests pass, including unregistered-H1 and changed-mask refusals.

## Training-side feasibility decision

The registered teacher-development rows are SafeSyn (38,757) and CID22 (3,785).
Stored targets already use score points (`--target-scale 1`). The invalid initial
100× conversion is retained under `dev-unit-error/`; it does not qualify L3.
Corrected pooled MAE, seeds 0 / 1 / 2:

| Model | SafeSyn | CID22 | Row-pooled |
|---|---:|---:|---:|
| A (research F) | 3.393523 / 3.398859 / 3.568893 | 1.500572 / 1.586669 / 1.601302 | 3.225105 / 3.237627 / 3.393835 |
| L1 | 2.956284 / 2.712661 / 3.163496 | 1.370633 / 1.314540 / 1.437120 | 2.815207 / 2.588269 / 3.009899 |

L1 seed-mean pooled error 2.804458 versus A 3.285522: delta −0.481064 score
points, within the registered +1 point allowance. L3 therefore ran. This is
training-side teacher evidence, not independent human generalization.

## Frozen fits

Full hashes, epochs and native certificates: [fit receipt](r12_linear_fits_2026-11-04.json).
All nine packed bakes are certified monotone with zero ramps and zero max pairs.

| Arm | Seed | Packed SHA-256 | Bytes |
|---|---:|---|---:|
| L1 | 0 | `6a9312a478d727195e5cb25ab72c2f3e04ee0f8e8b6aeebbc9f0d0aaa843a2b9` | 21814 |
| L1 | 1 | `985d2b234254f4a9c691cee9cc0cef6b4aec805b46ee966e297798eed649ce08` | 21881 |
| L1 | 2 | `2c54e0991eba3173c647ea285d7d031e2b54e7b27733ff1aebec14a35c29683e` | 21782 |
| L2 | 0 | `51a3ff42cbf48ddadbddaafe0df9b730871798af8f73d24827a88abfa2096dd5` | 20930 |
| L2 | 1 | `9d6a9bcb36a5aa844d4561303d580842680f09c170e0194a1508ec9b9b6c89bc` | 20933 |
| L2 | 2 | `012545710c8edfb074fb28f65b06d883091ca76b9e8eecf5d68bbdaa4fe9185d` | 20914 |
| L3 | 0 | `7230b6046d6f9df6345e5904eb79af98cd8986de3df30f14d3a8a8e17578c0ac` | 23957 |
| L3 | 1 | `3984473bbbc46cbc6f283ed1f205365b6a761733f0df34629b5760f2cd8a7bf1` | 23668 |
| L3 | 2 | `62ec6a8ee8f04417a75bae8f4005afaf9baabdbc2791153ec477819132eddaf9` | 24023 |

Artifact root, checked on dev: `/var/tmp/byc-linear-20261104/`. Model files live under
`fits/L1-s*/refit/production-f16.bin`, `fits/L2-s*/refit/production-f16.bin` and
`fits/L3-s*/pack_e059/refit/production-f16.bin`. `SOURCE_PINS.json` binds source files,
trainer/packer binaries and the immutable root. `MODELS.json` freezes the assessment
inventory. All logs and the refused pre-fit/unit-error attempts remain retained.

Execution through the shared heavy lock and `run-heavy --mem 24G`: six base fits
536 s, peak-RSS 2.00 GiB, min available 35,772 MiB, peak load 3.68; three
continuations 344 s, peak-RSS 1.96 GiB, min available 36,502 MiB, peak load 2.41.
These are execution receipts, not inference timing comparisons.

## Original-arm assessments

[Measured read tables](r12_linear_reads_2026-11-04.md) report every original
population beside A: all nine MOS collections, compression/non-codec subsets,
E36, G-CHROMA-L, G-STEER, C1–C6, identity, NEARID and the five HDR views.
Cells use the three seed means; the retained JSON has seed values and native
whole-reference/family intervals. These are descriptive comparisons after freezing,
not selection. The eight held-out MOS populations exclude MDID, which appears in
training; the common three remain IQA-ODI, NITS-IQA and VCL-FER. The historical
experiment-5 readout's arm-name heuristic treats unknown R12 names as the full
recipe, so its `heldout` aggregate is not used. Collection statistics are unchanged.

Dense L1 improves E36 inversion share by 0.248398 percentage points against A
(95% origin bootstrap −0.427255 to −0.060774), but all-nine MOS normalized median
error rises from 8.02794% to 8.35201%. L2 has 3.897773% E36 inversions against A's
3.396647%; its all-nine error is 8.58165%. L3's continuation yields 3.595252%
inversions and 8.26648% error. Each fitted seed passes 135/135 G-STEER cases,
C1–C6 on both grids, exact pixel identity, and NEARID N1/N2/N3. All nine fail
G-CHROMA-L level qualification; all pass its rank non-regression. Extended-XYB MAE
rises from A0's 3.576321 points to seed means 4.029702 / 4.934336 / 3.946562 for
L1 / L2 / L3. No original fitted arm meets the complete label-free product gates.

HDR uses the registered cube-root:203 banks and historical native predictor. It is
report-only and includes a teacher validation population. No HDR leg was fitted.
The small 4Kdtb EVAL sample has one family, so its family bootstrap is undefined.
Pooled ladder correlations are descriptive; within-ladder rank is the ladder read.

## Full sparse formula and head timing

The [full L2 coefficient table](r12_sparse_formula_2026-11-04.tsv) contains 144 damage
features, each with direct and fragility-product coefficients for all three seeds.
With damage D and reference fragility R, its algebraic raw head is
`100 − sum_i D_i * (a_i + b_i * R_i)`. All 288 available coefficients are non-negative;
all 696 other input coefficients are zero. Actual nonzero counts are 288 / 274 / 285.
The common output stage and TRAIN-fitted PCHIP dial remain nonlinear. The table folds
native scaler and layer coefficients in f64 for interpretation; the packed native
H1 preserves f32 scaling/dot/output operation order. A handwritten folded formula
is not asserted bit-identical to the served bake.

Paired native `Predictor::predict_transformed` zenbench measurements on dev include
input scaling and network forward, over 256 fixed positive logical-input rows.
They exclude feature extraction, derived-product generation and dial evaluation.
Default native dispatch; one call thread; no target-cpu=native. Thirty paired rounds
per seed, using identical gate-clean round indices for both heads. Retain every raw
round; no value-based trimming. Gate-clean counts are 20/22/15, 13/19/20 and 20/21/23
for L1/L2/L3. These descriptive clean-round seed means do not imply a pooled timing CI.

| Arm | A matched head (µs/pair) | Candidate head (µs/pair) | A / candidate |
|---|---:|---:|---:|
| L1 | 5.774279 | 2.581856 | 2.236483 |
| L2 | 5.609705 | 2.540460 | 2.208145 |
| L3 | 5.786397 | 2.551408 | 2.267923 |

The head batch resource receipt is 87 s, peak-RSS 0.18 GiB, minimum available
39,133 MiB, peak load 1.57. The read batch is 2,702 s, peak-RSS 8.08 GiB,
minimum available 32,886 MiB, peak load 8.29. Its exit status records three failed
reports caused by a missing copied `judge_pairs.tsv`, not failed predictions.
Those reports were replayed successfully without changing bakes, rows, targets,
statistics or gates: 8 s, peak-RSS 0.25 GiB, minimum available 40,520 MiB, peak load
5.29. The earlier command-logging and report failures remain preserved.

L2 still declares all 984 inputs; the native extraction plan uses declared IDs,
not nonzero coefficient inspection. Extraction work is unchanged in this experiment.
The original timing executable is retained at `heads/bin/prune_forward_bench-original`
with its source and hash receipt. Reproduce summaries with
`scripts/byconstruction/r12_linear_head_summary.py`; reproduce tables with
`scripts/byconstruction/r12_linear_tables.py`. Complete raw artifacts are under the
registered `/var/tmp/byc-linear-20261104/` root, not committed model binaries.

## Serving route

The internal ZNPR predictor and research BakeScorer serve the H1 and H2 containers;
no new wire format is needed. In the proposed public zensim crate at pinned revision
`bebc6ea45d2b6adb6ab70affc3e289707ef74fb0`,
[BakeScorer](https://github.com/imazen/zensim/blob/bebc6ea45d2b6adb6ab70affc3e289707ef74fb0/zensim/src/metric/bake.rs)
and [ProfileParams.mlp_bytes](https://github.com/imazen/zensim/blob/bebc6ea45d2b6adb6ab70affc3e289707ef74fb0/zensim/src/profile.rs)
are crate-private. Its public custom-profile path is not an arbitrary custom ZNPR
loader. These research bakes therefore need approved profile embedding/integration
before that public API can serve them. This experiment adds no public API.

## L0/L4 addition (registered before construction)

Registration `d3ec4dab` adds structurally pruned A and a freshly fitted two-piece
max-linear head, both over the same 984 inputs. Offset amendment `339de6b8` was
committed before any L4 construction: one learned non-positive offset on the second
piece, `max(a.x, b.x − t)`, with a,b,c,t non-negative. Native admission binds that
amendment explicitly. L4 uses the same base objective and fixed final epoch as L1/L2;
it has no added continuation arm. A and L3 have the registered continuation, which
is a training-protocol difference from L1/L2/L4 as well as a head comparison.

L0 removes only zero-encoder-column ReLU units with non-positive bias. The native
packer receives an explicit identity hidden-unit permutation so it retains live
accumulation order. Bit identity is an experimental gate, not inferred from the
algebra; fixture proof and the optional two-unit residual-removal diagnostic follow
the frozen L0 bakes. No inference tolerance is authorized.
