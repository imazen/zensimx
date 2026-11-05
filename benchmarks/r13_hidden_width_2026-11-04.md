# R13: A at proportional hidden widths (2026-11-04)

Training width affects the learned solution in these seed-0 runs. Full-recipe W4 does not recover
A’s MOS error or E36 ordering. W16/W32/W64/W256 retain one live max pair; W4 has none detected and W512
retains two substantial pairs. W4/W16/W512 improve steering to 135/135, but every width fails
chroma-level, so no width improves across the registered product-gate set. No model is adopted.

Registration `25e77292`; admission/execution `ccdb5c7a`; native read-owner reuse `d6e429d3`.
All six bakes were frozen in `7669afbe` before assessment; capacity orchestration repair `f6941f24`.
Fits completed on November 4 UTC; the EVAL and product assessments ran on November 5 UTC.
The control A is the frozen adopted W128 mixF3, seeds 0–2; no A refit.
Owner amendment `397d67be` cuts the original seed repetitions on cost and adds W4 seed 0 after W16.
Candidate seeds 1–2: **not run (owner cut on cost, 2026-11-04)**. W4 was registered only at seed 0.
Candidate across-seed stability is unmeasured. These are six seed-0 fits beside the three frozen A seeds.
W4 uses numerical source `ccdb5c7aa0d4b67d52f0e2874f325fbc84ddb11c` with the sole Rust overlay
`zensim-validate/src/subjective_training_leg.rs` for its amendment-bound admission. Its trainer SHA-256 is
`d14d9c8b69067fa0761608f12248c6801a9e4ace0a678ee11602beee6dd0d2c9`; complete file/binary pins are in `W4_SOURCE_PINS.json`.
Its admission/build source is `509c4f09d9c49ac9fe5c842e3eff28d94b9f36b4`.
Owner defines width as pairs = width/4: half the units form max pairs, half retain A’s ordinary threshold role.
W16 and W32 cannot keep 32 pairs, so this comparison is proportional by construction.

Width also changes the canonical Xavier scales and the number of initialization RNG draws:
`std1=sqrt(2/(inputs+hidden))`, `std2=sqrt(2/(hidden+outputs))`.
The output weights are drawn after the width-dependent encoder. Initialization and pair-sampling
RNG streams are separate, so changing width does not consume pair-sampler draws. These are
inherited trainer semantics, not an additional initialization intervention.

All retained fits use A’s 984 inputs, admitted TRAIN mixture and weights, 120 base epochs (fixed epoch 119),
then public_train ×256 continuation with original rehearsal, 60 epochs at 1e-5 (fixed epoch 59).
No EVAL result selects a width, checkpoint or setting. No AIC-2026 assessment statistics are committed.
Packing is canonical densify → f16 with protected final layer → CID22 TRAIN dial, E33 identity/tail2,
with all 984 inputs preserved and a native monotone certificate for each final bake.

The recipe’s final strip removes `zentrain.repro` and `zentrain.sample_coverage` into `shipped.bin`.
Native inspections of all layers/scalers/derived declarations/dial splines are exactly equal before
and after stripping. Assessments use the original packed bytes. Fit wall excludes this separate
metadata-only finalization; its execution receipt appears below.

Fit wall totals sum the recorded GNU time payload elapsed fields and exclude queued waits.
The W512 base elapsed field is printed at whole-second resolution; its recorded sum is not a centisecond-precise whole-fit measurement.

W4’s first setup attempt rejected the `H4` recipe token before optimization; no checkpoint was written.
The repair admits only the exact amendment-bound W4 specification and preserves the legacy default rejection.
That 1.24 s setup attempt is retained under `failed-attempts/W4-s0-recipe-parser-v1/` and excluded from the successful fit’s wall total.

The first capacity audit stopped because its explicit native certifier binding was absent.
Setting `VIZ_CERTIFIER` to the same frozen packing certifier repaired orchestration; the failed invocation and partial output remain under `failed-attempts/audit-certifier-env-v1/`.

| Width | Seed | Base final dev geomean3 | TRAIN dev MAE | E36 inversions (%) | All-nine MOS error (%) | Chroma level | Dead / residual / live | Largest pair (%) | Inputs 50/90/99% | Head (µs) | Fit wall (s) |
|---|---:|---:|---:|---:|---:|---|---|---:|---|---:|---:|
| 4 | 0 | 0.955900 | 3.289118 | 3.632725 | 8.25425 | fail | 2/1/1 | undefined | 12/60/118 | 2.568 | 252.38 |
| 16 | 0 | 0.957500 | 3.172636 | 3.458295 | 8.24386 | fail | 11/3/2 | 99.99986559 | 12/68/147 | 2.571 | 482.50 |
| 32 | 0 | 0.957500 | 3.173617 | 3.404791 | 8.18327 | fail | 23/7/2 | 99.99965369 | 12/67/146 | 2.898 | 716.21 |
| 64 | 0 | 0.957100 | 3.196792 | 3.388399 | 8.01711 | fail | 47/15/2 | 99.99921102 | 13/68/147 | 4.011 | 1244.90 |
| A (128) | 0 | 0.957200 | 3.225105 | 3.386235 | 7.94419 | reference A0 | 95/31/2 | 99.99846693 | 13/69/149 | paired A0 below | not refitted |
| A (128) | 1 | 0.955600 | 3.237627 | 3.362575 | 8.09039 | not rescored by this gate | 95/31/2 | 100.00000000 | 14/71/145 | not timed | not refitted |
| A (128) | 2 | 0.956600 | 3.393835 | 3.441131 | 8.04925 | not rescored by this gate | 125/1/2 | 99.99998959 | 14/72/151 | not timed | not refitted |
| 256 | 0 | 0.957200 | 3.211384 | 3.386544 | 7.97556 | fail | 191/63/2 | 99.99685395 | 13/68/149 | 10.634 | 4624.73 |
| 512 | 0 | 0.957300 | 3.484453 | 3.475305 | 8.03648 | fail | 508/0/4 | 53.26299170 | 13/71/154 | 20.831 | 8386.33 |

Base development geomean3 is recorded at the fixed final epoch; it does not select a different epoch.
TRAIN dev MAE pools SafeSyn and CID22 development rows in stored score-point units.
All-nine error averages collection-wise median residual / target p1–p99 span after assessment-only 4PL.
MDID is a training source; the full read tables separately show the eight other collections and common three.
Dead units have exact-zero encoder columns and nonpositive bias. Residual excludes dead units and has
|output coefficient| < 1e-3 of the largest. Largest pair share uses measured mean absolute raw contributions.
Concentration counts damage slots out of 492, jointly intervening on their direct and product paths;
single-slot effects overlap and do not form an additive score decomposition.

| Width | Matched A0 head (µs) | Clean paired rounds /30, candidate seed 0 |
|---|---|---|
| 4 | 5.872 | 4 |
| 16 | 5.389 | 9 |
| 32 | 6.332 | 17 |
| 64 | 6.272 | 19 |
| 256 | 6.144 | 23 |
| 512 | 6.607 | 16 |

Native head owner: R12’s pinned `Predictor::predict_transformed` paired zenbench instrument,
256 fixed positive logical rows, one call thread, default native dispatch, same clean-round indices.
Includes scaling/network forward; excludes extraction, derived products and dial. These are paired
descriptive means; no pooled confidence interval or full-image speed claim is implied.
W4 has only four clean rounds and W16 nine; their similar means do not establish a speed difference.

## Capacity on the HDR TRAIN feature bank

These are label-free forward/ablation diagnostics on the registered TRAIN bank, separate from the HDR quality panel.
Structural counts use the same fixed weights as the CID22 table; contributions and interventions use this bank’s rows.

| Width | Seed | Rows | Live max pairs | Largest pair (%) | Inputs 50/90/99% |
|---|---:|---:|---:|---:|---|
| 4 | 0 | 7390 | 0 | undefined | 11/54/112 |
| 16 | 0 | 7390 | 1 | 99.99986352 | 12/67/143 |
| 32 | 0 | 7390 | 1 | 99.99964650 | 12/66/139 |
| 64 | 0 | 7390 | 1 | 99.99918119 | 12/65/137 |
| 256 | 0 | 7390 | 1 | 99.99673102 | 12/65/138 |
| 512 | 0 | 7390 | 2 | 53.69431700 | 12/65/142 |

Full EVAL, E36, chroma, steering, C1–C6, identity, NEARID and five-view HDR tables:
[R13 measured reads](r13_hidden_width_reads_2026-11-04.md). HDR uses the registered cube-root:203 banks,
raw historical native scores before the dial, exactly as R12. HDR is report-only; no HDR leg was fitted.
These are reused mix/HC1/R12 assessments, not untouched confirmation. CLIC/CIQA/MMSPG and public_test stay closed.

The three A EVAL records, E36 control/population and five-view HDR point records exactly match
stored R12 outputs; A0’s five chroma anchor MAEs also match. `reads/CONTROL_REUSE_CHECK.json`
records this check. Additional verdict/identity executable hashes match the preserved October-28
tool manifest (`GATE_TOOL_PINS.json`); no executable or read input changed.

`FINAL_VALIDATION.json` verifies the six bakes, twelve capacity cells, six native timing pairs
and all 36 read receipts. Every steering packet retains the same 135 cases as A; identity has
744 rows per model and NEARID 648 rows / 24 references / 144 ladders. `read-source/` and
`read-tools/` preserve the pinned reader sources and executable bytes; `READ_TOOL_ARCHIVE.json`
maps their original locations to archived copies.

## Chroma level changes against A0

Score-point MAE change and 95% ladder bootstrap interval, from the native registered read.
Each width has at least one family with a strictly positive interval. All six pass the separate rank check.

| Family | W4 | W16 | W32 | W64 | W256 | W512 |
|---|---:|---:|---:|---:|---:|---:|
| ycc444 | -0.110317 [-0.235875, +0.017667] | +0.042015 [+0.001428, +0.082199] | +0.007294 [-0.019042, +0.034054] | +0.003856 [-0.005382, +0.013127] | -0.004782 [-0.010663, +0.001162] | +0.069431 [+0.029928, +0.109400] |
| ycc420 | +0.024216 [-0.104861, +0.155533] | +0.051795 [+0.015901, +0.088319] | +0.041907 [+0.014463, +0.070086] | +0.027742 [+0.018241, +0.037465] | +0.018503 [+0.012600, +0.024492] | -0.014247 [-0.055677, +0.026336] |
| xyb-capped | -0.085276 [-0.129643, -0.041589] | +0.115177 [+0.085644, +0.144496] | +0.042824 [+0.021790, +0.064126] | +0.019219 [+0.009911, +0.028709] | -0.000799 [-0.004934, +0.003460] | -0.001548 [-0.013565, +0.010966] |
| xyb-extended | +0.380209 [+0.249061, +0.502837] | -0.062619 [-0.144567, +0.020589] | -0.063882 [-0.119712, -0.006977] | -0.023258 [-0.042301, -0.003543] | -0.019227 [-0.029984, -0.008031] | +0.084771 [+0.048750, +0.122035] |
| band-ac | -0.081846 [-0.130355, -0.033038] | +0.032765 [+0.013339, +0.052315] | -0.007329 [-0.022996, +0.009127] | -0.004728 [-0.011459, +0.002133] | -0.015564 [-0.019768, -0.011304] | -0.001606 [-0.032656, +0.031497] |

## Individual dial contracts

Each cell is the measured value and native state. A lists seeds 0/1/2; candidates list seed 0.
C1 is monotonicity; C2 flat/clamp share; C3 negative-score share; C4 deepest negative score;
C5 identity-band violations; C6 cells above identity. C1–C6 pass on both grids.
This does not grade G-ADDR: its floor registry/mentor-truth inputs remain absent, as in R12.

### standard

| Contract | A | W4 | W16 | W32 | W64 | W256 | W512 |
|---|---:|---:|---:|---:|---:|---:|---:|
| C1 | 0.997453 (pass) / 0.998379 (pass) / 0.997916 (pass) | 0.994905 (pass) | 0.996989 (pass) | 0.997221 (pass) | 0.997453 (pass) | 0.997453 (pass) | 0.997684 (pass) |
| C2 | 0.000000 (pass) / 0.000000 (pass) / 0.000000 (pass) | 0.000000 (pass) | 0.000000 (pass) | 0.000000 (pass) | 0.000000 (pass) | 0.000000 (pass) | 0.000000 (pass) |
| C3 | 0.768000 (pass) / 0.741000 (pass) / 0.723500 (pass) | 0.816000 (pass) | 0.772500 (pass) | 0.770500 (pass) | 0.769000 (pass) | 0.768500 (pass) | 0.716000 (pass) |
| C4 | -204.532324 (pass) / -201.435230 (pass) / -201.361266 (pass) | -309.251580 (pass) | -209.253748 (pass) | -203.070431 (pass) | -203.713333 (pass) | -203.654904 (pass) | -187.395638 (pass) |
| C5 | 0.000000 (pass) / 0.000000 (pass) / 0.000000 (pass) | 0.000000 (pass) | 0.000000 (pass) | 0.000000 (pass) | 0.000000 (pass) | 0.000000 (pass) | 0.000000 (pass) |
| C6 | 0.000000 (pass) / 0.000000 (pass) / 0.000000 (pass) | 0.000000 (pass) | 0.000000 (pass) | 0.000000 (pass) | 0.000000 (pass) | 0.000000 (pass) | 0.000000 (pass) |

### ladder

| Contract | A | W4 | W16 | W32 | W64 | W256 | W512 |
|---|---:|---:|---:|---:|---:|---:|---:|
| C1 | 0.992349 (pass) / 0.993837 (pass) / 0.992349 (pass) | 0.986824 (pass) | 0.992031 (pass) | 0.992137 (pass) | 0.992243 (pass) | 0.992349 (pass) | 0.992031 (pass) |
| C2 | 0.000000 (pass) / 0.000000 (pass) / 0.000000 (pass) | 0.000000 (pass) | 0.000000 (pass) | 0.000000 (pass) | 0.000000 (pass) | 0.000000 (pass) | 0.000000 (pass) |
| C3 | 0.768000 (pass) / 0.741000 (pass) / 0.723500 (pass) | 0.816000 (pass) | 0.772500 (pass) | 0.770500 (pass) | 0.769000 (pass) | 0.768500 (pass) | 0.716000 (pass) |
| C4 | -204.532324 (pass) / -201.435230 (pass) / -201.361266 (pass) | -309.251580 (pass) | -209.253748 (pass) | -203.070431 (pass) | -203.713333 (pass) | -203.654904 (pass) | -187.395638 (pass) |
| C5 | 0.000000 (pass) / 0.000000 (pass) / 0.000000 (pass) | 0.000000 (pass) | 0.000000 (pass) | 0.000000 (pass) | 0.000000 (pass) | 0.000000 (pass) | 0.000000 (pass) |
| C6 | 0.000000 (pass) / 0.000000 (pass) / 0.000000 (pass) | 0.000000 (pass) | 0.000000 (pass) | 0.000000 (pass) | 0.000000 (pass) | 0.000000 (pass) | 0.000000 (pass) |

## Interpretation

Training width is load-bearing for the fitted solution and measured outcomes here, although candidate
across-seed stability is unmeasured. With the full continuation, W4 still has 8.25425% all-nine MOS
error versus A’s 7.94419/8.09039/8.04925%, and its E36 change against A’s seed mean is
+0.236079 percentage points [0.061760, 0.420477]. W16 also worsens both; W32 worsens MOS
while its E36 interval includes zero. W64 and W256 sit within A’s MOS seed range and their E36
intervals include zero; this does not establish W128 as necessary or optimal. W512 retains two
pairs with 53.263/46.737% of CID22 contribution (53.694/46.306% on HDR TRAIN), yet its E36
inversion share worsens. Collapse to one pair is therefore not universal. The packed W4 has
one live unit and one residual unit, with no coupled max pair detected; this is not an assertion
of exact global linearity. W4/W16/W512 beat A’s steering counts at 135/135 versus 133/134/134,
but all widths fail chroma-level; W4’s extended-XYB MAE rises by 0.380209 points
[0.249061, 0.502837]. All pass the registered dial/identity/NEARID bars, while W4 and W512
each lose one monotone NEARID ladder (131 versus A’s 132). No width gives an across-gate win.
Width includes the proportional pair allocation and inherited Xavier/RNG semantics above;
the experiment does not isolate those mechanisms.

HDR gains are mixed: W4 improves 4Kdtb within-reference inversions to 43 versus A’s 45/44/45,
but has 53 ladder inversions versus 50/51/49 and 125 UPIQ-all inversions versus 121/121/124.
Its teacher-VAL pooled SROCC improves to 0.899412 versus A’s 0.891399 seed mean, while its
teacher within-reference ordering loses one pair. These report-only results do not establish
a consistent HDR gain or authorize an HDR fit.

## Frozen final bakes

| Width | Seed | Packed SHA-256 | Bytes |
|---|---:|---|---:|
| W32 | 0 | `beaf83829ee6899f5474a486c7580fc7773daada9bb2a8c36c0c7ef988de171f` | 34156 |
| W256 | 0 | `5097267b5d4d74369169c48d74c883010158dcffcb36cda0ca0f4ec35ca2c387` | 37040 |
| W64 | 0 | `332ee1188188d4cae23f9b17c51d95e9629c1819159669a23ab19c6ef367face` | 35213 |
| W512 | 0 | `472256e7488b66dcd46b9387095f0901dc18c4edcd45606e7be53d63bdc89335` | 38790 |
| W16 | 0 | `472e51b0ff8f4f6a4828ef2a75e9250658c41bc5f7f0f427d7ea833b9cdbcf3c` | 34623 |
| W4 | 0 | `53ca43065c5d7bf33f1123851845af53b5f4bf8df353b0d457c74889600a2c00` | 27818 |

Complete fit checkpoints, source/binary identities, capacities on CID22 and HDR TRAIN banks,
per-job GNU time and run-heavy receipts, and raw read JSON are preserved under
`/var/tmp/byc-width-20261104/` (verified on dev). Mirror:
`/mnt/tower/output/zensim/r13-width-20261104/`; verification receipt `ARCHIVE_VERIFIED.json`.
Every base/continuation/pack/audit/read job takes and releases the shared heavy lock individually.

## Per-job resource receipts

- `logs/W16-s0-base.log`: run-heavy: done rc=0 302s | peak-RSS 2.05GiB | min-avail 38630MiB | peak-load 2.07
- `logs/W16-s0-continuation.log`: run-heavy: done rc=0 180s | peak-RSS 2.05GiB | min-avail 38708MiB | peak-load 1.90
- `logs/W16-s0-pack.log`: run-heavy: done rc=0 1s | peak-RSS 0.28GiB | min-avail 40536MiB | peak-load 1.43
- `logs/W256-s0-base.log`: run-heavy: done rc=0 3422s | peak-RSS 1.96GiB | min-avail 38818MiB | peak-load 2.42
- `logs/W256-s0-continuation.log`: run-heavy: done rc=0 1203s | peak-RSS 1.95GiB | min-avail 39172MiB | peak-load 2.28
- `logs/W256-s0-pack.log`: run-heavy: done rc=0 1s | peak-RSS 0.28GiB | min-avail 41719MiB | peak-load 1.86
- `logs/W32-s0-base.log`: run-heavy: done rc=0 472s | peak-RSS 1.91GiB | min-avail 39073MiB | peak-load 3.10
- `logs/W32-s0-continuation.log`: run-heavy: done rc=0 244s | peak-RSS 1.99GiB | min-avail 39486MiB | peak-load 2.02
- `logs/W32-s0-pack.log`: run-heavy: done rc=0 1s | peak-RSS 0.28GiB | min-avail 42010MiB | peak-load 1.38
- `logs/W4-s0-base.log`: run-heavy: done rc=0 125s | peak-RSS 2.11GiB | min-avail 38677MiB | peak-load 1.29
- `logs/W4-s0-continuation.log`: run-heavy: done rc=0 127s | peak-RSS 2.12GiB | min-avail 38707MiB | peak-load 1.48
- `logs/W4-s0-pack.log`: run-heavy: done rc=0 1s | peak-RSS 0.28GiB | min-avail 41092MiB | peak-load 1.36
- `logs/W512-s0-base.log`: run-heavy: done rc=0 6227s | peak-RSS 2.01GiB | min-avail 38515MiB | peak-load 5.01
- `logs/W512-s0-continuation.log`: run-heavy: done rc=0 2159s | peak-RSS 1.99GiB | min-avail 38653MiB | peak-load 4.32
- `logs/W512-s0-pack.log`: run-heavy: done rc=0 1s | peak-RSS 0.29GiB | min-avail 41160MiB | peak-load 2.25
- `logs/W64-s0-base.log`: run-heavy: done rc=0 864s | peak-RSS 1.99GiB | min-avail 39153MiB | peak-load 2.93
- `logs/W64-s0-continuation.log`: run-heavy: done rc=0 381s | peak-RSS 1.98GiB | min-avail 39230MiB | peak-load 3.39
- `logs/W64-s0-pack.log`: run-heavy: done rc=0 1s | peak-RSS 0.28GiB | min-avail 41794MiB | peak-load 2.55
- `logs/ablation-cid22-W16_s0.log`: run-heavy: done rc=0 40s | peak-RSS 0.13GiB | min-avail 40449MiB | peak-load 1.90
- `logs/ablation-cid22-W256_s0.log`: run-heavy: done rc=0 106s | peak-RSS 0.14GiB | min-avail 40712MiB | peak-load 1.77
- `logs/ablation-cid22-W32_s0.log`: run-heavy: done rc=0 43s | peak-RSS 0.13GiB | min-avail 40801MiB | peak-load 1.72
- `logs/ablation-cid22-W4_s0.log`: run-heavy: done rc=0 39s | peak-RSS 0.13GiB | min-avail 40657MiB | peak-load 1.50
- `logs/ablation-cid22-W512_s0.log`: run-heavy: done rc=0 187s | peak-RSS 0.14GiB | min-avail 40192MiB | peak-load 1.98
- `logs/ablation-cid22-W64_s0.log`: run-heavy: done rc=0 50s | peak-RSS 0.13GiB | min-avail 40628MiB | peak-load 1.88
- `logs/ablation-hdr-W16_s0.log`: run-heavy: done rc=0 28s | peak-RSS 0.13GiB | min-avail 40627MiB | peak-load 1.84
- `logs/ablation-hdr-W256_s0.log`: run-heavy: done rc=0 76s | peak-RSS 0.13GiB | min-avail 40533MiB | peak-load 1.67
- `logs/ablation-hdr-W32_s0.log`: run-heavy: done rc=0 30s | peak-RSS 0.13GiB | min-avail 40604MiB | peak-load 1.48
- `logs/ablation-hdr-W4_s0.log`: run-heavy: done rc=0 27s | peak-RSS 0.13GiB | min-avail 40572MiB | peak-load 2.15
- `logs/ablation-hdr-W512_s0.log`: run-heavy: done rc=0 130s | peak-RSS 0.14GiB | min-avail 40551MiB | peak-load 1.72
- `logs/ablation-hdr-W64_s0.log`: run-heavy: done rc=0 35s | peak-RSS 0.13GiB | min-avail 40517MiB | peak-load 1.64
- `logs/audit-cid22-W16_s0.log`: run-heavy: done rc=0 9s | peak-RSS 0.26GiB | min-avail 40612MiB | peak-load 1.80
- `logs/audit-cid22-W256_s0.log`: run-heavy: done rc=0 10s | peak-RSS 0.31GiB | min-avail 40720MiB | peak-load 1.83
- `logs/audit-cid22-W32_s0.log`: run-heavy: done rc=0 10s | peak-RSS 0.26GiB | min-avail 40866MiB | peak-load 2.45
- `logs/audit-cid22-W4_s0.log`: run-heavy: done rc=0 9s | peak-RSS 0.26GiB | min-avail 40552MiB | peak-load 1.52
- `logs/audit-cid22-W512_s0.log`: run-heavy: done rc=0 10s | peak-RSS 0.31GiB | min-avail 40488MiB | peak-load 1.72
- `logs/audit-cid22-W64_s0.log`: run-heavy: done rc=0 9s | peak-RSS 0.27GiB | min-avail 40377MiB | peak-load 1.79
- `logs/audit-hdr-W16_s0.log`: run-heavy: done rc=0 6s | peak-RSS 0.16GiB | min-avail 40482MiB | peak-load 1.65
- `logs/audit-hdr-W256_s0.log`: run-heavy: done rc=0 5s | peak-RSS 0.21GiB | min-avail 40613MiB | peak-load 1.29
- `logs/audit-hdr-W32_s0.log`: run-heavy: done rc=0 5s | peak-RSS 0.17GiB | min-avail 40620MiB | peak-load 1.58
- `logs/audit-hdr-W4_s0.log`: run-heavy: done rc=0 6s | peak-RSS 0.16GiB | min-avail 40637MiB | peak-load 1.86
- `logs/audit-hdr-W512_s0.log`: run-heavy: done rc=0 6s | peak-RSS 0.26GiB | min-avail 40512MiB | peak-load 1.30
- `logs/audit-hdr-W64_s0.log`: run-heavy: done rc=0 5s | peak-RSS 0.18GiB | min-avail 40549MiB | peak-load 1.47
- `logs/baseline-clippy.log`: run-heavy: done rc=0 57s | peak-RSS 1.07GiB | min-avail 39456MiB | peak-load 5.81
- `logs/build.log`: run-heavy: done rc=0 51s | peak-RSS 1.73GiB | min-avail 37648MiB | peak-load 5.20
- `logs/clippy.log`: run-heavy: done rc=0 1s | peak-RSS 0.38GiB | min-avail 41591MiB | peak-load 4.81
- `logs/development-cid22-F_s0.log`: run-heavy: done rc=0 0s | peak-RSS 0.19GiB | min-avail 40681MiB | peak-load 1.27
- `logs/development-cid22-F_s1.log`: run-heavy: done rc=0 0s | peak-RSS 0.19GiB | min-avail 40694MiB | peak-load 1.16
- `logs/development-cid22-F_s2.log`: run-heavy: done rc=0 0s | peak-RSS 0.19GiB | min-avail 40792MiB | peak-load 1.16
- `logs/development-cid22-W16_s0.log`: run-heavy: done rc=0 1s | peak-RSS 0.19GiB | min-avail 40636MiB | peak-load 1.27
- `logs/development-cid22-W256_s0.log`: run-heavy: done rc=0 0s | peak-RSS 0.19GiB | min-avail 40630MiB | peak-load 1.40
- `logs/development-cid22-W32_s0.log`: run-heavy: done rc=0 0s | peak-RSS 0.19GiB | min-avail 40715MiB | peak-load 1.40
- `logs/development-cid22-W4_s0.log`: run-heavy: done rc=0 0s | peak-RSS 0.19GiB | min-avail 40618MiB | peak-load 1.27
- `logs/development-cid22-W512_s0.log`: run-heavy: done rc=0 1s | peak-RSS 0.19GiB | min-avail 40539MiB | peak-load 1.29
- `logs/development-cid22-W64_s0.log`: run-heavy: done rc=0 0s | peak-RSS 0.19GiB | min-avail 40615MiB | peak-load 1.29
- `logs/development-safesyn-F_s0.log`: run-heavy: done rc=0 1s | peak-RSS 0.97GiB | min-avail 40725MiB | peak-load 1.35
- `logs/development-safesyn-F_s1.log`: run-heavy: done rc=0 1s | peak-RSS 0.97GiB | min-avail 40592MiB | peak-load 1.35
- `logs/development-safesyn-F_s2.log`: run-heavy: done rc=0 1s | peak-RSS 0.97GiB | min-avail 40675MiB | peak-load 1.35
- `logs/development-safesyn-W16_s0.log`: run-heavy: done rc=0 1s | peak-RSS 0.97GiB | min-avail 40814MiB | peak-load 1.47
- `logs/development-safesyn-W256_s0.log`: run-heavy: done rc=0 1s | peak-RSS 0.97GiB | min-avail 40979MiB | peak-load 0.56
- `logs/development-safesyn-W32_s0.log`: run-heavy: done rc=0 1s | peak-RSS 0.97GiB | min-avail 41157MiB | peak-load 0.56
- `logs/development-safesyn-W4_s0.log`: run-heavy: done rc=0 1s | peak-RSS 0.97GiB | min-avail 40857MiB | peak-load 1.47
- `logs/development-safesyn-W512_s0.log`: run-heavy: done rc=0 2s | peak-RSS 0.98GiB | min-avail 40681MiB | peak-load 1.42
- `logs/development-safesyn-W64_s0.log`: run-heavy: done rc=0 1s | peak-RSS 0.97GiB | min-avail 40945MiB | peak-load 1.46
- `logs/head-W16_s0.log`: run-heavy: done rc=0 10s | peak-RSS 0.19GiB | min-avail 42114MiB | peak-load 1.92
- `logs/head-W256_s0.log`: run-heavy: done rc=0 10s | peak-RSS 0.19GiB | min-avail 40612MiB | peak-load 1.87
- `logs/head-W32_s0.log`: run-heavy: done rc=0 10s | peak-RSS 0.18GiB | min-avail 40674MiB | peak-load 2.14
- `logs/head-W4_s0.log`: run-heavy: done rc=0 10s | peak-RSS 0.19GiB | min-avail 41807MiB | peak-load 5.40
- `logs/head-W512_s0.log`: run-heavy: done rc=0 10s | peak-RSS 0.19GiB | min-avail 41966MiB | peak-load 2.59
- `logs/head-W64_s0.log`: run-heavy: done rc=0 10s | peak-RSS 0.19GiB | min-avail 42058MiB | peak-load 2.66
- `logs/metadata-strip.log`: run-heavy: done rc=0 0s | peak-RSS 0.05GiB | min-avail 40833MiB | peak-load 2.49
- `logs/native-admission-tests.log`: run-heavy: done rc=0 62s | peak-RSS 1.83GiB | min-avail 37472MiB | peak-load 4.80
- `logs/read-chroma.log`: run-heavy: done rc=0 12s | peak-RSS 2.95GiB | min-avail 39732MiB | peak-load 5.76
- `logs/read-dial-W16_s0.log`: run-heavy: done rc=0 52s | peak-RSS 0.25GiB | min-avail 41066MiB | peak-load 6.05
- `logs/read-dial-W256_s0.log`: run-heavy: done rc=0 52s | peak-RSS 0.25GiB | min-avail 41996MiB | peak-load 7.98
- `logs/read-dial-W32_s0.log`: run-heavy: done rc=0 53s | peak-RSS 0.25GiB | min-avail 41911MiB | peak-load 6.08
- `logs/read-dial-W4_s0.log`: run-heavy: done rc=0 52s | peak-RSS 0.25GiB | min-avail 41027MiB | peak-load 6.19
- `logs/read-dial-W512_s0.log`: run-heavy: done rc=0 52s | peak-RSS 0.25GiB | min-avail 41232MiB | peak-load 5.26
- `logs/read-dial-W64_s0.log`: run-heavy: done rc=0 52s | peak-RSS 0.25GiB | min-avail 41932MiB | peak-load 2.83
- `logs/read-e36.log`: run-heavy: done rc=0 107s | peak-RSS 7.69GiB | min-avail 34984MiB | peak-load 3.98
- `logs/read-eval.log`: run-heavy: done rc=0 6s | peak-RSS 0.25GiB | min-avail 41642MiB | peak-load 3.05
- `logs/read-hdr.log`: run-heavy: done rc=0 1104s | peak-RSS 0.65GiB | min-avail 41080MiB | peak-load 3.24
- `logs/read-nearid-W16_s0.log`: run-heavy: done rc=0 10s | peak-RSS 0.25GiB | min-avail 41107MiB | peak-load 3.67
- `logs/read-nearid-W256_s0.log`: run-heavy: done rc=0 10s | peak-RSS 0.25GiB | min-avail 41889MiB | peak-load 5.40
- `logs/read-nearid-W32_s0.log`: run-heavy: done rc=0 10s | peak-RSS 0.25GiB | min-avail 41874MiB | peak-load 2.48
- `logs/read-nearid-W4_s0.log`: run-heavy: done rc=0 11s | peak-RSS 0.25GiB | min-avail 41091MiB | peak-load 3.52
- `logs/read-nearid-W512_s0.log`: run-heavy: done rc=0 11s | peak-RSS 0.25GiB | min-avail 40757MiB | peak-load 3.17
- `logs/read-nearid-W64_s0.log`: run-heavy: done rc=0 10s | peak-RSS 0.25GiB | min-avail 41906MiB | peak-load 2.00
- `logs/read-steer-W16_s0.log`: run-heavy: done rc=0 58s | peak-RSS 0.25GiB | min-avail 40350MiB | peak-load 6.74
- `logs/read-steer-W256_s0.log`: run-heavy: done rc=0 59s | peak-RSS 0.25GiB | min-avail 41120MiB | peak-load 6.12
- `logs/read-steer-W32_s0.log`: run-heavy: done rc=0 57s | peak-RSS 0.25GiB | min-avail 41442MiB | peak-load 5.79
- `logs/read-steer-W4_s0.log`: run-heavy: done rc=0 57s | peak-RSS 0.25GiB | min-avail 40450MiB | peak-load 6.65
- `logs/read-steer-W512_s0.log`: run-heavy: done rc=0 61s | peak-RSS 0.25GiB | min-avail 41349MiB | peak-load 5.76
- `logs/read-steer-W64_s0.log`: run-heavy: done rc=0 59s | peak-RSS 0.25GiB | min-avail 41321MiB | peak-load 7.19
- `w4-build.log`: run-heavy: done rc=0 106s | peak-RSS 1.83GiB | min-avail 39172MiB | peak-load 3.38
- `failed-attempts/W4-s0-recipe-parser-v1/logs/W4-s0-base.log`: run-heavy: done rc=1 1s | peak-RSS 0.17GiB | min-avail 40793MiB | peak-load 1.32
- `failed-attempts/audit-certifier-env-v1/logs/audit-cid22-W32_s0.log`: run-heavy: done rc=1 2s | peak-RSS 0.13GiB | min-avail 40866MiB | peak-load 1.15
