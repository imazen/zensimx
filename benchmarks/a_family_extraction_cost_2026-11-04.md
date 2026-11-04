# Public A: low-effect-family extraction cost, 2026-11-04

This study changes no feature arithmetic, model bytes, scores, or public API.
A is the adopted mixF3 seed-0 bake from the [weight audit](a_weight_audit_2026-11-04.md),
SHA-256 `185f43f7e5e1327c34ad18f0b73b96b872663726a0fac6ff524430d0d8430088`.
The question is extraction work saved by omitting the audit's eleven low-effect
signal families, rather than a quality verdict on that omission.

## Method and pins

Perf attribution and the tiny/small timings use **r5900xt, AMD Ryzen 9 5900XT,
Ubuntu 26.04.1, AVX2/v3 (Zen 3)**. After the owner released i270 at
16:53:14 UTC, four unstarted second-retry cells (1024²/large × 1/16 threads)
moved to **i270, Intel Core Ultra 7 270K Plus, Ubuntu 26.04, AVX2/v3**.
The frozen executable was built on i270 and copied with its SHA-256 verified;
it uses runtime dispatch rather than a host-native build. One-thread runs
are pinned to CPU 2; sixteen-thread runs use CPUs 0–15. On r5900xt this is
sixteen physical cores with SMT siblings 16–31 excluded. i270 is a hybrid
24-core CPU without SMT; its 0–15 affinity includes eight P cores and eight
E cores. The observed kernel PMU CPU lists are `cpu_core: 0–7` and
`cpu_atom: 8–23` under `/sys/bus/event_source/devices/`.
Every table labels its host, and fits never combine hosts. These numbers do
not measure Zen 4, Zen 5, AVX-512, ARM, or HDR performance. Both hosts' observed
CPU governors are `powersave`; no governor was changed.

The extraction/runtime source is `61d77a13352f6e845419008a2737e5f892a42421`;
the original metric source export was based on `9c5fd910`. The only subsequent
dependency change used in the final executable is the benchmark-only zenbench
pin and its clap dependencies. zenbench is `0.1.10`, revision
`82ff70a3f0c212cf88f53d6e9c3e02de13f32eec`; zenresize is revision
`e3975fb9d6d6b7baa96038a0eb8e27febb37c012`; zenjpeg is `0.8.4`;
zenpng is `0.1.4`; archmage is `0.9.30`. The final binary, source receipts,
lockfile and commands are retained with the raw evidence.
The executable SHA-256 is
`cdf7e8a2bfefe6a7030253581cbd171928c14f156e9b4d0fd2e101be9b488072`;
the benchmark Rust source SHA-256 is
`058017aec7657d8664a77616a1ea738431e04944fa9449616f3b9150767dbdc2`.
The source hash agrees between the committed owner and the remote build export.

The three timed arms are:

- `extract492`: `research::Request::for_bake_bytes(A)` through
  `research::extract`, with all 492 damage IDs and 12 fragility IDs.
- `extract336`: the same request width and revision, omitting the 156 damage
  IDs identified by `inputs-A0.tsv`. This represents removing those damage
  inputs and their 156 fragility products. The remaining 336 damage inputs
  still require all 12 fragility IDs: 348 raw IDs remain requested.
- `compute_A`: the unchanged `BakeScorer::compute` surface, including A's
  network and score mapping, with its normal reusable scratch.

The reduced arm uses the brief's research-request fallback. `BakeScorer` has
no external read-set override; changing a bake or inventing a reduced score
would mix a model change into this extraction experiment. Research requests
include per-call planning, allocation and provenance generation. Their fixed
cost therefore differs from the cached complete scorer. The reduced-request
timings are not claimed as complete reduced-model scoring times.

Each arm gets a 100 ms zenbench warmup target per image, followed by 32
randomized paired rounds with exactly one call per arm. There are 20 images
per size. The owner capped collection at the primary pass and two retry
passes across both hosts; host migration does not reset that count. The
selector asserts at most three native attempts for every image/thread cell. An image is a cell; one busy image does not invalidate a different
image's quiet attempt. Foreign `rustc`, `cargo`, `aws` or `perf` rejects the
affected image attempt. New runs record process snapshots before and after
each image and a 0.5-second census during it, in addition to zenbench's trace.
Earlier completed runs retain their per-image zenbench traces and are marked
`legacy ... direct census unrecorded` in the selection manifest; a direct
census receipt is never claimed retroactively.

The percentiles use the first **16 clean paired calls per image**, in
chronological order across admitted attempts. A resource-flagged round is
excluded jointly from all three arms. Fewer than 16 admitted calls is
**not measured (contended)**, with the image and evidence retained. Selection
never uses latency or arm ordering. Zenbench's unreliable status refuses an
attempt. Every table gives its measured image coverage rather than silently
substituting another image. The host is shared, not exclusive. Other sessions'
processes are never stopped. The normal background processes remain visible
to the resource gate; the exact benchmark worker and its admission monitor
are registered as benchmark work. For zenbench timings, standard foreign CPU,
heavy-process, temperature and memory checks remain enabled. Perf captures use
the explicit foreign-process census; they do not run zenbench's per-round
resource gate.
Calibration workloads and stack jitter are disabled. Every build and timed
run is serialized by the same host lock and capped by run-heavy at 16 GiB.
Release uses thin LTO, debug symbols and frame pointers, without
`-C target-cpu=native`.

All pairs are prepared before timing. The tiny/small cells use ten CID22
training sources and ten CLIC-1024 training sources, center-cropped to square
then Mitchell-downscaled to 64²/256². The medium cell uses twenty native
1024² CLIC-1024 training images. The large cell uses five canonical png-v3 SDR
TRAIN photographs from each of four imazen-26 folders, at native size when
their long edge is at most 4096 and Mitchell-downscaled otherwise. No image
is upscaled. Each prepared reference gets a real zenjpeg q40, 4:2:0 re-encode
and zenjpeg decode. The encoded JPEGs and both RGB buffers are retained and
hashed. No subjective labels or validation/test images are read.

Before timing each cell, the harness compares every retained raw feature's
f64 bits between the full request, reduced request and complete A scorer.
perf records full A scoring at one thread, one size at a time. FIFO control
enables sampling only after decode, hash checks and parity/warmup; it disables
sampling immediately after the scoring loop.

## Planner result

Both requests emit `basic+v2@w720/unknown#ee342d8c`: all 156 basic and all
348 v2 slots, 504 populated slots in a 720-position sparse layout. Removing
156 requested damage IDs does **not** remove a scale, channel, pyramid,
SSIM moment pass, dense pass, gradient pass, masking pass, or pooling pass.
The full/reduced provenance manifests agree. This is a measured producer
identity and a source-level work-boundary finding, not an estimate of a new
fine-grained implementation.

`feature_plan.rs::Plan::derive_with_layout_at` turns on the basic block when
any basic slot is read, and turns on `v2_blocks`, `gradient`, `blockiness`
and `transducer_bank` together for these v2 requests. All channel/scale cells
remain needed by retained inputs. `feature_v2_oracle.rs`'s revision-7
`dense_block_kernel_canon64_vec_v3` computes SSIM, deviations, bounded errors,
all eleven weighted pools and transducers in the same loop.
`fused_vblur_ssim_canon64_vec_v3` computes basic SSIM/MSE/contrast alongside
retained edge/detail/texture signals. `finish_channel_scale_at_revision`
finishes the v2 output pools together; `foldapp_streaming_walk_impl` finishes
edge-width change from adjacent-scale gradient sums regardless of an
individual edge-width output read. Those existing boundaries cannot skip
the eleven signals individually without an implementation change.

## Measured results

Across hosts, **155/160 distinct image/thread pairs** have sixteen admitted paired calls on at least one host. Five remain **not measured (contended)**. Each size was prepared and attempted with twenty images. No calls, percentiles or fits mix host clocks. r5900xt contributes 121 host-specific records and i270 75; overlapping records are not counted twice in 155/160.

Of the 121 r5900xt records, 65 contain selected calls admitted from legacy per-image zenbench traces without separately recorded census snapshots. That limitation is retained explicitly; those calls are not described as direct-census observations. The other r5900xt records and all 75 i270 records have direct admission receipts. i270 intentionally did not collect tiny/small cells.

### Timings

All numbers are **AVX2/v3**, milliseconds **p50 / p95** of individual calls. Each admitted image contributes exactly sixteen calls per arm. `compute_A` is the complete unchanged scorer; the two extraction arms are the research-request fallback.

| Host | Size | Threads | Extract 492 | Extract 336 | Complete A | Images | Calls/arm |
|---|---|---:|---:|---:|---:|---:|---:|
| r5900xt | 64 | 1 | 0.7766 / 0.8586 | 0.8450 / 0.9320 | 0.3586 / 0.4114 | 20/20 | 320 |
| r5900xt | 256 | 1 | 3.4917 / 5.3005 | 3.5616 / 5.5376 | 2.7060 / 2.8576 | 20/20 | 320 |
| r5900xt | 1024 | 1 | not measured (contended) | not measured (contended) | not measured (contended) | 0/20 | 0 |
| r5900xt | large | 1 | 502.2038 / 525.1857 | 500.1238 / 535.5811 | 470.8121 / 490.2268 | 10/20 | 160 |
| r5900xt | 64 | 16 | 0.9566 / 1.0773 | 1.0290 / 1.1730 | 0.4515 / 0.5899 | 18/20 | 288 |
| r5900xt | 256 | 16 | 2.2608 / 2.9079 | 2.3234 / 2.9415 | 1.4296 / 1.7705 | 20/20 | 320 |
| r5900xt | 1024 | 16 | 22.4031 / 28.2592 | 22.5828 / 30.7896 | 15.8458 / 18.7811 | 19/20 | 304 |
| r5900xt | large | 16 | 302.2195 / 322.7494 | 301.4822 / 324.2631 | 280.1481 / 299.3801 | 14/20 | 224 |
| i270 | 1024 | 1 | 29.8140 / 33.9601 | 29.8839 / 34.0244 | 27.3726 / 30.1008 | 20/20 | 320 |
| i270 | large | 1 | 343.5207 / 371.7594 | 343.0613 / 370.2414 | 327.4940 / 357.5431 | 15/20 | 240 |
| i270 | 1024 | 16 | 12.2007 / 17.2550 | 12.2791 / 17.0526 | 7.5965 / 13.5730 | 20/20 | 320 |
| i270 | large | 16 | 106.9843 / 120.5706 | 106.2575 / 120.8391 | 95.0313 / 105.2785 | 20/20 | 320 |

### Intercept and slope

OLS uses admitted per-image medians: **time (ms) = α (ms) + β (ms/MP) × measured megapixels**. r5900xt fits use its available size-floor records; i270 fits use only native 1024² and large photographs, so its intercept is especially weak evidence of small-image fixed work. Cohorts differ in content. These are in-process fits, not fresh-process startup measurements; negative intercepts are unconstrained fit results, not physical negative costs.

| Host | Threads | Arm | α ms | β ms/MP | R² | Admitted images / four-size floor |
|---|---:|---|---:|---:|---:|---:|
| r5900xt | 1 | `extract492` | 0.75401 | 40.47175 | 0.99640 | 50/80 |
| r5900xt | 1 | `extract336` | 0.85194 | 40.66472 | 0.99648 | 50/80 |
| r5900xt | 1 | `compute_A` | 0.18290 | 38.50441 | 0.99823 | 50/80 |
| r5900xt | 16 | `extract492` | -0.59571 | 25.29425 | 0.99920 | 71/80 |
| r5900xt | 16 | `extract336` | -0.48519 | 25.27889 | 0.99933 | 71/80 |
| r5900xt | 16 | `compute_A` | -2.69100 | 23.54911 | 0.99865 | 71/80 |
| i270 | 1 | `extract492` | -0.02782 | 28.64523 | 0.99836 | 35/80 |
| i270 | 1 | `extract336` | 0.11834 | 28.62954 | 0.99843 | 35/80 |
| i270 | 1 | `compute_A` | -1.70925 | 27.65953 | 0.99886 | 35/80 |
| i270 | 16 | `extract492` | 3.03540 | 8.78604 | 0.99600 | 40/80 |
| i270 | 16 | `extract336` | 3.06400 | 8.70047 | 0.99469 | 40/80 |
| i270 | 16 | `compute_A` | -0.80443 | 8.05520 | 0.99657 | 40/80 |

### Paired saving and gaps

Positive values below mean the reduced request saves time. These are means of within-image median differences, with 10,000 seeded image-cluster bootstrap resamples (seed 20261104). They are not differences between the pooled p50 columns. No positive saving has a 95% interval excluding zero. This does not establish equivalence or bound a future kernel rewrite.

| Host | Size | Threads | Full minus reduced ms | Bootstrap 95% interval ms |
|---|---|---:|---:|---:|
| r5900xt | 64 | 1 | -0.067843 | [-0.073364, -0.062160] |
| r5900xt | 256 | 1 | -0.139820 | [-0.330982, -0.010816] |
| r5900xt | large | 1 | -2.424231 | [-4.186001, -0.856916] |
| r5900xt | 64 | 16 | -0.076678 | [-0.085820, -0.066771] |
| r5900xt | 256 | 16 | -0.037432 | [-0.083132, 0.011979] |
| r5900xt | 1024 | 16 | -0.245071 | [-0.541048, 0.037133] |
| r5900xt | large | 16 | 0.129120 | [-0.905496, 1.092854] |
| i270 | 1024 | 1 | -0.126118 | [-0.214997, -0.040497] |
| i270 | large | 1 | 0.038382 | [-0.268720, 0.370153] |
| i270 | 1024 | 16 | 0.037405 | [-0.321569, 0.384964] |
| i270 | large | 16 | 1.013323 | [-0.236568, 2.240616] |

These five pairs never reach sixteen qualifying calls on any single host. Host clocks are never pooled to rescue them. The complete attempt timestamps, raw gate flags, image admission and source hashes are in the two `accepted-*/selection.json` files.

| Cell | Image | Qualifying r5900xt rounds | Qualifying i270 rounds | State |
|---|---|---:|---:|---|
| 64-16t | `64_00` | 11 | not collected | not measured (contended) |
| 64-16t | `64_01` | 2 | not collected | not measured (contended) |
| large-1t | `large_09` | 10 | 14 | not measured (contended) |
| large-1t | `large_13` | 2 | 6 | not measured (contended) |
| large-1t | `large_14` | 13 | 4 | not measured (contended) |

The five i270 large/1-thread gaps are `large_07` (2 clean), `large_09` (14), `large_13` (6), `large_14` (4), and `large_18` (4). `large_07` and `large_18` qualify separately on r5900xt; they are not inserted into the i270 percentile or fit. Both tiny/16-thread gaps had 96 native rounds across their three attempts; only 11 and 2 respectively qualified after image admission and round gates. Rejected attempts and pre-epoch samples remain in the evidence.

**Answer to the owner:** removing these 156 low-effect damage IDs removes no extraction pass in the current planner, and the timed fallback shows no positive paired saving whose 95% interval excludes zero. The eleven signals share their main kernels with retained signals, so their exclusive arithmetic cost is not separately measured. The shared-kernel percentages below are not a removable fraction. A finer per-signal implementation is a separate, unmeasured optimization experiment. A production model without these non-dead inputs needs retraining and a scores change at a release boundary.

The final independent checks pass: NumPy 2.5.2 verifies 121 r5900xt records (1,936 paired rounds) and 75 i270 records (1,200 paired rounds), all sixteen request-folder producer-manifest comparisons, all eleven measured percentile cells and twelve numerical fits. Pillow 12.3.0 / libjpeg-turbo 3.1.4.1 fully decodes and verifies all 80 JPEG hashes and dimensions; no oracle pixels enter the benchmark.

### Shared-kernel attribution

All percentages below are **r5900xt / Zen 3 / v3**, period-weighted executing-symbol self samples under the extraction call chain. A repeated percentage denotes the same shared bucket, not another charge. **Exclusive per-family time is not separately measured**: inlined fused arithmetic has no separate sampled symbol. These shares are not removable-time estimates.

| Family | Damage IDs | Main shared bucket | 64² | 256² | 1024² | Large |
|---|---:|---|---:|---:|---:|---:|
| basic × ssim | 36 | `basic_fused` | 13.591% | 12.506% | 10.484% | 8.876% |
| basic × mse | 12 | `basic_fused` | 13.591% | 12.506% | 10.484% | 8.876% |
| basic × contrast_inc | 12 | `basic_fused` | 13.591% | 12.506% | 10.484% | 8.876% |
| v2 × ssim | 12 | `v2_dense` | 19.654% | 23.430% | 24.335% | 21.393% |
| v2 × mse | 12 | `v2_dense` | 19.654% | 23.430% | 24.335% | 21.393% |
| v2 × ssim_soft_peak | 12 | `v2_dense` | 19.654% | 23.430% | 24.335% | 21.393% |
| v2 × masked_ssim | 12 | `v2_dense` | 19.654% | 23.430% | 24.335% | 21.393% |
| v2 × masked_art | 12 | `v2_dense` | 19.654% | 23.430% | 24.335% | 21.393% |
| v2 × masked_mse | 12 | `v2_dense` | 19.654% | 23.430% | 24.335% | 21.393% |
| v2 × iw_mse | 12 | `v2_dense` | 19.654% | 23.430% | 24.335% | 21.393% |
| v2 × edge_width_change | 12 | `v2_gradient` | 13.292% | 11.811% | 10.945% | 8.703% |

The three basic signals share blur moments and the fused reduction with retained edge, detail, texture and variance signals. The seven dense v2 signals share SSIM moments, error sums, activity weights and pooling with retained deviations, peaks, detail/IW outputs and transducers. Edge-width change shares gradients with retained GMS, ringing, banding and fragility; its adjacent-scale finalization remains in the shared fold and is not isolated by the gradient bucket. The common color and pyramid work is required by retained families too.

| Extraction bucket | 64² | 256² | 1024² | Large |
|---|---:|---:|---:|---:|
| `basic_fused` | 13.591% | 12.506% | 10.484% | 8.876% |
| `v2_dense` | 19.654% | 23.430% | 24.335% | 21.393% |
| `v2_gradient` | 13.292% | 11.811% | 10.945% | 8.703% |
| `shared_blur_pyramid` | 32.588% | 33.247% | 37.122% | 41.437% |
| `shared_color` | 5.326% | 8.862% | 10.012% | 9.073% |
| `shared_memory` | 9.749% | 5.062% | 2.784% | 5.440% |
| `retained_blockiness` | 1.719% | 2.213% | 1.937% | 1.578% |
| `other_extraction` | 4.080% | 2.868% | 2.382% | 3.500% |

| Size | All samples | Extraction samples | Extraction / whole sampled cycles |
|---|---:|---:|---:|
| 64 | 9408 | 9006 | 95.752% |
| 256 | 21679 | 21407 | 98.748% |
| 1024 | 21511 | 21481 | 99.860% |
| large | 27509 | 27499 | 99.964% |

The denominator is the sum of sample periods whose call chain includes `zensim::fold_engine::compute_fold_backed_stoppable`. Only each sample's executing symbol enters a bucket; callers do not add inclusive cost. Sampling is 997 Hz `cycles:u` with frame pointers. Exact period counts and all bucket symbols are retained in `perf-summary.json`. The family-containing bucket sums use these observed symbols (union over sizes):

`basic_fused`:

```text
<zensim::feature_v2::V1BasicSums>::accumulate
<zensim::feature_v2::V1BasicSums>::at_revision
<zensim::feature_v2::V1BasicSums>::finalize_into
<zensim::feature_v2::V1BasicSums>::merge
<zensim::fused::VWin>::at32::{closure#1}
<zensim::fused::VblurPools<zensim::featcanon::Lanes16F32>>::fold_into::<zensim::featcanon::Lanes16F64>
<zensim::fused::VblurPools<zensim::featcanon::Lanes16F64>>::finish_into
core::ptr::drop_glue::<zensim::fused::VWin>
zensim::feature_v2::fold_v1_one_band
zensim::fused::__arcane_fused_vblur_ssim_canon64_vec_v3::<zensim::featcanon::Lanes16F64>
zensim::fused::fused_vblur_features_ssim
```

`v2_dense`:

```text
<zensim::feature_v2::OnlineMoments>::merge
zensim::feature_v2::__arcane_dense_block_kernel_canon64_vec_v3::<zensim::featcanon::Lanes16F64, true>
zensim::feature_v2::dense_block_kernel_at_revision
```

`v2_gradient`:

```text
zensim::feature_v2::__arcane_gradient_block_kernel_canon64_vec_v3::<zensim::featcanon::Lanes16F64, false, false, false, false, true>
zensim::feature_v2::gradient_block_kernel_at_revision
zensim::feature_v2::gradient_border_pixel::<false, false, false, false, true>
zensim::feature_v2::gradient_interior_elem_canon::<zensim::featcanon::Lanes16F64, zensim::featcanon::Lanes16F64, false, false, false, false, true>
```

### Resource receipts

These are the accepted **r5900xt / v3** perf-record run-heavy receipts. RSS
covers the whole process, including the twenty resident image pairs and setup;
it is not scorer-only, extraction-only, or an isolated full/reduced-arm peak.
The one-minute load average can retain the preceding sixteen-thread run.

| Size | Exit | Elapsed s | Peak RSS GiB | Minimum available MiB | Peak load |
|---|---:|---:|---:|---:|---:|
| 64² | 0 | 11 | 0.03 | 44138 | 2.68 |
| 256² | 0 | 24 | 0.03 | 43866 | 4.48 |
| 1024² | 0 | 25 | 0.18 | 44844 | 10.92 |
| Large | 0 | 60 | 1.61 | 43369 | 6.68 |

The corresponding logs are `retry02/logs/perf-64.log`,
`retry01/logs/perf-256.log`, `logs/perf-1024.log` and `logs/perf-large.log`.
Symbol-processing receipts are retained separately and are not substituted for
these runtime peaks. Every run uses run-heavy's 16 GiB hard cgroup cap.

The following timing receipts select the largest measured whole-harness RSS among completed native attempts for each host/cell. They can include rejected timing rounds; they are not isolated arm peaks. Every row has exit zero. The source path and verbatim run-heavy line are in `RESOURCE_RECEIPTS.json`.

| Host / v3 | Size | Threads | Elapsed s | Peak RSS GiB | Minimum available MiB | Peak load |
|---|---|---:|---:|---:|---:|---:|
| r5900xt | 64 | 1 | 237 | 0.08 | 47972 | 3.64 |
| r5900xt | 64 | 16 | 227 | 0.08 | 48060 | 4.77 |
| r5900xt | 256 | 1 | 230 | 0.10 | 47876 | 4.71 |
| r5900xt | 256 | 16 | 229 | 0.16 | 44338 | 4.65 |
| r5900xt | 1024 | 1 | 294 | 0.25 | 47636 | 3.27 |
| r5900xt | 1024 | 16 | 265 | 0.62 | 47472 | 6.30 |
| r5900xt | large | 1 | 1311 | 1.70 | 46194 | 4.60 |
| r5900xt | large | 16 | 815 | 2.76 | 46159 | 10.95 |
| i270 | 1024 | 1 | 155 | 0.21 | 25179 | 4.31 |
| i270 | 1024 | 16 | 151 | 0.60 | 24776 | 2.69 |
| i270 | large | 1 | 795 | 1.67 | 23673 | 2.41 |
| i270 | large | 16 | 317 | 2.76 | 22603 | 10.51 |

## Exact commands and evidence

The raw evidence is stored outside git at
`/mnt/v/output/zensim/a-family-extraction-cost-2026-11-04/raw/`, with a second
archive copy. `SHA256SUMS` covers 2,731 regular source files; its SHA-256 is
`6e9b058ff89ede2274976e85efaac4cd7e7aae06490d638880a1457b4d753476`.
It includes the frozen binary/build export, exact source pins, prepared RGB
pairs/JPEGs, all raw source rounds, admission receipts, rejected attempts,
perf data and symbol reports, host-specific accepted views, summaries and
verification receipts. Transient FIFOs, locks, scratch and Python bytecode
are excluded. Archive links are relative; `ARCHIVE_LINKS.json` records their
original targets. To replay verification after relocation, pass
`--raw-root <archive-root>` to `a_family_cost_verify.py`.
After both copies completed, `sha256sum -c SHA256SUMS` passed all 2,731
checks independently on each copy. Their relocation maps agree and all
relocated targets resolve.

The committed owners are `zensim/examples/a_family_cost.rs`,
`scripts/bench/a_family_cost.sh`, `scripts/bench/a_family_cost_inputs.py` and
`scripts/bench/a_family_cost_report.py`; the justfile exposes the same owners.
The selector retains source hashes and round indices. The independent verifier
checks every retained round against those raw sources, recalculates percentiles
with NumPy and fits the regressions with NumPy's least-squares solver.

```sh
python3 scripts/bench/a_family_cost_inputs.py \
  --corpus /home/lilith/work/codec-corpus \
  --canonical /home/lilith/work/zen/imazen-26-png-v3/png-v3 \
  --out /home/lilith/tmp/zensim-cost-input \
  --remote-root /home/lilith/tmp/zensim-cost/input

# Build on i270; execute the identical verified binary and pairs on r5900xt:
export ZENSIM_BUILD_COMMIT=61d77a13352f6e845419008a2737e5f892a42421
bash a_family_cost.sh build /home/lilith/tmp/zensim-cost
bash a_family_cost.sh prepare /home/lilith/tmp/zensim-cost
~/work/claudehints/scripts/run-heavy --mem 16G --jobs 8 -- \
  python3 a_family_cost_shared_run.py /home/lilith/tmp/zensim-cost

# After release, the four unstarted second-retry cells run on i270:
export A_COST_CPUS_1=2 A_COST_CPUS_16=0-15
bash a_family_cost.sh timings /home/lilith/tmp/zensim-cost/retry02-released \
  1024:1 1024:16 large:1 large:16
# Their immutable timing/log directories are imported as retry02-i270.
# Run selection and reporting separately with --host r5900xt and --host i270:
flock ~/tmp/heavy.lock ~/work/claudehints/scripts/run-heavy --mem 16G --jobs 8 -- \
  python3 a_family_cost_select.py /home/lilith/tmp/zensim-cost \
  /home/lilith/tmp/zensim-cost/accepted-r5900xt --final --host r5900xt
flock ~/tmp/heavy.lock ~/work/claudehints/scripts/run-heavy --mem 16G --jobs 8 -- \
  python3 a_family_cost_report.py /home/lilith/tmp/zensim-cost/accepted-r5900xt \
  /home/lilith/tmp/zensim-cost/summary-r5900xt.json --host r5900xt
flock ~/tmp/heavy.lock ~/work/claudehints/scripts/run-heavy --mem 16G --jobs 8 -- \
  python3 a_family_cost_perf.py /home/lilith/tmp/zensim-cost/accepted-r5900xt \
  /home/lilith/tmp/zensim-cost/perf-summary.json --host r5900xt
flock ~/tmp/heavy.lock ~/work/claudehints/scripts/run-heavy --mem 16G --jobs 8 -- \
  python3 a_family_cost_verify.py /home/lilith/tmp/zensim-cost/accepted-r5900xt \
  /home/lilith/tmp/zensim-cost/summary-r5900xt.json /home/lilith/tmp/zensim-cost/verified-r5900xt.json
```

Fixture preparation used the original pre-admission-fix executable; the
feature/runtime source and prepared bytes are unchanged. All accepted perf
captures and timings use the final owner-aware build and bind its binary hash.
Discarded setup attempts remain under `.bak` names in the raw evidence;
no sample from them enters these results. The complete 64²/1-thread i270
cell is also kept outside this report's r5900xt comparisons. New i270 work
stopped when the host was reserved for the peer speed study. The setup exposed the old
zenbench pin's lack of explicit worker ownership and a foreign build outside
the cooperating lock. One pilot also overlapped perf symbol expansion;
the report commands now run under the same heavy-work lock with inline
symbol expansion disabled. A runner update also caused its controller to
fail after a successful native cell; complete native pairs and its zero-exit
admission receipt were preserved, and the controller resumed using atomic
script publication. Completed and interrupted attempts remain distinguishable.
At the owner's release, the r5900xt driver was terminated after its native
64²/16-thread attempt had finished with exit zero, before it could launch the
four remaining second-retry cells. The driver's exit 143 is intentional;
no native timing pair was interrupted. `BOUNDED_COMPLETE.json` records the
completed collection after the four i270 cells and all perf captures finish.

## Exact source files

CID22 and CLIC paths below are relative to
`/home/lilith/work/codec-corpus/`. `png-v3/` paths are relative to the canonical
`/home/lilith/work/zen/imazen-26-png-v3/` checkout, revision
`b1faae550eba4d29efee1de1a32954ef11a0e1ed`.
LFS pointer-only files were fetched from the corpus's public PNG-v3 endpoint
and SHA-256-checked against that checkout's pointer, without changing the
checkout. No quarantined copy was used.

| Cases | Source file (resolved on dev) | SHA-256 | Native size |
|---|---|---|---|
| 64_00, 256_00 | `CID22/CID22-512/training/1001682.png` | `2b76992d7b0578d0ca333c61cf3dac4938f3ae2c8c7a613dc3179fc3125ec7c4` | 512×512 |
| 64_01, 256_01 | `CID22/CID22-512/training/1028637.png` | `bf71611dd0ee392366468420449462a69f07ccc89e96f22ac2744332160ad85f` | 512×512 |
| 64_02, 256_02 | `CID22/CID22-512/training/1029604.png` | `ddf197a051d6ff0264456147b703a2d78e9cf734782fd1c4de142d78ff9b1f83` | 512×512 |
| 64_03, 256_03 | `CID22/CID22-512/training/106399.png` | `5649526c6d79faf3abb980170a728b2ec06c2244d8f75a6ed5f1cb918ef15772` | 512×512 |
| 64_04, 256_04 | `CID22/CID22-512/training/1080721.png` | `c7822d5bd89bed54d4c6f23e2104536c483024ed5c6290d55a97c80fa836f45c` | 512×512 |
| 64_05, 256_05 | `CID22/CID22-512/training/1082342.png` | `f0d4a3edb73360618ba276c299d5ebda60e724475f248ab689d50b12311631a1` | 512×512 |
| 64_06, 256_06 | `CID22/CID22-512/training/1089930.png` | `70bda85409b427474ba37d25dee5435ad32bea594b62cd4ba69592e527cb9870` | 512×512 |
| 64_07, 256_07 | `CID22/CID22-512/training/110472.png` | `4f35f46b9029e963fd49d24ab1fd0235234e037caf436a2e241e35aa418102b6` | 512×512 |
| 64_08, 256_08 | `CID22/CID22-512/training/1129482.png` | `9416992dcb64582212da548e6183bba0c97cf77c8368f3268ebafae8a23cc00b` | 512×512 |
| 64_09, 256_09 | `CID22/CID22-512/training/1130683.png` | `81165700b673e31b0df94c7c3398ee651241748f5dd98343b0352f79a5d8e366` | 512×512 |
| 64_10, 256_10, 1024_00 | `clic2025-1024/097cb426910ba8ce2525dd8bb7fb1777.png` | `29f637ce7934a542c116049d4e74f1885d88dc08f689d557d599aa95fc5049de` | 1024×1024 |
| 64_11, 256_11, 1024_01 | `clic2025-1024/0c49a5cce349020bbba2f97ae41e90ba.png` | `0d13217e6a93999fd3f3b28796df8b3dd12522984644376de00915fa78d39b8c` | 1024×1024 |
| 64_12, 256_12, 1024_02 | `clic2025-1024/100a02c269c5948392f283b2aa3bb4da.png` | `d24ef571eca0547cf33a19fd8212f97375309ce279d1d70d63d4a98f46ffd265` | 1024×1024 |
| 64_13, 256_13, 1024_03 | `clic2025-1024/100f17892cf7bb12beb12479641fcc4c.png` | `00f5340b9d27a7cc87c7a6c98c7344eeef4cfb596919113aa00a0bad60e5e9f0` | 1024×1024 |
| 64_14, 256_14, 1024_04 | `clic2025-1024/11f2b039b293758398b1a7a8afa64bb2.png` | `8d4d1b245599080091a930bfac471cf0c5eb3d66030c6a4242515cec76b81d81` | 1024×1024 |
| 64_15, 256_15, 1024_05 | `clic2025-1024/14ab4af28901fbeb1356b06d2d08ae06.png` | `c95b1c8f1900874cd844ed013879b105f6aa2c6d81538eb443f3375f68ed7ff5` | 1024×1024 |
| 64_16, 256_16, 1024_06 | `clic2025-1024/1e2f9d41529197f10d32bfa68a1e0bcc.png` | `7e5950540bb4a7a2045bcdcf310cd775b2f91828e81b41151be02a106fb80eb4` | 1024×1024 |
| 64_17, 256_17, 1024_07 | `clic2025-1024/22ea12c903e41583b7c469cb86040157.png` | `05726ce1eee2384d106db2a875fedf6c0a76ee44ace06aa95bed51979cbdd244` | 1024×1024 |
| 64_18, 256_18, 1024_08 | `clic2025-1024/28d24b9c83de066597ff96a68769884f.png` | `28d565117cff9242f93becf420081b1b117296719b2ff1a37e170e09640fe7cb` | 1024×1024 |
| 64_19, 256_19, 1024_09 | `clic2025-1024/2a2420f929cd47122eae364a2bc27710.png` | `b5e0d70de15cd5ed0edee94ae90e601e0decc781838b274b4a3d3eb17e9ee8e7` | 1024×1024 |
| 1024_10 | `clic2025-1024/2c1f84548ef99faec2b4f9bf12227c83.png` | `1761db9c5778e37ed12454c3314d1d970285ed454ee293772112c5eec1c6a875` | 1024×1024 |
| 1024_11 | `clic2025-1024/3140d643a1f57e80ff265507984d95a0.png` | `8af7fd39943c1c81aeaf6369a856d7771d56790fc84e7012d8e4663371c74e4b` | 1024×1024 |
| 1024_12 | `clic2025-1024/4a18abf66d65ad1a95ce455c8b6cd8de.png` | `ea740b930a5b1856fcc83a3ccb039dae70f194f50e8c0c05c9f67b161a81988b` | 1024×1024 |
| 1024_13 | `clic2025-1024/4cd6910a0b7b39365fda5df87618d091.png` | `2bbfa289d5b9c786ff1a0bb3b44e4f45af6b9180089d3a5cc7975b3fee21c8c4` | 1024×1024 |
| 1024_14 | `clic2025-1024/5dbdb989cf026f7ab3dd7167d7fc0fe2.png` | `ba707f5ec9eb1719b6f8daf8cabf80d2f5559152132f18aae143a4ca54b21f6a` | 1024×1024 |
| 1024_15 | `clic2025-1024/5e5ce43575fa67fdc0dd37146d7f479e.png` | `170a099e5c4b447675a1b541ded370a0645b41174578597836a3ecb407fb2beb` | 1024×1024 |
| 1024_16 | `clic2025-1024/60430844748882ff66b9a5598ef06dd6.png` | `9d076e5fda96e267ff47d6dc2f02449d3fff44e7d6f58e907a0c4e3fd930eedf` | 1024×1024 |
| 1024_17 | `clic2025-1024/7e499613c3e376ea93afb3649719abeb.png` | `0d1322274a51490648354bf55222ab7dfea3e78fba82209ea98b67704f3264ba` | 1024×1024 |
| 1024_18 | `clic2025-1024/870516c65d81fb9267de6865964083a9.png` | `81f50db6fcf01282479292252f934d972b2dd584ceba645a75da821e952f31a9` | 1024×1024 |
| 1024_19 | `clic2025-1024/8bb119b8ca17492392e1e3ea4b811b32.png` | `242ad398f86782161b09811f0bcc69e8752a99773ba772bb55d7f40634329f0b` | 1024×1024 |
| large_00 | `png-v3/1000-lilith-photos-general/1000_general_red-convertible-car_mission-hills-san-diego-california_note9_iso50-f2p4_20190514-100811_4032x3024.sdr.png` | `df355b76958e2482ed85ca92a354150586041ac3f2e4927509b0746b1d331754` | 4032×3024 |
| large_01 | `png-v3/1000-lilith-photos-general/1002_general_oceanfront-hotel-building_ka-anapali-beach-kaanapali_s21u_iso640-f1p8_20210608-052615_4000x3000.sdr.png` | `17bcef7fefb44b63f32d01ff504ca005f82a891742fce096aa4cac0dc46fc5aa` | 4000×3000 |
| large_02 | `png-v3/1000-lilith-photos-general/1004_general_paint-color-samples_s21u_iso200-f1p8_20220304-144007_3000x4000.sdr.png` | `231c517eef3fb98386cdb0bafea30414ac17cbab2626463c80e582242607df0c` | 3000×4000 |
| large_03 | `png-v3/1000-lilith-photos-general/1006_general_church-steeple-among-trees_s21u_iso50-f1p8_20220304-182325_3000x4000.sdr.png` | `de5c69eb0bab47f20b5a8a99ff0363e992cc68391be56c17a2def0c95a272c21` | 3000×4000 |
| large_04 | `png-v3/1000-lilith-photos-general/1008_general_city-view-through-window_tokyo-skytree-sumida_s23u_iso50-f2p2_20230706-142333_3000x4000.sdr.png` | `325742e72036878bb316aed3629d41b92f285f0ad44de845ac8d512b285d3d30` | 3000×4000 |
| large_05 | `png-v3/1200-lilith-interiors/1200_interiors_reflective-egg-sculptures_s23u_iso10-f1p7_20230704-121114_4000x3000.sdr.png` | `7d935f49472cb9f5282b48a2da3dde3eb69030f535260da6b9334a35335ec95a` | 4000×3000 |
| large_06 | `png-v3/1200-lilith-interiors/1202_interiors_abstract-textured-pattern_seattle-center-seattle_s23u_iso1000-f1p7_20230727-164305_3000x4000.sdr.png` | `020cbb1290d5ded0893319e24b9947597b202f438354e362c01bb4bf342b732e` | 3000×4000 |
| large_07 | `png-v3/1200-lilith-interiors/1204_interiors_rustic-outdoor-dining-area_s23u_iso125-f2p2_20230915-124518_3000x4000.sdr.png` | `23579a83cedb6e5009b089e7b1e9c949a6da11ed14a6d7873304b4a170ec5214` | 3000×4000 |
| large_08 | `png-v3/1200-lilith-interiors/1206_interiors_tree-like-sculpture-ceiling_azulik-city-arts-francisco-uh-may_s23u_iso80-f2p2_20230917-143302_3000x4000.sdr.png` | `fbb834595b932679385d9cc2be6cb916d7b7b10dcf244b82646da8f47e7f894f` | 3000×4000 |
| large_09 | `png-v3/1200-lilith-interiors/1208_interiors_woven-hammock-by-beach_hotel-zentik-project-valladolid_s23u_iso20-f1p7_20230918-110558_3000x4000.sdr.png` | `45af84bef5fa767823039322a4c600a16811c6adc748833743ca3cfa9d5325fd` | 3000×4000 |
| large_10 | `png-v3/1400-lilith-nature/1400_nature_snowy-mountain-landscape_colorado_s10_iso50-f2p2_20200226-105616_4608x3456.sdr.png` | `7516beec496ae6b44231d37b26d7408e02ddcf4e2dc1b4eb9ea8024c52bda9c1` | 4608×3456 |
| large_11 | `png-v3/1400-lilith-nature/1402_nature_rocky-ocean-coast_spitting-caves-east-honolulu_s21u_iso16-f1p8_20210605-110609_4000x3000.sdr.png` | `56afdc7e89a6b547088843e8d7779116cc95ce76e9bdc1bdb6ec066ce7b6d965` | 4000×3000 |
| large_12 | `png-v3/1400-lilith-nature/1404_nature_palm-trees-sunset_ka-anapali-beach-kaanapali_s21u_iso16-f1p8_20210607-180007_4000x3000.sdr.png` | `9dbdfdf7ad58b38ef0a3f5d962e3e974dc0c4151033c5da11d9bcf056afbc7cf` | 4000×3000 |
| large_13 | `png-v3/1400-lilith-nature/1406_nature_tropical-coastline-bay_waianapanapa-state-park-hana_s21u_iso50-f2p2_20210608-132330_4000x2252.sdr.png` | `3b3427766d5f93fbbbfc0692aee569ee11305e0d8389339ab43e4200fd57089a` | 4000×2252 |
| large_14 | `png-v3/1400-lilith-nature/1408_nature_coastal-cliff-road_maui-hawaii_s21u_iso16-f1p8_20210608-162813_4000x3000.sdr.png` | `b675fd60b1740385b5e3a54ad5afd6179132774085e904e3b1b2f3759280d884` | 4000×3000 |
| large_15 | `png-v3/1600-lilith-food/1600_food_steamed-dumplings-basket_university-city-san-diego-california_note9_iso64-f2p4_20190516-185153_4032x3024.sdr.png` | `bc074c930ec9f1635fa6654e03fde21b8b54f7f3b15a58fba8cca69320bce3db` | 4032×3024 |
| large_16 | `png-v3/1600-lilith-food/1602_food_caprese-salad-plate_s21u_iso250-f2p4_20210709-193445_4000x3000.sdr.png` | `efb255087fdba21ea53845626a643d4f10743d20d46d8a6450e030f45cc9f131` | 4000×3000 |
| large_17 | `png-v3/1600-lilith-food/1604_food_meat-and-fried-vegetables_l-eixample-barcelona-spain_s21u_iso640-f1p8_20220304-124452_3000x4000.sdr.png` | `0a45c1775fd42760a45c0ba51ceed1589e1babddafad983c105f2a11981138e7` | 3000×4000 |
| large_18 | `png-v3/1600-lilith-food/1608_food_single-cake-slice-plate_anedaitocho-kyoto-japan_s23u_iso320-f2p2_20230709-194142_4000x3000.sdr.png` | `f0a95910756b0386ca96169c2ac04b83b4decc78cfc5cc3e3e1f04b3692e88d3` | 4000×3000 |
| large_19 | `png-v3/1600-lilith-food/1610_food_rice-bowl-vegetables_mapunapuna-honolulu-hawaii_s23u_iso50-f2p2_20230712-183013_3000x4000.sdr.png` | `261cf9b9be8f7c92333f066059aae69f45a9b3107ed6dcba5e78f76120039fb5` | 3000×4000 |
