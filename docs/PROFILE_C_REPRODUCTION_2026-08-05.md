# Profile C — provenance + exact reproduction (2026-08-05)

> **⛔ F8 SUPERSEDED 2026-08-06 (campaign appendix V).** Every `F8 CID22 B9 band
> tail` figure in this document (0.139 vs ≥ 0.15, n=43) was computed on an
> ABSOLUTE-valued statistic over a degenerate band — 43 pairs from 11 of 49
> references spanning 0.0194 MOS, split-half reliability 0.711 against a 0.90
> bar — and its SIGN was negative for 109 of the 120 board cells that could be
> re-checked. The `7 of 8 floors` counts below are therefore stale: F8 now reads
> the SIGNED top/bottom USABLE bands of scheme `merged-decile-2026-08-06` with a
> derived floor of 0.09, and on the recut board every cell passes it. Re-run
> `freeze_check` after `promote_fulleval.py --rebuild-bands` for a current count.
> Record: `benchmarks/band_minimum_n_2026-08-06.md`; registry entry
> `f8-b9-abs-bar-superseded`.


**`ZensimProfile::C`** (`zensim-c`) ships the SOTA-944 campaign's wave-11
battery-selected cell, internal name **`W10L9_s4003_packed`**, as
`zensim/weights/c_sdr_mlp944_corrmix_2026-08-05.bin` —
**165,696 bytes, sha256
`1a2c8d522fed8034b279ff018aa052f19d0b9f419f12cf22cca303a0b4abb7f4`.**
`B` remains the default / `codec_target()`; `C` is an additional profile,
user-gated ship 2026-08-05. This document is written for a reader with a
fresh clone of `imazen/zensim` and R2 access: every input, command, commit,
and expected output needed to re-derive the shipped bytes and re-verify the
claimed numbers, with nothing assumed from the authoring session.

Authority records (in-repo): campaign doc
[`benchmarks/sota944_campaign_2026-08-03.md`](../benchmarks/sota944_campaign_2026-08-03.md)
— appendix **H** (wave 10: the KADID orientation fix + leave-one-out, results
H.R), appendix **K** (wave 11: k=8 seed depth + the full winner battery,
results K.R), appendix **N** (the fused folded-944 compare + H3-own-map loop
results, N.R), appendix **M** (the 2/3-shot loop panel pointer); wave records
under [`benchmarks/wave10/`](../benchmarks/wave10/) and
[`benchmarks/wave11/`](../benchmarks/wave11/).

---

## 1. Result summary

### 1.1 What C is

A 944-input MLP (folded-720+append+append2 feature regime, `--regime 944`),
944 → 128 → 1, LeakyReLU, f16 + zerobias-0.005, dial-packaged with a monotone
PCHIP output spline and **dead-column-pruned**: caller width 944, internal
layer-0 width 667 (277 `FeatureTransform::Drop` lines, all class-1
weight-dead, identity-gate BIT-identical on the 2,035-row anchor). It is the
first shipped bake trained after the KADID target-orientation fix
(`176c4268`) and the first shipped PRUNED bake — the runtime sizes feature
vectors by `Model::caller_input_width()`, never `n_inputs()` (`ae852b1b`).

### 1.2 The k=8 family (recipe confirmation at seed depth)

The recipe (wave-10 arm **L9**: the incumbent arm-H 944 mix with the
corrected KADID table and the `tkadis` teacher leg dropped) was confirmed at
k=8 seeds — the two wave-10 draws + six new wave-11 seeds, pooled per the
registered K.4 rule. From K.R1 (tables:
`benchmarks/wave11/wave11_{cells,family_summary}_2026-08-05.tsv`):

| headline axis | fam median [min, max] | L9 pair range | incumbent k=3 mean | K.5 call |
|---|---|---|---|---|
| CID22 | 0.88412 [0.87477, 0.88903] | [0.88671, 0.88903] | 0.87874 | HOLDS-WITHIN-NOISE |
| KonJND | 0.46604 [0.41035, 0.50741] | [0.42389, 0.49879] | 0.43297 | HOLDS |
| LIVE | 0.96389 [0.96081, 0.96770] | [0.96081, 0.96526] | 0.84293 | HOLDS |
| HF-NL per-ref | 0.66308 [0.46789, 0.74993] | [0.62120, 0.73334] | 0.25550 | HOLDS |
| dial mono | 99.52% [98.85%, 99.72%] | [99.53%, 99.66%] | 94.81% | HOLDS-WITHIN-NOISE |

Zero REGRESSION, zero COLLAPSE on any axis — registered outcome **(b)-mild**:
the breadth/HF-NL/dial gains are properties of the corrected mix across all
8 seeds; the wave-10 pair was mildly seed-lucky on CID22 point estimates
(±0.003). Selection over the pooled 8 (`freeze_check --select`, registered
E.4 rule): **`W10L9_s4003`** — 7/8 balanced floors, sel_comp 0.9579, M3a
0.8626 GOLD tie-break (`benchmarks/wave11/wave11_select_2026-08-05.txt`).

### 1.3 C's own verdict headline (the shipped packed bytes)

All values read from the committed verdict/fulleval of the shipped bytes
(`/mnt/v/output/zensim/bakes/sota944/verdicts/W10L9_s4003_packed.full.json`,
`bake_sha256 = 1a2c8d52…`; board fulleval
`/mnt/v/output/zensim/reports/fulleval/W10L9_s4003_packed.fulleval.json`):

| axis | value | note |
|---|---|---|
| CID22 SROCC | **0.88672** | gold human-MOS holdout |
| KonJND \|SROCC\| | **0.49883** | signed −0.49883 (PJND direction) |
| LIVE | **0.96043** | classic-IQA breadth |
| CSIQ | **0.93312** | classic-IQA breadth |
| nonphoto (ssim2 north star) | 0.92512 | |
| imazen26 real-codec (ssim2) | 0.92147 | |
| HF-NL per-ref mean | **0.7334** | class record (B = 0.8252) |
| hfnlproxy pooled | 0.43136 | pooled ≠ per-ref; both reported |
| AIC-3 / AIC-4 | 0.79681 / \|0.90187\| | |
| sdr25 \|SROCC\| | 0.95275 | comparator only, never primary |
| KADID / TID (signed) | +0.91314 / +0.93261 | train==val integrity guards, corrected-cohort KADID |
| dial monotonicity / tied | **99.319% / 0.0%** in DIAL units | first packaged 944 cell over the ≥93% dial bar |
| dial dynamic range / reach | 67.64 / 94.33 | |
| M3a coherence | **0.86238** (packed; parent 0.86259) | GOLD ≥ 0.85, post-`299ccc8c` instrument |
| corruption (companion head `corrhead944_s13`) | pass_q20 **0.79315** / pass_q10 **0.92560** | dial-alone 0.1875/0.0625 — broken by design post-720; the head is the owner |
| product_composite | 0.86025 | |
| balanced floors (`balanced-2026-08-04`) | **7/8** | only miss: F8 CID22 B9 band tail 0.139 vs ≥ 0.15 |

Packaging is FREE (K.R battery 1): per-axis |Δ(raw→packed)| ≤ 0.0004 on every
rank axis (CID22 +0.00001, KonJND +0.00004 — the f32-pack contingency does
not fire), M3a −0.0002.

LOO ×2 (occlusion, not ablation): both the BANDVIS lanes and the append2
block are KEEP — family Σ(|full|−|drop|) = +0.0266 / +0.0882; the winner
draws on append2 on all five read axes (K.R battery 4).

Loop value (appendix M/N + jxl-encoder records): in the 2/3-shot
loop-targeting panel the candidate splits — k2 (c), k3 (b)-plus
(`jxl-encoder/benchmarks/zensim_loop_23shot_sota944_2026-08-05.md` +
`zensim_loop_23shot_summary_2026-08-05.json`, jxl commit `1f89dc66`); with
its OWN attribution map under H3 magnitude steering it takes the board-best
inner census at k3: **17/27**, med |err| 1.66 vs generic 1.82, paired
18W/8L/1T at bytes 0.978 (N.R, jxl commit `4e4a7334`, zensim entry
`c28d29b8`).

vs shipped **B** (honest read, K.R3): C takes CID22 (+0.005), nonphoto
(+0.026), LIVE (+0.064), dial mono (+2.1 pp incl. packaged-unit), M3a
(+0.266) and carries the corruption head; B keeps KonJND (0.5186 vs 0.4988)
and HF-NL per-ref (0.8252 vs 0.7334). **B stays the default.**

---

## 2. Environment

| item | value |
|---|---|
| Training node | `lilith-lianli` — AMD Ryzen 9 7900X (12C/24T), 29 GiB RAM, Ubuntu 26.04 LTS. (The embedded repro's `cwd=/home/lilith/sota944` is the lianli staging root; the wave-10 local lane ran on the dev box below. `hostname` is empty in wave-10 embedded repros — node identity is inferred from the staging-root path, and is not load-bearing: one binary, seed-deterministic, both lanes.) |
| Dev/eval box (packaging + all verdicts) | WSL2 on AMD Ryzen 9 7950X — VM sees 28 cores / ~59 GiB (Linux 6.x, x86-64) |
| Toolchain | rustc stable (1.97.1 at ship time; workspace MSRV 1.93). Trainer + eval binaries are plain `cargo build --release` — no `-C target-cpu=native`. Binaries staged to lianli by `scp` (glibc-compatible; lianli has no local toolchain). |
| Trainer binary (wave 10, both lanes) | `zensim_mlp_train` built from the `zensim--wave10` workspace (pre-flat-buffer lineage, registration commit `ac260c42`); binary sha256 `e5db2498…` recorded in H.R0. The staged lane copies were cleaned post-wave; reproduction path is a rebuild — validity across builds is the registered `d869a186` gate (below). |
| Eval binaries | `bake_verdict` / `freeze_check` / `diffmap_block_coherence` from this repo, `cargo build --release -p zensim-validate`; wave-11's instrument state = `1ed606e5` (post-`3db5a215` M3a-instrument fix). |
| Trainer preset | `--regime 944` data preset is resolved by the tools themselves (`bake_verdict --regime 944` pins the ext944 roots + 944 grids, test-locked); the trainer takes explicit `--group` paths (below). |
| Cargo flags | none beyond `--release`; per-agent `CARGO_TARGET_DIR` only relocates build output. |
| Determinism | The trainer is deterministic in `--seed` across lanes and builds: the registered cross-build identity gate (`d869a186`) re-trained a seed on the flat-buffer build and got f64-exact `best_val` + payload sha256-IDENTICAL after stripping provenance metadata. Gate tool: `scripts/verify_bake_identity.sh a.bin b.bin` — compares every byte outside the embedded `zentrain.repro` plus the repro JSON minus `timestamp_epoch`/`argv[0]`/`cwd`/`trainer_source_dir`/`--out`. **Caveat: `--out` is recorded verbatim in the embedded argv, so `cmp` of two whole files only passes when the `--out` string also matches; use the script, not `cmp`.** |

---

## 3. Data provenance — the 10-group corrected mix

Every row below is read from the shipped bake's **embedded
`zentrain.repro`** (`zenpredict inspect
zensim/weights/c_sdr_mlp944_corrmix_2026-08-05.bin`, metadata key
`zentrain.repro`) and re-verified against the on-disk bytes on 2026-08-05
(`sha256sum` of each local file == embedded sha, 10/10 match). Weights are
`train_w:val_w` with per-group loss mode.

| group (w:v:mode) | rows | sha256 (embedded == local) | local path | R2 (`s3://zentrain/…`) | Tower |
|---|--:|---|---|---|---|
| safesyn (1.0:0.5:both) | 111,068 | `cab92322156e72a9…` | `/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01/ext_safesyn_full.parquet` | `ext944-canonical-2026-08-01/ext_safesyn_full.parquet` | `/mnt/tower/output/zensim-ext944-canonical-2026-08-01/` |
| cid22_train (1.0:2.0:both) | 17,611 | `5abe28a099ac5f08…` | same dir, `ext_cid22_train201.parquet` | `ext944-canonical-2026-08-01/ext_cid22_train201.parquet` | same |
| kadid (0.5:1.0:rank) | 10,125 | `286f1b239d88c483…` **CORRECTED** | same dir, `ext_kadid.parquet` | `ext944-canonical-2026-08-01/ext_kadid.parquet` | same |
| tid (0.5:1.0:rank) | 3,000 | `7efce8f2a8d459d1…` | same dir, `ext_tid.parquet` | `ext944-canonical-2026-08-01/ext_tid.parquet` | same |
| bigcodec (0.5:1.0:both) | 208,169 | `190cb9161e392984…` | `/mnt/v/zen/zensim-training/tbig_944_200k.parquet` | `sota944/tbig_944_200k.parquet` *(uploaded 2026-08-05, this pass)* | gap→fixed: `/mnt/tower/output/zensim-sota944-2026-08-03/data/` |
| kadis (0.15:1.0:both) | 50,000 | `0ac7c889bdf1a728…` | `/mnt/v/zen/zensim-training/kadis-944-2026-08-01/kadis_944_ssim2_50k.parquet` | `kadis-944-2026-08-01/kadis_944_ssim2_50k.parquet` | `/mnt/tower/output/zensim/kadis-944-2026-08-01/` |
| tsafesyn (0.5:1.0:both) | 111,068 | `e0d6f5b9581b86d1…` | `/mnt/v/output/zensim/bakes/sota944/teacher/safesyn_teacher944.parquet` | `sota944/teacher/safesyn_teacher944.parquet` *(uploaded 2026-08-05)* | gap→fixed: same Tower `data/` dir |
| ttbig (0.5:1.0:both) | 208,169 | `f3919601a72ed328…` | same teacher dir, `tbig_teacher944.parquet` | `sota944/teacher/tbig_teacher944.parquet` *(uploaded 2026-08-05)* | gap→fixed: same |
| konjnd_bpg (1.2:0.0:both) | 8,060 | `c87d8f268d0f9a89…` | ext944 dir, `konjnd_bpg_train_944.parquet` | `ext944-canonical-2026-08-01/konjnd_bpg_train_944.parquet` | ext944 Tower dir |
| konjnd_bpg_val (0.0:1.5:both) | 2,020 | `a5bfd51801f04232…` | ext944 dir, `konjnd_bpg_val_944.parquet` | `ext944-canonical-2026-08-01/konjnd_bpg_val_944.parquet` | ext944 Tower dir |

Packaging anchor (not a training group): `anchor944_dial.parquet` — 2,035
rows (safesyn_full 800 + cid22_train201 401 + kadid 405 + tid 429), target
`target_score` ∈ [−100.0, 95.6], sha256
`d74d36ef01db117729f0a3cc019265f588b59bc7f02046ddaa39282cf8508e56`, built by
the committed `scripts/canonical_corpus/build_anchor944_dial.py`. Local: the
ext944 dir above; R2: `ext944-canonical-2026-08-01/anchor944_dial.parquet`
*(uploaded 2026-08-05, this pass — it was Tower+local only)*; Tower:
`/mnt/tower/output/zensim-sota944-2026-08-03/anchor944_dial.parquet`.

R2 verification (2026-08-05, this pass): every URL above HEAD-checked;
objects already present (`ext944-canonical-2026-08-01/*`,
`kadis-944-2026-08-01/*`) carry byte sizes matching local and their
dir-level round-trip sha verification is recorded in
`~/work/zen/DATA_PROVENANCE.md`; objects uploaded this pass
(`sota944/tbig_944_200k.parquet`, `sota944/teacher/*`,
`ext944-canonical-2026-08-01/anchor944_dial.parquet`) were round-trip
sha256-verified in §7. Dataset builders + per-dir `_MANIFEST.json` (with
`build_commit`) live beside each table.

**The corrected-KADID history (load-bearing for reproduction).**
`ext_kadid.parquet` stored an INVERTED target until 2026-08-05 (Known Bugs:
"ext-lineage KADID target stored inverted"; annotation registry
`kadid-ext-root-inverted`, 188 pre-correction verdicts). Wave 10 part 1
(`176c4268`) rebuilt the CORRECT quality-oriented `(dmos−1)/4` target
**in place under the canonical name** at all three ext roots and preserved
the inverted bytes as `ext_kadid_INVERTED_2026-08-04.parquet` (ext944 sha
`4dde6be26d3c50dc…`) in the same directories. Consequences:
- C trained on the corrected table (`286f1b23…` in its embedded repro) —
  its KADID number is real and correctly signed (+0.9133).
- **Repro hazard for OTHER (pre-correction) bakes:** any embedded repro
  listing kadid sha `4dde6be2…` can only be reproduced against the preserved
  `_INVERTED_` file, not the canonical name. C is NOT affected.
- Orientation is now gated: `scripts/canonical_corpus/check_target_orientation.py`.

`tkadis` (the kadis teacher leg) is deliberately ABSENT — wave 10 measured it
as the mix's one clearly-negative-marginal leg (H.R2: dropping it moves
HF-NL per-ref +0.464, LIVE +0.061, dial mono +4.3 pp, all outside noise).

---

## 4. Step-by-step reproduction

### 4.0 Build

```sh
git clone https://github.com/imazen/zensimx && cd zensim
# sibling checkout of imazen/zenanalyze is required (path deps zenpredict/zenpredict-bake)
cargo build --release -p zensim-validate   # zensim_mlp_train, bake_verdict, bake_dial_refit, freeze_check
```

### 4.1 Train (seed 4003) — the exact embedded argv

Stage the 10 tables at any root and substitute paths; everything else is
verbatim from the embedded `zentrain.repro` (also in the committed sidecar
`/mnt/v/output/zensim/bakes/sota944/bakes/W10L9_s4003.bin.spec.json` and its
R2 copy `s3://zentrain/profiles/C-2026-08-05/W10L9_s4003.bin.spec.json`).
~14 min on the 7950X lane at ~11.3 GiB peak RSS (measured wave-10; run under
`run-heavy --mem 24G` or equivalent).

```sh
zensim_mlp_train \
  --group safesyn:DATA/ext944/ext_safesyn_full.parquet:1.0:0.5:both \
  --group cid22_train:DATA/ext944/ext_cid22_train201.parquet:1.0:2.0:both \
  --group kadid:DATA/ext944/ext_kadid.parquet:0.5:1.0:rank \
  --group tid:DATA/ext944/ext_tid.parquet:0.5:1.0:rank \
  --group bigcodec:DATA/tbig_944_200k.parquet:0.5:1.0:both \
  --group kadis:DATA/kadis944/kadis_944_ssim2_50k.parquet:0.15:1.0:both \
  --group tsafesyn:DATA/teacher/safesyn_teacher944.parquet:0.5:1.0:both \
  --group ttbig:DATA/teacher/tbig_teacher944.parquet:0.5:1.0:both \
  --group konjnd_bpg:DATA/ext944/konjnd_bpg_train_944.parquet:1.2:0.0:both \
  --group konjnd_bpg_val:DATA/ext944/konjnd_bpg_val_944.parquet:0.0:1.5:both \
  --n-hidden-layers 0 --target-column human_score --target-scale 100 \
  --epochs 120 --pairs-per-epoch 50000 --seed 4003 \
  --max-features 944 --allow-narrow-features --coarse-decay 1e-5 \
  <82 --feature-transform flags — verbatim in the embedded repro / spec.json: \
   36 winsor_p99 + 12 signed_cbrt on v1-fold slots, plus 24 winsor_p99:*:0,0 \
   BANDVIS-lane guards at f731..f919> \
  --out W10L9_s4003.bin
```

Expected: `best_val = 0.9226384934404903` (f64-exact under the determinism
gate); raw bake 509,913 B. Byte-level check vs the recorded artifact
(`s3://zentrain/profiles/C-2026-08-05/W10L9_s4003.bin`, sha256
`6b48328054b4a0cbded2e2c1c8cde3e46f507c75de289c8b4309a7625b2ea70c`):

```sh
scripts/verify_bake_identity.sh W10L9_s4003.bin W10L9_s4003.reference.bin
```

must report identity (payload bytes + repro-minus-provenance). Whole-file
sha equality additionally requires the same `--out` string and cannot match
`timestamp_epoch` — use the script, not `cmp` (§2 caveat). Note the exact
wave-10 binary (`e5db2498…`) predates the `031bd261` flat-buffer trainer:
rebuilding from current `main` is registered-valid via the `d869a186`
cross-build identity gate (f64-exact `best_val`, payload sha identical).

### 4.2 Package (dial spline + pack) — the frozen 2026-08-04 chain

Owners only, no Python bake editing (K.6 / the packaging-pass registration):

```sh
# 1) dial spline (add-spline: the owner for spline-less bakes)
bake_dial_refit add-spline --in W10L9_s4003.bin --out W10L9_s4003_dial.bin \
    --anchor anchor944_dial.parquet --target-col target_score
# expected: 390,449 B, sha256 411d6db5e7e3f4a2af8e1c35097eb690645b643b0f8c8edfe224bec60b2f8554

# 2) G-RANGE gate on the dial bake (honest FAIL — see §6)
bake_dial_refit gate --bake W10L9_s4003_dial.bin --corpus ext_cid22val.parquet

# 3) pack: zerobias -> PRUNE -> f16 -> spline refit ON the packed net
bake_dial_refit pack --in W10L9_s4003_dial.bin --out W10L9_s4003_packed.bin \
    --neg-tail --anchor anchor944_dial.parquet --target-col target_score \
    --verify ext_cid22val.parquet --verify-col human_score --verify-scale 100
# expected: 165,696 B, sha256 1a2c8d522fed8034b279ff018aa052f19d0b9f419f12cf22cca303a0b4abb7f4
#           zerobias L0 59,777/120,832, L1 84/128; prune 944 -> 667 (277 drops,
#           all class-1); identity gate BIT-identical on 2,035 anchor rows;
#           --verify SROCC 0.8867
```

The shipped `zensim/weights/c_sdr_mlp944_corrmix_2026-08-05.bin` is a byte
copy of step 3's output (sha pinned in-crate by
`profile_c_tests::weight_sha256_pinned`). Byte-exact REPRODUCTION of the
pack step on the recorded inputs is expected to be exact (same binary class:
the 2026-08-04 pass reproduced `pack --no-prune` byte-identically vs the
2026-05-27 shipped artifact; pruning is deterministic). If a future
`bake_dial_refit` changes defaults, `--no-prune` reproduces the pre-pruning
2026-08-04 byte behavior — C itself ships PRUNED (prune ON, the default).

### 4.3 Verify (rank + dial + coherence + freeze)

```sh
# full 12-corpus verdict on the frozen --regime 944 preset (test-pinned roots)
bake_verdict --bake W10L9_s4003_packed.bin --regime 944 --full-json out.full.json
```

Expected headline == §1.3 (CID22 0.88672, KonJND |0.49883|, LIVE 0.96043,
CSIQ 0.93312, dial mono 99.319%/0.0% tied, composite 0.86025).
**Comparability gate** (run before reading any number produced by a
different build): this repo's `bake_verdict` must numerically reproduce the
committed `W10L9_s4003_packed.full.json` — the wave-11 instrument gate
diffed **82,385 numeric fields with 0 mismatches** across builds
(`benchmarks/wave11/comparability_gate_2026-08-05.txt` — method: flatten
both `--full-json` outputs, compare every numeric leaf at tolerance 0 beyond
float formatting). **Ship-day confirmation (2026-08-05):** `bake_verdict`
built from the Profile C ship commits, run on the in-repo
`zensim/weights/c_sdr_mlp944_corrmix_2026-08-05.bin`, reproduces the
committed verdict on **82,675 shared numeric fields with 0 mismatches and
0 field-set differences** (5.8 s, 51,934 pair rows, 12 corpora).

```sh
# M3a coherence (the run_full_eval harness measures M3 + M3a and emits the board fulleval)
scripts/run_full_eval.sh W10L9_s4003_packed.bin W10L9_s4003_packed 944
```

Expected M3a **0.862 ± instrument noise** (committed: packed 0.862381,
parent 0.862585) on the post-`299ccc8c` instrument (the append2-coverage
fix — M3a measured on any earlier instrument reads ~0.05 low). M3
(legacy fold) ~0.156 — the fold is visualization-only; M3a is the gate.

```sh
# freeze surfaces
freeze_check --fulleval W10L9_s4003_packed.fulleval.json --profile balanced-2026-08-04
```

Expected: **7 of 8 floors PASS** — F8 (CID22 B9 band tail 0.139 vs ≥ 0.15,
n=43) is the only FAIL; M3a GOLD tier; packed rows measure dial-mono in real
dial units (99.3%). The §5 freeze bar (default `freeze_check`) additionally
FAILs CID22 0.8867 vs ≥ 0.89 — recorded, not contested; see §6.

```sh
# corruption ordering — owned by the companion head, not the dial
sota944_verdict.sh W10L9_s4003_packed.bin <stem>_corrjoint --corruption-head corrhead944_s13.bin
```

Expected head-joint pass_q20 0.79315 / pass_q10 0.92560 (head:
`s3://zentrain/profiles/C-2026-08-05/corrhead944_s13.bin`, 493,716 B, sha256
`31faffa48b82c16d…`); dial-alone 0.1875/0.0625 reported for honesty.

### 4.4 The loop study (closed-loop value; jxl-encoder repo)

- Panel doc: `jxl-encoder/benchmarks/zensim_loop_23shot_sota944_2026-08-05.md`
  (+ `.tsv` + `zensim_loop_23shot_summary_2026-08-05.json` — the stats owner;
  counts/medians are READ from it, never re-derived). Commit `1f89dc66`.
- H3-own-map arm: `zensim_loop_h3own_sota944_2026-08-05.tsv`, commit
  `4e4a7334`; zensim-side fused entry `c28d29b8`
  (`compute_folded944_score_and_attribution` + `Fused944Session`), campaign
  appendix N/N.R.
- Outcome: k2 parity/split, k3 (b)-plus; own-map 17/27 board-best inner
  census, med |err| 1.66 vs generic 1.82, paired 18W/8L/1T, bytes 0.978.

---

## 5. Git commit chain (imazen/zensim `main` unless stated)

| commit | why it is load-bearing |
|---|---|
| `176c4268` | WAVE 10 part 1 — corrects the inverted KADID target on all three ext roots (the fix C trains on; preserved `_INVERTED_` bytes beside it) |
| `ac260c42` | registers appendix H (wave 10) — the LOO design + trainer lineage the winner comes from |
| `e03508ec` | wave-7 KonJND BPG training leg at 944 (reference-disjoint; groups 9+10 of the mix) |
| `177ce09c` | WAVE 10 RESULTS — tkadis negative-marginal + KADID-fix breadth findings; defines arm L9 |
| `532e3a1f` | registers appendix K (wave 11) pre-fit — seeds, k=8 pooling rule, battery |
| `9066fe73` | wave-11 instruments — comparability gate + family-table tool |
| `78418478` | WAVE 11 RESULTS — k=8 confirmation, selection, full winner battery (C's numbers) |
| `ae852b1b` | pack-side dead-column pruning + the metric.rs caller-width fix (`caller_input_width()` sizing — why a pruned C scores correctly) |
| zenanalyze `88410ba6` | `FeatureTransform::Drop` + `Model::caller_input_width` (the wire format C's pruning rides on) |
| `031bd261` | trainer flat-buffer memory refactor (one matrix copy per lane) |
| `0ce3e2f2` | eval+trainer perf pass, BIT-identical outputs (the 4.3-4.5× bake_verdict used for all wave verdicts) |
| `d869a186` | cross-build trainer identity gate — pre/post-flat-buffer builds are one population (validates rebuild-based reproduction) |
| `299ccc8c` | M3a append2-coverage fix — every 944 M3a before it reads low; C's 0.862 is post-fix |
| `3db5a215` | M3a instrument compile fix after the flat-buffer refactor (wave-11 harvest window) |
| `c28d29b8` | the fused folded-944 score+attribution entry (the 944-class codec-loop call shape) |
| jxl-encoder `1f89dc66` | 2/3-shot loop panel with the sota944 candidate |
| jxl-encoder `4e4a7334` | H3-own-map loop results (17/27 at k3) |
| `af4417f8` | fix: `feature-regime-v2`-only builds compile (Fused944Session re-export gate) — found by this ship's test build |
| `4e33e9a6` | feat(profile): Profile C variant + weight + tests + provenance manifest (this ship) |
| *(docs commit)* | this document + CHANGELOG/README/cookbook/mapping-table records — the commit carrying this file |

---

## 6. Known caveats (honest, registered — none contested here)

1. **G-RANGE FAIL at both ends** (K.R battery 2): 4.473% (dial) / 4.497%
   (packed) of cid22val raw predictions sit above the spline's top knot
   (gate < 0.010%) — the worst the class has posted; near-top saturation at
   larger mass. The below-knot flat-bottom segment also collapses 68/866
   CSIQ + 70/779 LIVE pairs into one tie group (max KROCC wiggle 1.7e-3;
   CID22 rank rows BIT-identical). Both ends = the frozen §3d anchor's
   domain vs this bake's raw range. **Registered lever, deliberately not
   applied post-hoc:** amendment-2 anchor densification (near-top AND
   near-bottom).
2. **§5 freeze-bar CID22 row FAILs**: 0.8867 < 0.89 (the bar has been
   cleared only by the `W5_E1_k2` ensemble, 0.89425). The k=8 family median
   is 0.884. C ships as the balanced-profile candidate (7/8), not as a §5
   bar pass, and SOTA is NOT claimed.
3. **F8 B9 band tail** 0.139 vs ≥ 0.15 (n=43) — the 944 class's persistent
   miss; range-restricted per-band SROCC, read with its n.
4. **KonJND** 0.4988 < shipped-B's 0.5186; HF-NL per-ref 0.7334 < B's
   0.8252. The C-vs-B trade is stated in §1.3; B remains the default.
5. **ext_kadid corrected-in-place hazard**: OLD (pre-2026-08-05) embedded
   repros list sha `4dde6be2…` — those reproduce only against
   `ext_kadid_INVERTED_2026-08-04.parquet`, and the 188 pre-correction
   verdicts' KADID cells are annotated (`kadid-ext-root-inverted`,
   `benchmarks/eval_annotations.json`). C is post-fix and unaffected.
6. **Loop k2 split**: the 2-shot panel verdict is (c) at k2 (no win over
   the incumbent arms) and (b)-plus at k3; the own-map advantage is a
   k3 result (N.R). `ZENSIM_H3_GAIN` remains unswept (registered default).
7. **LOO is occlusion, not ablation** (E-M4 lesson) — "keeps both blocks"
   is a marginal-value statement on the fixed trained net.
8. **UPIQ / Korshunov / perf freeze rows** remain externally-owned ATTACH
   rows — not run for this profile ship (same status as every 944 candidate
   to date).
9. **`Zensim::compute()` cannot score C on non-identical pairs** — the
   standard 372-feature pipeline cannot feed a 944 bake; it fails loud
   (`ModelForwardFailed`, test-pinned). Scoring contract: folded-944
   extraction (`feature-regime-v2`) + `score_features_with_profile`, or the
   fused `compute_folded944_score_and_attribution` entry. HDR content is
   out of domain (route to `BHdr`).
10. **CI platform-test jobs on `main` are red for a PRE-EXISTING, tracked
    reason** (issue #55: `v1_golden_bytes` is single-tier-calibrated —
    fails on every non-AVX-512 runner since it landed; also #56, MSCN
    rsqrt vendor-nondeterminism). Verified at ship time: the identical
    failure signature (241/372 golden divergence) appears on the run for
    the pre-ship `main` head, and zero of the last 100 `main` CI runs were
    green. The Profile C ship's own gates — Format, Clippy (all-features,
    `-D warnings`), Feature permutations, MSRV, WASM SIMD128, join-safety,
    corruption census, API-leakage — all PASS on the ship commits, and the
    full workspace suite + all-features clippy are green locally
    (AVX-512 box). A latent `diff_cli` fixture race (masked for days
    behind the golden gate's fail-fast) was found and fixed during this
    ship.

---

## 7. Distribution record (R2 + Tower), sha-verified

Shipped artifact chain (all verified 2026-08-05):

| artifact | bytes | sha256 |
|---|--:|---|
| raw trainer output `W10L9_s4003.bin` | 509,913 | `6b48328054b4a0cbded2e2c1c8cde3e46f507c75de289c8b4309a7625b2ea70c` |
| dial `W10L9_s4003_dial.bin` | 390,449 | `411d6db5e7e3f4a2af8e1c35097eb690645b643b0f8c8edfe224bec60b2f8554` |
| **packed = shipped C weight** | 165,696 | `1a2c8d522fed8034b279ff018aa052f19d0b9f419f12cf22cca303a0b4abb7f4` |
| corruption head `corrhead944_s13.bin` | 493,716 | `31faffa48b82c16d58533167575b927c8b1f2c4c9542b99d7eb9f0dddb906621` |

Equality chain for the shipped weight: source bake
`/mnt/v/output/zensim/bakes/sota944/bakes/W10L9_s4003_packed.bin` == in-repo
`zensim/weights/c_sdr_mlp944_corrmix_2026-08-05.bin` == R2
`s3://zentrain/profiles/C-2026-08-05/c_sdr_mlp944_corrmix_2026-08-05.bin` ==
Tower `/mnt/tower/output/zensim/profiles/C-2026-08-05/c_sdr_mlp944_corrmix_2026-08-05.bin`
(and the wave-11 Tower mirror
`/mnt/tower/output/zensim/bakes/sota944-wave11-2026-08-05/bakes/W10L9_s4003_packed.bin`,
sha spot-checked in K.R5) — all `1a2c8d52…`, each copy round-trip
sha256-verified at write time.

R2 profile prefix `s3://zentrain/profiles/C-2026-08-05/`:
`c_sdr_mlp944_corrmix_2026-08-05.bin`, `W10L9_s4003.bin`,
`W10L9_s4003_dial.bin`, `W10L9_s4003.bin.spec.json`,
`corrhead944_s13.bin`, `PROFILE_C_REPRODUCTION_2026-08-05.md` (this
document; its committed copy in `docs/` is canonical).
Tower mirror `/mnt/tower/output/zensim/profiles/C-2026-08-05/` carries the
same file set.

Training-data uploads performed this pass (gaps found during link
verification, fixed): `s3://zentrain/sota944/tbig_944_200k.parquet`
(+`._MANIFEST.json`), `s3://zentrain/sota944/teacher/{safesyn_teacher944,
tbig_teacher944}.parquet` (+`_MANIFEST.json`),
`s3://zentrain/ext944-canonical-2026-08-01/anchor944_dial.parquet`
(+ manifest) — each round-trip sha-verified against the local canonical
copy; Tower copies of the three previously-Tower-missing tables placed at
`/mnt/tower/output/zensim-sota944-2026-08-03/data/`.

The `--regime 944` eval fixtures (ext944 val legs, dial/corruption grids,
kadis per-pair source) are the standing canonical datasets — locations +
manifests in `~/work/zen/DATA_PROVENANCE.md` (they were already
triple-mirrored before this ship).
