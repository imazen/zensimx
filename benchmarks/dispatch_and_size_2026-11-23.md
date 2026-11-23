# Dispatch & binary-size leaf optimizations — 2026-11-23

Lane: `swe-zensim` (`[LEAF]`), baseline `abec986d` (`chore(deps): upgrade
archmage/magetypes to 0.9.33`). Host: i134 (Intel i5-13400F), P-core
confinement via `systemd-run --scope -p AllowedCPUs` (zenbench rejects
taskset/group-affinity as insufficient on asymmetric topology; criterion
runs under `taskset -c 2`).

Three publication-free changes inside `zensim` only. No public API, feature,
weight, or tolerance change. Every floating-point arithmetic order is
preserved; the only structural change is *where* the same work is
instantiated.

## 1. What changed

### 1a. `edge8!` masked-load head/tail removed (`feature_v2_kernels.rs`)

The `edge8!` macro evaluated the first (`1..head_end`) and last
(`interior_end-8..interior_end`) interior columns through
`f32x8::load_t`/`store_t` masked lane ops, which lower to `vmaskmov`
on AVX2. It only ran when `vec_edges = !BANK && r4.is_none() &&
interior_end >= 9` — i.e. exactly when its bank/Rev4 side-effects were
inert — and its per-lane body is the same arithmetic as the existing
scalar fallback (`gradient_interior_elem_canon`, f32 canonical order).
The head/tail loops now call the scalar path unconditionally: identical
bits, and the masked-load instructions are gone.

### 1b. `go64!` const-generic matrix collapsed (`feature_v2_kernels.rs`)

`gradient_block_kernel_canon64_vec` was instantiated over
`<P, BANDVIS, BV_DSTACT, BANK, HPR, R7>` — a 6-way feature combination
times the `{HPR,R7}` forms per pool type, ~36 monomorphizations per
SIMD tier (144 `*canon64_vec*` symbols in the bench binary). `BANDVIS`,
`BV_DSTACT`, and `BANK` are branch-invariant for a whole image: they are
now runtime `bool` args threaded to `gradient_terms64`,
`gradient_terms32_v`, and `gradient_interior_elem_canon`. The dispatch
keeps const generics only where they change the arithmetic itself —
`<P, HPR, R7>` — so `Mode::Canon64` has 6 arms instead of 36. The
oracle `exact` path keeps its const generics (it is a test ruler).

### 1c. `foldapp_streaming_walk_impl` type-erased (`feature_v2.rs` and callers)

`foldapp_streaming_walk_impl::<S, D, ALL_CHANNELS>` monomorphized the
~200-line orchestration body plus the whole `StripPlaneProducer` chain
once per `(S, D)` source pair per channel shape (4 instantiations in the
bench binary, ~48–50 KB each). The producer's only source-type contact
is `row_bytes` once per row, so `&dyn ImageSource` costs two row-access
calls per row across the pair — cold next to the per-row conversion
work — while the downstream kernels are already
non-generic over `&[f32]` planes. `impl`/`rev5_block_walk`/
`stream_phase_*`/`convert_source_to_xyb*`/`Sampling::{pyramid,reference}`/
`reject_hdr_input`/`reflect_pad_*`/`prepare_v2_reference_impl*` now take
`impl ImageSource + ?Sized` (the existing `SubsetView` pattern), and the
impl takes `&dyn ImageSource` — one instantiation per `ALL_CHANNELS`.

### 1d. Oracle gating — verified, no change needed

The brief proposed gating `include!("feature_v2_oracle.rs")`. The file is
misnamed: it owns the *shipped* era-2/canon/canonical-64 kernel bodies
(`dense_block_kernel_era2`, `dense_block_kernel_canon*`,
`gradient_block_kernel_canon*`), so gating the include breaks the lib.
Its genuinely oracle-only items (`mod oracle`, `bench_*`/`harness_*`,
`*_exact` kernels) already carry `cfg(any(test, feature = "oracle"))` —
a production build excludes them today. A comment at the include site
now records this.

## 2. BEFORE attribution (i134, baseline `abec986d`)

`zensim-bench` (excluded workspace) fails dependency resolution at its
committed pins (`zenjpeg 4b0f1bfa` et al. bound `archmage <= 0.9.31-beta.1`,
incompatible with 0.9.33 — pre-existing breakage at the baseline commit).
Both phases built `zensim-bench` with pins synced to the revs the root
`Cargo.lock` already resolves (`zenjpeg f39ab835`, `zenwebp b123f328`,
`zenavif 4221a7c0`, `zenjxl 1d14f0e0`, `jxl-encoder c087fbd6`,
`zenjxl-decoder e1c366b0`, `zenanalyze 0bc01d3f`). Scratch-only patch; no
manifest changes are committed.

| metric | tier_isolation | fold_engine_bench |
|---|---|---|
| GNU `size` "text" column (`.text`+rodata/etc aggregate) | 8,343,428 B | 7,821,910 B |
| ELF `.text` section (`readelf -S`) | 6,916,473 B | 6,503,509 B |
| `gradient_block_kernel_canon64_vec` symbols | 144 | 144 |
| `foldapp_walk`/`rev5_block`/`StripPlaneProducer` symbols | 233 | – |
| `vmaskmov` insns (objdump) | 297 | – |

criterion `zensim/*` arms (`--bench --sample-size 10 --measurement-time 3
--warm-up-time 1`, `taskset -c 2`):

| arm | BEFORE |
|---|---|
| 256x256 v3(avx2) | 2.582 ms |
| 256x256 scalar | 11.483 ms |
| 512x512 v3(avx2) | 9.788 ms |
| 512x512 scalar | 44.493 ms |
| 1920x1080 v3(avx2) | 71.335 ms |
| 1920x1080 scalar | 346.81 ms |

`fold_engine_bench` `ZEN_FE_RSS` arms (30 iters × 1152², cpuset scope on
P-core 4, 3 runs — median):

| arm | BEFORE wall | BEFORE max RSS |
|---|---|---|
| score_fold | 1.39 s | 35.3 MB |
| score_buffered | 1.37 s | 35.1 MB |
| ref_fold | 1.26 s | 53.8 MB |
| ref_buffered | 1.27 s | 53.7 MB |
| poolctl_full | 2.32 s | 37.4 MB |

## 3. AFTER attribution

Identical build/run procedure. The first attempt measured nothing —
cargo's mtime fingerprint saw the rsynced tree as "unchanged" and reused
the BEFORE objects (`Finished in 0.15s`); the table below is the real
rebuild after the script's touch pass.

| metric | tier_isolation | fold_engine_bench |
|---|---|---|
| GNU `size` "text" column (`.text`+rodata/etc aggregate) | 6,468,856 B | 5,947,346 B |
| ELF `.text` section (`readelf -S`) | 5,085,161 B | 4,672,213 B |
| `gradient_block_kernel_canon64_vec` symbols | 22 | 22 |
| `foldapp_walk`/`rev5_block`/`StripPlaneProducer` symbols | 131 | 128 |
| `vmaskmov` insns (objdump) | 29 | 29 |

**Δ GNU size "text": −1,874,572 B (−22.5 %) tier_isolation; −1,874,564 B
(−24.0 %) fold_engine_bench. Δ ELF `.text` section: −1,831,312 B
(−26.5 %) tier_isolation; −1,831,296 B (−28.2 %) fold_engine_bench.**
`foldapp_streaming_walk_impl` shrank from
4 concrete instantiations (`{OwnedImage,RgbSlice}² × {true,false}`) to 2
(`<true>`/`<false>` over `&dyn ImageSource`), and `rev5_block_walk`
likewise instantiates once over `dyn`.

criterion `zensim/*` arms, same flags:

| arm | BEFORE | AFTER | Δ |
|---|---|---|---|
| 256x256 v3(avx2) | 2.582 ms | 2.780 ms | +7.7 % |
| 256x256 scalar | 11.483 ms | 11.586 ms | +0.9 % |
| 512x512 v3(avx2) | 9.788 ms | 10.130 ms | +3.5 % |
| 512x512 scalar | 44.493 ms | 45.372 ms | +2.0 % |
| 1920x1080 v3(avx2) | 71.335 ms | 72.177 ms | +1.2 % |
| 1920x1080 scalar | 346.81 ms | 346.46 ms | −0.1 % |

`ZEN_FE_RSS` arms (median of 3 × 30 iters × 1152²):

| arm | BEFORE wall | AFTER wall | Δ |
|---|---|---|---|
| score_fold | 1.39 s | 1.37 s | −1.4 % |
| score_buffered | 1.37 s | 1.37 s | 0.0 % |
| ref_fold | 1.26 s | 1.27 s | +0.8 % |
| ref_buffered | 1.27 s | 1.27 s | 0.0 % |
| poolctl_full | 2.32 s | 2.22 s | **−4.3 %** |

The scalar tier is flat and every end-to-end fold arm is flat-to-faster,
so the regression lives in the buffered v3 path and shrinks with size
(7.7 % → 3.5 % → 1.2 %): the runtime `bandvis`/`bv_dstact`/`gmsbank`
bools now evaluate where const generics used to fold them, and the edge
columns that ran masked `f32x8` lanes run the same arithmetic scalar.
That is the accepted trade for −1.87 MB; hoisting the flag checks to
chunk level would recover part of it if the small-image latency matters.

## 4. Parity / gate evidence

Commands run on i265 (Core Ultra 7 265K, asymmetric — bench mains need
`systemd-run --scope -p AllowedCPUs=0-7`; the zenbench CPU gate rejects
taskset as "group affinity insufficient"):

- `cargo test --release -p zensim --all-targets` — **all lib and
  integration tests pass**; run stops at `fold_engine_bench`, which
  panics at baseline too (`feat_*` arms + rev5 masked features — proven
  on the untouched baseline elf on i134). Other bench mains pass:
  `era2_accum_shape` ✓, `pu21_bench` ✓ (a concurrent rerun hit only
  zenbench's global admission lock).
- `cargo test --release -p zensim --features oracle --all-targets` —
  same result; oracle benches (`era2_dense_ab`, `featacc_extract_ab`,
  extract paths) compile and run to completion.
- `cargo test --release -p zensim --features training --test
  fold_engine_parity` — **11/11 pass, 0 ignored** (incl.
  `compute_is_bit_identical_across_engines`,
  `extended_features_are_bit_identical_across_engines`,
  `both_engines_are_bit_identical_across_rayon_pool_sizes`,
  `golden_real_fixture_is_bit_identical_across_engines`).
- `cargo test --release -p zensim --features training
  v1_372_bit_exact_to_fold_at_every_width` — **1/1 pass, 0 ignored**
  (19 geometries, bit-exact v1-372 vs the fold).
- `cargo clippy -p zensim --all-targets -- -D warnings` — clean.
- `cargo clippy --manifest-path zensim-bench/Cargo.toml --bench
  tier_isolation -- -D warnings` — clean (with lockfile-synced pins;
  committed pins don't resolve at all at baseline, and other
  `zensim-bench` targets have pre-existing `zenpredict` API drift —
  neither is this lane's code).
- `cargo fmt --check -p zensim` — clean.
- Feature-matrix `cargo check`: default, `oracle`, `viz-memory`, and
  `--no-default-features --features feature-regime-v2` — all clean.

## 5. Measurement notes

- `feat_buffered`/`feat_fold` (`fold_engine_bench`) and
  `extract_features/*` (`tier_isolation`) panic at baseline with
  `ModelLoadFailed`: rev5 + `codec_target()` reject extended masked
  features (`metric.rs` "formula revision 5 supports only basic + peaks +
  v2"). Pre-existing; identical in both phases; excluded from comparison.
- zenbench refuses an unconfined asymmetric box ("group affinity mask …
  insufficient"): runs use `systemd-run --scope -p AllowedCPUs=<P-core>`
  which satisfies `cpu_enforcement=cpuset`.
- rsync `-a` preserves old source mtimes; cargo's mtime fingerprint then
  reuses stale objects silently. The measure script touches all sources
  before building — verify `build-*.log` shows real compile, not
  `Finished in 0.15s`.
- The 29 residual `vmaskmov` insns all disassemble inside
  `zensim::feature_v2::__arcane_dense_block_kernel_era2_entry_v3` (the
  AVX2 tier of the era-2 dense kernel entry) — none of the removed
  `edge8!` sites contributed to them.
