# NEON runs the scalar canonical body — findings and a design for vector bodies

Status: **findings only, nothing implemented or measured.** On aarch64,
eleven canonical-arithmetic kernels currently run the scalar canonical body
instead of a lane-parallel NEON body. The owner set NEON aside on 2026-10-25;
this document records what was verified during the `neon-rev7-20261025`
exploration so a future implementation doesn't re-do the archaeology.

## Why NEON is scalar here

Two commits on 2026-10-03 routed NEON to the scalar canonical body after a
CI aarch64 failure in the REV4VEC gates:

- `d9f1592b4f` "fix(rev4vec): NEON runs the scalar canonical body" — the five
  Rev4 f32 canon drivers (sRGB/linear/PU→XYB, PQ/HLG rows).
- `1961e0805941` "fix(rev4vec2): x86_64-only lane-parallel canon kernels" —
  the six rev4vec2 f64 kernels (Rec64 blur, restore-cuts, dense, gradient,
  fused V-blur SSIM and edge).

Root cause: magetypes' NEON backend implements vector `min`/`max` with
`vminq`/`vmaxq` (AArch64 `FMIN`/`FMAX`), which **propagate** NaN
(`~/work/archmage/magetypes/src/simd/impls/arm_neon.rs`, e.g.
`F32x4Backend::max` → `vmaxq_f32`). The scalar canon clamps with
`f32::max`/`f32::min`, which **drop** NaN — `NaN.max(c) == c`. On a NaN lane
the two disagree (`d9f1592b4f`'s message: `exp2_midp(NaN)` gave NaN on NEON
where the scalar canon gives `MIN_POSITIVE`; the clamped linear→XYB chunk
produced NaN instead of finite values).

x86 needed no fix: `MAXPS`/`MINPS` return the second operand when either
input is NaN, which matches `f32::max`/`min` whenever the bound sits in the
second operand — the shape every canon clamp uses (`value.max(bound)`,
`value.min(bound)`). Consistent with the measured x86 tier parity, not
proven for every call site.

AArch64 has the matching instruction: `FMAXNM`/`FMINNM` (`vmaxnmq`/`vminnmq`)
returns the non-NaN operand — the same semantics `f32::max`/`f32::min`
compile to on aarch64.

## Sites on `main` (Rev1–Rev5 era)

All eleven have the same shape: `#[magetypes]` generates `v4x`/`v4`/`v3`
bodies, and hand-written `_neon` stubs call the scalar canonical body.
`incant!` already lists `neon`, so the stub is what dispatch lands on.
Line numbers are as of `bab88fde` (main, 2026-10-24).

| Site | File | What it is | NaN-sensitive ops in the shared path |
|---|---|---|---|
| `srgb_xyb_canon_vec_neon` | `zensim/src/color.rs:755` | sRGB8→XYB bulk conversion | `CanonChunk::mixed` clamps `mixed.max(0)` per lane |
| `linear_xyb_canon_vec_neon` | `zensim/src/color.rs:822` | linear f32→XYB (`CLAMP` const) | same `CanonChunk` clamp |
| `pu_xyb_canon_vec_neon` | `zensim/src/color.rs:2743` | perceptually-uniform→XYB | `exp2_midp` inside `pu_encode` (clamps internally) |
| `decode_pq_row_canon_vec_neon` | `zensim/src/transfer.rs:374` | PQ→linear row decode | `pow_midp` (→ `exp2_midp`) plus `.max`/`.min` in the body |
| `decode_hlg_row_canon_vec_neon` | `zensim/src/transfer.rs:667` | HLG→linear row decode | `exp_midp` + `pow_midp` plus `.max` in the body |
| `fused_blur_h_ssim_rec64_rows_neon` | `zensim/src/blur.rs:6492` | Rec64 horizontal SSIM blur window | see note below — the scalar row body itself has no float `max`/`min` |
| `restore_cuts_row_work_neon` | `zensim/src/feature_v2/restore_cuts.rs:498` | v2 restore-cuts row kernel | `.max`/`abs` in body |
| `gradient_block_kernel_canon64_vec_neon` | `zensim/src/feature_v2_kernels.rs:5291` | v2 gradient block kernel | `.min` in body; `.max(zero)` in the shared `*_v` helpers |
| `dense_block_kernel_canon64_vec_neon` | `zensim/src/feature_v2_oracle.rs:1824` | v2 dense block kernel (production canon64, colocated with the f64 oracle) | `dense_terms32_v` → `.max(zero)` helpers |
| `fused_vblur_ssim_canon64_vec_neon` | `zensim/src/fused.rs:6436` | fused V-blur SSIM fold | `.max`/`abs` in body and `ssim_d_local_v` |
| `fused_vblur_edge_canon64_vec_neon` | `zensim/src/fused.rs:7131` | fused V-blur edge fold | `.max`/`abs` in body |

Shared helpers carrying `.max(zero)` into the v2 kernels:
`ssim_d_local_v`, `bounded_excess_v`, `bounded_excess_pair_v`, `saturate_v`
(`zensim/src/feature_v2_kernels.rs:79–170`).

**Rec64 caveat.** The x86 tiers run bespoke AVX-512/AVX2 kernels
(`rec64_rows8x8_avx*` around `blur.rs:6245`), not `#[magetypes]` bodies, and
the scalar Rec64 row body (`fused_blur_h_rec64_row`, `blur.rs:5289`) contains
no float `max`/`min` — only `usize` index clamps. The stub comment carries
the same generic NaN justification, but a NEON port here is a new bespoke
kernel (or a new `#[magetypes]` f64x8 body), not a stub swap. Whether the
bulk 2026-10-03 fix deliberately swept it in or it was collateral is not
recorded; audit before writing one.

## On `explore/byconstruction` (Rev6/Rev7)

Same eleven sites, same function names — **no additional NEON stubs.** What
the branch adds is Rev6/Rev7 formula paths inside the same shared bodies
(e.g. `dense_block_kernel_canon64_vec<P, const R7: bool>` gains the `R7`
flag). A fix written on main must be forward-ported so the rev6/rev7 paths
are exercised too; conversely a fix written on the branch covers every
revision because the bodies are shared.

## Verified facts

- NEON vector `min`/`max` → `vminq`/`vmaxq`: NaN-propagating, operand-order
  independent. `vmaxnmq`/`vminnmq` give the canon's `f32::max`/`min`
  semantics. Same story at f64 width (`vmaxnmq_f64`/`vminnmq_f64`).
- Value-based NEON intrinsics are safe inside `#[target_feature]` functions
  — pointer loads/stores are the unsafe category. `vmaxnmq_f32`,
  `vminnmq_f32`, `vmaxnmq_f64`, `vminnmq_f64` all compile in a
  `#[target_feature(enable = "neon")]` fn:
  `~/work/archmage/tests/arm_safe_intrinsics.rs:42–43,101–102`. This is how
  `zensim`'s `#![forbid(unsafe_code)]` can stay: `#[magetypes]`/`#[rite]`
  emit the target-feature boundary, the intrinsics inside stay safe.
- magetypes NEON reprs (`simd/impls/arm_neon.rs`):
  `f32x16<NeonToken> = [float32x4_t; 4]` (impl at :4542),
  `f32x8<NeonToken> = [float32x4_t; 2]` (:295),
  `f64x8<NeonToken> = [float64x2_t; 4]` (:4840). `into_repr()` /
  `from_repr_t(token, repr)` are public, so a zensim-local NEON helper can
  unwrap, apply `v*nmq` per element, and rewrap without touching magetypes.
- Of the `*_midp` family, only `exp2_midp` clamps:
  `clamped = x.max(-126.0).min(128.0)` then `xi = round().min(127.0)`
  (scalar `det_math.rs:485–486`; vector
  `transcendentals_f32x16.rs:253,279`). `exp_midp`, `pow_midp`,
  `sigmoid_midp`, `silu_midp` inherit through it. `log2_midp`,
  `log10_midp`, `ln_midp`, `cbrt_midp` contain no `max`/`min` and are
  already NEON-correct.
- `f32::mul_add` on NEON (`vfmaq`) is a true fused FMA — same single
  rounding as the scalar canon on a fused tier. Not a divergence source.

## Design direction (not implemented)

Keep one shared generic body per kernel; route every NaN-sensitive op
through a zensim-local trait:

- A `CanonNum`-style trait with `canon_max`/`canon_min` implemented per
  concrete `(vector type, token)` pair — `f32x16`/`f32x8`/`f64x8` ×
  `X64V3Token`/`X64V4Token`/`X64V4xToken`/`NeonToken`. x86 impls delegate
  to `.max()`/`.min()` (already bit-identical, zero codegen change). The
  `NeonToken` impls unwrap the repr, emit `vmaxnmq`/`vminnmq` per element,
  rewrap. Rust's no-specialization means one small impl per pair — about
  twelve one-line bodies.
- `exp2_midp`'s two clamps need the same treatment: a NEON twin of the
  generated body (same op sequence, nm ops in place of `max`/`min`), or a
  zensim-local `canon_exp2_midp` calling the trait. The generated
  `transcendentals_f32x16.rs` body is the sequence to replicate.
- Then `neon` goes back in each `#[magetypes]` tier list, the eleven stubs
  are deleted, and the shared bodies' `.max(`/`.min(` become
  `x.canon_max(`/`x.canon_min(` (helpers get a `V: CanonNum`-style bound).
- Rec64 blur is separate (see caveat above): bespoke `rec64_rows8x8_neon`
  or a new `#[magetypes]` f64x8 body.

## Gates a future implementation must pass

From the (unexecuted) implementation brief
`~/tmp/handoff/neon-rev7-20261025/brief.md`:

1. `cargo test -p zensim --all-features` on aarch64, including the tier
   parity tests (`featcanon_*_parity`, `cross_tier`,
   `restore_cuts_parity`, `feature_invariants`).
2. A NaN regression test that fails under `FMAX`/`FMIN` and passes under
   `FMAXNM`/`FMINNM` — demonstrated by a temporary, uncommitted
   `vmaxq`-style edit.
3. Bit-identity: `byc_score_timing --dump` output identical between NEON
   and forced-scalar across synthetic geometries and case images at 1 and
   8 threads. Note `byc_score_timing --tier scalar` currently only masks
   x86 tokens — it needs an aarch64 extension before this comparison is
   possible on ARM.
4. Speed: paired timing rounds vs the scalar body on an aarch64 machine,
   per-kernel `perf record` shares. Nothing has been measured; how much of
   NEON zensim runtime is currently scalar is not measured either.

Constraints that survive any design: no public API, `#![forbid(unsafe_code)]`
stays, no test expectation relaxed, exact per-lane op order and reduction
trees preserved (no FMA where the canon doesn't use `mul_add`).
