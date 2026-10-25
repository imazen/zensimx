//! Featcanon (2026-09-25): canonical feature arithmetic behind
//! [`crate::feature_defs::FormulaRevision::Rev4`], plus the measurement
//! substrate that chose it (`benchmarks/featcanon_WORKLOG.md`).
//!
//! The problem: several extraction kernels emit different bits per SIMD tier
//! because (a) `mul_add` is fused on v3/v4/neon but plain `a*b+c` on the
//! magetypes scalar and wasm128 backends, (b) `reduce_add`'s tree is
//! backend-defined (16-lane native on v4, 2×8-lane on v3, array-order on
//! scalar), and (c) per-tier tail/chunk geometry differs (width-16 vs width-8
//! vs remainder paths). [`era2_reduce8`] established the
//! fix pattern for the dense block kernel; this module generalises it: one
//! body, fixed virtual-lane accumulation, a written-out reduction tree, and
//! `f32::mul_add` (the fused intrinsic on every target) wherever the formula
//! is a product-sum.
//!
//! **The arithmetic is chosen PER COMPUTATION, never per process.** Every
//! canonical leaf receives the revision of the computation it serves and asks
//! [`mode`] with it. `ZENSIM_FORMULA_REV` does not reach a leaf through this
//! module: it only chooses which revision a caller that did not name one
//! gets, and the entries refuse every request whose revision disagrees with
//! the process revision when either is Rev4 (see
//! [`crate::ssim_form::refuse_rev4_mix`]), because other formula gates still
//! read the process switch.
//!
//! **Served at Rev4 (REV4SERVE).** The served leaves are canonical too:
//! the PU front end runs `color::pu_xyb_canon` over the `_at_revision`
//! transfer decoders, `streaming::active_channels` routes every active
//! channel through the fused SSIM kernels at Rev4 (the edge-only
//! `blur::fused_blur_h_mu` chain and the MSE-only `sq_diff_sum` leaf never
//! run), and the fold walk's block kernels already select their canon arms.
//! Rev1–Rev3 bytes are unchanged; a Rev4 process serves Rev4 computations
//! bit-identically across tiers, and [`crate::ssim_form::refuse_rev4_mix`]
//! keeps every cross-boundary mix refused.
//!
//! **Measurement modes are not in a product build.** `ZENSIM_FEATCANON`
//! (`exact`/`c32`/`c64`/`neum`/`off`) and the f64 exact-oracle bodies exist
//! only with the `oracle` cargo feature, the crate's precedent for a ruler
//! that is not semantics. A product build knows two arithmetics: production
//! (Rev1–Rev3) and [`Mode::Canon64`] (Rev4).

use crate::feature_defs::FormulaRevision;

/// Arithmetic mode of the canonical leaves for ONE computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Mode {
    /// Production dispatch (no canonical routing): Rev1–Rev3.
    Off,
    /// Candidate (d) — the ORIGINAL Rev4 arithmetic: fused `mul_add`
    /// elements + fixed 8-virtual-lane f32 partials + fixed pairwise
    /// reduction (the era-2 shape) over the f32 sliding blur. Superseded by
    /// [`Mode::Canon64`] (rev4canon, 2026-09-30); it survives only as the
    /// `c32` oracle arm so the reviewed canon stays reproducible bit for
    /// bit — a product build cannot select it.
    #[cfg(feature = "oracle")]
    Canon32,
    /// f64 element evaluation + compensated f64 accumulation — the exact
    /// oracle arm. Measurement only.
    #[cfg(feature = "oracle")]
    Exact,
    /// THE REV4 CANON (rev4canon): fused `mul_add` elements + fixed
    /// 8-virtual-lane f64 partials + the same fixed pairwise reduction as
    /// (d), over the f64 sliding blur recurrence ([`BlurMode::Rec64`]).
    /// FEATACC's measured recommendation; the only arithmetic a product
    /// build can select at Rev4.
    Canon64,
    /// Candidate (f): fused `mul_add` elements + Neumaier-compensated f64
    /// accumulation. Measurement only.
    #[cfg(feature = "oracle")]
    CanonNeum,
}

impl Mode {
    /// `true` when the canonical/candidate path should run (any non-Off mode).
    pub(crate) const fn active(self) -> bool {
        !matches!(self, Mode::Off)
    }

    /// `true` when this is the exact-oracle arm (`oracle` builds only; a
    /// product build has no such arm).
    #[cfg(feature = "oracle")]
    #[inline(always)]
    pub(crate) const fn exact(self) -> bool {
        matches!(self, Mode::Exact)
    }

    /// Product-build sibling of the oracle `exact()`: never true, so
    /// `if mode.exact()` folds away wherever it is written unconditionally.
    #[cfg(not(feature = "oracle"))]
    #[allow(dead_code)]
    #[inline(always)]
    pub(crate) const fn exact(self) -> bool {
        false
    }
}

/// The canonical-leaf arithmetic for a computation at `revision`.
///
/// `revision` is the revision the leaf's CALLER is computing — the one it
/// already carries for its other formula gates (`fused_blur_h_ssim_at_revision`'s
/// `revision`, `FreeExtrasWork::revision()`, `V2NewFeatureToggles::formula_revision`,
/// `ZensimConfig::formula_revision`). `Rev4` selects [`Mode::Canon64`], the
/// featacc-measured recommendation landed by rev4canon; every earlier revision
/// selects [`Mode::Off`], so Rev1–Rev3 bytes cannot depend on this module.
///
/// With the `oracle` feature, `ZENSIM_FEATCANON` overrides the result for
/// measurement (`exact`/`c32`/`c64`/`neum`, or `off` to force production even
/// at Rev4). The override is read once per process; a product build does not
/// contain it.
///
/// REV4SERVE reach note: the `_at_revision` PU-XYB/PU21/transfer decoders
/// dispatch their canonical bodies on `mode(revision).active()`, so an
/// oracle build's `ZENSIM_FEATCANON` forces canonical PU/transfer bits at
/// Rev1–Rev3 where a product run uses the production bodies — the same
/// convention `srgb_to_positive_xyb_planar_into_at_revision` already had.
/// Measurement runs under that env get canon bits by design; product
/// builds are unaffected (`mode() == Off` below Rev4).
#[inline]
pub(crate) fn mode(revision: FormulaRevision) -> Mode {
    #[cfg(feature = "oracle")]
    if let Some(m) = measurement_override() {
        return m;
    }
    if revision >= FormulaRevision::Rev4 {
        Mode::Canon64
    } else {
        Mode::Off
    }
}

/// Process-bound arithmetic for legacy research helpers that receive no revision.
/// Raw wide extraction guards require matching process/request revisions before
/// these helpers run. Model-bound Rev5 serving instead passes the requested
/// revision explicitly to its canonical leaves and accumulator constructors.
#[inline]
pub(crate) fn compute_mode() -> Mode {
    mode(crate::ssim_form::active_revision())
}

/// `ZENSIM_FEATCANON`, read once — into a cell the featacc cost bench can
/// then overwrite in-process (`bench_set_measurement`), so one zenbench
/// group can interleave candidates fairly. `None` when unset/unrecognised.
#[cfg(feature = "oracle")]
fn measurement_override() -> Option<Mode> {
    *override_cell().read().unwrap_or_else(|e| e.into_inner())
}

/// The shared cell. Initialised from `ZENSIM_FEATCANON` on first read;
/// `bench_set_measurement` rewrites it.
#[cfg(feature = "oracle")]
fn override_cell() -> &'static std::sync::RwLock<Option<Mode>> {
    use std::sync::{OnceLock, RwLock};
    static M: OnceLock<RwLock<Option<Mode>>> = OnceLock::new();
    M.get_or_init(|| {
        RwLock::new(match std::env::var("ZENSIM_FEATCANON").as_deref() {
            Ok("exact") => Some(Mode::Exact),
            Ok("c32") => Some(Mode::Canon32),
            Ok("c64") => Some(Mode::Canon64),
            Ok("neum") => Some(Mode::CanonNeum),
            Ok("off") => Some(Mode::Off),
            _ => None,
        })
    })
}

/// `ZENSIM_FEATCANON_BLUR`, read once into a cell — same bench reasoning as
/// [`override_cell`]. `None` = unset/unrecognised (the default then applies).
#[cfg(feature = "oracle")]
fn blur_cell() -> &'static std::sync::RwLock<Option<BlurMode>> {
    use std::sync::{OnceLock, RwLock};
    static B: OnceLock<RwLock<Option<BlurMode>>> = OnceLock::new();
    B.get_or_init(|| {
        RwLock::new(match std::env::var("ZENSIM_FEATCANON_BLUR").as_deref() {
            Ok("rec") => Some(BlurMode::Rec),
            Ok("rec64") => Some(BlurMode::Rec64),
            Ok("local") => Some(BlurMode::Local),
            Ok("fresh") => Some(BlurMode::Fresh),
            _ => None,
        })
    })
}

/// featacc cost bench: set the measurement override (and, when `Some`, the
/// blur axis) in-process so a zenbench group can interleave candidates
/// instead of paying a process restart per mode. `None` restores "env
/// unset". Measurement-only: exists solely under `oracle`.
#[cfg(all(feature = "oracle", feature = "feature-regime-v2"))]
pub(crate) fn bench_set_measurement(mode: Option<Mode>, blur: Option<BlurMode>) {
    *override_cell().write().unwrap_or_else(|e| e.into_inner()) = mode;
    if let Some(b) = blur {
        *blur_cell().write().unwrap_or_else(|e| e.into_inner()) = Some(b);
    }
}

/// featacc D2: the measurement override, `Off` when unset — for kernels
/// whose production arithmetic is NOT canon-dispatched (dense era-1/era-2,
/// [`mode`], this never synthesises `Canon32`, so it cannot move a product
/// Rev4 byte: with no `ZENSIM_FEATCANON` in the environment it returns
/// `Off` and every kernel that consults it runs its shipped body verbatim.
/// `Off` itself is a real candidate ("the uncanonized baseline"), so the
/// dispatch sites gate on `active()` only when routing to a canon body.
#[cfg(feature = "oracle")]
#[inline]
pub(crate) fn measurement_mode() -> Mode {
    measurement_override().unwrap_or(Mode::Off)
}

// ============================================================================
// The blur-recurrence axis (featacc) — `Rec64` is the Rev4 canon; the rest
// of the axis is measurement-only
// ============================================================================
//
// The blurred planes feeding every feature kernel carry a THIRD kind of
// numerical error, separate from element rounding and accumulation order:
// the production box/SSIM blurs evaluate each window by updating the
// previous window's sum (`sum + add − rem`), so rounding error accrues
// with position rather than being re-bounded per output. `BlurMode` names
// the axis: `Rec64` is the Rev4 canon's choice; under `oracle`
// `ZENSIM_FEATCANON_BLUR` isolates it for measurement:

/// How the blur windows evaluate their sums. `Rec` is the shipped form for
/// Rev1–Rev3; `Rec64` is the Rev4 canon; `Local` is the Rev5 canon.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BlurMode {
    /// Production: per-column (V) / per-row (H) sliding f32 sums
    /// (`sum = sum + add − rem`).
    Rec,
    /// The same sliding recurrence in f64 — THE REV4 CANON's blur axis
    /// (rev4canon). Under `oracle` it is also a measurement arm measuring
    /// the f32 storage of the running sums, not the recurrence's existence.
    Rec64,
    /// THE REV5 CANON's blur axis (`localwin`): each output's 11-tap window
    /// is summed independently in f32 over the spec's fixed pair tree —
    /// `s2 = x[i]+x[i+1]; s4 = s2[i]+s2[i+2]; s8 = s4[i]+s4[i+4];
    /// w = s8 + s2[8] + x[10]` — with [`tap_mirror`] padding. No recurrence,
    /// so rounding cannot travel along a row or strip.
    Local,
    /// Per-position window re-summation in f64 — the strongest reference:
    /// no drift term at all.
    #[cfg(feature = "oracle")]
    Fresh,
}

/// `ZENSIM_FEATCANON_BLUR` (`rec`/`rec64`/`fresh`), read once — plus the
/// default: `exact` mode re-sums (`Fresh`); every other candidate keeps the
/// production recurrence so its column measures accumulation alone. In a
/// product build this is a constant `Rec`.
#[cfg(feature = "oracle")]
#[inline]
pub(crate) fn blur_axis() -> BlurMode {
    blur_cell()
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .unwrap_or_else(|| {
            if measurement_mode().exact() {
                BlurMode::Fresh
            } else {
                BlurMode::Rec
            }
        })
}

/// Non-oracle sibling: the blur axis does not exist, only `Rec` ships.
#[cfg(not(feature = "oracle"))]
#[inline(always)]
#[allow(dead_code)]
pub(crate) const fn blur_axis() -> BlurMode {
    BlurMode::Rec
}

/// The blur recurrence a canonical BODY runs under `mode` — distinct from
/// the dispatch signal [`blur_axis`], which only answers "did an oracle
/// knob ask for a non-production blur". The canon body's axis is the mode's
/// own: `Fresh` for the exact arm, `Rec` (the shipped f32 sliding sums) for
/// the superseded `c32` candidate so `ZENSIM_FEATCANON=c32` reproduces the
/// reviewed Rev4 canon bit for bit, `Rec64` — the Rev4 canon — otherwise.
/// An explicit `ZENSIM_FEATCANON_BLUR` still overrides under `oracle`.
#[cfg(feature = "oracle")]
#[inline]
pub(crate) fn canon_blur_axis(mode: Mode) -> BlurMode {
    blur_cell()
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .unwrap_or(match mode {
            Mode::Exact => BlurMode::Fresh,
            Mode::Canon64 => BlurMode::Rec64,
            _ => BlurMode::Rec,
        })
}

/// Non-oracle sibling: the only canon mode a product build can select is
/// `Canon64`, whose blur is the f64 recurrence — a constant here.
#[cfg(not(feature = "oracle"))]
#[inline(always)]
pub(crate) const fn canon_blur_axis(_mode: Mode) -> BlurMode {
    BlurMode::Rec64
}

/// The blur axis a canonical body computes at `revision` — [`canon_blur_axis`]
/// owns the mode's default; this adds the era split: a Rev5 computation runs
/// `Local` windows (the `localwin` era) wherever Rev4 ran `Rec64`. The exact
/// arm still re-sums (`Fresh`); an explicit `ZENSIM_FEATCANON_BLUR` still
/// overrides under `oracle`.
#[cfg(feature = "oracle")]
#[inline]
pub(crate) fn canon_blur_axis_for(
    mode: Mode,
    revision: crate::feature_defs::FormulaRevision,
) -> BlurMode {
    blur_cell()
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .unwrap_or(match mode {
            Mode::Exact => BlurMode::Fresh,
            _ if revision >= crate::feature_defs::FormulaRevision::Rev5 => BlurMode::Local,
            Mode::Canon64 => BlurMode::Rec64,
            _ => BlurMode::Rec,
        })
}

/// Non-oracle sibling: `Local` at Rev5+, `Rec64` below (canonical bodies only
/// run under an active canon mode, i.e. Rev4+).
#[cfg(not(feature = "oracle"))]
#[inline(always)]
pub(crate) fn canon_blur_axis_for(
    _mode: Mode,
    revision: crate::feature_defs::FormulaRevision,
) -> BlurMode {
    if revision >= crate::feature_defs::FormulaRevision::Rev5 {
        BlurMode::Local
    } else {
        BlurMode::Rec64
    }
}

/// The boundary map the blur recurrences implement, for the `Fresh` arm's
/// per-position window: reflect once toward the image, clamp if still out
/// (the kernels' `mirror_idx`/`vblur_add_idx`/`vblur_rem_idx` all reduce to
/// this: `j < 0 → -j`, `j ≥ n → 2(n−1) − j`, both clamped into `[0, n−1]`).
/// NOT the crate's periodic [`crate::feature_v2::reflect_101`] — the sliding
/// kernels' convention is what `fresh` must replay exactly, so a `rec` vs
/// `fresh` diff measures recurrence drift, not a boundary convention. Rev5's
/// `Local` windows pad with the same map, so a `local` vs `fresh` diff also
/// measures the f32 pair tree alone, not a boundary convention.
#[inline(always)]
pub(crate) fn tap_mirror(j: isize, n: usize) -> usize {
    let n1 = n - 1;
    if j < 0 {
        j.unsigned_abs().min(n1)
    } else if j as usize > n1 {
        (2 * n1).saturating_sub(j as usize).min(n1)
    } else {
        j as usize
    }
}

// ============================================================================
// Canonical accumulators — the three measured candidates
// ============================================================================

/// The era-2 horizontal reduction — **part of the semantics**.
///
/// Pairwise rather than sequential (tighter error, equally fixed), and written
/// out rather than delegated: `GenericF32x8::reduce_add` resolves to a
/// per-backend order (§14.2 of `benchmarks/era2_perf_break_2026-08-31.md`),
/// so calling it would make the reduction tree an unspecified, tier-dependent
/// operation — precisely what the era-2 identity theorem forbids.
///
/// Owned here, not in `feature_v2` (which re-exports it for the era-2 dense
/// kernel), because the canonical leaves compile in every build and
/// `feature_v2` exists only with `feature-regime-v2`.
#[inline(always)]
pub(crate) fn era2_reduce8(a: [f32; 8]) -> f64 {
    (((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7]))) as f64
}

/// The canonical lane count. 8 because it is the largest power of two that
/// every production tier maps onto without remainder (the era-2 choice,
/// `benchmarks/era2_perf_break_2026-08-31.md` §2.0).
pub(crate) const CANON_LANES: usize = 8;

/// Candidate (d): 8 f32 virtual lanes, term j → lane `j % 8`, tail folds into
/// the same lanes, closed by the fixed [`era2_reduce8`]
/// pairwise tree. Width-independent by construction.
///
/// rev4canon: the SUPERSEDED Rev4 canon — `oracle` builds only, as the `c32`
/// reproduction arm; the shipped Rev4 canon is [`LanesF64`].
#[cfg(feature = "oracle")]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct LanesF32(pub [f32; CANON_LANES]);

#[cfg(feature = "oracle")]
impl LanesF32 {
    #[inline(always)]
    pub(crate) fn zero() -> Self {
        Self([0.0; 8])
    }
    /// Add one full 8-wide chunk lane-wise.
    // dead_code until the chunked-pool kernels land; kept — `Pool::add`
    // covers the per-element callers first.
    #[allow(dead_code)]
    #[inline(always)]
    pub(crate) fn add_chunk(&mut self, c: &[f32; 8]) {
        for (l, &v) in self.0.iter_mut().zip(c.iter()) {
            *l += v;
        }
    }
    /// Add one element to lane `i` — the explicit-lane form for kernels whose
    /// terms arrive per-element rather than per-chunk.
    #[inline(always)]
    pub(crate) fn add_lane(&mut self, i: usize, v: f32) {
        self.0[i & (CANON_LANES - 1)] += v;
    }
    /// Fold a partial tail into the SAME lanes (element k → lane k).
    #[allow(dead_code)]
    #[inline(always)]
    pub(crate) fn add_tail(&mut self, t: &[f32]) {
        for (l, &v) in self.0.iter_mut().zip(t.iter()) {
            *l += v;
        }
    }
    /// The fixed pairwise reduction. `f32` adds in the written order, then one
    /// exact `f64` widening.
    #[inline(always)]
    pub(crate) fn reduce(self) -> f64 {
        era2_reduce8(self.0)
    }
}

/// THE REV4 CANON (rev4canon): 8 f64 virtual lanes — same lane mapping and
/// the same adjacent-pairwise tree shape as [`LanesF32`], but every partial
/// is f64, so per-term error is only the term's own f32 rounding. FEATACC's
/// recommendation (`c64`); under `oracle` also a measurement arm.
///
/// 64-byte aligned for the same reason as [`Lanes16F64`].
#[derive(Debug, Clone, Copy, Default)]
#[repr(align(64))]
pub(crate) struct LanesF64(pub [f64; CANON_LANES]);

/// dead_code until chunked-pool kernels land; see LanesF32::add_chunk.
#[allow(dead_code)]
impl LanesF64 {
    #[inline(always)]
    pub(crate) fn zero() -> Self {
        Self([0.0; 8])
    }
    #[inline(always)]
    pub(crate) fn add_chunk(&mut self, c: &[f32; 8]) {
        for (l, &v) in self.0.iter_mut().zip(c.iter()) {
            *l += v as f64;
        }
    }
    #[inline(always)]
    pub(crate) fn add_chunk_f64(&mut self, c: &[f64; 8]) {
        for (l, &v) in self.0.iter_mut().zip(c.iter()) {
            *l += v;
        }
    }
    #[inline(always)]
    pub(crate) fn add_lane(&mut self, i: usize, v: f64) {
        self.0[i & (CANON_LANES - 1)] += v;
    }
    #[inline(always)]
    pub(crate) fn add_tail(&mut self, t: &[f32]) {
        for (l, &v) in self.0.iter_mut().zip(t.iter()) {
            *l += v as f64;
        }
    }
    #[inline(always)]
    pub(crate) fn add_tail_f64(&mut self, t: &[f64]) {
        for (l, &v) in self.0.iter_mut().zip(t.iter()) {
            *l += v;
        }
    }
    /// The same adjacent-pairwise tree as `era2_reduce8`, in f64.
    #[inline(always)]
    pub(crate) fn reduce(self) -> f64 {
        (self.0[0] + self.0[1] + (self.0[2] + self.0[3]))
            + ((self.0[4] + self.0[5]) + (self.0[6] + self.0[7]))
    }
}

/// Rev5's sixteen virtual f64 lanes, independent of the hardware width.
///
/// 64-byte aligned: the canonical kernels add whole 8-lane chunks (`chunk8`, 64 bytes) every few pixels, and
/// at the default 8-byte alignment those accesses split cache lines whenever the caller's frame puts the
/// array off a line. Measured (benchmarks/byconstruction_2026-10-22.md): the Rev5 walk at 2048², one thread,
/// ran 9.3% faster with this alignment, and without it an unrelated call-site change cost 3.5% in the
/// unchanged dense kernel. Values are unchanged; this is memory alignment only.
#[derive(Debug, Clone, Copy, Default)]
#[repr(align(64))]
pub(crate) struct Lanes16F64(pub [f64; 16]);

impl Pool for Lanes16F64 {
    const LANES: usize = 16;
    #[inline(always)]
    fn zero() -> Self {
        Self([0.0; 16])
    }
    #[inline(always)]
    fn add(&mut self, lane: usize, v: f32) {
        self.0[lane & 15] += v as f64;
    }
    #[cfg(feature = "oracle")]
    #[inline(always)]
    fn add64(&mut self, lane: usize, v: f64) {
        self.0[lane & 15] += v;
    }
    #[inline(always)]
    fn fin(self) -> f64 {
        let mut a = self.0;
        for width in [8, 4, 2, 1] {
            for i in 0..width {
                a[i] = a[2 * i] + a[2 * i + 1];
            }
        }
        a[0]
    }
}

/// Revision 7's sixteen f32 row lanes (the `rowlanes` era,
/// [`crate::feature_defs::FormulaRevision::Rev7`]).
///
/// A pool's terms for the columns of a row's complete eight-column chunks
/// (`x < 8·⌊width/8⌋`) add here, lane `x & 15`, ascending `x`;
/// [`Self::fold_into`] then adds each lane to the matching f64 lane of the
/// pool's [`Lanes16F64`] and clears it, after every column `x` with
/// `(x + 1) % ROW_LANES_FOLD == 0` and at the end of the chunks. The row's
/// tail columns add to the f64 lanes directly. One f32 add per term replaces a
/// conversion and an f64 add, and no f32 lane takes more than 256 terms
/// between folds. The f64 lanes live for a dense band of 32 rows, or for a
/// fused-SSIM or gradient call, and are finished once there.
#[derive(Debug, Clone, Copy, Default)]
#[repr(align(64))]
pub(crate) struct Lanes16F32(pub [f32; 16]);

/// Columns between two folds of [`Lanes16F32`] into the f64 lanes.
pub(crate) const ROW_LANES_FOLD: usize = 4096;

impl Lanes16F32 {
    /// Lanes `offset..offset + 8` (`offset` is 0 or 8; indexed by its bit 3,
    /// so the access needs no range check).
    #[inline(always)]
    pub(crate) fn chunk8(&mut self, offset: usize) -> &mut [f32; 8] {
        debug_assert!(offset == 0 || offset == 8);
        &mut self.0.as_chunks_mut::<8>().0[(offset >> 3) & 1]
    }

    /// Add every lane to `pool`'s matching lane, keeping the lanes (the
    /// first half of a difference pair's fold, see the gradient kernel).
    #[inline(always)]
    pub(crate) fn add_into<P: Pool>(&self, pool: &mut P) {
        for l in 0..16 {
            pool.add(l, self.0[l]);
        }
    }

    /// Add every lane to `pool`'s matching lane (for [`Lanes16F64`],
    /// `lane += v as f64`), then clear.
    #[inline(always)]
    pub(crate) fn fold_into<P: Pool>(&mut self, pool: &mut P) {
        for l in 0..16 {
            pool.add(l, self.0[l]);
        }
        self.0 = [0.0; 16];
    }
}

impl Pool for Lanes16F32 {
    const LANES: usize = 16;
    #[inline(always)]
    fn zero() -> Self {
        Self([0.0; 16])
    }
    #[inline(always)]
    fn add(&mut self, lane: usize, v: f32) {
        self.0[lane & 15] += v;
    }
    /// Row lanes are folded into [`Lanes16F64`], never finished themselves;
    /// this reduces them by its tree for completeness.
    #[inline(always)]
    fn fin(self) -> f64 {
        Lanes16F64(self.0.map(f64::from)).fin()
    }
}

/// Fixed f64 lane storage used by the existing vector kernels.
pub(crate) trait Pool64: Pool {
    fn chunk8(&mut self, offset: usize) -> &mut [f64; 8];
}
impl Pool64 for LanesF64 {
    #[inline(always)]
    fn chunk8(&mut self, _offset: usize) -> &mut [f64; 8] {
        &mut self.0
    }
}
impl Pool64 for Lanes16F64 {
    /// Lanes `offset..offset + 8` (`offset` is 0 or 8; indexed by its bit 3,
    /// so the access needs no range check).
    #[inline(always)]
    fn chunk8(&mut self, offset: usize) -> &mut [f64; 8] {
        debug_assert!(offset == 0 || offset == 8);
        &mut self.0.as_chunks_mut::<8>().0[(offset >> 3) & 1]
    }
}

/// Candidate (f): Neumaier-compensated f64 running sum — the compensated form
/// already proven in `feature_v2::oracle` (`Neumaier`), reused so the lane's
/// compensation semantics have exactly one definition. Measurement only (the
/// candidate and the exact-oracle bodies).
#[cfg(feature = "oracle")]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Neum64 {
    sum: f64,
    comp: f64,
}

#[cfg(feature = "oracle")]
impl Neum64 {
    #[inline(always)]
    pub(crate) fn zero() -> Self {
        Self {
            sum: 0.0,
            comp: 0.0,
        }
    }
    #[inline(always)]
    pub(crate) fn add(&mut self, v: f64) {
        let t = self.sum + v;
        if self.sum.abs() >= v.abs() {
            self.comp += (self.sum - t) + v;
        } else {
            self.comp += (v - t) + self.sum;
        }
        self.sum = t;
    }
    #[inline(always)]
    pub(crate) fn reduce(self) -> f64 {
        self.sum + self.comp
    }
}

// ============================================================================
// The `Pool` abstraction — one body drives all three candidate accumulators
// ============================================================================

/// One canonical accumulator slot in a kernel body.
///
/// `add(lane, v)` puts term `v` into canonical lane `lane` (the term's
/// position mod 8 — kernels pass the column index). `fin` applies the fixed
/// reduction and returns f64. The three impls are the brief's measured
/// candidates: (d) f32 virtual lanes + fixed pairwise tree, (e) f64 virtual
/// lanes + the same tree, (f) sequential Neumaier compensation.
pub(crate) trait Pool: Copy {
    const LANES: usize = 8;
    fn zero() -> Self;
    fn add(&mut self, lane: usize, v: f32);
    /// f64 term form — used when the term was itself computed in f64
    /// (the measurement accumulators and the exact bodies).
    #[cfg(feature = "oracle")]
    fn add64(&mut self, lane: usize, v: f64) {
        self.add(lane, v as f32);
    }
    fn fin(self) -> f64;
}

#[cfg(feature = "oracle")]
impl Pool for LanesF32 {
    #[inline(always)]
    fn zero() -> Self {
        Self::zero()
    }
    #[inline(always)]
    fn add(&mut self, lane: usize, v: f32) {
        self.add_lane(lane, v);
    }
    #[inline(always)]
    fn fin(self) -> f64 {
        self.reduce()
    }
}

impl Pool for LanesF64 {
    #[inline(always)]
    fn zero() -> Self {
        Self::zero()
    }
    #[inline(always)]
    fn add(&mut self, lane: usize, v: f32) {
        self.add_lane(lane, v as f64);
    }
    /// Exact-width add — the f64-native terms of the canon feed through
    /// [`SumVar::add64`], which calls `add_lane` directly so this trait
    /// override stays oracle-only.
    #[cfg(feature = "oracle")]
    #[inline(always)]
    fn add64(&mut self, lane: usize, v: f64) {
        self.add_lane(lane, v);
    }
    #[inline(always)]
    fn fin(self) -> f64 {
        self.reduce()
    }
}

#[cfg(feature = "oracle")]
impl Pool for Neum64 {
    #[inline(always)]
    fn zero() -> Self {
        Self::zero()
    }
    #[inline(always)]
    fn add(&mut self, _lane: usize, v: f32) {
        Neum64::add(self, v as f64);
    }
    #[inline(always)]
    fn add64(&mut self, _lane: usize, v: f64) {
        Neum64::add(self, v);
    }
    #[inline(always)]
    fn fin(self) -> f64 {
        self.reduce()
    }
}

/// Sequential f64 accumulation as a `Pool` — the production shape of the
/// pools). `add(lane, v)` ignores `lane` and adds in arrival order, so a
/// `Pool`-generic kernel body instantiated at `f64` replays the shipped
/// sequence exactly.
impl Pool for f64 {
    #[inline(always)]
    fn zero() -> Self {
        0.0
    }
    #[inline(always)]
    fn add(&mut self, _lane: usize, v: f32) {
        *self += v as f64;
    }
    #[cfg(feature = "oracle")]
    #[inline(always)]
    fn add64(&mut self, _lane: usize, v: f64) {
        *self += v;
    }
    #[inline(always)]
    fn fin(self) -> f64 {
        self
    }
}

// ============================================================================
// `SumVar` / `WelfordVar` — runtime-mode accumulators for kernels whose
// production state is scalar f64 (a `Pool`-generic body would put a type
// parameter on structs like `LevelSums` that live inside long-lived,
// non-generic accumulator objects).
// ============================================================================

/// One accumulator slot whose arithmetic is chosen by [`measurement_mode`].
///
/// `Seq` is the shipped form (a plain `f64 += ` per term). Which candidate
/// `c32`/`c64`/`neum` maps to depends on the ELEMENTS the kernel sums —
/// [`SumVar::f32_elems`] for terms that are f32-representable (f32-computed
/// or f32-stored: an `as f32` narrowing of `f64::from(v)` is exact), and
/// [`SumVar::f64_elems`] for f64-native terms — for those, `c32` keeps the
/// sequential form since f32 lanes would change the ELEMENT precision, not
/// just the accumulation order (that would conflate the two axes the lane
/// exists to separate).
#[derive(Debug, Clone, Copy)]
pub(crate) enum SumVar {
    /// Sequential f64 — production accumulation.
    Seq(f64),
    /// 8 f32 lanes + [`era2_reduce8`] — the superseded `c32` candidate,
    /// oracle-only (rev4canon).
    #[cfg(feature = "oracle")]
    L32(LanesF32),
    /// 8 f64 lanes + the f64 pairwise tree — THE REV4 CANON (`c64`,
    /// rev4canon); under `oracle` also the `c64` measurement arm.
    L64(LanesF64),
    /// Sequential Neumaier compensation — candidate (f) and the oracle arm.
    #[cfg(feature = "oracle")]
    Neum(Neum64),
}

impl SumVar {
    /// Variant for kernels whose summed terms are f32-representable —
    /// the `c32` arm; under the `c64` canon f32-element pools ride the same
    /// `L64` lanes as f64 ones, so product code only needs `f64_elems`.
    #[cfg(feature = "oracle")]
    pub(crate) fn f32_elems(mode: Mode) -> Self {
        match mode {
            Mode::Off => Self::Seq(0.0),
            #[cfg(feature = "oracle")]
            Mode::Canon32 => Self::L32(LanesF32::zero()),
            Mode::Canon64 => Self::L64(LanesF64::zero()),
            #[cfg(feature = "oracle")]
            Mode::CanonNeum | Mode::Exact => Self::Neum(Neum64::zero()),
        }
    }

    /// Variant for kernels whose summed terms are f64-native — `c32` keeps
    /// `Seq` (production) so the f32-lane column never masks element error
    /// as accumulation error; `c64` takes the f64 lanes (the Rev4 canon).
    pub(crate) fn f64_elems(mode: Mode) -> Self {
        match mode {
            Mode::Off => Self::Seq(0.0),
            #[cfg(feature = "oracle")]
            Mode::Canon32 => Self::Seq(0.0),
            Mode::Canon64 => Self::L64(LanesF64::zero()),
            #[cfg(feature = "oracle")]
            Mode::CanonNeum | Mode::Exact => Self::Neum(Neum64::zero()),
        }
    }

    /// Constructor used at walk-setup — resolves the computation's own
    /// arithmetic via [`compute_mode`]: the `ZENSIM_FEATCANON` override
    /// under `oracle`, else [`Mode::Canon64`] at Rev4 and `Seq` below it.
    /// See [`compute_mode`] for why the process revision is the
    /// computation's here.
    #[cfg(feature = "oracle")]
    #[inline]
    pub(crate) fn f32_elems_meas() -> Self {
        Self::f32_elems(compute_mode())
    }
    #[inline]
    pub(crate) fn f64_elems_meas() -> Self {
        Self::f64_elems(compute_mode())
    }

    /// Accumulate one f64 term — `Seq`/`L64`/`Neum` keep it exact; `L32`
    /// narrows to f32 first (the caller-chosen variant encodes whether that
    /// narrowing is value-exact — it is whenever the term is f32-stored).
    #[inline(always)]
    pub(crate) fn add64(&mut self, lane: usize, v: f64) {
        match self {
            Self::Seq(s) => *s += v,
            #[cfg(feature = "oracle")]
            Self::L32(p) => Pool::add(p, lane, v as f32),
            Self::L64(p) => p.add_lane(lane, v),
            #[cfg(feature = "oracle")]
            Self::Neum(p) => Pool::add64(p, lane, v),
        }
    }

    /// The slot's running value (fold-time readout).
    #[inline]
    pub(crate) fn fin(&self) -> f64 {
        match self {
            Self::Seq(s) => *s,
            #[cfg(feature = "oracle")]
            Self::L32(p) => (*p).fin(),
            Self::L64(p) => (*p).fin(),
            #[cfg(feature = "oracle")]
            Self::Neum(p) => (*p).fin(),
        }
    }

    /// Fold a same-mode sibling's accumulated value into `self` —
    /// lane-aligned for the lane variants (the per-row cell → strip cell
    /// merge in the var-typed kernels).
    pub(crate) fn merge_from(&mut self, o: &Self) {
        match (self, o) {
            (Self::Seq(a), Self::Seq(b)) => *a += *b,
            #[cfg(feature = "oracle")]
            (Self::L32(a), Self::L32(b)) => {
                for j in 0..CANON_LANES {
                    a.0[j] += b.0[j];
                }
            }
            (Self::L64(a), Self::L64(b)) => {
                for j in 0..CANON_LANES {
                    a.0[j] += b.0[j];
                }
            }
            #[cfg(feature = "oracle")]
            (Self::Neum(a), Self::Neum(b)) => Neum64::add(a, b.reduce()),
            _ => panic!("SumVar::merge_from across modes"),
        }
    }
}

/// Welford `(n, mean, m2)` state whose update arithmetic follows the
/// measurement mode — for `mapdev` (row Welford + Chan merge) and
/// `gmsbank`'s per-cell deviation state.
///
/// There is no meaningful f32-Welford-lane candidate — Welford's value is
/// precisely that it is already a stable f64 sequential update — so
/// `c32` maps to [`WelfordVar::Seq`], i.e. the c32 column for the Welford
/// kernels IS the production accumulation. `c64` gives the 8-lane form
/// (each lane an independent Welford over the `x ≡ lane (mod 8)` substream,
/// pairwise Chan-merged), `neum`/`exact` the compensated form.
///
/// rev4canon: `Lanes` is the Rev4 canon; `Neum` stays measurement-only.
#[derive(Clone, Copy)]
pub(crate) enum WelfordVar {
    /// Production: sequential Welford in f64.
    Seq(WelfordCell),
    /// 8 independent Welford lanes, pairwise-merged at [`Self::stats`].
    Lanes([WelfordCell; 8]),
    /// Neumaier-compensated `mean`/`m2` updates.
    #[cfg(feature = "oracle")]
    Neum { n: u64, mean: Neum64, m2: Neum64 },
}

/// Plain f64 Welford triple — the production cell shape.
#[derive(Clone, Copy, Default)]
pub(crate) struct WelfordCell {
    pub(crate) n: u64,
    pub(crate) mean: f64,
    pub(crate) m2: f64,
}

impl WelfordCell {
    /// One sample. `recip_n1` is `1.0 / (n + 1)` — precomputed by callers
    /// that fold a whole row (the mapdev idiom); `0` here means "compute".
    #[inline(always)]
    pub(crate) fn push(&mut self, x: f64) {
        self.n += 1;
        let change = x - self.mean;
        self.mean += change / self.n as f64;
        self.m2 += change * (x - self.mean);
    }

    /// Chan's parallel merge.
    pub(crate) fn merge(&mut self, o: &Self) {
        if o.n == 0 {
            return;
        }
        if self.n == 0 {
            *self = *o;
            return;
        }
        let total = self.n + o.n;
        let shift = o.mean - self.mean;
        self.m2 += o.m2 + shift * shift * (self.n as f64 * o.n as f64 / total as f64);
        self.mean += shift * (o.n as f64 / total as f64);
        self.n = total;
    }
}

impl WelfordVar {
    pub(crate) fn for_mode(mode: Mode) -> Self {
        match mode {
            Mode::Off => Self::Seq(WelfordCell::default()),
            #[cfg(feature = "oracle")]
            Mode::Canon32 => Self::Seq(WelfordCell::default()),
            Mode::Canon64 => Self::Lanes([WelfordCell::default(); 8]),
            #[cfg(feature = "oracle")]
            Mode::CanonNeum | Mode::Exact => Self::Neum {
                n: 0,
                mean: Neum64::zero(),
                m2: Neum64::zero(),
            },
        }
    }

    /// One sample into canonical lane `lane` (`x` position mod 8).
    #[cfg_attr(not(feature = "oracle"), allow(unused_variables))]
    #[inline]
    pub(crate) fn push(&mut self, lane: usize, x: f64) {
        match self {
            Self::Seq(c) => c.push(x),
            Self::Lanes(lanes) => lanes[lane & 7].push(x),
            #[cfg(feature = "oracle")]
            Self::Neum { n, mean, m2 } => {
                *n += 1;
                let change = x - mean.reduce();
                mean.add(change / *n as f64);
                m2.add(change * (x - mean.reduce()));
            }
        }
    }

    /// `(n, mean, m2)` — merging the lanes pairwise first.
    #[inline]
    pub(crate) fn stats(&self) -> WelfordCell {
        match self {
            Self::Seq(c) => *c,
            Self::Lanes(lanes) => {
                // The same adjacent-pairwise tree as `era2_reduce8`.
                let m = |a: WelfordCell, b: WelfordCell| {
                    let mut a = a;
                    a.merge(&b);
                    a
                };
                m(
                    m(m(lanes[0], lanes[1]), m(lanes[2], lanes[3])),
                    m(m(lanes[4], lanes[5]), m(lanes[6], lanes[7])),
                )
            }
            #[cfg(feature = "oracle")]
            Self::Neum { n, mean, m2 } => WelfordCell {
                n: *n,
                mean: mean.reduce(),
                m2: m2.reduce(),
            },
        }
    }

    /// Fold a same-mode sibling's whole state into `self` — lane-aligned
    /// for `Lanes`, sequential for `Seq`/`Neum` (the per-row → strip cell
    /// merge in the var-typed kernels).
    pub(crate) fn merge_var(&mut self, o: &Self) {
        match (self, o) {
            (Self::Seq(a), Self::Seq(b)) => a.merge(b),
            (Self::Lanes(a), Self::Lanes(b)) => {
                for j in 0..CANON_LANES {
                    a[j].merge(&b[j]);
                }
            }
            #[cfg(feature = "oracle")]
            (Self::Neum { n, mean, m2 }, Self::Neum { .. }) => {
                let mut a = WelfordCell {
                    n: *n,
                    mean: mean.reduce(),
                    m2: m2.reduce(),
                };
                a.merge(&o.stats());
                *n = a.n;
                *mean = Neum64::zero();
                Neum64::add(mean, a.mean);
                *m2 = Neum64::zero();
                Neum64::add(m2, a.m2);
            }
            _ => panic!("WelfordVar::merge_var across modes"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// featcanon D1/D6: the mode is a function of the computation's revision
    /// alone — the process revision does not enter it — and a product build
    /// (no `oracle`) knows exactly two arithmetics.
    #[test]
    fn mode_is_a_function_of_the_computation_revision() {
        #[cfg(feature = "oracle")]
        assert!(
            std::env::var_os("ZENSIM_FEATCANON").is_none(),
            "this gate reads the unoverridden mapping; unset ZENSIM_FEATCANON"
        );
        for rev in [
            FormulaRevision::Rev1,
            FormulaRevision::Rev2,
            FormulaRevision::Rev3,
        ] {
            assert_eq!(mode(rev), Mode::Off, "{rev:?}");
        }
        assert_eq!(mode(FormulaRevision::Rev4), Mode::Canon64);
        // Same answers whatever ZENSIM_FORMULA_REV this process runs at.
        let _ = crate::ssim_form::active_revision();
        #[cfg(not(feature = "oracle"))]
        {
            // Exhaustive in a product build: adding a third arithmetic
            // without the `oracle` gate fails to compile here.
            let _: fn(Mode) = |m| match m {
                Mode::Off | Mode::Canon64 => {}
            };
        }
    }
}
