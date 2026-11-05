//! **Diagnostic-only** per-phase timing for the streaming fold walk.
//!
//! The fold-MT lane's first instruction is "profile the parallel walk, don't
//! guess". Instruction-count profiles (callgrind) answer a *serial* question —
//! they cannot see a thread idling at a barrier. This module answers the
//! parallel one directly: for each phase of the strip loop it records both the
//! **wall** span (which is the critical path) and the summed **busy** time of
//! the tasks inside it, so `busy / (wall × threads)` is a measured occupancy
//! rather than an inferred one.
//!
//! Nothing here can change a feature byte: every hook is a timestamp and an
//! atomic add, and the recorded values are never read by any kernel.
//!
//! # Enabling
//!
//! `ZENSIM_FOLD_TIMING=<N>` — dump a report to stderr and reset every `N`
//! completed walks (`ZENSIM_FOLD_TIMING=1` dumps per walk). Unset or `0`
//! disables, and then every hook is one relaxed atomic load of an already-
//! resolved `OnceLock` plus a predictable branch.
//!
//! # Reading the report
//!
//! | column | meaning |
//! |---|---|
//! | `wall` | summed wall time of that phase across every strip — the phase's share of the **critical path** |
//! | `busy` | summed per-task time inside that phase |
//! | `occ` | `busy / (wall × RAYON_NUM_THREADS)` — 1.00 means every thread was busy for the whole phase |
//!
//! A phase with `occ` near `1/threads` is running at degree 1 (serial); a
//! phase whose `wall` is large and whose `occ` is small is where the threads
//! are idling, which is the number this lane exists to move.

use core::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

/// Work events beside phase timings; zero overhead beyond the existing enable gate.
#[derive(Clone, Copy)]
pub(crate) enum Work {
    VerticalPlane,
    ActivityChain,
    PeakBand,
    Scale0XbCell,
    ScratchZeroElements,
    Scale0XbStoredRows,
    V2Cell,
}
static WORK: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];
#[inline]
pub(crate) fn work(kind: Work, amount: usize) {
    if on() {
        WORK[kind as usize].fetch_add(amount as u64, Ordering::Relaxed);
    }
}

/// One accumulator slot. Indices are `(phase, scale)`; see [`Phase`].
const N_PHASE: usize = 23;
const N_SCALE: usize = 8;

// Inline-const array repeat, so each element is its own `AtomicU64::new(0)`
// evaluation rather than N copies of one named const (which is what
// `clippy::declare_interior_mutable_const` exists to catch).
static NANOS: [AtomicU64; N_PHASE * N_SCALE] = [const { AtomicU64::new(0) }; N_PHASE * N_SCALE];
static COUNTS: [AtomicU64; N_PHASE * N_SCALE] = [const { AtomicU64::new(0) }; N_PHASE * N_SCALE];
static WALKS: AtomicU64 = AtomicU64::new(0);

/// Phases of the strip loop, in the order they execute.
// `PhaseAV2Planes` / `PhaseAAppendPlanes` trip `enum_variant_names` on the
// enum's own name, but the `PhaseA` prefix names the WALK's phase A (the
// plane-production fan-out) — the shared prefix is the thing being measured,
// not a restatement of the type. Renaming them would break the correspondence
// with `stream_phase_a`.
#[allow(clippy::enum_variant_names)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub(crate) enum Phase {
    /// `producer.next_strip()` — decode + XYB + the downscale cascade. Serial
    /// by construction today; this is the row the "parallelise the producer"
    /// candidate is judged on.
    Producer = 0,
    /// Wall of the phase-A fan-out (3-way over channels).
    AWall = 1,
    /// Summed per-channel busy time inside phase A.
    ABusy = 2,
    /// Wall of the consuming fan-out: phase B alone on the SPLIT arm, the
    /// whole fused (A then B) span on the fused arm.
    BWall = 3,
    /// Summed per-channel busy time inside phase B.
    BBusy = 4,
    /// Summed per-BAND busy time inside `fold_v1_basic_bands`.
    BandBusy = 5,
    /// Serial work between the two fan-outs (mean-offset side channel,
    /// `refy` binding, retention copies).
    Between = 6,
    /// Whole-walk wall, recorded at scale slot 0.
    Walk = 7,
    /// Wall of the strip-local `fold_v1_basic_bands` call (one per channel;
    /// summed across channels, so it exceeds `BWall` when parallel).
    FoldWall = 8,
    /// Wall of `fused_blur_h_ssim` (one per channel, summed).
    BlurHWall = 9,
    /// Scale-0 XYB conversion inside the producer.
    ProdConvert = 10,
    /// Downscale cascade inside the producer.
    ProdDownscale = 11,
    /// Summed per-BAND busy time inside `fused_blur_h_ssim_banded`.
    BlurBandBusy = 12,
    /// The `mean_offset` side channel's per-strip pass.
    MeanOffset = 13,
    // ---- v2-block decomposition (v2-cost lane, 2026-08-31) ----
    // Summed per-(strip, channel) busy time of each v2-era kernel inside
    // phase B, plus the phase-A work that exists ONLY to feed them. These
    // rows are what makes the `v1_only`-vs-944 delta attributable to a
    // kernel at the PRODUCTION tier — callgrind cannot profile `v4x`
    // (valgrind masks AVX-512 out of CPUID), and `v4x` is exactly where
    // the dense kernel's `POOL_SIMD` path lives.
    /// `dense_block_kernel` — the 22-of-29 v2 slots per (ch, scale).
    DenseKernel = 14,
    /// `gradient_block_kernel` — GMS/ringing/banding/grad sums (+BANDVIS).
    GradKernel = 15,
    /// `append_block_kernel` — the f720+ append-204 block.
    AppendKernel = 16,
    /// `csfw_block_kernel` — the f944+ CSFW block (Y only, default off).
    CsfwKernel = 17,
    /// `blockiness_sparse_strip_wide` — the sparse lattice pass.
    BlockKernel = 18,
    /// Phase A's v2-ONLY plane chain: four `box_blur_v_from_copy` sweeps
    /// (mu1/mu2/ssq/s12) plus the activity chain (`abs_diff_into` +
    /// `box_blur_1pass_into`). Skipped entirely by a `v1_only` request.
    PhaseAV2Planes = 19,
    /// Phase A's `bs2` chain (`square_into` + H + V blur), which exists
    /// only for the append block's sigma split, plus the optional
    /// BANDVIS dst-activity twin.
    PhaseAAppendPlanes = 20,
    /// REV4 kernels — `gridblk_strip_wide` inside phase B plus the
    /// `dst_y_edge_mask_strip` bleed-mask build (serial, pre-fan-out).
    /// Default-off families; zero counts when every rev4 flag is off.
    Rev4Kernel = 22,
}

#[inline(always)]
fn interval() -> u64 {
    static IV: OnceLock<u64> = OnceLock::new();
    *IV.get_or_init(|| {
        std::env::var("ZENSIM_FOLD_TIMING")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(0)
    })
}

/// True when timing is enabled. One resolved-`OnceLock` load when off.
#[inline(always)]
pub(crate) fn on() -> bool {
    interval() != 0
}

/// Start a span. Returns `None` (and does no work) when timing is off.
#[inline(always)]
pub(crate) fn start() -> Option<Instant> {
    if on() { Some(Instant::now()) } else { None }
}

/// Close a span opened by [`start`] into `(phase, scale)`.
#[inline(always)]
pub(crate) fn stop(t: Option<Instant>, phase: Phase, scale: usize) {
    if let Some(t0) = t {
        record(phase, scale, t0.elapsed().as_nanos() as u64);
    }
}

#[inline(always)]
fn record(phase: Phase, scale: usize, nanos: u64) {
    let i = (phase as usize) * N_SCALE + scale.min(N_SCALE - 1);
    NANOS[i].fetch_add(nanos, Ordering::Relaxed);
    COUNTS[i].fetch_add(1, Ordering::Relaxed);
}

fn sum(phase: Phase) -> (u64, u64) {
    let base = (phase as usize) * N_SCALE;
    let mut n = 0;
    let mut c = 0;
    for k in 0..N_SCALE {
        n += NANOS[base + k].load(Ordering::Relaxed);
        c += COUNTS[base + k].load(Ordering::Relaxed);
    }
    (n, c)
}

fn per_scale(phase: Phase) -> [u64; 4] {
    let base = (phase as usize) * N_SCALE;
    core::array::from_fn(|k| NANOS[base + k].load(Ordering::Relaxed))
}

/// Called once per completed walk; dumps + resets on the configured interval.
pub(crate) fn walk_done(walk_nanos: u64) {
    if !on() {
        return;
    }
    record(Phase::Walk, 0, walk_nanos);
    let n = WALKS.fetch_add(1, Ordering::Relaxed) + 1;
    if n.is_multiple_of(interval()) {
        dump(n);
        reset();
    }
}

fn threads() -> f64 {
    std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| *v > 0.0)
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1.0, |v| v.get() as f64))
}

fn dump(walks: u64) {
    let counts: [u64; 7] = std::array::from_fn(|i| WORK[i].swap(0, Ordering::Relaxed));
    eprintln!(
        "WORK_CENSUS vertical_planes={} activity_chains={} peak_bands={} scale0_xb_cells={} scratch_zero_elements={} scale0_xb_stored_rows={} v2_cells={}",
        counts[0], counts[1], counts[2], counts[3], counts[4], counts[5], counts[6]
    );

    if std::env::var_os("ZENSIM_FOLD_CENSUS_ASSERT").is_some() {
        assert!(counts[6] > 0, "census must visit actual v2 cells");
        assert_eq!(
            counts[0],
            5 * counts[6],
            "four moment planes + one activity plane per cell"
        );
        assert_eq!(counts[1], counts[6], "one activity chain per cell");
        assert_eq!(
            (counts[2], counts[3], counts[5]),
            (0, 0, 0),
            "unread peaks or scale-zero chroma"
        );
        if walks > 1 {
            assert_eq!(
                counts[4], 0,
                "overwritten scratch must be reused after warmup"
            );
        }
        eprintln!(
            "WORK_CENSUS_GATE {}",
            if walks > 1 { "PASS" } else { "WARMUP" }
        );
    }

    let thr = threads();
    let (walk, _) = sum(Phase::Walk);
    let w = walk.max(1) as f64;
    let ms = |v: u64| v as f64 / 1e6 / walks as f64;
    let pct = |v: u64| 100.0 * v as f64 / w;

    eprintln!(
        "\n=== ZENSIM FOLD TIMING === walks={walks} threads={thr:.0} \
         walk={:.3} ms/walk",
        ms(walk)
    );
    eprintln!(
        "{:<16} {:>10} {:>8} {:>10} {:>7} {:>9}  per-scale ms (0..3)",
        "phase", "ms/walk", "% wall", "busy ms", "occ", "calls"
    );
    let row = |name: &str, wall: Phase, busy: Option<Phase>| {
        let (wn, wc) = sum(wall);
        let (bn, occ) = match busy {
            Some(b) => {
                let (bn, _) = sum(b);
                (bn, bn as f64 / (wn.max(1) as f64 * thr))
            }
            None => (0, f64::NAN),
        };
        let ps = per_scale(wall);
        eprintln!(
            "{:<16} {:>10.3} {:>8.1} {:>10.3} {:>7} {:>9}  {:.2} {:.2} {:.2} {:.2}",
            name,
            ms(wn),
            pct(wn),
            ms(bn),
            if occ.is_nan() {
                "-".to_string()
            } else {
                format!("{occ:.3}")
            },
            wc,
            ms(ps[0]),
            ms(ps[1]),
            ms(ps[2]),
            ms(ps[3]),
        );
    };
    // NOTE on reading this: the FUSED per-channel arm records its whole
    // (A then B) span under `BWall`, so on that arm `phaseA` and
    // `between` are legitimately 0.000 and `consume` is the whole compute.
    // Under the SPLIT arm the two are separate rows. `blur_h` reads 0 as well
    // whenever the self-blur band shape runs — the H blur then happens inside
    // the band, and its time is inside `fold(sum)`.
    row("producer", Phase::Producer, None);
    row("  convert", Phase::ProdConvert, None);
    row("  downscale", Phase::ProdDownscale, None);
    row("phaseA(split)", Phase::AWall, Some(Phase::ABusy));
    row("  blur_h(sum)", Phase::BlurHWall, Some(Phase::BlurBandBusy));
    row("mean_offset", Phase::MeanOffset, None);
    row("between", Phase::Between, None);
    row("consume", Phase::BWall, Some(Phase::BBusy));
    row("  fold(sum)", Phase::FoldWall, Some(Phase::BandBusy));
    // v2-block decomposition (busy sums, not wall spans — these are the
    // per-(strip, channel) kernel bodies inside the phase-B fan-out, so
    // their sum exceeds `consume`'s wall whenever the fan-out is parallel).
    row("  v2:dense", Phase::DenseKernel, None);
    row("  v2:gradient", Phase::GradKernel, None);
    row("  v2:append", Phase::AppendKernel, None);
    row("  v2:csfw", Phase::CsfwKernel, None);
    row("  v2:blockiness", Phase::BlockKernel, None);
    row("  v2:rev4", Phase::Rev4Kernel, None);
    row("  v2:planesA", Phase::PhaseAV2Planes, None);
    row("  v2:planesApp", Phase::PhaseAAppendPlanes, None);
    let (v2sum, _) = {
        let mut n = 0u64;
        let mut c = 0u64;
        for ph in [
            Phase::DenseKernel,
            Phase::GradKernel,
            Phase::AppendKernel,
            Phase::CsfwKernel,
            Phase::BlockKernel,
            Phase::Rev4Kernel,
            Phase::PhaseAV2Planes,
            Phase::PhaseAAppendPlanes,
        ] {
            let (a, b) = sum(ph);
            n += a;
            c += b;
        }
        (n, c)
    };
    eprintln!("  v2:TOTAL(busy)  {:>10.3} {:>8.1}", ms(v2sum), pct(v2sum));
    let (p, _) = sum(Phase::Producer);
    let (a, _) = sum(Phase::AWall);
    let (bt, _) = sum(Phase::Between);
    let (b, _) = sum(Phase::BWall);
    let (mo, _) = sum(Phase::MeanOffset);
    let acct = p + a + bt + b + mo;
    eprintln!(
        "accounted {:.3} ms/walk = {:.1} % of walk; unaccounted {:.3} ms/walk",
        ms(acct),
        pct(acct),
        ms(walk.saturating_sub(acct))
    );
}

fn reset() {
    for s in NANOS.iter().chain(COUNTS.iter()) {
        s.store(0, Ordering::Relaxed);
    }
}
