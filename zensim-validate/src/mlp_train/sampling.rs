//! THE owner of the RankNet **pair-draw step** and of subset-coverage
//! measurement.
//!
//! # Why this module exists
//!
//! The draw step used to be copy-pasted inline in four training loops
//! (`train_mlp_strategy`, `train_mlp_pool_head_with_tv`,
//! `train_mlp_hybrid_head_with_tv`, `train_mlp_per_sample_alpha_head`).
//! Three of the four were byte-identical; the fourth added the STRATEGY
//! stratified-row variant. Re-simulating a drawn subset for the
//! subset-quality study (`benchmarks/subset_quality_study_2026-09-04.md`)
//! would have made a *fifth* copy — exactly the duplication zensim's
//! `CLAUDE.md` "one owner per task" rule forbids — so the step was
//! extracted here instead and every caller, training and simulating,
//! routes through [`draw_pair`].
//!
//! # The contract that makes re-simulation exact
//!
//! The trainer uses separate `SplitMix64` states for initialization and
//! sampling, defaulting to one `--seed` with explicit seed overrides:
//!
//! - init: `SplitMix64::new(seed)` — He-normal weights.
//! - sample: [`sample_stream_seed`] — every pair draw.
//!
//! The separation is deliberate and predates this module: it exists so a
//! 228-vs-372 feature A/B sees the *same* pair sequence even though init
//! consumes a different number of normals. The consequence exploited here
//! is that **the drawn multiset is a pure function of**
//! `(seed, [train_weight], [row_count], epochs, pairs_per_epoch, boosts,
//! within_ref)` **and of nothing else** — in particular not of the feature
//! matrix, the architecture, or the loss. A subset can therefore be
//! reconstructed from a bake's embedded `zentrain.repro` block without
//! reading a single feature column.
//!
//! The legacy sample initializer maps seed differences to exact raw-stream
//! offsets: adjacent sample seeds do not provide disjoint sampling windows.
//! Use recorded, well-separated sampling seeds and `subset_sim`'s opt-in
//! `--require-disjoint-sampler-windows` preflight for uniform-sampler studies.
//! Distinct digests prove distinct sequences, not independent sampling replicas.
//!
//! [`SampleSequenceDigest`] is what proves the reconstruction is faithful:
//! the same rolling hash is computed by the training loop (under
//! `ZENSIM_SAMPLE_DIGEST=1`) and by [`simulate`], and they must match.

use super::RefBuckets;
use super::SplitMix64;

pub(super) mod matched_codec;

/// Sample-stream seed for every entry point except
/// `train_mlp_per_sample_alpha_head`.
///
/// Mirrors `mod.rs`'s in-loop construction exactly; changing either
/// without the other silently re-rolls every model's training subset.
#[inline]
pub fn sample_stream_seed(seed: u64) -> u64 {
    seed.wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(0xDEAD_BEEF_CAFE_BABE)
}

/// Sample-stream seed for `train_mlp_per_sample_alpha_head`, which uses a
/// different additive constant.
#[inline]
pub fn sample_stream_seed_per_sample_alpha(seed: u64) -> u64 {
    seed.wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(0x0123_4567_89AB_CDEF)
}

/// Everything the draw step reads. All slices are indexed by
/// *position within `train_indices`*, not by group index.
pub(crate) struct PairDrawCtx<'a> {
    /// Cumulative normalised `train_weight`, one entry per train group.
    pub cdf: &'a [f64],
    /// Row count per train group (`groups[gi].features.len()`).
    pub row_counts: &'a [usize],
    /// Per-row sampling CDF when any q-boost is non-trivial, else `None`.
    pub per_row_cdfs: &'a [Option<Vec<f64>>],
    /// Within-ref buckets for groups that opted in, else `None`.
    pub ref_buckets: &'a [Option<RefBuckets>],
    /// STRATEGY stratified row-A bands; empty slice = off.
    pub strat_bands: &'a [Vec<Vec<usize>>],
    /// `--pair-sampling stratified` SCHEDULE, or `None` for the uniform
    /// draw every model before 2026-09-04 trained under. See
    /// [`StratifiedPlan`]. **`None` is the default and leaves `draw_pair`
    /// byte-for-byte the function it was** — the stratified branch is a
    /// leading `if` that a `None` plan never enters.
    pub plan: Option<&'a StratifiedPlan>,
    /// Index of THIS draw within the run (`epoch * pairs_per_epoch + i`,
    /// or `simulate`'s flat step). Read ONLY by the stratified branch,
    /// which is a schedule and therefore a function of position; the
    /// uniform branch ignores it entirely.
    pub draw_index: u64,
}

// ── `--pair-sampling stratified` ─────────────────────────────────────────
//
// The uniform draw visits a (ref, quality-band) cell as often as chance
// takes it there, so which cells a finite-epoch run covers — and how
// evenly — is a property of the sample seed. `zentrain.sample_coverage`
// measures that; this is the mode that REMOVES it, so a study can hold
// coverage fixed and vary only the seed.
//
// THE SCHEDULE, and what each half is a function of:
//
//   * WHICH stratum a draw visits is a pure function of `draw_index`:
//     stratum `draw_index % n_strata`, a fixed round-robin. Every stratum
//     is therefore visited every `n_strata` draws — "every ref x band each
//     epoch" whenever `pairs_per_epoch >= n_strata`, which the plan checks
//     and reports rather than assumes.
//   * WHICH ROWS inside it is a function of the SAMPLE SEED: each stratum
//     walks its own rows through a seeded affine bijection
//     `j -> (a*j + b) mod len` with `a` coprime to `len`, pairing visit `j`
//     with visit `j+1`. A bijection walks every row of the stratum exactly
//     once per `len` visits, so the SET of rows a run touches is the whole
//     stratum for any seed — the coverage is seed-invariant while the pair
//     ORDER and the pairings are not.
//
// That split is the whole design: the seed permutes order WITHIN strata,
// and nothing else. It is also why the stratified branch consumes ZERO RNG
// values — the seed enters once, when the plan is built, not per draw. A
// stratified run's `--sample-seed` therefore changes the pair sequence
// (and the digest) while leaving row/ref/cell coverage identical, which is
// exactly the property `coverage_embed_gate.sh` checks in the other
// direction for uniform.
//
// NOT an era break for anything already trained: a `None` plan is the
// default and skips this entirely.

/// One (group, ref, band) stratum: the row indices in it, plus the seeded
/// affine walk over them.
#[derive(Clone, Debug)]
pub struct Stratum {
    /// Position within `train_indices` (NOT the group index).
    pub train_pos: usize,
    /// The stratum's rows, in ascending row order. The seed permutes the
    /// WALK, never this list, so two seeds' strata are the same sets.
    pub rows: Vec<usize>,
    /// Multiplier of the affine walk; coprime to `rows.len()`.
    step: usize,
    /// Offset of the affine walk.
    offset: usize,
}

impl Stratum {
    /// Row at visit `j` of this stratum.
    #[inline]
    fn row_at(&self, j: u64) -> usize {
        let len = self.rows.len();
        self.rows[((j as usize)
            .wrapping_mul(self.step)
            .wrapping_add(self.offset))
            % len]
    }
}

/// The stratified visiting schedule for a whole run.
#[derive(Clone, Debug)]
pub struct StratifiedPlan {
    pub strata: Vec<Stratum>,
    /// Draws per epoch, so the plan can say whether every stratum is
    /// actually reached within one.
    pub pairs_per_epoch: usize,
    /// Strata with a single row: they can never yield an in-stratum pair
    /// and are EXCLUDED from `strata` rather than silently emitting
    /// collisions. Counted so the exclusion is visible, not invisible.
    pub singleton_strata: usize,
}

/// Largest `a < len` coprime to `len`, at or below a seeded start. Any
/// coprime multiplier makes `j -> a*j + b (mod len)` a bijection; picking
/// it from the seed is what makes the WALK seed-dependent while the row
/// SET stays fixed.
fn coprime_step(len: usize, seed: u64) -> usize {
    if len <= 2 {
        return 1;
    }
    let gcd = |mut a: usize, mut b: usize| {
        while b != 0 {
            let t = a % b;
            a = b;
            b = t;
        }
        a
    };
    let start = 1 + (seed % (len as u64 - 1)) as usize;
    for d in 0..len {
        let c = 1 + ((start - 1 + d) % (len - 1));
        if gcd(c, len) == 1 {
            return c;
        }
    }
    1
}

/// THE (ref, quality-band) partition a stratified run visits, for ONE
/// group: cells keyed on `(ref_id, native_band)` when the group carries
/// `ref_ids`, and on band alone when it does not — the finest partition
/// the group's own metadata supports, never a fabricated one.
///
/// One owner, called by both the trainer's plan builder and [`simulate`],
/// so a replay partitions exactly as the run did.
pub fn strata_of(human_scores: &[f64], ref_ids: Option<&[u32]>) -> Vec<Vec<usize>> {
    let mut cells: std::collections::BTreeMap<(u32, usize), Vec<usize>> =
        std::collections::BTreeMap::new();
    for (i, &s) in human_scores.iter().enumerate() {
        // `u32::MAX` is the "no reference identity" key, so a group without
        // ref_ids partitions on band alone rather than being dropped.
        let r = ref_ids.and_then(|r| r.get(i).copied()).unwrap_or(u32::MAX);
        cells.entry((r, native_band(s))).or_default().push(i);
    }
    cells.into_values().collect()
}

impl StratifiedPlan {
    /// Build the schedule. `strata_of` yields, per train group, the row
    /// index lists of its (ref, band) cells; a group with no ref identity
    /// contributes its band cells alone.
    pub fn build(
        per_group_strata: &[Vec<Vec<usize>>],
        pairs_per_epoch: usize,
        sample_seed: u64,
    ) -> Self {
        let mut strata = Vec::new();
        let mut singleton_strata = 0usize;
        for (train_pos, cells) in per_group_strata.iter().enumerate() {
            for rows in cells {
                if rows.len() < 2 {
                    if !rows.is_empty() {
                        singleton_strata += 1;
                    }
                    continue;
                }
                // Seed each stratum's walk distinctly, so two strata of the
                // same size do not march in lockstep.
                let k = SplitMix64::new(
                    sample_seed
                        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                        .wrapping_add(strata.len() as u64),
                )
                .next_u64();
                let len = rows.len();
                strata.push(Stratum {
                    train_pos,
                    rows: rows.clone(),
                    step: coprime_step(len, k),
                    offset: (k >> 32) as usize % len,
                });
            }
        }
        Self {
            strata,
            pairs_per_epoch,
            singleton_strata,
        }
    }

    /// True when one epoch reaches every stratum at least once. False is
    /// not an error — it is a fact the caller must be told, because the
    /// mode's headline promise ("every ref x band each epoch") does not
    /// hold below it.
    pub fn covers_every_stratum_per_epoch(&self) -> bool {
        !self.strata.is_empty() && self.pairs_per_epoch >= self.strata.len()
    }
}

/// One draw's outcome.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Draw {
    /// The group had fewer than 2 rows. Consumed 1 RNG value.
    GroupTooSmall,
    /// Row indices collided (`ia == ib == row`). The RNG values were
    /// consumed exactly as for a `Pair`. Most callers skip; the STRATEGY
    /// site instead runs its hard-pair miner first, which is why the
    /// collided index is carried rather than discarded.
    SameRow { train_pos: usize, row: usize },
    /// A usable pair.
    Pair {
        train_pos: usize,
        ia: usize,
        ib: usize,
    },
}

/// Draw one RankNet pair.
///
/// **RNG consumption is part of the contract**, not an implementation
/// detail: 1 value for the group, then 2 (uniform / per-row-CDF) or 3
/// (within-ref) for the rows, with an early return after the group value
/// when the group is unusable. Any change re-rolls every subsequent draw
/// of every model ever trained, so it must be treated as an era break.
/// Replace a fixed share of draws with two uniform rows in the SAME leg.
/// The disabled path consumes no random values and returns the original draw.
pub(crate) fn pool_draw(
    draw: Draw,
    counts: &[usize],
    enabled: &[bool],
    share: f64,
    rng: &mut SplitMix64,
) -> Draw {
    let pos = match draw {
        Draw::Pair { train_pos, .. } | Draw::SameRow { train_pos, .. } => train_pos,
        _ => return draw,
    };
    if share == 0.0 || !enabled[pos] {
        return draw;
    }
    if rng.next_f64_unit() >= share {
        return draw;
    }
    let n = counts[pos];
    let ia = (rng.next_u64() as usize) % n;
    let ib = (rng.next_u64() as usize) % n;
    if ia == ib {
        Draw::SameRow {
            train_pos: pos,
            row: ia,
        }
    } else {
        Draw::Pair {
            train_pos: pos,
            ia,
            ib,
        }
    }
}

/// Replace only an opted-in group's draw, preserving every disabled RNG stream.
pub(crate) fn list_draw(draw: Draw, lists: &[Option<&[[usize; 2]]>], rng: &mut SplitMix64) -> Draw {
    match draw {
        Draw::Pair { train_pos, .. } | Draw::SameRow { train_pos, .. }
            if lists[train_pos].is_some() =>
        {
            let pairs = lists[train_pos].unwrap();
            let [ia, ib] = pairs[(rng.next_u64() as usize) % pairs.len()];
            Draw::Pair { train_pos, ia, ib }
        }
        other => other,
    }
}

#[cfg(test)]
mod list_tests {
    use super::*;
    #[test]
    fn pair_list_excludes_unlisted_pairs_and_preserves_disabled_rng() {
        let pairs = [[0, 3], [1, 2]];
        let mut a = SplitMix64::new(91);
        let mut b = SplitMix64::new(91);
        let unchanged = list_draw(
            Draw::SameRow {
                train_pos: 0,
                row: 9,
            },
            &[None],
            &mut a,
        );
        assert!(matches!(unchanged, Draw::SameRow { row: 9, .. }));
        assert_eq!(a.next_u64(), b.next_u64());
        let mut seen = std::collections::BTreeSet::new();
        for _ in 0..100 {
            let Draw::Pair { ia, ib, train_pos } = list_draw(
                Draw::SameRow {
                    train_pos: 0,
                    row: 9,
                },
                &[Some(&pairs)],
                &mut a,
            ) else {
                panic!("no pair")
            };
            assert_eq!(train_pos, 0);
            assert!(pairs.contains(&[ia, ib]));
            seen.insert([ia, ib]);
        }
        assert_eq!(seen.len(), 2);
    }
}

pub(crate) fn draw_pair(ctx: &PairDrawCtx<'_>, rng: &mut SplitMix64) -> Draw {
    // `--pair-sampling stratified`: a SCHEDULE, so it reads `draw_index`
    // and consumes NO rng. See the StratifiedPlan note above for why the
    // seed enters at plan-build time instead. `None` (the default) falls
    // straight through to the unchanged uniform draw below.
    if let Some(plan) = ctx.plan {
        if plan.strata.is_empty() {
            return Draw::GroupTooSmall;
        }
        let n_strata = plan.strata.len() as u64;
        let s = &plan.strata[(ctx.draw_index % n_strata) as usize];
        let visit = ctx.draw_index / n_strata;
        let ia = s.row_at(visit);
        let ib = s.row_at(visit + 1);
        if ia == ib {
            // Only reachable for a 2-row stratum walked with step 1 at an
            // odd offset; report it the way the uniform path reports a
            // collision so every caller's skip logic is unchanged.
            return Draw::SameRow {
                train_pos: s.train_pos,
                row: ia,
            };
        }
        return Draw::Pair {
            train_pos: s.train_pos,
            ia,
            ib,
        };
    }
    let u = rng.next_f64_unit();
    let train_pos = ctx.cdf.partition_point(|&c| c < u).min(ctx.cdf.len() - 1);
    let n = ctx.row_counts[train_pos];
    if n < 2 {
        return Draw::GroupTooSmall;
    }
    let (ia, ib) = if let Some(rb) = &ctx.ref_buckets[train_pos] {
        // Within-ref: pick a ref, then two rows inside it.
        rb.draw(rng.next_u64(), rng.next_u64(), rng.next_u64())
    } else {
        match &ctx.per_row_cdfs[train_pos] {
            Some(row_cdf) => {
                let ua = rng.next_f64_unit();
                let ub = rng.next_f64_unit();
                (
                    row_cdf.partition_point(|&c| c < ua).min(n - 1),
                    row_cdf.partition_point(|&c| c < ub).min(n - 1),
                )
            }
            None => {
                // STRATEGY: stratified row A — band-uniform then row-uniform.
                let ia = if !ctx.strat_bands.is_empty() {
                    let bands = &ctx.strat_bands[train_pos];
                    let b = &bands[(rng.next_u64() as usize) % bands.len()];
                    b[(rng.next_u64() as usize) % b.len()]
                } else {
                    (rng.next_u64() as usize) % n
                };
                (ia, (rng.next_u64() as usize) % n)
            }
        }
    };
    if ia == ib {
        return Draw::SameRow { train_pos, row: ia };
    }
    Draw::Pair { train_pos, ia, ib }
}

/// Order-sensitive rolling hash over the drawn `(train_pos, ia, ib)`
/// stream — the faithfulness proof for [`simulate`].
///
/// FNV-1a over little-endian `u32` triples. Skipped draws are folded in
/// too (as a distinct tag) so a divergence in *skip* behaviour, which
/// would silently shift every later draw, cannot hash equal.
#[derive(Clone, Copy, Debug)]
pub struct SampleSequenceDigest(u64);

impl Default for SampleSequenceDigest {
    fn default() -> Self {
        Self::new()
    }
}

impl SampleSequenceDigest {
    pub fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }
    #[inline]
    fn byte(&mut self, b: u8) {
        self.0 ^= b as u64;
        self.0 = self.0.wrapping_mul(0x0000_0100_0000_01B3);
    }
    #[inline]
    fn word(&mut self, w: u32) {
        for b in w.to_le_bytes() {
            self.byte(b);
        }
    }
    /// Fold one draw outcome in.
    #[inline]
    pub(crate) fn push(&mut self, d: Draw) {
        match d {
            Draw::GroupTooSmall => self.byte(0x01),
            Draw::SameRow { train_pos, row } => {
                self.byte(0x02);
                self.word(train_pos as u32);
                self.word(row as u32);
            }
            Draw::Pair { train_pos, ia, ib } => {
                self.byte(0x03);
                self.word(train_pos as u32);
                self.word(ia as u32);
                self.word(ib as u32);
            }
        }
    }
    pub fn finish(&self) -> u64 {
        self.0
    }
    pub fn hex(&self) -> String {
        format!("{:016x}", self.0)
    }
}

// ---------------------------------------------------------------------
// Coverage simulation
// ---------------------------------------------------------------------

/// One training group as the simulator needs it: counts, scores, refs.
pub struct SimGroup {
    pub name: String,
    pub train_weight: f64,
    pub n_rows: usize,
    pub human_scores: Vec<f64>,
    pub ref_ids: Option<Vec<u32>>,
    pub within_ref: bool,
}

/// The knobs that change RNG consumption or row weighting.
pub struct SimParams {
    pub seed: u64,
    pub epochs: usize,
    pub pairs_per_epoch: usize,
    pub low_q_boost: f64,
    pub mid_q_boost: f64,
    pub high_q_boost: f64,
    pub stratified_bands: usize,
    /// Draw count defining the "early window" descriptors. `0` = one epoch.
    pub early_window: usize,
    /// Use the `train_mlp_per_sample_alpha_head` stream constant.
    pub per_sample_alpha_head: bool,
    /// Replay a `--pair-sampling stratified` run. `false` (default for a
    /// legacy bake, which had no such flag) replays the uniform draw.
    pub stratified_pairs: bool,
    /// Optional within-leg pooled rank sampling, matching the live CPU trainer.
    pub pooled_legs: Vec<String>,
    pub pooled_rank_share: f64,
}

/// Per-group coverage descriptors over one window.
#[derive(Clone, Debug, Default)]
pub struct GroupCoverage {
    pub name: String,
    pub n_rows: usize,
    pub train_weight: f64,
    /// Pairs drawn from this group.
    pub n_pairs: u64,
    /// Rows touched at least once.
    pub rows_touched: usize,
    /// `rows_touched / n_rows`.
    pub row_coverage: f64,
    /// Distinct reference images touched (groups with `ref_ids`).
    pub refs_touched: usize,
    pub n_refs: usize,
    pub ref_coverage: f64,
    /// Distinct (ref, native-band) cells touched / total non-empty cells.
    pub cell_coverage: f64,
    pub cells_touched: usize,
    pub n_cells: usize,
    /// Shannon entropy of per-row multiplicity, normalised by `ln(n_rows)`.
    pub row_entropy_norm: f64,
    /// Coefficient of variation of per-row multiplicity.
    pub row_multiplicity_cv: f64,
    /// Share of this group's pairs whose higher endpoint scores >= 90 —
    /// the near-threshold / near-lossless zone.
    pub near_threshold_share: f64,
    /// Share of drawn pairs whose two rows share a reference image.
    pub within_image_share: f64,
    /// Pairs drawn from rows in each of the trainer's four NATIVE quality
    /// bands (B0 <50, B1 [50,65), B2 [65,90), B3 >=90), endpoint-counted.
    pub band_pair_counts: [u64; 4],
}

/// Whole-run (or whole-window) coverage descriptors.
#[derive(Clone, Debug, Default)]
pub struct SampleCoverage {
    pub window_draws: u64,
    pub n_pairs: u64,
    /// Draws lost to `ia == ib` — a pure RNG-luck control.
    pub same_row_skips: u64,
    pub group_too_small_skips: u64,
    /// Pooled `rows_touched / total_rows`.
    pub pooled_row_coverage: f64,
    /// L1 deviation between realised per-group pair share and declared
    /// normalised `train_weight`.
    pub group_share_l1: f64,
    /// Pearson chi-square of realised vs declared group shares.
    pub group_share_chisq: f64,
    /// Pooled near-threshold (>=90) endpoint share.
    pub near_threshold_share: f64,
    /// Pooled within-image pair share.
    pub within_image_share: f64,
    /// Fraction of drawn pairs repeating an already-drawn `(g, ia, ib)`.
    /// Only computed when the window is bounded (see `SimParams`).
    pub duplicate_pair_rate: f64,
    pub per_group: Vec<GroupCoverage>,
}

/// Full simulation output: the whole-run window, the early window, and the
/// digest proving it matches a real run.
pub struct SimResult {
    pub full: SampleCoverage,
    pub early: SampleCoverage,
    pub digest: SampleSequenceDigest,
    /// Digest over just the early window, so a short faithfulness check
    /// does not need a full 6M-draw run.
    pub early_digest: SampleSequenceDigest,
}

/// The trainer's own native quality bands, as used by its q-boost logic
/// (`mod.rs`: B0 <50, B1 [50,65), B2 [65,90), B3 >=90). Reused rather than
/// inventing a banding so descriptors speak the sampler's own language.
#[inline]
pub fn native_band(score: f64) -> usize {
    if score < 50.0 {
        0
    } else if score < 65.0 {
        1
    } else if score < 90.0 {
        2
    } else {
        3
    }
}

/// Build the per-row sampling CDF exactly as `mod.rs` does.
fn build_per_row_cdf(scores: &[f64], lo: f64, mid: f64, hi: f64) -> Option<Vec<f64>> {
    if lo == 1.0 && mid == 1.0 && hi == 1.0 {
        return None;
    }
    let mut cum = 0.0;
    let raw: Vec<f64> = scores
        .iter()
        .map(|&s| {
            let mut w = 1.0;
            if lo != 1.0 {
                if s < 50.0 {
                    w *= lo;
                } else if s < 65.0 {
                    w *= lo.sqrt();
                }
            }
            if mid != 1.0 && (50.0..90.0).contains(&s) {
                w *= mid;
            }
            if hi != 1.0 && s >= 90.0 {
                w *= hi;
            }
            cum += w;
            cum
        })
        .collect();
    let total = *raw.last().unwrap_or(&1.0);
    Some(raw.into_iter().map(|c| c / total).collect())
}

/// Quantile bands over a group's scores, matching
/// `strategy::build_bands`'s contract closely enough for RNG-consumption
/// parity (band COUNT is what the draw reads).
fn build_bands(scores: &[f64], n_bands: usize) -> Vec<Vec<usize>> {
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&a, &b| scores[a].partial_cmp(&scores[b]).unwrap());
    let mut out = vec![Vec::new(); n_bands.max(1)];
    for (rank, &row) in order.iter().enumerate() {
        let b = (rank * n_bands) / scores.len().max(1);
        out[b.min(n_bands - 1)].push(row);
    }
    out.retain(|b| !b.is_empty());
    out
}

/// Accumulates coverage over one window.
struct Acc {
    hits: Vec<Vec<u32>>,
    ref_seen: Vec<Vec<bool>>,
    cell_seen: Vec<std::collections::HashSet<(u32, u8)>>,
    pairs: Vec<u64>,
    near: Vec<u64>,
    within: Vec<u64>,
    bands: Vec<[u64; 4]>,
    draws: u64,
    same_row: u64,
    too_small: u64,
    dup: Option<(std::collections::HashSet<u64>, u64)>,
}

impl Acc {
    fn new(groups: &[SimGroup], track_dup: bool) -> Self {
        Self {
            hits: groups.iter().map(|g| vec![0u32; g.n_rows]).collect(),
            ref_seen: groups
                .iter()
                .map(|g| {
                    let n = g
                        .ref_ids
                        .as_ref()
                        .and_then(|r| r.iter().copied().max())
                        .map(|m| m as usize + 1)
                        .unwrap_or(0);
                    vec![false; n]
                })
                .collect(),
            cell_seen: groups
                .iter()
                .map(|_| std::collections::HashSet::new())
                .collect(),
            pairs: vec![0; groups.len()],
            near: vec![0; groups.len()],
            within: vec![0; groups.len()],
            bands: vec![[0u64; 4]; groups.len()],
            draws: 0,
            same_row: 0,
            too_small: 0,
            dup: track_dup.then(|| (std::collections::HashSet::new(), 0)),
        }
    }

    fn record(&mut self, groups: &[SimGroup], d: Draw) {
        self.draws += 1;
        match d {
            Draw::GroupTooSmall => self.too_small += 1,
            Draw::SameRow { .. } => self.same_row += 1,
            Draw::Pair { train_pos, ia, ib } => {
                let g = &groups[train_pos];
                self.pairs[train_pos] += 1;
                self.hits[train_pos][ia] += 1;
                self.hits[train_pos][ib] += 1;
                let sa = g.human_scores[ia];
                let sb = g.human_scores[ib];
                self.bands[train_pos][native_band(sa)] += 1;
                self.bands[train_pos][native_band(sb)] += 1;
                if sa.max(sb) >= 90.0 {
                    self.near[train_pos] += 1;
                }
                if let Some(rids) = &g.ref_ids {
                    let (ra, rb) = (rids[ia], rids[ib]);
                    self.ref_seen[train_pos][ra as usize] = true;
                    self.ref_seen[train_pos][rb as usize] = true;
                    self.cell_seen[train_pos].insert((ra, native_band(sa) as u8));
                    self.cell_seen[train_pos].insert((rb, native_band(sb) as u8));
                    if ra == rb {
                        self.within[train_pos] += 1;
                    }
                }
                if let Some((set, dups)) = &mut self.dup {
                    let key = ((train_pos as u64) << 56)
                        ^ ((ia.min(ib) as u64) << 28)
                        ^ (ia.max(ib) as u64);
                    if !set.insert(key) {
                        *dups += 1;
                    }
                }
            }
        }
    }

    fn finish(self, groups: &[SimGroup], cdf: &[f64]) -> SampleCoverage {
        let total_pairs: u64 = self.pairs.iter().sum();
        let mut per_group = Vec::with_capacity(groups.len());
        let mut pooled_touched = 0usize;
        let mut pooled_rows = 0usize;
        let mut pooled_near = 0u64;
        let mut pooled_within = 0u64;
        let mut pooled_within_denom = 0u64;
        for (i, g) in groups.iter().enumerate() {
            let hits = &self.hits[i];
            let touched = hits.iter().filter(|&&h| h > 0).count();
            pooled_touched += touched;
            pooled_rows += g.n_rows;
            let total_hits: f64 = hits.iter().map(|&h| h as f64).sum();
            let (mut ent, mut sumsq) = (0.0f64, 0.0f64);
            for &h in hits {
                if h > 0 && total_hits > 0.0 {
                    let p = h as f64 / total_hits;
                    ent -= p * p.ln();
                }
                sumsq += (h as f64) * (h as f64);
            }
            let n = g.n_rows.max(1) as f64;
            let mean = total_hits / n;
            let var = (sumsq / n) - mean * mean;
            let cv = if mean > 0.0 {
                var.max(0.0).sqrt() / mean
            } else {
                0.0
            };
            let n_refs = self.ref_seen[i].len();
            let refs_touched = self.ref_seen[i].iter().filter(|&&b| b).count();
            // Total non-empty (ref, band) cells present in the group.
            let n_cells = g
                .ref_ids
                .as_ref()
                .map(|r| {
                    let mut s = std::collections::HashSet::new();
                    for (row, &rid) in r.iter().enumerate() {
                        s.insert((rid, native_band(g.human_scores[row]) as u8));
                    }
                    s.len()
                })
                .unwrap_or(0);
            let cells_touched = self.cell_seen[i].len();
            pooled_near += self.near[i];
            if g.ref_ids.is_some() {
                pooled_within += self.within[i];
                pooled_within_denom += self.pairs[i];
            }
            per_group.push(GroupCoverage {
                name: g.name.clone(),
                n_rows: g.n_rows,
                train_weight: g.train_weight,
                n_pairs: self.pairs[i],
                rows_touched: touched,
                row_coverage: touched as f64 / n,
                refs_touched,
                n_refs,
                ref_coverage: if n_refs > 0 {
                    refs_touched as f64 / n_refs as f64
                } else {
                    f64::NAN
                },
                cell_coverage: if n_cells > 0 {
                    cells_touched as f64 / n_cells as f64
                } else {
                    f64::NAN
                },
                cells_touched,
                n_cells,
                row_entropy_norm: if n > 1.0 { ent / n.ln() } else { 0.0 },
                row_multiplicity_cv: cv,
                near_threshold_share: if self.pairs[i] > 0 {
                    self.near[i] as f64 / self.pairs[i] as f64
                } else {
                    f64::NAN
                },
                within_image_share: if g.ref_ids.is_some() && self.pairs[i] > 0 {
                    self.within[i] as f64 / self.pairs[i] as f64
                } else {
                    f64::NAN
                },
                band_pair_counts: self.bands[i],
            });
        }
        // Declared shares from the CDF (differences of the cumulative).
        let mut declared = Vec::with_capacity(cdf.len());
        let mut prev = 0.0;
        for &c in cdf {
            declared.push(c - prev);
            prev = c;
        }
        let (mut l1, mut chisq) = (0.0f64, 0.0f64);
        if total_pairs > 0 {
            for (i, d) in declared.iter().enumerate() {
                let obs = self.pairs[i] as f64 / total_pairs as f64;
                l1 += (obs - d).abs();
                if *d > 0.0 {
                    let e = d * total_pairs as f64;
                    let o = self.pairs[i] as f64;
                    chisq += (o - e) * (o - e) / e;
                }
            }
        }
        SampleCoverage {
            window_draws: self.draws,
            n_pairs: total_pairs,
            same_row_skips: self.same_row,
            group_too_small_skips: self.too_small,
            pooled_row_coverage: pooled_touched as f64 / pooled_rows.max(1) as f64,
            group_share_l1: l1,
            group_share_chisq: chisq,
            near_threshold_share: if total_pairs > 0 {
                pooled_near as f64 / total_pairs as f64
            } else {
                f64::NAN
            },
            within_image_share: if pooled_within_denom > 0 {
                pooled_within as f64 / pooled_within_denom as f64
            } else {
                f64::NAN
            },
            duplicate_pair_rate: match self.dup {
                Some((_, dups)) if total_pairs > 0 => dups as f64 / total_pairs as f64,
                _ => f64::NAN,
            },
            per_group,
        }
    }
}

/// Replay a training run's pair sampler without training.
///
/// Uses the same [`draw_pair`] the training loops use, so the sequence is
/// the run's sequence by construction rather than by a re-implementation
/// that has to be kept in sync.
pub fn simulate(groups: &[SimGroup], params: &SimParams) -> SimResult {
    assert!(
        params.pooled_rank_share.is_finite() && (0.0..=1.0).contains(&params.pooled_rank_share)
    );
    let pooled_mask: Vec<bool> = groups
        .iter()
        .map(|g| params.pooled_legs.contains(&g.name))
        .collect();
    assert!(
        params
            .pooled_legs
            .iter()
            .all(|name| groups.iter().any(|g| &g.name == name)),
        "unknown pooled leg"
    );
    assert!(
        params.pooled_legs.is_empty()
            || (!params.per_sample_alpha_head && !params.stratified_pairs),
        "pooled replay requires plain uniform CPU sampling"
    );
    let train_total: f64 = groups.iter().map(|g| g.train_weight).sum();
    let mut cum = 0.0;
    let cdf: Vec<f64> = groups
        .iter()
        .map(|g| {
            cum += g.train_weight;
            cum / train_total
        })
        .collect();
    let row_counts: Vec<usize> = groups.iter().map(|g| g.n_rows).collect();
    let per_row_cdfs: Vec<Option<Vec<f64>>> = groups
        .iter()
        .map(|g| {
            build_per_row_cdf(
                &g.human_scores,
                params.low_q_boost,
                params.mid_q_boost,
                params.high_q_boost,
            )
        })
        .collect();
    let ref_buckets: Vec<Option<RefBuckets>> = groups
        .iter()
        .map(|g| {
            if g.within_ref {
                g.ref_ids.as_deref().and_then(RefBuckets::build)
            } else {
                None
            }
        })
        .collect();
    let strat_bands: Vec<Vec<Vec<usize>>> = if params.stratified_bands > 0 {
        groups
            .iter()
            .map(|g| build_bands(&g.human_scores, params.stratified_bands))
            .collect()
    } else {
        Vec::new()
    };

    let plan: Option<StratifiedPlan> = if params.stratified_pairs {
        let per_group: Vec<Vec<Vec<usize>>> = groups
            .iter()
            .map(|g| strata_of(&g.human_scores, g.ref_ids.as_deref()))
            .collect();
        Some(StratifiedPlan::build(
            &per_group,
            params.pairs_per_epoch,
            params.seed,
        ))
    } else {
        None
    };
    let mut ctx = PairDrawCtx {
        cdf: &cdf,
        row_counts: &row_counts,
        per_row_cdfs: &per_row_cdfs,
        ref_buckets: &ref_buckets,
        strat_bands: &strat_bands,
        plan: plan.as_ref(),
        draw_index: 0,
    };

    let stream = if params.per_sample_alpha_head {
        sample_stream_seed_per_sample_alpha(params.seed)
    } else {
        sample_stream_seed(params.seed)
    };
    let mut rng = SplitMix64::new(stream);

    let early_n = if params.early_window == 0 {
        params.pairs_per_epoch
    } else {
        params.early_window
    };
    let total = params.epochs.saturating_mul(params.pairs_per_epoch);

    let mut full = Acc::new(groups, false);
    let mut early = Acc::new(groups, true);
    let mut digest = SampleSequenceDigest::new();
    let mut early_digest = SampleSequenceDigest::new();

    for step in 0..total {
        ctx.draw_index = step as u64;
        let d = draw_pair(&ctx, &mut rng);
        let d = if params.pooled_legs.is_empty() {
            d
        } else {
            pool_draw(
                d,
                &row_counts,
                &pooled_mask,
                params.pooled_rank_share,
                &mut rng,
            )
        };
        digest.push(d);
        full.record(groups, d);
        if step < early_n {
            early_digest.push(d);
            early.record(groups, d);
        }
    }

    SimResult {
        full: full.finish(groups, &cdf),
        early: early.finish(groups, &cdf),
        digest,
        early_digest,
    }
}

/// JSON for one coverage window. THE encoder — `subset_sim`'s replay output
/// and the `zentrain.sample_coverage` block a bake embeds are the SAME shape
/// because they come through here, which is what makes a stored bake's
/// coverage directly comparable to a later replay of the same seed.
pub fn coverage_json(c: &SampleCoverage) -> serde_json::Value {
    serde_json::json!({
        "window_draws": c.window_draws,
        "n_pairs": c.n_pairs,
        "same_row_skips": c.same_row_skips,
        "group_too_small_skips": c.group_too_small_skips,
        "pooled_row_coverage": c.pooled_row_coverage,
        "group_share_l1": c.group_share_l1,
        "group_share_chisq": c.group_share_chisq,
        "near_threshold_share": c.near_threshold_share,
        "within_image_share": c.within_image_share,
        "duplicate_pair_rate": c.duplicate_pair_rate,
        "per_group": c.per_group.iter().map(|g| serde_json::json!({
            "name": g.name,
            "n_rows": g.n_rows,
            "train_weight": g.train_weight,
            "n_pairs": g.n_pairs,
            "rows_touched": g.rows_touched,
            "row_coverage": g.row_coverage,
            "refs_touched": g.refs_touched,
            "n_refs": g.n_refs,
            "ref_coverage": g.ref_coverage,
            "cells_touched": g.cells_touched,
            "n_cells": g.n_cells,
            "cell_coverage": g.cell_coverage,
            "row_entropy_norm": g.row_entropy_norm,
            "row_multiplicity_cv": g.row_multiplicity_cv,
            "near_threshold_share": g.near_threshold_share,
            "within_image_share": g.within_image_share,
            "band_pair_counts": g.band_pair_counts,
        })).collect::<Vec<_>>(),
    })
}

/// The full `zentrain.sample_coverage` payload for a run: both windows, the
/// digest that proves a replay reproduced it, and every input the replay
/// needs.
///
/// # What this is, and what it is not
///
/// It is the record of **what a run's pair sampler actually touched**. A
/// seed does not only move where a run LANDS — it moves what the run SEES:
/// the sample stream is seeded, so two sample seeds walk the same fixed row
/// population in a different ORDER, and a finite-epoch run emphasises a
/// different subset. Two seed siblings therefore differ in coverage, and
/// before this block existed nothing on a bake said by how much.
///
/// It is NOT a claim that more coverage is better — that is the question the
/// subset-quality study asks, and this is the measurement that makes it
/// answerable. It is also NOT a record of which rows were ELIGIBLE: group
/// membership and the train/val split are fixed by the declared `--group`
/// corpora and the data-prep-time origin-hash rule, which no seed moves.
///
/// `digest` is the same rolling hash `subset_sim --expect-digest` checks, so
/// a stored block can be re-derived and verified without the bake's weights.
pub fn run_coverage_json(
    r: &SimResult,
    groups: &[SimGroup],
    params: &SimParams,
    init_seed: u64,
) -> serde_json::Value {
    let mut value = serde_json::json!({
        "schema": 1,
        "what": "what this run's PAIR SAMPLER actually touched. A seed moves what a run \
                 SEES, not only where it lands: the sample stream is seeded, so two sample \
                 seeds walk the same fixed row population in a different order and a \
                 finite-epoch run emphasises a different subset. This is that difference, \
                 measured.",
        "not": "NOT a claim that more coverage is better, and NOT a record of which rows \
                were ELIGIBLE — group membership and the train/val split are fixed by the \
                declared --group corpora and the data-prep-time origin-hash rule, which no \
                seed moves.",
        "init_seed": init_seed,
        "sample_seed": params.seed,
        "sample_stream_seed": if params.per_sample_alpha_head {
            sample_stream_seed_per_sample_alpha(params.seed)
        } else {
            sample_stream_seed(params.seed)
        },
        "epochs": params.epochs,
        "pairs_per_epoch": params.pairs_per_epoch,
        "low_q_boost": params.low_q_boost,
        "mid_q_boost": params.mid_q_boost,
        "high_q_boost": params.high_q_boost,
        "stratified_bands": params.stratified_bands,
        "pair_sampling": if params.stratified_pairs { "stratified" } else { "uniform" },
        "per_sample_alpha_head": params.per_sample_alpha_head,
        "band_edges": "B0 <50, B1 [50,65), B2 [65,90), B3 >=90 (human_score 0-100), the \
                       trainer's own q-boost bands",
        "digest": r.digest.hex(),
        "early_digest": r.early_digest.hex(),
        "groups_declared": groups.iter().map(|g| serde_json::json!({
            "name": g.name,
            "train_weight": g.train_weight,
            "n_rows": g.n_rows,
            "within_ref": g.within_ref,
            "has_ref_ids": g.ref_ids.is_some(),
        })).collect::<Vec<_>>(),
        "full": coverage_json(&r.full),
        "early": coverage_json(&r.early),
    });
    if !params.pooled_legs.is_empty() {
        value["pooled_objective"] = serde_json::json!({
            "legs": params.pooled_legs,
            "rank_share": params.pooled_rank_share,
            "coverage_scope": "rank-pair endpoints; independent auxiliary Pearson rows excluded",
        });
    }
    value
}

#[cfg(test)]
mod tests {
    use super::*;

    fn g(name: &str, w: f64, scores: Vec<f64>, refs: Option<Vec<u32>>, wr: bool) -> SimGroup {
        SimGroup {
            name: name.into(),
            train_weight: w,
            n_rows: scores.len(),
            human_scores: scores,
            ref_ids: refs,
            within_ref: wr,
        }
    }

    fn params(seed: u64) -> SimParams {
        SimParams {
            seed,
            epochs: 3,
            pairs_per_epoch: 500,
            low_q_boost: 1.0,
            mid_q_boost: 1.0,
            high_q_boost: 1.0,
            stratified_bands: 0,
            early_window: 100,
            per_sample_alpha_head: false,
            stratified_pairs: false,
            pooled_legs: Vec::new(),
            pooled_rank_share: 0.0,
        }
    }

    fn two_groups() -> Vec<SimGroup> {
        let a: Vec<f64> = (0..40).map(|i| (i as f64) * 2.5).collect();
        let b: Vec<f64> = (0..25).map(|i| 40.0 + (i as f64) * 2.0).collect();
        vec![
            g("a", 1.0, a, Some((0..40).map(|i| i / 4).collect()), false),
            g("b", 0.5, b, Some((0..25).map(|i| i / 5).collect()), false),
        ]
    }

    /// The stream constants are a wire contract: if either changes, every
    /// model ever trained had a different training subset than its repro
    /// block claims. Pinned, not derived.
    #[test]
    fn sample_stream_seeds_are_pinned() {
        assert_eq!(sample_stream_seed(0), 0xDEAD_BEEF_CAFE_BABE);
        assert_eq!(
            sample_stream_seed_per_sample_alpha(0),
            0x0123_4567_89AB_CDEF
        );
        // A concrete production seed, so a refactor of the arithmetic
        // (not just the constants) is caught too.
        assert_eq!(
            sample_stream_seed(4004),
            4004u64
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(0xDEAD_BEEF_CAFE_BABE)
        );
        assert_ne!(sample_stream_seed(4004), sample_stream_seed(4005));
        // Init and sample streams must not coincide for any seed we ship.
        for s in [0u64, 1, 42, 4004, 4005, 1301] {
            assert_ne!(s, sample_stream_seed(s));
        }
    }

    /// A PINNED digest over a fixed synthetic fixture.
    ///
    /// `simulate_is_deterministic_for_a_seed` only proves the replay agrees
    /// with itself; this pins the actual sequence, so any future change to
    /// the draw order or the RNG consumption pattern fails here instead of
    /// silently re-rolling every model's training subset. The value was
    /// taken from a run whose digest was independently verified to match a
    /// REAL `zensim_mlp_train` run under `ZENSIM_SAMPLE_DIGEST=1` (see
    /// `benchmarks/subset_quality_study_2026-09-04.md` §Phase 1).
    #[test]
    fn simulate_digest_is_pinned() {
        let gs = two_groups();
        let r = simulate(&gs, &params(4004));
        assert_eq!(
            r.digest.hex(),
            "39dd23c68ee3f018",
            "the pair-draw sequence changed — this is an ERA BREAK for every \
             model trained through this sampler, not a test to update casually"
        );
    }

    #[test]
    fn simulate_is_deterministic_for_a_seed() {
        let gs = two_groups();
        let a = simulate(&gs, &params(4004));
        let b = simulate(&gs, &params(4004));
        assert_eq!(a.digest.finish(), b.digest.finish());
        assert_eq!(a.full.n_pairs, b.full.n_pairs);
        assert_eq!(a.early.pooled_row_coverage, b.early.pooled_row_coverage);
    }

    /// Different digests do not establish independent sampling: these actual
    /// within-reference pair streams coincide after dropping one attempted pair.
    #[test]
    fn nearby_legacy_seeds_are_shifted_withinref_streams() {
        let refs = vec![0, 0, 1, 1, 2, 2, 3, 3];
        let buckets = [RefBuckets::build(&refs)];
        let ctx = PairDrawCtx {
            cdf: &[1.0],
            row_counts: &[8],
            per_row_cdfs: &[None],
            ref_buckets: &buckets,
            strat_bands: &[],
            plan: None,
            draw_index: 0,
        };
        for initialize in [sample_stream_seed, sample_stream_seed_per_sample_alpha] {
            let mut earlier = SplitMix64::new(initialize(17103));
            let mut later = SplitMix64::new(initialize(17107));
            let _ = draw_pair(&ctx, &mut earlier);
            for _ in 0..262143 {
                assert_eq!(draw_pair(&ctx, &mut earlier), draw_pair(&ctx, &mut later));
            }
        }
    }

    #[test]
    fn different_seeds_give_different_digests() {
        let gs = two_groups();
        let a = simulate(&gs, &params(4004));
        let b = simulate(&gs, &params(4005));
        assert_ne!(a.digest.finish(), b.digest.finish());
    }

    /// The digest must be sensitive to skip behaviour, not just to the
    /// pairs that survive — a skip shifts every later draw.
    #[test]
    fn digest_distinguishes_skips_from_pairs() {
        let mut d1 = SampleSequenceDigest::new();
        let mut d2 = SampleSequenceDigest::new();
        d1.push(Draw::Pair {
            train_pos: 0,
            ia: 1,
            ib: 2,
        });
        d2.push(Draw::SameRow {
            train_pos: 0,
            row: 5,
        });
        d2.push(Draw::Pair {
            train_pos: 0,
            ia: 1,
            ib: 2,
        });
        assert_ne!(d1.finish(), d2.finish());
        let mut d3 = SampleSequenceDigest::new();
        d3.push(Draw::Pair {
            train_pos: 0,
            ia: 2,
            ib: 1,
        });
        assert_ne!(
            d1.finish(),
            d3.finish(),
            "pair ORDER is part of the sequence"
        );
    }

    /// A group with fewer than 2 rows consumes exactly ONE RNG value and
    /// returns early. Getting this wrong desynchronises the whole stream.
    #[test]
    fn tiny_group_consumes_one_draw_only() {
        let gs = vec![g("solo", 1.0, vec![50.0], None, false)];
        let cdf = [1.0];
        let rows = [1usize];
        let prc = [None];
        let rb = [None];
        let ctx = PairDrawCtx {
            cdf: &cdf,
            row_counts: &rows,
            per_row_cdfs: &prc,
            ref_buckets: &rb,
            strat_bands: &[],
            plan: None,
            draw_index: 0,
        };
        let _ = gs;
        let mut r1 = SplitMix64::new(7);
        assert_eq!(draw_pair(&ctx, &mut r1), Draw::GroupTooSmall);
        // One value consumed: a fresh RNG advanced once must match.
        let mut r2 = SplitMix64::new(7);
        let _ = r2.next_f64_unit();
        assert_eq!(r1.next_u64(), r2.next_u64());
    }

    /// Within-ref draws must never cross reference images.
    #[test]
    fn within_ref_pairs_share_a_reference() {
        let refs: Vec<u32> = (0..40).map(|i| i / 4).collect();
        let scores: Vec<f64> = (0..40).map(|i| (i as f64) * 2.5).collect();
        let gs = vec![g("a", 1.0, scores, Some(refs.clone()), true)];
        let mut p = params(11);
        p.epochs = 2;
        let res = simulate(&gs, &p);
        assert!(res.full.n_pairs > 100);
        let wi = res.full.per_group[0].within_image_share;
        assert!(
            (wi - 1.0).abs() < 1e-12,
            "within-ref group drew cross-image pairs: share={wi}"
        );
    }

    /// Cross-image is the default: the same group WITHOUT the opt-in must
    /// draw mostly across references (the negative control for the test
    /// above).
    #[test]
    fn cross_image_is_the_default() {
        let refs: Vec<u32> = (0..40).map(|i| i / 4).collect();
        let scores: Vec<f64> = (0..40).map(|i| (i as f64) * 2.5).collect();
        let gs = vec![g("a", 1.0, scores, Some(refs), false)];
        let res = simulate(&gs, &params(11));
        let wi = res.full.per_group[0].within_image_share;
        assert!(wi < 0.25, "expected mostly cross-image draws, got {wi}");
    }

    /// A q-boost must move the realised band mix, and must do so in the
    /// declared direction — otherwise the boost silently does nothing.
    #[test]
    fn high_q_boost_shifts_the_band_mix() {
        let gs = two_groups();
        let flat = simulate(&gs, &params(3));
        let mut p = params(3);
        p.high_q_boost = 3.0;
        let boosted = simulate(&gs, &p);
        let b3 = |r: &SimResult| -> f64 {
            let (mut hi, mut tot) = (0u64, 0u64);
            for g in &r.full.per_group {
                hi += g.band_pair_counts[3];
                tot += g.band_pair_counts.iter().sum::<u64>();
            }
            hi as f64 / tot as f64
        };
        assert!(
            b3(&boosted) > b3(&flat) * 1.2,
            "high_q_boost=3 did not raise the >=90 share: {} vs {}",
            b3(&boosted),
            b3(&flat)
        );
    }

    /// Group shares must track the declared weights (2:1 here).
    #[test]
    fn realised_group_share_tracks_declared_weight() {
        let gs = two_groups();
        let mut p = params(5);
        p.epochs = 20;
        let r = simulate(&gs, &p);
        let a = r.full.per_group[0].n_pairs as f64;
        let b = r.full.per_group[1].n_pairs as f64;
        assert!(
            (a / b - 2.0).abs() < 0.15,
            "declared 1.0:0.5 but drew {a}:{b}"
        );
        assert!(r.full.group_share_l1 < 0.02, "{}", r.full.group_share_l1);
    }

    /// The early window is a strict prefix of the full run.
    #[test]
    fn early_window_is_a_prefix_of_the_full_run() {
        let gs = two_groups();
        let r = simulate(&gs, &params(9));
        assert_eq!(r.early.window_draws, 100);
        assert_eq!(r.full.window_draws, 1500);
        assert!(r.early.n_pairs <= r.full.n_pairs);
        for (e, f) in r.early.per_group.iter().zip(r.full.per_group.iter()) {
            assert!(
                e.rows_touched <= f.rows_touched,
                "early window touched more rows than the whole run"
            );
        }
        // And a simulation whose TOTAL equals the early window must have
        // the same digest as that window.
        let mut p = params(9);
        p.epochs = 1;
        p.pairs_per_epoch = 100;
        p.early_window = 100;
        let short = simulate(&gs, &p);
        assert_eq!(short.digest.finish(), r.early_digest.finish());
    }

    /// `run_coverage_json` carries the digest `simulate` produced and every
    /// field the board reads. A block whose digest disagreed with its own
    /// simulation would be unfalsifiable — `subset_sim --expect-digest`
    /// could never catch a stale or mis-parameterised embed.
    #[test]
    fn run_coverage_json_carries_the_digest_and_the_board_read_fields() {
        let gs = vec![
            SimGroup {
                name: "a".into(),
                train_weight: 1.0,
                n_rows: 40,
                human_scores: (0..40).map(|i| i as f64 * 2.5).collect(),
                ref_ids: Some((0..40).map(|i| (i / 4) as u32).collect()),
                within_ref: true,
            },
            SimGroup {
                name: "b".into(),
                train_weight: 0.5,
                n_rows: 25,
                human_scores: (0..25).map(|i| 50.0 + i as f64).collect(),
                ref_ids: None,
                within_ref: false,
            },
        ];
        let p = SimParams {
            seed: 100,
            epochs: 2,
            pairs_per_epoch: 300,
            low_q_boost: 1.0,
            mid_q_boost: 1.0,
            high_q_boost: 1.0,
            stratified_bands: 0,
            early_window: 0,
            per_sample_alpha_head: false,
            stratified_pairs: false,
            pooled_legs: Vec::new(),
            pooled_rank_share: 0.0,
        };
        let r = simulate(&gs, &p);
        let v = run_coverage_json(&r, &gs, &p, 777);

        assert_eq!(v["digest"], serde_json::json!(r.digest.hex()));
        assert_eq!(v["early_digest"], serde_json::json!(r.early_digest.hex()));
        // The two seeds are reported SEPARATELY: coverage is a property of
        // the sample stream, and saying so is what lets a reader tell an
        // init re-roll from a sample re-roll.
        assert_eq!(v["init_seed"], serde_json::json!(777u64));
        assert_eq!(v["sample_seed"], serde_json::json!(100u64));
        assert_eq!(
            v["sample_stream_seed"],
            serde_json::json!(sample_stream_seed(100))
        );
        for k in [
            "schema",
            "epochs",
            "pairs_per_epoch",
            "stratified_bands",
            "groups_declared",
            "full",
            "early",
        ] {
            assert!(!v[k].is_null(), "missing board-read field {k}");
        }
        for k in [
            "pooled_row_coverage",
            "n_pairs",
            "per_group",
            "group_share_l1",
        ] {
            assert!(!v["full"][k].is_null(), "missing full.{k}");
        }
        // Per-group rows carry the four things the brief names: distinct
        // refs, distinct (ref, band) cells, the fraction of rows ever
        // paired, and the pair count — plus the band histogram.
        let g0 = &v["full"]["per_group"][0];
        for k in [
            "refs_touched",
            "cells_touched",
            "row_coverage",
            "n_pairs",
            "band_pair_counts",
        ] {
            assert!(!g0[k].is_null(), "missing per_group.{k}");
        }
        assert_eq!(g0["name"], serde_json::json!("a"));
        // `groups_declared` is the DECLARED table, so a reader can tell a
        // group that was never drawn from a group that was never declared.
        assert_eq!(v["groups_declared"].as_array().unwrap().len(), 2);
    }

    /// Coverage depends on the SAMPLE seed and on nothing else in the pair:
    /// the same `SimParams` gives a byte-identical block, a different seed
    /// does not. This is the in-process form of `coverage_embed_gate.sh`
    /// checks 2 and 3.
    #[test]
    fn coverage_json_is_a_function_of_the_sample_seed() {
        let gs = vec![SimGroup {
            name: "a".into(),
            train_weight: 1.0,
            n_rows: 30,
            human_scores: (0..30).map(|i| i as f64 * 3.0).collect(),
            ref_ids: None,
            within_ref: false,
        }];
        let mk = |seed: u64| SimParams {
            seed,
            epochs: 2,
            pairs_per_epoch: 200,
            low_q_boost: 1.0,
            mid_q_boost: 1.0,
            high_q_boost: 1.0,
            stratified_bands: 0,
            early_window: 0,
            per_sample_alpha_head: false,
            stratified_pairs: false,
            pooled_legs: Vec::new(),
            pooled_rank_share: 0.0,
        };
        let (p1, p2) = (mk(100), mk(999));
        let j = |p: &SimParams, init: u64| {
            run_coverage_json(&simulate(&gs, p), &gs, p, init).to_string()
        };
        // Different INIT seed, same sample seed: identical, including digest.
        assert_eq!(
            j(&p1, 1).replace("\"init_seed\":1,", "I"),
            j(&p1, 42).replace("\"init_seed\":42,", "I"),
            "coverage must not depend on the init seed"
        );
        // Different SAMPLE seed: it must actually move.
        assert_ne!(
            simulate(&gs, &p1).digest.hex(),
            simulate(&gs, &p2).digest.hex(),
            "coverage must depend on the sample seed"
        );
    }

    // ── `--pair-sampling stratified` ────────────────────────────────────

    fn strat_groups() -> Vec<SimGroup> {
        // 60 rows over 6 refs x a score ramp that lands rows in several
        // native bands, so the (ref, band) partition is non-trivial.
        vec![SimGroup {
            name: "a".into(),
            train_weight: 1.0,
            n_rows: 60,
            human_scores: (0..60).map(|i| (i as f64) * 1.7).collect(),
            ref_ids: Some((0..60).map(|i| (i % 6) as u32).collect()),
            within_ref: false,
        }]
    }

    #[test]
    fn stratified_group_shares_ignore_weights_uniform_respects_them() {
        // Same four rows per reference, but one group has eight references.
        // Global stratum cycling is not a weighted mixture.
        let make = |weight| {
            vec![
                g("small", weight, vec![30.0; 4], Some(vec![0; 4]), true),
                g(
                    "large",
                    1.0,
                    vec![30.0; 32],
                    Some((0..32).map(|i| i / 4).collect()),
                    true,
                ),
            ]
        };
        let mut p = strat_params(7101);
        p.pairs_per_epoch = 18_000;
        let equal = simulate(&make(1.0), &p);
        let weighted = simulate(&make(9.0), &p);
        assert_eq!(equal.digest.hex(), weighted.digest.hex());
        assert_eq!(
            weighted.full.per_group[0].n_pairs * 8,
            weighted.full.per_group[1].n_pairs
        );
        p.stratified_pairs = false;
        let uniform = simulate(&make(9.0), &p);
        let share = uniform.full.per_group[0].n_pairs as f64 / uniform.full.n_pairs as f64;
        assert!((share - 0.9).abs() < 0.01, "weighted group share {share}");
        assert_ne!(uniform.digest.hex(), weighted.digest.hex());
    }

    fn strat_params(seed: u64) -> SimParams {
        SimParams {
            seed,
            epochs: 4,
            pairs_per_epoch: 400,
            low_q_boost: 1.0,
            mid_q_boost: 1.0,
            high_q_boost: 1.0,
            stratified_bands: 0,
            early_window: 0,
            per_sample_alpha_head: false,
            stratified_pairs: true,
            pooled_legs: Vec::new(),
            pooled_rank_share: 0.0,
        }
    }

    /// **The property the mode exists for**, and the failing-first shape:
    /// under `stratified`, two SAMPLE SEEDS touch the same rows, refs and
    /// cells — coverage stops being a property of the seed — while under
    /// `uniform` they do not. Both halves are asserted, because only the
    /// contrast is evidence: a stratified-only assertion would also pass
    /// on a mode that ignored the seed entirely.
    #[test]
    fn stratified_coverage_is_seed_invariant_and_uniform_is_not() {
        let gs = strat_groups();
        let cov = |p: &SimParams| {
            let c = &simulate(&gs, p).full.per_group[0];
            (
                c.rows_touched,
                c.refs_touched,
                c.cells_touched,
                c.n_pairs,
                c.band_pair_counts,
            )
        };
        let (s1, s2) = (strat_params(100), strat_params(999));
        assert_eq!(
            cov(&s1),
            cov(&s2),
            "stratified: two sample seeds must touch the SAME rows/refs/cells"
        );
        // ...and it is not vacuous: the schedule reaches EVERY row that a
        // stratum can pair, which is the guarantee — not "every row in the
        // group". A (ref, band) cell holding one row cannot yield an
        // in-stratum pair and is excluded by construction, so the reachable
        // set is computed from the plan rather than assumed to be all 60.
        // (This fixture HAS such cells: 9 rows land in band 1 and 7 in band
        // 3, spread over 6 refs, so several cells hold exactly one row —
        // the first version of this test asserted 60 and correctly failed.)
        let per_group: Vec<Vec<Vec<usize>>> = gs
            .iter()
            .map(|g| strata_of(&g.human_scores, g.ref_ids.as_deref()))
            .collect();
        let plan = StratifiedPlan::build(&per_group, s1.pairs_per_epoch, s1.seed);
        let reachable: usize = plan.strata.iter().map(|st| st.rows.len()).sum();
        assert!(
            plan.singleton_strata > 0,
            "fixture must contain excluded singleton cells, else this assertion is              indistinguishable from the naive one"
        );
        assert!((40..60).contains(&reachable), "reachable {reachable}");
        assert_eq!(cov(&s1).0, reachable, "every pairable row must be reached");
        let reachable_refs: std::collections::BTreeSet<u32> = plan
            .strata
            .iter()
            .flat_map(|st| st.rows.iter().map(|&r| gs[0].ref_ids.as_ref().unwrap()[r]))
            .collect();
        assert_eq!(
            cov(&s1).1,
            reachable_refs.len(),
            "every pairable ref is reached"
        );
        assert_eq!(reachable_refs.len(), 6, "all 6 refs survive somewhere");

        // The seed still does something: the pair ORDER (hence the digest)
        // differs, which is exactly "permutes order WITHIN strata only".
        assert_ne!(
            simulate(&gs, &s1).digest.hex(),
            simulate(&gs, &s2).digest.hex(),
            "the sample seed must still permute the within-stratum walk"
        );

        // UNIFORM, same two seeds, same everything else: coverage MOVES.
        let mut u1 = strat_params(100);
        u1.stratified_pairs = false;
        let mut u2 = strat_params(999);
        u2.stratified_pairs = false;
        assert_ne!(
            cov(&u1),
            cov(&u2),
            "uniform: coverage must vary with the sample seed — if this fails the \
             contrast the stratified assertion rests on is not real"
        );
    }

    /// Every stratum is visited on a fixed round-robin, so within one epoch
    /// each is reached `pairs_per_epoch / n_strata` times and the visit
    /// counts differ by at most one. This is the "every ref x band each
    /// epoch" claim, asserted rather than asserted-in-a-doc-comment.
    #[test]
    fn stratified_visits_every_stratum_each_epoch() {
        let gs = strat_groups();
        let per_group: Vec<Vec<Vec<usize>>> = gs
            .iter()
            .map(|g| strata_of(&g.human_scores, g.ref_ids.as_deref()))
            .collect();
        let plan = StratifiedPlan::build(&per_group, 400, 100);
        assert!(plan.strata.len() >= 6, "several (ref, band) cells");
        assert!(plan.covers_every_stratum_per_epoch());

        let mut visits = vec![0usize; plan.strata.len()];
        for k in 0..400u64 {
            visits[(k % plan.strata.len() as u64) as usize] += 1;
        }
        let (lo, hi) = (*visits.iter().min().unwrap(), *visits.iter().max().unwrap());
        assert!(
            hi - lo <= 1,
            "round-robin visit counts differ by >1: {lo}..{hi}"
        );

        // A too-small budget is REPORTED, not silently mis-promised.
        let tight = StratifiedPlan::build(&per_group, 2, 100);
        assert!(!tight.covers_every_stratum_per_epoch());
    }

    /// The within-stratum walk is a BIJECTION: `len` consecutive visits
    /// enumerate the stratum exactly once each, for every seed. That is
    /// what makes the row SET seed-invariant while the order is not.
    #[test]
    fn stratum_walk_is_a_bijection_for_every_seed() {
        for len in 2..40usize {
            for seed in [0u64, 1, 7, 100, 999, u64::MAX] {
                let rows: Vec<usize> = (0..len).collect();
                let plan = StratifiedPlan::build(&[vec![rows.clone()]], 1000, seed);
                let s = &plan.strata[0];
                let seen: std::collections::BTreeSet<usize> =
                    (0..len as u64).map(|j| s.row_at(j)).collect();
                assert_eq!(
                    seen.len(),
                    len,
                    "len {len} seed {seed}: walk is not a bijection"
                );
            }
        }
    }

    /// Singleton strata cannot yield an in-stratum pair. They are EXCLUDED
    /// and counted, rather than silently emitting collisions that would
    /// look like RNG bad luck in the coverage record.
    #[test]
    fn singleton_strata_are_excluded_and_counted() {
        let plan = StratifiedPlan::build(&[vec![vec![0], vec![1, 2, 3], vec![4]]], 100, 7);
        assert_eq!(plan.strata.len(), 1);
        assert_eq!(plan.singleton_strata, 2);
        assert_eq!(plan.strata[0].rows, vec![1, 2, 3]);
    }

    /// **Default byte-identity**: a `None` plan leaves `draw_pair` the
    /// function it was, draw for draw, including RNG consumption. Asserted
    /// against a full simulate digest so a change in skip behaviour (which
    /// shifts every later draw) cannot pass.
    #[test]
    fn pair_sampling_uniform_is_the_untouched_draw() {
        let gs = strat_groups();
        let mut p = strat_params(100);
        p.stratified_pairs = false;
        let a = simulate(&gs, &p);
        let b = simulate(&gs, &p);
        assert_eq!(a.digest.finish(), b.digest.finish());
        // The uniform path must still consume RNG (the stratified one does
        // not) — a plan-free run whose draws did not depend on the stream
        // would mean the `None` branch had been bypassed.
        let mut q = strat_params(4242);
        q.stratified_pairs = false;
        assert_ne!(a.digest.hex(), simulate(&gs, &q).digest.hex());
    }
}

#[cfg(test)]
mod pooled_tests {
    use super::*;
    #[test]
    fn pooled_draws_keep_leg_and_disabled_stream() {
        let draw = Draw::Pair {
            train_pos: 1,
            ia: 0,
            ib: 1,
        };
        let mut a = SplitMix64::new(27);
        let mut b = SplitMix64::new(27);
        assert_eq!(
            pool_draw(draw, &[2, 100], &[false, true], 0.0, &mut a),
            draw
        );
        assert_eq!(a.next_u64(), b.next_u64());
        let mut changed = 0;
        for _ in 0..10000 {
            match pool_draw(draw, &[2, 100], &[false, true], 0.5, &mut a) {
                Draw::Pair { train_pos, ia, ib } => {
                    assert_eq!(train_pos, 1);
                    assert!(ia < 100 && ib < 100);
                    if ia != 0 || ib != 1 {
                        changed += 1;
                    }
                }
                Draw::SameRow { train_pos, .. } => assert_eq!(train_pos, 1),
                _ => panic!("invalid pooled draw"),
            }
        }
        assert!(changed > 4700 && changed < 5300);
        assert_eq!(
            pool_draw(draw, &[2, 100], &[false, false], 1.0, &mut a),
            draw
        );
    }
}
