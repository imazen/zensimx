//! THE v1 HF-energy ratio family — one owner for three expressions that were
//! hand-copied at three sites, and the home of the **F17** fix.
//!
//! # The family, and which member is broken
//!
//! v1's basic block ends with three ratio slots built from two pooled second
//! moments (`var_* = Σ(x − μ)²/n`) and one pooled first absolute moment
//! (`mad_* = Σ|x − μ|/n`):
//!
//! ```text
//! var_loss     = max(0, 1 − var_dst/var_src)      bounded [0, 1]
//! tex_loss     = max(0, 1 − mad_dst/mad_src)      bounded [0, 1]
//! contrast_inc = max(0, var_dst/var_src − 1)      UNBOUNDED ABOVE
//! ```
//!
//! All three are `max(0, ·)` of a difference over the **source** term. That
//! bounds the two `loss` members structurally — their numerator can never
//! exceed their denominator — and bounds nothing at all for the `gain` member,
//! whose numerator is the distorted term. A flat source drives `var_src → 0`
//! past the [`VAR_SRC_FLOOR`] gate while the distorted image still carries HF,
//! and the ratio runs away. **The gate is a threshold, not a stabiliser**: it
//! decides *whether* to divide, never *what* to divide by.
//!
//! # F17, measured
//!
//! Over the R6 rev1 tables — 216,756 real pairs, eight corpora, all 372 slots
//! (`docs/PLAN_FEATURE_REV2_2026-09-05.md` §11.2): the twelve `contrast_inc`
//! slots are the **top twelve of all 372 by maximum**, worst **36,465.74**, and
//! the thirteenth slot is **1.972**. The population separates with no overlap.
//! Against the gold photographic holdout's own p99.9 over those slots
//! (CID22, 0.34687) the worst value is **×105,127**. Its two siblings max at
//! exactly `1.000000`, as the algebra above says they must.
//!
//! This is **not** F4's shape. F4's 5.8e6 lives in a bigcodec sweep with no
//! local pixels and fires on ZERO of those 216,756 rows; F17 fires on five
//! distortion corpora and on the training leg, on 0.0198 % of cells.
//!
//! # The fixed form already exists one block over
//!
//! v2's `HF_GAIN` — the same quantity in the same units (squared XYB energy),
//! computed per pixel instead of per pool — is
//! `bounded_excess_pair(hf_dst², hf_src², C_HF)`, i.e.
//! `max(0, a−b)/(a+b+C_HF)`, with `C_HF` declared as *"stabilizer for the HF
//! gain/loss/mag-loss bounded-excess forms"*. So F17 is not a missing idea; it
//! is one block that did not use the idea. [`HfGainForm::BoundedExcess`] is
//! that owner's form applied here.
//!
//! # Why an arm and not just a clamp
//!
//! Because `contrast_inc` is nonzero on 12 % (CID22) to 52 % (KADID) of cells,
//! any bounded form necessarily moves healthy content — so *which* bounded form
//! is a measurement, pre-registered as §11 of the rev2 plan and decided by it,
//! exactly as `ssim_form` handled F4.

use crate::feature_defs::FormulaRevision;

/// The `var_src` / `mad_src` floor below which the ratio is not formed.
///
/// A THRESHOLD, not a stabiliser — it chooses whether to divide, and the value
/// it divides by is unchanged on the other side of it. Owned here so it cannot
/// drift between the three sites that used to spell it out.
pub(crate) const VAR_SRC_FLOOR: f64 = 1e-10;

/// The HF bounded-excess stabiliser, v1 — same value and same derivation as
/// `feature_v2::C_HF`, declared independently so a future v2-only change
/// cannot silently move v1's bytes. `C2_V2` carries the same note for the same
/// reason, in the opposite direction. Held equal by
/// [`tests::v1_c_hf_matches_the_v2_owner`].
pub(crate) const C_HF_V1: f64 = 1e-4;

/// The cap [`HfGainForm::CappedExcess`] applies — **derived, not chosen**.
///
/// `1.0` is exactly the bound the two `loss` members of this family already
/// have, so capping the gain member there gives the family one shared range
/// without introducing a constant from outside it. It is also the point at
/// which the distorted region carries twice the source's HF energy.
///
/// Its cost, stated because the arm is carried to measure it: `g > 1` on
/// **2.5854 %** of the 2,601,072 measured cells, and every one of them
/// receives the same value — so this arm, alone in the family, does not
/// preserve severity order. See [`HfGainForm::preserves_order`].
pub(crate) const GAIN_CAP: f64 = 1.0;

/// Which form the HF *gain* member uses.
///
/// The revision axis is owned by [`crate::feature_defs`]; this enum is the
/// arithmetic a revision SELECTS — the same split [`crate::ssim_form`] uses.
/// Four bounded/compressing arms exist because the choice between them is a
/// measurement (`docs/PLAN_FEATURE_REV2_2026-09-05.md` §11), not an argument.
///
/// Throughout, `a = var_dst`, `b = var_src`, and `g = max(0, a/b − 1)`.
// Every variant ends in `Excess` because every arm IS an excess form —
// `max(0, dst - src)` under four different normalisations. Dropping the suffix
// would leave `Ratio`, `Bounded`, `Log1p`, `Saturating`, `Capped`, which read
// as unrelated ideas rather than as one family; the shared word is the point.
#[allow(clippy::enum_variant_names)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum HfGainForm {
    /// `g`. The shipped form, **unbounded above** (F17).
    #[default]
    RatioExcess,
    /// `max(0, a−b) / (a + b + C_HF)` — [`crate::feature_v2`]'s own
    /// `bounded_excess`, the crate's existing owner for this exact quantity.
    ///
    /// Bounded `[0, 1)`. Same numerator as [`Self::RatioExcess`]; the
    /// denominator is the stabilised one. Introduces no constant — `C_HF` is
    /// this family's declared stabiliser.
    ///
    /// Its cost: the denominator swap rescales even the smallest values, since
    /// `(a−b)/(a+b+C) → g/2` as `g → 0`. It also leaves the family spelled two
    /// ways — the gain member as a bounded excess, the two loss members still
    /// as `max(0, 1 − ratio)` — so making the family consistent under this arm
    /// would move 36 slots instead of 12.
    BoundedExcess,
    /// `ln(1 + g)`. Monotone, zero-preserving, agrees with
    /// [`Self::RatioExcess`] to first order — and **still unbounded**, which is
    /// exactly why it is carried: the log family is the obvious cheap answer
    /// and this is what measuring it costs rather than assuming it.
    Log1pExcess,
    /// `g / (g + 1)`, i.e. the crate's own `saturate(g, 1)` idiom.
    ///
    /// Bounded `[0, 1)`, strictly increasing, agrees with
    /// [`Self::RatioExcess`] to **first** order (`g − g² + …`), and `c = 1` is
    /// the unique scale at which it does — so it introduces no constant, the
    /// same derivation [`crate::ssim_form::SsimLumaForm::Lorentz`] uses.
    ///
    /// It also has an identity the other arms do not:
    /// `g/(g+1) = max(0, 1 − var_src/var_dst)`, the exact src↔dst reflection of
    /// `var_loss = max(0, 1 − var_dst/var_src)`. So it makes the family
    /// symmetric by changing only the member that is broken.
    SaturatingExcess,
    /// `min(g, `[`GAIN_CAP`]`)`. **Exact** below the cap and flat above it —
    /// the [`crate::ssim_form::SsimLumaForm::Clamp`] analogue. Bounds without
    /// touching anything below a doubling of HF energy, at the cost of
    /// flattening order among the 2.59 % of cells above it.
    CappedExcess,
}

impl HfGainForm {
    /// Resolve a per-comparison revision while retaining explicit research defaults.
    pub(crate) fn at_revision(revision: Option<crate::feature_defs::FormulaRevision>) -> Self {
        let rev = crate::ssim_form::effective_revision(
            revision.unwrap_or_else(crate::ssim_form::active_revision),
        );
        if rev == crate::ssim_form::active_revision() {
            active_gain_form()
        } else {
            Self::for_revision(rev)
        }
    }

    /// The form a registered revision selects.
    pub(crate) const fn for_revision(rev: FormulaRevision) -> Self {
        match rev {
            FormulaRevision::Rev1 => Self::RatioExcess,
            FormulaRevision::Rev2
            | FormulaRevision::Rev3
            | FormulaRevision::Rev4
            | FormulaRevision::Rev5
            | FormulaRevision::Rev6
            | FormulaRevision::Rev7 => Self::REV2_HFGAIN,
        }
    }

    /// The arm revision 2 ships — **`SaturatingExcess`, DECIDED BY MEASUREMENT**
    /// (`benchmarks/feature_rev2_2026-09-05.md` §11, R6b).
    ///
    /// Named ONCE so the R6b probe and the kernels cannot disagree about what
    /// "rev2" means for this family.
    ///
    /// # How the pre-registered rule selected it
    ///
    /// Five arms were extracted from ONE binary over 216,756 rows (the seven
    /// human eval corpora + the full 196,086-row safesyn training leg), fitted
    /// through the shipped Profile-D recipe at two slices × two solvers, and
    /// graded on the gates `docs/PLAN_FEATURE_REV2_2026-09-05.md` §11.8 fixed
    /// before any of it ran. Rule 1 (H3/H4/H5) left exactly one arm standing:
    ///
    /// * [`Self::Log1pExcess`] fails **H3** — log growth is a compression, not
    ///   a bound, and it measures **10.504** where the other arms cannot exceed
    ///   1. Carried precisely so that cost is measured rather than assumed.
    /// * [`Self::BoundedExcess`] fails **H5** with **263,195** adjacent-pair
    ///   inversions: it reads the MAGNITUDE of `var_src`, so it does not bound
    ///   the shipped statistic, it replaces it with a scale-dependent one.
    /// * [`Self::CappedExcess`] fails **H5** with **67,224** new ties — F4's
    ///   `Clamp` analogue, which was free there (0 moved cells) and is not here
    ///   (2.59 % of cells exceed `g = 1`).
    /// * `SaturatingExcess` passes all three: max **0.99997** against its
    ///   declared bound of 1, **0** inversions, **0** new ties.
    ///
    /// Rule 2 then fired on its own terms — it is the sole survivor AND wins a
    /// strict majority of {CID22, KonJND, AIC-3} with CI-excluding paired
    /// bootstrap deltas in 2 of the 4 variants (CID22 **+0.0027…+0.0090**,
    /// AIC-3 **+0.0009…+0.0032**). Its two constant-free derivations survived
    /// the measurement: it agrees with revision 1 to FIRST order in `g`, and
    /// `g/(g+1) = max(0, 1 − var_src/var_dst)` makes the family symmetric by
    /// changing only the member that is broken.
    ///
    /// # What it buys, and what it costs — both measured
    ///
    /// **LIVE goes 0.7357 → 0.9500** (+0.214) and 0.7960 → 0.9503 on the
    /// `lasso` variants, with TID +0.033/+0.040 and KADID +0.021/+0.031: the
    /// unbounded slot was actively wrecking the fit on the corpora where it
    /// fires (LIVE holds 122 of 779 rows above 100).
    ///
    /// **KonJND regresses, on every arm and every variant**
    /// (`satexcess` −0.013…−0.080, CI-excluding on 3 of 4). Two structurally
    /// different bounded maps produce the same sign, so this is a property of
    /// BOUNDING `contrast_inc`, not of this arm's shape: the near-threshold
    /// corpus is using the unbounded magnitude and no bounded form returns it.
    /// If that axis is worth recovering it is an APPEND slot under the
    /// append-only numbering, not a reason to keep an unbounded one.
    ///
    /// The dial (H7) does not regress: monotonicity within 0.0011 of revision 1,
    /// tied rate 0.0000–0.0001, negative-tail fraction within 0.022 (and
    /// BETTER on `s228_bvls`), at a reach cost of 2.2–11.8 points.
    pub(crate) const REV2_HFGAIN: Self = Self::SaturatingExcess;

    /// Whether the arm has a **structural** upper bound (gate H3).
    ///
    /// [`Self::Log1pExcess`] answers `false` on purpose: log growth is not a
    /// bound, and an arm that merely makes the number small enough to look
    /// tidy is the failure mode this flag exists to name.
    pub(crate) const fn bounds_gain(self) -> bool {
        matches!(
            self,
            Self::BoundedExcess | Self::SaturatingExcess | Self::CappedExcess
        )
    }

    /// Whether the arm is strictly increasing in `var_dst` **at a fixed
    /// `var_src`** — the LOCAL ordering, within one cell's sweep.
    ///
    /// Only [`Self::CappedExcess`] is not, and it is not by construction.
    ///
    /// ⚠ This is NOT gate H5, and the difference cost a wrong prediction: H5
    /// is the GLOBAL ordering, across cells with different `var_src`, and an
    /// arm can be locally monotone and still re-rank the population. See
    /// [`Self::depends_only_on_ratio`].
    ///
    /// Read by [`tests::order_preservation_matches_the_declared_flag`], which
    /// holds the flag equal to the arm's actual behaviour.
    #[allow(dead_code)]
    pub(crate) const fn preserves_order(self) -> bool {
        !matches!(self, Self::CappedExcess)
    }

    /// Whether the arm is a function of the RATIO `g` alone — the property
    /// gate **H5** actually measures, and the one that decides whether the arm
    /// preserves the shipped slot's ordering over a whole corpus.
    ///
    /// [`Self::BoundedExcess`] is the exception, and it is worth stating why
    /// rather than merely flagging: `max(0, a−b)/(a+b+C)` depends on the
    /// MAGNITUDE of `b`, not only on `a/b`. Two cells with the same `g` and
    /// different `var_src` get different values, so it does not merely bound
    /// the shipped statistic — it replaces it with a scale-dependent one.
    /// MEASURED on CID22 alone: 2,895 adjacent-pair inversions against the
    /// revision-1 order. Predicted the opposite before measuring; the
    /// prediction was made from `d/da` at fixed `b`, which is
    /// [`Self::preserves_order`]'s question, not this one.
    ///
    /// Mirrored by `scripts/r6b_arm_delta.py`'s `ARM_RATIO_ONLY`, and pinned
    /// against the arithmetic by
    /// [`tests::only_bounded_excess_depends_on_the_magnitude`].
    #[allow(dead_code)]
    pub(crate) const fn depends_only_on_ratio(self) -> bool {
        !matches!(self, Self::BoundedExcess)
    }

    /// The arm's least upper bound, or `None` when it has none.
    pub(crate) const fn upper_bound(self) -> Option<f64> {
        match self {
            Self::RatioExcess | Self::Log1pExcess => None,
            Self::BoundedExcess | Self::SaturatingExcess => Some(1.0),
            Self::CappedExcess => Some(GAIN_CAP),
        }
    }

    /// Stable lower-case token — the `ZENSIM_HF_GAIN` spelling, and the arm
    /// label in every R6b artefact.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::RatioExcess => "ratio",
            Self::BoundedExcess => "bexcess",
            Self::Log1pExcess => "log1p",
            Self::SaturatingExcess => "satexcess",
            Self::CappedExcess => "cap",
        }
    }
}

/// **The form the active revision selects.**
///
/// Read ONCE per finalize, not per call site — modelled on
/// [`crate::ssim_form::active_luma_form`] and for its stated reason: *"a switch
/// that some call sites honour and others do not is the same defect"*.
///
/// `ZENSIM_HF_GAIN` is a MEASUREMENT override and nothing else. R6b has to fit
/// the monotone linear class against every arm on ONE set of pixels from ONE
/// binary, or the comparison is confounded by a rebuild. No shipping path sets
/// it; a shipping path selects a REVISION, through the equal-byte-length
/// `ZENSIM_FORMULA_REV` that [`crate::ssim_form::active_revision`] owns.
#[inline]
pub(crate) fn active_gain_form() -> HfGainForm {
    use std::sync::OnceLock;
    static FORM: OnceLock<HfGainForm> = OnceLock::new();
    *FORM.get_or_init(|| match std::env::var("ZENSIM_HF_GAIN").as_deref() {
        Ok("ratio") => HfGainForm::RatioExcess,
        Ok("bexcess") => HfGainForm::BoundedExcess,
        Ok("log1p") => HfGainForm::Log1pExcess,
        Ok("satexcess") => HfGainForm::SaturatingExcess,
        Ok("cap") => HfGainForm::CappedExcess,
        _ => HfGainForm::for_revision(crate::ssim_form::active_revision()),
    })
}

/// **`contrast_inc` / `hf_energy_gain`** — the one member with an arm.
///
/// Returns exactly `0.0` whenever `var_src` is at or below [`VAR_SRC_FLOOR`],
/// or whenever `var_dst <= var_src`, **for every arm** — the zero-preservation
/// invariant (gate H4) that keeps the slot a `Difference` form and the identity
/// vector zero.
#[inline]
pub(crate) fn hf_energy_gain(form: HfGainForm, var_src: f64, var_dst: f64) -> f64 {
    // `!(x > y)` rather than `x <= y` DELIBERATELY: it is the negation of the
    // shipped `if var_src > 1e-10 { … } else { 0.0 }`, so a NaN `var_src` takes
    // the zero branch exactly as it always has. `x <= y` would take the other
    // one and divide.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(var_src > VAR_SRC_FLOOR) {
        return 0.0;
    }
    let g = match form {
        // Bit-identical to the expression this replaced: same subtraction of
        // 1.0 from the ratio, same `.max(0.0)`, same order.
        HfGainForm::RatioExcess => (var_dst / var_src - 1.0).max(0.0),
        HfGainForm::BoundedExcess => (var_dst - var_src).max(0.0) / (var_dst + var_src + C_HF_V1),
        HfGainForm::Log1pExcess => (var_dst / var_src - 1.0).max(0.0).ln_1p(),
        HfGainForm::SaturatingExcess => {
            let g = (var_dst / var_src - 1.0).max(0.0);
            g / (g + 1.0)
        }
        // `.max(0.0).min(cap)` rather than `.clamp(0.0, cap)`: the `.max(0.0)`
        // is the SHIPPED expression's own floor, and keeping it visible is what
        // makes "exact below the cap" readable as an edit to one end only.
        #[allow(clippy::manual_clamp)]
        HfGainForm::CappedExcess => (var_dst / var_src - 1.0).max(0.0).min(GAIN_CAP),
    };
    // The boundedness claim, checked rather than asserted in prose — the same
    // shape `ssim_form`'s scalar tail uses. Debug-only; this is a per-cell
    // finalize, not a per-pixel loop, so the cost is 12 checks per image.
    debug_assert!(
        match form.upper_bound() {
            Some(hi) => (0.0..=hi).contains(&g),
            None => g >= 0.0,
        },
        "{} produced {g} outside its declared bound {:?}",
        form.as_str(),
        form.upper_bound()
    );
    debug_assert_eq!(form.bounds_gain(), form.upper_bound().is_some());
    g
}

/// **`var_loss` / `hf_energy_loss`** — bounded `[0, 1]` by construction, no arm.
///
/// Here so the family's gate has one owner; the arithmetic is unchanged and
/// [`tests::loss_members_are_bounded_by_construction`] states the bound as a
/// property rather than a comment.
#[inline]
pub(crate) fn hf_energy_loss(var_src: f64, var_dst: f64) -> f64 {
    if var_src > VAR_SRC_FLOOR {
        (1.0 - var_dst / var_src).max(0.0)
    } else {
        0.0
    }
}

/// **`tex_loss` / `hf_mag_loss`** — the L1 sibling of [`hf_energy_loss`],
/// likewise bounded `[0, 1]` and likewise without an arm.
#[inline]
pub(crate) fn hf_mag_loss(mad_src: f64, mad_dst: f64) -> f64 {
    if mad_src > VAR_SRC_FLOOR {
        (1.0 - mad_dst / mad_src).max(0.0)
    } else {
        0.0
    }
}

/// `∂ hf_energy_gain / ∂ var_dst`, for the attribution density's integrand.
///
/// The map has to differentiate the form the SCORE uses or it describes a
/// metric nobody is running, so the derivative lives beside the form and
/// switches with it. Zero wherever the value is pinned at 0 (`var_dst <=
/// var_src`), matching `AttrCoeffs`'s existing branch, and zero above the cap
/// for [`HfGainForm::CappedExcess`], which is flat there.
// Used by `attribution`'s density integrand, which is `custom-profiles`-gated;
// on a build without that feature it is reachable only from this module's
// finite-difference tests.
#[allow(dead_code)]
#[inline]
pub(crate) fn hf_energy_gain_d_var_dst(form: HfGainForm, var_src: f64, var_dst: f64) -> f64 {
    // Same NaN-preserving negation as `hf_energy_gain`; see its comment.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(var_src > VAR_SRC_FLOOR) || var_dst <= var_src {
        return 0.0;
    }
    match form {
        HfGainForm::RatioExcess => 1.0 / var_src,
        HfGainForm::BoundedExcess => {
            let den = var_dst + var_src + C_HF_V1;
            (2.0 * var_src + C_HF_V1) / (den * den)
        }
        // d/da ln(a/b) = 1/a.
        HfGainForm::Log1pExcess => 1.0 / var_dst,
        // g/(g+1) = 1 - b/a  ⇒  d/da = b/a².
        HfGainForm::SaturatingExcess => var_src / (var_dst * var_dst),
        HfGainForm::CappedExcess => {
            if var_dst / var_src - 1.0 < GAIN_CAP {
                1.0 / var_src
            } else {
                0.0
            }
        }
    }
}

/// The attribution integrand's coefficient, in the ACCUMULATOR's units:
/// `∂ hf_energy_gain / ∂ Σ(dst − μ2)²`.
///
/// [`hf_energy_gain_d_var_dst`] is the mathematical owner; this is the wrapper
/// the density's call site uses, and it exists so the shipped arm keeps its
/// exact spelling. `var_x = Σx·inv_n`, so the chain rule gives
/// `inv_n · ∂f/∂var_dst`, which for [`HfGainForm::RatioExcess`] is
/// `inv_n / var_src` — mathematically `1/Σsrc²`, but not the same f64 rounding.
/// Spelling that arm as the division the site already performed keeps revision
/// 1's density BIT-identical; every other arm goes through the chain rule.
// Used by `attribution`'s density integrand, which is `custom-profiles`-gated;
// on a build without that feature it is reachable only from this module's
// finite-difference tests.
#[allow(dead_code)]
#[inline]
pub(crate) fn hf_energy_gain_d_sum_dst_sq(
    form: HfGainForm,
    hf_sq_src: f64,
    hf_sq_dst: f64,
    inv_n: f64,
) -> f64 {
    match form {
        HfGainForm::RatioExcess => 1.0 / hf_sq_src,
        _ => inv_n * hf_energy_gain_d_var_dst(form, hf_sq_src * inv_n, hf_sq_dst * inv_n),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ARMS: [HfGainForm; 5] = [
        HfGainForm::RatioExcess,
        HfGainForm::BoundedExcess,
        HfGainForm::Log1pExcess,
        HfGainForm::SaturatingExcess,
        HfGainForm::CappedExcess,
    ];

    /// `(var_src, var_dst)` pairs reaching the healthy regime, the gate, and
    /// F17's pathology — the last row is the measured 36,465.7 configuration.
    fn cases() -> Vec<(f64, f64)> {
        let mut v = Vec::new();
        for &b in &[0.0, 1e-12, 1e-10, 1e-9, 1e-4, 1e-2, 0.25, 1.0, 100.0] {
            for &ratio in &[0.0, 0.5, 0.9999, 1.0, 1.0001, 1.5, 2.0, 10.0, 1e3, 36_466.7] {
                v.push((b, b * ratio));
            }
        }
        v.push((1e-9, 3.6e-5)); // var_src just past the gate, real HF in dst
        v
    }

    /// The exact expression the three sites carried, transcribed verbatim —
    /// the control for the extraction.
    fn legacy_gain(var_src: f64, var_dst: f64) -> f64 {
        if var_src > 1e-10 {
            (var_dst / var_src - 1.0).max(0.0)
        } else {
            0.0
        }
    }
    fn legacy_loss(var_src: f64, var_dst: f64) -> f64 {
        if var_src > 1e-10 {
            (1.0 - var_dst / var_src).max(0.0)
        } else {
            0.0
        }
    }

    /// **The extraction gate.** The legacy arm must reproduce the replaced
    /// expression BIT-for-BIT, not merely closely — that is what lets the
    /// three-site rewrite claim to be inert.
    #[test]
    fn legacy_arm_is_bit_identical_to_the_expression_it_replaced() {
        for (b, a) in cases() {
            let got = hf_energy_gain(HfGainForm::RatioExcess, b, a);
            let want = legacy_gain(b, a);
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "gain diverged at var_src={b} var_dst={a}: {got} vs {want}"
            );
            let got_l = hf_energy_loss(b, a);
            let want_l = legacy_loss(b, a);
            assert_eq!(
                got_l.to_bits(),
                want_l.to_bits(),
                "loss diverged at var_src={b} var_dst={a}"
            );
            let got_m = hf_mag_loss(b, a);
            assert_eq!(got_m.to_bits(), legacy_loss(b, a).to_bits());
        }
    }

    /// **F17, stated as a failing property of the shipped form.** If this ever
    /// stops failing for `RatioExcess`, the pathology was fixed elsewhere and
    /// this module's reason to exist has changed.
    #[test]
    fn only_the_ratio_and_log_arms_are_unbounded() {
        // The measured worst configuration: a near-flat source past the gate
        // against a distorted region carrying real HF.
        let (b, a) = (1e-9, 3.6466e-5);
        assert!(
            hf_energy_gain(HfGainForm::RatioExcess, b, a) > 3.6e4,
            "the F17 pathology should reproduce here"
        );
        for form in ARMS {
            let g = hf_energy_gain(form, b, a);
            match form.upper_bound() {
                Some(hi) => assert!(
                    (0.0..=hi).contains(&g),
                    "{form:?} claims bound {hi} but produced {g}"
                ),
                None => assert!(form == HfGainForm::RatioExcess || form == HfGainForm::Log1pExcess),
            }
            assert_eq!(form.bounds_gain(), form.upper_bound().is_some());
        }
        // Log growth is a compression, not a bound — named, not implied.
        assert!(!HfGainForm::Log1pExcess.bounds_gain());
        assert!(hf_energy_gain(HfGainForm::Log1pExcess, b, a) < 11.0);
    }

    /// **H3, structurally.** Every bounding arm stays inside its declared
    /// bound across the whole sweep, not just the headline case.
    #[test]
    fn bounding_arms_respect_their_declared_bound_everywhere() {
        for form in ARMS.iter().copied().filter(|f| f.bounds_gain()) {
            let hi = form.upper_bound().unwrap();
            for (b, a) in cases() {
                let g = hf_energy_gain(form, b, a);
                assert!(
                    g.is_finite() && (0.0..=hi).contains(&g),
                    "{form:?} produced {g} outside [0, {hi}] at var_src={b} var_dst={a}"
                );
            }
        }
    }

    /// **H4 — zero preservation.** Every arm returns EXACTLY 0 wherever the
    /// shipped form does. This is what keeps the identity vector zero and the
    /// slot a `Difference` form under any revision.
    #[test]
    fn every_arm_preserves_the_zeros_of_the_shipped_form() {
        for (b, a) in cases() {
            if legacy_gain(b, a) != 0.0 {
                continue;
            }
            for form in ARMS {
                let g = hf_energy_gain(form, b, a);
                assert_eq!(
                    g.to_bits(),
                    0.0f64.to_bits(),
                    "{form:?} broke zero-preservation at var_src={b} var_dst={a}: {g}"
                );
            }
        }
    }

    /// **H5 — order preservation.** Each arm that claims it is strictly
    /// increasing in `var_dst` above `var_src`; `CappedExcess` claims it is not
    /// and must actually tie above the cap, so the flag is not decorative.
    #[test]
    fn order_preservation_matches_the_declared_flag() {
        let b = 0.25;
        let ratios: Vec<f64> = (0..400).map(|i| 1.0 + (i as f64) * 0.05).collect();
        for form in ARMS {
            let vals: Vec<f64> = ratios
                .iter()
                .map(|r| hf_energy_gain(form, b, b * r))
                .collect();
            let strict = vals.windows(2).all(|w| w[1] > w[0]);
            assert_eq!(
                strict,
                form.preserves_order(),
                "{form:?} declares preserves_order = {} but strict = {strict}",
                form.preserves_order()
            );
        }
        // And the cap actually ties, rather than merely failing strictness for
        // some floating-point reason.
        let hi1 = hf_energy_gain(HfGainForm::CappedExcess, b, b * 5.0);
        let hi2 = hf_energy_gain(HfGainForm::CappedExcess, b, b * 5000.0);
        assert_eq!(hi1, hi2);
        assert_eq!(hi1, GAIN_CAP);
    }

    /// H5's actual subject: which arms are functions of the RATIO alone.
    ///
    /// The arm that is not re-ranks the population even though it is monotone
    /// in `var_dst` at fixed `var_src` — the distinction the first draft of
    /// this module got wrong.
    #[test]
    fn only_bounded_excess_depends_on_the_magnitude() {
        // Same ratio g = 1 (a = 2b), three magnitudes three orders apart.
        let pairs = [(1e-6f64, 2e-6f64), (1e-3, 2e-3), (1.0, 2.0)];
        for form in ARMS {
            let vals: Vec<f64> = pairs
                .iter()
                .map(|&(b, a)| hf_energy_gain(form, b, a))
                .collect();
            let same = vals.windows(2).all(|w| w[0] == w[1]);
            assert_eq!(
                same,
                form.depends_only_on_ratio(),
                "{form:?} declares depends_only_on_ratio = {} but values are {vals:?}",
                form.depends_only_on_ratio()
            );
        }
        // And the magnitude-dependent arm really does invert an order the
        // shipped form fixes: a SMALLER `g` at a larger magnitude outranks a
        // larger `g` at a tiny one.
        let (lo_b, lo_a) = (1.0f64, 1.5f64); //  g = 0.5, large magnitude
        let (hi_b, hi_a) = (1e-5f64, 3e-5f64); // g = 2.0, tiny magnitude
        assert!(
            hf_energy_gain(HfGainForm::RatioExcess, lo_b, lo_a)
                < hf_energy_gain(HfGainForm::RatioExcess, hi_b, hi_a)
        );
        assert!(
            hf_energy_gain(HfGainForm::BoundedExcess, lo_b, lo_a)
                > hf_energy_gain(HfGainForm::BoundedExcess, hi_b, hi_a),
            "BoundedExcess should invert this pair — that IS the H5 failure"
        );
    }

    /// The two derivations that make `SaturatingExcess` constant-free, checked
    /// rather than argued: it is the src↔dst reflection of `hf_energy_loss`,
    /// and it agrees with the shipped form to first order in `g`.
    #[test]
    fn saturating_excess_is_the_reflection_of_the_loss_member() {
        for &(b, a) in &[(0.25, 0.5), (1e-4, 3e-4), (2.0, 2.5), (0.5, 50.0)] {
            let got = hf_energy_gain(HfGainForm::SaturatingExcess, b, a);
            let reflected = hf_energy_loss(a, b); // swap src and dst
            assert!(
                (got - reflected).abs() < 1e-15,
                "reflection identity failed at ({b}, {a}): {got} vs {reflected}"
            );
        }
        // First-order agreement: |arm − g| / g² → 1 as g → 0.
        for &eps in &[1e-3, 1e-4, 1e-5] {
            let (b, a) = (1.0, 1.0 + eps);
            let g = hf_energy_gain(HfGainForm::RatioExcess, b, a);
            let s = hf_energy_gain(HfGainForm::SaturatingExcess, b, a);
            assert!(
                ((g - s) / (g * g) - 1.0).abs() < 1e-2,
                "1st-order agreement failed at eps={eps}: g={g} s={s}"
            );
        }
        // `BoundedExcess` by contrast rescales by ~1/2 even as g → 0 — the
        // reason §11.5 predicts it perturbs healthy cells hardest.
        let (b, a) = (1.0, 1.0 + 1e-5);
        let g = hf_energy_gain(HfGainForm::RatioExcess, b, a);
        let be = hf_energy_gain(HfGainForm::BoundedExcess, b, a);
        assert!(
            (be / g - 0.5).abs() < 1e-3,
            "expected ~g/2, got {be} vs {g}"
        );
    }

    /// The attribution integrand differentiates the form the score uses.
    /// Central differences against each arm, so a future arm cannot land with
    /// a copied derivative.
    #[test]
    fn gain_derivative_matches_a_central_difference_for_every_arm() {
        for form in ARMS {
            for &(b, a) in &[
                (0.25, 0.5),
                (1e-4, 3e-4),
                (2.0, 2.5),
                (1.0, 1.2),
                (1.0, 50.0),
            ] {
                // Skip the cap's own kink, where a derivative is not defined.
                if form == HfGainForm::CappedExcess && ((a / b - 1.0) - GAIN_CAP).abs() < 1e-6 {
                    continue;
                }
                let h = a * 1e-6;
                let fd =
                    (hf_energy_gain(form, b, a + h) - hf_energy_gain(form, b, a - h)) / (2.0 * h);
                let an = hf_energy_gain_d_var_dst(form, b, a);
                let scale = an.abs().max(fd.abs()).max(1e-12);
                assert!(
                    (an - fd).abs() / scale < 1e-4,
                    "{form:?} d/d(var_dst) at ({b}, {a}): analytic {an} vs FD {fd}"
                );
            }
            // Pinned-at-zero region: the value is flat, so the coefficient is 0.
            assert_eq!(hf_energy_gain_d_var_dst(form, 1.0, 0.5), 0.0);
            assert_eq!(hf_energy_gain_d_var_dst(form, 1e-12, 5.0), 0.0);
        }
    }

    /// The density wrapper reproduces the shipped site's spelling BIT-exactly
    /// at revision 1, and is the chain rule at every other arm.
    #[test]
    fn density_coefficient_is_bit_identical_at_rev1_and_chain_rule_elsewhere() {
        for &(sum_src, sum_dst, n) in &[
            (12.5f64, 30.0f64, 4096.0f64),
            (1e-6, 1.0, 65536.0),
            (3.25, 3.26, 1024.0),
        ] {
            let inv_n = 1.0 / n;
            let got = hf_energy_gain_d_sum_dst_sq(HfGainForm::RatioExcess, sum_src, sum_dst, inv_n);
            assert_eq!(
                got.to_bits(),
                (1.0f64 / sum_src).to_bits(),
                "rev1 density coefficient changed spelling"
            );
            for form in ARMS
                .iter()
                .copied()
                .filter(|f| *f != HfGainForm::RatioExcess)
            {
                let want = inv_n * hf_energy_gain_d_var_dst(form, sum_src * inv_n, sum_dst * inv_n);
                let got = hf_energy_gain_d_sum_dst_sq(form, sum_src, sum_dst, inv_n);
                assert_eq!(got.to_bits(), want.to_bits(), "{form:?}");
            }
        }
    }

    /// The `loss` members' bound is a property of the algebra, not a clamp —
    /// stated as a test so a future edit that breaks it fails loudly.
    #[test]
    fn loss_members_are_bounded_by_construction() {
        for (b, a) in cases() {
            let l = hf_energy_loss(b, a);
            let m = hf_mag_loss(b, a);
            assert!((0.0..=1.0).contains(&l), "hf_energy_loss = {l}");
            assert!((0.0..=1.0).contains(&m), "hf_mag_loss = {m}");
        }
    }

    /// Every arm token round-trips, and the tokens are the `ZENSIM_HF_GAIN`
    /// spellings the R6b scripts use.
    #[test]
    fn arm_tokens_are_unique_and_stable() {
        let mut seen = std::collections::BTreeSet::new();
        for form in ARMS {
            assert!(
                seen.insert(form.as_str()),
                "duplicate token {}",
                form.as_str()
            );
        }
        assert_eq!(
            seen.iter().copied().collect::<Vec<_>>(),
            ["bexcess", "cap", "log1p", "ratio", "satexcess"]
        );
    }

    /// Revision 1 is the shipped form, and revision 2 resolves through the one
    /// named constant.
    #[test]
    fn revision_selection_goes_through_the_named_arm() {
        assert_eq!(
            HfGainForm::for_revision(FormulaRevision::Rev1),
            HfGainForm::RatioExcess
        );
        assert_eq!(
            HfGainForm::for_revision(FormulaRevision::Rev2),
            HfGainForm::REV2_HFGAIN
        );
    }

    #[cfg(feature = "feature-regime-v2")]
    #[test]
    fn v1_c_hf_matches_the_v2_owner() {
        assert_eq!(C_HF_V1, crate::feature_v2::C_HF);
    }
}
