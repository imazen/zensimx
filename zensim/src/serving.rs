//! **THE servability census** — the roster of profiles this build ships, and
//! the gates that every one of them serves its own bake.
//!
//! ## Why it is here and not in `feature_plan`
//!
//! It used to be `feature_plan::servability_census`, which is gated on
//! `feature-regime-v2`. So the one instrument that asks *"can this build serve
//! what it ships?"* did not exist in the builds that could not — and on
//! 2026-09-06 that is exactly what happened: the shipped `A`, `B`, `BHdr` and
//! `D` bakes went DENSE (`cb2f412d`) while the gather that serves a dense
//! declaration lived behind `feature-regime-v2`, so every
//! `--no-default-features` consumer silently got the positional-prefix
//! reading. A census that cannot run in the broken configuration is not a
//! census.
//!
//! Everything here needs only [`ZensimProfile`], [`Zensim::compute`] and
//! `mlp::Model`. The gates that need a `Plan` (revision agreement, the
//! id-space-vs-`Plan::for_bake` cross-check) stay in `feature_plan` and
//! import the roster from here, so there is still exactly one roster.

use crate::ZensimProfile;

/// A deterministic non-identical SDR pair. 64×64 is the pyramid minimum,
/// so it exercises the real 4-scale walk with no reflect-pad.
pub(crate) fn pair(w: usize, h: usize) -> (Vec<[u8; 3]>, Vec<[u8; 3]>) {
    let mut src = vec![[0u8; 3]; w * h];
    let mut dst = vec![[0u8; 3]; w * h];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let v = ((x * 7 + y * 13) % 251) as u8;
            src[i] = [v, v.wrapping_add(40), v.wrapping_mul(3)];
            // A structured, non-trivial distortion: quantize + shift.
            dst[i] = [
                v & 0xF0,
                v.wrapping_add(37),
                v.wrapping_mul(3).wrapping_sub(9),
            ];
        }
    }
    (src, dst)
}

/// The same reference as [`pair`], distorted by dropping ONE bit per channel.
///
/// A near-invisible distortion, and the one that first exposed the dense
/// mis-serving in `zensim-wasm-tests` — a score near the top of the range is
/// where a wrong feature vector is least likely to look wrong. Kept next to
/// [`pair`] rather than reaching for `tests/common/distortions.rs`, which is
/// integration-test scaffolding a `src` module cannot see.
pub(crate) fn pair_lsb(w: usize, h: usize) -> (Vec<[u8; 3]>, Vec<[u8; 3]>) {
    let (src, _) = pair(w, h);
    let dst = src
        .iter()
        .map(|p| [p[0] & 0xFE, p[1] & 0xFE, p[2] & 0xFE])
        .collect();
    (src, dst)
}

/// Every profile this build ships, with its name.
///
/// `pub(crate)` so the layout census (`feature_layout::tests`) and the plan
/// gates (`feature_plan::servability_census`) enumerate the SAME roster rather
/// than keeping second lists that could drift past a feature flag — the roster
/// is `#[cfg]`-dependent, which is exactly the kind of list a copy gets wrong.
#[cfg_attr(
    not(any(feature = "deprecated-profiles", feature = "candidate-profiles")),
    allow(unused_mut)
)]
pub(crate) fn shipped_profiles() -> Vec<(&'static str, ZensimProfile)> {
    let mut v: Vec<(&'static str, ZensimProfile)> = vec![
        ("B", ZensimProfile::B),
        ("BHdr", ZensimProfile::BHdr),
        ("PreviewV0_1", ZensimProfile::PreviewV0_1),
        ("PreviewV0_2", ZensimProfile::PreviewV0_2),
    ];
    #[cfg(feature = "deprecated-profiles")]
    {
        // `A` is deprecated but SHIPPED, and the contract is about what
        // ships. Censusing it is the point.
        #[allow(deprecated)]
        v.push(("A", ZensimProfile::A));
    }
    #[cfg(feature = "candidate-profiles")]
    {
        v.push(("C", ZensimProfile::C));
        v.push(("CHdr", ZensimProfile::CHdr));
        v.push(("D", ZensimProfile::D));
    }
    v
}

/// The number of (profile, bake) pairs [`shipped_profiles`] contributes a
/// SCORING bake for, under the ACTIVE feature set — the floor the layout and
/// plan censuses check against so a census that silently sees nothing still
/// fails loud.
///
/// `PreviewV0_1` / `PreviewV0_2` are always in the roster but carry no MLP
/// bake (`scoring_bake_bytes()` yields nothing for them), so the floor tracks
/// the SAME `#[cfg]` gates as `shipped_profiles` itself — `A` (1 bake) behind
/// `deprecated-profiles`, `C`+`CHdr`+`D` (3 bakes) behind
/// `candidate-profiles`. A bare constant here is precisely the "second list
/// that could drift past a feature flag" `shipped_profiles`'s own doc warns
/// about: found 2026-09-06 when the CI permutation matrix's `--features
/// feature-regime-v2` cell (neither extra feature on) hit a hardcoded `>= 5`
/// that only B+BHdr (2 bakes) can ever satisfy without them.
pub(crate) fn expected_min_bake_count() -> usize {
    2 // B, BHdr — unconditional
        + usize::from(cfg!(feature = "deprecated-profiles")) // A
        + 3 * usize::from(cfg!(feature = "candidate-profiles")) // C, CHdr, D
}

/// How many of [`expected_min_bake_count`]'s bakes are DENSE
/// (`zentrain.feature_ids`-declared) under the active feature set.
///
/// Every scoring bake in the current roster declares its consumed IDs.
pub(crate) fn expected_min_dense_count() -> usize {
    expected_min_bake_count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{RgbSlice, Zensim};

    /// One census row.
    #[derive(Debug)]
    struct Row {
        name: String,
        declared_width: usize,
        outcome: Result<usize, String>,
    }

    fn census_profile(name: &str, p: ZensimProfile) -> Row {
        let (w, h) = (64usize, 64usize);
        let (src, dst) = pair(w, h);
        let declared_width = p
            .params()
            .scoring_bake_bytes()
            .filter_map(|b| crate::mlp::Model::from_bytes(b).ok())
            .map(|m| crate::feature_layout::declared_layout(&m).walk_width())
            .max()
            .unwrap_or(0);
        let z = Zensim::new(p);
        let outcome = z
            .compute(&RgbSlice::new(&src, w, h), &RgbSlice::new(&dst, w, h))
            .map(|r| r.features().len())
            .map_err(|e| format!("{e:?}"));
        Row {
            name: name.to_string(),
            declared_width,
            outcome,
        }
    }

    /// **THE contract gate.** Every shipped profile serves a non-identical
    /// pair, and emits at least the width its widest bake declares.
    ///
    /// The census REPORT is printed on every run (`--nocapture` to see it) so
    /// "what cannot be served today, and why" is a measurement rather than an
    /// inference from reading `profile.rs`.
    #[test]
    fn every_shipped_profile_is_servable() {
        let rows: Vec<Row> = shipped_profiles()
            .into_iter()
            .map(|(n, p)| census_profile(n, p))
            .collect();
        // The PRE-increment-1 rule, stated exactly as `prep_bake_input_f32`
        // enforced it: the extraction emitted at most the v1 width, so a bake
        // was servable iff `caller_input_width() <= v1_width + 4` (the `+4` is
        // the optional size-axis augmentation). Reproduced here rather than
        // recalled, so the BEFORE column of the census is derived from the
        // removed condition rather than from memory.
        let v1_width = crate::NUM_SCALES
            * 3
            * (crate::metric::FEATURES_PER_CHANNEL_EXTENDED
                + crate::metric::FEATURES_PER_CHANNEL_IW);
        let old_rule = |declared: usize| declared <= v1_width + 4;
        let mut report = String::from(
            "\nSERVABILITY CENSUS — shipped profiles\n\
             (BEFORE = the removed `prep_bake_input_f32` rule: declared <= v1_width + 4)\n\
             profile      declared  emitted  BEFORE  NOW\n",
        );
        let (mut before_ok, mut after_ok) = (0usize, 0usize);
        let mut unservable = Vec::new();
        for r in &rows {
            if old_rule(r.declared_width) {
                before_ok += 1;
            }
            match &r.outcome {
                Ok(emitted) => {
                    after_ok += 1;
                    report.push_str(&format!(
                        "  {:<10} {:>8}  {:>7}  {:>6}  SERVED\n",
                        r.name,
                        r.declared_width,
                        emitted,
                        if old_rule(r.declared_width) {
                            "served"
                        } else {
                            "REFUSED"
                        }
                    ));
                    assert!(
                        *emitted >= r.declared_width,
                        "{}: emitted {emitted} < declared {}",
                        r.name,
                        r.declared_width
                    );
                }
                Err(e) => {
                    report.push_str(&format!(
                        "  {:<10} {:>8}  {:>7}  {:>6}  NOT SERVED: {e}\n",
                        r.name,
                        r.declared_width,
                        "-",
                        if old_rule(r.declared_width) {
                            "served"
                        } else {
                            "REFUSED"
                        }
                    ));
                    unservable.push(format!("{} ({e})", r.name));
                }
            }
        }
        report.push_str(&format!(
            "  ---- servable: {before_ok}/{} BEFORE, {after_ok}/{} NOW\n",
            rows.len(),
            rows.len()
        ));
        println!("{report}");
        assert!(
            unservable.is_empty(),
            "{} shipped profile(s) cannot be served: {}\n{report}",
            unservable.len(),
            unservable.join(", ")
        );
    }

    /// **THE cross-feature-set gate, expressed in-process.**
    ///
    /// Every shipped profile's score on two fixed 64×64 pairs, pinned. This
    /// test compiles and runs under EVERY cargo feature permutation, which is
    /// the whole point: the failure it catches is invisible from inside any
    /// single build, because a wrong feature vector produces a plausible
    /// number and no error.
    ///
    /// ## The tolerance, derived — and the derivation that was WRONG first
    ///
    /// Not bit-exact, because these values legitimately move across the
    /// arithmetic classes this crate is built for. The first version of this
    /// gate set `TOL = 1e-2` from a population of x86-64 builds only
    /// (AVX-512+threads against neither, max 1.048e-5) and **i686 CI turned it
    /// red on the first push**: `PreviewV0_1`'s single-LSB cell read
    /// 98.378873 against the pinned 98.394763, a delta of 1.589e-2. That was a
    /// defect in the DERIVATION, not in a kernel — the population was one
    /// architecture wide.
    ///
    /// MEASURED 2026-09-06 over **160 cells** (8 profiles × 4 geometries × 5
    /// distortion strengths) on the four arms CI actually runs
    /// (`benchmarks/dense_serving_ungate_2026-09-06.md` §2d):
    ///
    /// | arms | max \|Δ\| | cells > 1e-2 |
    /// |---|--:|--:|
    /// | x86-64 AVX-512 vs AVX2 | **2.1411e-5** | 0 |
    /// | x86-64 (either) vs **i686 scalar** | **2.8221e-2** | 9 |
    /// | x86-64 (either) vs **wasm32 simd128** | **2.8221e-2** | 9 |
    /// | i686 scalar vs wasm32 simd128 | **0** (bit-identical, 160/160) | 0 |
    ///
    /// Two arithmetic classes, not four: every backend with a FUSED
    /// multiply-add (AVX-512, AVX2, and — by CI evidence — NEON) agrees to
    /// 2.1e-5, and magetypes' scalar + wasm128 backends implement `mul_add` as
    /// an UNFUSED `a*b+c` (`e1324192` measured the same 1-ULP split on the ring
    /// tests) and are bit-identical to each other. On a near-identical pair the
    /// dissimilarity features sit at the f32 cancellation floor, so a 1-ULP
    /// difference in form becomes an O(1) RELATIVE difference there, and
    /// `100 − 18·d^0.7` has unbounded slope as `d → 0`. That is the whole
    /// mechanism, and it is a KNOWN, deliberately-unfixed upstream property —
    /// the kernel-side fusing fix was tried in `e1324192` and rejected because
    /// it shifted `sigma_sq`/`sigma12` and broke
    /// `cross_platform::pixel_format_equivalence`.
    ///
    /// So the bar is set from two populations, **both measured, each on the
    /// population that actually bounds it**:
    ///
    /// * **noise** — cross-arithmetic-class spread, max **2.8221e-2** over the
    ///   160 cells above. This gate's OWN 16 cells top out at 1.5890e-2; the
    ///   wider number is used deliberately, because 16 cells under-sample a
    ///   floor an unseen platform could push on.
    /// * **signal** — the smallest movement a real mis-serve produces ON THIS
    ///   GATE'S OWN CELLS: **0.945195** points. MEASURED directly, by bypassing
    ///   the gather in a local build and re-running this test: all 8
    ///   dense-profile cells moved, from `B`/single-LSB **0.945195** to
    ///   `D`/quantize+shift **172.261448** (`D` inverts, to −155.77). Measured
    ///   here, not inferred from the serving-matrix population.
    ///
    /// `TOL = 0.16` is their GEOMETRIC MIDPOINT
    /// (`sqrt(2.8221e-2 · 0.945195) = 0.1633`, to two significant figures) —
    /// **5.67× above the conservative noise (10.07× above this gate's own
    /// cells) and 5.91× below the smallest real defect**. Balanced in log
    /// space, every input measured on a stated population, rather than a
    /// number chosen to make CI pass.
    ///
    /// The tight complement lives in `scripts/serving_matrix.sh`, which
    /// requires **bit-exact** agreement across feature sets WITHIN one
    /// architecture. Loose across classes, exact within one.
    ///
    /// If a deliberate change moves a shipped score, this test FAILS and the
    /// pin is updated in the same commit with the measurement that justifies
    /// it. That is the intended workflow — never widen `TOL` to make a build
    /// pass; widen it only by re-deriving it on a larger MEASURED population,
    /// which is what happened here, once, and is recorded as such.
    #[test]
    fn every_shipped_profile_scores_its_pinned_value() {
        /// See the doc comment: the geometric midpoint of the measured
        /// cross-class noise (2.8221e-2) and the smallest real defect measured
        /// on these very cells (0.945195).
        const TOL: f64 = 0.16;
        // (profile, quantize+shift pair, single-LSB pair). Captured from the
        // default build at 2026-09-06 with the dense gather live. B was
        // remeasured on the owner-approved diff+frag-820 rotation (2026-10-19).
        let pins: &[(&str, f64, f64)] = &[
            ("A", 38.570_207_155_687, 93.833_130_065_237),
            ("B", 22.196_183_902_237_486, 97.800_779_621_166_41),
            ("BHdr", 60.136_845_729_586, 95.561_810_768_036),
            ("PreviewV0_1", 40.823_327_058_061, 98.394_762_958_421),
            ("PreviewV0_2", 46.958_140_610_559, 98.185_741_680_479),
            // C saturates the single-LSB cell at exactly 100 — noted, not
            // hidden: that cell cannot discriminate for C, and its
            // quantize+shift cell carries the profile here. The September 7
            // activity correction moves C/CHdr to canonical training features.
            ("C", 38.633_627_095_004, 100.0),
            ("CHdr", 66.079_811_960_673, 97.132_037_328_304),
            ("D", 16.490_362_421_318, 97.179_188_906_774),
        ];
        let (w, h) = (64usize, 64usize);
        let (src, dst) = pair(w, h);
        let (lsrc, ldst) = pair_lsb(w, h);
        let mut checked = 0usize;
        let mut fails = Vec::new();
        for (name, p) in shipped_profiles() {
            let Some(&(_, want_strong, want_lsb)) = pins.iter().find(|(n, _, _)| *n == name) else {
                panic!(
                    "{name} is in the shipped roster with no pinned score — add one \
                     (measured, from the default build), never skip a profile here"
                );
            };
            let z = Zensim::new(p);
            for (label, r, d, want) in [
                ("quantize+shift", &src, &dst, want_strong),
                ("single-LSB", &lsrc, &ldst, want_lsb),
            ] {
                let got = z
                    .compute(&RgbSlice::new(r, w, h), &RgbSlice::new(d, w, h))
                    .unwrap_or_else(|e| panic!("{name}/{label}: shipped profile refused: {e}"))
                    .score();
                eprintln!("SERVING PIN {name}/{label}: {got:.17}");
                checked += 1;
                if (got - want).abs() > TOL {
                    fails.push(format!(
                        "{name}/{label}: {got:.12} vs pinned {want:.12} (delta {:.6})",
                        got - want
                    ));
                }
            }
        }
        assert_eq!(
            checked,
            2 * shipped_profiles().len(),
            "every profile in the roster must be scored on both pairs"
        );
        assert!(
            fails.is_empty(),
            "{} shipped score(s) moved — a mis-served bake looks exactly like this, \
             so investigate BEFORE re-pinning:\n  {}",
            fails.len(),
            fails.join("\n  ")
        );
    }

    /// The cross-BUILD gate's example must carry the SAME roster, gated the
    /// SAME way.
    ///
    /// `zensim/examples/serving_matrix.rs` cannot call [`shipped_profiles`] —
    /// it is `pub(crate)`, and exporting a test fixture would be a public-API
    /// delta — so it keeps its own `#[cfg]`-dependent list. That is the drift
    /// [`shipped_profiles`]'s own doc warns about, and the consequence is
    /// specific: a profile the example forgets, or gates differently, is a
    /// profile `scripts/serving_matrix.sh` silently stops diffing across
    /// builds.
    ///
    /// So both bodies are read with `include_str!` and reduced to the same
    /// thing: the set of `(gating feature, profile)` pairs. A textual check is
    /// weaker than calling the function and strictly stronger than nothing —
    /// and it is `#[cfg]`-INDEPENDENT, which a runtime comparison of the two
    /// lists could never be (the example's source text names `A` whether or
    /// not `deprecated-profiles` is on).
    #[test]
    fn the_serving_matrix_example_carries_the_same_roster() {
        /// `(gating feature or "" for unconditional, profile name)` for every
        /// `ZensimProfile::X` named in `body`.
        ///
        /// Both bodies have the same shape — an unconditional `vec![…]`
        /// followed by `#[cfg(feature = "…")] { … }` blocks and nothing after
        /// — so splitting on the cfg marker recovers the gate for each name.
        fn pairs(body: &str) -> Vec<(String, String)> {
            let mut out = Vec::new();
            let mut push_names = |gate: &str, seg: &str| {
                for chunk in seg.split("ZensimProfile::").skip(1) {
                    let name: String = chunk
                        .chars()
                        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                        .collect();
                    if !name.is_empty() {
                        out.push((gate.to_string(), name));
                    }
                }
            };
            let mut segs = body.split("#[cfg(feature = \"");
            push_names("", segs.next().unwrap_or(""));
            for seg in segs {
                let (gate, rest) = seg.split_once("\")]").unwrap_or(("?", seg));
                push_names(gate, rest);
            }
            out.sort();
            out
        }

        const EXAMPLE: &str = include_str!("../examples/serving_matrix.rs");
        const OWNER: &str = include_str!("serving.rs");
        // Only each function's BODY, so a name in prose does not count.
        let body_of = |src: &str, sig: &str| -> String {
            src.split_once(sig)
                .and_then(|(_, rest)| rest.split_once("\n}"))
                .map(|(body, _)| body.to_string())
                .unwrap_or_else(|| panic!("could not find `{sig}`"))
        };
        let example = pairs(&body_of(EXAMPLE, "fn profiles()"));
        let owner = pairs(&body_of(OWNER, "pub(crate) fn shipped_profiles()"));
        assert!(
            example.len() >= 8,
            "the scan must SEE the roster, saw {example:?} — if `profiles()` was \
             reshaped, fix this scan rather than letting it pass"
        );
        assert_eq!(
            example, owner,
            "serving_matrix.rs's roster and `shipped_profiles` disagree; the \
             cross-build gate would silently stop diffing the difference"
        );
    }

    /// **A published crate that does not compile serves nothing.**
    ///
    /// `zensim/Cargo.toml`'s `include` is an ALLOWLIST, so a
    /// `include_bytes!("../weights/…")` whose file is not listed produces a
    /// `.crate` that fails to build — and nothing in a workspace checkout can
    /// notice, because the file is right there on disk.
    ///
    /// MEASURED 2026-09-06: six files were missing. `cb2f412d` pointed A / B /
    /// BHdr / D at `*_byid_2026-09-06.bin` and `c_sdr_purity944` /
    /// `c_hdr_l1t1944` replaced `c_sdr_mlp944_corrmix`, while `include` still
    /// named the superseded paths; `cargo package --list` shipped seven
    /// weights of which one was even referenced.
    ///
    /// Reads both files through `include_str!`, so the gate needs no
    /// filesystem and runs anywhere the crate builds.
    #[test]
    fn every_included_bake_is_packaged() {
        const MANIFEST: &str = include_str!("../Cargo.toml");
        // Every `src` file that embeds weight bytes. Listed rather than
        // globbed, because a test with no filesystem cannot glob — a new
        // embedder adds a line here. The `checked >= 7` floor below is what
        // catches this list going stale in the other direction (a scan that
        // silently matches nothing).
        const SOURCES: &[(&str, &str)] = &[
            ("profile.rs", include_str!("profile.rs")),
            ("metric.rs", include_str!("metric.rs")),
            ("corruption_head.rs", include_str!("corruption_head.rs")),
        ];
        let include_block = MANIFEST
            .split_once("\ninclude = [")
            .and_then(|(_, rest)| rest.split_once("\n]"))
            .map(|(block, _)| block)
            .expect("Cargo.toml must carry an `include = [ .. ]` allowlist");
        let mut missing = Vec::new();
        let mut checked = 0usize;
        for (file, text) in SOURCES {
            for chunk in text.split("include_bytes!(\"../weights/").skip(1) {
                let Some((path, _)) = chunk.split_once('"') else {
                    continue;
                };
                checked += 1;
                if !include_block.contains(&format!("\"weights/{path}\"")) {
                    missing.push(format!("{file}: weights/{path}"));
                }
            }
        }
        assert!(
            checked >= 7,
            "the gate must SEE the embeds, saw {checked} — if `include_bytes!` \
             moved or was reformatted, fix this scan rather than letting it pass"
        );
        assert!(
            missing.is_empty(),
            "{} embedded weight file(s) are NOT in Cargo.toml's `include` \
             allowlist, so `cargo package` produces a crate that cannot \
             compile:\n  {}",
            missing.len(),
            missing.join("\n  ")
        );
    }

    /// The dense declaration must be READ in this build, whatever features are
    /// on — the structural half of the pinned-score gate.
    ///
    /// A negative control rides along: for every dense bake, the gathered
    /// vector must actually DIFFER from the positional prefix the old code
    /// took, so "the gather ran" is distinguishable from "the gather is a
    /// no-op here".
    #[test]
    fn dense_bakes_resolve_to_a_dense_layout_and_the_gather_is_not_a_no_op() {
        let walk: Vec<f64> = (0..crate::feature_defs::full_width(crate::NUM_SCALES))
            .map(|i| i as f64 + 0.5)
            .collect();
        let mut dense = 0usize;
        for (name, p) in shipped_profiles() {
            for bytes in p.params().scoring_bake_bytes() {
                let Ok(m) = crate::mlp::Model::from_bytes(bytes) else {
                    continue;
                };
                let layout = crate::feature_layout::declared_layout(&m);
                if layout.is_identity() {
                    continue;
                }
                dense += 1;
                assert!(
                    crate::declared_feature_ids(&m).is_some(),
                    "{name}: dense layout without an explicit id declaration"
                );
                let mut gathered = Vec::new();
                layout.gather(&walk, &mut gathered);
                let prefix = &walk[..layout.width()];
                assert_ne!(
                    gathered.as_slice(),
                    prefix,
                    "{name}: the dense gather equals the positional prefix, so this gate \
                     cannot tell a served bake from a mis-served one"
                );
            }
        }
        assert_eq!(
            dense,
            expected_min_dense_count(),
            "expected exactly the dense bakes this feature set ships"
        );
    }
}
