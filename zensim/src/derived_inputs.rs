//! Derived model inputs declared in bake metadata (E33 `fx1`).
//!
//! A bake normally feeds its declared feature ids (`zentrain.feature_ids`)
//! straight into layer 0, one id per input. A bake carrying
//! [`DERIVED_INPUTS_KEY`] instead lists its model inputs explicitly, over the
//! declared ids:
//!
//! ```text
//! zensim-derived-inputs v1
//! in 13
//! ...
//! product 13 422
//! ...
//! ```
//!
//! * `in <id>` feeds one declared id unchanged.
//! * `product <a> <b>` feeds the IEEE f32 product of the two declared ids'
//!   f32 values. `a` must be a [`Form::Difference`] slot and `b` a
//!   [`Form::ReferenceOnly`] slot at the SAME scale and channel, so a
//!   reference-only value can only scale a difference and is exactly `±0`
//!   whenever the difference is (E33 registration §5.1, §7 E4).
//!
//! The declared ids are the bake's READ SET (what the planner extracts); the
//! derived list is the model's input order. Every declared id must be read by
//! at least one entry and the entry count must equal the model's caller width.
//!
//! **Fail-closed on older runtimes.** Before this module, `BakeScorer`
//! refused any bake whose declared id count differed from its caller width.
//! A derived bake always differs (products add inputs), so a runtime that
//! does not know this key refuses it instead of mis-serving it.
//!
//! [`Form::Difference`]: crate::feature_defs::Form::Difference
//! [`Form::ReferenceOnly`]: crate::feature_defs::Form::ReferenceOnly

use crate::feature_set_id::SlotSet;
pub(crate) mod shaping;
use shaping::{DamageShape, ReferenceGate};

/// The metadata key. Its value is utf8 text in the format above.
pub(crate) const DERIVED_INPUTS_KEY: &str = "zensim.derived_inputs";
const HEADER: &str = "zensim-derived-inputs v1";

/// One model input, by position in the bake's dense declared layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum DerivedInput {
    Direct(u32),
    Product(u32, u32),
    Shaped(u32, DamageShape),
    ShapedProduct(u32, u32, DamageShape),
    Separate(u32, u32, DamageShape, ReferenceGate),
}

/// A validated derived-input declaration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DerivedInputs {
    inputs: Vec<DerivedInput>,
    gather_width: usize,
}

impl DerivedInputs {
    /// Parse and validate `text` against the bake's declared ids.
    pub(crate) fn parse(
        text: &str,
        declared: &SlotSet,
        n_scales: usize,
    ) -> Result<Self, &'static str> {
        use crate::feature_defs::{Form, def_at};
        let ids: Vec<usize> = declared.iter_slots().collect();
        let pos_of = |tok: &str| -> Result<u32, &'static str> {
            let id: usize = tok
                .parse()
                .map_err(|_| "derived inputs: unparseable feature id")?;
            ids.binary_search(&id)
                .map(|p| p as u32)
                .map_err(|_| "derived inputs: id is not a declared feature id")
        };
        let mut lines = text.lines().map(str::trim).filter(|l| !l.is_empty());
        let research = match lines.next() {
            Some(HEADER) => false,
            Some("zensim-derived-inputs research-shaping-v1") => true,
            _ => return Err("derived inputs: unknown header"),
        };
        let mut inputs = Vec::new();
        let mut used = vec![false; ids.len()];
        for line in lines {
            let toks: Vec<&str> = line.split_ascii_whitespace().collect();
            let entry = match toks.as_slice() {
                ["in", a] => DerivedInput::Direct(pos_of(a)?),
                ["shape", a, params @ ..] if research => {
                    let pa = pos_of(a)?;
                    let da = def_at(ids[pa as usize], n_scales)
                        .ok_or("derived inputs: id outside the registry")?;
                    if da.signal.form != Form::Difference {
                        return Err("shape: direct input must be Difference");
                    }
                    DerivedInput::Shaped(pa, DamageShape::parse(params)?)
                }
                [
                    op @ ("product" | "shape-product" | "separate"),
                    a,
                    b,
                    params @ ..,
                ] => {
                    let (pa, pb) = (pos_of(a)?, pos_of(b)?);
                    let da = def_at(ids[pa as usize], n_scales)
                        .ok_or("derived inputs: id outside the registry")?;
                    let db = def_at(ids[pb as usize], n_scales)
                        .ok_or("derived inputs: id outside the registry")?;
                    if da.signal.form != Form::Difference {
                        return Err("derived inputs: product factor a must be a Difference slot");
                    }
                    if db.signal.form != Form::ReferenceOnly {
                        return Err(
                            "derived inputs: product factor b must be a ReferenceOnly slot",
                        );
                    }
                    if da.scale != db.scale || da.channel != db.channel {
                        return Err("derived inputs: product factors must share scale and channel");
                    }
                    match *op {
                        "product" if params.is_empty() => DerivedInput::Product(pa, pb),
                        "shape-product" if research => {
                            DerivedInput::ShapedProduct(pa, pb, DamageShape::parse(params)?)
                        }
                        "separate" if research && params.len() == 8 => DerivedInput::Separate(
                            pa,
                            pb,
                            DamageShape::parse(&params[..5])?,
                            ReferenceGate::parse(&params[5..])?,
                        ),
                        _ => return Err("derived inputs: unknown entry"),
                    }
                }
                _ => return Err("derived inputs: unknown entry"),
            };
            match entry {
                DerivedInput::Direct(p) | DerivedInput::Shaped(p, _) => used[p as usize] = true,
                DerivedInput::Product(a, b)
                | DerivedInput::ShapedProduct(a, b, _)
                | DerivedInput::Separate(a, b, _, _) => {
                    used[a as usize] = true;
                    used[b as usize] = true;
                }
            }
            inputs.push(entry);
        }
        if inputs.is_empty() {
            return Err("derived inputs: no entries");
        }
        let mut sorted = inputs.clone();
        sorted.sort_unstable();
        if sorted.windows(2).any(|w| w[0] == w[1]) {
            return Err("derived inputs: duplicate entry");
        }
        if used.iter().any(|u| !u) {
            return Err("derived inputs: a declared id is read by no entry");
        }
        Ok(Self {
            inputs,
            gather_width: ids.len(),
        })
    }

    /// Validated input positions for the exact score chain rule.
    pub(crate) fn entries(&self) -> &[DerivedInput] {
        &self.inputs
    }

    /// Model (caller) input width: one per entry.
    pub(crate) fn model_width(&self) -> usize {
        self.inputs.len()
    }

    /// Width of the gathered declared layout these entries index.
    pub(crate) fn gather_width(&self) -> usize {
        self.gather_width
    }

    /// The declared feature ids the LIVE model inputs read: a live `in` reads
    /// its id, a live `product` reads both factors. `declared` is the bake's
    /// ascending declared id list; `live` is indexed by model input.
    /// Only the feature planner (`feature-regime-v2`) asks.
    pub(crate) fn read_slots(&self, declared: &[usize], live: &[bool]) -> Option<SlotSet> {
        if live.len() != self.inputs.len() || declared.len() != self.gather_width {
            return None;
        }
        let mut ids = Vec::new();
        for (entry, &on) in self.inputs.iter().zip(live) {
            if !on {
                continue;
            }
            match *entry {
                DerivedInput::Direct(p) | DerivedInput::Shaped(p, _) => {
                    ids.push(declared[p as usize])
                }
                DerivedInput::Product(a, b)
                | DerivedInput::ShapedProduct(a, b, _)
                | DerivedInput::Separate(a, b, _, _) => {
                    ids.push(declared[a as usize]);
                    ids.push(declared[b as usize]);
                }
            }
        }
        Some(SlotSet::from_slots(ids))
    }

    /// Build the model's f32 input row from a gathered declared-layout row.
    ///
    /// Each value is narrowed to f32 first, exactly as the plain path's
    /// `prep_bake_input_f32` does; a product is the IEEE f32 multiply of the
    /// two narrowed factors (registration §5.1 "Arithmetic").
    pub(crate) fn fill_f32(&self, gathered: &[f64], out: &mut Vec<f32>) {
        debug_assert_eq!(gathered.len(), self.gather_width);
        out.clear();
        out.extend(self.inputs.iter().map(|e| match *e {
            DerivedInput::Direct(p) => gathered[p as usize] as f32,
            DerivedInput::Shaped(p, shape) => shape.eval(gathered[p as usize] as f32),
            DerivedInput::ShapedProduct(a, b, shape) => {
                shape.eval(gathered[a as usize] as f32 * gathered[b as usize] as f32)
            }
            DerivedInput::Separate(a, b, shape, gate) => {
                shape.eval(gathered[a as usize] as f32) * gate.eval(gathered[b as usize] as f32)
            }
            DerivedInput::Product(a, b) => {
                gathered[a as usize] as f32 * gathered[b as usize] as f32
            }
        }));
    }
}

/// The bake's derived-input declaration, validated against its declared ids
/// and caller width. `Ok(None)` when the key is absent.
pub(crate) fn from_model(model: &crate::mlp::Model) -> Result<Option<DerivedInputs>, &'static str> {
    let metadata = model.metadata();
    if metadata.get(DERIVED_INPUTS_KEY).is_none() {
        return Ok(None);
    }
    let text = metadata
        .get_utf8(DERIVED_INPUTS_KEY)
        .map_err(|_| "derived inputs: not utf8")?;
    let declared = crate::feature_layout::declared_ids(model)
        .ok_or("derived inputs require a valid zentrain.feature_ids declaration")?;
    let parsed = DerivedInputs::parse(text, &declared, crate::NUM_SCALES)?;
    if parsed.model_width() != model.caller_input_width() {
        return Err("derived inputs: entry count differs from the model input width");
    }
    if let Some(ts) = model.feature_transforms() {
        use zenpredict::FeatureTransform::*;
        let spans = crate::fold_engine::caller_col_spans(model, model.n_inputs())
            .ok_or("derived inputs: transform arities do not tile layer zero")?;
        if spans.len() != parsed.model_width() || spans.iter().any(|&(a, b)| b != a + 1) {
            return Err("derived inputs: transforms must preserve scalar caller arity");
        }
        for (i, &t) in ts.iter().enumerate() {
            let p = model
                .feature_transform_params()
                .map_or(&[][..], |p| p[i].as_slice());
            let valid = p.iter().all(|v| v.is_finite())
                && match t {
                    Identity | Log1p | SignedLog1p => p.is_empty(),
                    YeoJohnson => p.len() == 1,
                    SoftClip => p.len() == 2 && p[0] > 0. && p[1] > 0.,
                    SoftSign => p.len() == 1 && p[0] > 0.,
                    _ => false,
                };
            if !valid || t.apply_with_params(0., p) != 0. {
                return Err(
                    "derived inputs: transform must be smooth, increasing and zero-preserving (R14)",
                );
            }
        }
    }
    Ok(Some(parsed))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn declared() -> SlotSet {
        // s0 Y: basic ssim_mean 13, v2 ssim_mean 401, v2 fragility 422;
        // s1 X: v2 fragility 480.
        SlotSet::from_slots([13, 401, 422, 480])
    }

    fn parse(text: &str) -> Result<DerivedInputs, &'static str> {
        DerivedInputs::parse(text, &declared(), crate::NUM_SCALES)
    }

    #[test]
    fn products_pair_same_cell_difference_with_reference_only() {
        let d = parse(
            "zensim-derived-inputs v1\nin 13\nin 401\nproduct 13 422\nproduct 401 422\nin 480\n",
        )
        .unwrap();
        assert_eq!(d.model_width(), 5);
        assert_eq!(d.gather_width(), 4);
        let mut out = Vec::new();
        d.fill_f32(&[0.5, -0.0, 0.25, 0.75], &mut out);
        assert_eq!(out, vec![0.5, -0.0, 0.125, -0.0, 0.75]);
        assert_eq!(out[3].to_bits(), (-0.0f32).to_bits());
    }

    #[test]
    fn zero_difference_gives_exact_zero_products() {
        let d = parse("zensim-derived-inputs v1\nin 13\nproduct 13 422\nin 401\nin 480\n").unwrap();
        let mut out = Vec::new();
        for frag in [1.0, 0.25, 1e-30, f64::MIN_POSITIVE] {
            d.fill_f32(&[0.0, 0.0, frag, 0.5], &mut out);
            assert_eq!(out[1], 0.0);
        }
    }

    /// A one-layer bake over declared ids 13 (s0 Y basic ssim_mean), 401
    /// (s0 Y v2 ssim_mean) and 422 (s0 Y fragility) whose four model inputs
    /// are `in 13`, `in 401`, `product 13 422`, `product 401 422`.
    fn derived_bake(derived: Option<&str>) -> Vec<u8> {
        let mut metadata = vec![
            serde_json::json!({
                "key": "zentrain.feature_ids", "type": "utf8", "text": "13 401 422"
            }),
            // The process revision, so the v2 ids are extractable (as `bake_over` does).
            serde_json::json!({
                "key": "zentrain.formula_revision", "type": "utf8",
                "text": (crate::ssim_form::active_revision() as u8 + 1).to_string()
            }),
        ];
        if let Some(text) = derived {
            metadata.push(serde_json::json!({
                "key": DERIVED_INPUTS_KEY, "type": "utf8", "text": text
            }));
        }
        let recipe = serde_json::json!({
            "schema_hash": 1, "scaler_mean": [0.0, 0.0, 0.0, 0.0],
            "scaler_scale": [1.0, 2.0, 0.5, 4.0], "metadata": metadata,
            "layers": [{"in_dim":4,"out_dim":1,"activation":"identity","dtype":"f32",
                        "weights":[-1.0, -2.0, -3.0, -4.0],"biases":[100.0]}]
        });
        zenpredict_bake::bake_from_json_str(&recipe.to_string()).expect("bake the recipe")
    }

    #[cfg(feature = "feature-regime-v2")]
    const FX: &str = "zensim-derived-inputs v1\nin 13\nin 401\nproduct 13 422\nproduct 401 422\n";

    // Serving a derived bake needs the feature planner: without
    // `feature-regime-v2`, `BakeScorer::new` refuses it by design.
    #[test]
    #[cfg(feature = "feature-regime-v2")]
    fn served_score_is_the_declared_products_and_identity_is_the_pin() {
        let bytes = derived_bake(Some(FX));
        let model = zenpredict::Model::from_bytes(&bytes).unwrap();
        assert_eq!(model.caller_input_width(), 4);
        let mut scorer = crate::BakeScorer::new(&model).unwrap();
        let mut row = vec![0.0f64; 720];
        // A perfect copy: every difference 0, fragility nonzero.
        row[422] = 0.731;
        assert_eq!(scorer.score_features(&row, 96, 96, None).unwrap(), 100.0);
        row[13] = 0.25;
        row[401] = 0.125;
        let (d13, d401, f) = (0.25f32, 0.125f32, 0.731f64 as f32);
        let mut predictor = zenpredict::Predictor::new(&model);
        let expected = predictor.predict(&[d13, d401, d13 * f, d401 * f]).unwrap()[0];
        let served = scorer.score_features(&row, 96, 96, None).unwrap();
        assert_eq!(served, f64::from(expected));
        assert!(served < 100.0);
    }

    #[test]
    #[cfg(feature = "feature-regime-v2")]
    fn read_set_includes_reference_only_factors() {
        let bytes = derived_bake(Some(FX));
        let model = zenpredict::Model::from_bytes(&bytes).unwrap();
        let scorer = crate::BakeScorer::new(&model).unwrap();
        assert_eq!(scorer.consumed_feature_ids().unwrap(), [13, 401, 422]);
        assert_eq!(
            crate::declared_feature_ids(&model),
            Some(vec![13, 401, 422])
        );
    }

    #[test]
    #[cfg(feature = "feature-regime-v2")]
    fn fd_sensitivity_follows_the_products() {
        let bytes = derived_bake(Some(FX));
        let model = zenpredict::Model::from_bytes(&bytes).unwrap();
        let mut scorer = crate::BakeScorer::new(&model).unwrap();
        let mut row = vec![0.0f64; 720];
        row[13] = 0.5;
        row[401] = 0.25;
        row[422] = 0.5;
        let g = scorer
            .score_features_fd_gradient(&row, 96, 96, None)
            .unwrap();
        // d score / d f13 = -(1/1 + 3*f/0.5) = -4; d / d f422 = -(3*d13/0.5 + 4*d401/4) = -3.25.
        // The forward is f32: one ulp of a score near 98 (7.6e-6) over 2·eps = 1e-3 is ~0.008, so
        // the central difference is accurate to a few hundredths, not 1e-3.
        assert!((g[13] + 4.0).abs() < 0.05, "{}", g[13]);
        assert!((g[422] + 3.25).abs() < 0.05, "{}", g[422]);
        // At zero difference the fragility has no effect at all.
        row[13] = 0.0;
        row[401] = 0.0;
        let g = scorer
            .score_features_fd_gradient(&row, 96, 96, None)
            .unwrap();
        assert_eq!(g[422], 0.0);
    }

    #[test]
    fn a_runtime_without_the_declaration_refuses_the_bake() {
        // What an older runtime sees: three declared ids, four inputs.
        let bytes = derived_bake(None);
        let model = zenpredict::Model::from_bytes(&bytes).unwrap();
        assert!(crate::BakeScorer::new(&model).is_err());
    }

    #[test]
    fn malformed_declarations_refuse_at_load() {
        for text in [
            "zensim-derived-inputs v1\nin 13\nin 401\nproduct 13 422\n",
            "zensim-derived-inputs v1\nin 13\nin 401\nproduct 422 13\nproduct 401 422\n",
            "zensim-derived-inputs v1\nin 13\nin 401\nin 422\nproduct 401 422\n\nproduct 13 401\n",
        ] {
            let bytes = derived_bake(Some(text));
            let model = zenpredict::Model::from_bytes(&bytes).unwrap();
            assert!(
                crate::BakeScorer::new(&model).is_err(),
                "{text:?} must refuse"
            );
        }
    }

    #[test]
    fn refusals() {
        for (text, why) in [
            ("in 13\n", "header"),
            (
                "zensim-derived-inputs v2\nin 13\nin 401\nin 422\nin 480\n",
                "version",
            ),
            ("zensim-derived-inputs v1\n", "empty"),
            (
                "zensim-derived-inputs v1\nin 14\nin 401\nin 422\nin 480\n",
                "undeclared",
            ),
            (
                "zensim-derived-inputs v1\nin 13\nin 401\nin 422\n",
                "unused declared id",
            ),
            (
                "zensim-derived-inputs v1\nin 13\nin 13\nin 401\nin 422\nin 480\n",
                "duplicate",
            ),
            (
                "zensim-derived-inputs v1\nproduct 422 13\nin 401\nin 480\n",
                "factor order",
            ),
            (
                "zensim-derived-inputs v1\nproduct 13 401\nin 422\nin 480\n",
                "two differences",
            ),
            (
                "zensim-derived-inputs v1\nproduct 13 480\nin 401\nin 422\n",
                "cross cell",
            ),
            (
                "zensim-derived-inputs v1\nsum 13 422\nin 401\nin 480\n",
                "unknown kind",
            ),
        ] {
            assert!(parse(text).is_err(), "{why} must refuse");
        }
    }
}
