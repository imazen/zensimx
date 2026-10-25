//! E33 `fx1` derived inputs on the training side.
//!
//! The registration (`benchmarks/e33_registration_2026-10-09.md` §5.1, §10)
//! declares Arm C's model inputs as the 410 difference ids followed by one
//! `difference × fragility` product per difference, fragility taken at the
//! difference's own scale and channel. This module owns that declaration for
//! the trainer and the bake tools:
//!
//! * [`Fx1Declaration`] — the JSON declaration file (`zensim-fx1-v1`).
//! * [`Fx1Declaration::extend_compact`] / [`Fx1Declaration::extend_dense`] —
//!   append product columns to loaded training rows, after the logical table
//!   width, as IEEE f32 products of the f32-stored factors.
//! * [`Fx1Declaration::compact_bake`] — rewrite a trainer bake (identity
//!   width `base + products`, dropped rows pinned to zero) into the servable
//!   form zensim reads: `zentrain.feature_ids` = the read set (differences and
//!   fragility factors), `zensim.derived_inputs` = the model-input list, layer
//!   0 holding exactly the kept rows in model-input order. A bit-identity gate
//!   compares the compact bake through `zensim::BakeScorer` against the
//!   original weights on deterministic probe rows.
//!
//! The pairing rule itself (Difference × same-cell ReferenceOnly) is enforced
//! by zensim when the compact bake loads, so no second copy of the feature
//! registry lives here.

use crate::mlp_train::CompactRows;
use zenpredict::{Model, WeightStorage};
use zenpredict_bake::{BakeLayer, BakeMetadataEntry, BakeRequest, bake};

/// Metadata key zensim reads (`zensim::derived_inputs::DERIVED_INPUTS_KEY`;
/// kept in sync by [`tests::compact_bake_serves_through_bakescorer`], which
/// fails if zensim stops honouring this key).
pub const DERIVED_INPUTS_KEY: &str = "zensim.derived_inputs";
/// Schema tag of the declaration file.
pub const FX1_SCHEMA: &str = "zensim-fx1-v1";
const HEADER: &str = "zensim-derived-inputs v1";

/// One layer copied out of a bake, each keeping its own dtype.
struct OwnedLayer {
    in_dim: usize,
    out_dim: usize,
    activation: zenpredict::Activation,
    dtype: zenpredict::WeightDtype,
    weights: Vec<f32>,
    biases: Vec<f32>,
}

/// The `fx1` declaration file.
#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
pub struct Fx1Declaration {
    pub schema: String,
    /// Direct model inputs, ascending feature ids.
    pub direct: Vec<usize>,
    /// `[difference id, reference-only id]`, in model-input order.
    pub products: Vec<[usize; 2]>,
    /// Direct ids that have no product (must be empty for E33 Arm C).
    #[serde(default)]
    pub unpaired: Vec<usize>,
}

impl Fx1Declaration {
    /// Parse and structurally validate a declaration.
    pub fn parse(text: &str) -> Result<Self, String> {
        let d: Fx1Declaration =
            serde_json::from_str(text).map_err(|e| format!("fx1 declaration: {e}"))?;
        if d.schema != FX1_SCHEMA {
            return Err(format!(
                "fx1 declaration: schema {:?} is not {FX1_SCHEMA}",
                d.schema
            ));
        }
        if d.direct.is_empty() || d.direct.windows(2).any(|w| w[0] >= w[1]) {
            return Err(
                "fx1 declaration: direct ids must be nonempty, ascending and unique".into(),
            );
        }
        let mut seen_a = std::collections::BTreeSet::new();
        for &[a, b] in &d.products {
            if d.direct.binary_search(&a).is_err() {
                return Err(format!(
                    "fx1 declaration: product factor {a} is not a direct input"
                ));
            }
            if d.direct.binary_search(&b).is_ok() {
                return Err(format!(
                    "fx1 declaration: reference factor {b} is also a direct input (no raw fragility input)"
                ));
            }
            if !seen_a.insert(a) {
                return Err(format!("fx1 declaration: difference {a} has two products"));
            }
        }
        let mut unpaired: Vec<usize> = d
            .direct
            .iter()
            .copied()
            .filter(|id| !seen_a.contains(id))
            .collect();
        unpaired.sort_unstable();
        let mut listed = d.unpaired.clone();
        listed.sort_unstable();
        if unpaired != listed {
            return Err(format!(
                "fx1 declaration: unpaired list {listed:?} differs from the direct ids without a product {unpaired:?}"
            ));
        }
        Ok(d)
    }

    /// Read and parse a declaration file; returns it with the file's SHA-256.
    pub fn load(path: &std::path::Path) -> Result<(Self, String), String> {
        let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
        let text =
            std::str::from_utf8(&bytes).map_err(|_| format!("{}: not utf8", path.display()))?;
        let sha = {
            use sha2::Digest;
            let digest = sha2::Sha256::digest(&bytes);
            digest
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>()
        };
        Ok((Self::parse(text)?, sha))
    }

    /// Every feature id the model reads: direct ids plus reference factors.
    pub fn read_set(&self) -> Vec<usize> {
        let mut ids: Vec<usize> = self.direct.clone();
        ids.extend(self.products.iter().map(|p| p[1]));
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    /// Ids a training loader must retain: the read set.
    pub fn load_ids(&self) -> Vec<usize> {
        self.read_set()
    }

    /// Logical column of product `j` in a table of logical width `base`.
    pub fn product_column(&self, base: usize, j: usize) -> usize {
        base + j
    }

    /// Whether the declaration reads ids past zensim's registry (research columns only a table can supply).
    pub fn reads_research_ids(&self) -> bool {
        let width = zensim::research::full_width();
        self.read_set().iter().any(|&id| id >= width)
    }

    /// The compact model row for one full-width table row: direct ids in declaration order, then the products,
    /// factors narrowed to f32 first (the trainer's `extend_compact` layout).
    pub fn compact_row(&self, row: &[f64], out: &mut Vec<f32>) {
        out.clear();
        out.extend(self.direct.iter().map(|&id| row[id] as f32));
        out.extend(
            self.products
                .iter()
                .map(|&[a, b]| Self::product_f32(row[a] as f32, row[b] as f32)),
        );
    }

    /// The trainer's kept logical columns after extension: the direct ids,
    /// then one column per product past `base`.
    pub fn kept_columns(&self, base: usize) -> Vec<usize> {
        let mut kept = self.direct.clone();
        kept.extend((0..self.products.len()).map(|j| self.product_column(base, j)));
        kept
    }

    /// `zensim.derived_inputs` text: `in` per direct id, then `product`s.
    pub fn metadata_text(&self) -> String {
        let mut s = String::from(HEADER);
        s.push('\n');
        for id in &self.direct {
            s.push_str(&format!("in {id}\n"));
        }
        for [a, b] in &self.products {
            s.push_str(&format!("product {a} {b}\n"));
        }
        s
    }

    fn product_f32(a: f32, b: f32) -> f32 {
        a * b
    }

    /// Replace a compact group (kept = [`Self::load_ids`]) by its extended
    /// form: direct ids, then products at `base + j`. Reference factors that
    /// are not direct inputs leave the kept set.
    pub fn extend_compact(&self, rows: &mut CompactRows, base: usize) -> Result<(), String> {
        let load = self.load_ids();
        let kept: Vec<usize> = rows.kept.iter().map(|&k| k as usize).collect();
        if kept != load {
            return Err(format!(
                "fx1: compact rows keep {} columns, the declaration loads {}",
                kept.len(),
                load.len()
            ));
        }
        if rows.n_features != base {
            return Err(format!(
                "fx1: compact rows have logical width {}, expected {base}",
                rows.n_features
            ));
        }
        let col = |id: usize| load.binary_search(&id).expect("declared id is loaded");
        let direct: Vec<usize> = self.direct.iter().map(|&id| col(id)).collect();
        let products: Vec<(usize, usize)> = self
            .products
            .iter()
            .map(|&[a, b]| (col(a), col(b)))
            .collect();
        let k_in = load.len();
        let k_out = direct.len() + products.len();
        let mut data = Vec::with_capacity(rows.n_rows * k_out);
        for r in 0..rows.n_rows {
            let row = &rows.data[r * k_in..(r + 1) * k_in];
            data.extend(direct.iter().map(|&c| row[c]));
            data.extend(
                products
                    .iter()
                    .map(|&(a, b)| Self::product_f32(row[a], row[b])),
            );
        }
        rows.data = data;
        rows.kept = self.kept_columns(base).iter().map(|&c| c as u32).collect();
        rows.n_features = base + self.products.len();
        Ok(())
    }

    /// Extend a dense row-major buffer of width `base` to `base + products`.
    /// Factors are narrowed to f32 first, matching the compact path.
    pub fn extend_dense(
        &self,
        flat: &mut Vec<f64>,
        n_rows: usize,
        base: usize,
    ) -> Result<(), String> {
        if flat.len() != n_rows * base {
            return Err(format!(
                "fx1: dense rows hold {} values, expected {n_rows}×{base}",
                flat.len()
            ));
        }
        let width = base + self.products.len();
        let mut out = Vec::with_capacity(n_rows * width);
        for r in 0..n_rows {
            let row = &flat[r * base..(r + 1) * base];
            out.extend_from_slice(row);
            out.extend(
                self.products
                    .iter()
                    .map(|&[a, b]| f64::from(Self::product_f32(row[a] as f32, row[b] as f32))),
            );
        }
        *flat = out;
        Ok(())
    }

    /// Rewrite a trainer bake of identity width `base + products` into the
    /// compact derived form, then gate it. See the module docs.
    pub fn compact_bake(&self, bytes: &[u8], base: usize) -> Result<Vec<u8>, String> {
        let model =
            Model::from_bytes(bytes).map_err(|e| format!("fx1 compact: parse bake: {e:?}"))?;
        let width = base + self.products.len();
        if model.n_inputs() != width || model.caller_input_width() != width {
            return Err(format!(
                "fx1 compact: bake has {} inputs, expected base {base} + {} products",
                model.n_inputs(),
                self.products.len()
            ));
        }
        if model.feature_transforms().is_some() {
            return Err("fx1 compact: feature transforms are not supported".into());
        }
        if model
            .metadata()
            .get(zensim::ZENTRAIN_FEATURE_IDS_KEY)
            .is_some()
            || model.metadata().get(DERIVED_INPUTS_KEY).is_some()
        {
            return Err("fx1 compact: bake already declares its inputs".into());
        }
        let rows = self.kept_columns(base);
        let mut layers: Vec<OwnedLayer> = Vec::new();
        for (li, l) in model.layers().enumerate() {
            let (weights, dtype) = match &l.weights {
                WeightStorage::F32(w) => (w.to_vec(), zenpredict::WeightDtype::F32),
                WeightStorage::F16(w) => (
                    w.iter().map(|b| zenpredict::f16_bits_to_f32(*b)).collect(),
                    zenpredict::WeightDtype::F16,
                ),
                WeightStorage::I8 { .. } => {
                    return Err(format!("fx1 compact: layer {li} is i8; start from f32/f16"));
                }
            };
            layers.push(OwnedLayer {
                in_dim: l.in_dim,
                out_dim: l.out_dim,
                activation: l.activation,
                dtype,
                weights,
                biases: l.biases.to_vec(),
            });
        }
        // Every dropped layer-0 row must already be exactly zero: the trainer's
        // keep mask pins them, and a nonzero dropped row would be a fit that
        // read an undeclared input.
        {
            let (in_dim, out_dim, w) = (layers[0].in_dim, layers[0].out_dim, &layers[0].weights);
            let mut keep = vec![false; in_dim];
            for &r in &rows {
                keep[r] = true;
            }
            for (r, kept) in keep.iter().enumerate() {
                if !kept && w[r * out_dim..(r + 1) * out_dim].iter().any(|&x| x != 0.0) {
                    return Err(format!("fx1 compact: dropped layer-0 row {r} is not zero"));
                }
            }
        }
        let out_dim0 = layers[0].out_dim;
        let w0: Vec<f32> = rows
            .iter()
            .flat_map(|&r| {
                layers[0].weights[r * out_dim0..(r + 1) * out_dim0]
                    .iter()
                    .copied()
            })
            .collect();
        layers[0].in_dim = rows.len();
        layers[0].weights = w0;
        let pick = |v: &[f32]| rows.iter().map(|&r| v[r]).collect::<Vec<f32>>();
        let scaler_mean = pick(model.scaler_mean());
        let scaler_scale = pick(model.scaler_scale());
        let mut md: Vec<(String, zenpredict::MetadataType, Vec<u8>)> = model
            .metadata()
            .iter()
            .map(|e| (e.key.to_string(), e.kind, e.value.to_vec()))
            .collect();
        let read_set = self.read_set();
        md.push((
            zensim::ZENTRAIN_FEATURE_IDS_KEY.to_string(),
            zenpredict::MetadataType::Utf8,
            read_set
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(" ")
                .into_bytes(),
        ));
        md.push((
            DERIVED_INPUTS_KEY.to_string(),
            zenpredict::MetadataType::Utf8,
            self.metadata_text().into_bytes(),
        ));
        let bl: Vec<BakeLayer<'_>> = layers
            .iter()
            .map(|l| BakeLayer {
                in_dim: l.in_dim,
                out_dim: l.out_dim,
                activation: l.activation,
                dtype: l.dtype,
                weights: &l.weights,
                biases: &l.biases,
            })
            .collect();
        let me: Vec<BakeMetadataEntry<'_>> = md
            .iter()
            .map(|(k, t, v)| BakeMetadataEntry {
                key: k,
                kind: *t,
                value: v,
            })
            .collect();
        let out = bake(&BakeRequest {
            schema_hash: model.schema_hash(),
            flags: 0,
            scaler_mean: &scaler_mean,
            scaler_scale: &scaler_scale,
            layers: &bl,
            feature_bounds: &[],
            metadata: &me,
            output_specs: &[],
            discrete_sets: &[],
            sparse_overrides: &[],
            feature_order: None,
            output_order: None,
            compressed: true,
            hu_permutations: None,
        })
        .map_err(|e| format!("fx1 compact: serialize: {e:?}"))?;
        if self.reads_research_ids() {
            // By-construction exploration: ids past zensim's registry (the byc_globaff descriptor) cannot be
            // served by `BakeScorer`, so its half of the gate cannot run. `byc_predict` forwards these bakes
            // through zenpredict with the same row layout.
            eprintln!(
                "fx1 compact: declaration reads ids at or past zensim's registry width {}; \
                 the BakeScorer gate is skipped (research bake, zensim cannot serve it)",
                zensim::research::full_width()
            );
        } else {
            self.gate(bytes, &out, base, 64)?;
        }
        Ok(out)
    }

    /// Bit-identity gate: the compact bake served by `BakeScorer` equals the
    /// original weights forwarded on the extended row with every unkept
    /// column zeroed (the densify gate's strong form).
    fn gate(&self, original: &[u8], compact: &[u8], base: usize, n: usize) -> Result<(), String> {
        let orig = Model::from_bytes(original).map_err(|e| format!("fx1 gate: {e:?}"))?;
        let dense = Model::from_bytes(compact).map_err(|e| format!("fx1 gate: {e:?}"))?;
        let mut scorer = zensim::BakeScorer::new(&dense)
            .map_err(|e| format!("fx1 gate: zensim refuses the compact bake: {e}"))?;
        let mut predictor = zenpredict::Predictor::new(&orig);
        let read_set = self.read_set();
        let walk = read_set.last().copied().unwrap_or(0) + 1;
        let width = base + self.products.len();
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            (z >> 11) as f64 / (1u64 << 53) as f64
        };
        let direct_mask: Vec<bool> = {
            let mut m = vec![false; base];
            for &id in &self.direct {
                m[id] = true;
            }
            m
        };
        for i in 0..n {
            // Row 0 is a perfect copy: differences 0, reference factors live.
            let mut row = vec![0.0f64; walk.max(base)];
            for &id in &read_set {
                row[id] = if i == 0 && direct_mask[id] {
                    0.0
                } else {
                    next() * 2.0
                };
            }
            let mut ext = vec![0f32; width];
            for &id in &self.direct {
                ext[id] = row[id] as f32;
            }
            for (j, &[a, b]) in self.products.iter().enumerate() {
                ext[base + j] = Self::product_f32(row[a] as f32, row[b] as f32);
            }
            let want = predictor
                .predict(&ext)
                .map_err(|e| format!("fx1 gate: original forward: {e:?}"))?[0];
            let got = scorer
                .score_features(&row, 0, 0, None)
                .map_err(|e| format!("fx1 gate: compact forward: {e}"))?;
            if got.to_bits() != f64::from(want).to_bits() {
                return Err(format!(
                    "fx1 gate FAILED at probe {i}: compact {got:?} vs original {want:?}"
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decl() -> Fx1Declaration {
        Fx1Declaration::parse(
            r#"{"schema":"zensim-fx1-v1","direct":[13,401],"products":[[13,422],[401,422]],"unpaired":[]}"#,
        )
        .unwrap()
    }

    #[test]
    fn declaration_refusals() {
        for (text, why) in [
            (r#"{"schema":"x","direct":[13],"products":[]}"#, "schema"),
            (
                r#"{"schema":"zensim-fx1-v1","direct":[401,13],"products":[]}"#,
                "order",
            ),
            (
                r#"{"schema":"zensim-fx1-v1","direct":[13],"products":[[14,422]]}"#,
                "factor a",
            ),
            (
                r#"{"schema":"zensim-fx1-v1","direct":[13,422],"products":[[13,422]]}"#,
                "raw ref",
            ),
            (
                r#"{"schema":"zensim-fx1-v1","direct":[13],"products":[[13,422],[13,480]]}"#,
                "two products",
            ),
            (
                r#"{"schema":"zensim-fx1-v1","direct":[13,401],"products":[[13,422]]}"#,
                "unlisted unpaired",
            ),
        ] {
            assert!(Fx1Declaration::parse(text).is_err(), "{why}");
        }
        let ok = Fx1Declaration::parse(
            r#"{"schema":"zensim-fx1-v1","direct":[13,401],"products":[[13,422]],"unpaired":[401]}"#,
        );
        assert!(ok.is_ok());
    }

    #[test]
    fn compact_extension_matches_dense_extension() {
        let d = decl();
        let base = 720;
        // Two rows over the load set [13, 401, 422].
        let mut c = CompactRows {
            data: vec![0.5, 0.25, 0.75, 0.0, -0.0, 0.3],
            n_rows: 2,
            n_features: base,
            kept: vec![13, 401, 422],
        };
        d.extend_compact(&mut c, base).unwrap();
        assert_eq!(c.kept, vec![13, 401, 720, 721]);
        assert_eq!(c.n_features, 722);
        assert_eq!(&c.data[..4], &[0.5, 0.25, 0.5 * 0.75, 0.25 * 0.75]);
        assert_eq!(&c.data[4..6], &[0.0, -0.0]);
        assert_eq!(c.data[6].to_bits(), 0.0f32.to_bits());
        assert_eq!(c.data[7].to_bits(), (-0.0f32).to_bits());
        let mut flat = vec![0.0f64; 2 * base];
        for (r, vals) in [[0.5, 0.25, 0.75], [0.0, -0.0, 0.3]].iter().enumerate() {
            flat[r * base + 13] = vals[0];
            flat[r * base + 401] = vals[1];
            flat[r * base + 422] = vals[2];
        }
        d.extend_dense(&mut flat, 2, base).unwrap();
        for r in 0..2 {
            for j in 0..2 {
                assert_eq!(
                    (flat[r * 722 + 720 + j] as f32).to_bits(),
                    c.data[r * 4 + 2 + j].to_bits()
                );
            }
        }
    }

    /// A trainer-shaped bake: identity width 722 (720 + two products), only
    /// rows 13, 401, 720, 721 live, nonneg-style output.
    fn trainer_bake() -> Vec<u8> {
        let width = 722;
        let hidden = 3;
        let mut w1 = vec![0.0f64; width * hidden];
        for (k, &r) in [13usize, 401, 720, 721].iter().enumerate() {
            for h in 0..hidden {
                w1[r * hidden + h] = 0.1 + 0.05 * (k * hidden + h) as f64;
            }
        }
        let mut scale = vec![1.0f64; width];
        scale[13] = 0.5;
        scale[720] = 0.25;
        let recipe = serde_json::json!({
            "schema_hash": 7, "scaler_mean": vec![0.0; width], "scaler_scale": scale,
            "metadata": [{"key":"zentrain.formula_revision","type":"utf8","text":"5"}],
            "layers": [
                {"in_dim":width,"out_dim":hidden,"activation":"relu","dtype":"f32",
                 "weights":w1,"biases":vec![0.0; hidden]},
                {"in_dim":hidden,"out_dim":1,"activation":"identity","dtype":"f32",
                 "weights":[-1.0,-2.0,-0.5],"biases":[100.0]}]
        });
        zenpredict_bake::bake_from_json_str(&recipe.to_string()).unwrap()
    }

    #[test]
    fn compact_bake_serves_through_bakescorer() {
        let d = decl();
        let out = d.compact_bake(&trainer_bake(), 720).unwrap();
        let m = Model::from_bytes(&out).unwrap();
        assert_eq!(m.caller_input_width(), 4);
        assert_eq!(zensim::declared_feature_ids(&m), Some(vec![13, 401, 422]));
        let mut s = zensim::BakeScorer::new(&m).unwrap();
        let mut row = vec![0.0; 720];
        row[422] = 0.6;
        // Perfect copy: exactly the pin.
        assert_eq!(s.score_features(&row, 0, 0, None).unwrap(), 100.0);
        row[13] = 0.2;
        assert!(s.score_features(&row, 0, 0, None).unwrap() < 100.0);
    }

    #[test]
    fn read_set_owners_name_derived_bakes_correctly() {
        let out = decl().compact_bake(&trainer_bake(), 720).unwrap();
        let m = Model::from_bytes(&out).unwrap();
        let r = crate::feature_set::bake_feature_set_ref(&m, "rev5_localwin").unwrap();
        assert_eq!(r.slots.iter_slots().collect::<Vec<_>>(), vec![13, 401, 422]);
        assert!(crate::block_profile::profile(&m).is_err());
        // The wire is the gathered declared read set (like a dense bake), not the model's input
        // width (direct inputs plus products), so a 720-wide table carries no layout shortfall.
        let declared = zensim::declared_feature_ids(&m).unwrap().len();
        assert!(m.caller_input_width() > declared);
        assert_eq!(r.layout, Some(declared));
        let table = crate::feature_set::FeatureSetRef {
            id: r.id.clone(),
            slots: r.slots.clone(),
            layout: Some(720),
            source: "test table".to_string(),
            inferred: false,
        };
        assert!(
            crate::feature_set::check(&r, &table)
                .iter()
                .all(|m| m.kind != crate::feature_set::MismatchKind::LayoutDiffers)
        );
    }

    #[test]
    fn compact_bake_refuses_a_live_dropped_row() {
        let d = decl();
        let mut bytes_model = trainer_bake();
        // Rebuild with row 500 live.
        let width = 722;
        let hidden = 3;
        let mut w1 = vec![0.0f64; width * hidden];
        for &r in &[13usize, 401, 500, 720, 721] {
            w1[r * hidden] = 0.3;
        }
        let recipe = serde_json::json!({
            "schema_hash": 7, "scaler_mean": vec![0.0; width], "scaler_scale": vec![1.0; width],
            "metadata": [], "layers": [
                {"in_dim":width,"out_dim":hidden,"activation":"relu","dtype":"f32",
                 "weights":w1,"biases":vec![0.0; hidden]},
                {"in_dim":hidden,"out_dim":1,"activation":"identity","dtype":"f32",
                 "weights":[-1.0,-1.0,-1.0],"biases":[100.0]}]
        });
        bytes_model =
            zenpredict_bake::bake_from_json_str(&recipe.to_string()).unwrap_or(bytes_model);
        let err = d.compact_bake(&bytes_model, 720).unwrap_err();
        assert!(err.contains("row 500"), "{err}");
    }
}
