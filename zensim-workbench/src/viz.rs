//! Browser inspection uses the bake scorer and the fold observer, never JS math.
use super::{BAKES, model, rgb};
use wasm_bindgen::prelude::*;
use zensim::{BakeScorer, RgbSlice, research};

fn quantile(v: &[f64], p: f64) -> f64 {
    let mut v = v.to_vec();
    v.sort_by(f64::total_cmp);
    let x = p * (v.len() - 1) as f64;
    let i = x.floor() as usize;
    v[i] + (v[x.ceil() as usize] - v[i]) * (x - i as f64)
}

/// A browser-local pixel pair. The retained production planes are reused by maps.
#[wasm_bindgen]
pub struct AtlasSession {
    atlas: research::DetectorAtlas,
    summary: String,
}
impl AtlasSession {
    /// Plain-error entry for the independent native parity runner.
    pub fn from_rgba(
        metric: &str,
        reference: &[u8],
        distorted: &[u8],
        width: u32,
        height: u32,
        stride: usize,
    ) -> Result<Self, String> {
        let (_, bytes, sha) = BAKES.iter().find(|b| b.0 == metric).ok_or("unknown bake")?;
        let (r, d) = (
            rgb(reference, width, height, stride)?,
            rgb(distorted, width, height, stride)?,
        );
        let identical = r == d;
        let (r, d) = (
            RgbSlice::new(&r, width as usize, height as usize),
            RgbSlice::new(&d, width as usize, height as usize),
        );
        let (features, atlas) =
            research::extract_with_detectors(bytes, &r, &d).map_err(|e| e.to_string())?;
        let m = model(metric)?;
        let score = BakeScorer::new(&m)
            .map_err(|e| e.to_string())?
            .score_features_with_identity(&features, width, height, None, identical)
            .map_err(|e| e.to_string())?;
        Ok(Self {
            atlas,
            summary: serde_json::json!({"score":score,"features":features,"bake_sha256":sha})
                .to_string(),
        })
    }
    /// Plain-error map entry for the native parity runner.
    pub fn detector_json(&self, id: usize) -> Result<String, String> {
        let map = self.atlas.map(id).map_err(|e| e.to_string())?;
        let v: Vec<_> = map.values.iter().map(|&v| f64::from(v)).collect();
        let range = if v.is_empty() {
            vec![]
        } else {
            vec![quantile(&v, 0.02), quantile(&v, 0.98)]
        };
        Ok(serde_json::json!({"id":id,"width":map.width,"height":map.height,"values":map.values,
            "range":map.range,"percentile_range":range,"role":map.role,"pooling_grid":map.pooling_grid}).to_string())
    }
}
#[wasm_bindgen]
impl AtlasSession {
    /// Opaque RGBA8 rows, including explicit stride; no uploads or native decoder.
    #[wasm_bindgen(constructor)]
    pub fn new(
        metric: &str,
        reference: &[u8],
        distorted: &[u8],
        width: u32,
        height: u32,
        stride: usize,
    ) -> Result<AtlasSession, JsError> {
        Self::from_rgba(metric, reference, distorted, width, height, stride)
            .map_err(|e| JsError::new(&e))
    }
    /// Score and pooled features from the retained fold.
    pub fn summary(&self) -> String {
        self.summary.clone()
    }
    /// Formula operands at one basic detector cell, calculated in native Rust.
    pub fn basic(&self, id: usize, x: usize, y: usize) -> Result<String, JsError> {
        let values = self
            .atlas
            .basic_terms(id, x, y)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(serde_json::json!({"x":x,"y":y,"names":["mu_r","mu_d","src","dst","square_moment","error_moment","variance_sum","error_variance","d1","edge","artifact","detail","mse"],"values":values,"formula_replay_bit_identity":true}).to_string())
    }
    /// Native detector samples, native quantiles, and an explicit global label.
    pub fn detector(&self, id: usize) -> Result<String, JsError> {
        self.detector_json(id).map_err(|e| JsError::new(&e))
    }
}

/// A compact bank of consumed raw registry slots, expanded only for serving.
#[wasm_bindgen]
pub struct LensSession {
    metric: String,
    rows: Vec<Vec<f64>>,
    slots: Vec<usize>,
}
impl LensSession {
    /// Plain-error constructor for native/browser parity.
    pub fn from_sample(metric: &str, slots: &[u32], values: &[f32]) -> Result<Self, String> {
        let m = model(metric)?;
        if slots.is_empty()
            || values.len() != 200 * slots.len()
            || values.iter().any(|v| !v.is_finite())
        {
            return Err("expected 200 finite rows of declared raw slots".into());
        }
        let _ = m; // loading validates the selected bake before admitting data
        let width = slots.iter().copied().max().ok_or("no slots")? as usize + 1;
        if width > research::full_width() {
            return Err("raw slot outside registry".into());
        }
        let rows = values
            .chunks(slots.len())
            .map(|input| {
                let mut row = vec![0.0; width];
                for (&id, &v) in slots.iter().zip(input) {
                    row[id as usize] = f64::from(v);
                }
                row
            })
            .collect();
        Ok(Self {
            metric: metric.into(),
            rows,
            slots: slots.iter().map(|&s| s as usize).collect(),
        })
    }
    /// All 200 ICE traces and flags are returned; the UI may thin painted traces.
    pub fn curve_json(&self, feature: usize, grid: &[f32]) -> Result<String, String> {
        self.curve_impl(&[feature], grid, false)
    }
    /// Joint strength axis for a damage family; reference fragility stays fixed.
    pub fn family_json(&self, features: &[usize]) -> Result<String, String> {
        self.curve_impl(
            features,
            &(0..21).map(|j| j as f32 / 10.0).collect::<Vec<_>>(),
            true,
        )
    }
    fn curve_impl(
        &self,
        features: &[usize],
        grid: &[f32],
        strength: bool,
    ) -> Result<String, String> {
        if features.is_empty()
            || features.iter().any(|id| !self.slots.contains(id))
            || grid.len() != 21
            || grid.iter().any(|v| !v.is_finite())
        {
            return Err("expected sampled raw slots and 21 finite grid points".into());
        }
        let feature = features[0];
        let m = model(&self.metric)?;
        let mut scorer = BakeScorer::new(&m).map_err(|e| e.to_string())?;
        let mut ice = Vec::new();
        let mut flags = Vec::new();
        for (i, input) in self.rows.iter().enumerate() {
            let mut row = input.clone();
            let mut curve = Vec::new();
            for &x in grid {
                for &id in features {
                    row[id] = if strength {
                        input[id] * f64::from(x)
                    } else {
                        f64::from(x)
                    };
                }
                curve.push(
                    scorer
                        .score_features(&row, 512, 512, None)
                        .map_err(|e| e.to_string())?,
                );
            }
            for j in 1..curve.len() {
                let delta = curve[j] - curve[j - 1];
                if delta > 0.0 {
                    flags.push(serde_json::json!({"row":i,"point":j,"delta":delta,"severity":if delta < 0.001 {"note"} else if delta < 0.05 {"warning"} else if delta < 0.5 {"serious"} else {"critical"}}));
                }
            }
            ice.push(curve);
        }
        let stats = |traces: &[Vec<f64>]| {
            let mean: Vec<_> = (0..grid.len())
                .map(|j| traces.iter().map(|r| r[j]).sum::<f64>() / traces.len() as f64)
                .collect();
            let band = |p| {
                (0..grid.len())
                    .map(|j| quantile(&traces.iter().map(|r| r[j]).collect::<Vec<_>>(), p))
                    .collect::<Vec<_>>()
            };
            (mean, band(0.25), band(0.75))
        };
        let centered_ice: Vec<Vec<f64>> = ice
            .iter()
            .map(|r| r.iter().map(|v| v - r[0]).collect())
            .collect();
        let (mean, p25, p75) = stats(&ice);
        let (centered_mean, centered_p25, centered_p75) = stats(&centered_ice);
        Ok(serde_json::json!({"x":grid,"ice":ice,"mean":mean,"p25":p25,"p75":p75,
            "centered_ice":centered_ice,"centered_mean":centered_mean,"centered_p25":centered_p25,"centered_p75":centered_p75,
            "rise_flags":flags,"rows":200,"raw_feature":feature,"geometry":[512,512],"raw_feature_intervention":true,"features":features,"family_strength":strength}).to_string())
    }
    /// The existing lens definitions on the compact sample: raw damage
    /// ablation, plus score separation by empirical damage deciles.
    pub fn diagnostics_json(&self, feature: usize, ablate: &[usize]) -> Result<String, String> {
        if !self.slots.contains(&feature)
            || ablate.is_empty()
            || ablate.iter().any(|id| !self.slots.contains(id))
        {
            return Err("sampled raw damage slots required".into());
        }
        let m = model(&self.metric)?;
        let mut scorer = BakeScorer::new(&m).map_err(|e| e.to_string())?;
        let mut scores = Vec::new();
        let mut deltas = Vec::new();
        let mut values = Vec::new();
        for original in &self.rows {
            let score = scorer
                .score_features(original, 512, 512, None)
                .map_err(|e| e.to_string())?;
            let mut row = original.clone();
            for &id in ablate {
                row[id] = 0.0;
            }
            let delta = scorer
                .score_features(&row, 512, 512, None)
                .map_err(|e| e.to_string())?
                - score;
            scores.push(score);
            deltas.push(delta);
            values.push(original[feature]);
        }
        let mut order: Vec<_> = (0..values.len()).collect();
        order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
        let edges: Vec<_> = (1..10)
            .map(|q| quantile(&values, q as f64 / 10.0))
            .collect();
        let mut buckets = vec![Vec::new(); 10];
        for i in order {
            buckets[edges.partition_point(|v| values[i] > *v)].push(i);
        }
        let bins:Vec<_>=buckets.into_iter().enumerate().filter(|(_,b)|!b.is_empty()).map(|(bin,b)| {
            let v:Vec<_>=b.iter().map(|&i|scores[i]).collect();
            serde_json::json!({"bin":bin,"low":values[b[0]],"high":values[*b.last().unwrap()],"count":b.len(),"median":quantile(&v,0.5),"p10":quantile(&v,0.1),"p90":quantile(&v,0.9)})
        }).collect();
        Ok(serde_json::json!({"rows":200,"raw_feature":feature,"features":ablate,"scores":scores,"delta":deltas,
            "mean_abs_delta":deltas.iter().map(|v|v.abs()).sum::<f64>()/deltas.len() as f64,
            "median_delta":quantile(&deltas,0.5),"p10":quantile(&deltas,0.1),"p90":quantile(&deltas,0.9),"separation":bins,
            "definition":"Zero the declared raw damage slots, retaining observed reference-only fragility; BakeScorer recomputes products. Effects overlap and are not additive. Separation uses empirical raw-feature deciles and served scores, not human ratings."}).to_string())
    }
}
#[wasm_bindgen]
impl LensSession {
    /// Binary sample decoded as f32; slots are registry identities, not indices.
    #[wasm_bindgen(constructor)]
    pub fn new(metric: &str, slots: &[u32], values: &[f32]) -> Result<LensSession, JsError> {
        Self::from_sample(metric, slots, values).map_err(|e| JsError::new(&e))
    }
    /// Joint family response with a native 0–2 strength grid.
    pub fn family(&self, features: &[u32]) -> Result<String, JsError> {
        self.family_json(&features.iter().map(|&v| v as usize).collect::<Vec<_>>())
            .map_err(|e| JsError::new(&e))
    }
    /// Ablation and separation, computed through the native bake scorer.
    pub fn diagnostics(&self, feature: usize, ablate: &[u32]) -> Result<String, JsError> {
        self.diagnostics_json(
            feature,
            &ablate.iter().map(|&v| v as usize).collect::<Vec<_>>(),
        )
        .map_err(|e| JsError::new(&e))
    }
    /// The bake's forward pass, including native calibration and fragility join.
    pub fn curve(&self, feature: usize, grid: &[f32]) -> Result<String, JsError> {
        self.curve_json(feature, grid).map_err(|e| JsError::new(&e))
    }
}

/// SHA-256 for archive-entry verification, including LAN HTTP without WebCrypto.
#[wasm_bindgen]
pub fn blob_sha256(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Identity of the actual embedded bake, available before a pixel pair is loaded.
#[wasm_bindgen]
pub fn bake_sha256(metric: &str) -> Result<String, JsError> {
    BAKES
        .iter()
        .find(|b| b.0 == metric)
        .map(|b| b.2.to_owned())
        .ok_or_else(|| JsError::new("unknown bake"))
}

/// SHA-256 of the exact metric source and manifest compiled into this module.
#[wasm_bindgen]
pub fn metric_source_sha256() -> String {
    env!("ZENSIM_WORKBENCH_METRIC_SOURCE_SHA256").into()
}
