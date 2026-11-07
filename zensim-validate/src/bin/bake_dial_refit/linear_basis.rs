//! Shared private basis owner for native fitting and cost qualification.
use serde::Deserialize;

#[derive(Deserialize)]
pub(super) struct Spec {
    pub(super) schema: String,
    pub(super) terms: Vec<(usize, Option<usize>)>,
    pub(super) scales: Vec<f32>,
    /// Empty for linear; otherwise two strictly positive knots per term.
    pub(super) knots: Vec<[f32; 2]>,
    pub(super) basis_scales: Vec<f32>,
    #[serde(default)]
    pub(super) calibration_ids: Vec<usize>,
    #[serde(default)]
    pub(super) calibration_targets: Vec<f64>,
}

impl Spec {
    pub(super) fn width(&self) -> usize {
        self.terms.len() * if self.knots.is_empty() { 1 } else { 3 }
    }

    pub(super) fn validate(&self, input_width: usize) -> Result<(), String> {
        if self.schema != "shaped-linear-basis-v1"
            || self.terms.is_empty()
            || self.terms.len() != self.scales.len()
            || self.basis_scales.len() != self.width()
            || self
                .terms
                .iter()
                .any(|&(a, b)| a >= input_width || b.is_some_and(|v| v >= input_width))
            || self
                .scales
                .iter()
                .chain(&self.basis_scales)
                .any(|v| !v.is_finite() || *v <= 0.)
            || (!self.knots.is_empty()
                && (self.knots.len() != self.terms.len()
                    || self
                        .knots
                        .iter()
                        .any(|k| !k[1].is_finite() || !(0. < k[0] && k[0] < k[1]))))
        {
            return Err("invalid registered basis".into());
        }
        Ok(())
    }

    pub(super) fn apply(&self, raw: &[f32], out: &mut [f32]) -> Result<(), String> {
        for (j, &(a, b)) in self.terms.iter().enumerate() {
            let value = b.map_or(raw[a], |b| raw[a] * raw[b]);
            let x = value / self.scales[j];
            if self.knots.is_empty() {
                out[j] = x / self.basis_scales[j];
            } else {
                if x < 0. {
                    return Err("negative damage input in monotone segment basis".into());
                }
                let [a, b] = self.knots[j];
                out[3 * j] = x.min(a) / self.basis_scales[3 * j];
                out[3 * j + 1] = (x - a).max(0.).min(b - a) / self.basis_scales[3 * j + 1];
                out[3 * j + 2] = (x - b).max(0.) / self.basis_scales[3 * j + 2];
            }
        }
        if out.iter().any(|v| !v.is_finite()) {
            return Err("nonfinite shaped feature".into());
        }
        Ok(())
    }
}
