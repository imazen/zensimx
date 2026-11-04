use crate::{
    Result, audit,
    bank::Bank,
    constraints, lens,
    model::{self, Input},
};
use serde::Serialize;
use std::{collections::BTreeMap, path::Path};
use zenpredict::Model;
#[derive(Serialize)]
pub(crate) struct Drift {
    pub feature: usize,
    pub factor: Option<usize>,
    pub primary_input: usize,
    pub secondary_input: Option<usize>,
    pub primary_l1: f64,
    pub secondary_l1: Option<f64>,
    pub primary_l2: f64,
    pub secondary_l2: Option<f64>,
    pub l2_norm_delta: Option<f64>,
    pub indexwise_l2_delta: Option<f64>,
}
fn norm(row: &[f32]) -> (f64, f64) {
    (
        row.iter().map(|v| f64::from(v.abs())).sum(),
        row.iter()
            .map(|&v| f64::from(v).powi(2))
            .sum::<f64>()
            .sqrt(),
    )
}
pub(crate) fn run(
    primary: &Model,
    secondary: &Model,
    inputs: &[Input],
    other_inputs: &[Input],
    bank: &Bank,
    paths: [&Path; 2],
    feature: usize,
) -> Result<serde_json::Value> {
    let ca = constraints::inspect(primary, inputs, paths[0])?;
    let cb = constraints::inspect(secondary, other_inputs, paths[1])?;
    let aa = audit::run(primary, inputs, bank, None, crate::hash(paths[0])?, &ca)?;
    let ab = audit::run(
        secondary,
        other_inputs,
        bank,
        None,
        crate::hash(paths[1])?,
        &cb,
    )?;
    let widths = [primary.layer(0).out_dim, secondary.layer(0).out_dim];
    let wa = model::weights(primary, 0);
    let wb = model::weights(secondary, 0);
    let lookup: BTreeMap<_, _> = other_inputs
        .iter()
        .map(|i| ((i.feature, i.factor), i.index))
        .collect();
    let mut drift = Vec::new();
    for input in inputs {
        let a = &wa[input.index * widths[0]..(input.index + 1) * widths[0]];
        let na = norm(a);
        let j = lookup.get(&(input.feature, input.factor)).copied();
        let b = j.map(|j| &wb[j * widths[1]..(j + 1) * widths[1]]);
        let nb = b.map(norm);
        drift.push(Drift {
            feature: input.feature,
            factor: input.factor,
            primary_input: input.index,
            secondary_input: j,
            primary_l1: na.0,
            secondary_l1: nb.map(|n| n.0),
            primary_l2: na.1,
            secondary_l2: nb.map(|n| n.1),
            l2_norm_delta: nb.map(|n| na.1 - n.1),
            indexwise_l2_delta: b.filter(|b| b.len() == a.len()).map(|b| {
                a.iter()
                    .zip(b)
                    .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
                    .sum::<f64>()
                    .sqrt()
            }),
        });
    }
    let mut diff = BTreeMap::new();
    for key in aa.counts.keys().chain(ab.counts.keys()) {
        diff.insert(
            key.clone(),
            *aa.counts.get(key).unwrap_or(&0) as i64 - *ab.counts.get(key).unwrap_or(&0) as i64,
        );
    }
    let pa = lens::pdp(primary, bank, feature)?;
    let pb = lens::pdp(secondary, bank, feature)?;
    Ok(
        serde_json::json!({"primary_sha256":aa.bake_sha256,"secondary_sha256":ab.bake_sha256,"bank_sha256":bank.sha256,"sampled_row_indices":bank.row_indices,"primary_counts":aa.counts,"secondary_counts":ab.counts,"audit_count_delta":diff,"weight_drift":drift,"pdp":[pa,pb],"weight_drift_caveat":"L1/L2 row norms are invariant to hidden-unit permutation. Indexwise drift compares stored unit indices without neuron alignment; it is descriptive, not evidence of functional change."}),
    )
}
