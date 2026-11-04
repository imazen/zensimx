use crate::{
    Result,
    bank::Bank,
    model::{self, Input},
};
use serde::Serialize;
use std::{collections::BTreeMap, io::Write, path::Path};
use zenpredict::{Activation, Model};
const EPS: f64 = 1e-6;
const Z: f64 = 3.5;
const MODIFIED_Z_SCALE: f64 = 0.6744897501960817;

#[derive(Serialize)]
pub(crate) struct Flag {
    pub kind: String,
    pub severity: String,
    pub location: String,
    pub layer: Option<usize>,
    pub from: Option<usize>,
    pub to: Option<usize>,
    pub value: f64,
    pub statistic: Option<f64>,
    pub threshold: f64,
    pub reason: String,
}
#[derive(Serialize)]
pub(crate) struct Unit {
    pub unit: usize,
    pub activation_rate: f64,
    pub mean_abs_activation: f64,
    pub min: f32,
    pub max: f32,
    pub top_inputs: Vec<(usize, f32)>,
}
#[derive(Serialize)]
pub(crate) struct Audit {
    pub schema: u32,
    pub bake_sha256: String,
    pub bank_sha256: String,
    pub bank_path: String,
    pub bank_provenance: serde_json::Value,
    pub bank_rows: usize,
    pub sampled_rows: usize,
    pub sampled_row_indices: Vec<usize>,
    pub weight_count: usize,
    pub counts: BTreeMap<String, usize>,
    pub flags: Vec<Flag>,
    pub hidden_units: Vec<Unit>,
    pub thresholds: serde_json::Value,
    pub constraints: serde_json::Value,
}
struct Scope {
    layer: Option<usize>,
    from: Option<usize>,
    to: Option<usize>,
}
impl Scope {
    fn input(i: usize) -> Self {
        Self {
            layer: Some(0),
            from: Some(i),
            to: None,
        }
    }
    fn weight(l: usize, i: usize, j: usize) -> Self {
        Self {
            layer: Some(l),
            from: Some(i),
            to: Some(j),
        }
    }
    fn location(&self) -> String {
        match (self.layer, self.from, self.to) {
            (Some(l), Some(i), Some(j)) => format!("layer {l}, input {i} → unit {j}"),
            (Some(l), Some(i), None) => format!("layer {l}, input {i}"),
            (Some(l), None, Some(j)) => format!("layer {l}, unit {j}"),
            _ => "model".into(),
        }
    }
}
#[expect(
    clippy::too_many_arguments,
    reason = "flag fields mirror the auditable JSON schema at each call site"
)]
fn flag(
    flags: &mut Vec<Flag>,
    kind: &str,
    severity: &str,
    s: Scope,
    value: f64,
    statistic: Option<f64>,
    threshold: f64,
    reason: &str,
) {
    flags.push(Flag {
        kind: kind.into(),
        severity: severity.into(),
        location: s.location(),
        layer: s.layer,
        from: s.from,
        to: s.to,
        value,
        statistic,
        threshold,
        reason: reason.into(),
    });
}
pub(crate) fn quantile(values: &[f64], p: f64) -> f64 {
    assert!(!values.is_empty());
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let pos = (sorted.len() - 1) as f64 * p;
    let i = pos.floor() as usize;
    let j = (i + 1).min(sorted.len() - 1);
    sorted[i] + (sorted[j] - sorted[i]) * (pos - i as f64)
}
fn center(values: &[f64]) -> (f64, f64) {
    let m = quantile(values, 0.5);
    let dev: Vec<_> = values.iter().map(|v| (v - m).abs()).collect();
    (m, quantile(&dev, 0.5))
}
fn robust(flags: &mut Vec<Flag>, kind: &str, s: Scope, v: f64, c: (f64, f64)) {
    if c.1 == 0.0 {
        if (v - c.0).abs() > EPS {
            flag(
                flags,
                kind,
                "note",
                s,
                v,
                None,
                Z,
                "MAD is zero; nonzero deviation from median (no finite robust z-score)",
            );
        }
    } else {
        let z = MODIFIED_Z_SCALE * (v - c.0) / c.1;
        if z.abs() > Z {
            flag(
                flags,
                kind,
                "note",
                s,
                v,
                Some(z),
                Z,
                "absolute modified z-score exceeds threshold; diagnostic, not a correctness defect",
            );
        }
    }
}
pub(crate) fn weight_statistics(
    weights: &[f32],
    in_dim: usize,
    out_dim: usize,
) -> serde_json::Value {
    let rows: Vec<_> = weights
        .chunks(out_dim)
        .map(|r| center(&r.iter().map(|&v| f64::from(v)).collect::<Vec<_>>()))
        .collect();
    let columns: Vec<_> = (0..out_dim)
        .map(|j| {
            center(
                &(0..in_dim)
                    .map(|i| f64::from(weights[i * out_dim + j]))
                    .collect::<Vec<_>>(),
            )
        })
        .collect();
    let z = |value: f32, c: (f64, f64)| {
        if c.1 == 0.0 {
            None
        } else {
            Some(MODIFIED_Z_SCALE * (f64::from(value) - c.0) / c.1)
        }
    };
    serde_json::json!({"rows":rows,"columns":columns,"row_z":weights.iter().enumerate().map(|(k,&v)|z(v,rows[k/out_dim])).collect::<Vec<_>>(),"column_z":weights.iter().enumerate().map(|(k,&v)|z(v,columns[k%out_dim])).collect::<Vec<_>>(),"zero_mad":"z is undefined and serialized as null","modified_z_threshold":Z})
}
pub(crate) fn run(
    model: &Model,
    inputs: &[Input],
    bank: &Bank,
    training: Option<&Bank>,
    bake_sha256: String,
    constraints: &crate::constraints::Constraints,
) -> Result<Audit> {
    let mut flags = Vec::new();
    if constraints.certificate["certified"] != true {
        flag(
            &mut flags,
            "certificate",
            "critical",
            Scope {
                layer: None,
                from: None,
                to: None,
            },
            0.0,
            None,
            0.0,
            constraints.certificate["reason"]
                .as_str()
                .unwrap_or("canonical certificate refused"),
        );
    }
    let mut weight_count = 0;
    for l in 0..model.n_layers() {
        let layer = model.layer(l);
        let w = model::weights(model, l);
        weight_count += w.len();
        let rows: Vec<_> = w
            .chunks(layer.out_dim)
            .map(|r| center(&r.iter().map(|&v| f64::from(v)).collect::<Vec<_>>()))
            .collect();
        let cols: Vec<_> = (0..layer.out_dim)
            .map(|j| {
                center(
                    &(0..layer.in_dim)
                        .map(|i| f64::from(w[i * layer.out_dim + j]))
                        .collect::<Vec<_>>(),
                )
            })
            .collect();
        for i in 0..layer.in_dim {
            for j in 0..layer.out_dim {
                let v = f64::from(w[i * layer.out_dim + j]);
                if !v.is_finite() {
                    return Err("nonfinite model weight".into());
                }
                // The target family uses positive damage→hidden and negative
                // hidden→distance-oriented score weights. Report signs in
                // both layers instead of silently trusting the first layer.
                if (l == 0 && v < 0.0 && !constraints.exempt.contains(&(i, j)))
                    || (l == model.n_layers() - 1 && l > 0 && v > 0.0)
                {
                    flag(
                        &mut flags,
                        "sign",
                        "critical",
                        Scope::weight(l, i, j),
                        v,
                        Some(v),
                        0.0,
                        "weight violates the monotone damage network sign constraint",
                    );
                }
                robust(
                    &mut flags,
                    "row_outlier",
                    Scope::weight(l, i, j),
                    v,
                    rows[i],
                );
                robust(
                    &mut flags,
                    "column_outlier",
                    Scope::weight(l, i, j),
                    v,
                    cols[j],
                );
            }
            let maximum = w[i * layer.out_dim..(i + 1) * layer.out_dim]
                .iter()
                .map(|v| f64::from(v.abs()))
                .fold(0.0, f64::max);
            if l == 0 && maximum == 0.0 {
                flag(
                    &mut flags,
                    "dead-input",
                    "critical",
                    Scope::input(i),
                    0.0,
                    Some(0.0),
                    0.0,
                    "all incoming-model-row outgoing weights are exactly zero",
                );
            }
            if maximum <= EPS {
                flag(
                    &mut flags,
                    "dead_input",
                    "critical",
                    Scope {
                        layer: Some(l),
                        from: Some(i),
                        to: None,
                    },
                    maximum,
                    Some(maximum),
                    EPS,
                    "all outgoing weights are within the near-zero threshold",
                );
            }
        }
        let biases: Vec<_> = layer.biases.iter().map(|&v| f64::from(v)).collect();
        let c = center(&biases);
        for (j, &v) in biases.iter().enumerate() {
            robust(
                &mut flags,
                "bias_outlier",
                Scope {
                    layer: Some(l),
                    from: None,
                    to: Some(j),
                },
                v,
                c,
            );
        }
    }
    for (i, &scale) in model.scaler_scale().iter().enumerate() {
        if scale < 0.0 {
            flag(
                &mut flags,
                "scaler_sign",
                "critical",
                Scope::input(i),
                f64::from(scale),
                Some(f64::from(scale)),
                0.0,
                "negative scaler reverses a damage input's effective sign",
            );
        }
    }
    let columns: Vec<Vec<f64>> = inputs
        .iter()
        .map(|i| bank.rows.iter().map(|r| f64::from(i.value(r))).collect())
        .collect();
    let means: Vec<_> = columns
        .iter()
        .map(|c| c.iter().sum::<f64>() / c.len() as f64)
        .collect();
    let norms: Vec<_> = columns
        .iter()
        .zip(&means)
        .map(|(c, m)| c.iter().map(|v| (v - m) * (v - m)).sum::<f64>().sqrt())
        .collect();
    let normalized: Vec<Vec<f64>> = columns
        .iter()
        .enumerate()
        .map(|(i, c)| {
            c.iter()
                .map(|v| {
                    if norms[i] == 0.0 {
                        0.0
                    } else {
                        (v - means[i]) / norms[i]
                    }
                })
                .collect()
        })
        .collect();
    for (i, c) in columns.iter().enumerate() {
        if norms[i] == 0.0 {
            flag(
                &mut flags,
                "constant_input",
                "critical",
                Scope::input(i),
                c[0],
                Some(0.0),
                0.0,
                "input is constant across the recorded bank sample",
            );
        }
        if let Some(t) = training {
            let tc: Vec<_> = t
                .rows
                .iter()
                .map(|r| f64::from(inputs[i].value(r)))
                .collect();
            let lo = quantile(&tc, 0.01);
            let hi = quantile(&tc, 0.99);
            let outside = c.iter().filter(|&&v| v < lo || v > hi).count();
            if outside > 0 {
                flag(
                    &mut flags,
                    "drift",
                    "critical",
                    Scope::input(i),
                    outside as f64,
                    Some(outside as f64 / c.len() as f64),
                    0.0,
                    &format!(
                        "sample observations outside training sample p1={lo}, p99={hi}; descriptive tail exposure, not a test"
                    ),
                );
            }
        }
        if norms[i] == 0.0 {
            continue;
        }
        for j in 0..i {
            if norms[j] == 0.0 {
                continue;
            }
            let corr = zenstats::pearson(&normalized[i], &normalized[j]);
            if corr.abs() > 0.999 {
                flag(
                    &mut flags,
                    "duplicate_input",
                    "critical",
                    Scope::input(i),
                    corr,
                    Some(corr),
                    0.999,
                    &format!(
                        "absolute sample Pearson correlation with input {j} exceeds threshold"
                    ),
                );
            }
        }
    }
    let layer = model.layer(0);
    let weights = model::weights(model, 0);
    let hidden_model = model::layer_model(model, 0)?;
    let output_model = model::layer_model(model, 1)?;
    let mut hidden = zenpredict::Predictor::new(&hidden_model);
    let mut output = zenpredict::Predictor::new(&output_model);
    let mut complete = zenpredict::Predictor::new(model);
    let mut activations = Vec::new();
    for row in &bank.rows {
        let input: Vec<_> = inputs.iter().map(|i| i.value(row)).collect();
        let h = hidden.predict(&input)?.to_vec();
        let separate = output.predict(&h)?;
        let direct = complete.predict(&input)?;
        if separate
            .iter()
            .zip(direct)
            .any(|(a, b)| a.to_bits() != b.to_bits())
        {
            return Err("diagnostic layer execution differs from native full forward bits".into());
        }
        activations.push(h);
    }
    let mut units = Vec::new();
    for j in 0..layer.out_dim {
        let vals: Vec<_> = activations.iter().map(|a| a[j]).collect();
        let active = vals.iter().filter(|&&v| v > 0.0).count();
        let min = vals.iter().copied().fold(f32::INFINITY, f32::min);
        let max = vals.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut top: Vec<_> = (0..layer.in_dim)
            .map(|i| (i, weights[i * layer.out_dim + j]))
            .collect();
        top.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()));
        top.truncate(8);
        units.push(Unit {
            unit: j,
            activation_rate: active as f64 / vals.len() as f64,
            mean_abs_activation: vals.iter().map(|v| f64::from(v.abs())).sum::<f64>()
                / vals.len() as f64,
            min,
            max,
            top_inputs: top,
        });
        if vals.iter().all(|v| v.abs() <= EPS as f32) {
            flag(
                &mut flags,
                "dead_hidden",
                "critical",
                Scope {
                    layer: Some(0),
                    from: None,
                    to: Some(j),
                },
                f64::from(max),
                Some(0.0),
                EPS,
                "hidden unit is near zero over the recorded sample",
            );
        }
        // ReLU/smooth hinge have a flat inactive region; positive ReLU is
        // linear, not saturated. Identity/leaky ReLU have no flat tail.
        if matches!(layer.activation, Activation::Relu | Activation::SmoothHinge) && active == 0 {
            flag(
                &mut flags,
                "saturated_hidden",
                "critical",
                Scope {
                    layer: Some(0),
                    from: None,
                    to: Some(j),
                },
                f64::from(max),
                Some(0.0),
                0.0,
                "hidden unit is always in its flat inactive region in this sample",
            );
        }
    }
    let mut counts = BTreeMap::new();
    for kind in [
        "dead-input",
        "sign",
        "row_outlier",
        "column_outlier",
        "bias_outlier",
        "dead_input",
        "constant_input",
        "dead_hidden",
        "saturated_hidden",
        "duplicate_input",
        "drift",
        "scaler_sign",
    ] {
        counts.insert(kind.into(), 0);
    }
    for f in &flags {
        *counts.entry(f.kind.clone()).or_default() += 1;
    }
    let severity = |s: &str| match s {
        "critical" => 0,
        "serious" => 1,
        "warning" => 2,
        _ => 3,
    };
    flags.sort_by(|a, b| {
        severity(&a.severity)
            .cmp(&severity(&b.severity))
            .then_with(|| {
                b.statistic
                    .unwrap_or(b.value)
                    .abs()
                    .total_cmp(&a.statistic.unwrap_or(a.value).abs())
            })
    });
    Ok(Audit {
        schema: 1,
        bake_sha256,
        bank_sha256: bank.sha256.clone(),
        bank_path: bank.path.clone(),
        bank_provenance: bank.provenance.clone(),
        bank_rows: bank.total_rows,
        sampled_rows: bank.rows.len(),
        sampled_row_indices: bank.row_indices.clone(),
        weight_count,
        constraints: serde_json::to_value(constraints)?,
        counts,
        flags,
        hidden_units: units,
        thresholds: serde_json::json!({"near_zero":EPS,"modified_z":Z,"duplicate_abs_pearson":0.999,"drift_training_percentiles":[0.01,0.99],"drift_available":training.is_some(),"bank_scope":if bank.rows.len()==bank.total_rows {"every observed bank row"} else {"deterministic evenly spaced observed sample; not whole-bank certification"},"zero_mad":"nonzero median deviations flagged with null statistic; no fabricated finite z-score"}),
    })
}
pub(crate) fn write_tsv(path: &Path, a: &Audit) -> Result<()> {
    let mut f = std::io::BufWriter::new(
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?,
    );
    writeln!(
        f,
        "severity\tkind\tlocation\tvalue\tstatistic\tthreshold\treason"
    )?;
    for v in &a.flags {
        writeln!(
            f,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}",
            v.severity,
            v.kind,
            v.location,
            v.value,
            v.statistic
                .map_or_else(|| "undefined (MAD=0)".into(), |v| v.to_string()),
            v.threshold,
            v.reason
        )?;
    }
    Ok(())
}
// Presentation severity may change; every original measured flag must survive.
pub(crate) fn verify_prior(audit: &Audit, baseline: &Path, out: &Path) -> Result<()> {
    let prior: serde_json::Value = serde_json::from_reader(std::fs::File::open(baseline)?)?;
    if prior["bake_sha256"] != audit.bake_sha256 || prior["bank_sha256"] != audit.bank_sha256 {
        return Err("audit baseline identity differs".into());
    }
    let key = |mut value: serde_json::Value| -> Result<String> {
        value
            .as_object_mut()
            .ok_or("invalid flag")?
            .remove("severity");
        Ok(serde_json::to_string(&value)?)
    };
    let mut remaining = BTreeMap::<String, usize>::new();
    for f in &audit.flags {
        *remaining.entry(key(serde_json::to_value(f)?)?).or_default() += 1;
    }
    let flags = prior["flags"].as_array().ok_or("baseline flags missing")?;
    for f in flags {
        let k = key(f.clone())?;
        let count = remaining
            .get_mut(&k)
            .ok_or("an original measured flag was dropped or changed")?;
        if *count == 0 {
            return Err("an original duplicate flag was dropped".into());
        }
        *count -= 1;
    }
    for (kind, count) in prior["counts"]
        .as_object()
        .ok_or("baseline counts missing")?
    {
        if kind != "dead-input" && count.as_u64() != audit.counts.get(kind).map(|&v| v as u64) {
            return Err(format!("original {kind} count changed").into());
        }
    }
    crate::json(
        &out.join("flag-preservation.json"),
        &serde_json::json!({"baseline_sha256":crate::hash(baseline)?,"original_flags":flags.len(),"matched_original_flags":flags.len(),"new_flags":audit.flags.len()-flags.len(),"all_original_numeric_fields_unchanged":true,"severity_presentation_allowed_to_change":true}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn robust_outlier_and_zero_mad_are_explicit() {
        let mut f = Vec::new();
        robust(&mut f, "row", Scope::input(0), 8.0, (1.0, 1.0));
        assert_eq!(f.len(), 1);
        assert!(f[0].statistic.unwrap() > Z);
        robust(&mut f, "row", Scope::input(0), 2.0, (0.0, 0.0));
        assert!(f[1].statistic.is_none());
        robust(&mut f, "row", Scope::input(0), 0.0, (0.0, 0.0));
        assert_eq!(f.len(), 2);
    }
    #[test]
    fn interpolated_quantiles_preserve_observations() {
        assert_eq!(quantile(&[9.0, 1.0, 5.0], 0.5), 5.0);
        assert_eq!(quantile(&[0.0, 10.0], 0.1), 1.0);
    }
    #[cfg(feature = "real-bake-tests")]
    #[test]
    fn f_bake_has_no_sign_violation() {
        let path = std::env::var("VIZ_F_BAKE").expect("caller must set VIZ_F_BAKE");
        let bank_path = std::env::var("VIZ_F_BANK").expect("caller must set VIZ_F_BANK");
        let m = Model::from_bytes(&std::fs::read(&path).unwrap()).unwrap();
        let inputs = model::inputs(&m).unwrap();
        let b = crate::bank::load(Path::new(&bank_path), &inputs, usize::MAX).unwrap();
        let constraints = crate::constraints::inspect(&m, &inputs, Path::new(&path)).unwrap();
        let a = run(
            &m,
            &inputs,
            &b,
            None,
            crate::hash(Path::new(&path)).unwrap(),
            &constraints,
        )
        .unwrap();
        assert_eq!(a.counts["sign"], 0);
        assert_eq!(a.counts["scaler_sign"], 0);
        assert_eq!(a.counts["dead-input"], 0);
        let weights = model::weights_document(&m, &inputs);
        assert_eq!(weights["nonzero_counts"][371], 33);
        assert_eq!(
            weights["capacity"]["counts"],
            serde_json::json!({"live":2,"residual":31,"dead":95})
        );
        assert_eq!(weights["capacity"]["units"].as_array().unwrap().len(), 128);
        assert!(
            !a.flags
                .iter()
                .any(|f| f.kind == "dead-input" && f.from == Some(371))
        );
        let expected: serde_json::Value =
            serde_json::from_str(include_str!("../f-audit-census.json")).unwrap();
        assert_eq!(a.bake_sha256, expected["bake_sha256"].as_str().unwrap());
        assert_eq!(a.bank_sha256, expected["bank_sha256"].as_str().unwrap());
        assert_eq!(
            a.bank_rows,
            expected["bank_rows"].as_u64().unwrap() as usize
        );
        assert_eq!(a.sampled_rows, a.bank_rows);
        assert_eq!(
            a.weight_count,
            expected["weight_count"].as_u64().unwrap() as usize
        );
        let expected: BTreeMap<String, usize> =
            serde_json::from_value(expected["counts"].clone()).unwrap();
        assert_eq!(a.counts, expected);
    }
}
