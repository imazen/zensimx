//! Compact browser bank; numeric summaries stay with the Rust exporter.
use crate::{Result, bank::Bank, model::Input};
use std::{collections::BTreeSet, io::Write, path::Path};

pub(crate) fn bank(
    model: &zenpredict::Model,
    inputs: &[Input],
    bank: &Bank,
    out: &Path,
) -> Result<()> {
    if bank.rows.len() < 200 {
        return Err("compact bank requires 200 observed rows".into());
    }
    let slots: Vec<_> = inputs
        .iter()
        .flat_map(|i| [Some(i.feature), i.factor])
        .flatten()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let selected: Vec<_> = (0..200).map(|i| i * (bank.rows.len() - 1) / 199).collect();
    let mut sample = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(out.join("sample.f32le"))?;
    let mut quantiles = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(out.join("quantiles.f32le"))?;
    let mut quantization_score_max_delta = 0.0f64;
    let mut scorer = zensim::BakeScorer::new(model)?;
    for &index in &selected {
        let row = &bank.rows[index];
        let mut expanded = vec![0.0; slots.last().copied().ok_or("empty raw slots")? + 1];
        let mut original = expanded.clone();
        for &id in &slots {
            let value = row[id] as f32;
            if !value.is_finite() {
                return Err("f32 sample overflow".into());
            }
            sample.write_all(&value.to_le_bytes())?;
            expanded[id] = f64::from(value);
            original[id] = row[id];
        }
        let a = scorer.score_features(&expanded, 512, 512, None)?;
        let b = scorer.score_features(&original, 512, 512, None)?;
        quantization_score_max_delta = quantization_score_max_delta.max((a - b).abs());
    }
    let mut ranges = Vec::new();
    for &id in &slots {
        let values: Vec<_> = bank.rows.iter().map(|r| r[id]).collect();
        ranges.push([
            crate::audit::quantile(&values, 0.01),
            crate::audit::quantile(&values, 0.99),
        ]);
        for j in 0..=20 {
            quantiles.write_all(
                &(crate::audit::quantile(&values, j as f64 / 20.0) as f32).to_le_bytes(),
            )?;
        }
    }
    let mut model_quantiles = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(out.join("model-quantiles.f32le"))?;
    for input in inputs {
        let values: Vec<_> = bank
            .rows
            .iter()
            .map(|row| f64::from(input.value(row)))
            .collect();
        for j in 0..=20 {
            model_quantiles.write_all(
                &(crate::audit::quantile(&values, j as f64 / 20.0) as f32).to_le_bytes(),
            )?;
        }
    }
    crate::json(
        &out.join("bank.json"),
        &serde_json::json!({"schema":"zensim-pages-bank-v1",
        "rows":200,"population_rows":bank.total_rows,"slots":slots,
        "row_indices":selected.iter().map(|&i|bank.row_indices[i]).collect::<Vec<_>>(),
        "population_sha256":bank.sha256,"exporter_metric_source_sha256":env!("ZENSIM_VIZ_METRIC_SOURCE_SHA256"),"formula_revision":7,"feature_set_id":bank.provenance["feature_set_id"],"hdr_front_end":bank.provenance["hdr_front_end"],"ranges_p1_p99":ranges,"sample":"sample.f32le","quantiles":"quantiles.f32le","model_quantiles":"model-quantiles.f32le",
        "model_inputs":inputs.len(),"model_quantile_layout":"input-index-major-f32le p0,p5,...,p100; derived input before scaler",
        "fragility_formula":format!(r"b_i=1-\frac{{\overline{{g_r}}}}{{\overline{{g_r}}+{}}}",zensim::feature_v2::C_PJND_GRAD),
        "sample_layout":"row-major-f32le consumed raw registry slots",
        "quantile_layout":"slot-major-f32le 21 points p0,p5,...,p100",
        "quantization_score_max_delta":quantization_score_max_delta,
        "geometry":[512,512],"data_role":"TRAIN inspection; teacher targets are not human ratings"}),
    )
}
