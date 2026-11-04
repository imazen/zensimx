use crate::{Result, audit::quantile, bank::Bank, model::Input};
use serde::Serialize;
use zenpredict::Model;

#[derive(Serialize)]
pub(crate) struct Pdp {
    pub feature: usize,
    pub row_indices: Vec<usize>,
    pub family: Option<String>,
    pub features: Vec<usize>,
    pub axis_label: String,
    pub x: Vec<f64>,
    pub percentiles: Vec<f64>,
    pub mean: Vec<f64>,
    pub ice: Vec<Vec<f64>>,
    pub monotonicity_violations: Vec<(usize, usize, f64)>,
}
#[derive(Serialize)]
pub(crate) struct Bin {
    pub bin: usize,
    pub low: f64,
    pub high: f64,
    pub count: usize,
    pub median: f64,
    pub p10: f64,
    pub p90: f64,
    pub human: Option<[f64; 3]>,
}
#[derive(Serialize)]
pub(crate) struct Attribution {
    pub label: String,
    pub features: Vec<usize>,
    pub mean_abs_delta: f64,
    pub median_delta: f64,
    pub p10: f64,
    pub p90: f64,
}
#[derive(Serialize)]
pub(crate) struct Lens {
    pub bake_sha256: String,
    pub bank_sha256: String,
    pub bank_provenance: serde_json::Value,
    pub row_indices: Vec<usize>,
    pub scores: Vec<f64>,
    pub pdp: Pdp,
    pub separation: Vec<Bin>,
    pub separation_axis: String,
    pub attribution: Vec<Attribution>,
    pub input_attribution: Vec<Attribution>,
    pub attribution_definition: String,
    pub geometry: [u32; 2],
    pub raw_feature_intervention: bool,
}
fn score(s: &mut zensim::BakeScorer<'_>, row: &[f64]) -> Result<f64> {
    let v = s.score_features(row, 512, 512, None)?;
    if !v.is_finite() {
        return Err("nonfinite served score".into());
    }
    Ok(v)
}
fn curves(
    model: &Model,
    bank: &Bank,
    features: Vec<usize>,
    family: Option<&str>,
    x: Vec<f64>,
    percentiles: Vec<f64>,
) -> Result<Pdp> {
    if bank.rows.len() < 200 {
        return Err("PDP needs at least 200 observed rows".into());
    }
    let mut s = zensim::BakeScorer::new(model)?;
    let mut ice = Vec::new();
    let mut violations = Vec::new();
    let n = bank.rows.len().min(200);
    let selected: Vec<_> = (0..n)
        .map(|i| i * (bank.rows.len() - 1) / (n - 1))
        .collect();
    for (i, &selected_row) in selected.iter().enumerate() {
        let observed = &bank.rows[selected_row];
        let mut row = observed.clone();
        let mut curve = Vec::new();
        for &v in &x {
            for &id in &features {
                row[id] = if family.is_some() {
                    observed[id] * v
                } else {
                    v
                };
            }
            curve.push(score(&mut s, &row)?);
        }
        for j in 1..curve.len() {
            if curve[j] > curve[j - 1] {
                violations.push((i, j, curve[j] - curve[j - 1]));
            }
        }
        ice.push(curve);
    }
    let mean = (0..x.len())
        .map(|j| ice.iter().map(|r| r[j]).sum::<f64>() / ice.len() as f64)
        .collect();
    let feature = features[0];
    Ok(Pdp {
        feature,
        row_indices: selected.iter().map(|&i| bank.row_indices[i]).collect(),
        family: family.map(str::to_owned),
        features,
        axis_label: family.map_or_else(
            || format!("f{feature} · feature units"),
            |f| format!("{f} · multiplier of each observed damage value"),
        ),
        x,
        percentiles,
        mean,
        ice,
        monotonicity_violations: violations,
    })
}
pub(crate) fn pdp(model: &Model, bank: &Bank, feature: usize) -> Result<Pdp> {
    let values: Vec<_> = bank.rows.iter().map(|r| r[feature]).collect();
    let percentiles: Vec<_> = (0..21).map(|i| 0.01 + 0.98 * i as f64 / 20.0).collect();
    let x = percentiles.iter().map(|&p| quantile(&values, p)).collect();
    curves(model, bank, vec![feature], None, x, percentiles)
}
pub(crate) fn family_pdp(
    model: &Model,
    inputs: &[Input],
    bank: &Bank,
    family: &str,
) -> Result<Pdp> {
    let ids: std::collections::BTreeSet<_> = inputs
        .iter()
        .filter(|i| i.family == family)
        .map(|i| i.feature)
        .collect();
    if ids.is_empty() {
        return Err("unknown registry family".into());
    }
    curves(
        model,
        bank,
        ids.into_iter().collect(),
        Some(family),
        (0..21).map(|i| i as f64 / 10.0).collect(),
        Vec::new(),
    )
}
fn separation(values: &[f64], human: &[Option<f64>], scores: &[f64]) -> Vec<Bin> {
    let mut order: Vec<_> = (0..values.len()).collect();
    order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
    let edges: Vec<_> = (1..10).map(|q| quantile(values, q as f64 / 10.0)).collect();
    let mut buckets = vec![Vec::new(); 10];
    for i in order {
        let b = edges.partition_point(|v| values[i] > *v);
        buckets[b].push(i);
    }
    buckets
        .into_iter()
        .enumerate()
        .filter(|(_, b)| !b.is_empty())
        .map(|(bin, b)| {
            let v: Vec<_> = b.iter().map(|&i| scores[i]).collect();
            let h: Vec<_> = b.iter().filter_map(|&i| human[i]).collect();
            Bin {
                bin,
                low: values[b[0]],
                high: values[*b.last().unwrap()],
                count: b.len(),
                median: quantile(&v, 0.5),
                p10: quantile(&v, 0.1),
                p90: quantile(&v, 0.9),
                human: if h.len() == b.len() {
                    Some([quantile(&h, 0.5), quantile(&h, 0.1), quantile(&h, 0.9)])
                } else {
                    None
                },
            }
        })
        .collect()
}
fn ablate(
    model: &Model,
    bank: &Bank,
    scores: &[f64],
    label: String,
    ids: Vec<usize>,
) -> Result<Attribution> {
    let mut s = zensim::BakeScorer::new(model)?;
    let mut delta = Vec::new();
    for (r, &base) in bank.rows.iter().zip(scores) {
        let mut row = r.clone();
        for &id in &ids {
            row[id] = 0.0;
        }
        delta.push(score(&mut s, &row)? - base);
    }
    Ok(Attribution {
        label,
        features: ids,
        mean_abs_delta: delta.iter().map(|v| v.abs()).sum::<f64>() / delta.len() as f64,
        median_delta: quantile(&delta, 0.5),
        p10: quantile(&delta, 0.1),
        p90: quantile(&delta, 0.9),
    })
}
#[derive(Serialize)]
pub(crate) struct CurveView {
    pub pdp: Pdp,
    pub separation: Vec<Bin>,
    pub separation_axis: String,
}
pub(crate) fn view(
    model: &Model,
    inputs: &[Input],
    bank: &Bank,
    feature: usize,
    family: Option<&str>,
    scores: &[f64],
) -> Result<CurveView> {
    let pdp = if let Some(family) = family {
        family_pdp(model, inputs, bank, family)?
    } else {
        pdp(model, bank, feature)?
    };
    let (mut values, separation_axis) = (
        bank.rows.iter().map(|r| r[feature]).collect::<Vec<_>>(),
        format!("f{feature} · feature units"),
    );
    let separation_axis = if let Some(family) = family {
        let direct: Vec<_> = inputs
            .iter()
            .filter(|i| i.family == family && i.factor.is_none())
            .collect();
        if direct.is_empty() {
            return Err("family has no direct damage inputs".into());
        }
        values = bank
            .rows
            .iter()
            .map(|r| {
                (direct
                    .iter()
                    .map(|i| (r[i.feature] / f64::from(model.scaler_scale()[i.index])).powi(2))
                    .sum::<f64>()
                    / direct.len() as f64)
                    .sqrt()
            })
            .collect();
        format!("{family} · RMS damage in scaler-standardized units")
    } else {
        separation_axis
    };
    let separation = separation(&values, &bank.human, scores);
    Ok(CurveView {
        pdp,
        separation,
        separation_axis,
    })
}
pub(crate) fn run(
    model: &Model,
    inputs: &[Input],
    bank: &Bank,
    feature: usize,
    family: Option<&str>,
    bake_sha256: String,
    checkpoint: Option<&std::path::Path>,
) -> Result<Lens> {
    if !inputs.iter().any(|i| i.feature == feature) {
        return Err("selected feature is not a model damage input".into());
    }
    let mut s = zensim::BakeScorer::new(model)?;
    let scores = bank
        .rows
        .iter()
        .map(|r| score(&mut s, r))
        .collect::<Result<Vec<_>>>()?;
    let CurveView {
        pdp,
        separation,
        separation_axis,
    } = view(model, inputs, bank, feature, family, &scores)?;
    let mut groups = std::collections::BTreeMap::<String, std::collections::BTreeSet<usize>>::new();
    for input in inputs {
        if input.form != "difference" {
            return Err("identity ablation requires difference-form raw inputs".into());
        }
        groups
            .entry(input.family.clone())
            .or_default()
            .insert(input.feature);
    }
    use std::io::Write;
    let mut ledger = checkpoint
        .map(|out| {
            std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(out.join("ablations.jsonl"))
        })
        .transpose()?;
    let mut completed = 0usize;
    let required = groups.len() + inputs.iter().filter(|i| i.factor.is_none()).count();
    let mut record = |kind: &str, result: &Attribution| -> Result<()> {
        completed += 1;
        if let Some(file) = ledger.as_mut() {
            serde_json::to_writer(
                &mut *file,
                &serde_json::json!({"kind":kind,"result":result}),
            )?;
            writeln!(file)?;
            file.flush()?;
        }
        if let Some(out) = checkpoint {
            let data = serde_json::json!({"stage":"ablations","completed":completed,"required":required,"last":result.label,"pid":std::process::id(),"bake_sha256":bake_sha256,"bank_sha256":bank.sha256});
            let temp = out.join(".progress.next");
            std::fs::write(&temp, serde_json::to_vec(&data)?)?;
            std::fs::rename(temp, out.join("progress.json"))?;
        }
        Ok(())
    };
    let mut attribution = Vec::new();
    for (label, ids) in groups {
        let result = ablate(model, bank, &scores, label, ids.into_iter().collect())?;
        record("family", &result)?;
        attribution.push(result);
    }
    attribution.sort_by(|a, b| b.mean_abs_delta.total_cmp(&a.mean_abs_delta));
    let mut input_attribution = Vec::new();
    for input in inputs.iter().filter(|i| i.factor.is_none()) {
        let result = ablate(
            model,
            bank,
            &scores,
            format!("f{} · {}", input.feature, input.name),
            vec![input.feature],
        )?;
        record("input", &result)?;
        input_attribution.push(result);
    }
    input_attribution.sort_by(|a, b| b.mean_abs_delta.total_cmp(&a.mean_abs_delta));
    Ok(Lens{bake_sha256,bank_sha256:bank.sha256.clone(),bank_provenance:bank.provenance.clone(),row_indices:bank.row_indices.clone(),scores,pdp,separation,separation_axis,attribution,input_attribution,attribution_definition:"Set damage slots to their exact identity value 0; retain each row's reference-only fragility. BakeScorer recomputes every derived product. Family effects overlap and are not additive. Input ranking is by raw damage slot, jointly removing its derived products.".into(),geometry:[512,512],raw_feature_intervention:true})
}
pub(crate) fn bulk(
    model: &Model,
    inputs: &[Input],
    probes: &[Input],
    bank: &Bank,
    bake_sha256: String,
    out: &std::path::Path,
) -> Result<()> {
    let base = run(
        model,
        inputs,
        bank,
        inputs[0].feature,
        None,
        bake_sha256.clone(),
        Some(out),
    )?;
    let mut summary = serde_json::to_value(&base)?;
    let object = summary.as_object_mut().unwrap();
    object.remove("pdp");
    object.remove("separation");
    object.remove("separation_axis");
    crate::json(&out.join("summary.json"), &summary)?;
    let ids: std::collections::BTreeSet<_> = probes.iter().map(|i| i.feature).collect();
    let families: std::collections::BTreeSet<_> =
        inputs.iter().map(|i| i.family.as_str()).collect();
    let total = ids.len() + families.len();
    let mut files = serde_json::Map::new();
    let mut family_files = serde_json::Map::new();
    let mut flags = 0;
    let mut completed = 0;
    for id in ids {
        let data = view(model, inputs, bank, id, None, &base.scores)?;
        flags += data.pdp.monotonicity_violations.len();
        let name = format!("curve-f{id}.json");
        crate::json(&out.join(&name), &data)?;
        files.insert(format!("f{id}"), serde_json::json!(name));
        completed += 1;
        println!("curve f{id}: {completed}/{total}");
        let progress = serde_json::json!({"stage":"feature curves","bake_sha256":bake_sha256,"bank_sha256":bank.sha256,"completed":completed,"required":total,"last_feature":id,"pid":std::process::id()});
        std::fs::write(out.join("progress.json"), serde_json::to_vec(&progress)?)?;
    }
    for family in families {
        let data = view(
            model,
            inputs,
            bank,
            inputs[0].feature,
            Some(family),
            &base.scores,
        )?;
        flags += data.pdp.monotonicity_violations.len();
        let name = format!("curve-family-{family}.json");
        crate::json(&out.join(&name), &data)?;
        family_files.insert(family.into(), serde_json::json!(name));
        completed += 1;
        println!("family {family}: {completed}/{total}");
    }
    crate::json(
        &out.join("index.json"),
        &serde_json::json!({"bake_sha256":bake_sha256,"bank_sha256":bank.sha256,"bank_rows":bank.total_rows,"observed_rows":bank.rows.len(),"ice_rows":bank.rows.len().min(200),"summary":"summary.json","lenses":files,"family_lenses":family_files,"completed":completed,"required":total,"monotonicity_flag_count":flags}),
    )?;
    Ok(())
}
#[cfg(all(test, feature = "real-bake-tests"))]
mod tests {
    use super::*;
    use std::path::Path;
    #[test]
    fn every_exported_ice_point_equals_direct_served_score() {
        let bake = std::env::var("VIZ_F_BAKE").expect("VIZ_F_BAKE required");
        let path = std::env::var("VIZ_F_BANK").expect("VIZ_F_BANK required");
        let m = Model::from_bytes(&std::fs::read(bake).unwrap()).unwrap();
        let inputs = crate::model::inputs(&m).unwrap();
        let bank = crate::bank::load(Path::new(&path), &inputs, 200).unwrap();
        let feature = inputs[0].feature;
        let p = pdp(&m, &bank, feature).unwrap();
        let mut direct = zensim::BakeScorer::new(&m).unwrap();
        for (i, row) in bank.rows.iter().enumerate() {
            for (j, &x) in p.x.iter().enumerate() {
                let mut changed = row.clone();
                changed[feature] = x;
                assert_eq!(
                    p.ice[i][j].to_bits(),
                    direct
                        .score_features(&changed, 512, 512, None)
                        .unwrap()
                        .to_bits()
                );
            }
        }
        let family = inputs[0].family.as_str();
        let family_curve = family_pdp(&m, &inputs, &bank, family).unwrap();
        for (i, observed) in bank.rows.iter().enumerate() {
            for (j, &multiplier) in family_curve.x.iter().enumerate() {
                let mut changed = observed.clone();
                for &id in &family_curve.features {
                    changed[id] = observed[id] * multiplier;
                }
                assert_eq!(
                    family_curve.ice[i][j].to_bits(),
                    direct
                        .score_features(&changed, 512, 512, None)
                        .unwrap()
                        .to_bits()
                );
            }
        }
    }
}
