use crate::{Result, bank::Bank, model::Input, pixels};
use std::{collections::BTreeMap, path::Path, process::Command};
use zenpredict::Model;

pub(crate) fn pass(
    bake: &Path,
    reference: &Path,
    distorted: &Path,
    out: &Path,
    row_padding: usize,
) -> Result<()> {
    let bytes = std::fs::read(bake)?;
    let req = zensim::research::Request::for_bake_bytes(&bytes)?;
    let a = pixels::decode(reference)?.with_padding(row_padding);
    let b = pixels::decode(distorted)?.with_padding(row_padding);
    let result = zensim::research::extract(&req, &a, &b)?;
    let mut doc: serde_json::Value = serde_json::from_str(&result.manifest_json())?;
    doc["values"] = serde_json::json!(result.values());
    doc["bits"] = serde_json::json!(
        result
            .values()
            .iter()
            .map(|v| v.to_bits().to_string())
            .collect::<Vec<_>>()
    );
    crate::json(out, &doc)
}
pub(crate) struct Paths<'a> {
    pub bake: &'a Path,
    pub reference: &'a Path,
    pub distorted: &'a Path,
    pub out: &'a Path,
}
pub(crate) fn run(
    model: &Model,
    inputs: &[Input],
    bank: &Bank,
    paths: Paths<'_>,
    row_padding: usize,
    percentile_banks: &[Bank],
) -> Result<()> {
    let Paths {
        bake,
        reference,
        distorted,
        out,
    } = paths;
    let maps = out.join("maps");
    std::fs::create_dir_all(&maps)?;
    let bake = std::fs::canonicalize(bake)?;
    let a = std::fs::canonicalize(reference)?;
    let b = std::fs::canonicalize(distorted)?;
    let out = std::fs::canonicalize(out)?;
    // Two child processes give immutable process-global formula/dispatch state.
    // Only one has the retained-map hook enabled. No unsafe env mutation.
    for instrumented in [false, true] {
        let file = out.join(if instrumented {
            "instrumented.json"
        } else {
            "control.json"
        });
        let mut child = Command::new(std::env::current_exe()?);
        child
            .arg("atlas-pass")
            .arg(&bake)
            .arg("--reference")
            .arg(&a)
            .arg("--distorted")
            .arg(&b)
            .arg("--out")
            .arg(&file)
            .arg("--row-padding")
            .arg(row_padding.to_string())
            .env("ZENSIM_FORMULA_REV", "7");
        if instrumented {
            child.env("ZENSIM_VIZ_MAP_DIR", &maps);
        } else {
            child.env_remove("ZENSIM_VIZ_MAP_DIR");
        }
        if !child.status()?.success() {
            return Err("atlas extraction child failed".into());
        }
    }
    let measured: serde_json::Value =
        serde_json::from_slice(&std::fs::read(out.join("instrumented.json"))?)?;
    let control: serde_json::Value =
        serde_json::from_slice(&std::fs::read(out.join("control.json"))?)?;
    if measured["bits"] != control["bits"] {
        return Err("retained map export changed pooled feature bits".into());
    }
    let records: Vec<serde_json::Value> =
        serde_json::from_slice(&std::fs::read(maps.join("maps.json"))?)?;
    // Display quantiles describe the actual encoded, sampled map pixels.
    // Cache by shared detector filename; use the same quantile owner as PDP.
    let mut display_ranges = BTreeMap::new();
    let records = records
        .into_iter()
        .map(|mut record| -> Result<_> {
            if let Some(file) = record["map_file"].as_str() {
                let range = if let Some(range) = display_ranges.get(file) {
                    *range
                } else {
                    let image = pixels::decode(&maps.join(file))?;
                    let lo = record["map_range"][0]
                        .as_f64()
                        .ok_or("missing map minimum")?;
                    let hi = record["map_range"][1]
                        .as_f64()
                        .ok_or("missing map maximum")?;
                    let values: Vec<_> = (0..image.height)
                        .flat_map(|y| image.row(y))
                        .map(|p| lo + f64::from(p[0]) / 255.0 * (hi - lo))
                        .collect();
                    let range = [
                        crate::audit::quantile(&values, 0.02),
                        crate::audit::quantile(&values, 0.98),
                    ];
                    display_ranges.insert(file.to_owned(), range);
                    range
                };
                record["map_robust_range"] = serde_json::json!(range);
            }
            Ok(record)
        })
        .collect::<Result<Vec<_>>>()?;
    let map_by_id: BTreeMap<_, _> = records
        .into_iter()
        .map(|r| (r["id"].as_u64().unwrap() as usize, r))
        .collect();
    let av = pixels::decode(&a)?;
    let bv = pixels::decode(&b)?;
    if av.width != bv.width || av.height != bv.height {
        return Err("pair dimensions differ".into());
    }
    let diff = pixels::RgbImage {
        pixels: av
            .pixels
            .iter()
            .zip(&bv.pixels)
            .map(|(a, b)| {
                [
                    a[0].abs_diff(b[0]),
                    a[1].abs_diff(b[1]),
                    a[2].abs_diff(b[2]),
                ]
            })
            .collect(),
        width: av.width,
        height: av.height,
        stride: av.width,
    };
    for (name, image) in [
        ("reference.png", &av),
        ("distorted.png", &bv),
        ("difference.png", &diff),
    ] {
        pixels::write(&out.join(name), &pixels::preview(image))?;
    }
    let req = zensim::research::Request::for_bake_bytes(&std::fs::read(&bake)?)?;
    let extraction = zensim::research::extract(
        &req,
        &zensim::source::RgbSlice::new(&av.pixels, av.width, av.height),
        &zensim::source::RgbSlice::new(&bv.pixels, bv.width, bv.height),
    )?;
    let values = extraction.values();
    let provenance: BTreeMap<_, _> = extraction
        .provenance()
        .iter()
        .map(|p| (usize::from(p.id), p))
        .collect();
    let mut features = Vec::new();
    let pdp_percentiles = crate::lens::pdp_percentiles();
    let plot_range = |values: &[f64]| {
        [
            crate::audit::quantile(values, pdp_percentiles[0]),
            crate::audit::quantile(values, *pdp_percentiles.last().unwrap()),
        ]
    };
    for input in inputs {
        let p = provenance
            .get(&input.feature)
            .ok_or("missing input provenance")?;
        if p.form != "difference" {
            return Err(
                "atlas model inputs must be difference slots or their fragility products".into(),
            );
        }
        let mut f=map_by_id.get(&input.feature).cloned().unwrap_or_else(||serde_json::json!({"map_file":null,"map_status":"map unavailable for this declared input"}));
        if let Some(file) = f["terms_file"].as_str() {
            f["terms_file"] = serde_json::json!(format!("maps/{file}"));
        }
        let value = if input.factor.is_none() {
            values[input.feature]
        } else {
            f64::from(input.value(values))
        };
        let column: Vec<_> = bank
            .rows
            .iter()
            .map(|r| {
                if input.factor.is_none() {
                    r[input.feature]
                } else {
                    f64::from(input.value(r))
                }
            })
            .collect();
        let less = column.iter().filter(|&&v| v < value).count();
        let equal = column.iter().filter(|&&v| v == value).count();
        let percentile = (less as f64 + equal as f64 * 0.5) / column.len() as f64;
        f["key"] = serde_json::json!(format!("input{}", input.index));
        f["id"] = serde_json::json!(input.feature);
        f["name"] = serde_json::json!(if input.factor.is_some() {
            format!("{} × fragility", p.name)
        } else {
            p.name.clone()
        });
        f["family"] = serde_json::json!(p.family);
        f["signal"] = serde_json::json!(input.signal);
        f["analysis_family"] = serde_json::json!(input.analysis_family);
        f["short_name"] = serde_json::json!(input.short_name);
        f["scale"] = serde_json::json!(p.scale);
        f["channel"] = serde_json::json!(p.channel);
        f["pooling"] = serde_json::json!(p.statistic);
        f["fragility"] = serde_json::json!(input.factor.is_some());
        f["value"] = serde_json::json!(value);
        f["pooled_value"] = serde_json::json!(values[input.feature]);
        f["model_input_value"] = serde_json::json!(input.value(values));
        f["percentile"] = serde_json::json!(percentile);
        let extrema = |values: &[f64]| {
            [
                values.iter().copied().fold(f64::INFINITY, f64::min),
                values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            ]
        };
        let mut bank_ranges = serde_json::Map::new();
        bank_ranges.insert(bank.sha256.clone(), serde_json::json!(extrema(&column)));
        let mut bank_plot_ranges = serde_json::Map::new();
        bank_plot_ranges.insert(bank.sha256.clone(), serde_json::json!(plot_range(&column)));
        let mut percentiles = serde_json::Map::new();
        percentiles.insert(bank.sha256.clone(), serde_json::json!(percentile));
        for other in percentile_banks {
            let values: Vec<_> = other
                .rows
                .iter()
                .map(|r| {
                    if input.factor.is_none() {
                        r[input.feature]
                    } else {
                        f64::from(input.value(r))
                    }
                })
                .collect();
            bank_ranges.insert(other.sha256.clone(), serde_json::json!(extrema(&values)));
            bank_plot_ranges.insert(other.sha256.clone(), serde_json::json!(plot_range(&values)));
            let below = values.iter().filter(|&&v| v < value).count();
            let equal = values.iter().filter(|&&v| v == value).count();
            percentiles.insert(
                other.sha256.clone(),
                serde_json::json!((below as f64 + equal as f64 * 0.5) / values.len() as f64),
            );
        }
        f["percentiles"] = serde_json::Value::Object(percentiles);
        f["bank_ranges"] = serde_json::Value::Object(bank_ranges);
        f["bank_plot_ranges"] = serde_json::Value::Object(bank_plot_ranges);

        if let Some(file) = f["map_file"].as_str() {
            f["map_file"] = serde_json::json!(format!("maps/{file}"));
        }
        if let Some(factor) = input.factor {
            let v = values[factor] as f32;
            f["fragility_slot"] = serde_json::json!(factor);
            f["fragility_value"] = serde_json::json!(v);
            if let Some(range) = f["map_range"].as_array() {
                f["map_range"] = serde_json::json!(
                    range
                        .iter()
                        .map(|a| a.as_f64().unwrap() * f64::from(v))
                        .collect::<Vec<_>>()
                );
            }
            if let Some(range) = f["map_robust_range"].as_array() {
                f["map_robust_range"] = serde_json::json!(
                    range
                        .iter()
                        .map(|a| a.as_f64().unwrap() * f64::from(v))
                        .collect::<Vec<_>>()
                );
            }
        }
        features.push(f);
    }
    let populated = control["emitted_slot_count"]
        .as_u64()
        .ok_or("control manifest has no populated-slot count")?;
    let score = zensim::BakeScorer::new(model)?.score_features(
        values,
        av.width as u32,
        av.height as u32,
        None,
    )?;
    let meta = serde_json::json!({"name":format!("{} / {}",a.file_stem().unwrap().to_string_lossy(),b.file_stem().unwrap().to_string_lossy()),"width":av.width,"height":av.height,"reference":"reference.png","distorted":"distorted.png","difference":"difference.png","source_paths":[a,b],"source_sha256":[crate::hash(reference)?,crate::hash(distorted)?],"bake_sha256":crate::hash(&bake)?,"bank_sha256":bank.sha256,"bank_sample_rows":bank.rows.len(),"formula_revision":7,"pooled_bit_identity":true,"pooled_bits_checked":populated,"layout_bits_checked":values.len(),"score":score,"features":features,"map_encoding":"8-bit grayscale; per-map range; nearest subsampling to <=512; shared detector files across pooling variants","decode":"Imazen PNG/JPEG RGB8; no U16 downconversion; explicit no-alpha ingress"});
    crate::json(&out.join("meta.json"), &meta)?;
    println!(
        "atlas exported {} model inputs; {} populated slots across {} layout positions bit-identical",
        inputs.len(),
        populated,
        values.len()
    );
    Ok(())
}
