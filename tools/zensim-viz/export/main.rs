#![forbid(unsafe_code)]

mod atlas;
mod audit;
mod bank;
mod compare;
mod constraints;
mod lens;
mod model;
mod pixels;
mod provenance;

use serde::Serialize;
use std::{
    collections::BTreeMap,
    error::Error,
    path::{Path, PathBuf},
};
type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn hash(path: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = [0; 65536];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}
fn json(path: &Path, value: &impl Serialize) -> Result<()> {
    let f = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    serde_json::to_writer(std::io::BufWriter::new(f), value)?;
    Ok(())
}
fn main() -> Result<()> {
    if std::env::var("ZENSIM_FORMULA_REV").as_deref() != Ok("7") {
        return Err("set ZENSIM_FORMULA_REV=7 for revision-7 inspection".into());
    }
    let mut args = std::env::args().skip(1);
    let command = args.next().ok_or("usage: zensim-viz-export audit BAKE --bank TRAIN.parquet --out DIRECTORY [--max-rows 512] [--training-bank TRAIN.parquet]")?;
    let bake_path = PathBuf::from(args.next().ok_or("missing bake")?);
    let mut opts = BTreeMap::new();
    let mut percentile_paths = Vec::new();
    let allowed: &[&str] = match command.as_str() {
        "weights" => &["--out"],
        "curve-audit" => &["--out", "--baseline"],
        "audit" => &[
            "--bank",
            "--out",
            "--max-rows",
            "--training-bank",
            "--baseline-audit",
        ],
        "mlp-bank" => &["--bank", "--out", "--max-rows", "--other"],
        "mlp-lens" => &["--bank", "--out", "--max-rows", "--feature", "--family"],
        "compare" => &["--bank", "--out", "--max-rows", "--feature", "--other"],
        "atlas" => &[
            "--bank",
            "--out",
            "--max-rows",
            "--reference",
            "--distorted",
            "--row-padding",
            "--percentile-bank",
        ],
        "atlas-pass" => &["--out", "--reference", "--distorted", "--row-padding"],
        _ => return Err("unknown command".into()),
    };
    while let Some(key) = args.next() {
        if !allowed.contains(&key.as_str()) {
            return Err(format!("unknown option {key} for {command}").into());
        }
        let value = args.next().ok_or("option needs a value")?;
        if key == "--percentile-bank" {
            percentile_paths.push(value);
        } else if opts.insert(key.clone(), value).is_some() {
            return Err(format!("repeated option {key}").into());
        }
    }
    let out = PathBuf::from(opts.get("--out").ok_or("--out is required")?);
    if command == "curve-audit" {
        std::fs::create_dir_all(&out)?;
        provenance::write(&out)?;
        return lens::summarize(&bake_path, &out, opts.get("--baseline").map(Path::new));
    }
    if command == "atlas-pass" {
        let reference = PathBuf::from(opts.get("--reference").ok_or("--reference required")?);
        let distorted = PathBuf::from(opts.get("--distorted").ok_or("--distorted required")?);
        return atlas::pass(
            &bake_path,
            &reference,
            &distorted,
            &out,
            opts.get("--row-padding")
                .map(|v| v.parse())
                .transpose()?
                .unwrap_or(0),
        );
    }
    if command == "weights" {
        let model = zenpredict::Model::from_bytes(&std::fs::read(&bake_path)?)?;
        let inputs = model::inputs(&model)?;
        std::fs::create_dir_all(&out)?;
        provenance::write(&out)?;
        json(
            &out.join("weights.json"),
            &model::weights_document(&model, &inputs),
        )?;
        return json(
            &out.join("identity.json"),
            &serde_json::json!({"bake_sha256":hash(&bake_path)?}),
        );
    }
    if command != "audit" && std::env::var("ZENSIM_FORMULA_REV").as_deref() != Ok("7") {
        return Err("set ZENSIM_FORMULA_REV=7".into());
    }
    let bank_path = PathBuf::from(opts.get("--bank").ok_or("--bank is required")?);
    let max_rows = opts
        .get("--max-rows")
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(
            if matches!(
                command.as_str(),
                "audit" | "atlas" | "mlp-bank" | "mlp-lens"
            ) {
                usize::MAX
            } else {
                512usize
            },
        );
    if max_rows == 0 {
        return Err("--max-rows must be positive".into());
    }
    if matches!(command.as_str(), "mlp-lens" | "mlp-bank" | "compare") && max_rows < 200 {
        return Err("--max-rows must be >=200 for the lens".into());
    }
    let model = zenpredict::Model::from_bytes(&std::fs::read(&bake_path)?)?;
    let inputs = model::inputs(&model)?;
    let other_model = opts
        .get("--other")
        .map(|p| -> Result<_> { Ok(zenpredict::Model::from_bytes(&std::fs::read(p)?)?) })
        .transpose()?;
    let other_inputs = other_model.as_ref().map(model::inputs).transpose()?;
    let mut bank_inputs = inputs.clone();
    if let Some(other) = &other_inputs {
        bank_inputs.extend_from_slice(other);
    }
    let bank = bank::load(&bank_path, &bank_inputs, max_rows)?;
    let percentile_banks = percentile_paths
        .iter()
        .map(|p| bank::load(Path::new(p), &inputs, max_rows))
        .collect::<Result<Vec<_>>>()?;
    let training = opts
        .get("--training-bank")
        .map(|p| bank::load(Path::new(p), &inputs, max_rows))
        .transpose()?;
    std::fs::create_dir_all(&out)?;
    provenance::write(&out)?;
    match command.as_str() {
        "audit" => {
            let constraints = constraints::inspect(&model, &inputs, &bake_path)?;
            let audit = audit::run(
                &model,
                &inputs,
                &bank,
                training.as_ref(),
                hash(&bake_path)?,
                &constraints,
            )?;
            if let Some(path) = opts.get("--baseline-audit") {
                audit::verify_prior(&audit, Path::new(path), &out)?;
            }
            json(&out.join("audit.json"), &audit)?;
            audit::write_tsv(&out.join("audit.tsv"), &audit)?;
            json(
                &out.join("weights.json"),
                &model::weights_document(&model, &inputs),
            )?;
            println!(
                "{} weights; {} sampled / {} bank rows; {} flags",
                audit.weight_count,
                bank.rows.len(),
                bank.total_rows,
                audit.flags.len()
            );
            for (kind, count) in &audit.counts {
                println!("{kind}\t{count}");
            }
            for flag in &audit.flags {
                println!(
                    "{}\t{}\t{}\t{}",
                    flag.severity, flag.kind, flag.location, flag.reason
                );
            }
        }
        "compare" => {
            let other = other_model.as_ref().ok_or("--other required")?;
            let other_inputs = other_inputs.as_ref().ok_or("other inputs missing")?;
            let feature = opts
                .get("--feature")
                .map(|v| v.parse())
                .transpose()?
                .unwrap_or(inputs[0].feature);
            let doc = compare::run(
                &model,
                other,
                &inputs,
                other_inputs,
                &bank,
                [&bake_path, Path::new(opts.get("--other").unwrap())],
                feature,
            )?;
            json(&out.join(format!("compare-f{feature}.json")), &doc)?;
        }
        "atlas" => {
            let reference = PathBuf::from(opts.get("--reference").ok_or("--reference required")?);
            let distorted = PathBuf::from(opts.get("--distorted").ok_or("--distorted required")?);
            atlas::run(
                &model,
                &inputs,
                &bank,
                atlas::Paths {
                    bake: &bake_path,
                    reference: &reference,
                    distorted: &distorted,
                    out: &out,
                },
                opts.get("--row-padding")
                    .map(|v| v.parse())
                    .transpose()?
                    .unwrap_or(0),
                &percentile_banks,
            )?;
        }
        "mlp-bank" => lens::bulk(
            &model,
            &inputs,
            &bank_inputs,
            &bank,
            hash(&bake_path)?,
            &out,
        )?,
        "mlp-lens" => {
            if std::env::var("ZENSIM_FORMULA_REV").as_deref() != Ok("7") {
                return Err("set ZENSIM_FORMULA_REV=7".into());
            }
            let feature = opts
                .get("--feature")
                .map(|v| v.parse())
                .transpose()?
                .unwrap_or(inputs[0].feature);
            let doc = lens::run(
                &model,
                &inputs,
                &bank,
                feature,
                opts.get("--family").map(String::as_str),
                hash(&bake_path)?,
                Some(&out),
            )?;
            let name = opts.get("--family").map_or_else(
                || format!("lens-f{feature}.json"),
                |f| format!("lens-family-{f}.json"),
            );
            json(&out.join(name), &doc)?;
        }
        _ => return Err(format!("unknown command {command}").into()),
    }
    Ok(())
}
