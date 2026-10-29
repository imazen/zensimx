#![forbid(unsafe_code)]
//! Exercise the exact private native trainer metadata owner without fitting a model.
use serde_json::Value;
use std::collections::BTreeSet;
use std::path::Path;
use std::process::Command;

#[path = "../../../zensim-validate/src/bin/zensim_mlp_train/chroma_admission.rs"]
mod chroma_admission;

fn document(path: &Path, pin: &str) -> Result<Value, String> {
    if pin.len() != 64
        || !pin
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err("metadata pin required".into());
    }
    let hash = Command::new("sha256sum")
        .arg("--")
        .arg(path)
        .output()
        .map_err(|e| e.to_string())?;
    if !hash.status.success() || hash.stdout.get(..64) != Some(pin.as_bytes()) {
        return Err("metadata pin changed".into());
    }
    serde_json::from_slice(&std::fs::read(path).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())
}

fn main() -> Result<(), String> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: chroma-admission-check ROOT OUTPUT_RECEIPT".into());
    }
    let root = Path::new(&args[1]);
    let packet: Value = serde_json::from_slice(
        &std::fs::read(root.join("datasets/receipt.json")).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let record = packet["groups"]
        .as_array()
        .ok_or("groups required")?
        .iter()
        .find(|r| r["name"] == "chroma_teacher")
        .ok_or("chroma teacher missing")?;
    let declaration = document(
        Path::new(
            &(record["path"]
                .as_str()
                .ok_or("table path required")?
                .to_owned()
                + ".manifest.json"),
        ),
        record["manifest_sha256"]
            .as_str()
            .ok_or("manifest pin required")?,
    )?;
    let selected: Vec<usize> =
        serde_json::from_value(packet["feature_ids"].clone()).map_err(|e| e.to_string())?;
    let guard = document(
        Path::new(
            packet["family_guard_path"]
                .as_str()
                .ok_or("guard path required")?,
        ),
        packet["family_guard_sha256"]
            .as_str()
            .ok_or("guard pin required")?,
    )?;
    let excluded: BTreeSet<String> =
        serde_json::from_value(guard["protected_families"].clone()).map_err(|e| e.to_string())?;
    chroma_admission::admit(&declaration, record, &selected, &excluded, document)?;
    let mut negative_controls = Vec::new();
    for (field, value) in [
        ("extractor_sha256", Value::from("wrong")),
        ("teacher_helper_sha256", Value::from("wrong")),
        ("target_transform", Value::from("clipped")),
    ] {
        let mut changed = declaration.clone();
        changed[field] = value;
        if chroma_admission::admit(&changed, record, &selected, &excluded, document).is_ok() {
            return Err(format!("negative control accepted: {field}"));
        }
        negative_controls.push(field.to_owned());
    }
    let mut changed = record.clone();
    changed["mode"] = Value::from("withinref,rank");
    if chroma_admission::admit(&declaration, &changed, &selected, &excluded, document).is_ok() {
        return Err("rank-only control accepted".into());
    }
    negative_controls.push("rank-only-mode".into());
    for (field, value) in [
        ("mlic_role", "dev"),
        ("mlic_role", "calibration"),
        ("codec_family", "band-dc"),
        ("codec_family", "ycc444"),
        ("mlic_role", "new-train"),
    ] {
        let substitute = |path: &Path, pin: &str| {
            let mut v = document(path, pin)?;
            if path == Path::new(declaration["keys_path"].as_str().unwrap()) {
                v[0][field] = Value::from(value);
            }
            Ok(v)
        };
        if chroma_admission::admit(&declaration, record, &selected, &excluded, substitute).is_ok() {
            return Err(format!("negative key control accepted: {field}/{value}"));
        }
        negative_controls.push(format!("{field}/{value}"));
    }
    let receipt = serde_json::json!({"passed":true,"native_owner":"zensim_mlp_train/chroma_admission.rs","rows":record["rows"],"selected_features":selected.len(),"negative_controls_refused":negative_controls,"training_launched":false});
    if Path::new(&args[2]).exists() {
        return Err("preserve existing native admission receipt".into());
    }
    std::fs::write(&args[2], serde_json::to_vec_pretty(&receipt).unwrap())
        .map_err(|e| e.to_string())?;
    Ok(())
}
