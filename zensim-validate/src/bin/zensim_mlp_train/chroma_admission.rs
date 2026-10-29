//! Frozen chroma teacher metadata/key admission; no target-table reads here.
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

const EXTRACTOR: &str = "ed4957417339779dd8ed25fe59a6166326ba9db86d5c1530b1e1c788c9d614d8";
const PRODUCER: &str = "b77883be18db5933a9db4b9e2670a2ba88a9729b";
const FEATURES: &str = "basic+peaks+v2@w1825/rev7_rowlanes#36c3f3af";
const TEACHER: &str = "2e6d5f77f54062c9a08c6390668eb075474ff8772926720b5475d8ca179eb3a6";

pub(super) fn admit(
    declaration: &Value,
    record: &Value,
    selected: &[usize],
    excluded: &BTreeSet<String>,
    document: impl Fn(&Path, &str) -> Result<Value, String>,
) -> Result<(), String> {
    let d = declaration;
    if record["name"] != "chroma_teacher"
        || record["teacher"] != true
        || record["subjective"] != false
        || record["mode"] != "withinref,both"
        || record["val_weight"].as_f64() != Some(0.)
        || d["schema"] != "chroma-teacher-leg-v1"
        || d["loss"] != "within-source-rank-and-scale"
        || d["target_transform"] != "identity-ssimulacra2-unclipped"
        || d["target_column"] != "human_score"
        || d["target_orientation"] != "quality"
        || d["native_role"] != "TRAIN"
        || d["selection"] != "fixed-final-epoch"
        || d["nominal_weight"].as_f64() != Some(1.)
        || d["formula_revision"] != 7
        || d["tier"] != "v3"
        || d["feature_ids"] != serde_json::json!(selected)
        || d["feature_set_id"] != FEATURES
        || d["producer_commit"] != PRODUCER
        || d["extractor_sha256"] != EXTRACTOR
        || d["teacher_helper_sha256"] != TEACHER
        || d["table_sha256"] != record["table_sha256"]
        || d["keys_sha256"] != record["keys_sha256"]
    {
        return Err("chroma teacher role/loss/producer binding differs".into());
    }
    let read = |path_key: &str, pin_key: &str| {
        document(
            Path::new(
                d[path_key]
                    .as_str()
                    .ok_or("chroma metadata path required")?,
            ),
            d[pin_key].as_str().ok_or("chroma metadata pin required")?,
        )
    };
    let admission = read("admission_summary_path", "admission_summary_sha256")?;
    let bank = read("feature_bank_manifest_path", "feature_bank_manifest_sha256")?;
    let ac = read("ac_admission_summary_path", "ac_admission_summary_sha256")?;
    let ac_kept = ac["totals"]["band-ac_kept"]
        .as_u64()
        .ok_or("AC kept count required")?;
    let fit_rows = 232888
        + ["fit", "new-train"]
            .iter()
            .map(|role| {
                ac["per_role"][role]["band-ac_kept"]
                    .as_u64()
                    .ok_or("AC role count required")
            })
            .collect::<Result<Vec<_>, _>>()?
            .iter()
            .sum::<u64>();
    if admission["qualified_full_grid"] != true
        || admission["cells"] != 696312
        || admission["views"] != 9162
        || bank["qualified_for_leg"] != true
        || bank["feature_set_id"] != FEATURES
        || bank["binary_sha256"] != EXTRACTOR
        || bank["build_commit"] != PRODUCER
        || bank["rows"] != 251685 + ac_kept
        || bank["band_ac_included"] != true
        || ac["qualified_full_grid"] != true
        || ac["cells"] != 146592
        || ac["views"] != 9162
        || ac["training_allowed"] != true
        || bank["ac_admission_summary_sha256"] != d["ac_admission_summary_sha256"]
    {
        return Err("complete frozen chroma admission and feature reconciliation required".into());
    }
    let keys = read("keys_path", "keys_sha256")?;
    let keys = keys.as_array().ok_or("chroma ordered keys required")?;
    if keys.len() as u64 != fit_rows || d["rows"] != keys.len() || record["rows"] != keys.len() {
        return Err("chroma fitting population differs".into());
    }
    let mut observations = BTreeSet::new();
    let mut counts = BTreeMap::<String, usize>::new();
    let mut ladders = BTreeMap::<String, BTreeSet<u64>>::new();
    for (i, key) in keys.iter().enumerate() {
        let text = |name: &str| {
            key[name]
                .as_str()
                .filter(|s| !s.is_empty())
                .ok_or("chroma key field required")
        };
        let source = text("source")?;
        if key["row"] != i
            || key["role"] != "fit"
            || !matches!(text("mlic_role")?, "fit" | "new-train")
            || !matches!(
                text("codec_family")?,
                "ycc444" | "ycc420" | "xyb-capped" | "xyb-extended" | "band-ac"
            )
            || key["reference"] != source
            || excluded.contains(text("family")?)
            || !observations.insert(text("observation")?.to_owned())
        {
            return Err("chroma row/role/family identity differs".into());
        }
        *counts.entry(source.to_owned()).or_default() += 1;
        if !ladders
            .entry(text("ladder")?.to_owned())
            .or_default()
            .insert(key["rung"].as_u64().ok_or("chroma rung required")?)
        {
            return Err("duplicate chroma ladder rung".into());
        }
    }
    if ladders.values().any(|r| r.len() < 3) || serde_json::json!(counts) != record["source_counts"]
    {
        return Err("chroma admitted source/ladder population differs".into());
    }
    let weight =
        1. / (counts.values().map(|n| 1. - 1. / *n as f64).sum::<f64>() / counts.len() as f64);
    if record["train_weight"].as_f64() != Some(weight) {
        return Err("chroma nominal-weight acceptance correction differs".into());
    }
    Ok(())
}
