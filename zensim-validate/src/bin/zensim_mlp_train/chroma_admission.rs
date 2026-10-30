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
        || !matches!(d["nominal_weight"].as_f64(), Some(1.0 | 0.5 | 0.25 | 0.125))
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
    // Optional declared view-size filter: fitting views with a long edge of 64 px or less are dropped, because
    // SSIMULACRA2 is not valid there (benchmarks/ssim2_small_view_validity_2026-10-30.md). Absent = the full leg.
    let filter = if d.get("view_filter_path").is_some() || d.get("drop_long_edge_at_most").is_some() {
        let f = read("view_filter_path", "view_filter_sha256")?;
        if f["schema"] != "chroma-view-size-filter-v1"
            || f["drop_long_edge_at_most"] != 64
            || d["drop_long_edge_at_most"] != 64
            || f["bank_keys_sha256"] != bank["keys_sha256"]
        {
            return Err("chroma view-size filter binding differs".into());
        }
        Some(f)
    } else {
        None
    };
    let expected_kept = |role: &str, family: &str| -> Result<u64, String> {
        let summary = if family == "band-ac" { &ac } else { &admission };
        let admitted = summary["per_role"][role][format!("{family}_kept")]
            .as_u64()
            .ok_or("family admission count required")?;
        let Some(f) = &filter else {
            return Ok(admitted);
        };
        let counts = &f["per_role"][role][family];
        let (kept, dropped) = (
            counts["kept"].as_u64().ok_or("filter kept count required")?,
            counts["dropped"].as_u64().ok_or("filter dropped count required")?,
        );
        if kept + dropped != admitted {
            return Err("chroma view-size filter population differs".into());
        }
        Ok(kept)
    };
    let fit_rows = if filter.is_some() {
        let mut n = 0;
        for role in ["fit", "new-train"] {
            for family in ["ycc444", "ycc420", "xyb-capped", "xyb-extended", "band-ac"] {
                n += expected_kept(role, family)?;
            }
        }
        n
    } else {
        fit_rows
    };
    let keys = read("keys_path", "keys_sha256")?;
    let keys = keys.as_array().ok_or("chroma ordered keys required")?;
    if keys.len() as u64 != fit_rows || d["rows"] != keys.len() || record["rows"] != keys.len() {
        return Err("chroma fitting population differs".into());
    }
    let mut observations = BTreeSet::new();
    let mut counts = BTreeMap::<String, usize>::new();
    let mut ladders = BTreeMap::<String, BTreeSet<u64>>::new();
    let mut role_families = BTreeMap::<(String, String), u64>::new();
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
            || (filter.is_some() && !key["long_edge"].as_u64().is_some_and(|e| e > 64))
        {
            return Err("chroma row/role/family identity differs".into());
        }
        *counts.entry(source.to_owned()).or_default() += 1;
        *role_families
            .entry((
                text("mlic_role")?.to_owned(),
                text("codec_family")?.to_owned(),
            ))
            .or_default() += 1;
        if !ladders
            .entry(text("ladder")?.to_owned())
            .or_default()
            .insert(key["rung"].as_u64().ok_or("chroma rung required")?)
        {
            return Err("duplicate chroma ladder rung".into());
        }
    }
    for role in ["fit", "new-train"] {
        for family in ["ycc444", "ycc420", "xyb-capped", "xyb-extended", "band-ac"] {
            let expected = expected_kept(role, family)?;
            if role_families
                .get(&(role.to_owned(), family.to_owned()))
                .copied()
                .unwrap_or(0)
                != expected
            {
                return Err("chroma labelled family/role population differs".into());
            }
        }
    }
    if ladders.values().any(|r| r.len() < 3) || serde_json::json!(counts) != record["source_counts"]
    {
        return Err("chroma admitted source/ladder population differs".into());
    }
    // Registered scales (1, 1/2, 1/4, 1/8) are exact in f64, so scale 1 keeps the round-1 bits.
    let nominal = d["nominal_weight"].as_f64().ok_or("chroma nominal weight required")?;
    let weight =
        nominal / (counts.values().map(|n| 1. - 1. / *n as f64).sum::<f64>() / counts.len() as f64);
    if record["train_weight"].as_f64() != Some(weight) {
        return Err("chroma nominal-weight acceptance correction differs".into());
    }
    Ok(())
}
