//! D6's single admitted AIC population. This is metadata/key admission; the
//! caller hashes payloads only after all ordinary groups pass admission too.
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use zensim_validate::{mlp_train::GroupLossMode, train_manifest};

type Group = (String, PathBuf, f64, f64, bool, GroupLossMode);
const INPUTS: [(&str, &str); 3] = [
    (
        "source_table_sha256",
        "a2969c70f2e1c96f18348b0b8e66a28e41ee87ad839636e8329e63408b138b71",
    ),
    (
        "pairs_sha256",
        "8ce5a1e913d3c4a9c10848560125b3449542301921d5e9316bfe14b1bb854d66",
    ),
    (
        "assignment_sha256",
        "d67be7b4a5d09dac07ca9ba30ae2acbcfc13bac9369ccd61290f509ab869f267",
    ),
];
const FSID: &str = "basic+peaks+v2@w1825/rev7_rowlanes#36c3f3af";

fn hash(path: &Path) -> Result<String, String> {
    super::group_admission::safe(path)?;
    train_manifest::sha256_file(path).map_err(|e| e.to_string())
}
fn json(path: &Path) -> Result<Value, String> {
    super::group_admission::safe(path)?;
    serde_json::from_slice(&std::fs::read(path).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())
}
fn text<'a>(v: &'a Value, key: &str) -> Result<&'a str, String> {
    v[key].as_str().ok_or_else(|| format!("D6 {key} required"))
}
fn declaration(d: &Value) -> Result<(), String> {
    if d["schema"] != "aic2026-public-train-rev7-v1"
        || d["ruling"] != "D6"
        || d["split"] != "public_train"
        || d["role"] != "TRAIN"
        || d["use"] != "fit"
        || d["formula_revision"] != 7
        || d["feature_set_id"] != FSID
        || d["target_column"] != "human_score"
        || d["target_transform"] != "negate-JND_score"
        || INPUTS.iter().any(|&(key, pin)| d[key] != pin)
    {
        return Err("D6 admits exactly revision-7 AIC2026 public_train".into());
    }
    Ok(())
}

pub(super) fn admit(
    authority: &Path,
    authority_pin: &str,
    groups: &[Group],
) -> Result<BTreeMap<String, Value>, String> {
    if authority_pin.len() != 64 || hash(authority)? != authority_pin {
        return Err("D6 independently pinned fold authority required".into());
    }
    let a = json(authority)?;
    if a["schema"] != "aic2026-public-train-rev7-folds-v1"
        || a["ruling"] != "D6"
        || a["population"] != "WG1 m112107 public_train"
        || INPUTS.iter().any(|&(key, pin)| a[key] != pin)
    {
        return Err("D6 public_train fold authority differs".into());
    }
    let candidates: Vec<_> = groups
        .iter()
        .filter(|g| g.0 == "aic_public_train")
        .collect();
    if candidates.len() != 1 {
        return Err("D6 requires one explicitly named public_train leg".into());
    }
    let g = candidates[0];
    if !g.2.is_finite() || g.2 <= 0.0 || g.3 != 0.0 || g.4 || g.5 != GroupLossMode::Rank {
        return Err("D6 requires a positive fit-only pooled-rank leg".into());
    }
    let path = g.1.canonicalize().map_err(|e| e.to_string())?;
    let records = a["tables"]
        .as_object()
        .ok_or("D6 table inventory required")?;
    let matches: Vec<_> = records
        .values()
        .filter(|r| {
            r["path"]
                .as_str()
                .is_some_and(|p| Path::new(p).canonicalize().ok().as_ref() == Some(&path))
        })
        .collect();
    if matches.len() != 1 {
        return Err("D6 table is absent from the pinned fold inventory".into());
    }
    let r = matches[0];
    let manifest = PathBuf::from(format!("{}.manifest.json", g.1.display()));
    let d = json(&manifest)?;
    declaration(&d)?;
    if hash(&manifest)? != text(r, "manifest_sha256")? || d["table_sha256"] != r["table_sha256"] {
        return Err("D6 table declaration pin differs".into());
    }
    let omitted = if d["omitted_source_fold"].is_null() {
        None
    } else {
        Some(
            d["omitted_source_fold"]
                .as_u64()
                .filter(|&k| k < 7)
                .ok_or("D6 source fold differs")?,
        )
    };
    let assignment_path = Path::new(text(&d, "assignment_path")?);
    if hash(assignment_path)? != INPUTS[2].1 {
        return Err("D6 source assignment pin differs".into());
    }
    let assignment_document = json(assignment_path)?;
    let assignment = assignment_document["assignment"]
        .as_object()
        .ok_or("D6 assignment required")?;
    if a["assignment"] != assignment_document["assignment"] || assignment.len() != 28 {
        return Err("D6 authority source assignment differs".into());
    }
    let keys_path = Path::new(text(&d, "keys_path")?);
    if hash(keys_path)? != text(&d, "keys_sha256")? {
        return Err("D6 ordered key pin differs".into());
    }
    let keys_document = json(keys_path)?;
    let keys = keys_document.as_array().ok_or("D6 ordered keys required")?;
    let mut seen = BTreeSet::new();
    let mut counts = BTreeMap::<String, usize>::new();
    for (i, k) in keys.iter().enumerate() {
        let fields: BTreeSet<_> = k
            .as_object()
            .ok_or("D6 key object required")?
            .keys()
            .map(String::as_str)
            .collect();
        let source = text(k, "source")?;
        let key = text(k, "key")?;
        if fields
            != ["row", "key", "source", "split", "source_fold"]
                .into_iter()
                .collect()
            || k["row"].as_u64() != Some(i as u64)
            || k["split"] != "public_train"
            || key.is_empty()
            || source.is_empty()
            || !seen.insert(key)
            || k["source_fold"]
                .as_u64()
                .is_none_or(|f| f >= 7 || Some(f) == omitted)
            || assignment.get(source) != Some(&k["source_fold"])
        {
            return Err("D6 ordered source-key role/fold differs".into());
        }
        *counts.entry(source.to_owned()).or_default() += 1;
    }
    let n = if omitted.is_some() { 3360 } else { 3920 };
    let sources = if omitted.is_some() { 24 } else { 28 };
    if keys.len() != n
        || d["rows"].as_u64() != Some(n as u64)
        || counts.len() != sources
        || counts.values().any(|&n| n != 140)
        || d["sources"] != serde_json::json!(counts.keys().collect::<Vec<_>>())
    {
        return Err("D6 source-fold population differs".into());
    }
    Ok([(g.0.clone(), d)].into_iter().collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn other_split_and_extraction_declarations_are_rejected_without_file_reads() {
        assert!(admit(Path::new("must-not-open"), "", &[]).is_err());
        let mut d = serde_json::json!({"schema":"aic2026-public-train-rev7-v1", "ruling":"D6",
            "split":"public_train", "role":"TRAIN", "use":"fit", "formula_revision":7,
            "feature_set_id":FSID, "target_column":"human_score", "target_transform":"negate-JND_score"});
        for (key, pin) in INPUTS {
            d[key] = serde_json::json!(pin);
        }
        declaration(&d).unwrap();
        for (key, value) in [
            ("split", "public_test"),
            ("split", "aic3"),
            ("role", "EVAL"),
            ("feature_set_id", "other"),
            ("source_table_sha256", "wrong"),
        ] {
            let mut changed = d.clone();
            changed[key] = serde_json::json!(value);
            assert!(declaration(&changed).is_err());
        }
    }
}
