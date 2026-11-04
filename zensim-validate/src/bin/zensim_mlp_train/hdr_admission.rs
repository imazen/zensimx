//! Registered native-HDR TRAIN legs (paths P2/P3): metadata, ordered keys and population only; no target-table read.
//! Mirrors `admit_hdr_leg` in scripts/rev4_featpot/subjective_training_leg.py; registration in
//! docs/HDR_P2P3_EXECUTION_2026-10-29.md.
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

const FEATURES: &str = "basic+peaks+v2@w1825/rev7_rowlanes#36c3f3af";

/// Registered producer sets, keyed by the HDR front end the banks were extracted through (mirrors
/// `HDR_PRODUCERS` in subjective_training_leg.py): front end, native (hdr944_extract) commit and
/// extractor sha, UPIQ (upiq_pu_score) commit and extractor sha. `pu21` is the 2026-10-29 fleet
/// extraction; banks and declarations from before the field exist carry no `hdr_front_end` and mean
/// `pu21`. `cube-root:203` is HC1's dev re-extraction (e36_subset_test_2026-11-01.md, HC1).
const PRODUCERS: [(&str, &str, &str, &str, &str); 2] = [
    (
        "pu21",
        "a86520ca68dc92f98abfd8a526d81f8abf112de6",
        "63bed9a64ed265d2e8b8bad4aabaae5d952215afda935ec911a0c82d1c204aa7",
        "d86b3b44383433e56ebc6161ee6c8db12639c5af",
        "73d1374f1cbb2ca2f5166812d69ee49e22486174b56337263fa27d9689c79214",
    ),
    (
        "cube-root:203",
        "94095e060047cbae3c257188f4f9e1b2a785ea54",
        "b61ae3f50e903f522a1265ab9b5506b80fe0c254323aa9b8974a7fbb05890d1c",
        "94095e060047cbae3c257188f4f9e1b2a785ea54",
        "ea4298a3ee3fca22cc2b4c73c92301142818321287128ed93c43a14c6af934b3",
    ),
];

/// (producer commit, extractor sha) of a registered front end for a UPIQ or native leg.
fn producer(front_end: &str, upiq: bool) -> Option<(&'static str, &'static str)> {
    let set = PRODUCERS.iter().find(|p| p.0 == front_end)?;
    Some(if upiq { (set.3, set.4) } else { (set.1, set.2) })
}

/// name, human, mode, loss, target transform, rows, produced by upiq_pu_score (else hdr944_extract)
type LegSpec = (
    &'static str,
    bool,
    &'static str,
    &'static str,
    &'static str,
    u64,
    bool,
);
const LEGS: [LegSpec; 4] = [
    (
        "hdr_upiq380",
        true,
        "withinref,rank",
        "within-source-rank",
        "score=100+10*human_JOD;no-clipping",
        330,
        true,
    ),
    (
        "hdr_4kdtb_train",
        true,
        "withinref,rank",
        "within-source-rank",
        "orient-native-quality;MOS-0-100",
        48,
        false,
    ),
    (
        "hdr_ladder_chroma",
        false,
        "withinref,both",
        "within-source-rank-and-scale",
        "identity-ssimulacra2-unclipped",
        7177,
        false,
    ),
    (
        "hdr_teacher_hdrvdp3",
        false,
        "withinref,both",
        "within-source-rank-and-scale",
        "score=10*q_jod;no-clipping",
        7390,
        false,
    ),
];

/// TRAIN rows left when dev fold 0 / 1 is held out (registered; mirrors `HDR_FOLD_ROWS` in subjective_training_leg.py).
fn fold_rows(name: &str) -> [u64; 2] {
    match name {
        "hdr_upiq380" => [150, 180],
        "hdr_4kdtb_train" => [0, 48],
        "hdr_ladder_chroma" => [2709, 4468],
        _ => [4484, 2906],
    }
}

fn fold_of(family: &str) -> u8 {
    let digest = Sha256::digest(format!("hdr-p2p3-dev-fold-v1:{family}").as_bytes());
    digest[31] & 1 // the integer is big-endian: its parity is the last byte's low bit
}

pub(super) fn admit(
    declaration: &Value,
    record: &Value,
    feature_ids: &Value, // the authority cell's `feature_ids` (the leg declarations bind these, as the subjective legs do)
    excluded: &BTreeSet<String>,
    document: impl Fn(&Path, &str) -> Result<Value, String>,
) -> Result<(), String> {
    let d = declaration;
    let leg = LEGS
        .iter()
        .find(|l| record["name"] == l.0)
        .ok_or("unregistered HDR leg")?;
    let (name, human, mode, loss, transform, rows, upiq) = *leg;
    let front_end = d["hdr_front_end"].as_str().unwrap_or("pu21");
    let (commit, extractor) = producer(front_end, upiq).ok_or("unregistered HDR front end")?;
    let nominal = d["nominal_weight"]
        .as_f64()
        .filter(|w| w.is_finite() && *w > 0.);
    let checks: [(&str, bool); 22] = [
        ("schema", d["schema"] != "hdr-native-leg-v1"),
        ("leg", d["leg"] != name),
        ("loss", d["loss"] != loss),
        ("target_transform", d["target_transform"] != transform),
        ("target_column", d["target_column"] != "human_score"),
        ("target_orientation", d["target_orientation"] != "quality"),
        ("native_role", d["native_role"] != "TRAIN"),
        ("selection", d["selection"] != "fixed-final-epoch"),
        ("formula_revision", d["formula_revision"] != 7),
        ("tier", d["tier"] != "v3"),
        ("feature_ids", d["feature_ids"] != *feature_ids),
        ("feature_set_id", d["feature_set_id"] != FEATURES),
        ("producer_commit", d["producer_commit"] != commit),
        ("extractor_sha256", d["extractor_sha256"] != extractor),
        ("nominal_weight", nominal.is_none()),
        ("table_sha256", d["table_sha256"] != record["table_sha256"]),
        ("keys_sha256", d["keys_sha256"] != record["keys_sha256"]),
        ("record.teacher", record["teacher"] != !human),
        ("record.subjective", record["subjective"] != human),
        ("record.mode", record["mode"] != mode),
        (
            "record.val_weight",
            record["val_weight"].as_f64() != Some(0.),
        ),
        ("record.rows", record["rows"] != d["rows"]),
    ];
    if let Some((which, _)) = checks.iter().find(|(_, failed)| *failed) {
        return Err(format!(
            "HDR leg role/loss/producer binding differs: {which} ({name})"
        ));
    }
    let fold = match &d["dev_fold"] {
        Value::Null => None,
        v => Some(
            v.as_u64()
                .filter(|f| *f < 2)
                .ok_or("HDR dev fold must be 0 or 1")? as u8,
        ),
    };
    let expected_rows = fold.map_or(rows, |f| fold_rows(name)[f as usize]);
    if d["rows"] != expected_rows {
        return Err("HDR leg population (or dev fold) differs".into());
    }
    let read = |path_key: &str, pin_key: &str| {
        document(
            Path::new(d[path_key].as_str().ok_or("HDR metadata path required")?),
            d[pin_key].as_str().ok_or("HDR metadata pin required")?,
        )
    };
    let bank = read("feature_bank_manifest_path", "feature_bank_manifest_sha256")?;
    if bank["schema"] != "hdr-native-rev7-bank-v1"
        || bank["build_commit"] != commit
        || bank["hdr_front_end"].as_str().unwrap_or("pu21") != front_end
        || bank["research"]["feature_set_id"] != FEATURES
        || bank["labels_read"] != false
    {
        return Err("HDR native feature bank identity differs".into());
    }
    let keys = read("keys_path", "keys_sha256")?;
    let keys = keys.as_array().ok_or("HDR ordered keys required")?;
    if keys.len() as u64 != expected_rows {
        return Err("HDR leg population differs".into());
    }
    let mut observations = BTreeSet::new();
    let mut counts = BTreeMap::<String, usize>::new();
    for (i, key) in keys.iter().enumerate() {
        let object = key.as_object().ok_or("HDR key object required")?;
        let text = |n: &str| {
            key[n]
                .as_str()
                .filter(|s| !s.is_empty())
                .ok_or("HDR key field required")
        };
        let fields: BTreeSet<&str> = object.keys().map(String::as_str).collect();
        if fields
            != BTreeSet::from([
                "observation",
                "source",
                "family",
                "reference",
                "role",
                "tier",
                "row",
            ])
            || key["row"] != i
            || key["role"] != "fit"
            || key["tier"] != "T2"
            || key["reference"] != key["source"]
            || excluded.contains(text("family")?)
            || !observations.insert(text("observation")?.to_owned())
        {
            return Err("HDR leg row/role/family identity differs".into());
        }
        if fold == Some(fold_of(text("family")?)) {
            return Err("HDR leg holds a family of its own held-out dev fold".into());
        }
        text("reference")?;
        *counts.entry(text("source")?.to_owned()).or_default() += 1;
    }
    if serde_json::json!(counts) != record["source_counts"] || counts.values().any(|n| *n < 2) {
        return Err("HDR leg source population differs".into());
    }
    let weight = nominal.unwrap_or(0.)
        / (counts.values().map(|n| 1. - 1. / *n as f64).sum::<f64>() / counts.len() as f64);
    if record["train_weight"].as_f64() != Some(weight) {
        return Err("HDR leg nominal-weight acceptance correction differs".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn fixture() -> (Value, Value, Value, Value) {
        let (name, _, mode, loss, transform, rows, upiq) = LEGS[1];
        let (commit, extractor) = producer("pu21", upiq).unwrap();
        let keys: Vec<Value> = (0..rows)
            .map(|i| {
                let s = format!("4kdtb:S{}", i / 12);
                json!({"row": i, "observation": format!("o{i}"), "source": s, "reference": s,
                       "family": format!("hdr:{s}"), "role": "fit", "tier": "T2"})
            })
            .collect();
        let mut counts = BTreeMap::new();
        for k in &keys {
            *counts
                .entry(k["source"].as_str().unwrap().to_owned())
                .or_insert(0usize) += 1;
        }
        let weight =
            0.5 / (counts.values().map(|n| 1. - 1. / *n as f64).sum::<f64>() / counts.len() as f64);
        let d = json!({"schema": "hdr-native-leg-v1", "leg": name, "loss": loss, "nominal_weight": 0.5,
            "target_orientation": "quality", "target_transform": transform, "target_column": "human_score",
            "selection": "fixed-final-epoch", "formula_revision": 7, "tier": "v3", "feature_ids": [1, 2],
            "feature_set_id": FEATURES, "producer_commit": commit, "extractor_sha256": extractor,
            "native_role": "TRAIN", "rows": rows, "table_sha256": "t", "keys_sha256": "k", "keys_path": "keys",
            "feature_bank_manifest_path": "bank", "feature_bank_manifest_sha256": "b"});
        let r = json!({"name": name, "teacher": false, "subjective": true, "train_weight": weight,
            "val_weight": 0.0, "mode": mode, "table_sha256": "t", "keys_sha256": "k", "rows": rows,
            "source_counts": counts});
        let bank = json!({"schema": "hdr-native-rev7-bank-v1", "build_commit": commit,
            "research": {"feature_set_id": FEATURES}, "labels_read": false});
        (d, r, bank, Value::Array(keys))
    }

    fn run(
        d: &Value,
        r: &Value,
        bank: &Value,
        keys: &Value,
        excluded: &BTreeSet<String>,
    ) -> Result<(), String> {
        admit(d, r, &json!([1, 2]), excluded, |p, _| {
            Ok(if p == Path::new("bank") {
                bank.clone()
            } else {
                keys.clone()
            })
        })
    }

    #[test]
    fn registered_leg_admits_and_each_deviation_refuses() {
        let (d, r, bank, keys) = fixture();
        let none = BTreeSet::new();
        run(&d, &r, &bank, &keys, &none).unwrap();
        let mut bad = d.clone();
        bad["loss"] = json!("within-source-rank-and-scale");
        assert!(run(&bad, &r, &bank, &keys, &none).is_err());
        let mut bad = d.clone();
        bad["producer_commit"] = json!("0");
        assert!(run(&bad, &r, &bank, &keys, &none).is_err());
        let mut bad = r.clone();
        bad["train_weight"] = json!(r["train_weight"].as_f64().unwrap() * 1.01);
        assert!(run(&d, &bad, &bank, &keys, &none).is_err());
        let mut bad = bank.clone();
        bad["labels_read"] = json!(true);
        assert!(run(&d, &r, &bad, &keys, &none).is_err());
        let mut bad = d.clone();
        bad["dev_fold"] = json!(0);
        assert!(run(&bad, &r, &bank, &keys, &none).is_err());
        assert_eq!(fold_of("hdr:upiq:narwaria:01"), {
            let h = Sha256::digest(b"hdr-p2p3-dev-fold-v1:hdr:upiq:narwaria:01");
            h[31] & 1
        });
        let reserved = BTreeSet::from(["hdr:4kdtb:S0".to_owned()]);
        assert!(run(&d, &r, &bank, &keys, &reserved).is_err());
        let mut bad = r.clone();
        bad["name"] = json!("hdr_other");
        assert!(run(&d, &bad, &bank, &keys, &none).is_err());
    }

    #[test]
    fn cube_root_producer_set_admits_and_front_ends_never_mix() {
        let (d, r, bank, keys) = fixture();
        let none = BTreeSet::new();
        let (commit, extractor) = producer("cube-root:203", false).unwrap();
        assert_ne!(extractor, producer("pu21", false).unwrap().1);
        let mut cube = d.clone();
        cube["hdr_front_end"] = json!("cube-root:203");
        cube["producer_commit"] = json!(commit);
        cube["extractor_sha256"] = json!(extractor);
        let mut cube_bank = bank.clone();
        cube_bank["build_commit"] = json!(commit);
        cube_bank["hdr_front_end"] = json!("cube-root:203");
        run(&cube, &r, &cube_bank, &keys, &none).unwrap();
        // a cube-root declaration over a 2026-10-29 bank (no field, meaning pu21) refuses
        assert!(run(&cube, &r, &bank, &keys, &none).is_err());
        // pu21 pins under the cube-root label refuse; an unregistered front end refuses
        let mut bad = cube.clone();
        bad["producer_commit"] = d["producer_commit"].clone();
        bad["extractor_sha256"] = d["extractor_sha256"].clone();
        assert!(run(&bad, &r, &cube_bank, &keys, &none).is_err());
        let mut bad = cube.clone();
        bad["hdr_front_end"] = json!("cube-root:100");
        assert!(run(&bad, &r, &cube_bank, &keys, &none).is_err());
        // a pu21 declaration over a cube-root bank refuses
        assert!(run(&d, &r, &cube_bank, &keys, &none).is_err());
        // the UPIQ leg takes the set's upiq_pu_score pins
        assert_eq!(producer("cube-root:203", true).unwrap().0, commit);
    }
}
