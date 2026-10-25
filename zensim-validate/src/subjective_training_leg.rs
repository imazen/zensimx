//! Private binary module: receipt-bound unnamed subjective ingress.
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use zensim_validate::{mlp_train::GroupLossMode, train_manifest};

type Group = (String, PathBuf, f64, f64, bool, GroupLossMode);

fn safe(path: &Path) -> Result<(), String> {
    super::group_admission::safe(path)
}
fn hash(path: &Path) -> Result<String, String> {
    safe(path)?;
    train_manifest::sha256_file(path).map_err(|e| e.to_string())
}
fn json(path: &Path, pin: &str) -> Result<Value, String> {
    if pin.len() != 64
        || !pin
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
    {
        return Err("independently pinned subjective authority required".into());
    }
    if hash(path)? != pin {
        return Err("subjective metadata/key pin changed".into());
    }
    serde_json::from_slice(&std::fs::read(path).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())
}
fn string<'a>(v: &'a Value, key: &str) -> Result<&'a str, String> {
    v[key]
        .as_str()
        .ok_or_else(|| format!("subjective {key} required"))
}

fn run_binding(c: &Value, args: &super::Args, width: usize) -> Result<(), String> {
    let limit = if c["route"] == "loso" { 3 } else { 10 };
    if !matches!(c["route"].as_str(), Some("full" | "d1" | "loso"))
        || c["seed_index"].as_u64().is_none_or(|v| v >= limit)
        || c["epochs"] != 120
        || args.epochs != 120
        || c["pairs_per_epoch"] != 50000
        || args.pairs_per_epoch != 50000
        || c["selected_epoch"] != 119
        || args.hidden != 128
        || args.max_features != width
        || c["init_seed"].as_u64() != args.init_seed
        || c["sample_seed"].as_u64() != args.sample_seed
        || args.init_seed.is_none()
        || args.sample_seed.is_none()
        || !args.nonneg_distance
        || args.nonneg_pin != 100.
        || !args.feature_transform.is_empty()
        || args.auto_transforms.is_some()
    {
        return Err("subjective registered cell/seed/full-budget/head binding differs".into());
    }
    Ok(())
}

fn family_masks(c: &Value, records: &[Value]) -> Result<(), String> {
    if c["route"] != "loso" {
        return Ok(());
    }
    let masks = json(
        Path::new(string(c, "all_leg_masks_path")?),
        string(c, "all_leg_masks_sha256")?,
    )?;
    if masks["schema"] != "source-family-native-mask-v1"
        || masks["excluded_families"] != c["excluded_families"]
    {
        return Err("LOSO native family-mask closure differs".into());
    }
    let excluded = c["excluded_families"]
        .as_array()
        .ok_or("LOSO exclusions required")?;
    let groups = masks["groups"]
        .as_object()
        .ok_or("all-leg family masks required")?;
    let names: BTreeSet<_> = records
        .iter()
        .map(|r| string(r, "name"))
        .collect::<Result<_, _>>()?;
    if groups.keys().map(String::as_str).collect::<BTreeSet<_>>() != names {
        return Err("LOSO every inherited and added group must be masked".into());
    }
    for r in records {
        let m = &groups[string(r, "name")?];
        let families = m["families"]
            .as_array()
            .ok_or("LOSO ordered families required")?;
        if m["manifest_sha256"] != r["manifest_sha256"]
            || m["keys_sha256"] != r["keys_sha256"]
            || r["rows"].as_u64() != Some(families.len() as u64)
            || families
                .iter()
                .any(|f| f.as_str().is_none_or(str::is_empty) || excluded.contains(f))
        {
            return Err("LOSO inherited/added family leak or mask pin mismatch".into());
        }
    }
    for stage in ["calibration", "preprocessing"] {
        let m = &masks[stage];
        let families = m["families"]
            .as_array()
            .ok_or("LOSO scaler/calibration family inventory required")?;
        if families
            .iter()
            .any(|f| f.as_str().is_none_or(str::is_empty) || excluded.contains(f))
        {
            return Err("LOSO scaler/calibration family leak".into());
        }
        let key = format!("{stage}_manifest_sha256");
        if c[&key] != m["manifest_sha256"] {
            return Err("LOSO preprocessing/calibration pin differs".into());
        }
        json(
            Path::new(string(m, "manifest_path")?),
            string(m, "manifest_sha256")?,
        )?;
    }
    Ok(())
}

pub(super) fn admit(
    authority_path: &Path,
    authority_pin: &str,
    cell: &str,
    groups: &[Group],
    selected: &[usize],
    run: Option<&super::Args>,
) -> Result<BTreeMap<String, Value>, String> {
    let a = json(authority_path, authority_pin)?;
    if a["schema"] == "subjective-training-authority-v2" {
        return admit_datasets(&a, cell, groups, selected, run);
    }
    if a["schema"] != "subjective-training-authority-v1"
        || a["state"] != "approved"
        || a["artifact_visibility"] != "private"
        || a["selection"] != "fixed-final-epoch"
    {
        return Err("subjective authority not admitted".into());
    }
    let c = &a["cells"][cell];
    if let Some(args) = run {
        run_binding(c, args, 1825)?;
    }
    let ids: Vec<usize> =
        serde_json::from_value(c["feature_ids"].clone()).map_err(|e| e.to_string())?;
    if ids != selected || c["formula_revision"] != 5 || c["tier"] != "v3" {
        return Err("subjective feature/producer contract differs".into());
    }
    let records = c["groups"]
        .as_array()
        .ok_or("subjective group inventory required")?;
    let names: BTreeSet<_> = groups.iter().map(|g| g.0.as_str()).collect();
    let expected: BTreeSet<_> = records
        .iter()
        .map(|r| string(r, "name"))
        .collect::<Result<_, _>>()?;
    if names != expected || groups.len() != names.len() || records.len() != names.len() {
        return Err("complete unique cell group inventory required".into());
    }
    let mut pending = Vec::new();
    for g in groups {
        let r = records.iter().find(|r| r["name"] == g.0).unwrap();
        let path = PathBuf::from(string(r, "path")?);
        safe(&path)?;
        safe(&g.1)?;
        let resolved = |p: &Path| p.canonicalize().map_err(|e| e.to_string());
        if resolved(&path)? != resolved(&g.1)?
            || !g.2.is_finite()
            || !g.3.is_finite()
            || r["train_weight"].as_f64() != Some(g.2)
            || r["val_weight"].as_f64() != Some(g.3)
        {
            return Err("subjective cell group path/weight mismatch".into());
        }
        let mode = if g.4 { "withinref," } else { "" }.to_string()
            + match g.5 {
                GroupLossMode::Rank => "rank",
                GroupLossMode::Both => "both",
                GroupLossMode::Mse => "mse",
            };
        if r["mode"] != mode {
            return Err("subjective cell loss mode mismatch".into());
        }
        let manifest = PathBuf::from(format!("{}.manifest.json", g.1.display()));
        safe(&manifest)?;
        safe(Path::new(string(r, "keys_path")?))?;
        pending.push((r, manifest));
    }
    family_masks(c, records)?;
    let declarations: Vec<_> = pending
        .iter()
        .map(|(r, p)| Ok((*r, json(p, string(r, "manifest_sha256")?)?)))
        .collect::<Result<_, String>>()?;
    let mut admitted = BTreeMap::new();
    for (r, d) in &declarations {
        if r["subjective"] != true {
            continue;
        }
        if d["schema"] != "subjective-training-leg-v1"
            || d["loss"] != "within-source-rank"
            || d["nominal_weight"] != 8
            || d["target_orientation"] != "quality"
            || d["target_transform"] != "negate-pointwise"
            || d["target_column"] != "human_score"
            || d["selection"] != "fixed-final-epoch"
            || d["formula_revision"] != 5
            || d["tier"] != "v3"
            || d["feature_ids"] != c["feature_ids"]
            || d["producer_commit"] != c["producer_commit"]
            || r["name"] != "subjective_rank"
            || r["mode"] != "withinref,rank"
            || r["val_weight"].as_f64() != Some(0.)
            || r["train_weight"].as_f64().is_none_or(|v| v <= 0.)
            || d["table_sha256"] != r["table_sha256"]
        {
            return Err("subjective role/loss/producer/target binding differs".into());
        }
        admitted.insert(string(r, "name")?.to_owned(), d.clone());
    }
    if admitted.len() != 1 {
        return Err("authority requires one aggregated unnamed subjective leg".into());
    }
    for (r, d) in &declarations {
        if r["subjective"] != true {
            continue;
        }
        let keys = json(
            Path::new(string(r, "keys_path")?),
            string(r, "keys_sha256")?,
        )?;
        let keys = keys.as_array().ok_or("subjective ordered keys required")?;
        if keys.is_empty()
            || r["rows"].as_u64() != Some(keys.len() as u64)
            || d["rows"] != r["rows"]
        {
            return Err("subjective ordered key count differs".into());
        }
        let sources = c["sources"]
            .as_array()
            .ok_or("subjective source inventory required")?;
        let excluded = c["excluded_families"]
            .as_array()
            .ok_or("subjective exclusions required")?;
        let mut observations = BTreeSet::new();
        let mut counts = BTreeMap::<&str, usize>::new();
        let mut references = BTreeMap::new();
        let allowed: BTreeSet<_> = [
            "observation",
            "source",
            "family",
            "reference",
            "role",
            "tier",
            "row",
        ]
        .into_iter()
        .collect();
        for (i, k) in keys.iter().enumerate() {
            let fields: BTreeSet<_> = k
                .as_object()
                .ok_or("subjective key object required")?
                .keys()
                .map(String::as_str)
                .collect();
            let source = string(k, "source")?;
            let reference = string(k, "reference")?;
            if fields != allowed
                || k["row"].as_u64() != Some(i as u64)
                || k["role"] != "fit"
                || k["tier"] != "T2"
                || !sources.contains(&k["source"])
                || excluded.contains(&k["family"])
                || !observations.insert(string(k, "observation")?)
                || reference.is_empty()
                || ["source", "family", "observation"]
                    .iter()
                    .any(|f| !string(k, f).is_ok_and(|s| !s.is_empty()))
            {
                return Err("subjective source/observation/order/role mismatch".into());
            }
            if references
                .insert(source, reference)
                .is_some_and(|old| old != reference)
            {
                return Err("subjective source must have one pairing reference".into());
            }
            *counts.entry(source).or_default() += 1;
        }
        if serde_json::to_value(&counts).map_err(|e| e.to_string())? != r["source_counts"]
            || counts.values().any(|&n| n < 2)
            || counts.keys().copied().collect::<BTreeSet<_>>()
                != sources.iter().filter_map(Value::as_str).collect()
            || references.values().copied().collect::<BTreeSet<_>>().len() != references.len()
        {
            return Err("subjective source population differs from authority".into());
        }
        let corrected =
            8. / (counts.values().map(|&n| 1. - 1. / n as f64).sum::<f64>() / counts.len() as f64);
        if r["train_weight"].as_f64() != Some(corrected) {
            return Err("subjective nominal-eight acceptance correction differs".into());
        }
    }
    Ok(admitted)
}

// Fresh revision-7 authority. The historical v1 experiment above retains its
// exact revision, single group and nominal-eight contract.
fn admit_datasets(
    a: &Value,
    cell: &str,
    groups: &[Group],
    selected: &[usize],
    run: Option<&super::Args>,
) -> Result<BTreeMap<String, Value>, String> {
    if a["state"] != "approved"
        || a["artifact_visibility"] != "private"
        || a["selection"] != "fixed-final-epoch"
    {
        return Err("dataset authority not admitted".into());
    }
    let c = &a["cells"][cell];
    if let Some(args) = run {
        let width = c["width"]
            .as_u64()
            .filter(|&w| matches!(w, 1825 | 1853))
            .ok_or("dataset declared training width required")? as usize;
        run_binding(c, args, width)?;
    }
    let ids: Vec<usize> =
        serde_json::from_value(c["read_ids"].clone()).map_err(|e| e.to_string())?;
    if ids != selected || c["formula_revision"] != 7 || c["tier"] != "v3" {
        return Err("dataset feature/producer contract differs".into());
    }
    let guard = json(
        Path::new(string(c, "family_guard_path")?),
        string(c, "family_guard_sha256")?,
    )?;
    if guard["schema"] != "dataset-family-guard-v1" || guard["inherited_leaks"] != 0 {
        return Err("dataset family guard differs".into());
    }
    let excluded: BTreeSet<String> =
        serde_json::from_value(guard["protected_families"].clone()).map_err(|e| e.to_string())?;
    let records = c["groups"]
        .as_array()
        .ok_or("dataset group inventory required")?;
    let names: BTreeSet<_> = groups.iter().map(|g| g.0.as_str()).collect();
    let expected: BTreeSet<_> = records
        .iter()
        .map(|r| string(r, "name"))
        .collect::<Result<_, _>>()?;
    if names != expected || names.len() != groups.len() || expected.len() != records.len() {
        return Err("complete unique dataset cell inventory required".into());
    }
    let mut pending = Vec::new();
    for g in groups {
        let r = records.iter().find(|r| r["name"] == g.0).unwrap();
        let path = PathBuf::from(string(r, "path")?);
        safe(&path)?;
        safe(&g.1)?;
        if path.canonicalize().map_err(|e| e.to_string())?
            != g.1.canonicalize().map_err(|e| e.to_string())?
            || !g.2.is_finite()
            || !g.3.is_finite()
            || r["train_weight"].as_f64() != Some(g.2)
            || r["val_weight"].as_f64() != Some(g.3)
            || r["mode"] != "withinref,rank" && r["subjective"] == true
        {
            return Err("dataset cell group path/mode/weight mismatch".into());
        }
        let mode = if g.4 { "withinref," } else { "" }.to_string()
            + match g.5 {
                GroupLossMode::Rank => "rank",
                GroupLossMode::Both => "both",
                GroupLossMode::Mse => "mse",
            };
        if r["mode"] != mode {
            return Err("dataset cell loss mode mismatch".into());
        }
        let manifest = PathBuf::from(format!("{}.manifest.json", g.1.display()));
        safe(&manifest)?;
        safe(Path::new(string(r, "keys_path")?))?;
        pending.push((r, manifest));
    }
    let declarations: Vec<_> = pending
        .iter()
        .map(|(r, p)| Ok((*r, json(p, string(r, "manifest_sha256")?)?)))
        .collect::<Result<_, String>>()?;
    let mut admitted = BTreeMap::new();
    for (r, d) in declarations {
        if r["subjective"] != true {
            continue;
        }
        if d["schema"] != "subjective-training-leg-v2"
            || !matches!(
                d["loss"].as_str(),
                Some("within-source-rank" | "hard-pair-preference")
            )
            || d["nominal_weight"].as_f64() != Some(0.5)
            || d["target_orientation"] != "quality"
            || d["target_transform"] != "orient-native-quality"
            || d["target_column"] != "human_score"
            || d["formula_revision"] != 7
            || d["tier"] != "v3"
            || d["feature_ids"] != c["feature_ids"]
            || d["selection"] != "fixed-final-epoch"
            || d["native_role"] != "TRAIN"
            || d["producer_commit"] != c["producer_commit"]
            || !string(r, "name")?.starts_with("subjective_rank_")
            || r["val_weight"].as_f64() != Some(0.)
            || r["train_weight"].as_f64().is_none_or(|v| v <= 0.)
            || d["table_sha256"] != r["table_sha256"]
        {
            return Err("dataset role/loss/producer/target binding differs".into());
        }
        let keys = json(
            Path::new(string(r, "keys_path")?),
            string(r, "keys_sha256")?,
        )?;
        let keys = keys.as_array().ok_or("dataset ordered keys required")?;
        if keys.is_empty()
            || r["rows"].as_u64() != Some(keys.len() as u64)
            || d["rows"] != r["rows"]
        {
            return Err("dataset ordered key count differs".into());
        }
        let allowed: BTreeSet<_> = [
            "observation",
            "source",
            "family",
            "reference",
            "role",
            "tier",
            "row",
        ]
        .into_iter()
        .collect();
        let mut observations = BTreeSet::new();
        let mut counts = BTreeMap::<&str, usize>::new();
        for (i, k) in keys.iter().enumerate() {
            let fields: BTreeSet<_> = k
                .as_object()
                .ok_or("dataset key object required")?
                .keys()
                .map(String::as_str)
                .collect();
            if fields != allowed
                || k["row"].as_u64() != Some(i as u64)
                || k["role"] != "fit"
                || k["tier"] != "T2"
                || excluded.contains(string(k, "family")?)
                || !observations.insert(string(k, "observation")?)
                || k["source"] != k["reference"]
                || ["observation", "source", "family", "reference"]
                    .iter()
                    .any(|f| !string(k, f).is_ok_and(|s| !s.is_empty()))
            {
                return Err("dataset key role/order/family mismatch".into());
            }
            *counts.entry(string(k, "source")?).or_default() += 1;
        }
        if serde_json::to_value(&counts).map_err(|e| e.to_string())? != r["source_counts"]
            || counts
                .values()
                .any(|&n| n < 2 || d["loss"] == "hard-pair-preference" && n != 2)
        {
            return Err("dataset source population mismatch".into());
        }
        let corrected =
            0.5 / (counts.values().map(|&n| 1. - 1. / n as f64).sum::<f64>() / counts.len() as f64);
        // Python 3.12+ sums with compensation; Iterator::sum uses ordinary
        // ordered addition. Bind the CLI weight exactly above, allowing only
        // final arithmetic roundoff when independently checking its formula.
        if r["train_weight"]
            .as_f64()
            .is_none_or(|w| (w - corrected).abs() > 32. * f64::EPSILON * corrected.abs())
        {
            return Err(format!(
                "dataset nominal-half acceptance correction differs: {} stored={} computed={corrected:.17}",
                string(r, "name")?,
                r["train_weight"]
            ));
        }
        admitted.insert(string(r, "name")?.to_owned(), d);
    }
    if admitted.is_empty() {
        return Err("dataset authority has no subjective groups".into());
    }
    Ok(admitted)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQUENCE: AtomicU64 = AtomicU64::new(0);
    struct Fixture {
        root: PathBuf,
        table: PathBuf,
        keys: Value,
        declaration: Value,
        authority: Value,
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.root).unwrap();
        }
    }
    impl Fixture {
        fn new() -> Self {
            let root = std::env::temp_dir().join(format!(
                "subjective-fixture-{}-{}",
                std::process::id(),
                SEQUENCE.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&root).unwrap();
            let table = root.join("invented.parquet");
            std::fs::write(
                &table,
                b"not parquet; metadata phase must not parse or hash targets",
            )
            .unwrap();
            let keys = json!([
                {"observation":"o0","source":"s","family":"f","reference":"r","role":"fit","tier":"T2","row":0},
                {"observation":"o1","source":"s","family":"f","reference":"r","role":"fit","tier":"T2","row":1}]);
            let declaration = json!({"schema":"subjective-training-leg-v1","loss":"within-source-rank","nominal_weight":8,
                "target_orientation":"quality","target_transform":"negate-pointwise","target_column":"human_score",
                "selection":"fixed-final-epoch","formula_revision":5,"tier":"v3","feature_ids":[0,1],
                "producer_commit":"invented-producer","table_sha256":"f".repeat(64),"rows":2});
            let authority = json!({"schema":"subjective-training-authority-v1","state":"approved","artifact_visibility":"private",
                "selection":"fixed-final-epoch","cells":{"cell":{"feature_ids":[0,1],"formula_revision":5,"tier":"v3",
                    "producer_commit":"invented-producer","sources":["s"],"excluded_families":[],"groups":[{
                        "name":"subjective_rank","path":table,"train_weight":16.,"val_weight":0.,"mode":"withinref,rank",
                        "subjective":true,"table_sha256":"f".repeat(64),"keys_path":root.join("keys.json"),"rows":2,"source_counts":{"s":2}}]}}});
            Self {
                root,
                table,
                keys,
                declaration,
                authority,
            }
        }
        fn call(&mut self) -> Result<BTreeMap<String, Value>, String> {
            self.call_named("subjective_rank", 16.)
        }
        fn call_named(
            &mut self,
            name: &str,
            weight: f64,
        ) -> Result<BTreeMap<String, Value>, String> {
            let keys = self.root.join("keys.json");
            let declaration = PathBuf::from(format!("{}.manifest.json", self.table.display()));
            std::fs::write(&keys, serde_json::to_vec(&self.keys).unwrap()).unwrap();
            std::fs::write(&declaration, serde_json::to_vec(&self.declaration).unwrap()).unwrap();
            self.authority["cells"]["cell"]["groups"][0]["manifest_sha256"] =
                json!(hash(&declaration).unwrap());
            self.authority["cells"]["cell"]["groups"][0]["keys_sha256"] =
                json!(hash(&keys).unwrap());
            let authority = self.root.join("authority.json");
            std::fs::write(&authority, serde_json::to_vec(&self.authority).unwrap()).unwrap();
            admit(
                &authority,
                &hash(&authority).unwrap(),
                "cell",
                &[(
                    name.into(),
                    self.table.clone(),
                    weight,
                    0.,
                    true,
                    GroupLossMode::Rank,
                )],
                &[0, 1],
                None,
            )
        }
    }
    #[test]
    fn dataset_v2_keeps_legacy_authority_and_enforces_family_and_pair_keys() {
        let mut f = Fixture::new();
        let guard = f.root.join("guard.json");
        std::fs::write(&guard, br#"{"schema":"dataset-family-guard-v1","protected_families":["reserved"],"inherited_leaks":0}"#).unwrap();
        f.authority["schema"] = json!("subjective-training-authority-v2");
        let c = &mut f.authority["cells"]["cell"];
        c["formula_revision"] = json!(7);
        c["read_ids"] = c["feature_ids"].clone();
        c["family_guard_path"] = json!(guard);
        c["family_guard_sha256"] = json!(hash(&guard).unwrap());
        c["groups"][0]["name"] = json!("subjective_rank_fixture");
        c["groups"][0]["train_weight"] = json!(1.);
        f.declaration["schema"] = json!("subjective-training-leg-v2");
        f.declaration["formula_revision"] = json!(7);
        f.declaration["nominal_weight"] = json!(0.5);
        f.declaration["target_transform"] = json!("orient-native-quality");
        f.declaration["native_role"] = json!("TRAIN");
        f.declaration["loss"] = json!("hard-pair-preference");
        for k in f.keys.as_array_mut().unwrap() {
            k["reference"] = k["source"].clone();
        }
        assert_eq!(
            f.call_named("subjective_rank_fixture", 1.).unwrap().len(),
            1
        );
        f.keys[0]["family"] = json!("reserved");
        assert!(
            f.call_named("subjective_rank_fixture", 1.)
                .unwrap_err()
                .contains("family")
        );
        f.keys[0]["family"] = json!("f");
        f.declaration["native_role"] = json!("EVAL");
        assert!(
            f.call_named("subjective_rank_fixture", 1.)
                .unwrap_err()
                .contains("role")
        );
    }
    #[test]
    fn subjective_native_metadata_never_hashes_or_parses_targets() {
        let mut f = Fixture::new();
        // Intentionally false table pin, and no parquet header. Admission must
        // still succeed here; payload verification is a distinct later phase.
        assert_eq!(f.call().unwrap().len(), 1);
    }
    #[test]
    fn subjective_native_rejects_roles_order_producer_and_leaks() {
        let mut f = Fixture::new();
        for (field, value) in [
            ("role", json!("dev")),
            ("row", json!(5)),
            ("source", json!("late")),
            ("observation", json!("o0")),
        ] {
            let old = f.keys[1][field].clone();
            f.keys[1][field] = value;
            assert!(f.call().unwrap_err().contains("mismatch"));
            f.keys[1][field] = old;
        }
        f.declaration["producer_commit"] = json!("wrong-producer");
        assert!(f.call().unwrap_err().contains("producer"));
        f.declaration["producer_commit"] = json!("invented-producer");
        f.authority["cells"]["cell"]["excluded_families"] = json!(["f"]);
        assert!(f.call().unwrap_err().contains("mismatch"));
    }
    #[test]
    fn subjective_native_rejects_added_group_and_wrong_weight() {
        let mut f = Fixture::new();
        let mut late = f.authority["cells"]["cell"]["groups"][0].clone();
        late["name"] = json!("late");
        f.authority["cells"]["cell"]["groups"]
            .as_array_mut()
            .unwrap()
            .push(late);
        assert!(f.call().unwrap_err().contains("inventory"));
        f.authority["cells"]["cell"]["groups"]
            .as_array_mut()
            .unwrap()
            .pop();
        f.authority["cells"]["cell"]["groups"][0]["val_weight"] = json!(1.);
        assert!(f.call().unwrap_err().contains("weight"));
    }
    #[test]
    fn subjective_native_all_leg_masks_reject_inherited_and_scaler_leaks() {
        let f = Fixture::new();
        let record = serde_json::json!({"name":"inherited","manifest_sha256":"a".repeat(64),
            "keys_sha256":"b".repeat(64),"rows":1});
        let path = f.root.join("masks.json");
        let mut masks = serde_json::json!({"schema":"source-family-native-mask-v1","excluded_families":["excluded"],
            "groups":{"inherited":{"manifest_sha256":"a".repeat(64),"keys_sha256":"b".repeat(64),"families":["excluded"]}}});
        std::fs::write(&path, serde_json::to_vec(&masks).unwrap()).unwrap();
        let mut c = serde_json::json!({"route":"loso","excluded_families":["excluded"],
            "all_leg_masks_path":path,"all_leg_masks_sha256":hash(&path).unwrap()});
        assert!(
            family_masks(&c, std::slice::from_ref(&record))
                .unwrap_err()
                .contains("leak")
        );
        masks["groups"]["inherited"]["families"] = serde_json::json!(["allowed"]);
        masks["calibration"] = serde_json::json!({"families":["excluded"]});
        std::fs::write(&path, serde_json::to_vec(&masks).unwrap()).unwrap();
        c["all_leg_masks_sha256"] = serde_json::json!(hash(&path).unwrap());
        assert!(
            family_masks(&c, std::slice::from_ref(&record))
                .unwrap_err()
                .contains("scaler/calibration")
        );
    }
}
