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

/// Registered continuation protocols beyond the frozen 0/4/8-epoch contract
/// (byconstruction record, 2026-10-27): each names its populations with their
/// public_train weight, and its retained checkpoint epochs. Every one runs 60
/// epochs at constant LR 1e-5, fresh Adam, a checkpoint dump every epoch. The
/// Python mirror holds the same table.
pub(super) const DIAGNOSTIC_EPOCHS: u64 = 60;
type Protocol = (&'static str, &'static [(&'static str, f64)], &'static [u64]);
/// Stage D (byconstruction record, 2026-10-27): the one registered rehearsal
/// leg repeats the human leg with its ranking pairs drawn within each older
/// dataset, named by the keys' member sets.
pub(super) const WITHIN_SET_GROUP: &str = "human_xref";
const XREF_PROTOCOL: &str = "aic-share-xref-v1";
const DATASET_OF_MEMBER: [(&str, &str); 6] = [
    ("kadid_train", "kadid"),
    ("kadid_select", "kadid"),
    ("tid2013", "tid2013"),
    ("cid22_a25", "cid22_a25"),
    ("konfig_train", "konfig"),
    ("konfig_val", "konfig"),
];

/// The rehearsal leg's record: the base human leg's own record (path,
/// weights, manifest and keys) under its name, pairing within each dataset.
fn xref_record(base_groups: &[Value]) -> Result<Value, String> {
    let human: Vec<_> = base_groups
        .iter()
        .filter(|r| r["name"] == "human")
        .collect();
    if human.len() != 1 || human[0]["mode"] != "withinref,rank" || human[0]["subjective"] == true {
        return Err("rehearsal requires the base within-reference human leg".into());
    }
    let mut record = human[0].clone();
    record["name"] = Value::from(WITHIN_SET_GROUP);
    record["mode"] = Value::from("withinset,rank");
    Ok(record)
}

/// Dense per-row dataset ids of the rehearsal leg, in table order, from its
/// keys file; any unregistered member set refuses.
pub(super) fn dataset_ids(table: &Path, rows: usize) -> Result<Vec<u32>, String> {
    let columns = ["pair_key", "source_row_id", "ref_basename", "member_set"];
    let keys =
        super::training_keys::read(&table.with_extension("keys.parquet"), &columns, &columns)?;
    if keys.rows.len() != rows {
        return Err("rehearsal keys and table row counts differ".into());
    }
    let member = keys.column("member_set")?;
    let mut ids = BTreeMap::new();
    keys.rows
        .iter()
        .map(|row| {
            let (_, dataset) = DATASET_OF_MEMBER
                .iter()
                .find(|(m, _)| *m == row[member])
                .ok_or("unregistered rehearsal member set")?;
            let next = ids.len() as u32;
            Ok(*ids.entry(*dataset).or_insert(next))
        })
        .collect()
}

const DIAGNOSTIC_PROTOCOLS: [Protocol; 3] = [
    (
        "aic-matched-continuation-diagnostic-v1",
        &[("mixed-aic16", 16.0), ("aic-only", 16.0)],
        &[7, 59],
    ),
    (
        "aic-share-frontier-v1",
        &[
            ("mixed-aic64", 64.0),
            ("mixed-aic256", 256.0),
            ("mixed-aic1024", 1024.0),
        ],
        &[7, 15, 29, 59],
    ),
    (
        XREF_PROTOCOL,
        &[("mixed-aic256-xref", 256.0), ("mixed-aic1024-xref", 1024.0)],
        &[7, 15, 29, 59],
    ),
];

/// The public_train weight and retained checkpoints of a continuation's
/// registered protocol and population.
fn diagnostic_protocol(w: &Value) -> Option<(f64, &'static [u64])> {
    let (_, populations, checkpoints) =
        DIAGNOSTIC_PROTOCOLS.iter().find(|p| w["protocol"] == p.0)?;
    let weight = populations.iter().find(|p| w["population"] == p.0)?.1;
    Some((weight, checkpoints))
}

/// The registered population of the continuation that this run's pinned
/// version-4 authority binds to its cell; None for any other authority. The
/// recipe owners (matched codecs) call this after `admit` has bound the run.
pub(super) fn continuation_population(args: &super::Args) -> Result<Option<String>, String> {
    let (Some(path), Some(pin), Some(cell)) = (
        &args.subjective_authority,
        &args.subjective_authority_sha256,
        &args.subjective_cell,
    ) else {
        return Ok(None);
    };
    let a = json(path, pin)?;
    if a["schema"] != "subjective-training-authority-v4" {
        return Ok(None);
    }
    let w = &a["cells"][cell.as_str()]["continuation"];
    diagnostic_protocol(w).ok_or("registered continuation protocol required")?;
    Ok(Some(string(w, "population")?.to_owned()))
}

fn run_binding(
    c: &Value,
    args: &super::Args,
    width: usize,
    diagnostic: bool,
) -> Result<(), String> {
    let continuation = c.get("continuation");
    if args.init_from.is_some() && continuation.is_none() {
        return Err("checkpoint initialization requires its continuation binding".into());
    }
    if let Some(w) = continuation {
        let path = args
            .init_from
            .as_ref()
            .ok_or("continuation checkpoint required")?;
        let budget = if diagnostic {
            DIAGNOSTIC_PROTOCOLS.iter().any(|p| w["protocol"] == p.0)
                && args.epochs as u64 == DIAGNOSTIC_EPOCHS
                && args.dump_checkpoints_every == 1
                && args.dump_checkpoints_dir.is_some()
        } else {
            w.get("protocol").is_none() && matches!(args.epochs, 0 | 4 | 8)
        };
        if w["checkpoint_sha256"] != hash(path)?
            || w["checkpoint"] != path.to_string_lossy().as_ref()
            || w["epochs"].as_u64() != Some(args.epochs as u64)
            || !budget
            || w["lr"].as_f64() != Some(args.lr)
            || args.lr != 0.00001
            || args.lr_schedule != "constant"
            || w["lr_schedule"] != "constant"
        {
            return Err("continuation checkpoint/budget/schedule binding differs".into());
        }
    }
    let limit = if c["route"] == "loso" { 3 } else { 10 };
    if !matches!(c["route"].as_str(), Some("full" | "d1" | "loso"))
        || c["seed_index"].as_u64().is_none_or(|v| v >= limit)
        || c["epochs"] != 120
        || (continuation.is_none() && args.epochs != 120)
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
    if a["schema"] == "subjective-training-authority-v2"
        || a["schema"] == "subjective-training-authority-v3"
        || a["schema"] == "subjective-training-authority-v4"
    {
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
        run_binding(c, args, 1825, false)?;
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
    let filtered;
    let diagnostic = a["schema"] == "subjective-training-authority-v4";
    if a["schema"] != "subjective-training-authority-v3"
        && !diagnostic
        && a["cells"][cell].get("continuation").is_some()
    {
        return Err("continuation requires version-3 authority".into());
    }
    let aic_only = if diagnostic {
        diagnostic_binding(a, cell, groups)?
    } else {
        false
    };
    let groups = if a["schema"] == "subjective-training-authority-v3" || diagnostic {
        if a["cells"][cell].get("continuation").is_none() {
            return Err("continuation authority requires its checkpoint binding".into());
        }
        filtered = groups
            .iter()
            .filter(|g| g.0 != "aic_public_train")
            .cloned()
            .collect::<Vec<_>>();
        filtered.as_slice()
    } else {
        groups
    };
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
        run_binding(c, args, width, diagnostic)?;
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
        let pairing = if g.0 == WITHIN_SET_GROUP {
            "withinset,"
        } else if g.4 {
            "withinref,"
        } else {
            ""
        };
        let mode = pairing.to_string()
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
        if d["schema"] == "chroma-teacher-leg-v1" {
            // The leg declares the cell's direct ids; `selected` is the (larger) read set of a derived declaration.
            let direct: Vec<usize> =
                serde_json::from_value(c["feature_ids"].clone()).map_err(|e| e.to_string())?;
            super::chroma_admission::admit(&d, r, &direct, &excluded, json)?;
            admitted.insert(string(r, "name")?.to_owned(), d);
            continue;
        }
        if d["schema"] == "hdr-native-leg-v1" {
            super::hdr_admission::admit(&d, r, selected, &excluded, json)?;
            admitted.insert(string(r, "name")?.to_owned(), d);
            continue;
        }
        if r["subjective"] != true {
            continue;
        }
        let votes = d["schema"] == "subjective-training-leg-v3";
        let legacy = d["schema"] == "subjective-training-leg-v2"
            && matches!(
                d["loss"].as_str(),
                Some("within-source-rank" | "hard-pair-preference")
            )
            && d["target_transform"] == "orient-native-quality"
            && string(r, "name")?.starts_with("subjective_rank_");
        let vote_contract = votes
            && d["loss"] == "binomial-two-vote"
            && d["target_transform"] == "native-vote-probability"
            && d["collection"] == "bapps"
            && d["native_votes_per_pair"] == 2
            && d["nominal_weight"].as_f64() == Some(0.5)
            && r["name"] == "subjective_vote_bapps"
            && d["keys_path"] == r["keys_path"]
            && d["keys_sha256"] == r["keys_sha256"];
        if !(legacy || vote_contract)
            || d["nominal_weight"]
                .as_f64()
                .is_none_or(|w| !w.is_finite() || w <= 0.)
            || d["target_orientation"] != "quality"
            || d["target_column"] != "human_score"
            || d["formula_revision"] != 7
            || d["tier"] != "v3"
            || d["feature_ids"] != c["feature_ids"]
            || d["selection"] != "fixed-final-epoch"
            || d["native_role"] != "TRAIN"
            || d["producer_commit"] != c["producer_commit"]
            || r["val_weight"].as_f64() != Some(0.)
            || r["train_weight"].as_f64().is_none_or(|v| v <= 0.)
            || d["table_sha256"] != r["table_sha256"]
        {
            return Err("dataset role/loss/producer/target binding differs".into());
        }
        if votes {
            safe(Path::new(string(&d, "native_votes_path")?))?;
            let pin = string(&d, "native_votes_sha256")?;
            if pin.len() != 64
                || !pin
                    .bytes()
                    .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
            {
                return Err("native vote payload pin required".into());
            }
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
                .any(|&n| n < 2 || (d["loss"] == "hard-pair-preference" || votes) && n != 2)
        {
            return Err("dataset source population mismatch".into());
        }
        let corrected = d["nominal_weight"].as_f64().unwrap()
            / (counts.values().map(|&n| 1. - 1. / n as f64).sum::<f64>() / counts.len() as f64);
        // Python 3.12+ sums with compensation; Iterator::sum uses ordinary
        // ordered addition. Bind the CLI weight exactly above, allowing only
        // final arithmetic roundoff when independently checking its formula.
        if r["train_weight"]
            .as_f64()
            .is_none_or(|w| (w - corrected).abs() > 32. * f64::EPSILON * corrected.abs())
        {
            return Err(format!(
                "dataset nominal-weight acceptance correction differs: {} stored={} computed={corrected:.17}",
                string(r, "name")?,
                r["train_weight"]
            ));
        }
        admitted.insert(string(r, "name")?.to_owned(), d);
    }
    // The AIC-only diagnostic arm explicitly excludes every training leg,
    // including all subjective legs; its exclusion is bound to the base above.
    if admitted.is_empty() && !aic_only {
        return Err("dataset authority has no subjective groups".into());
    }
    Ok(admitted)
}

/// Bind a diagnostic authority to its unmodified, independently pinned
/// version-2 base. Returns whether this is the AIC-only arm.
///
/// The mixed arm keeps every base group. The AIC-only arm keeps exactly the
/// base's validation-only groups (train weight zero, base order) and lists
/// every training group (train weight positive, base order) as excluded.
/// Every other field of the authority and of every cell must equal the base.
fn diagnostic_binding(a: &Value, cell: &str, groups: &[Group]) -> Result<bool, String> {
    let base = json(
        Path::new(string(a, "base_authority")?),
        string(a, "base_authority_sha256")?,
    )?;
    if base["schema"] != "subjective-training-authority-v2" {
        return Err("diagnostic base must be the original version-2 authority".into());
    }
    let (Some(top), Some(base_top)) = (a.as_object(), base.as_object()) else {
        return Err("diagnostic authority objects required".into());
    };
    let outer = ["schema", "base_authority", "base_authority_sha256", "cells"];
    let strip = |o: &serde_json::Map<String, Value>, skip: &[&str]| -> BTreeMap<String, Value> {
        o.iter()
            .filter(|(k, _)| !skip.contains(&k.as_str()))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    };
    if strip(top, &outer) != strip(base_top, &outer) {
        return Err("diagnostic authority changed a base field".into());
    }
    let (Some(cells), Some(base_cells)) = (a["cells"].as_object(), base["cells"].as_object())
    else {
        return Err("diagnostic cell inventory required".into());
    };
    if cells.keys().ne(base_cells.keys()) {
        return Err("diagnostic cell inventory differs from its base".into());
    }
    for (name, value) in cells {
        if name != cell && value != &base_cells[name] {
            return Err("diagnostic authority changed another cell".into());
        }
    }
    let (Some(c), Some(b)) = (
        a["cells"][cell].as_object(),
        base["cells"][cell].as_object(),
    ) else {
        return Err("diagnostic cell required".into());
    };
    if b.contains_key("continuation")
        || strip(c, &["groups", "continuation"]) != strip(b, &["groups", "continuation"])
    {
        return Err("diagnostic cell changed a base binding".into());
    }
    let w = &a["cells"][cell]["continuation"];
    let (aic_weight, checkpoints) =
        diagnostic_protocol(w).ok_or("diagnostic protocol/population required")?;
    let aic_only = w["population"] == "aic-only";
    let base_groups = base["cells"][cell]["groups"]
        .as_array()
        .ok_or("diagnostic base group inventory required")?;
    let weight = |r: &Value| r["train_weight"].as_f64().filter(|v| v.is_finite());
    if base_groups.iter().any(|r| weight(r).is_none_or(|v| v < 0.)) {
        return Err("diagnostic base train weights required".into());
    }
    let (mut retained, excluded): (Vec<Value>, Vec<Value>) = if aic_only {
        base_groups
            .iter()
            .cloned()
            .partition(|r| weight(r) == Some(0.))
    } else {
        (base_groups.clone(), Vec::new())
    };
    if w["protocol"] == XREF_PROTOCOL {
        retained.push(xref_record(base_groups)?);
    }
    if w["epochs"].as_u64() != Some(DIAGNOSTIC_EPOCHS)
        || w["aic_weight"].as_f64() != Some(aic_weight)
        || w["dump_checkpoints_every"] != 1
        || w["retained_checkpoint_epochs"] != serde_json::json!(checkpoints)
        || w["optimizer_state"] != "fresh"
        || a["cells"][cell]["groups"] != Value::Array(retained.clone())
        || w["excluded_groups"] != Value::Array(excluded.clone())
        || aic_only && (retained.is_empty() || excluded.is_empty())
    {
        return Err("diagnostic protocol/population/group partition differs".into());
    }
    let aic: Vec<_> = groups
        .iter()
        .filter(|g| g.0 == "aic_public_train")
        .collect();
    if aic.len() != 1 || aic[0].2 != aic_weight {
        return Err(format!(
            "diagnostic requires one public_train leg at weight {aic_weight}"
        ));
    }
    Ok(aic_only)
}

/// Only called after all metadata admission and payload hashes, with the
/// loaded targets at native scale. No vote counts enter metadata admission.
pub fn native_votes(d: &Value, targets: &[f64], refs: Option<&[u32]>) -> Result<(), String> {
    if d["schema"] != "subjective-training-leg-v3" {
        return Ok(());
    }
    let votes = json(
        Path::new(string(d, "native_votes_path")?),
        string(d, "native_votes_sha256")?,
    )?;
    let keys = json(
        Path::new(string(d, "keys_path")?),
        string(d, "keys_sha256")?,
    )?;
    let keys = keys.as_array().ok_or("native vote ordered keys required")?;
    let comparisons = votes["comparisons"]
        .as_array()
        .ok_or("native vote comparisons required")?;
    let refs = refs.ok_or("native vote pairing references required")?;
    if votes["schema"] != "bapps-native-votes-v1"
        || comparisons.is_empty()
        || comparisons.len().checked_mul(2) != Some(keys.len())
        || keys.len() != targets.len()
        || refs.len() != targets.len()
    {
        return Err("native vote population differs".into());
    }
    let mut paired_refs = BTreeSet::new();
    for (i, v) in comparisons.iter().enumerate() {
        let fields: BTreeSet<_> = v
            .as_object()
            .ok_or("native vote object required")?
            .keys()
            .map(String::as_str)
            .collect();
        if fields
            != ["source", "n_a", "n_b", "n_tie", "n", "p_a"]
                .into_iter()
                .collect()
        {
            return Err("native vote count fields required".into());
        }
        let na = v["n_a"]
            .as_u64()
            .ok_or("native integer vote count required")?;
        let nb = v["n_b"]
            .as_u64()
            .ok_or("native integer vote count required")?;
        let p = na as f64 / 2.;
        if na.checked_add(nb) != Some(2)
            || v["n_tie"].as_u64() != Some(0)
            || v["n"].as_u64() != Some(2)
            || v["p_a"].as_f64() != Some(p)
        {
            return Err("native two-vote count/probability binding differs".into());
        }
        let source = string(v, "source")?;
        for (side, probability, row) in [("a", p, 2 * i), ("b", 1. - p, 2 * i + 1)] {
            if keys[row]["source"] != source
                || keys[row]["observation"] != format!("{source}:{side}")
                || targets[row] != probability
            {
                return Err("native vote ordered key/target binding differs".into());
            }
        }
        if keys[2 * i]["family"] != keys[2 * i + 1]["family"]
            || refs[2 * i] != refs[2 * i + 1]
            || !paired_refs.insert(refs[2 * i])
        {
            return Err("native vote pair family/reference differs".into());
        }
    }
    Ok(())
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
        for nominal in [0.25, 1.] {
            f.declaration["nominal_weight"] = json!(nominal);
            f.authority["cells"]["cell"]["groups"][0]["train_weight"] = json!(nominal * 2.);
            assert_eq!(
                f.call_named("subjective_rank_fixture", nominal * 2.)
                    .unwrap()
                    .len(),
                1
            );
        }
        f.declaration["nominal_weight"] = json!(0.5);
        f.authority["cells"]["cell"]["groups"][0]["train_weight"] = json!(1.);
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

    fn vote_fixture() -> Fixture {
        let mut f = Fixture::new();
        let guard = f.root.join("guard.json");
        std::fs::write(&guard, br#"{"schema":"dataset-family-guard-v1","protected_families":["reserved"],"inherited_leaks":0}"#).unwrap();
        f.authority["schema"] = json!("subjective-training-authority-v2");
        let c = &mut f.authority["cells"]["cell"];
        c["formula_revision"] = json!(7);
        c["read_ids"] = c["feature_ids"].clone();
        c["family_guard_path"] = json!(guard);
        c["family_guard_sha256"] = json!(hash(&guard).unwrap());
        c["groups"][0]["name"] = json!("subjective_vote_bapps");
        c["groups"][0]["train_weight"] = json!(1.);
        for (i, k) in f.keys.as_array_mut().unwrap().iter_mut().enumerate() {
            k["reference"] = k["source"].clone();
            k["observation"] = json!(format!("s:{}", ["a", "b"][i]));
        }
        let keys = f.root.join("keys.json");
        std::fs::write(&keys, serde_json::to_vec(&f.keys).unwrap()).unwrap();
        let d = &mut f.declaration;
        d["schema"] = json!("subjective-training-leg-v3");
        d["formula_revision"] = json!(7);
        d["nominal_weight"] = json!(0.5);
        d["loss"] = json!("binomial-two-vote");
        d["target_transform"] = json!("native-vote-probability");
        d["native_role"] = json!("TRAIN");
        d["collection"] = json!("bapps");
        d["native_votes_per_pair"] = json!(2);
        d["keys_path"] = json!(keys);
        d["keys_sha256"] = json!(hash(&keys).unwrap());
        d["native_votes_path"] = json!(f.root.join("votes.json"));
        d["native_votes_sha256"] = json!("a".repeat(64));
        f
    }

    #[test]
    fn vote_metadata_admission_never_opens_response_payload_and_refuses_contract_drift() {
        let mut f = vote_fixture();
        // The response file deliberately does not exist during admission.
        assert_eq!(f.call_named("subjective_vote_bapps", 1.).unwrap().len(), 1);
        assert!(!f.root.join("votes.json").exists());
        for (field, value) in [
            ("loss", json!("hard-pair-preference")),
            ("native_role", json!("EVAL")),
            ("nominal_weight", json!(1.)),
            ("collection", json!("pieapp")),
            ("keys_sha256", json!("b".repeat(64))),
        ] {
            let old = f.declaration[field].clone();
            f.declaration[field] = value;
            assert!(
                f.call_named("subjective_vote_bapps", 1.).is_err(),
                "{field}"
            );
            f.declaration[field] = old;
        }
    }

    #[test]
    fn native_votes_bind_integer_counts_to_ordered_sides_targets_and_references() {
        let mut f = vote_fixture();
        let path = f.root.join("votes.json");
        let base = json!({"schema":"bapps-native-votes-v1", "comparisons":[
            {"source":"s", "n_a":1, "n_b":1, "n_tie":0, "n":2, "p_a":0.5}]});
        let mut check = |votes: &Value, targets: &[f64], refs: &[u32]| {
            std::fs::write(&path, serde_json::to_vec(votes).unwrap()).unwrap();
            f.declaration["native_votes_sha256"] = json!(hash(&path).unwrap());
            native_votes(&f.declaration, targets, Some(refs))
        };
        for na in 0..=2 {
            let mut v = base.clone();
            v["comparisons"][0]["n_a"] = json!(na);
            v["comparisons"][0]["n_b"] = json!(2 - na);
            v["comparisons"][0]["p_a"] = json!(f64::from(na) / 2.);
            check(&v, &[f64::from(na) / 2., 1. - f64::from(na) / 2.], &[0, 0]).unwrap();
        }
        assert!(check(&base, &[1., 0.], &[0, 0]).is_err());
        assert!(check(&base, &[0.5, 0.5], &[0, 1]).is_err());
        for (field, value) in [
            ("n_a", json!(0.5)),
            ("n_a", json!(true)),
            ("n_b", json!(0)),
            ("n_tie", json!(1)),
            ("n", json!(3)),
            ("p_a", json!(1.)),
            ("source", json!("different")),
        ] {
            let mut v = base.clone();
            v["comparisons"][0][field] = value;
            assert!(check(&v, &[0.5, 0.5], &[0, 0]).is_err(), "{field}");
        }
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

    const DIAGNOSTIC_PROTOCOL: &str = DIAGNOSTIC_PROTOCOLS[0].0;
    const FRONTIER_PROTOCOL: &str = DIAGNOSTIC_PROTOCOLS[1].0;
    /// (protocol, public_train weight, retained checkpoints) of a population.
    fn registered(population: &str) -> (&'static str, f64, &'static [u64]) {
        DIAGNOSTIC_PROTOCOLS
            .iter()
            .find_map(|(p, pops, ck)| {
                pops.iter()
                    .find(|(n, _)| *n == population)
                    .map(|(_, w)| (*p, *w, *ck))
            })
            .unwrap()
    }

    /// Version-2 base (one training leg, one validation-only leg) and its
    /// diagnostic derivation, written to disk with independent pins.
    fn diagnostic_fixture(root: &Path, population: &str) -> (Value, Value, PathBuf) {
        let training = json!({"name":"trained","path":root.join("t.parquet"),"train_weight":2.5,"val_weight":0.,
            "mode":"withinref,rank","subjective":true});
        let validation = json!({"name":"held_development","path":root.join("v.parquet"),"train_weight":0.,
            "val_weight":1.,"mode":"withinref,rank"});
        let base = json!({"schema":"subjective-training-authority-v2","state":"approved",
            "artifact_visibility":"private","selection":"fixed-final-epoch",
            "cells":{"full-s0":{"route":"full","width":1853,"groups":[training.clone(), validation.clone()]},
                     "full-s1":{"route":"full","width":1853,"groups":[]}}});
        let base_path = root.join("dataset-authority.json");
        std::fs::write(&base_path, serde_json::to_vec(&base).unwrap()).unwrap();
        let (retained, excluded) = if population == "aic-only" {
            (json!([validation]), json!([training]))
        } else {
            (base["cells"]["full-s0"]["groups"].clone(), json!([]))
        };
        let mut a = base.clone();
        a["schema"] = json!("subjective-training-authority-v4");
        a["base_authority"] = json!(base_path);
        a["base_authority_sha256"] = json!(hash(&base_path).unwrap());
        a["cells"]["full-s0"]["groups"] = retained;
        let (protocol, weight, checkpoints) = registered(population);
        a["cells"]["full-s0"]["continuation"] = json!({"checkpoint":"x","checkpoint_sha256":"y","epochs":60,
            "lr":0.00001,"lr_schedule":"constant","optimizer_state":"fresh","protocol":protocol,
            "population":population,"aic_weight":weight,"dump_checkpoints_every":1,
            "retained_checkpoint_epochs":checkpoints,"excluded_groups":excluded});
        (base, a, base_path)
    }
    fn aic(weight: f64) -> Vec<Group> {
        vec![(
            "aic_public_train".into(),
            PathBuf::from("aic.parquet"),
            weight,
            0.,
            false,
            GroupLossMode::Rank,
        )]
    }
    #[test]
    fn rehearsal_leg_pairs_within_each_registered_dataset() {
        use arrow::array::{ArrayRef, Int64Array, StringArray};
        use arrow::record_batch::RecordBatch;
        use parquet::arrow::ArrowWriter;
        use std::sync::Arc;
        let root = std::env::temp_dir().join(format!("rehearsal-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let table = root.join("human.parquet");
        let write = |members: &[&str]| {
            let n = members.len();
            let keys: Vec<String> = (0..n).map(|i| format!("k{i}")).collect();
            let batch = RecordBatch::try_from_iter([
                ("pair_key", Arc::new(StringArray::from(keys)) as ArrayRef),
                (
                    "source_row_id",
                    Arc::new(Int64Array::from((0..n as i64).collect::<Vec<_>>())) as ArrayRef,
                ),
                (
                    "ref_basename",
                    Arc::new(StringArray::from(vec!["r"; n])) as ArrayRef,
                ),
                (
                    "member_set",
                    Arc::new(StringArray::from(members.to_vec())) as ArrayRef,
                ),
            ])
            .unwrap();
            let file = std::fs::File::create(table.with_extension("keys.parquet")).unwrap();
            let mut writer = ArrowWriter::try_new(file, batch.schema(), None).unwrap();
            writer.write(&batch).unwrap();
            writer.close().unwrap();
        };
        write(&[
            "kadid_train",
            "tid2013",
            "kadid_select",
            "konfig_val",
            "cid22_a25",
            "konfig_train",
        ]);
        assert_eq!(dataset_ids(&table, 6).unwrap(), vec![0, 1, 0, 2, 3, 2]);
        assert!(dataset_ids(&table, 5).unwrap_err().contains("row counts"));
        write(&["kadid_train", "pipal"]);
        assert!(dataset_ids(&table, 2).unwrap_err().contains("member set"));
        std::fs::remove_dir_all(&root).unwrap();
        // The flag belongs to the one registered leg, and that leg pairs no other way.
        let parse = super::super::parse_group_spec;
        let (name, _, _, _, within, _) = parse("human_xref:t.parquet:32:0:withinset,rank").unwrap();
        assert!(name == WITHIN_SET_GROUP && within);
        for spec in [
            "human:t.parquet:32:0:withinset,rank",
            "human_xref:t.parquet:32:0:withinref,rank",
            "human_xref:t.parquet:32:0:withinset,withinref,rank",
            "human_xref:t.parquet:32:0:rank",
        ] {
            assert!(parse(spec).is_err(), "{spec}");
        }
    }

    #[test]
    fn rehearsal_protocol_adds_exactly_the_human_legs_dataset_pairing() {
        let root = std::env::temp_dir().join(format!("xref-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let human = json!({"name":"human","path":root.join("h.parquet"),"train_weight":32.37,"val_weight":0.,
            "mode":"withinref,rank","subjective":false,"manifest_sha256":"m","keys_path":root.join("h.keys.parquet")});
        let validation = json!({"name":"human_development","path":root.join("d.parquet"),"train_weight":0.,
            "val_weight":1.,"mode":"withinref,rank"});
        let base = json!({"schema":"subjective-training-authority-v2","state":"approved",
            "artifact_visibility":"private","selection":"fixed-final-epoch",
            "cells":{"full-s0":{"route":"full","width":1853,"groups":[human.clone(), validation.clone()]}}});
        let base_path = root.join("dataset-authority.json");
        std::fs::write(&base_path, serde_json::to_vec(&base).unwrap()).unwrap();
        let mut xref = human.clone();
        xref["name"] = json!(WITHIN_SET_GROUP);
        xref["mode"] = json!("withinset,rank");
        for (population, weight) in [("mixed-aic256-xref", 256.), ("mixed-aic1024-xref", 1024.)] {
            let mut a = base.clone();
            a["schema"] = json!("subjective-training-authority-v4");
            a["base_authority"] = json!(base_path);
            a["base_authority_sha256"] = json!(hash(&base_path).unwrap());
            a["cells"]["full-s0"]["groups"] = json!([human, validation, xref]);
            a["cells"]["full-s0"]["continuation"] = json!({"checkpoint":"x","checkpoint_sha256":"y","epochs":60,
                "lr":0.00001,"lr_schedule":"constant","optimizer_state":"fresh","protocol":XREF_PROTOCOL,
                "population":population,"aic_weight":weight,"dump_checkpoints_every":1,
                "retained_checkpoint_epochs":[7, 15, 29, 59],"excluded_groups":[]});
            assert!(!diagnostic_binding(&a, "full-s0", &aic(weight)).unwrap());
            let refuse = |mutate: &dyn Fn(&mut Value)| {
                let mut changed = a.clone();
                mutate(&mut changed);
                assert!(diagnostic_binding(&changed, "full-s0", &aic(weight)).is_err());
            };
            refuse(&|v| v["cells"]["full-s0"]["groups"] = json!([human, validation]));
            refuse(&|v| v["cells"]["full-s0"]["groups"][2]["mode"] = json!("withinref,rank"));
            refuse(&|v| v["cells"]["full-s0"]["groups"][2]["train_weight"] = json!(64.));
            refuse(&|v| {
                v["cells"]["full-s0"]["continuation"]["protocol"] = json!(FRONTIER_PROTOCOL)
            });
            refuse(&|v| {
                v["cells"]["full-s0"]["continuation"]["population"] = json!("mixed-aic256")
            });
        }
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn diagnostic_authority_binds_its_unmodified_base_and_group_partition() {
        let root = std::env::temp_dir().join(format!("diagnostic-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        for population in [
            "mixed-aic16",
            "aic-only",
            "mixed-aic64",
            "mixed-aic256",
            "mixed-aic1024",
        ] {
            let (_, a, base_path) = diagnostic_fixture(&root, population);
            let (protocol, weight, _) = registered(population);
            assert_eq!(
                diagnostic_binding(&a, "full-s0", &aic(weight)).unwrap(),
                population == "aic-only"
            );
            // The public_train leg must be present exactly once at its registered weight.
            assert!(diagnostic_binding(&a, "full-s0", &aic(8.)).is_err());
            assert!(diagnostic_binding(&a, "full-s0", &[]).is_err());
            let refuse = |mutate: &dyn Fn(&mut Value), needle: &str| {
                let mut changed = a.clone();
                mutate(&mut changed);
                let error = diagnostic_binding(&changed, "full-s0", &aic(weight)).unwrap_err();
                assert!(error.contains(needle), "{population}: {error}");
            };
            // A population is admitted only under its own protocol, weight and checkpoints.
            let other = if protocol == DIAGNOSTIC_PROTOCOL {
                FRONTIER_PROTOCOL
            } else {
                DIAGNOSTIC_PROTOCOL
            };
            refuse(
                &|v| v["cells"]["full-s0"]["continuation"]["protocol"] = json!(other),
                "protocol",
            );
            refuse(
                &|v| v["cells"]["full-s0"]["continuation"]["aic_weight"] = json!(weight * 2.),
                "protocol",
            );
            refuse(
                &|v| {
                    v["cells"]["full-s0"]["continuation"]["retained_checkpoint_epochs"] =
                        json!([7, 15, 29, 58])
                },
                "protocol",
            );
            refuse(
                &|v| v["cells"]["full-s0"]["continuation"]["epochs"] = json!(8),
                "protocol",
            );
            refuse(
                &|v| v["cells"]["full-s0"]["continuation"]["protocol"] = json!("other"),
                "protocol",
            );
            refuse(
                &|v| v["cells"]["full-s0"]["continuation"]["dump_checkpoints_every"] = json!(7),
                "protocol",
            );
            refuse(
                &|v| v["cells"]["full-s0"]["continuation"]["population"] = json!("aic"),
                "population",
            );
            refuse(
                &|v| {
                    v["cells"]["full-s0"]["continuation"]["excluded_groups"] = json!([{"name":"x"}])
                },
                "partition",
            );
            refuse(
                &|v| v["cells"]["full-s0"]["groups"] = json!([]),
                "partition",
            );
            refuse(
                &|v| v["cells"]["full-s0"]["width"] = json!(1825),
                "base binding",
            );
            refuse(
                &|v| v["cells"]["full-s1"]["route"] = json!("d1"),
                "another cell",
            );
            refuse(&|v| v["state"] = json!("draft"), "base field");
            refuse(
                &|v| v["base_authority_sha256"] = json!("0".repeat(64)),
                "pin",
            );
            // A base that is itself derived cannot anchor a diagnostic.
            let mut derived = a.clone();
            let mut v3 =
                serde_json::from_slice::<Value>(&std::fs::read(&base_path).unwrap()).unwrap();
            v3["schema"] = json!("subjective-training-authority-v3");
            let v3_path = root.join(format!("v3-{population}.json"));
            std::fs::write(&v3_path, serde_json::to_vec(&v3).unwrap()).unwrap();
            derived["base_authority"] = json!(v3_path);
            derived["base_authority_sha256"] = json!(hash(&v3_path).unwrap());
            assert!(
                diagnostic_binding(&derived, "full-s0", &aic(weight))
                    .unwrap_err()
                    .contains("version-2")
            );
        }
        std::fs::remove_dir_all(&root).unwrap();
    }
    #[test]
    fn continuation_budgets_stay_separate_and_checkpoints_require_a_binding() {
        use clap::Parser;
        let root = std::env::temp_dir().join(format!("diagnostic-run-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let checkpoint = root.join("initial.bin");
        std::fs::write(&checkpoint, b"invented checkpoint bytes").unwrap();
        let args = |epochs: &str, every: &str, init: bool| {
            let mut argv = vec![
                "zensim_mlp_train",
                "--group",
                "g:x.parquet:1:0",
                "--out",
                "o.bin",
                "--epochs",
                epochs,
                "--pairs-per-epoch",
                "50000",
                "--hidden",
                "128",
                "--max-features",
                "1853",
                "--init-seed",
                "1101",
                "--sample-seed",
                "101",
                "--nonneg-distance",
                "--lr",
                "0.00001",
                "--lr-schedule",
                "constant",
                "--dump-checkpoints-every",
                every,
                "--dump-checkpoints-dir",
                "ckpt",
            ];
            if init {
                argv.extend(["--init-from", checkpoint.to_str().unwrap()]);
            }
            super::super::Args::parse_from(argv)
        };
        let cell = |protocol: Option<&str>, epochs: u64| {
            let mut w = json!({"checkpoint":checkpoint,"checkpoint_sha256":hash(&checkpoint).unwrap(),
                "epochs":epochs,"lr":0.00001,"lr_schedule":"constant"});
            if let Some(p) = protocol {
                w["protocol"] = json!(p);
            }
            json!({"route":"full","seed_index":0,"epochs":120,"pairs_per_epoch":50000,"selected_epoch":119,
                "init_seed":1101,"sample_seed":101,"continuation":w})
        };
        // Frozen registration: 0/4/8 epochs only, and never under a protocol label.
        run_binding(&cell(None, 8), &args("8", "7", true), 1853, false).unwrap();
        assert!(run_binding(&cell(None, 60), &args("60", "1", true), 1853, false).is_err());
        assert!(
            run_binding(
                &cell(Some(DIAGNOSTIC_PROTOCOL), 8),
                &args("8", "7", true),
                1853,
                false
            )
            .is_err()
        );
        // Diagnostic: exactly 60 epochs with every-epoch dumps, only under version 4.
        run_binding(
            &cell(Some(DIAGNOSTIC_PROTOCOL), 60),
            &args("60", "1", true),
            1853,
            true,
        )
        .unwrap();
        assert!(
            run_binding(
                &cell(Some(DIAGNOSTIC_PROTOCOL), 60),
                &args("60", "59", true),
                1853,
                true
            )
            .is_err()
        );
        assert!(run_binding(&cell(None, 8), &args("8", "7", true), 1853, true).is_err());
        // The frontier shares the budget; an unregistered protocol label does not.
        run_binding(
            &cell(Some(FRONTIER_PROTOCOL), 60),
            &args("60", "1", true),
            1853,
            true,
        )
        .unwrap();
        assert!(
            run_binding(
                &cell(Some("aic-share-frontier-v2"), 60),
                &args("60", "1", true),
                1853,
                true
            )
            .is_err()
        );
        // A checkpoint without its binding no longer falls back to a full fit.
        let mut plain = cell(None, 8);
        plain.as_object_mut().unwrap().remove("continuation");
        run_binding(&plain, &args("120", "119", false), 1853, false).unwrap();
        assert!(
            run_binding(&plain, &args("120", "119", true), 1853, false)
                .unwrap_err()
                .contains("binding")
        );
        std::fs::remove_dir_all(&root).unwrap();
    }
}
