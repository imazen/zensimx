//! MlicOrder admission. Keys and model provenance precede prediction payloads.
use super::{Args, group_admission, training_keys};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use zensim_validate::{mlp_train::GroupLossMode, train_manifest};

pub(super) const GROUP: &str = "human_codec_strata";
const ROOT_SHA: &str = "a7fd4af0c2bb5a8981be636e5c1cb1a51db57496b7c60d7bf6da01d71dd16952";
const FOLDS: &[&str] = &["tid2013", "konfig", "cid22_a25", "pipal", "pairs", "mos7"];
type Group = (String, PathBuf, f64, f64, bool, GroupLossMode);

fn read(path: &Path, expected: &str) -> Result<Value, String> {
    group_admission::safe(path)?;
    if train_manifest::sha256_file(path).map_err(|e| e.to_string())? != expected {
        return Err(format!("matched-codec pin changed: {}", path.display()));
    }
    serde_json::from_slice(&std::fs::read(path).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())
}

fn string<'a>(v: &'a Value, key: &str) -> Result<&'a str, String> {
    v[key]
        .as_str()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| format!("matched-codec {key} required"))
}

#[derive(Clone)]
struct Row {
    index: usize,
    key: String,
    reference: String,
    jpeg: bool,
}

pub(super) struct Admitted {
    manifest: Value,
    manifest_sha: String,
    refs: Vec<String>,
    rows: Vec<Row>,
}

pub(super) fn admit(args: &Args, groups: &[Group]) -> Result<Option<Admitted>, String> {
    let matched: Vec<_> = groups.iter().filter(|g| g.0 == GROUP).collect();
    let Some(path) = &args.matched_codec_manifest else {
        if !matched.is_empty() || args.matched_codec_sha256.is_some() {
            return Err("matched-codec group requires its pinned manifest".into());
        }
        return Ok(None);
    };
    if matched.len() != 1
        || args.historical_replay.is_some()
        || args.hdr_consensus_research
        || !args.rank_pair_list.is_empty()
        || args.n_hidden_layers != 1
        || args.pool_head
        || args.hybrid_head
        || args.per_sample_alpha_head
        || args.minibatch_size != 1
        || args.pair_sampling != "uniform"
        || !args.no_sample_coverage
        || args.epochs != 120
        || args.pairs_per_epoch != 50000
        || args.subjective_authority.is_none()
    {
        return Err("matched codecs require the registered strict plain CPU cell".into());
    }
    let g = matched[0];
    if g.2 <= 0. || g.3 != 0. || !g.4 || g.5 != GroupLossMode::Rank {
        return Err("matched-codec group must be fit-only within-reference rank".into());
    }
    let pin = args
        .matched_codec_sha256
        .as_deref()
        .ok_or("matched-codec manifest pin required")?;
    let d = read(path, pin)?;
    let fold = string(&d, "fold")?;
    let seed = d["seed_index"]
        .as_u64()
        .ok_or("matched-codec seed index required")?;
    if d["schema"] != "mlic-matched-codecs-v1"
        || d["role"] != "TRAIN"
        || d["candidate"] != "MlicOrder"
        || !FOLDS.contains(&fold)
        || seed > 2
        || d["policy"]
            != "bidirectional-nearest-frozen-Mlic;quarter-accepted-reference;affine-alternate-cycles"
        || d["original_root_sha256"] != ROOT_SHA
        || d["init_seed"].as_u64() != Some(args.init_seed.unwrap_or(args.seed))
        || d["sample_seed"].as_u64() != Some(args.sample_seed.unwrap_or(args.seed))
    {
        return Err("unregistered matched-codec population, policy or seed".into());
    }
    let root = Path::new(string(&d, "original_root")?);
    let frozen = read(&root.join("wide/frozen.json"), ROOT_SHA)?;
    let receipt = read(
        &root.join("wide/main/real/receipt.json"),
        frozen["wide_receipts"]["main/real"]
            .as_str()
            .ok_or("original receipt pin required")?,
    )?;
    let leg = if ["tid2013", "konfig", "cid22_a25"].contains(&fold) {
        format!("human_without_{fold}")
    } else {
        "human_all".into()
    };
    let fit = &receipt["legs"][&leg]["fit"];
    if d["table_sha256"] != fit["sha256"]
        || d["keys_sha256"] != fit["keys_sha256"]
        || g.1.canonicalize().map_err(|e| e.to_string())?
            != root
                .join(string(fit, "rel")?)
                .canonicalize()
                .map_err(|e| e.to_string())?
    {
        return Err("matched-codec fitting population differs from original Mlic".into());
    }
    let table_manifest: Value = serde_json::from_slice(
        &std::fs::read(format!("{}.manifest.json", g.1.display())).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    if table_manifest["table_sha256"] != fit["sha256"] {
        return Err("matched-codec table declaration changed".into());
    }
    // Bind the exact original cell, rather than accepting a similarly named model.
    let old = root
        .parent()
        .and_then(Path::parent)
        .ok_or("original study root required")?;
    let cell = old.join(format!("cells/Mlic/without_{fold}_s{seed}"));
    let record = read(
        &cell.join("byc_record.json"),
        string(&d, "original_cell_record_sha256")?,
    )?;
    if record["fold"] != fold
        || record["seed"] != seed
        || record["last_bin_sha256"] != d["matching_model_sha256"]
        || Path::new(string(&d, "matching_model")?) != cell.join("refit/last.bin")
    {
        return Err("matched-codec model is not the corresponding frozen Mlic cell".into());
    }
    let authority = read(
        &cell.join("dataset-authority.json"),
        string(&d, "original_authority_sha256")?,
    )?;
    let cells = authority["cells"]
        .as_object()
        .ok_or("original cell authority required")?;
    if cells.len() != 1
        || cells
            .values()
            .any(|c| c["init_seed"] != d["init_seed"] || c["sample_seed"] != d["sample_seed"])
    {
        return Err("matched-codec original cell seeds changed".into());
    }
    let keys = training_keys::read(
        &g.1.with_extension("keys.parquet"),
        &["pair_key", "ref_basename", "member_set"],
        &["pair_key", "source_row_id", "ref_basename", "member_set"],
    )?;
    if keys.file_digest != string(&d, "keys_sha256")? {
        return Err("matched-codec fitting keys changed".into());
    }
    let (key, reference, member) = (
        keys.column("pair_key")?,
        keys.column("ref_basename")?,
        keys.column("member_set")?,
    );
    let banks = d["codec_banks"]
        .as_object()
        .ok_or("matched-codec bank identity required")?;
    if banks.keys().map(String::as_str).collect::<BTreeSet<_>>()
        != BTreeSet::from(["kadid_train", "kadid_select"])
    {
        return Err("matched-codec bank members differ".into());
    }
    let mut codec = BTreeMap::new();
    for (member, bank) in banks {
        let path = Path::new(string(bank, "keys_path")?);
        group_admission::safe(path)?;
        let keys = training_keys::read(
            path,
            &["pair_key", "ref_group", "codec"],
            &[
                "pair_key",
                "row_id",
                "ref_pixels_sha256",
                "dist_pixels_sha256",
                "ref_group",
                "codec",
                "knob",
                "width",
                "height",
                "pixels_identical",
                "ref_path",
                "dist_path",
                "n_stimuli",
            ],
        )?;
        if keys.file_digest != string(bank, "keys_sha256")? {
            return Err("codec bank keys changed".into());
        }
        let (k, r, c) = (
            keys.column("pair_key")?,
            keys.column("ref_group")?,
            keys.column("codec")?,
        );
        for row in keys.rows {
            let identity = (row[r].clone(), row[c].clone());
            if codec
                .insert((member.clone(), row[k].clone()), identity.clone())
                .is_some_and(|old| old != identity)
            {
                return Err("ambiguous promoted codec identity".into());
            }
        }
    }
    let mut rows = Vec::new();
    let mut refs = Vec::new();
    let mut seen = BTreeSet::new();
    for (index, row) in keys.rows.into_iter().enumerate() {
        if !seen.insert(row[key].clone()) {
            return Err("duplicate fitting pair key".into());
        }
        refs.push(row[reference].clone());
        if !row[member].starts_with("kadid_") {
            continue;
        }
        let (source, codec) = codec
            .get(&(row[member].clone(), row[key].clone()))
            .ok_or("KADID key has no promoted codec")?;
        if source != &row[reference] {
            return Err("promoted reference differs from fitting reference".into());
        }
        if ["kadid_09", "kadid_10"].contains(&codec.as_str()) {
            rows.push(Row {
                index,
                key: row[key].clone(),
                reference: source.clone(),
                jpeg: codec == "kadid_10",
            });
        }
    }
    if rows.is_empty() {
        return Err("matched-codec cell has no eligible rows".into());
    }
    group_admission::safe(Path::new(string(&d, "predictions")?))?;
    Ok(Some(Admitted {
        manifest: d,
        manifest_sha: pin.into(),
        refs,
        rows,
    }))
}

fn nearest_pairs(rows: &[Row], predictions: &[f64]) -> Result<Vec<[usize; 2]>, String> {
    let mut references = BTreeMap::<&str, Vec<&Row>>::new();
    for row in rows {
        references.entry(&row.reference).or_default().push(row);
    }
    let mut pairs = BTreeMap::new();
    for rows in references.values() {
        for row in rows {
            let other = rows
                .iter()
                .filter(|p| p.jpeg != row.jpeg)
                .min_by(|a, b| {
                    (predictions[row.index] - predictions[a.index])
                        .abs()
                        .total_cmp(&(predictions[row.index] - predictions[b.index]).abs())
                        .then_with(|| a.key.cmp(&b.key))
                })
                .ok_or("reference lacks one registered codec")?;
            let (a, b) = if row.jpeg {
                (*row, *other)
            } else {
                (*other, *row)
            };
            pairs.insert((&a.key, &b.key), [a.index, b.index]);
        }
    }
    Ok(pairs.into_values().collect())
}

impl Admitted {
    pub(super) fn payload(&self) -> Result<(Vec<[usize; 2]>, Value), String> {
        let d = &self.manifest;
        for (path, pin) in [
            ("matching_model", "matching_model_sha256"),
            ("predictions", "predictions_sha256"),
        ] {
            if train_manifest::sha256_file(Path::new(string(d, path)?))
                .map_err(|e| e.to_string())?
                != string(d, pin)?
            {
                return Err(format!("matched-codec {path} changed"));
            }
        }
        let mut reader = csv::ReaderBuilder::new()
            .delimiter(b'\t')
            .from_path(string(d, "predictions")?)
            .map_err(|e| e.to_string())?;
        if reader
            .headers()
            .map_err(|e| e.to_string())?
            .iter()
            .collect::<Vec<_>>()
            != ["row_idx", "pred"]
        {
            return Err("matched-codec prediction columns differ".into());
        }
        let mut predictions = Vec::new();
        for row in reader.records() {
            let row = row.map_err(|e| e.to_string())?;
            if row[0].parse::<usize>().map_err(|e| e.to_string())? != predictions.len() {
                return Err("matched-codec prediction order changed".into());
            }
            let p: f64 = row[1].parse().map_err(|_| "invalid matching prediction")?;
            if !p.is_finite() {
                return Err("nonfinite matching prediction".into());
            }
            predictions.push(p);
        }
        if predictions.len() != self.refs.len() {
            return Err("matching prediction population differs".into());
        }
        let pairs = nearest_pairs(&self.rows, &predictions)?;
        let report = json!({"manifest_sha256":self.manifest_sha, "manifest":d,
            "pairs":pairs.iter().map(|&[a,b]| json!({"a":a,"b":b,"gap":(predictions[a]-predictions[b]).abs()})).collect::<Vec<_>>()});
        Ok((pairs, report))
    }

    pub(super) fn validate_loaded(&self, refs: Option<&[u32]>) -> Result<(), String> {
        let refs = refs.ok_or("loaded matched-codec references missing")?;
        if refs.len() != self.refs.len() {
            return Err("loaded matched-codec row count changed".into());
        }
        let mut forward = BTreeMap::new();
        let mut reverse = BTreeMap::new();
        for (name, id) in self.refs.iter().zip(refs) {
            if forward.insert(name, id).is_some_and(|old| old != id)
                || reverse.insert(id, name).is_some_and(|old| old != name)
            {
                return Err("loaded matched-codec reference identity changed".into());
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn private_group_requires_manifest_before_any_table_read() {
        let args = <Args as clap::Parser>::parse_from([
            "trainer",
            "--group",
            "human_codec_strata:absent.parquet:1:0:withinref,rank",
            "--out",
            "absent.bin",
        ]);
        let group = (
            GROUP.into(),
            PathBuf::from("absent.parquet"),
            1.,
            0.,
            true,
            GroupLossMode::Rank,
        );
        assert!(
            admit(&args, &[group])
                .err()
                .unwrap()
                .contains("pinned manifest")
        );
        assert!(admit(&args, &[]).unwrap().is_none());
    }
    #[test]
    fn bidirectional_union_retains_every_row_and_breaks_ties_by_key() {
        let rows: Vec<_> = [("a", true), ("b", true), ("d", false), ("c", false)]
            .into_iter()
            .enumerate()
            .map(|(index, (key, jpeg))| Row {
                index,
                key: key.into(),
                reference: "r".into(),
                jpeg,
            })
            .collect();
        let pairs = nearest_pairs(&rows, &[0., 2., 1., 1.]).unwrap();
        assert_eq!(pairs, vec![[0, 3], [0, 2], [1, 3]]);
        let reversed: Vec<_> = rows.iter().rev().cloned().collect();
        assert_eq!(nearest_pairs(&reversed, &[0., 2., 1., 1.]).unwrap(), pairs);
        assert!(nearest_pairs(&rows[..2], &[0., 2.]).is_err());
    }
}
