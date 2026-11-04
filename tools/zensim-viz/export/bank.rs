use crate::{Result, model::Input};
use arrow::array::{Array, Float32Array, Float64Array};
use parquet::arrow::{ProjectionMask, arrow_reader::ParquetRecordBatchReaderBuilder};
use serde::Serialize;
use std::{collections::BTreeSet, path::Path};

#[derive(Serialize)]
pub(crate) struct Bank {
    pub path: String,
    pub sha256: String,
    pub total_rows: usize,
    pub provenance: serde_json::Value,
    pub row_indices: Vec<usize>,
    pub rows: Vec<Vec<f64>>,
    pub human: Vec<Option<f64>>,
}
pub(crate) fn load(path: &Path, inputs: &[Input], max_rows: usize) -> Result<Bank> {
    // Refuse AIC before opening a table. Manifest provenance is checked too:
    // a neutral filename must not permit a prohibited source through.
    let lower = path.to_string_lossy().to_lowercase();
    if lower.contains("aic") {
        return Err("AIC tables are outside visualizer read scope".into());
    }
    let sidecar = path.with_extension("parquet.manifest.json");
    let manifest = if sidecar.exists() {
        sidecar
    } else {
        path.parent()
            .ok_or("bank has no parent")?
            .join("_MANIFEST.json")
    };
    if !manifest.exists() {
        return Err(
            "bank needs its producer manifest; refusing identity inference from width".into(),
        );
    }
    let bank_sha256;
    let human_allowed;
    let provenance;
    let mut producer_slots;
    {
        let text = std::fs::read_to_string(&manifest)?;
        if text.to_lowercase().contains("aic") {
            return Err("bank manifest mentions AIC; refusing before reading table".into());
        }
        let doc: serde_json::Value = serde_json::from_str(&text)?;
        bank_sha256 = crate::hash(path)?;
        let revision = doc
            .get("formula_revision")
            .or_else(|| doc["research"].get("formula_revision"))
            .ok_or("manifest has no formula revision")?;
        if revision != 7 && revision != "Rev7" {
            return Err("bank manifest is not revision 7".into());
        }
        let set = doc["feature_set_id"]
            .as_str()
            .or_else(|| doc["research"]["feature_set_id"].as_str())
            .ok_or("manifest lacks feature-set identity")?;
        let fsid = zensim::feature_set_id::FeatureSetId::parse(set)
            .ok_or("invalid feature-set identity")?;
        let request = zensim::research::Request::for_set(&fsid)?;
        producer_slots = Some(request.want().clone());
        if let Some(measured) = doc["measured_ids"].as_array() {
            let ids: Vec<usize> = measured
                .iter()
                .map(|v| {
                    v.as_u64()
                        .and_then(|n| usize::try_from(n).ok())
                        .filter(|&n| n < zensim::research::full_width())
                        .ok_or("manifest measured_ids contains an invalid slot")
                })
                .collect::<std::result::Result<_, _>>()?;
            producer_slots = Some(zensim::feature_set_id::SlotSet::from_slots(ids));
        }
        provenance = serde_json::json!({"manifest_sha256":crate::hash(&manifest)?,"feature_set_id":set,"formula_revision":revision,"hdr_front_end":doc["hdr_front_end"],"data_role":doc["data_role"]});
        let expected = doc["table_sha256"]
            .as_str()
            .or_else(|| doc["features_parquet_sha256"].as_str())
            .ok_or("producer manifest has no table SHA-256")?;
        if expected != bank_sha256 {
            return Err("bank bytes differ from manifest hash".into());
        }
        let role = doc["data_role"].as_str().unwrap_or("").to_lowercase();
        human_allowed = role.contains("human") && !role.contains("oracle");
    }
    let ids: BTreeSet<_> = inputs
        .iter()
        .flat_map(|i| [Some(i.feature), i.factor])
        .flatten()
        .collect();
    let slots = producer_slots.ok_or("manifest did not resolve producer slots")?;
    for &id in &ids {
        if !slots.contains(id) {
            return Err(format!("producer does not populate consumed f{id}").into());
        }
    }
    let width = ids.last().ok_or("no inputs")? + 1;
    let builder = ParquetRecordBatchReaderBuilder::try_new(std::fs::File::open(path)?)?;
    let total_rows = usize::try_from(builder.metadata().file_metadata().num_rows())?;
    if total_rows == 0 {
        return Err("bank has no observed rows".into());
    }
    let schema = builder.schema().clone();
    let mut selected = Vec::new();
    for id in &ids {
        selected.push(schema.index_of(&format!("f{id}"))?);
    }
    // CID22 oracle banks have peer labels, not human responses. Only a column
    // explicitly named human_score is shown as human evidence.
    if human_allowed && let Ok(i) = schema.index_of("human_score") {
        selected.push(i);
    }
    let mask = ProjectionMask::roots(builder.parquet_schema(), selected);
    let n = max_rows.min(total_rows);
    let wanted: BTreeSet<_> = (0..n)
        .map(|i| {
            if n == 1 {
                0
            } else {
                i * (total_rows - 1) / (n - 1)
            }
        })
        .collect();
    let mut bank = Bank {
        path: path.to_string_lossy().into(),
        sha256: bank_sha256,
        total_rows,
        provenance,
        row_indices: Vec::new(),
        rows: Vec::new(),
        human: Vec::new(),
    };
    let mut offset = 0;
    for batch in builder
        .with_projection(mask)
        .with_batch_size(2048)
        .build()?
    {
        let batch = batch?;
        for r in 0..batch.num_rows() {
            if !wanted.contains(&(offset + r)) {
                continue;
            }
            let mut row = vec![f64::NAN; width];
            for id in &ids {
                let column = batch
                    .column_by_name(&format!("f{id}"))
                    .ok_or("projected column missing")?;
                row[*id] = number(column.as_ref(), r)?;
                if !row[*id].is_finite() {
                    return Err(format!("nonfinite f{id} at row {}", offset + r).into());
                }
            }
            bank.human.push(
                batch
                    .column_by_name("human_score")
                    .filter(|a| !a.is_null(r))
                    .map(|a| number(a.as_ref(), r))
                    .transpose()?,
            );
            bank.rows.push(row);
            bank.row_indices.push(offset + r);
        }
        offset += batch.num_rows();
    }
    if bank.rows.len() != n {
        return Err("bank sample coverage differs from footer".into());
    }
    Ok(bank)
}
fn number(a: &dyn Array, r: usize) -> Result<f64> {
    if a.is_null(r) {
        return Err("null bank value".into());
    }
    if let Some(a) = a.as_any().downcast_ref::<Float32Array>() {
        Ok(f64::from(a.value(r)))
    } else if let Some(a) = a.as_any().downcast_ref::<Float64Array>() {
        Ok(a.value(r))
    } else {
        Err(format!("expected float bank column, got {:?}", a.data_type()).into())
    }
}

#[cfg(test)]
mod admission_tests {
    use super::*;
    #[test]
    fn prohibited_path_is_rejected_before_io() {
        let error = load(Path::new("AIC-2026-must-never-be-opened.parquet"), &[], 200)
            .err()
            .expect("must refuse");
        assert_eq!(
            error.to_string(),
            "AIC tables are outside visualizer read scope"
        );
    }
    #[test]
    fn prohibited_manifest_is_rejected_before_table_hashing() {
        let scratch = std::env::var_os("TMPDIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                std::path::PathBuf::from(std::env::var_os("HOME").expect("HOME required"))
                    .join("tmp/zensim-viz")
            });
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory = scratch.join(format!("admission-{}-{stamp}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let table = directory.join("neutral.parquet");
        // Deliberately no table exists: an attempted hash would fail differently.
        std::fs::write(
            table.with_extension("parquet.manifest.json"),
            r#"{"source":"AIC-2026"}"#,
        )
        .unwrap();
        let error = load(&table, &[], 200).err().expect("must refuse");
        assert_eq!(
            error.to_string(),
            "bank manifest mentions AIC; refusing before reading table"
        );
    }
}
