//! Structural inventory of the exact native cost workload representations.
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::path::Path;
use zenpredict::{Model, WeightStorage};
use zensim::corruption_head::CorruptionHead;

fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(
        args.len() == 3,
        "usage: shaping_model_inventory WORKLOAD NEW_OUTPUT"
    );
    let path = Path::new(&args[1]);
    let bytes = std::fs::read(path)?;
    let workload: Value = serde_json::from_slice(&bytes)?;
    ensure!(
        workload["schema"] == "shaping-cost-v1",
        "native cost workload required"
    );
    let mut models = Vec::new();
    let mut names = std::collections::BTreeSet::new();
    for arm in workload["arms"].as_array().context("model arms")? {
        let name = arm["id"].as_str().context("model ID")?;
        ensure!(names.insert(name), "duplicate model ID");
        let model_path = path
            .parent()
            .context("workload parent")?
            .join(arm["model"].as_str().context("model path")?);
        let data = std::fs::read(&model_path)?;
        let pin = digest(&data);
        ensure!(arm["sha256"] == pin, "model hash differs");
        let structure = match arm["kind"].as_str().context("model kind")? {
            "tree" => {
                let model =
                    CorruptionHead::from_bytes(&data).map_err(|e| anyhow::anyhow!("{e}"))?;
                json!({"trees":model.n_trees(),"nodes":model.n_nodes(),
                    "caller_input_width":model.caller_input_width(),
                    "declared_feature_ids":model.declared_feature_ids()})
            }
            "bake" | "shaped_additive" => {
                let model = Model::from_bytes(&data).map_err(|e| anyhow::anyhow!("{e}"))?;
                let mut layers = Vec::new();
                let mut coefficients = 0;
                for layer in model.layers() {
                    let weights = match layer.weights {
                        WeightStorage::F32(w) => w,
                        _ => anyhow::bail!("registered cost models require f32 weights"),
                    };
                    coefficients += weights.len() + layer.biases.len();
                    layers.push(json!({"in_dim":layer.in_dim,"out_dim":layer.out_dim,
                        "activation":format!("{:?}",layer.activation),
                        "stored_weights":weights.len(),"stored_biases":layer.biases.len(),
                        "nonzero_weights":weights.iter().filter(|&&v|v!=0.).count()}));
                }
                json!({"inputs":model.n_inputs(),"layers":layers,
                    "expanded_stored_coefficients":coefficients,
                    "independent_trainable_parameters":Value::Null,
                    "count_scope":"includes frozen routing and tied/expanded weights; training-owner parameter accounting required separately"})
            }
            _ => anyhow::bail!("unknown model kind"),
        };
        ensure!(
            digest(&std::fs::read(&model_path)?) == pin,
            "model changed during inventory"
        );
        models.push(
            json!({"id":name,"model_sha256":pin,"model_bytes":data.len(),"structure":structure}),
        );
    }
    ensure!(!models.is_empty(), "no model arms");
    ensure!(
        digest(&std::fs::read(path)?) == digest(&bytes),
        "workload changed"
    );
    let record = json!({"schema":"shaping-model-inventory-v1","workload_sha256":digest(&bytes),
        "binary_sha256":digest(&std::fs::read(std::env::current_exe()?)?),
        "models":models,"timing_measured":false,"training_parameter_accounting_complete":false});
    use std::io::Write;
    let mut output = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[2])?;
    output.write_all(&serde_json::to_vec_pretty(&record)?)?;
    Ok(())
}
