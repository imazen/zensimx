//! Per-cell witnesses for the canonical monotonicity certificate. Overall
//! certification is delegated to mlp_train::certify_monotone_bake through its
//! existing native CLI. This module locates max-pair/gate witnesses for display;
//! it does not replace that certificate or infer it from a model's name.
use crate::{
    Result,
    model::{self, Input},
};
use serde::Serialize;
use std::{collections::BTreeSet, path::Path, process::Command};
use zenpredict::Model;

#[derive(Serialize)]
pub(crate) struct Constraints {
    pub certificate: serde_json::Value,
    pub certificate_binary_sha256: String,
    pub max_pairs: Vec<(usize, usize)>,
    pub gated_weights: Vec<(usize, usize)>,
    #[serde(skip)]
    pub exempt: BTreeSet<(usize, usize)>,
}
pub(crate) fn inspect(model: &Model, inputs: &[Input], bake: &Path) -> Result<Constraints> {
    let tool = std::env::var("VIZ_CERTIFIER")
        .map_err(|_| "set VIZ_CERTIFIER to the canonical bake_dial_refit binary")?;
    let output = Command::new(&tool)
        .args(["certify-monotone", "--bake"])
        .arg(bake)
        .output()?;
    let cert: serde_json::Value = serde_json::from_slice(&output.stdout).map_err(|e| {
        format!(
            "certificate command returned no JSON: {e}; {}",
            String::from_utf8_lossy(&output.stderr)
        )
    })?;
    if cert["certified"] == true
        && (cert["ramps"].as_u64() != Some(0) || cert["free_ramps"].as_u64() != Some(0))
    {
        return Err("per-cell witness view currently supports max pairs and fragility gates; ramp view is not implemented".into());
    }
    if model.n_layers() != 2 {
        return Err("witness view requires a one-hidden-layer network".into());
    }
    let l = model.layer(0);
    let w = model::weights(model, 0);
    let out = model::weights(model, 1);
    let scales = model.scaler_scale();
    let mut exempt = BTreeSet::new();
    let mut gates = Vec::new();
    // A nonlinear input transform invalidates the linear gated-weight slope
    // exemption. Mirror the canonical certificate; max-pair witnesses remain.
    for p in inputs.iter().filter(|i| i.factor.is_some() && !model.has_nontrivial_feature_transforms()) {
        let Some(d) = inputs
            .iter()
            .find(|i| i.factor.is_none() && i.feature == p.feature)
        else {
            return Err("product lacks direct witness".into());
        };
        for j in 0..l.out_dim {
            let a = f64::from(w[d.index * l.out_dim + j]) / f64::from(scales[d.index]);
            let b = f64::from(w[p.index * l.out_dim + j]) / f64::from(scales[p.index]);
            if b < 0.0 && a >= 0.0 && a + b >= 0.0 {
                exempt.insert((p.index, j));
                gates.push((p.index, j));
            }
        }
    }
    let signed =
        |j: usize| (0..l.in_dim).any(|i| w[i * l.out_dim + j] < 0.0 && !exempt.contains(&(i, j)));
    let mut used = BTreeSet::new();
    let mut pairs = Vec::new();
    for v in 0..l.out_dim {
        if used.contains(&v) || !signed(v) || out[v] == 0.0 {
            continue;
        }
        let u = (0..l.out_dim).find(|&u| {
            !used.contains(&u)
                && u != v
                && l.biases[u] == 0.0
                && out[u].to_bits() == out[v].to_bits()
                && !signed(u)
                && (0..l.in_dim).all(|i| {
                    let a = f64::from(w[i * l.out_dim + u]);
                    let b = f64::from(w[i * l.out_dim + v]);
                    a >= 0.0 && a + b >= 0.0
                })
        });
        if let Some(u) = u {
            used.insert(u);
            used.insert(v);
            pairs.push((u, v));
        }
    }
    if cert["certified"] == true && Some(pairs.len() as u64) != cert["max_pairs"].as_u64() {
        return Err("located max-pair witnesses differ from canonical certificate".into());
    }
    for &(_, v) in &pairs {
        for i in 0..l.in_dim {
            exempt.insert((i, v));
        }
    }
    // A structurally zero output makes its encoder unreachable, which the
    // canonical certificate permits. Label it separately from a live pair.
    for (j, &weight) in out.iter().enumerate() {
        if weight == 0.0 {
            for i in 0..l.in_dim {
                exempt.insert((i, j));
            }
        }
    }
    Ok(Constraints {
        certificate: cert,
        certificate_binary_sha256: crate::hash(Path::new(&tool))?,
        max_pairs: pairs,
        gated_weights: gates,
        exempt,
    })
}

#[cfg(all(test, feature = "real-bake-tests"))]
mod tests {
    use super::*;
    fn fixture(second: f32) -> (Model, std::path::PathBuf) {
        let weights = [2.0, second, 1.0, 1.0];
        let hidden_biases = [0.0, -1.0];
        let out_weights = [-1.0, -1.0];
        let out_bias = [100.0];
        let layers = [
            zenpredict_bake::BakeLayer {
                in_dim: 2,
                out_dim: 2,
                activation: zenpredict::Activation::Relu,
                dtype: zenpredict::WeightDtype::F32,
                weights: &weights,
                biases: &hidden_biases,
            },
            zenpredict_bake::BakeLayer {
                in_dim: 2,
                out_dim: 1,
                activation: zenpredict::Activation::Identity,
                dtype: zenpredict::WeightDtype::F32,
                weights: &out_weights,
                biases: &out_bias,
            },
        ];
        let bytes = zenpredict_bake::BakeRequest::builder(0, 0, &[0.0; 2], &[1.0; 2], &layers)
            .bake()
            .unwrap();
        let root = std::path::PathBuf::from(
            std::env::var("TMPDIR").expect("caller must set disk-backed TMPDIR"),
        );
        let path = root.join(format!(
            "viz-constraint-{}-{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&path, &bytes).unwrap();
        (Model::from_bytes(&bytes).unwrap(), path)
    }
    #[test]
    fn a_valid_negative_max_pair_is_distinguished_from_a_real_sign_failure() {
        let inputs: Vec<_> = (0..2)
            .map(|index| Input {
                index,
                feature: index,
                factor: None,
                name: format!("f{index}"),
                family: "fixture".into(),
                signal: "fixture".into(),
                analysis_family: "fixture × fixture".into(),
                short_name: format!("f{index}"),
                scale: 0,
                channel: "y".into(),
                pooling: "mean".into(),
                form: "difference".into(),
            })
            .collect();
        let (m, path) = fixture(-1.0);
        let c = inspect(&m, &inputs, &path).unwrap();
        assert_eq!(c.certificate["certified"], true);
        assert_eq!(c.max_pairs.len(), 1);
        let (u, v) = c.max_pairs[0];
        assert_eq!(m.layer(0).biases[u], 0.0);
        assert!(m.layer(0).biases[v] < 0.0);
        assert!(c.exempt.contains(&(0, v)));
        let (bad, path) = fixture(-3.0);
        let c = inspect(&bad, &inputs, &path).unwrap();
        assert_eq!(c.certificate["certified"], false);
        let bad_weights = model::weights(&bad, 0);
        let bad_unit = bad_weights[..2].iter().position(|v| *v < 0.0).unwrap();
        assert!(!c.exempt.contains(&(0, bad_unit)));
        let bank = crate::bank::Bank {
            path: "invented fixture".into(),
            sha256: "invented fixture".into(),
            total_rows: 200,
            provenance: serde_json::json!({"fixture":true}),
            row_indices: (0..200).collect(),
            rows: (0..200)
                .map(|i| vec![i as f64 / 100.0, (i % 17) as f64 / 17.0])
                .collect(),
            human: vec![None; 200],
        };
        let report =
            crate::audit::run(&bad, &inputs, &bank, None, "invented fixture".into(), &c).unwrap();
        assert!(
            report
                .flags
                .iter()
                .any(|f| f.kind == "sign" && f.from == Some(0) && f.to == Some(bad_unit))
        );
        assert!(report.flags.iter().any(|f| f.kind == "certificate"));
    }
}
