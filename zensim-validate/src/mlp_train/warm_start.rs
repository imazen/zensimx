//! Private checkpoint reader. The CLI and trainer retain separate admission
//! responsibilities: this reader validates the network being initialized.
use super::{MlpHyperparams, hidden_activation_for};
use zenpredict::{Activation, Model, WeightStorage};

pub(super) struct InitialWeights {
    pub(super) scaler_mean: Vec<f64>,
    pub(super) scaler_scale: Vec<f64>,
    pub(super) w1: Vec<f64>,
    pub(super) b1: Vec<f64>,
    pub(super) w2: Vec<f64>,
    pub(super) b2: Vec<f64>,
}

fn flag<'a>(argv: &'a [String], name: &str) -> Result<Option<&'a str>, String> {
    let positions: Vec<_> = argv
        .iter()
        .enumerate()
        .filter(|(_, s)| s.as_str() == name)
        .collect();
    if positions.len() > 1 {
        return Err(format!("duplicate checkpoint option {name}"));
    }
    Ok(positions
        .first()
        .and_then(|(i, _)| argv.get(i + 1).map(String::as_str)))
}

impl InitialWeights {
    pub(super) fn load(
        bytes: &[u8],
        n_features: usize,
        hp: &MlpHyperparams,
    ) -> Result<Self, String> {
        let model = Model::from_bytes(bytes).map_err(|e| format!("initial checkpoint: {e}"))?;
        if model.layers().count() != 2
            || hp.n_hidden_layers != 1
            || hp.pool_head
            || hp.hybrid_head
            || hp.per_sample_alpha_head
            || hp.skip_connection
            || hp.monotone_ramps > 0
            || hp.free_ramps > 0
            || !hp.monotone_product_pairs.is_empty()
            || hp.smooth_hinge
            || hp.monotone_feature_pin.is_some()
            || hp.steer_pairs.is_some()
        {
            return Err("initial checkpoint requires the plain two-layer head without auxiliary parameterizations".into());
        }
        let hidden = if hp.nonneg_distance {
            Activation::Relu
        } else {
            hidden_activation_for(hp.leaky_alpha)
        };
        let l0 = model.layer(0);
        let l1 = model.layer(1);
        if l0.out_dim != hp.n_hidden
            || l1.in_dim != hp.n_hidden
            || l1.out_dim != 1
            || l0.activation != hidden
            || l1.activation != Activation::Identity
        {
            return Err("initial checkpoint head, hidden size or activation differs".into());
        }
        if model.feature_transforms().is_some()
            || hp.feature_transforms.as_ref().is_some_and(|t| {
                t.iter()
                    .any(|&t| t != zenpredict::FeatureTransform::Identity)
            })
            || hp
                .feature_transform_params
                .as_ref()
                .is_some_and(|p| p.iter().any(|p| !p.is_empty()))
        {
            return Err("initial checkpoint feature transforms are unsupported".into());
        }
        let metadata = model.metadata();
        let repro: serde_json::Value = serde_json::from_str(
            metadata
                .get_utf8("zentrain.repro")
                .map_err(|_| "initial checkpoint needs embedded training reproduction metadata")?,
        )
        .map_err(|e| format!("initial checkpoint reproduction metadata: {e}"))?;
        let argv: Vec<String> = serde_json::from_value(repro["argv"].clone())
            .map_err(|_| "initial checkpoint needs its original training options")?;
        for (name, expected) in [
            ("--nonneg-distance", hp.nonneg_distance),
            ("--nonneg-thresholds", hp.nonneg_thresholds),
            ("--monotone-cbc", hp.monotone_cbc),
            ("--smooth-hinge", hp.smooth_hinge),
            ("--skip-connection", hp.skip_connection),
            ("--pool-head", hp.pool_head),
            ("--hybrid-head", hp.hybrid_head),
            ("--per-sample-alpha-head", hp.per_sample_alpha_head),
        ] {
            if argv.iter().any(|s| s == name) != expected {
                return Err(format!(
                    "initial checkpoint parameterization differs: {name}"
                ));
            }
        }
        for (name, default, expected) in [
            ("--hidden", 128, hp.n_hidden),
            ("--max-pairs", 0, hp.max_pairs),
            ("--monotone-ramps", 0, hp.monotone_ramps),
            ("--free-ramps", 0, hp.free_ramps),
            ("--n-hidden-layers", 1, hp.n_hidden_layers),
        ] {
            let actual = flag(&argv, name)?
                .map(|v| v.parse::<usize>())
                .transpose()
                .map_err(|_| format!("initial checkpoint malformed {name}"))?
                .unwrap_or(default);
            if actual != expected {
                return Err(format!(
                    "initial checkpoint parameterization differs: {name}"
                ));
            }
        }
        let old_pin = flag(&argv, "--nonneg-pin")?
            .map(str::parse::<f64>)
            .transpose()
            .map_err(|_| "initial checkpoint malformed identity pin")?
            .unwrap_or(100.0);
        if hp.nonneg_distance && old_pin != hp.nonneg_pin {
            return Err("initial checkpoint identity pin differs".into());
        }
        let columns = if let Some(path) = &hp.steer_derived_inputs {
            let (decl, sha) = crate::derived_inputs::Fx1Declaration::load(path)?;
            if metadata.get_utf8("zensim.derived_inputs").ok()
                != Some(decl.metadata_text().as_str())
                || metadata.get_utf8("zentrain.derived_inputs_sha256").ok() != Some(sha.as_str())
                || zensim::declared_feature_ids(&model) != Some(decl.read_set())
            {
                return Err("initial checkpoint derived inputs or read set differs".into());
            }
            let base = n_features
                .checked_sub(decl.products.len())
                .ok_or("initial checkpoint feature width differs")?;
            decl.kept_columns(base)
        } else {
            if metadata.get("zensim.derived_inputs").is_some() || model.n_inputs() != n_features {
                return Err("initial checkpoint feature layout differs".into());
            }
            if let Some(mask) = super::input_keep_mask() {
                if let Some(ids) = zensim::declared_feature_ids(&model) {
                    let expected: Vec<_> = mask
                        .iter()
                        .enumerate()
                        .filter_map(|(i, &k)| k.then_some(i))
                        .collect();
                    if ids != expected {
                        return Err("initial checkpoint read set differs".into());
                    }
                }
            }
            (0..n_features).collect()
        };
        if columns.len() != model.n_inputs() || columns.iter().any(|&c| c >= n_features) {
            return Err("initial checkpoint model-input layout differs".into());
        }
        let weights = |layer: usize| -> Result<Vec<f64>, String> {
            match &model.layer(layer).weights {
                WeightStorage::F32(w) if w.iter().all(|x| x.is_finite()) => {
                    Ok(w.iter().map(|&x| f64::from(x)).collect())
                }
                _ => Err("initial checkpoint must have finite unpacked f32 weights".into()),
            }
        };
        if model.scaler_mean().iter().any(|x| !x.is_finite())
            || model
                .scaler_scale()
                .iter()
                .any(|&x| !x.is_finite() || x <= 0.0)
            || l0.biases.iter().chain(l1.biases).any(|x| !x.is_finite())
        {
            return Err("initial checkpoint has invalid scaling or biases".into());
        }
        if hp.nonneg_distance && model.scaler_mean().iter().any(|&x| x != 0.0) {
            return Err("initial checkpoint violates scale-only standardization".into());
        }
        if hp.monotone_cbc && hp.nonneg_distance {
            super::certify_monotone_bake(&model, hp.nonneg_pin as f32)?;
        }
        let compact_w1 = weights(0)?;
        let w2 = weights(1)?;
        if hp.max_pairs * 2 > hp.n_hidden {
            return Err("initial checkpoint max-pair count exceeds its hidden width".into());
        }
        for j in 0..hp.max_pairs {
            let u = 2 * j;
            let v = u + 1;
            if l0.biases[u] != 0.0
                || w2[u].to_bits() != w2[v].to_bits()
                || compact_w1
                    .chunks_exact(hp.n_hidden)
                    .any(|row| row[u] < 0.0 || row[u] + row[v] < 0.0)
            {
                return Err("initial checkpoint max-pair parameterization differs".into());
            }
        }
        let mut result = Self {
            scaler_mean: vec![0.0; n_features],
            scaler_scale: vec![1.0; n_features],
            w1: vec![0.0; n_features * hp.n_hidden],
            b1: l0.biases.iter().map(|&x| f64::from(x)).collect(),
            w2,
            b2: l1.biases.iter().map(|&x| f64::from(x)).collect(),
        };
        for (i, &column) in columns.iter().enumerate() {
            result.scaler_mean[column] = f64::from(model.scaler_mean()[i]);
            result.scaler_scale[column] = f64::from(model.scaler_scale()[i]);
            result.w1[column * hp.n_hidden..(column + 1) * hp.n_hidden]
                .copy_from_slice(&compact_w1[i * hp.n_hidden..(i + 1) * hp.n_hidden]);
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::super::bake_two_layer_znpr_v3;
    use super::*;
    use zenpredict::WeightDtype;

    fn fixture() -> (Vec<u8>, MlpHyperparams) {
        let hp = MlpHyperparams {
            n_hidden: 2,
            nonneg_distance: true,
            nonneg_thresholds: true,
            monotone_cbc: true,
            leaky_alpha: 0.0,
            ..MlpHyperparams::default()
        };
        let bytes = bake_two_layer_znpr_v3(
            &[0.0, 0.0],
            &[0.25, 2.0],
            &[1.0, 2.0, 3.0, 4.0],
            &[-0.5, -1.0],
            &[-2.0, -3.0],
            &[100.0],
            2,
            2,
            1,
            WeightDtype::F32,
            None,
            None,
            Activation::Relu,
        );
        let repro = serde_json::json!({"argv":["trainer", "--hidden", "2", "--nonneg-distance",
            "--nonneg-thresholds", "--monotone-cbc"]})
        .to_string();
        let bytes =
            zenpredict_bake::append_metadata_utf8(&bytes, "zentrain.repro", &repro).unwrap();
        (bytes, hp)
    }

    #[test]
    fn unpacked_weights_and_scaler_roundtrip_exactly() {
        let (bytes, hp) = fixture();
        let init = InitialWeights::load(&bytes, 2, &hp).unwrap();
        let after = bake_two_layer_znpr_v3(
            &init.scaler_mean,
            &init.scaler_scale,
            &init.w1,
            &init.b1,
            &init.w2,
            &init.b2,
            2,
            2,
            1,
            WeightDtype::F32,
            None,
            None,
            Activation::Relu,
        );
        let original = Model::from_bytes(&bytes).unwrap();
        let model = Model::from_bytes(&after).unwrap();
        let mut original = zenpredict::Predictor::new(&original);
        let mut model = zenpredict::Predictor::new(&model);
        let input = [0.0_f32, 0.125, 0.25, 1.0, 2.0, 32.0];
        for &x in &input {
            for &y in &input {
                let a = original.predict(&[x, y]).unwrap()[0].to_bits();
                let b = model.predict(&[x, y]).unwrap()[0].to_bits();
                assert_eq!(a, b);
            }
        }
    }

    #[test]
    fn refuses_mismatched_width_head_and_monotone_parameters() {
        let (bytes, hp) = fixture();
        assert!(InitialWeights::load(&bytes, 3, &hp).is_err());
        for changed in [
            MlpHyperparams {
                n_hidden: 3,
                ..hp.clone()
            },
            MlpHyperparams {
                nonneg_thresholds: false,
                ..hp.clone()
            },
            MlpHyperparams {
                max_pairs: 1,
                ..hp.clone()
            },
            MlpHyperparams {
                monotone_cbc: false,
                ..hp.clone()
            },
            MlpHyperparams {
                nonneg_pin: 99.0,
                ..hp.clone()
            },
            MlpHyperparams {
                hybrid_head: true,
                ..hp.clone()
            },
        ] {
            assert!(InitialWeights::load(&bytes, 2, &changed).is_err());
        }
    }

    #[test]
    fn refuses_packed_or_unproven_checkpoints() {
        let (bytes, hp) = fixture();
        let missing = bake_two_layer_znpr_v3(
            &[0.0, 0.0],
            &[1.0, 1.0],
            &[1.0, 2.0, 3.0, 4.0],
            &[0.0, 0.0],
            &[-2.0, -3.0],
            &[100.0],
            2,
            2,
            1,
            WeightDtype::F32,
            None,
            None,
            Activation::Relu,
        );
        assert!(InitialWeights::load(&missing, 2, &hp).is_err());
        let half = bake_two_layer_znpr_v3(
            &[0.0, 0.0],
            &[1.0, 1.0],
            &[1.0, 2.0, 3.0, 4.0],
            &[0.0, 0.0],
            &[-2.0, -3.0],
            &[100.0],
            2,
            2,
            1,
            WeightDtype::F16,
            None,
            None,
            Activation::Relu,
        );
        let source = Model::from_bytes(&bytes).unwrap();
        let half = zenpredict_bake::append_metadata_utf8(
            &half,
            "zentrain.repro",
            source.metadata().get_utf8("zentrain.repro").unwrap(),
        )
        .unwrap();
        assert!(InitialWeights::load(&half, 2, &hp).is_err());
    }
}
