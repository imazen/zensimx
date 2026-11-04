//! R14 smooth, zero-preserving input-transform admission.
use zenpredict::FeatureTransform;

pub const REGISTRATION: &str = "5b5a013482db4944042883995ad6892f92244c69";

/// Analytic positive finite one-sided derivative at zero. This is a structural
/// real-arithmetic contract; floating-point plateau gates remain separate.
pub fn slope_at_zero(t: FeatureTransform, p: &[f32]) -> Result<f64, String> {
    use FeatureTransform::*;
    if p.iter().any(|v| !v.is_finite()) {
        return Err("R14 transform parameters must be finite".into());
    }
    let slope = match t {
        Identity | Log1p | SignedLog1p if p.is_empty() => 1.0,
        YeoJohnson if p.len() == 1 => 1.0,
        SoftClip if p.len() == 2 && p[0] > 0.0 && p[1] > 0.0 => 1.0,
        SoftSign if p.len() == 1 && p[0] > 0.0 => 1.0 / f64::from(p[0]),
        _ => {
            return Err(format!(
                "R14 disallowed transform or invalid parameters: {} {p:?}",
                t.as_token()
            ));
        }
    };
    if t.apply_with_params(0.0, p) != 0.0 || !slope.is_finite() || slope <= 0.0 {
        return Err("R14 transform violates zero/finite-positive-slope contract".into());
    }
    Ok(slope)
}

pub fn validate(ts: &[FeatureTransform], ps: Option<&[Vec<f32>]>) -> Result<(), String> {
    if ps.is_some_and(|p| p.len() != ts.len()) {
        return Err("R14 transform parameter width differs".into());
    }
    for (i, &t) in ts.iter().enumerate() {
        slope_at_zero(t, ps.map_or(&[], |p| p[i].as_slice()))
            .map_err(|e| format!("input {i}: {e}"))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn smooth_allowlist_has_analytic_finite_positive_zero_slopes() {
        for (t, p, s) in [
            (FeatureTransform::Identity, vec![], 1.),
            (FeatureTransform::Log1p, vec![], 1.),
            (FeatureTransform::SignedLog1p, vec![], 1.),
            (FeatureTransform::YeoJohnson, vec![-2.], 1.),
            (FeatureTransform::SoftClip, vec![0.25, 0.25], 1.),
            (FeatureTransform::SoftSign, vec![0.25], 4.),
        ] {
            assert_eq!(slope_at_zero(t, &p).unwrap(), s);
            let mut prev = 0.;
            for i in 1..=1000 {
                let v = t.apply_with_params(i as f32 / 100., &p);
                assert!(v >= prev);
                prev = v;
            }
        }
    }
    #[test]
    fn roots_flats_arity_changes_and_invalid_params_are_refused() {
        for (t, p) in [
            (FeatureTransform::SignedPow, vec![0.5]),
            (FeatureTransform::SignedSqrt, vec![]),
            (FeatureTransform::SignedCbrt, vec![]),
            (FeatureTransform::ClipThenLog1p, vec![0.]),
            (FeatureTransform::WinsorP99, vec![0., 1.]),
            (FeatureTransform::Drop, vec![]),
            (FeatureTransform::Log, vec![]),
            (FeatureTransform::SoftClip, vec![0., 1.]),
            (FeatureTransform::SoftSign, vec![-1.]),
            (FeatureTransform::YeoJohnson, vec![f32::NAN]),
            (FeatureTransform::YeoJohnson, vec![]),
        ] {
            assert!(slope_at_zero(t, &p).is_err());
        }
    }
}

#[cfg(test)]
mod admission_tests {
    use super::*;
    #[test]
    fn registration_is_required_and_does_not_authorize_roots() {
        let mut hp = super::super::MlpHyperparams {
            nonneg_distance: true,
            monotone_cbc: true,
            feature_transforms: Some(vec![FeatureTransform::SoftClip]),
            feature_transform_params: Some(vec![vec![0.5, 0.5]]),
            ..Default::default()
        };
        assert!(
            std::panic::catch_unwind(|| super::super::validate_training_capabilities(
                &hp, false, 0
            ))
            .is_err()
        );
        hp.r14_transforms_registration = Some(REGISTRATION.into());
        super::super::validate_training_capabilities(&hp, false, 0);
        hp.feature_transforms = Some(vec![FeatureTransform::SignedSqrt]);
        hp.feature_transform_params = None;
        assert!(
            std::panic::catch_unwind(|| super::super::validate_training_capabilities(
                &hp, false, 0
            ))
            .is_err()
        );
    }
}

#[cfg(test)]
mod certificate_tests {
    use super::*;
    #[test]
    fn certificate_accepts_smooth_transforms_and_refuses_roots_and_signed_gate_shortcuts() {
        let base = serde_json::json!({"schema_hash":1,"scaler_mean":[0.,0.],"scaler_scale":[1.,1.],
            "metadata":[{"key":"zentrain.feature_transforms","type":"utf8","text":"soft_clip\nsoft_sign"},
                {"key":"zentrain.feature_transform_params","type":"utf8","text":"0.5,0.5\n2"}],
            "layers":[{"in_dim":2,"out_dim":1,"activation":"relu","dtype":"f32","weights":[1.,2.],"biases":[0.]},
                {"in_dim":1,"out_dim":1,"activation":"identity","dtype":"f32","weights":[-1.],"biases":[100.]}]});
        let bytes = zenpredict_bake::bake_from_json_str(&base.to_string()).unwrap();
        let model = zenpredict::Model::from_bytes(&bytes).unwrap();
        assert!(super::super::certify_monotone_bake(&model, 100.).is_ok());
        let mut invalid = base.clone();
        invalid["metadata"][0]["text"] = serde_json::json!("signed_sqrt\nsoft_sign");
        invalid["metadata"][1]["text"] = serde_json::json!("\n2");
        let bytes = zenpredict_bake::bake_from_json_str(&invalid.to_string()).unwrap();
        assert!(
            super::super::certify_monotone_bake(
                &zenpredict::Model::from_bytes(&bytes).unwrap(),
                100.
            )
            .is_err()
        );
        let mut gated = base;
        gated["metadata"].as_array_mut().unwrap().push(serde_json::json!({"key":"zensim.derived_inputs","type":"utf8","text":"zensim-derived-inputs v1\nin 13\nproduct 13 422"}));
        gated["layers"][0]["weights"] = serde_json::json!([1., -0.5]);
        let bytes = zenpredict_bake::bake_from_json_str(&gated.to_string()).unwrap();
        assert!(
            super::super::certify_monotone_bake(
                &zenpredict::Model::from_bytes(&bytes).unwrap(),
                100.
            )
            .is_err()
        );
    }
}
