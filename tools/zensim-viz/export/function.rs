//! Algebraic head inspection. Curves execute the existing native predictors;
//! they are basis slices, not bank evidence or a memorization experiment.
use crate::{
    Result,
    constraints::Constraints,
    model::{self, Input},
};
use serde_json::{Value, json};
use zenpredict::{Activation, Model, Predictor};

fn slice(model: &Model, inputs: &[Input], units: &[usize]) -> Result<Value> {
    let layer = model.layer(0);
    let weights = model::weights(model, 0);
    let output = model::weights(model, 1);
    let selected = *units.last().ok_or("empty function term")?;
    // A raw input axis with all other raw slots (including fragility) zero.
    // Products are thus exactly zero; no unattainable independent product is
    // silently substituted for the declared raw-feature intervention.
    let axis = inputs
        .iter()
        .filter(|i| i.factor.is_none())
        .filter(|i| weights[i.index * layer.out_dim + selected] > 0.0)
        .max_by(|a, b| {
            let gain = |i: &Input| {
                f64::from(weights[i.index * layer.out_dim + selected])
                    / f64::from(model.scaler_scale()[i.index])
            };
            gain(a).total_cmp(&gain(b))
        })
        .ok_or("function basis slice needs a positive direct-input coefficient")?;
    let gain = f64::from(weights[axis.index * layer.out_dim + selected])
        / f64::from(model.scaler_scale()[axis.index]);
    let threshold = -f64::from(layer.biases[selected]) / gain;
    let high = if threshold > 0.0 {
        2.0 * threshold
    } else {
        2.0 / gain
    };
    if !high.is_finite() || high > f64::from(f32::MAX) {
        return Err("nonfinite function slice".into());
    }
    let first = model::layer_model(model, 0)?;
    let last = model::layer_model(model, 1)?;
    let mut hidden = Predictor::new(&first);
    let mut final_layer = Predictor::new(&last);
    let mut complete = Predictor::new(model);
    let mut scorer = zensim::BakeScorer::new(model)?;
    let mut points = Vec::new();
    for k in 0..=64 {
        let x = (high * k as f64 / 64.0) as f32;
        let mut row = vec![0.0; zensim::research::full_width()];
        row[axis.feature] = f64::from(x);
        let gathered: Vec<_> = inputs.iter().map(|i| i.value(&row)).collect();
        let h = hidden.predict_transformed(&gathered)?.to_vec();
        let network = complete.predict_transformed(&gathered)?[0];
        if final_layer.predict(&h)?[0].to_bits() != network.to_bits() {
            return Err("function layer replay differs from complete native predictor bits".into());
        }
        let terms: Vec<_> = units
            .iter()
            .map(|&j| f64::from(output[j]) * f64::from(h[j]))
            .collect();
        let mut point = json!({"x":x,"activations":units.iter().map(|&j| h[j]).collect::<Vec<_>>(),
            "weighted_terms":terms,"term_sum":terms.iter().sum::<f64>(),"network":network,
            "served":scorer.score_features(&row,512,512,None)?});
        if let [u, v] = units {
            let z = f64::from(x) / f64::from(model.scaler_scale()[axis.index]);
            let a = f64::from(weights[axis.index * layer.out_dim + u]) * z;
            let b = (f64::from(weights[axis.index * layer.out_dim + u])
                + f64::from(weights[axis.index * layer.out_dim + v]))
                * z
                + f64::from(layer.biases[*v]);
            point["branch_a"] = json!(f64::from(output[*u]) * a);
            point["branch_b"] = json!(f64::from(output[*u]) * b);
        }
        points.push(point);
    }
    Ok(
        json!({"input":axis.index,"feature":axis.feature,"threshold":threshold,"points":points,
        "native_layer_replay_bit_identity":true,
        "definition":"Raw feature axis; every other raw slot and fragility is zero. Native Predictor activations and BakeScorer scores, 512×512, no codec hint. A mathematical basis slice, not an observed or necessarily attainable image; not PDP."}),
    )
}

pub(crate) fn inspect(model: &Model, inputs: &[Input], constraints: &Constraints) -> Result<Value> {
    if constraints.certificate["certified"] != true {
        return Err(
            "paired function exhibit requires the canonical monotonicity certificate".into(),
        );
    }
    if model.n_layers() != 2
        || model.layer(0).activation != Activation::Relu
        || model.layer(1).activation != Activation::Identity
        || model.n_outputs() != 1
    {
        return Err(
            "function exhibit requires one ReLU hidden layer and one identity output".into(),
        );
    }
    if model.has_nontrivial_feature_transforms() || model.scaler_mean().iter().any(|&v| v != 0.0) {
        return Err("raw-axis function exhibit requires untransformed, scale-only inputs".into());
    }
    if model
        .scaler_scale()
        .iter()
        .any(|&v| !v.is_finite() || v <= 0.0)
    {
        return Err("function exhibit requires positive finite scales".into());
    }
    let meta = zensim::bake_metadata::parse_bake_metadata(model)?;
    if meta.hybrid_head.is_some() || meta.minmax_head.is_some() || meta.per_sample_alpha.is_some() {
        return Err("function exhibit does not yet describe auxiliary output heads".into());
    }
    let l = model.layer(0);
    let w = model::weights(model, 0);
    let c = model::weights(model, 1);
    if w.iter()
        .chain(c.iter())
        .chain(l.biases.iter())
        .chain(model.layer(1).biases.iter())
        .any(|v| !v.is_finite())
    {
        return Err("nonfinite function coefficient".into());
    }
    let document = model::weights_document(model, inputs);
    let capacity = &document["capacity"]["units"];
    let included: Vec<_> = (0..l.out_dim)
        .filter(|&j| c[j] != 0.0 && capacity[j]["status"] != "dead")
        .collect();
    let paired: std::collections::BTreeSet<_> = constraints
        .max_pairs
        .iter()
        .flat_map(|&(u, v)| [u, v])
        .collect();
    let pairs = constraints
        .max_pairs
        .iter()
        .enumerate()
        .map(|(index, &(u, v))| -> Result<Value> {
            let mut coefficients: Vec<_> = inputs
                .iter()
                .map(|i| {
                    json!({"input":i.index,
            "a":w[i.index*l.out_dim+u],"delta":w[i.index*l.out_dim+v],
            "b":f64::from(w[i.index*l.out_dim+u])+f64::from(w[i.index*l.out_dim+v])})
                })
                .collect();
            coefficients.sort_by(|a, b| {
                let magnitude = |r: &Value| {
                    r["a"]
                        .as_f64()
                        .unwrap()
                        .abs()
                        .max(r["delta"].as_f64().unwrap().abs())
                };
                magnitude(b)
                    .total_cmp(&magnitude(a))
                    .then(a["input"].as_u64().cmp(&b["input"].as_u64()))
            });
            Ok(
                json!({"index":index,"units":[u,v],"coefficient":c[u],"bias":l.biases[v],
            "coefficients":coefficients,"slice":slice(model,inputs,&[u,v])?}),
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let ordinary=included.iter().filter(|j|!paired.contains(j)).map(|&j| -> Result<Value> {
        let mut input_order:Vec<_>=(0..l.in_dim).collect();
        input_order.sort_by(|&a,&b|w[b*l.out_dim+j].abs().total_cmp(&w[a*l.out_dim+j].abs()).then(a.cmp(&b)));
        Ok(json!({"unit":j,"coefficient":c[j],"bias":l.biases[j],"status":capacity[j]["status"],"input_order":input_order,"slice":slice(model,inputs,&[j])?}))
    }).collect::<Result<Vec<_>>>()?;
    let nonconstant = included
        .iter()
        .filter(|&&j| (0..l.in_dim).any(|i| w[i * l.out_dim + j] != 0.0))
        .count();
    let zero_output_nondead: Vec<_> = (0..l.out_dim)
        .filter(|&j| c[j] == 0.0 && capacity[j]["status"] != "dead")
        .collect();
    let ordinary_nonconstant = included
        .iter()
        .filter(|&&j| !paired.contains(&j) && (0..l.in_dim).any(|i| w[i * l.out_dim + j] != 0.0))
        .count();
    let certified_line_bound = (constraints.certificate["certified"] == true)
        .then_some(constraints.max_pairs.len() + ordinary_nonconstant + 1);
    let switches = constraints.max_pairs.len() + ordinary_nonconstant;
    let affine_forms_bound = 1u128
        .checked_shl(switches as u32)
        .map_or_else(|| format!("2^{switches}"), |v| v.to_string());
    let spline = meta
        .output_spline
        .as_ref()
        .map(|s| json!({"x":s.xs,"y":s.ys,"derivatives":s.derivs}));
    Ok(
        json!({"schema":"zensim-function-v1","output_bias":model.layer(1).biases[0],
        "certificate":constraints.certificate,"pairs":pairs,"ordinary_terms":ordinary,
        "included_units":included,"zero_output_nondead_units":zero_output_nondead,"capacity":document["capacity"],
        "nonconstant_output_gates":nonconstant,"normalized_line_segments_upper_bound":nonconstant+1,
        "certified_line_segments_upper_bound":certified_line_bound,
        "binary_switches_upper_bound":switches,"global_affine_forms_upper_bound":affine_forms_bound,
        "global_bound_scope":"Algebraic upper bound before output calibration on the certified domain: each max pair or remaining nonconstant ReLU chooses between two affine branches. Their sum has at most 2^K affine formulas for K switches; some choices may be unreachable or identical. This is not a measured region count or a statistical capacity/generalization guarantee.",
        "certified_line_bound_scope":"On the certified nonnegative damage domain (declared fragility factors in [0,1]), each max of two affine planes adds at most one switch on a straight line in standardized inputs; each remaining nonconstant ReLU adds at most one. This bound excludes feature extraction, raw-to-product nonlinearities and output calibration. It does not count reachable regions or measure memorization.",
        "line_bound_scope":"Before the output head/calibration, along a straight line in standardized model inputs: each nonconstant affine preactivation crosses zero at most once. An upper bound, not a measured number of reachable pieces; raw fragility products and pixel descriptors are nonlinear.",
        "memorization":"Not established. Structural complexity and small coefficients do not prove absence of memorization or generalization. All nonzero residual terms are retained; no random-label fit, training/held-out gap, membership or duplicate-source experiment is performed here.",
        "algebra_scope":"Real-arithmetic explanation of stored coefficients, not a replacement scorer. Pair c·max(A,B) holds where the partner A is nonnegative; the native certificate proves this on nonnegative damage inputs with declared fragility factors in [0,1]. Reassociation need not preserve f32 bits. Curves use native activations and complete serving arithmetic.",
        "preprocessing":"d_i = raw feature or raw feature × declared fragility; z_i = (d_i − mean_i) / scale_i. Means are zero and scales positive in this export. Products are formed before standardization.",
        "output":{"tanh_pin_scale":meta.tanh_pin_scale,"spline":spline,
            "codec_calibration_present":model.metadata().get("zentrain.per_codec_calibration").is_some(),
            "scope":"Native serving may apply an output head, monotone calibration spline and codec calibration after the scalar network. Basis curves call BakeScorer rather than rebuilding these stages in JavaScript."}}),
    )
}

#[cfg(all(test, feature = "real-bake-tests"))]
mod tests {
    use super::*;
    #[test]
    fn frozen_f_keeps_every_residual_term_and_replays_pair_switches() {
        let path = std::env::var("VIZ_F_BAKE").expect("VIZ_F_BAKE is required");
        let model = Model::from_bytes(&std::fs::read(&path).unwrap()).unwrap();
        let inputs = model::inputs(&model).unwrap();
        let constraints =
            crate::constraints::inspect(&model, &inputs, std::path::Path::new(&path)).unwrap();
        let function = inspect(&model, &inputs, &constraints).unwrap();
        assert_eq!(function["pairs"].as_array().unwrap().len(), 1);
        assert_eq!(function["pairs"][0]["units"], json!([126, 127]));
        assert_eq!(function["ordinary_terms"].as_array().unwrap().len(), 31);
        assert_eq!(function["included_units"].as_array().unwrap().len(), 33);
        assert_eq!(function["certified_line_segments_upper_bound"], 33);
        assert_eq!(function["global_affine_forms_upper_bound"], "4294967296");
        let w = model::weights(&model, 0);
        for row in function["pairs"][0]["coefficients"].as_array().unwrap() {
            let i = row["input"].as_u64().unwrap() as usize;
            assert_eq!(row["a"], json!(w[i * 128 + 126]));
            assert_eq!(row["delta"], json!(w[i * 128 + 127]));
        }
        let points = function["pairs"][0]["slice"]["points"].as_array().unwrap();
        assert_eq!(points.len(), 65);
        assert_eq!(points[0]["activations"][1], json!(0.0));
        assert!(points[64]["activations"][1].as_f64().unwrap() > 0.0);
        assert_eq!(
            function["pairs"][0]["slice"]["native_layer_replay_bit_identity"],
            true
        );
    }
}
