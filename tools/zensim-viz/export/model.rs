use crate::Result;
use serde::Serialize;
use zenpredict::{Model, WeightStorage};

#[derive(Clone, Debug, Serialize)]
pub(crate) struct Input {
    pub index: usize,
    pub feature: usize,
    pub factor: Option<usize>,
    pub name: String,
    pub family: String,
    pub signal: String,
    pub analysis_family: String,
    pub short_name: String,
    pub scale: u8,
    pub channel: String,
    pub pooling: String,
    pub form: String,
}
impl Input {
    pub fn matches_family(&self, family: &str) -> bool {
        self.family == family || self.analysis_family == family
    }
    pub fn value(&self, row: &[f64]) -> f32 {
        let a = row[self.feature] as f32;
        self.factor.map_or(a, |b| a * row[b] as f32)
    }
}
pub(crate) fn inputs(model: &Model) -> Result<Vec<Input>> {
    let text = model.metadata().get_utf8("zensim.derived_inputs")?;
    let mut lines = text.lines().filter(|s| !s.trim().is_empty());
    if lines.next() != Some("zensim-derived-inputs v1") {
        return Err("unsupported derived declaration".into());
    }
    let mut inputs: Vec<Input> = lines
        .enumerate()
        .map(|(index, line)| -> Result<Input> {
            let words: Vec<_> = line.split_whitespace().collect();
            let (feature, factor) = match words.as_slice() {
                ["in", a] => (a.parse()?, None),
                ["product", a, b] => (a.parse()?, Some(b.parse()?)),
                _ => return Err("invalid derived input".into()),
            };
            Ok(Input {
                family: String::new(),
                signal: String::new(),
                analysis_family: String::new(),
                short_name: String::new(),
                scale: 0,
                channel: String::new(),
                pooling: String::new(),
                form: String::new(),
                index,
                feature,
                factor,
                name: factor
                    .map_or_else(|| format!("f{feature}"), |b| format!("f{feature} × f{b}")),
            })
        })
        .collect::<Result<_>>()?;
    if inputs.len() != model.n_inputs() {
        return Err("derived count differs from model width".into());
    }
    // The public scorer validates declarations, revisions and supported output stages.
    zensim::BakeScorer::new(model)?;
    let slots = zensim::feature_set_id::SlotSet::from_slots(inputs.iter().map(|i| i.feature));
    let request = zensim::research::Request::for_slots(slots, zensim::research::full_width());
    let pixels = vec![[128u8; 3]; 32 * 32];
    let image = zensim::source::RgbSlice::new(&pixels, 32, 32);
    let catalog = zensim::research::extract(&request, &image, &image)?;
    for input in &mut inputs {
        let p = catalog
            .provenance()
            .iter()
            .find(|p| usize::from(p.id) == input.feature && p.populated)
            .ok_or("input missing from registry provenance")?;
        input.name = if input.factor.is_some() {
            format!("{} × fragility", p.name)
        } else {
            p.name.clone()
        };
        input.family = p.family.into();
        input.signal = signal_name(&p.name, p.family);
        input.analysis_family = format!("{} × {}", p.family, input.signal);
        input.short_name = format!(
            "{}_{}_s{}_{}{}",
            p.family,
            input.signal,
            p.scale,
            p.channel,
            if input.factor.is_some() {
                " × fragility"
            } else {
                ""
            }
        );
        input.scale = p.scale;
        input.channel = p.channel.into();
        input.pooling = p.statistic.into();
        input.form = p.form.into();
    }
    Ok(inputs)
}
// Registry names encode signal, pool, scale and channel. Pool variants share a
// signal group; the registry's broad family remains unchanged in `family`.
pub(crate) fn signal_name(name: &str, family: &str) -> String {
    let prefix = name.rsplit_once("_s").map_or(name, |(a, _)| a);
    let mut prefix = prefix;
    for suffix in ["_mean", "_4th", "_2nd", "_max", "_l8"] {
        if let Some(value) = prefix.strip_suffix(suffix) {
            prefix = value;
            break;
        }
    }
    if prefix == "v2_ssim_var" || prefix == "v2_ssim_skew" {
        prefix = "v2_ssim";
    }
    prefix
        .strip_prefix(&format!("{family}_"))
        .unwrap_or(prefix)
        .into()
}
pub(crate) fn weights(model: &Model, layer: usize) -> Vec<f32> {
    let l = model.layer(layer);
    match l.weights {
        WeightStorage::F32(w) => w.to_vec(),
        WeightStorage::F16(w) => w.iter().map(|&v| zenpredict::f16_bits_to_f32(v)).collect(),
        WeightStorage::I8 { weights, scales } => weights
            .iter()
            .enumerate()
            .map(|(i, &v)| f32::from(v) * scales[i % l.out_dim])
            .collect(),
    }
}
pub(crate) fn weights_document(model: &Model, inputs: &[Input]) -> serde_json::Value {
    let first = model.layer(0);
    let w = weights(model, 0);
    let mut row_order: Vec<_> = (0..inputs.len()).collect();
    row_order.sort_by_key(|&i| {
        (
            &inputs[i].family,
            &inputs[i].signal,
            inputs[i].feature,
            inputs[i].factor,
            i,
        )
    });
    let energy: Vec<f64> = (0..first.out_dim)
        .map(|j| {
            (0..first.in_dim)
                .map(|i| f64::from(w[i * first.out_dim + j].abs()))
                .sum()
        })
        .collect();
    let mut column_order: Vec<_> = (0..first.out_dim).collect();
    column_order.sort_by(|&a, &b| energy[b].total_cmp(&energy[a]).then(a.cmp(&b)));
    let per_input_columns: Vec<Vec<usize>> = w
        .chunks(first.out_dim)
        .map(|row| {
            let mut order: Vec<_> = (0..row.len()).collect();
            order.sort_by(|&a, &b| row[b].abs().total_cmp(&row[a].abs()).then(a.cmp(&b)));
            order
        })
        .collect();
    let top_inputs_by_unit: Vec<Vec<usize>> = (0..first.out_dim)
        .map(|j| {
            let mut order: Vec<_> = (0..first.in_dim).collect();
            order.sort_by(|&a, &b| {
                w[b * first.out_dim + j]
                    .abs()
                    .total_cmp(&w[a * first.out_dim + j].abs())
                    .then(a.cmp(&b))
            });
            order.truncate(32);
            order
        })
        .collect();
    let nonzero_counts: Vec<_> = w
        .chunks(first.out_dim)
        .map(|row| row.iter().filter(|&&v| v != 0.0).count())
        .collect();
    let input_l1: Vec<f64> = w
        .chunks(first.out_dim)
        .map(|row| row.iter().map(|v| f64::from(v.abs())).sum())
        .collect();
    serde_json::json!({"inputs":inputs,"row_order":row_order,"column_order":column_order,"per_input_columns":per_input_columns,"top_inputs_by_unit":top_inputs_by_unit,"nonzero_counts":nonzero_counts,"input_l1":input_l1,"scaler_mean":model.scaler_mean(),"scaler_scale":model.scaler_scale(),
        "layers": (0..model.n_layers()).map(|i| { let l=model.layer(i); serde_json::json!({"layer":i,"in_dim":l.in_dim,"out_dim":l.out_dim,"activation":format!("{:?}",l.activation),"weights":weights(model,i),"biases":l.biases,"stats":crate::audit::weight_statistics(&weights(model,i),l.in_dim,l.out_dim)}) }).collect::<Vec<_>>()})
}
// Compose diagnostic layer views with the existing baker, then execute
// them with Predictor. This preserves the one owner of inference arithmetic.
pub(crate) fn layer_model(model: &Model, index: usize) -> Result<Model> {
    if model.has_nontrivial_feature_transforms() {
        return Err("layer inspection refuses nontrivial input transforms".into());
    }
    let l = model.layer(index);
    if matches!(l.weights, WeightStorage::I8 { .. }) {
        return Err("layer inspection does not yet support i8 post-dot scaling".into());
    }
    let w = weights(model, index);
    let zeros = vec![0.0; l.in_dim];
    let ones = vec![1.0; l.in_dim];
    let (mean, scale) = if index == 0 {
        (model.scaler_mean(), model.scaler_scale())
    } else {
        (zeros.as_slice(), ones.as_slice())
    };
    let layers = [zenpredict_bake::BakeLayer {
        in_dim: l.in_dim,
        out_dim: l.out_dim,
        activation: l.activation,
        dtype: zenpredict::WeightDtype::F32,
        weights: &w,
        biases: l.biases,
    }];
    let bytes = zenpredict_bake::BakeRequest::builder(model.schema_hash(), 0, mean, scale, &layers)
        .bake()?;
    Ok(Model::from_bytes(&bytes)?)
}
