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
    pub scale: u8,
    pub channel: String,
    pub pooling: String,
    pub form: String,
}
impl Input {
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
        input.scale = p.scale;
        input.channel = p.channel.into();
        input.pooling = p.statistic.into();
        input.form = p.form.into();
    }
    Ok(inputs)
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
    serde_json::json!({"inputs":inputs,"scaler_mean":model.scaler_mean(),"scaler_scale":model.scaler_scale(),
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
