//! Internal positive-segment export shared by the trainer and repair command.
//! Input coordinates are ReLU(z-knot). Bounded segments use width-ReLU(width-x),
//! avoiding subtraction of two large positive hinges at a saturated segment.
use zenpredict::Activation;

pub(crate) struct Layer {
    pub ni: usize,
    pub no: usize,
    pub activation: Activation,
    pub weights: Vec<f32>,
    pub biases: Vec<f32>,
}

pub(crate) fn bounded(no: usize, knots: &[f64], slopes: &[f32]) -> [Layer; 3] {
    let n = no * knots.len();
    assert_eq!(slopes.len(), n);
    assert!(slopes.iter().all(|s| s.is_finite() && *s >= 0.));
    assert!(knots.windows(2).all(|p| p[1] > p[0]));
    let layer = |ni, no, activation| Layer {
        ni,
        no,
        activation,
        weights: vec![0.; ni * no],
        biases: vec![0.; no],
    };
    let mut reverse = layer(n, n, Activation::Relu);
    let mut recover = layer(n, n, Activation::Identity);
    let mut combine = layer(n, no, Activation::Identity);
    for j in 0..no {
        for k in 0..knots.len() {
            let i = j * knots.len() + k;
            if let Some(&end) = knots.get(k + 1) {
                let width = (end - knots[k]) as f32;
                reverse.weights[i * n + i] = -1.;
                reverse.biases[i] = width;
                recover.weights[i * n + i] = -1.;
                recover.biases[i] = width;
            } else {
                reverse.weights[i * n + i] = 1.;
                recover.weights[i * n + i] = 1.;
            }
            combine.weights[i * no + j] = slopes[i];
        }
    }
    [reverse, recover, combine]
}
