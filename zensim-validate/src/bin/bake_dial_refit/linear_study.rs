//! Private study adapter: existing Gram Cholesky, ZNPR baker/Predictor and dial owners.
use super::*;
use serde::Serialize;
use sha2::{Digest, Sha256};
use zensim_validate::gram_lasso::{StandardizedGram, ridge_solve};
use zensim_validate::npz::{Npz, NpzF64Entry, write_npz_f64};

#[derive(clap::Args)]
pub(super) struct Args {
    #[arg(long)]
    gram: PathBuf,
    #[arg(long)]
    inputs: PathBuf,
    #[arg(long)]
    spec: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long)]
    lambda: f64,
    #[arg(long)]
    positive: bool,
}

#[path = "linear_basis.rs"]
mod linear_basis;
use linear_basis::Spec;

fn system(g: Vec<f64>, c: Vec<f64>) -> StandardizedGram {
    let n = c.len();
    StandardizedGram {
        n_feat: n,
        g,
        c,
        w_total: 1.,
        mu: vec![0.; n],
        sd: vec![1.; n],
        ybar: 0.,
    }
}

fn gradient(s: &StandardizedGram, w: &[f64], lambda: f64) -> Vec<f64> {
    s.g.chunks_exact(s.n_feat)
        .zip(&s.c)
        .zip(w)
        .map(|((row, c), wi)| row.iter().zip(w).map(|(g, w)| g * w).sum::<f64>() - c + lambda * wi)
        .collect()
}

/// Lawson-Hanson active-set steps on the positive definite regularized Gram.
/// Principal systems reuse the existing Cholesky owner. No factorized readout.
fn solve(s: &StandardizedGram, lambda: f64, positive: bool) -> Result<(Vec<f64>, usize), String> {
    if !lambda.is_finite() || lambda <= 0. {
        return Err("this certificate requires positive lambda".into());
    }
    let n = s.n_feat;
    if !positive {
        return ridge_solve(s, lambda).map(|w| (w, 1));
    }
    let mut w = vec![0.; n];
    let mut passive = vec![false; n];
    for iteration in 0..(20 * n + 100) {
        let g = gradient(s, &w, lambda);
        let entering = (0..n)
            .filter(|&i| !passive[i] && g[i] < -1e-10)
            .min_by(|&a, &b| g[a].total_cmp(&g[b]));
        let Some(entering) = entering else {
            return Ok((w, iteration));
        };
        passive[entering] = true;
        let mut feasible = false;
        for _ in 0..(n + 1) {
            let ids: Vec<_> = (0..n).filter(|&i| passive[i]).collect();
            let mut subg = Vec::with_capacity(ids.len() * ids.len());
            for &i in &ids {
                for &j in &ids {
                    subg.push(s.g[i * n + j]);
                }
            }
            let subc = ids.iter().map(|&i| s.c[i]).collect();
            let z = ridge_solve(&system(subg, subc), lambda)?;
            if z.iter().all(|v| *v > 0.) {
                w.fill(0.);
                for (&i, &v) in ids.iter().zip(&z) {
                    w[i] = v;
                }
                feasible = true;
                break;
            }
            let alpha = ids
                .iter()
                .zip(&z)
                .filter(|(_, v)| **v <= 0.)
                .map(|(&i, &v)| w[i] / (w[i] - v))
                .fold(1., f64::min);
            if !alpha.is_finite() || !(0. ..=1.).contains(&alpha) {
                return Err("invalid active-set boundary step".into());
            }
            for (&i, &v) in ids.iter().zip(&z) {
                w[i] += alpha * (v - w[i]);
                if w[i] <= 1e-14 {
                    w[i] = 0.;
                    passive[i] = false;
                }
            }
        }
        if !feasible {
            return Err("active-set inner iteration limit".into());
        }
    }
    Err("active-set outer iteration limit; no convergence claim".into())
}

#[derive(Serialize)]
struct Certificate {
    lambda: f64,
    positive: bool,
    active_set_iterations: usize,
    nonzero_coefficients: usize,
    kkt_infinity: f64,
    objective_gap_upper_bound: f64,
    weighted_half_mse: f64,
    regularized_objective: f64,
}

fn certify(
    s: &StandardizedGram,
    w: &[f64],
    y2: f64,
    lambda: f64,
    positive: bool,
    iterations: usize,
) -> Result<Certificate, String> {
    if w.iter().any(|v| !v.is_finite() || (positive && *v < 0.)) {
        return Err("nonfinite or infeasible solution".into());
    }
    let g = gradient(s, w, lambda);
    let residual: Vec<_> = g
        .iter()
        .zip(w)
        .map(|(&g, &w)| if positive && w == 0. { g.min(0.) } else { g })
        .collect();
    let kkt = residual.iter().fold(0_f64, |v, x| v.max(x.abs()));
    let gap = residual.iter().map(|v| v * v).sum::<f64>() / (2. * lambda);
    let norm2 = w.iter().map(|v| v * v).sum::<f64>();
    let data_loss = 0.5
        * (y2
            + w.iter()
                .zip(&g)
                .zip(&s.c)
                .map(|((&w, &g), &c)| w * (g - c))
                .sum::<f64>()
            - lambda * norm2);
    if !gap.is_finite() || kkt > 1e-7 || gap > 1e-6 || data_loss < -1e-9 {
        return Err(format!(
            "optimality gate failed: KKT={kkt}, gap={gap}, half-MSE={data_loss}"
        ));
    }
    Ok(Certificate {
        lambda,
        positive,
        active_set_iterations: iterations,
        nonzero_coefficients: w.iter().filter(|v| **v != 0.).count(),
        kkt_infinity: kkt,
        objective_gap_upper_bound: gap,
        weighted_half_mse: data_loss,
        regularized_objective: data_loss + lambda * norm2 / 2.,
    })
}

pub(super) fn run(a: &Args) -> Result<(), String> {
    let spec_bytes = std::fs::read(&a.spec).map_err(|e| e.to_string())?;
    let spec: Spec = serde_json::from_slice(&spec_bytes).map_err(|e| e.to_string())?;
    let gram = Npz::open(&a.gram)?;
    let ga = gram.get("g")?;
    let ca = gram.get("c")?;
    let y2 = gram.get("y2")?.scalar_f64()?;
    let n = spec.width();
    if ca.shape != [n] || ga.shape != [n, n] || !y2.is_finite() {
        return Err("Gram shape or target moment mismatch".into());
    }
    let s = system(ga.f64s()?.to_vec(), ca.f64s()?.to_vec());
    if s.g.iter().chain(&s.c).any(|v| !v.is_finite()) {
        return Err("nonfinite Gram".into());
    }
    for i in 0..n {
        for j in 0..n {
            if (s.g[i * n + j] - s.g[j * n + i]).abs() > 1e-12 {
                return Err("asymmetric Gram".into());
            }
        }
    }
    let inputs = Npz::open(&a.inputs)?;
    let x = inputs.get("x")?;
    if x.shape.len() != 2 {
        return Err("input matrix rank mismatch".into());
    }
    let (rows, width) = (x.shape[0], x.shape[1]);
    let raw = x.f32s()?;
    if raw.iter().any(|v| !v.is_finite()) {
        return Err("nonfinite raw input".into());
    }
    spec.validate(width)?;
    if a.out.join("RESULT.json").exists() || a.out.join("model.bin").exists() {
        return Err("preserve existing study output".into());
    }
    std::fs::create_dir_all(&a.out).map_err(|e| e.to_string())?;
    let (w, iterations) = solve(&s, a.lambda, a.positive)?;
    let certificate = certify(&s, &w, y2, a.lambda, a.positive, iterations)?;
    let basis_metadata = serde_json::json!({"schema":spec.schema, "terms":spec.terms,
        "scales":spec.scales, "knots":spec.knots, "basis_scales":spec.basis_scales});
    let metadata = [OwnedMeta {
        key: "zentrain.research_basis".into(),
        kind: MetadataType::Utf8,
        value: serde_json::to_vec(&basis_metadata).map_err(|e| e.to_string())?,
    }];
    let layer = OwnedLayer {
        in_dim: n,
        out_dim: 1,
        activation: Activation::Identity,
        dtype: WeightDtype::F32,
        weights: w.iter().map(|v| (-100. * v) as f32).collect(),
        biases: vec![100.],
    };
    let bytes = bake_layers(&vec![0.; n], &vec![1.; n], &[layer], &metadata, 0);
    std::fs::write(a.out.join("model.bin"), &bytes).map_err(|e| e.to_string())?;
    let model = Model::from_bytes(&bytes).map_err(|e| format!("bake parse: {e:?}"))?;
    let mut predictor = zenpredict::Predictor::new(&model);
    let mut basis = vec![0_f32; n];
    let mut native = Vec::with_capacity(rows);
    let mut optimum = Vec::with_capacity(rows);
    let mut hasher = Sha256::new();
    for row in raw.chunks_exact(width) {
        spec.apply(row, &mut basis)?;
        for &value in &basis {
            hasher.update(value.to_le_bytes());
        }
        optimum.push(
            100. * (1.
                - basis
                    .iter()
                    .zip(&w)
                    .map(|(&x, &w)| x as f64 * w)
                    .sum::<f64>()),
        );
        native.push(
            predictor
                .predict(&basis)
                .map_err(|e| format!("predict: {e:?}"))?[0] as f64,
        );
    }
    let mut zero = vec![0.; n];
    spec.apply(&vec![0.; width], &mut zero)?;
    let identity = predictor
        .predict(&zero)
        .map_err(|e| format!("identity: {e:?}"))?[0];
    if identity != 100. {
        return Err("native identity changed".into());
    }
    let max_drift = native
        .iter()
        .zip(&optimum)
        .map(|(a, b)| (a - b).abs())
        .fold(0., f64::max);
    write_npz_f64(
        &a.out.join("predictions.npz"),
        &[
            NpzF64Entry {
                name: "native",
                shape: &[rows],
                data: &native,
            },
            NpzF64Entry {
                name: "optimum",
                shape: &[rows],
                data: &optimum,
            },
            NpzF64Entry {
                name: "coefficients",
                shape: &[n],
                data: &w,
            },
        ],
    )?;
    let mut calibration = serde_json::json!({"status":"not_requested"});
    if !spec.calibration_ids.is_empty() {
        if spec.calibration_ids.len() != spec.calibration_targets.len()
            || spec.calibration_ids.iter().any(|i| *i >= rows)
            || spec.calibration_targets.iter().any(|v| !v.is_finite())
        {
            return Err("invalid calibration membership".into());
        }
        let predictions: Vec<_> = spec.calibration_ids.iter().map(|&i| native[i]).collect();
        match zensim_validate::dial_spline::fit_identity_pinned_knots(
            &predictions,
            &spec.calibration_targets,
            17,
            100.,
            2.,
        ) {
            Ok((xs, ys)) => {
                let payload = spline_payload(&xs, &ys);
                let checks = zensim_validate::dial_spline::check_identity_pinned(&payload, 100.)?;
                let curve = spline::parse_payload(&payload).ok_or("invalid spline payload")?;
                let mapped: Vec<_> = native.iter().map(|&p| spline::apply(p, &curve)).collect();
                std::fs::write(a.out.join("calibration.bin"), &payload)
                    .map_err(|e| e.to_string())?;
                write_npz_f64(
                    &a.out.join("calibrated.npz"),
                    &[NpzF64Entry {
                        name: "native",
                        shape: &[rows],
                        data: &mapped,
                    }],
                )?;
                calibration = serde_json::json!({"status":"measured", "knots":xs.len(), "checks":format!("{checks:?}")});
            }
            Err(reason) => {
                calibration = serde_json::json!({"status":"refused", "reason":reason});
            }
        }
    }
    let basis_sha256: String = hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    let result = serde_json::json!({"schema":"shaped-linear-native-v1", "certificate":certificate,
        "basis_sha256":basis_sha256, "rows":rows, "basis_width":n,
        "coefficients":w, "identity":identity, "maximum_f32_score_difference":max_drift,
        "calibration":calibration, "pixel_qualified":false,
        "certificate_scope":"f64 coefficients and supplied PSD Gram; not the rounded f32 bake"});
    std::fs::write(
        a.out.join("RESULT.json"),
        serde_json::to_vec_pretty(&result).unwrap(),
    )
    .map_err(|e| e.to_string())?;
    println!(
        "{}",
        serde_json::json!({"rows":rows,"width":n,"certificate":result["certificate"]})
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn free_and_active_bound_solutions_have_verified_gap() {
        let s = system(vec![2., 1., 1., 2.], vec![1., -1.]);
        let (free, _) = solve(&s, 0.5, false).unwrap();
        assert!((free[0] - 2. / 3.).abs() < 1e-12);
        assert!((free[1] + 2. / 3.).abs() < 1e-12);
        let (bounded, count) = solve(&s, 0.5, true).unwrap();
        assert!((bounded[0] - 0.4).abs() < 1e-14);
        assert_eq!(bounded[1], 0.);
        let c = certify(&s, &bounded, 4., 0.5, true, count).unwrap();
        assert!(c.objective_gap_upper_bound < 1e-25);
        assert!(certify(&s, &[0., 0.], 4., 0.5, true, 0).is_err());
        assert!(certify(&s, &[-0.1, 0.], 4., 0.5, true, 0).is_err());
    }

    #[test]
    fn correlated_singular_and_zero_columns_are_regularized() {
        let s = system(vec![1., 1., 0., 1., 1., 0., 0., 0., 0.], vec![1., 1., 0.]);
        for positive in [false, true] {
            let (w, count) = solve(&s, 1e-4, positive).unwrap();
            let expected = 1. / 2.0001;
            assert!((w[0] - expected).abs() < 1e-10);
            assert!((w[1] - expected).abs() < 1e-10);
            assert_eq!(w[2], 0.);
            certify(&s, &w, 1., 1e-4, positive, count).unwrap();
        }
        assert!(solve(&s, 0., true).is_err());
    }

    #[test]
    fn positive_segments_are_anchored_continuous_and_ordered() {
        let s = Spec {
            schema: "shaped-linear-basis-v1".into(),
            terms: vec![(0, None)],
            scales: vec![1.],
            knots: vec![[1., 2.]],
            basis_scales: vec![1.; 3],
            calibration_ids: vec![],
            calibration_targets: vec![],
        };
        s.validate(1).unwrap();
        let mut previous = vec![0.; 3];
        for x in [0., 1e-8, 0.5, 1., 1.5, 2., 3., 1e10] {
            let mut row = vec![0.; 3];
            s.apply(&[x], &mut row).unwrap();
            assert!(row.iter().zip(&previous).all(|(a, b)| a >= b));
            if x == 0. {
                assert_eq!(row, vec![0.; 3]);
            }
            previous = row;
        }
        assert!(s.apply(&[-1.], &mut [0.; 3]).is_err());
    }
}
