//! Exhaustive small-word oracle for the production conditional normalizer.
#[path = "../src/mlp_train/contrast_precision.rs"]
mod contrast_precision;

use contrast_precision::{Point, normalize, precision};

#[test]
fn exhaustive_modulo_draws_preserve_conditional_mean_one() {
    for bits in [3, 5, 8] {
        let points = [
            Point { x: 0., se: 0. },
            Point { x: 0.4, se: 0.1 },
            Point { x: 2., se: 0.3 },
        ];
        let scores = [1., 1., 2.];
        for exact_ties in [false, true] {
            let legal = |a: usize, b: usize| !exact_ties || scores[a] != scores[b];
            let norm = normalize(&points, 0.5, bits, legal).unwrap();
            let mut sum = 0.;
            let mut precision_sum = 0.;
            let mut precision_correction = 0.;
            let mut count = 0_u128;
            for u in 0..(1_usize << bits) {
                for v in 0..(1_usize << bits) {
                    let (a, b) = (u % 3, v % 3);
                    if a != b && legal(a, b) {
                        // Independent direct formula, not the precision helper.
                        let variance = points[a].se.max(0.05).powi(2)
                            + points[b].se.max(0.05).powi(2)
                            + 0.25 * (points[a].x - points[b].x).powi(2);
                        // The exhaustive oracle adds tens of thousands of
                        // terms; compensate that accumulation independently
                        // of the owner's compact multiplicity calculation.
                        let term = 1. / variance - precision_correction;
                        let next = precision_sum + term;
                        precision_correction = (next - precision_sum) - term;
                        precision_sum = next;
                        sum += norm.weight(points[a], points[b], 0.5);
                        count += 1;
                    }
                }
            }
            assert_eq!(norm.legal_draw_multiplicity, count);
            assert_eq!(norm.legal_ordered_pairs, if exact_ties { 4 } else { 6 });
            assert!((sum / count as f64 - 1.).abs() < 2e-12);
            assert!((precision_sum / count as f64 - norm.mean_precision).abs() < 2e-12);
        }
    }
}

#[test]
fn equal_precision_single_pair_symmetry_and_empty_support() {
    let points = [Point { x: 1., se: 0.2 }; 3];
    let equal = normalize(&points, 0.6, 64, |_, _| true).unwrap();
    assert_eq!(equal.weight(points[0], points[1], 0.6), 1.);
    let unequal = [Point { x: 0., se: 0. }, Point { x: 4., se: 1. }];
    let single = normalize(&unequal, 0.7, 64, |_, _| true).unwrap();
    assert_eq!(single.weight(unequal[0], unequal[1], 0.7), 1.);
    assert_eq!(single.weight(unequal[1], unequal[0], 0.7), 1.);
    assert!(normalize(&points, 0.6, 64, |_, _| false).is_none());
    assert!(normalize(&points[..1], 0.6, 64, |_, _| true).is_none());
    assert!(normalize(&[], 0.6, 64, |_, _| true).is_none());
}

#[test]
fn linking_term_and_item_floor_are_preserved() {
    let a = Point { x: 0., se: 0. };
    let b = Point { x: 2., se: 0. };
    assert_eq!(precision(a, a, 3.), 1. / (2. * 0.05_f64.powi(2)));
    assert_eq!(precision(a, b, 0.5), 1. / (2. * 0.05_f64.powi(2) + 1.));
    assert_eq!(precision(a, b, 0.5), precision(b, a, 0.5));
    let points = [a, b, Point { x: 1., se: 0. }];
    let norm = normalize(&points, 0.5, 64, |_, _| true).unwrap();
    assert_ne!(norm.weight(a, b, 0.5), 1.);
}

#[test]
fn invalid_uncertainties_and_linking_inputs_refuse() {
    let valid = Point { x: 1., se: 0.1 };
    for invalid in [
        Point {
            x: f64::NAN,
            se: 0.1,
        },
        Point { x: -1., se: 0.1 },
        Point { x: 1., se: -0.1 },
        Point {
            x: 1.,
            se: f64::INFINITY,
        },
    ] {
        assert!(
            std::panic::catch_unwind(|| normalize(&[valid, invalid], 0.5, 64, |_, _| true))
                .is_err()
        );
    }
    for sigma in [-0.1, f64::NAN, f64::INFINITY] {
        assert!(
            std::panic::catch_unwind(|| normalize(&[valid; 2], sigma, 64, |_, _| true)).is_err()
        );
    }
}
