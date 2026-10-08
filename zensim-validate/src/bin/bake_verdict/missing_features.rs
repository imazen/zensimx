//! Equality for a cached feature layout with explicitly uncomputed NaN slots.
//! Consumed inputs must be finite before inference; this comparator preserves
//! missingness and the owner's existing finite-feature tolerance.
pub(super) fn near_equal_rows(a: &[f64], b: &[f64], epsilon: f64) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| {
            (x.is_nan() && y.is_nan())
                || (x.is_finite() && y.is_finite() && (x - y).abs() <= epsilon)
        })
}

#[cfg(test)]
mod tests {
    use super::near_equal_rows;

    #[test]
    fn matching_uncomputed_slots_preserve_codec_saturation() {
        assert!(near_equal_rows(
            &[f64::NAN, 1.0, 2.0],
            &[f64::NAN, 1.0, 2.0],
            1e-5
        ));
        assert!(!near_equal_rows(&[f64::NAN, 1.0], &[0.0, 1.0], 1e-5));
        assert!(!near_equal_rows(&[0.0, 1.0], &[f64::NAN, 1.0], 1e-5));
        assert!(!near_equal_rows(&[f64::INFINITY], &[f64::INFINITY], 1e-5));
    }

    #[test]
    fn finite_tolerance_and_layout_remain_unchanged() {
        assert!(near_equal_rows(
            &[0.0, f64::NAN],
            &[0.999e-5, f64::NAN],
            1e-5
        ));
        assert!(!near_equal_rows(
            &[0.0, f64::NAN],
            &[1.001e-5, f64::NAN],
            1e-5
        ));
        assert!(!near_equal_rows(&[1.0], &[1.0, f64::NAN], 1e-5));
    }
}
