//! Two same-order reductions sharing the weight read and independent accumulators.

#[inline]
pub(super) fn dot_pair(weights: &[f64], a: &[f64], b: &[f64]) -> (f64, f64) {
    assert_eq!(weights.len(), a.len());
    assert_eq!(a.len(), b.len());
    // Iterator::sum's identity also preserves the sign of an all-negative-zero sum.
    let zero: f64 = std::iter::empty::<f64>().sum();
    let (mut ga, mut gb) = (zero, zero);
    for ((w, da), db) in weights.iter().zip(a).zip(b) {
        ga += w * da;
        gb += w * db;
    }
    (ga, gb)
}
