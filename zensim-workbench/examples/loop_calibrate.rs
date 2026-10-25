//! Fit each metric map's placement on butteraugli's scale for jxl-encoder's
//! loop, on the encodes zenmetrics' `loop_norm_calibrate --save-pairs` wrote,
//! by zenmetrics' documented method (`crates/zenmetrics-browser/examples/
//! loop_norm_calibrate.rs`): per 8×8 block, the map's local error (the
//! module's `loop_map_values` at scale 1, exponent 1) raised to `exponent`
//! and reduced by the 16-norm × 1.2 tile norm; for each exponent the scale is
//! the mean log ratio to butteraugli's block distance over blocks where both
//! are positive and butteraugli's is at least 0.05; the exponent with the
//! smallest log RMSE wins.
//!
//! ```text
//! ZENSIM_WORKBENCH_BAKES=... cargo run --release --example loop_calibrate -- PAIRS_DIR OUT.tsv
//! ```
//!
//! `LOOP_CAL_TAIL` adds a fit to each pair's worst tiles (largest and 99th
//! percentile; `OUT.tail.tsv`), the blocks that decide whether the loop
//! reaches its target. `LOOP_CAL_MAPS=a,b` fits only those maps.
//! `LOOP_CAL_LINEAR=DIR` scores each pair's distorted image from
//! `DIR/IDX.dist.linear.f32` (interleaved linear-light RGB `f32`, the encoder's
//! unrounded reconstruction, as zenmetrics' `margarine-loop-replay` writes it)
//! through `loop_map_values_linear`, the input `steer_rd --linear` gives the
//! loop, instead of the saved sRGB8 rounding.
//!
//! Fit: the CID22-512 *training* images. Held out: the gb82-sc screenshots.
//! The pair set's CID22-512 validation images are left out entirely (they are
//! a protected population in zensim's data splits).

use std::io::Write as _;

const EXPONENTS: [f64; 22] = [
    0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75,
    2.0, 2.5, 3.0, 4.0,
];
const FLOOR: f64 = 0.05;

/// One pair: its set (`fit` or `hold`), the map's local error at scale 1 and
/// exponent 1, butteraugli's block distances, width and height.
type Pair<'a> = (&'a str, Vec<f32>, Vec<f32>, usize, usize, f64);
const K_TILE_NORM: f64 = 1.2;

fn tile_norms(values: &[f32], width: usize, height: usize, exponent: f64) -> Vec<f64> {
    let (bx, by) = (width.div_ceil(8), height.div_ceil(8));
    let mut sum = vec![0f64; bx * by];
    let mut count = vec![0u32; bx * by];
    for y in 0..height {
        for x in 0..width {
            let b = (y / 8) * bx + x / 8;
            sum[b] += (f64::from(values[y * width + x]).max(0.0).powf(exponent)).powi(16);
            count[b] += 1;
        }
    }
    sum.iter()
        .zip(&count)
        .map(|(&s, &n)| K_TILE_NORM * (s / f64::from(n)).powf(1.0 / 16.0))
        .collect()
}

/// `(log RMSE, Pearson r of logs, blocks)` of `scale · m` against `b`.
fn fit_stats(pairs: &[(Vec<f64>, Vec<f32>)], scale: f64) -> (f64, f64, usize) {
    let (mut xs, mut ys) = (Vec::new(), Vec::new());
    for (m, b) in pairs {
        for (&mv, &bv) in m.iter().zip(b) {
            if mv > 0.0 && f64::from(bv) >= FLOOR {
                xs.push((scale * mv).ln());
                ys.push(f64::from(bv).ln());
            }
        }
    }
    let n = xs.len();
    if n < 2 {
        return (f64::NAN, f64::NAN, n);
    }
    let rmse = (xs
        .iter()
        .zip(&ys)
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        / n as f64)
        .sqrt();
    let (mx, my) = (
        xs.iter().sum::<f64>() / n as f64,
        ys.iter().sum::<f64>() / n as f64,
    );
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for (x, y) in xs.iter().zip(&ys) {
        sxy += (x - mx) * (y - my);
        sxx += (x - mx).powi(2);
        syy += (y - my).powi(2);
    }
    (rmse, sxy / (sxx * syy).sqrt(), n)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = std::path::PathBuf::from(args.next().expect("PAIRS_DIR"));
    let out = std::path::PathBuf::from(args.next().expect("OUT.tsv"));
    let index = std::fs::read_to_string(dir.join("pairs.tsv")).expect("pairs.tsv");
    let mut rows = Vec::new();
    for line in index.lines().skip(1) {
        let f: Vec<&str> = line.split('\t').collect();
        let (idx, image) = (f[0].to_string(), f[2].to_string());
        let set = if image.contains("CID22-512/training/") {
            "fit"
        } else if image.starts_with("gb82-sc/") {
            "hold"
        } else {
            continue; // CID22-512 validation: protected, never read
        };
        let (w, h): (usize, usize) = (f[4].parse().expect("width"), f[5].parse().expect("height"));
        let distance: f64 = f[3].parse().expect("distance");
        rows.push((idx, set, w, h, distance));
    }
    let read = |p: std::path::PathBuf| {
        std::fs::read(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
    };
    let mut tsv = String::from(
        "metric\tmap\tscale\texponent\tfit_log_rmse\tfit_log_r\tfit_blocks\thold_log_rmse\thold_log_r\thold_blocks\n",
    );
    let mut tail_tsv = String::from(
        "metric\tmap\ttail\tscale\texponent\tfit_log_rmse\tfit_log_r\tfit_pairs\thold_log_rmse\thold_log_r\thold_pairs\n",
    );
    let metrics: Vec<String> =
        serde_json::from_str(&zensim_workbench::available_metrics()).expect("metrics");
    // LOOP_CAL_MAPS=a,b: only those maps.
    let only_maps: Option<Vec<String>> = std::env::var("LOOP_CAL_MAPS")
        .ok()
        .map(|v| v.split(',').map(str::to_string).collect());
    let linear_dir = std::env::var_os("LOOP_CAL_LINEAR").map(std::path::PathBuf::from);
    for metric in &metrics {
        for map in zensim_workbench::map_names() {
            if only_maps.as_ref().is_some_and(|m| !m.contains(&map)) {
                continue;
            }
            // Per pair: the map's local error (scale 1, exponent 1) and butteraugli's block distances.
            let pairs: Vec<Pair> = rows
                .iter()
                .map(|(idx, set, w, h, distance)| {
                    let r = read(dir.join(format!("{idx}.ref.rgba")));
                    let (values, _) = match &linear_dir {
                        Some(ld) => {
                            let f: Vec<f32> = read(ld.join(format!("{idx}.dist.linear.f32")))
                                .as_chunks::<4>()
                                .0
                                .iter()
                                .map(|c| f32::from_le_bytes(*c))
                                .collect();
                            assert_eq!(f.len(), w * h * 3, "{idx}: linear reconstruction size");
                            let plane = |c: usize| {
                                f.iter().skip(c).step_by(3).copied().collect::<Vec<f32>>()
                            };
                            let (pr, pg, pb) = (plane(0), plane(1), plane(2));
                            zensim_workbench::loop_map_values_linear(
                                metric,
                                &map,
                                &r,
                                w * 4,
                                &pr,
                                &pg,
                                &pb,
                                *w,
                                *w as u32,
                                *h as u32,
                                1.0,
                                1.0,
                            )
                        }
                        None => {
                            let d = read(dir.join(format!("{idx}.dist.rgba")));
                            zensim_workbench::loop_map_values(
                                metric,
                                &map,
                                &r,
                                &d,
                                *w as u32,
                                *h as u32,
                                w * 4,
                                1.0,
                                1.0,
                            )
                        }
                    }
                    .expect("loop map");
                    let b: Vec<f32> = read(dir.join(format!("{idx}.butter.f32")))
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|c| f32::from_le_bytes(*c))
                        .collect();
                    (*set, values, b, *w, *h, *distance)
                })
                .collect();
            let mut best: Option<(f64, f64, f64)> = None;
            for &e in &EXPONENTS {
                let fit: Vec<(Vec<f64>, Vec<f32>)> = pairs
                    .iter()
                    .filter(|p| p.0 == "fit")
                    .map(|p| (tile_norms(&p.1, p.3, p.4, e), p.2.clone()))
                    .collect();
                let (mut s, mut n) = (0.0, 0usize);
                for (m, b) in &fit {
                    for (&mv, &bv) in m.iter().zip(b) {
                        if mv > 0.0 && f64::from(bv) >= FLOOR {
                            s += f64::from(bv).ln() - mv.ln();
                            n += 1;
                        }
                    }
                }
                if n == 0 {
                    continue;
                }
                let scale = (s / n as f64).exp();
                let (rmse, _, _) = fit_stats(&fit, scale);
                if best.is_none_or(|b| rmse < b.0) {
                    best = Some((rmse, scale, e));
                }
            }
            let Some((_, scale, e)) = best else {
                eprintln!("{metric}/{map}: no positive blocks");
                continue;
            };
            let set = |name: &str| -> Vec<(Vec<f64>, Vec<f32>)> {
                pairs
                    .iter()
                    .filter(|p| p.0 == name)
                    .map(|p| (tile_norms(&p.1, p.3, p.4, e), p.2.clone()))
                    .collect()
            };
            let (fr, fp, fnn) = fit_stats(&set("fit"), scale);
            let (hr, hp, hn) = fit_stats(&set("hold"), scale);
            // Per encoded distance: the mean log ratio of the placed block
            // distances to butteraugli's (positive: the placed map reads
            // worse than butteraugli, so the loop over-refines there), for
            // this fit and for the placement the module ships.
            // LOOP_CAL_PER_DISTANCE: the same fit separately at each encoded
            // distance (fit images), as `d<distance>=<scale>,<exponent>` pairs
            // ready for loop-converge's LOOP_PLACEMENT.
            if std::env::var_os("LOOP_CAL_PER_DISTANCE").is_some() {
                let mut ds: Vec<f64> = pairs.iter().map(|p| p.5).collect();
                ds.sort_by(f64::total_cmp);
                ds.dedup();
                let mut parts = Vec::new();
                for &d in &ds {
                    let mut best: Option<(f64, f64, f64)> = None;
                    for &ex in &EXPONENTS {
                        let fit: Vec<(Vec<f64>, Vec<f32>)> = pairs
                            .iter()
                            .filter(|p| p.0 == "fit" && p.5 == d)
                            .map(|p| (tile_norms(&p.1, p.3, p.4, ex), p.2.clone()))
                            .collect();
                        let (mut s, mut n) = (0.0, 0usize);
                        for (m, b) in &fit {
                            for (&mv, &bv) in m.iter().zip(b) {
                                if mv > 0.0 && f64::from(bv) >= FLOOR {
                                    s += f64::from(bv).ln() - mv.ln();
                                    n += 1;
                                }
                            }
                        }
                        if n == 0 {
                            continue;
                        }
                        let sc = (s / n as f64).exp();
                        let (rmse, _, _) = fit_stats(&fit, sc);
                        if best.is_none_or(|b| rmse < b.0) {
                            best = Some((rmse, sc, ex));
                        }
                    }
                    if let Some((rmse, sc, ex)) = best {
                        parts.push(format!("{d}={sc:.4e},{ex}"));
                        eprintln!(
                            "  per-distance d{d}: scale {sc:.4e} exponent {ex} fit rmse {rmse:.3}"
                        );
                    }
                }
                eprintln!("  LOOP_PLACEMENT={}", parts.join(";"));
            }
            if std::env::var_os("LOOP_CAL_BY_DISTANCE").is_some() {
                let shipped: Vec<serde_json::Value> =
                    serde_json::from_str(&zensim_workbench::loop_map_defaults()).expect("defaults");
                let shipped = shipped
                    .iter()
                    .find(|d| d["metric"] == metric.as_str() && d["map"] == map.as_str())
                    .map(|d| {
                        (
                            d["scale"].as_f64().unwrap(),
                            d["exponent"].as_f64().unwrap(),
                        )
                    });
                let mut ds: Vec<f64> = pairs.iter().map(|p| p.5).collect();
                ds.sort_by(f64::total_cmp);
                ds.dedup();
                for which in ["fit", "hold"] {
                    let mut line = format!("  {which} bias by distance (this fit");
                    if let Some((ss, se)) = shipped {
                        line += &format!(" | shipped {ss:.4e}/{se}");
                    }
                    line += "):";
                    for &d in &ds {
                        let bias = |sc: f64, ex: f64| -> (f64, usize) {
                            let (mut s, mut n) = (0.0, 0usize);
                            for p in pairs.iter().filter(|p| p.0 == which && p.5 == d) {
                                let m = tile_norms(&p.1, p.3, p.4, ex);
                                for (&mv, &bv) in m.iter().zip(&p.2) {
                                    if mv > 0.0 && f64::from(bv) >= FLOOR {
                                        s += (sc * mv).ln() - f64::from(bv).ln();
                                        n += 1;
                                    }
                                }
                            }
                            (if n > 0 { s / n as f64 } else { f64::NAN }, n)
                        };
                        let (b, n) = bias(scale, e);
                        line += &format!(" d{d} {b:+.2}");
                        if let Some((ss, se)) = shipped {
                            line += &format!("|{:+.2}", bias(ss, se).0);
                        }
                        line += &format!(" (n{n})");
                    }
                    eprintln!("{line}");
                }
            }
            // LOOP_CAL_TAIL: the same search, fitted to each pair's worst
            // tiles instead of every tile (the largest, and the 99th
            // percentile): the blocks that decide whether the loop reaches
            // its target and what its score reads. Bias by distance as
            // above, for this tail fit and for the bulk fit.
            if std::env::var_os("LOOP_CAL_TAIL").is_some() {
                let quantile = |v: &[f64], q: f64| -> f64 {
                    let mut s: Vec<f64> = v.to_vec();
                    s.sort_by(f64::total_cmp);
                    s[((s.len() - 1) as f64 * q).round() as usize]
                };
                for (stat, q) in [("max", 1.0), ("p99", 0.99)] {
                    let butter: Vec<f64> = pairs
                        .iter()
                        .map(|p| {
                            quantile(&p.2.iter().map(|&b| f64::from(b)).collect::<Vec<_>>(), q)
                        })
                        .collect();
                    // (set, distance, log butteraugli, log placed at scale 1) per pair.
                    let logs = |ex: f64| -> Vec<(&str, f64, f64, f64)> {
                        pairs
                            .iter()
                            .zip(&butter)
                            .filter_map(|(p, &b)| {
                                let m = quantile(&tile_norms(&p.1, p.3, p.4, ex), q);
                                (m > 0.0 && b >= FLOOR).then(|| (p.0, p.5, b.ln(), m.ln()))
                            })
                            .collect()
                    };
                    let stats =
                        |l: &[(&str, f64, f64, f64)], which: &str, ls: f64| -> (f64, f64, usize) {
                            let r: Vec<(f64, f64)> = l
                                .iter()
                                .filter(|x| x.0 == which)
                                .map(|x| (x.3 + ls, x.2))
                                .collect();
                            let n = r.len() as f64;
                            let rmse =
                                (r.iter().map(|(m, b)| (m - b).powi(2)).sum::<f64>() / n).sqrt();
                            let (mx, my) = (
                                r.iter().map(|x| x.0).sum::<f64>() / n,
                                r.iter().map(|x| x.1).sum::<f64>() / n,
                            );
                            let cov: f64 = r.iter().map(|(x, y)| (x - mx) * (y - my)).sum();
                            let vx: f64 = r.iter().map(|(x, _)| (x - mx).powi(2)).sum();
                            let vy: f64 = r.iter().map(|(_, y)| (y - my).powi(2)).sum();
                            (rmse, cov / (vx * vy).sqrt(), r.len())
                        };
                    let mut best: Option<(f64, f64, f64)> = None;
                    for &ex in &EXPONENTS {
                        let l = logs(ex);
                        let fit: Vec<f64> = l
                            .iter()
                            .filter(|x| x.0 == "fit")
                            .map(|x| x.2 - x.3)
                            .collect();
                        if fit.is_empty() {
                            continue;
                        }
                        let ls = fit.iter().sum::<f64>() / fit.len() as f64;
                        let (rmse, _, _) = stats(&l, "fit", ls);
                        if best.is_none_or(|b| rmse < b.0) {
                            best = Some((rmse, ls, ex));
                        }
                    }
                    let Some((_, ls, ex)) = best else { continue };
                    let l = logs(ex);
                    let (fr, fp, fnn) = stats(&l, "fit", ls);
                    let (hr, hp, hn) = stats(&l, "hold", ls);
                    eprintln!(
                        "{metric}/{map} tail {stat}: scale {:.4e} exponent {ex} · fit rmse {fr:.3} r {fp:.3} (n{fnn}) · held-out rmse {hr:.3} r {hp:.3} (n{hn})",
                        ls.exp()
                    );
                    tail_tsv.push_str(&format!(
                        "{metric}\t{map}\t{stat}\t{:.6e}\t{ex}\t{fr:.4}\t{fp:.4}\t{fnn}\t{hr:.4}\t{hp:.4}\t{hn}\n",
                        ls.exp()
                    ));
                    let lb = logs(e);
                    let mut ds: Vec<f64> = pairs.iter().map(|p| p.5).collect();
                    ds.sort_by(f64::total_cmp);
                    ds.dedup();
                    for which in ["fit", "hold"] {
                        let mut line =
                            format!("  {which} {stat} bias by distance (tail fit | bulk fit):");
                        for &d in &ds {
                            let bias = |l: &[(&str, f64, f64, f64)], lsc: f64| -> f64 {
                                let v: Vec<f64> = l
                                    .iter()
                                    .filter(|x| x.0 == which && x.1 == d)
                                    .map(|x| x.3 + lsc - x.2)
                                    .collect();
                                v.iter().sum::<f64>() / v.len() as f64
                            };
                            line += &format!(
                                " d{d} {:+.2}|{:+.2}",
                                bias(&l, ls),
                                bias(&lb, scale.ln())
                            );
                        }
                        eprintln!("{line}");
                    }
                }
            }
            tsv.push_str(&format!(
                "{metric}\t{map}\t{scale:.6e}\t{e}\t{fr:.4}\t{fp:.4}\t{fnn}\t{hr:.4}\t{hp:.4}\t{hn}\n"
            ));
            eprintln!(
                "{metric}/{map}: scale {scale:.4e} exponent {e} · fit rmse {fr:.3} r {fp:.3} · held-out rmse {hr:.3} r {hp:.3}"
            );
        }
    }
    std::fs::File::create(&out)
        .and_then(|mut f| f.write_all(tsv.as_bytes()))
        .expect("write tsv");
    if std::env::var_os("LOOP_CAL_TAIL").is_some() {
        std::fs::write(out.with_extension("tail.tsv"), tail_tsv).expect("write tail tsv");
    }
}
