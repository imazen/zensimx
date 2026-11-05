//! Opt-in STEER research mode: complete native pixel repair forwards.
//! Ordinary serving and all existing map arithmetic remain unchanged.
use super::*;

pub(super) type Gains = Vec<([usize; 4], f64)>;

pub(super) fn selected() -> Result<bool, ZensimError> {
    match std::env::var("ZENSIM_STEER_EXACT") {
        Err(std::env::VarError::NotPresent) => Ok(false),
        Ok(value) if value == "off" => Ok(false),
        Ok(value) if value == "all-blocks-v1" => Ok(true),
        _ => Err(refuse("unknown STEER exact policy")),
    }
}
fn refuse(reason: &'static str) -> ZensimError {
    ZensimError::ModelForwardFailed { reason }
}

pub(super) fn admit(
    scorer: &BakeScorer<'_>,
    source: &impl ImageSource,
    bin: usize,
    hdr: bool,
) -> Result<(), ZensimError> {
    if hdr
        || source.is_hdr()
        || source.pixel_format() != crate::PixelFormat::Srgb8Rgb
        || source.color_primaries() != crate::ColorPrimaries::Srgb
        || bin != 1
        || !scorer.members.is_empty()
        || scorer.disposition.is_some()
        || scorer.plan()?.compute.formula_revision != crate::feature_defs::FormulaRevision::Rev5
    {
        return Err(refuse(
            "STEER exact requires a Rev5 isolated SDR sRGB RGB8 bake and bin1",
        ));
    }
    Ok(())
}

pub(super) fn gains(
    scorer: &mut BakeScorer<'_>,
    source: &impl ImageSource,
    distorted: &impl ImageSource,
    codec: Option<&str>,
    base: f64,
) -> Result<Gains, ZensimError> {
    if codec.is_some()
        || distorted.is_hdr()
        || distorted.pixel_format() != crate::PixelFormat::Srgb8Rgb
        || distorted.color_primaries() != crate::ColorPrimaries::Srgb
        || source.width() != distorted.width()
        || source.height() != distorted.height()
    {
        return Err(refuse(
            "STEER exact requires matching SDR RGB8 inputs without codec routing",
        ));
    }
    let (w, h) = (source.width(), source.height());
    let n = w
        .checked_mul(h)
        .ok_or_else(|| refuse("STEER exact geometry overflow"))?;
    let mut pixels: Vec<[u8; 3]> = Vec::new();
    pixels
        .try_reserve_exact(n)
        .map_err(|_| refuse("STEER exact pixel allocation failed"))?;
    for y in 0..h {
        let row = distorted.row_bytes(y);
        if row.len() < w * 3 || source.row_bytes(y).len() < w * 3 {
            return Err(refuse("STEER exact short pixel row"));
        }
        pixels.extend_from_slice(&crate::source::packed_row::<[u8; 3]>(row, w));
    }
    let count = [8usize, 16, 32, 64]
        .into_iter()
        .try_fold(0usize, |n, b| {
            w.div_ceil(b)
                .checked_mul(h.div_ceil(b))
                .and_then(|k| n.checked_add(k))
        })
        .ok_or_else(|| refuse("STEER exact inventory overflow"))?;
    let mut table = Vec::new();
    table
        .try_reserve_exact(count)
        .map_err(|_| refuse("STEER exact gain allocation failed"))?;
    for b in [8usize, 16, 32, 64] {
        for y0 in (0..h).step_by(b) {
            for x0 in (0..w).step_by(b) {
                let (x1, y1) = ((x0 + b).min(w), (y0 + b).min(h));
                for y in y0..y1 {
                    let row = crate::source::packed_row::<[u8; 3]>(source.row_bytes(y), w);
                    pixels[y * w + x0..y * w + x1].copy_from_slice(&row[x0..x1]);
                }
                // Full native extraction and the complete serving score, including
                // identity. Never reconstruct a repaired row by subtract/re-add.
                let repaired = crate::RgbSlice::try_new(&pixels, w, h)?;
                let score = scorer.compute(source, &repaired, None)?.score();
                let gain = score - base;
                if !gain.is_finite() {
                    return Err(refuse("STEER exact nonfinite native gain"));
                }
                table.push(([x0, y0, x1, y1], gain));
                for y in y0..y1 {
                    let row = crate::source::packed_row::<[u8; 3]>(distorted.row_bytes(y), w);
                    pixels[y * w + x0..y * w + x1].copy_from_slice(&row[x0..x1]);
                }
            }
        }
    }
    Ok(table)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::StridedBytes;

    #[test]
    fn steer_smooth_native_tier_stride_parity() {
        let mut receipts = Vec::new();
        for tier in ["native", "v3", "scalar"] {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "metric::bake::steer_exact::tests::r_exact_native_parity",
                    "--exact",
                    "--nocapture",
                ])
                .env("STEER_EXACT_TEST_CHILD", "1")
                .env("STEER_SMOOTH_TEST_TIER", tier)
                .env("ZENSIM_FORMULA_REV", "5")
                .env("ZENSIM_STEER_EXACT", "all-blocks-v1")
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            let stdout = String::from_utf8(output.stdout).unwrap();
            // Serial libtest writes its test banner before uncaptured stdout.
            // The receipt can therefore share that line rather than start it.
            let mut bit_receipts = stdout.lines().filter_map(|line| {
                line.split_once("STEER_SMOOTH_BITS ")
                    .map(|(_, receipt)| receipt)
            });
            let receipt = bit_receipts
                .next()
                .unwrap_or_else(|| panic!("child omitted its steering receipt:\n{stdout}"))
                .to_string();
            assert!(
                bit_receipts.next().is_none(),
                "child printed multiple steering receipts:\n{stdout}"
            );
            receipts.push(receipt);
            println!("STEER_SMOOTH_NATIVE tier={tier} rectangles=91 stride=true identity=true");
        }
        assert_eq!(receipts[0], receipts[1], "native versus v3 score/features");
        assert_eq!(
            receipts[0], receipts[2],
            "native versus scalar score/features"
        );
    }

    #[test]
    fn r_exact_native_parity() {
        const PATH: &str = "metric::bake::steer_exact::tests::r_exact_native_parity";
        if std::env::var("STEER_EXACT_TEST_CHILD").as_deref() != Ok("1") {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([PATH, "--exact", "--nocapture"])
                .env("STEER_EXACT_TEST_CHILD", "1")
                .env("ZENSIM_FORMULA_REV", "5")
                .env("ZENSIM_STEER_EXACT", "all-blocks-v1")
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            let stdout = String::from_utf8_lossy(&output.stdout);
            assert!(stdout.contains("STEER_EXACT_NATIVE_PARITY"));
            print!("{stdout}");
            return;
        }
        let smooth = std::env::var("STEER_SMOOTH_TEST_TIER").ok();
        #[cfg(target_arch = "x86_64")]
        match smooth.as_deref() {
            Some("v3") => {
                archmage::X64V4Token::dangerously_disable_token_process_wide(true).unwrap();
            }
            Some("scalar") => {
                archmage::X64V2Token::dangerously_disable_token_process_wide(true).unwrap();
            }
            _ => (),
        }
        let json = r#"{
            "schema_hash":1,"scaler_mean":[0,0],"scaler_scale":[1,1],
            "metadata":[{"key":"zentrain.feature_ids","type":"utf8","text":"13 459"},{"key":"zentrain.formula_revision","type":"utf8","text":"5"}],
            "layers":[{"in_dim":2,"out_dim":2,"activation":"relu","dtype":"f16","weights":[0.5,-0.25,-0.125,0.5],"biases":[0,0]},
            {"in_dim":2,"out_dim":1,"activation":"identity","dtype":"f16","weights":[-0.5,-0.25],"biases":[100]}] }"#;
        let json = if smooth.is_some() {
            json.replace("\"relu\"", "\"smooth_hinge\"")
        } else {
            json.to_string()
        };
        let bytes = zenpredict_bake::bake_from_json_str(&json).unwrap();
        let model = crate::mlp::Model::from_bytes(&bytes).unwrap();
        let (w, h) = (65, 49);
        let r: Vec<[u8; 3]> = (0..w * h)
            .map(|i| {
                [
                    (i * 17 % 251) as u8,
                    (i * 31 % 251) as u8,
                    (i * 7 % 251) as u8,
                ]
            })
            .collect();
        let d: Vec<[u8; 3]> = r
            .iter()
            .enumerate()
            .map(|(i, p)| {
                if i % 7 == 0 {
                    [p[0] / 2, p[1] / 2, p[2] / 2]
                } else {
                    *p
                }
            })
            .collect();
        let source = crate::RgbSlice::new(&r, w, h);
        let distorted = crate::RgbSlice::new(&d, w, h);
        let mut scorer = BakeScorer::new(&model).unwrap().with_parallel(false);
        assert!(scorer.prepare_steering(&source, 8).is_err());
        let mapped = scorer
            .prepare_steering(&source, 1)
            .unwrap()
            .compute(&distorted, None)
            .unwrap();
        let mut oracle = BakeScorer::new(&model).unwrap().with_parallel(false);
        let base = oracle.compute(&source, &distorted, None).unwrap();
        assert_eq!(mapped.result().score().to_bits(), base.score().to_bits());
        assert_eq!(mapped.result().features(), base.features());
        if smooth.is_some() {
            println!(
                "STEER_SMOOTH_BITS {:?}",
                (
                    base.score().to_bits(),
                    base.features()
                        .iter()
                        .map(|f| f.to_bits())
                        .collect::<Vec<_>>()
                )
            );
        }
        let stride = w * 3 + 11;
        let mut rp = vec![239; stride * h];
        let mut dp = rp.clone();
        for y in 0..h {
            rp[y * stride..y * stride + w * 3]
                .copy_from_slice(bytemuck::cast_slice(&r[y * w..(y + 1) * w]));
            dp[y * stride..y * stride + w * 3]
                .copy_from_slice(bytemuck::cast_slice(&d[y * w..(y + 1) * w]));
        }
        let sr = StridedBytes::new(&rp, w, h, stride, crate::PixelFormat::Srgb8Rgb);
        let sd = StridedBytes::new(&dp, w, h, stride, crate::PixelFormat::Srgb8Rgb);
        let mut padded = BakeScorer::new(&model).unwrap().with_parallel(false);
        let padded_map = padded
            .prepare_steering(&sr, 1)
            .unwrap()
            .compute(&sd, None)
            .unwrap();
        assert_eq!(padded_map.result().features(), base.features());
        assert_eq!(
            padded_map.result().score().to_bits(),
            base.score().to_bits()
        );
        let table = mapped.exact_refinement.as_ref().unwrap();
        assert_eq!(table.len(), 91, "complete four-grid inventory for 65x49");
        // Different grids can produce the same clipped edge rectangle. Check
        // the full multiset, retaining each grid's required occurrence.
        let mut required = Vec::new();
        for side in [8usize, 16, 32, 64] {
            for row in 0..h.div_ceil(side) {
                for column in 0..w.div_ceil(side) {
                    required.push([
                        column * side,
                        row * side,
                        ((column + 1) * side).min(w),
                        ((row + 1) * side).min(h),
                    ]);
                }
            }
        }
        let mut observed: Vec<_> = table.iter().map(|(rect, _)| *rect).collect();
        observed.sort_unstable();
        required.sort_unstable();
        assert_eq!(observed, required, "complete four-grid rectangle multiset");
        for &(rect, gain) in table {
            let [x0, y0, x1, y1] = rect;
            let mut edit = d.clone();
            for y in y0..y1 {
                edit[y * w + x0..y * w + x1].copy_from_slice(&r[y * w + x0..y * w + x1]);
            }
            let truth = oracle
                .compute(&source, &crate::RgbSlice::new(&edit, w, h), None)
                .unwrap()
                .score()
                - base.score();
            assert_eq!(gain.to_bits(), truth.to_bits(), "{rect:?}");
            assert_eq!(
                padded_map.refinement_gain(x0, y0, x1, y1).to_bits(),
                truth.to_bits()
            );
        }
        assert!(mapped.refinement_gain(1, 1, 9, 9).is_nan());
        let identity = scorer
            .prepare_steering(&source, 1)
            .unwrap()
            .compute(&source, None)
            .unwrap();
        assert_eq!(identity.result().score(), 100.);
        assert!(
            identity
                .exact_refinement
                .as_ref()
                .unwrap()
                .iter()
                .all(|(_, g)| *g == 0.)
        );
        let mut worker = scorer.prepare_steering(&source, 1).unwrap();
        assert!(worker.compute(&distorted, Some("jxl")).is_err());
        println!(
            "STEER_EXACT_NATIVE_PARITY rectangles={} strided=true identity=true unsupported_query=NaN",
            table.len()
        );
    }
}
