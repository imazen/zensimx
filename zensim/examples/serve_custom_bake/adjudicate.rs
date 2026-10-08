//! Explicit label-free packet audit through the existing serving owners.
//! `--adjudicate PACKET.json MODEL...`; one formula revision per process.
//! Cached inputs are f64 bit strings, preserving NaNs without JSON floats.
use serde::Deserialize;
use serde_json::json;
use zensim::research::{self, Request};
use zensim::{BakeScorer, RgbSlice};

#[derive(Deserialize)]
struct Case {
    key: String,
    ref_path: Option<String>,
    dist_path: Option<String>,
    repair_bounds: Option<[usize; 4]>,
    feature_bits: Option<Vec<u64>>,
}

pub(super) fn run(args: &[String]) {
    assert!(args.len() >= 2, "packet and explicit models required");
    let revision = std::env::var("ZENSIM_FORMULA_REV").expect("explicit revision");
    let cases: Vec<Case> =
        serde_json::from_slice(&std::fs::read(&args[0]).expect("read explicitly admitted packet"))
            .expect("parse packet");
    let mut models = Vec::new();
    for path in &args[1..] {
        let bytes = std::fs::read(path).expect("read pinned model");
        let model = zenpredict::Model::from_bytes(&bytes).expect("parse model");
        let declared = model.metadata().get_utf8("zentrain.formula_revision");
        assert_eq!(declared.unwrap_or("1"), revision);
        let request = Request::for_bake_bytes(&bytes).expect("canonical request");
        let mut scorer = BakeScorer::new(&model).unwrap().with_parallel(false);
        let ids = scorer.consumed_feature_ids().expect("complete read set");
        let mut raw = BakeScorer::new(&model)
            .unwrap()
            .with_parallel(false)
            .without_output_calibration();
        let spline = scorer.metadata().output_spline.as_ref().map(|s| {
            json!({"xs":s.xs,"ys":s.ys,"derivs":s.derivs,
                "lower_floor":2.0*s.ys[0]-s.ys[s.ys.len()-1]})
        });
        let zero = vec![0.0; model.caller_input_width().max(720)];
        let zero_score = scorer.score_features(&zero, 0, 0, None).unwrap();
        let mut rows = Vec::new();
        for (i, case) in cases.iter().enumerate() {
            if i % 500 == 0 {
                eprintln!("adjudicate {path}: {i}/{}", cases.len());
            }
            let stored = case
                .feature_bits
                .as_ref()
                .map(|bits| bits.iter().map(|&v| f64::from_bits(v)).collect::<Vec<_>>());
            let mut row = json!({"key":case.key});
            if let Some(features) = &stored {
                row["cached_score"] = json!(scorer.score_features(features, 0, 0, None).unwrap());
                row["uncalibrated_score"] =
                    json!(raw.score_features(features, 0, 0, None).unwrap());
            }
            if let Some(path) = &case.ref_path {
                let (pixels, w, h) = super::load_rgb(path);
                let source = RgbSlice::new(&pixels, w as usize, h as usize);
                let mut distorted = if let Some(path) = &case.dist_path {
                    let (dest, dw, dh) = super::load_rgb(path);
                    assert_eq!((w, h), (dw, dh));
                    dest
                } else {
                    pixels.clone()
                };
                if let Some([x0, y0, x1, y1]) = case.repair_bounds {
                    assert!(x0 < x1 && y0 < y1 && x1 <= w as usize && y1 <= h as usize);
                    for y in y0..y1 {
                        let range = y * w as usize + x0..y * w as usize + x1;
                        distorted[range.clone()].copy_from_slice(&pixels[range]);
                    }
                }
                let identical = pixels == distorted;
                let dest = RgbSlice::new(&distorted, w as usize, h as usize);
                let canonical = research::extract(&request, &source, &dest).unwrap();
                let features = canonical.values();
                let roundtrip: Vec<f64> = features.iter().map(|&x| f64::from(x as f32)).collect();
                let native = scorer.compute(&source, &dest, None).unwrap();
                row["width"] = json!(w);
                row["height"] = json!(h);
                row["pixel_score"] = json!(native.score());
                row["served_feature_score"] = json!(
                    scorer
                        .score_features(native.features(), w, h, None)
                        .unwrap()
                );
                row["served_vs_canonical_mismatch_ids"] = json!(
                    ids.iter()
                        .filter(|&&id| {
                            native.features()[usize::from(id)].to_bits()
                                != features[usize::from(id)].to_bits()
                        })
                        .copied()
                        .collect::<Vec<_>>()
                );
                row["native_feature_score"] =
                    json!(scorer.score_features(features, w, h, None).unwrap());
                row["native_f32_feature_score"] =
                    json!(scorer.score_features(&roundtrip, w, h, None).unwrap());
                row["native_uncalibrated_score"] =
                    json!(raw.score_features(features, w, h, None).unwrap());
                row["identity_aware_score"] = json!(
                    scorer
                        .score_features_with_identity(features, w, h, None, identical)
                        .unwrap()
                );
                row["consumed_native_bits"] = json!(
                    ids.iter()
                        .map(|&id| features[usize::from(id)].to_bits())
                        .collect::<Vec<_>>()
                );
                row["consumed_native_f32_bits"] = json!(
                    ids.iter()
                        .map(|&id| (features[usize::from(id)] as f32).to_bits())
                        .collect::<Vec<_>>()
                );
                row["nonzero_consumed"] = json!(
                    ids.iter()
                        .filter_map(|&id| {
                            let value = features[usize::from(id)];
                            (value != 0.0).then_some((id, value))
                        })
                        .collect::<Vec<_>>()
                );
                if let Some(stored) = &stored {
                    row["f64_mismatch_ids"] = json!(
                        ids.iter()
                            .filter(|&&id| {
                                stored[usize::from(id)].to_bits()
                                    != features[usize::from(id)].to_bits()
                            })
                            .copied()
                            .collect::<Vec<_>>()
                    );
                    row["f32_roundtrip_mismatch_ids"] = json!(
                        ids.iter()
                            .filter(|&&id| {
                                stored[usize::from(id)].to_bits()
                                    != roundtrip[usize::from(id)].to_bits()
                            })
                            .copied()
                            .collect::<Vec<_>>()
                    );
                }
            }
            rows.push(row);
        }
        models.push(json!({"path":path,"revision":revision,"consumed_ids":ids,
            "zero_score":zero_score,"output_spline":spline,"rows":rows}));
    }
    println!(
        "{}",
        json!({"schema":"adjudicate-native-v1","packet":args[0],"models":models})
    );
}
