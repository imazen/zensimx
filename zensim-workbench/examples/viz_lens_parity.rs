//! Exact native forward and statistics for both compact browser banks.
fn main() {
    let mut args = std::env::args().skip(1);
    let root = std::path::PathBuf::from(args.next().expect("bank root"));
    let out = args.next().expect("output JSON");
    let spec = std::env::var("ZENSIM_WORKBENCH_BAKES").expect("frozen A bake");
    let path = spec.strip_prefix("A=").expect("one A bake");
    let model = zenpredict::Model::from_bytes(&std::fs::read(path).unwrap()).unwrap();
    let mut direct = zensim::BakeScorer::new(&model).unwrap();
    let mut result = serde_json::Map::new();
    for name in ["sdr", "hdr"] {
        let dir = root.join(name);
        let header: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("bank.json")).unwrap()).unwrap();
        let slots: Vec<u32> = serde_json::from_value(header["slots"].clone()).unwrap();
        let floats = |file: &str| {
            let data = std::fs::read(dir.join(file)).unwrap();
            assert_eq!(data.len() % 4, 0, "truncated f32 bank");
            data.as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect::<Vec<_>>()
        };
        let sample = floats("sample.f32le");
        let grids = floats("quantiles.f32le");
        let session = zensim_workbench::LensSession::from_sample("A", &slots, &sample).unwrap();
        let mut curves = Vec::new();
        let mut diagnostics = Vec::new();
        for id in [0, 3, 9, 143, 372, 393, 398, 709, 712] {
            let index = slots
                .iter()
                .position(|&s| s == id)
                .expect("registered selected slot");
            diagnostics.push(
                serde_json::from_str::<serde_json::Value>(
                    &session
                        .diagnostics_json(id as usize, &[id as usize])
                        .unwrap(),
                )
                .unwrap(),
            );
            curves.push(
                serde_json::from_str::<serde_json::Value>(
                    &session
                        .curve_json(id as usize, &grids[index * 21..index * 21 + 21])
                        .unwrap(),
                )
                .unwrap(),
            );
        }
        let mut direct_checks = 0;
        for curve in &curves {
            let id = curve["raw_feature"].as_u64().unwrap() as usize;
            let index = slots.iter().position(|&s| s as usize == id).unwrap();
            for row in [0, 99, 199] {
                for point in [0, 10, 20] {
                    let mut raw = vec![0.0; slots.iter().max().copied().unwrap() as usize + 1];
                    for (column, &slot) in slots.iter().enumerate() {
                        raw[slot as usize] = f64::from(sample[row * slots.len() + column]);
                    }
                    raw[id] = f64::from(grids[index * 21 + point]);
                    let expected = direct.score_features(&raw, 512, 512, None).unwrap();
                    let actual = curve["ice"][row][point].as_f64().unwrap();
                    assert!(
                        (actual - expected).abs() <= 1e-12 * expected.abs().max(1.0),
                        "independent direct score: {name}/f{id}/row{row}/point{point}"
                    );
                    direct_checks += 1;
                }
            }
        }
        let mut families = Vec::new();
        for (family, features) in [
            ("basic", (0..156).collect::<Vec<_>>()),
            (
                "v2",
                (372..720)
                    .filter(|id| (id - 372) % 29 != 21)
                    .collect::<Vec<_>>(),
            ),
        ] {
            families.push(serde_json::json!({"family":family,"features":features,
                "curve":serde_json::from_str::<serde_json::Value>(&session.family_json(&features).unwrap()).unwrap(),
                "diagnostics":serde_json::from_str::<serde_json::Value>(&session.diagnostics_json(features[0],&features).unwrap()).unwrap()}));
        }
        result.insert(
            name.into(),
            serde_json::json!({"header":header,"curves":curves,"diagnostics":diagnostics,"families":families,"independent_direct_checks":direct_checks}),
        );
    }
    std::fs::write(out, serde_json::to_vec(&result).unwrap()).unwrap();
    println!(
        "native lens: both banks, 18 raw-feature interventions, 200 × 21 scores each, centered and uncentered bands"
    );
}
