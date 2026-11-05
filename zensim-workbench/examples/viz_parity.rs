//! Independent native control, retained observer, and browser fixtures.
use std::io::Write;
fn main() {
    let bake = std::fs::read(std::env::args().nth(1).expect("A bake argument")).unwrap();
    let out = std::path::PathBuf::from(std::env::args().nth(2).expect("output directory"));
    std::fs::create_dir_all(&out).unwrap();
    let req = zensim::research::Request::for_bake_bytes(&bake).unwrap();
    let mut cases = Vec::new();
    for (w, h, padding, identical) in [
        (16, 19, 12, false),
        (65, 67, 20, false),
        (96, 80, 0, false),
        (97, 81, 8, true),
        (1024, 768, 16, false),
    ] {
        let stride = w * 4 + padding;
        let mut r = vec![173; stride * h];
        let mut d = r.clone();
        let mut rr = Vec::new();
        let mut dd = Vec::new();
        let px = |x: usize, y: usize| {
            let v = ((x * 7 + y * 13) % 200 + 20) as u8;
            [v, v / 2 + 40, 255 - v, 255]
        };
        for y in 0..h {
            for x in 0..w {
                let a = px(x, y);
                let b = px((x + 1).min(w - 1), y);
                let c = [
                    ((u16::from(a[0]) + u16::from(b[0])) / 2) as u8,
                    ((u16::from(a[1]) + u16::from(b[1])) / 2 + 2).min(255) as u8,
                    ((u16::from(a[2]) + u16::from(b[2])) / 2) as u8,
                    255,
                ];
                r[y * stride + x * 4..y * stride + x * 4 + 4].copy_from_slice(&a);
                d[y * stride + x * 4..y * stride + x * 4 + 4].copy_from_slice(&c);
                rr.push([a[0], a[1], a[2]]);
                dd.push([c[0], c[1], c[2]]);
            }
        }
        if identical {
            d.clone_from(&r);
            dd.clone_from(&rr);
        }
        let a = zensim::RgbSlice::new(&rr, w, h);
        let b = zensim::RgbSlice::new(&dd, w, h);
        let control = zensim::research::extract(&req, &a, &b).unwrap();
        let (observed, _) = zensim::research::extract_with_detectors(&bake, &a, &b).unwrap();
        assert_eq!(control.values().len(), observed.len());
        for (i, (a, b)) in control.values().iter().zip(&observed).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "pooled slot {i}");
        }
        let session =
            zensim_workbench::AtlasSession::from_rgba("A", &r, &d, w as u32, h as u32, stride)
                .unwrap();
        let summary: serde_json::Value = serde_json::from_str(&session.summary()).unwrap();
        let legacy: serde_json::Value = serde_json::from_str(
            &zensim_workbench::score_json("A", &r, &d, w as u32, h as u32, stride).unwrap(),
        )
        .unwrap();
        let legacy_score = legacy["values"][0]["value"].as_f64().unwrap();
        assert!(
            (summary["score"].as_f64().unwrap() - legacy_score).abs()
                <= 1e-6 * legacy_score.abs().max(1.0),
            "independent legacy score"
        );
        let maps: Vec<serde_json::Value> = req
            .want()
            .iter_slots()
            .filter(|&id| w < 512 || [0, 13, 39, 143, 372, 393, 398, 709].contains(&id))
            .map(|id| serde_json::from_str(&session.detector_json(id).unwrap()).unwrap())
            .collect();
        let mut basic = Vec::new();
        for id in (0..156).step_by(13) {
            let map: serde_json::Value =
                serde_json::from_str(&session.detector_json(id).unwrap()).unwrap();
            let (mw, mh) = (
                map["width"].as_u64().unwrap() as usize,
                map["height"].as_u64().unwrap() as usize,
            );
            for (x, y) in [(0, 0), (mw / 2, mh / 2), (mw - 1, mh - 1)] {
                basic.push(serde_json::json!({"id":id,"terms":serde_json::from_str::<serde_json::Value>(&session.basic(id,x,y).unwrap()).unwrap()}));
            }
        }
        let reference = format!("r-{w}-{h}.rgba");
        let distorted = format!("d-{w}-{h}.rgba");
        std::fs::write(out.join(&reference), r).unwrap();
        std::fs::write(out.join(&distorted), d).unwrap();
        cases.push(serde_json::json!({"width":w,"height":h,"stride":stride,"reference":reference,"distorted":distorted,
            "basic":basic,"identical":identical,"summary":summary,"legacy_score":legacy_score,"maps":maps,
            "native_pooled_bit_checks":control.values().len(),"native_pooled_bit_mismatches":0}));
    }
    let mut f = std::fs::File::create(out.join("native.json")).unwrap();
    f.write_all(serde_json::to_string(&cases).unwrap().as_bytes())
        .unwrap();
    println!(
        "native: 5 dimension/stride/identity cases, every pooled bit identical, every consumed detector exported"
    );
}
