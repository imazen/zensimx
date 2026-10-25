#![cfg(all(feature = "training", feature = "feature-regime-v2"))]
use archmage::testing::{CompileTimePolicy, for_each_token_permutation};
use zensim::{
    RgbSlice,
    feature_set_id::SlotSet,
    research::{self, Request},
};

#[test]
fn palette_tier_and_old_family_parity() {
    let _guard = archmage::testing::lock_token_testing();
    for (w, h) in [(1, 1), (17, 19), (64, 64), (97, 63), (131, 65)] {
        let src: Vec<_> = (0..w * h)
            .map(|i| [(i % 251) as u8, (i * 13 % 251) as u8, (i * 31 % 251) as u8])
            .collect();
        let dst: Vec<_> = src.iter().map(|p| [p[0] / 2, p[1], p[2]]).collect();
        let (s, d) = (RgbSlice::new(&src, w, h), RgbSlice::new(&dst, w, h));
        let palette = SlotSet::from_ranges([(1825, 1867)]);
        let req = Request::for_slots(palette.clone(), 1867)
            .with_era_label("palette_v2")
            .dense();
        let sparse_req = Request::for_slots(palette.clone(), 1867).with_era_label("palette_v2");
        assert_eq!(sparse_req.validate().unwrap(), palette);
        let sparse = research::extract(&sparse_req, &s, &d).unwrap();
        assert!(sparse.values()[..1825].iter().all(|v| v.to_bits() == 0));
        assert_eq!(
            sparse.feature_set_id().unwrap().compute().to_string(),
            "palette"
        );
        let base = research::extract(&req, &s, &d).unwrap();
        assert_eq!(base.values(), &sparse.values()[1825..]);
        assert_eq!(base.values().len(), 42);
        let fsid = base.feature_set_id().unwrap();
        assert!(fsid.compute().to_string() == "palette");
        assert_eq!(Request::for_set(fsid).unwrap().validate().unwrap(), palette);
        let report = for_each_token_permutation(CompileTimePolicy::Warn, |_| {
            let got = research::extract(&req, &s, &d).unwrap();
            assert_eq!(
                base.values()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                got.values().iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        });
        assert!(report.permutations_run >= 2);
        let old = SlotSet::from_ranges([(0, 228), (372, 720)]);
        let oldreq = Request::for_slots(old.clone(), 1825);
        let mixed = Request::for_slots(old.union(&palette), 1867);
        let oldv = research::extract(&oldreq, &s, &d).unwrap();
        let newv = research::extract(&mixed, &s, &d).unwrap();
        for i in 0..1825 {
            assert_eq!(
                oldv.values()[i].to_bits(),
                newv.values()[i].to_bits(),
                "old slot {i}"
            );
        }
        let identity = research::extract(&req, &s, &s).unwrap();
        assert!(identity.values().iter().all(|v| v.to_bits() == 0));
    }
}
#[test]
fn palette_respects_strided_rows() {
    use zensim::source::{ImageSource, PixelFormat};
    struct Rows {
        data: Vec<u8>,
        stride: usize,
    }
    impl ImageSource for Rows {
        fn width(&self) -> usize {
            17
        }
        fn height(&self) -> usize {
            19
        }
        fn alpha_mode(&self) -> zensim::source::AlphaMode {
            zensim::source::AlphaMode::Opaque
        }
        fn pixel_format(&self) -> PixelFormat {
            PixelFormat::Srgb8Rgb
        }
        fn row_bytes(&self, y: usize) -> &[u8] {
            &self.data[y * self.stride..y * self.stride + 17 * 3]
        }
    }
    let src: Vec<_> = (0..17 * 19)
        .map(|i| [(i % 251) as u8, (i * 13 % 251) as u8, (i * 31 % 251) as u8])
        .collect();
    let mut rows = Rows {
        data: vec![77; 19 * 64],
        stride: 64,
    };
    for y in 0..19 {
        for x in 0..17 {
            rows.data[y * 64 + x * 3..y * 64 + x * 3 + 3].copy_from_slice(&src[y * 17 + x]);
        }
    }
    let s = RgbSlice::new(&src, 17, 19);
    let req = Request::for_slots(SlotSet::from_ranges([(1825, 1867)]), 1867).dense();
    assert_eq!(
        research::extract(&req, &s, &rows).unwrap().values(),
        &[0.0; 42]
    );
}

#[test]
fn superseded_palette_revision_refuses() {
    let want = SlotSet::from_ranges([(1825, 1867)]);
    let old = Request::for_slots(want.clone(), 1867).with_era_label("palette_v1");
    assert!(matches!(
        old.validate(),
        Err(research::ResearchError::RevisionUnavailable { .. })
    ));
    let old_named = Request::for_slots(want, 1867)
        .at_revision(research::RevisionRef::Named("palette_v1".into()));
    assert!(matches!(
        old_named.validate(),
        Err(research::ResearchError::RevisionUnavailable { .. })
    ));
}

#[test]
fn private_palette_identity_preserves_supported_token_vocabulary() {
    use zensim::feature_set_id::{ComputeParts, ComputeToken, FeatureSetId};
    // 20 through gmsnative, plus globaff (formula revision 6).
    assert_eq!(ComputeToken::ALL.len(), 21);
    assert_eq!(ComputeToken::Gridblk as u32, 13);
    assert_eq!(ComputeToken::parse("palette"), None);
    let id = FeatureSetId::parse("palette@w1867/palette_v2#30b09cd1").unwrap();
    assert_eq!(id.compute().to_string(), "palette");
    assert_eq!(id.compute().iter().count(), 0);
    let slots = SlotSet::from_ranges([(1825, 1867)]);
    assert_eq!(Request::for_set(&id).unwrap().validate().unwrap(), slots);
    let supported = ComputeToken::ALL
        .iter()
        .copied()
        .fold(ComputeParts::EMPTY, ComputeParts::with);
    assert!(
        !supported
            .to_string()
            .split('+')
            .any(|name| name == "palette")
    );
    let mixed = ComputeParts::parse(&format!("{}+palette", supported)).unwrap();
    assert_eq!(
        mixed.iter().collect::<Vec<_>>(),
        supported.iter().collect::<Vec<_>>()
    );
    assert_eq!(ComputeParts::parse(&mixed.to_string()), Some(mixed));
}
