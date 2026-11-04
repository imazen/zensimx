//! Pixel prefixes must be independent of trailing row bytes and byte alignment.
use crate::feature_defs::FormulaRevision;
use crate::source::{AlphaMode, ColorPrimaries, GamutMapping, ImageSource, PixelFormat};

const FORMATS: [PixelFormat; 5] = [
    PixelFormat::Srgb8Rgb,
    PixelFormat::Srgb8Rgba,
    PixelFormat::Srgb8Bgra,
    PixelFormat::Srgb16Rgba,
    PixelFormat::LinearF32Rgba,
];
const REVISIONS: [FormulaRevision; 5] = [
    FormulaRevision::Rev1,
    FormulaRevision::Rev2,
    FormulaRevision::Rev3,
    FormulaRevision::Rev4,
    FormulaRevision::Rev5,
];
const MODES: [(AlphaMode, ColorPrimaries, GamutMapping); 6] = [
    (AlphaMode::Opaque, ColorPrimaries::Srgb, GamutMapping::Clip),
    (
        AlphaMode::Straight,
        ColorPrimaries::Srgb,
        GamutMapping::Clip,
    ),
    (AlphaMode::Unknown, ColorPrimaries::Srgb, GamutMapping::Clip),
    (
        AlphaMode::Opaque,
        ColorPrimaries::DisplayP3,
        GamutMapping::Clip,
    ),
    (
        AlphaMode::Straight,
        ColorPrimaries::DisplayP3,
        GamutMapping::Clip,
    ),
    (
        AlphaMode::Straight,
        ColorPrimaries::Bt2020,
        GamutMapping::Preserve,
    ),
];

/// u128 backing gives the reference a guaranteed alignment. Byte offsets and
/// arbitrary strides deliberately remove that guarantee from candidate rows.
struct Rows {
    storage: Vec<u128>,
    width: usize,
    height: usize,
    stride: usize,
    offset: usize,
    format: PixelFormat,
    alpha: AlphaMode,
    primaries: ColorPrimaries,
    gamut: GamutMapping,
    hdr: bool,
}

impl Rows {
    fn new(
        w: usize,
        h: usize,
        format: PixelFormat,
        padding: usize,
        offset: usize,
        seed: usize,
    ) -> Self {
        let bpp = format.bytes_per_pixel();
        let stride = w * bpp + padding;
        let mut storage = vec![0u128; (offset + h * stride).div_ceil(16)];
        let data: &mut [u8] = bytemuck::cast_slice_mut(&mut storage);
        data.fill(0xa5);
        for y in 0..h {
            for x in 0..w {
                let i = x + y * w + seed;
                let v = [i * 37 % 256, i * 19 % 256, i * 97 % 256, i * 53 % 256];
                let pixel = &mut data[offset + y * stride + x * bpp..][..bpp];
                match format {
                    PixelFormat::Srgb8Rgb => {
                        pixel.copy_from_slice(&[v[0] as u8, v[1] as u8, v[2] as u8])
                    }
                    PixelFormat::Srgb8Rgba => pixel.copy_from_slice(&v.map(|c| c as u8)),
                    PixelFormat::Srgb8Bgra => {
                        pixel.copy_from_slice(&[v[2] as u8, v[1] as u8, v[0] as u8, v[3] as u8])
                    }
                    PixelFormat::Srgb16Rgba => {
                        for (dst, c) in pixel.as_chunks_mut::<2>().0.iter_mut().zip(v) {
                            dst.copy_from_slice(&((c * 257) as u16).to_ne_bytes());
                        }
                    }
                    PixelFormat::LinearF32Rgba => {
                        for (dst, c) in pixel.as_chunks_mut::<4>().0.iter_mut().zip(v) {
                            dst.copy_from_slice(&(c as f32 / 255.0).to_ne_bytes());
                        }
                    }
                }
            }
        }
        Self {
            storage,
            width: w,
            height: h,
            stride,
            offset,
            format,
            alpha: AlphaMode::Opaque,
            primaries: ColorPrimaries::Srgb,
            gamut: GamutMapping::Clip,
            hdr: false,
        }
    }
    fn mode(mut self, mode: (AlphaMode, ColorPrimaries, GamutMapping)) -> Self {
        (self.alpha, self.primaries, self.gamut) = mode;
        self
    }
}

impl ImageSource for Rows {
    fn width(&self) -> usize {
        self.width
    }
    fn height(&self) -> usize {
        self.height
    }
    fn pixel_format(&self) -> PixelFormat {
        self.format
    }
    fn alpha_mode(&self) -> AlphaMode {
        self.alpha
    }
    fn color_primaries(&self) -> ColorPrimaries {
        self.primaries
    }
    fn gamut_mapping(&self) -> GamutMapping {
        self.gamut
    }
    fn is_hdr(&self) -> bool {
        self.hdr
    }
    fn row_bytes(&self, y: usize) -> &[u8] {
        let bytes: &[u8] = bytemuck::cast_slice(&self.storage);
        let start = self.offset + y * self.stride;
        &bytes[start..start + self.stride]
    }
}

fn bits(planes: [Vec<f32>; 3]) -> [Vec<u32>; 3] {
    planes.map(|p| p.into_iter().map(f32::to_bits).collect())
}

fn chunked(
    source: &impl ImageSource,
    pw: usize,
    parallel: bool,
    chunk: usize,
    revision: FormulaRevision,
) -> [Vec<u32>; 3] {
    let mut out: [Vec<f32>; 3] = std::array::from_fn(|_| vec![0.0; pw * source.height()]);
    let [a, b, c] = &mut out;
    crate::streaming::convert_source_to_xyb_into_slices_chunked(
        source, a, b, c, pw, parallel, 13, chunk, revision,
    );
    bits(out)
}

#[test]
fn padded_rows_xyb_packed_formats() {
    let _tokens = archmage::testing::lock_token_testing();
    for format in FORMATS {
        for (w, h) in [(7, 5), (17, 9), (33, 65)] {
            for mode in MODES {
                let reference = Rows::new(w, h, format, 0, 0, 0).mode(mode);
                for padding in 0..=format.bytes_per_pixel() + 1 {
                    for offset in [0, 1] {
                        let candidate = Rows::new(w, h, format, padding, offset, 0).mode(mode);
                        for revision in REVISIONS {
                            for parallel in [false, true] {
                                for pw in [w, w + 5] {
                                    for chunk in [7, 64] {
                                        assert_eq!(
                                            chunked(&candidate, pw, parallel, chunk, revision),
                                            chunked(&reference, pw, parallel, chunk, revision),
                                            "{format:?} {w}x{h} {mode:?} pad={padding} offset={offset} {revision:?} parallel={parallel} pw={pw} chunk={chunk}"
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn padded_rows_native_sdr() {
    let _tokens = archmage::testing::lock_token_testing();
    for format in [PixelFormat::Srgb16Rgba, PixelFormat::LinearF32Rgba] {
        for (w, h) in [(7, 5), (33, 65)] {
            for mode in &MODES[..5] {
                let reference = Rows::new(w, h, format, 0, 0, 0).mode(*mode);
                let expected: Vec<_> = crate::streaming::native_sdr_linear_rgb(&reference)
                    .unwrap()
                    .into_iter()
                    .flatten()
                    .map(f32::to_bits)
                    .collect();
                for padding in 0..=format.bytes_per_pixel() + 1 {
                    for offset in [0, 1] {
                        let candidate = Rows::new(w, h, format, padding, offset, 0).mode(*mode);
                        let actual: Vec<_> = crate::streaming::native_sdr_linear_rgb(&candidate)
                            .unwrap()
                            .into_iter()
                            .flatten()
                            .map(f32::to_bits)
                            .collect();
                        assert_eq!(
                            actual, expected,
                            "{format:?} {w}x{h} {mode:?} pad={padding} offset={offset}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn padded_rows_hdr_frontend() {
    let _tokens = archmage::testing::lock_token_testing();
    use crate::feature_v2::HdrEncoding;
    for format in [PixelFormat::Srgb16Rgba, PixelFormat::LinearF32Rgba] {
        for primaries in [
            ColorPrimaries::Srgb,
            ColorPrimaries::DisplayP3,
            ColorPrimaries::Bt2020,
        ] {
            let mode = (AlphaMode::Opaque, primaries, GamutMapping::Clip);
            let (w, h) = (17, 9);
            let reference = Rows::new(w, h, format, 0, 0, 0).mode(mode);
            for padding in 0..=format.bytes_per_pixel() + 1 {
                for offset in [0, 1] {
                    let candidate = Rows::new(w, h, format, padding, offset, 0).mode(mode);
                    for encoding in [
                        HdrEncoding::Linear,
                        HdrEncoding::Pq { peak_nits: 1000.0 },
                        HdrEncoding::Hlg {
                            peak_nits: 1000.0,
                            ambient_lux: 5.0,
                        },
                    ] {
                        for revision in REVISIONS {
                            let convert = |source: &Rows| {
                                let mut out = std::array::from_fn(|_| vec![0.0; w * h]);
                                crate::feature_v2_stream::hdr_source_to_xyb(
                                    source, encoding, &mut out, revision,
                                );
                                bits(out)
                            };
                            assert_eq!(
                                convert(&candidate),
                                convert(&reference),
                                "{format:?} {primaries:?} {encoding:?} {revision:?} pad={padding} offset={offset}"
                            );
                        }
                    }
                }
            }
        }
    }
}

fn result_bits(r: crate::ZensimResult) -> (u64, u64, [u64; 3], Vec<u64>) {
    (
        r.score().to_bits(),
        r.raw_distance().to_bits(),
        r.mean_offset().map(f64::to_bits),
        r.features().iter().map(|v| v.to_bits()).collect(),
    )
}

#[test]
fn padded_rows_subset_and_strip_producer() {
    let _tokens = archmage::testing::lock_token_testing();
    use crate::feature_v2::{HdrEncoding, HdrFrontEnd};
    use crate::feature_v2_stream::{FrontEnd, Side, StripPlaneProducer};
    for format in FORMATS {
        for mode in [MODES[0], MODES[1], MODES[4]] {
            let reference = Rows::new(33, 65, format, 0, 0, 0).mode(mode);
            for padding in 0..=format.bytes_per_pixel() + 1 {
                for offset in [0, 1] {
                    let candidate = Rows::new(33, 65, format, padding, offset, 0).mode(mode);
                    let a = crate::source::SubsetView::new(&reference, 1, 63);
                    let b = crate::source::SubsetView::new(&candidate, 1, 63);
                    for revision in [FormulaRevision::Rev3, FormulaRevision::Rev5] {
                        assert_eq!(
                            chunked(&a, 40, false, 7, revision),
                            chunked(&b, 40, false, 7, revision)
                        );
                        let mut fronts = vec![FrontEnd::Sdr];
                        if matches!(format, PixelFormat::Srgb16Rgba | PixelFormat::LinearF32Rgba) {
                            fronts.extend([
                                FrontEnd::Hdr(HdrEncoding::Linear, HdrFrontEnd::Pu21),
                                FrontEnd::Hdr(
                                    HdrEncoding::Pq { peak_nits: 1000.0 },
                                    HdrFrontEnd::Pu21,
                                ),
                                FrontEnd::Hdr(
                                    HdrEncoding::Hlg {
                                        peak_nits: 1000.0,
                                        ambient_lux: 5.0,
                                    },
                                    HdrFrontEnd::Pu21,
                                ),
                            ]);
                        }
                        for front in fronts {
                            let produce = |source: &Rows| {
                                let mut pool = Vec::new();
                                let mut producer = StripPlaneProducer::new_with_ref_feed(
                                    source, source, false, &mut pool, front, None, None, revision,
                                    false,
                                );
                                let mut metadata = Vec::new();
                                let mut values = Vec::new();
                                while let Some(info) = producer.next_strip() {
                                    metadata.push((
                                        info.scale,
                                        info.y0,
                                        info.strip_h,
                                        info.plane_w,
                                        info.plane_h,
                                    ));
                                    for side in [Side::Source, Side::Distorted] {
                                        for ch in 0..3 {
                                            let mut window =
                                                vec![0.0; info.plane_w * info.wide_h()];
                                            producer.fill_wide(side, ch, &info, &mut window);
                                            values.extend(window.into_iter().map(f32::to_bits));
                                        }
                                    }
                                }
                                (metadata, values)
                            };
                            assert_eq!(
                                produce(&candidate),
                                produce(&reference),
                                "{format:?} {mode:?} {front:?} {revision:?} pad={padding} offset={offset}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn padded_rows_public_sdr_and_hdr() {
    let _tokens = archmage::testing::lock_token_testing();
    for format in FORMATS {
        let (w, h) = (131, 129);
        for parallel in [false, true] {
            let z = crate::Zensim::new(crate::ZensimProfile::B).with_parallel(parallel);
            let reference = Rows::new(w, h, format, 0, 0, 0);
            let distorted = Rows::new(w, h, format, 0, 0, 7);
            let expected = result_bits(z.compute(&reference, &distorted).unwrap());
            for padding in 0..=format.bytes_per_pixel() + 1 {
                for offset in [0, 1] {
                    let candidate = Rows::new(w, h, format, padding, offset, 0);
                    let changed = Rows::new(w, h, format, padding, offset, 7);
                    assert_eq!(
                        result_bits(z.compute(&candidate, &changed).unwrap()),
                        expected,
                        "SDR {format:?} pad={padding} offset={offset} parallel={parallel}"
                    );
                }
            }
        }
    }
    for parallel in [false, true] {
        let z = crate::Zensim::new(crate::ZensimProfile::B).with_parallel(parallel);
        let mut reference = Rows::new(17, 9, PixelFormat::LinearF32Rgba, 0, 0, 0);
        let mut distorted = Rows::new(17, 9, PixelFormat::LinearF32Rgba, 0, 0, 7);
        reference.hdr = true;
        distorted.hdr = true;
        let expected = result_bits(z.compute(&reference, &distorted).unwrap());
        for padding in 0..=17 {
            for offset in [0, 1] {
                let mut candidate =
                    Rows::new(17, 9, PixelFormat::LinearF32Rgba, padding, offset, 0);
                let mut changed = Rows::new(17, 9, PixelFormat::LinearF32Rgba, padding, offset, 7);
                candidate.hdr = true;
                changed.hdr = true;
                assert_eq!(
                    result_bits(z.compute(&candidate, &changed).unwrap()),
                    expected,
                    "HDR pad={padding} offset={offset} parallel={parallel}"
                );
            }
        }
    }
}

#[test]
fn padded_rows_indexed_readers() {
    let _tokens = archmage::testing::lock_token_testing();
    for format in FORMATS {
        let reference = Rows::new(17, 9, format, 0, 0, 0);
        let distorted = Rows::new(17, 9, format, 0, 0, 7);
        for padding in 0..=format.bytes_per_pixel() + 1 {
            for offset in [0, 1] {
                let candidate = Rows::new(17, 9, format, padding, offset, 0);
                let changed = Rows::new(17, 9, format, padding, offset, 7);
                assert!(crate::metric::images_byte_identical(&candidate, &reference));
                assert!(!crate::metric::images_byte_identical(
                    &candidate, &distorted
                ));
                let a = crate::metric::reflect_pad_to_size(&reference, 32);
                let b = crate::metric::reflect_pad_to_size(&candidate, 32);
                assert!(crate::metric::images_byte_identical(&a, &b));
                #[cfg(feature = "classification")]
                {
                    let a = crate::streaming::compute_delta_stats(&reference, &distorted).unwrap();
                    let b = crate::streaming::compute_delta_stats(&candidate, &changed).unwrap();
                    // Debug uses round-trip float formatting; every field is
                    // finite for these fixtures, so all bits remain observable.
                    assert_eq!(format!("{a:?}"), format!("{b:?}"));
                }
                if format == PixelFormat::Srgb8Rgb {
                    assert_eq!(
                        crate::palette::extract(&candidate, &changed)
                            .unwrap()
                            .map(f64::to_bits),
                        crate::palette::extract(&reference, &distorted)
                            .unwrap()
                            .map(f64::to_bits)
                    );
                }
            }
        }
    }
}
