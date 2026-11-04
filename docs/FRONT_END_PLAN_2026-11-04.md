# Front end plan (2026-11-04): one descriptor-driven ingestion for 8/10/12/16-bit and float, SDR and HDR

Owner, 2026-11-04: "What would a better front end be? We need native support for sdr and hdr samples > 8bit,
including 10 and 12 and 16bit integer and float." This is a design proposal. Nothing here is implemented, and
nothing below changes a shipped score unless it says so.

## Where the front end stands (verified in `zensim/src` on 2026-11-04)

Two front ends feed one feature pipeline (`FrontEnd::Sdr` and `FrontEnd::Hdr(HdrEncoding)` in
`feature_v2_stream.rs`); everything after the scale-0 planes is shared.

- **SDR.** `ImageSource` with `PixelFormat::{Srgb8Rgb, Srgb8Rgba, Srgb8Bgra, Srgb16Rgba, LinearF32Rgba}`. Opaque
  u8 sRGB goes through a captured LUT with magetypes SIMD (`color::srgb_to_positive_xyb_planar_into`) and is the
  byte-identical contract; u16 sRGB is normalised by 65535 and linear f32 goes through the f32 path
  (`linear_to_positive_xyb_planar_into`). Opsin, then cube-root XYB.
- **HDR.** `Zensim::compute_pu_linear`, `compute_pu_linear_planar` and `compute_hdr(…, HdrEncoding)` with
  `HdrEncoding::{Linear, Pq { peak_nits }, Hlg { peak_nits, ambient_lux }}`; containers `LinearF32Rgba` (absolute
  nits) or `Srgb16Rgba` (PQ or HLG code values over 65535). The display model is `transfer.rs` (spec constants,
  pinned to pycvvdp 0.5.4). Opsin, then PU21 per channel divided by PU21(100 cd/m²), then the opponent axes
  (`color::linear_to_pu_xyb_planar_into`). SDR entry points refuse sources whose `is_hdr()` is true.
- **Precision.** Rows and planes are f32 after decode (`hdr_row: Vec<[f32; 3]>`); nothing is re-quantised to
  eight bits. The `pixels_identical` shortcut compares decoded bytes, at any depth.
- **zenpixels bridge** (`zenpixels_compat.rs`): Rgb8, Rgba8, Rgbx8, Bgra8, Bgrx8, Rgba16 (sRGB) and RgbaF32
  (linear) only; grey, F16, Rgb16 without alpha, non-sRGB encoded transfers, narrow range and non-BT.709
  primaries are refused on the SDR path.

Gaps against the requirement: no notion of significant bits, so a 10- or 12-bit value in a u16 container reads as
65535-scaled; no f16; no grey; no RGB16 without alpha; no BT.709 or gamma-2.2 encoded transfers; no narrow range;
PQ and HLG only through the 16-bit RGBA container; two public entry points with different signatures; `is_hdr` as a
boolean side channel instead of a declared transfer and luminance frame.

What the ecosystem already offers, to reuse rather than rewrite:

- **zenpixels 0.2.16** describes Rgb8, Rgba8, Rgb16, Rgba16, RgbF16, RgbaF16, RgbF32, RgbaF32 and the grey, Bgra8,
  Rgbx8 and Bgrx8 variants; transfers Linear, Srgb, Bt709, Pq, Gamma22, Hlg; primaries Bt709, Bt2020, DisplayP3,
  AdobeRgb; `SignalRange::{Full, Narrow}`; alpha modes; and a `ColorContext` with ICC, CICP and `diffuse_white`
  (the cd/m² that relative-linear 1.0 represents). It has no significant-bits field.
- **fast-ssim2 0.9.1** solved the same requirement for SSIMULACRA2: one API where the descriptor decides, the
  captured u8 LUT for bit parity, u16 and f32 through the sRGB polynomial, native HDR for PQ and HLG in u8, u16
  and f32 and for linear f32 with BT.709, BT.2020 or Display P3 primaries and a diffuse-white anchor, keeping the
  reference's cube-root XYB for HDR. Its `source.rs` funnel (layouts, alpha, narrow range, gamut matrices, EOTFs)
  is the port target. On UPIQ it reads 0.765 pooled (our 2026-11-04 run) against the revision-7 PU-XYB models'
  0.775 to 0.792, so the anchored cube-root encoder is competitive without PU21, though the comparison crosses
  metrics.
- **zenpixels-convert** does transfer-aware conversion, gamut mapping and format negotiation for anything the front
  end should not take natively (packed 10-bit layouts such as P010 or v210, CMYK, ICC-described sources).

## Proposal

### 1. One front door: `zenpixels::PixelSlice` with its descriptor and colour context

`PixelSlice` plus `PixelDescriptor` plus `ColorContext` becomes the single ingestion type for SDR and HDR, for the
full-image and the streaming entry. `ImageSource` and `PixelFormat` stay as adapters behind it so existing callers
keep working. The descriptor carries depth and layout, transfer, primaries, signal range and alpha mode; the context
carries the luminance anchor and ICC/CICP.

Significant bits are missing from zenpixels: add `PixelDescriptor::significant_bits` (10 or 12 in a u16 container,
16 by default; a decoder scales by `2^n − 1` and rejects values above it). zenpixels is ours; this is a small,
additive change and goes through a pull request. Packed 10-bit layouts are not taken natively; zenpixels-convert
unpacks them to u16 first.

### 2. Decode stage: descriptor to f32 linear light in a declared luminance frame

One module (`frontend/decode.rs`) turns any accepted descriptor into planar f32 linear RGB rows plus a luminance
frame, row by row so the strip producer streams as it does today:

- Integer code values become f32 by their significant-bit maximum. Opaque u8 sRGB keeps the captured LUT and the
  byte-identical contract. Everything else evaluates the transfer in f32: the IEC sRGB piecewise curve, the BT.709
  inverse OETF, gamma 2.2, PQ (ST 2084), HLG (BT.2100 inverse OETF and OOTF with peak and ambient) and linear. All
  of these except the u8 LUT exist already in `transfer.rs` or in fast-ssim2's funnel; the sRGB polynomial from
  fast-ssim2's SIMD path replaces per-sample `powf` for u16 and float input.
- f16 widens to f32 (zenpixels `SampleType::F16`); grey expands to three channels; narrow range expands to full
  before the transfer; alpha composites per the declared mode onto documented backgrounds; BT.2020 and Display P3
  convert to BT.709-linear through the exact matrices, keeping negative and over-range values (no clipping before
  the opsin), as fast-ssim2 does.
- The luminance frame is explicit: SDR sources are relative light with 1.0 at reference white (the default when
  nothing is signalled, documented as 100 cd/m², the value the PU path already normalises to); HDR sources are
  absolute nits (anchor 1.0 = 1 nit, the PQ and HLG decodes, or a signalled diffuse white).
- SIMD through magetypes tokens for every EOTF; `#[rite]` helpers, one `#[arcane]` entry per row chunk; the u8 LUT
  path is untouched.

### 3. Encoder stage: chosen by regime, scores unchanged

- **SDR regime** (transfer sRGB, BT.709, gamma 2.2 or relative linear, and no anchor above the SDR reference):
  cube-root XYB as today. u8 sRGB stays byte-identical. Higher-depth SDR evaluates the same mathematics in f32, so a
  16-bit image that holds 8-bit values scores within the LUT's f32 rounding of the u8 path; that distance is
  measured and stated in the parity record, not assumed.
- **HDR regime** (PQ, HLG, absolute-nits linear, or a signalled diffuse white above the SDR reference): PU21-XYB
  divided by PU21(100) as today, the path validated on UPIQ.
- The regime is decided in one function from the descriptor and the frame, never from a side flag, and is reported
  in the result so a caller can see which path scored the pair. Feeding HDR into the SDR mathematics becomes
  impossible by construction.

### 4. Later, as a revision-8 era break: one encoder

Two candidates, to be measured with the models we have before any training decision: (a) PU21-XYB for everything,
SDR at a nominal white (every SDR score changes; a retrain follows); (b) the anchored cube-root encoder for
everything, fast-ssim2's native approach (SDR unchanged by construction when the anchor is the SDR white; HDR scores
change). What (b) costs zensim against the PU path is not measured: the experiment is one day, extracting the UPIQ,
4Kdtb and HDRTEACH banks through the anchored cube-root front end, forwarding the adopted mix, and comparing with
the PU banks' 0.783 / 0.884 / 0.945 (pooled, Narwaria, Korshunov). Neither candidate is needed for the depth and
format support above.

### 5. Tests and contracts

- The revision-7 golden scores (`tests/profiles.rs`) and the split parity run against the research crate stay
  unchanged for u8 sRGB.
- Depth equivalence: 16-, 12- and 10-bit integer and f16 and f32 encodings of one 8-bit sRGB image score within a
  stated tolerance of the u8 path (expected at the 1e-3 level from LUT rounding; measured, not assumed), and
  exactly equal to each other where the values are exactly representable.
- SIMD-tier bit identity (v4x, v4, v3, scalar) and strip-versus-whole parity on every decode path; PQ and HLG
  golden constants (already in `transfer.rs`); BT.2020 and P3 matrix goldens; narrow-range and alpha-background
  goldens; typed errors for unsupported descriptors, never a silent fallback.
- A parity record under `benchmarks/` with the measured distances, by depth and transfer.

### 6. Work plan

1. zenpixels: `significant_bits` on the descriptor (pull request, additive).
2. zensim: the decode stage, ported from fast-ssim2's funnel and `transfer.rs`, behind the existing `ImageSource`
   adapters; `zenpixels_compat` widened to every layout above; no public signature changes yet.
3. zensim: regime selection and a `compute` entry taking `PixelSlice` for both SDR and HDR, with
   `compute_pu_linear` and `compute_hdr` kept as thin wrappers (pull request; `CHANGELOG` entry).
4. The tests and the parity record.
5. zenmetrics: route zensim's HDR feeding through the descriptor path, and move its SSIMULACRA2 HDR feeding off the
   PU21 path that fast-ssim2 0.9.1 deprecates (its `hdr_feeding` table still names `IntegratedPuNits`).
6. Optional, after an owner decision: the single-encoder experiment of section 4 and a revision 8.

### 7. What this does not fix

No shipped model trains on HDR rows, so HDR calibration of the dial and JND output is untested; luminance-dependent
contrast sensitivity beyond PU21's encoding is not modelled (HDR_PLAN chunk 3); and 10- and 12-bit content is scored
through the same features as 8-bit content, so banding and dither at those depths are seen only through them. The
front end widens what zensim can ingest correctly; it does not by itself make the models better on that input.
