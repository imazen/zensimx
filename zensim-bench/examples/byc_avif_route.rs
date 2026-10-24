//! By-construction exploration (`benchmarks/byconstruction_2026-10-22.md`, "AVIF decode drift"): decode each file
//! through the decode owner (`shared/zen_decode.rs`, the bank extractor's path) and project the native buffer to RGB8
//! three ways, printing the native descriptor and each route's pixel SHA-256 (the extractor audit's
//! `distorted_pixels_sha256` definition), so a route can be matched to an earlier build's audit hash.
//!
//! Routes: `owner` (the owner's projection: `RowConverter` to RGB8_SRGB), `relabel` (`RowConverter` from the native
//! descriptor relabelled with an unknown transfer to RGB8, i.e. depth only), `round` (16-bit samples to 8-bit by
//! `round(v · 255 / 65535)`, by hand).
//!
//!   byc_avif_route FILE...

#[path = "shared/zen_decode.rs"]
mod zen_decode;

use sha2::{Digest, Sha256};
use zenpixels::{PixelDescriptor, TransferFunction};
use zenpixels_convert::converter::RowConverter;

fn sha(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn convert(src: PixelDescriptor, dst: PixelDescriptor, data: &[u8], stride: usize, w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0u8; w * 3 * h];
    RowConverter::new(src, dst)
        .expect("plan")
        .convert_rows(data, stride, &mut out, w * 3, w as u32, h as u32)
        .expect("convert");
    out
}

fn main() {
    for path in std::env::args().skip(1) {
        let bytes = std::fs::read(&path).expect("read");
        let native = zen_decode::decode_native_bytes(&bytes, &path).expect("decode");
        let pb = &native.pixels;
        let desc = pb.descriptor();
        let (w, h) = (pb.width() as usize, pb.height() as usize);
        let slice = pb.as_slice();
        let (stride, data) = (slice.stride(), slice.as_strided_bytes());
        let owner = native.to_rgb8(&path).expect("project").pixels;
        println!("{path}\tdescriptor\t{desc:?}");
        println!("{path}\towner\t{}", sha(&owner));
        let relabel = convert(desc.with_transfer(TransferFunction::Unknown), PixelDescriptor::RGB8, data, stride, w, h);
        println!("{path}\trelabel\t{}", sha(&relabel));
        if desc.layout_compatible(PixelDescriptor::RGB16) {
            let mut round = Vec::with_capacity(w * 3 * h);
            for row in 0..h {
                let r = &data[row * stride..row * stride + w * 6];
                round.extend(r.as_chunks::<2>().0.iter().map(|b| {
                    let v = u32::from(u16::from_ne_bytes(*b));
                    ((v * 255 + 32767) / 65535) as u8
                }));
            }
            println!("{path}\tround\t{}", sha(&round));
        }
    }
}
