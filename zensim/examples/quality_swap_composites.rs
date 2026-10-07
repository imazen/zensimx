//! Compose registered block swaps using the shared zen PNG IO owner.
//! The caller supplies frozen ranked indices; this tool does no scoring.
#![forbid(unsafe_code)]
#[path = "support/zen_io.rs"]
mod zen_io;

use imgref::ImgRef;
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Job {
    low: PathBuf,
    high: PathBuf,
    output: PathBuf,
    bounds: Vec<[usize; 4]>,
    selected: Vec<usize>,
}

fn compose(
    low: ImgRef<'_, [u8; 3]>,
    high: ImgRef<'_, [u8; 3]>,
    bounds: &[[usize; 4]],
    selected: &[usize],
) -> Vec<[u8; 3]> {
    assert_eq!((low.width(), low.height()), (high.width(), high.height()));
    let w = low.width();
    let h = low.height();
    let mut out: Vec<_> = low.rows().flat_map(|row| row.iter().copied()).collect();
    for &i in selected {
        let [x0, y0, x1, y1] = bounds[i];
        assert!(
            x0 < x1 && x1 <= w && y0 < y1 && y1 <= h,
            "invalid block geometry"
        );
        for (y, row) in high.rows().enumerate().take(y1).skip(y0) {
            out[y * w + x0..y * w + x1].copy_from_slice(&row[x0..x1]);
        }
    }
    out
}

fn main() {
    let manifest = std::env::args().nth(1).expect("jobs JSON required");
    let jobs: Vec<Job> = serde_json::from_slice(&std::fs::read(manifest).unwrap()).unwrap();
    for job in jobs {
        assert!(!job.output.exists(), "fresh output required");
        let (low, w, h) = zen_io::decode_rgb8(&job.low);
        let (high, hw, hh) = zen_io::decode_rgb8(&job.high);
        assert_eq!((w, h), (hw, hh));
        let pixels = compose(
            ImgRef::new(&low, w, h),
            ImgRef::new(&high, w, h),
            &job.bounds,
            &job.selected,
        );
        let encoded = zen_io::encode_png_rgb8(&pixels, w, h);
        std::fs::write(&job.output, encoded).unwrap();
        let (roundtrip, rw, rh) = zen_io::decode_rgb8(&job.output);
        assert_eq!((rw, rh), (w, h));
        assert_eq!(roundtrip, pixels, "written PNG changes composite pixels");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn swaps_obey_stride_and_partial_edge_bounds() {
        let low = [[1, 2, 3]; 10];
        let high = [[4, 5, 6]; 12];
        let l = ImgRef::new_stride(&low, 3, 2, 5);
        let h = ImgRef::new_stride(&high, 3, 2, 6);
        let bounds = [[2, 0, 3, 2]];
        assert_eq!(
            compose(l, h, &bounds, &[0]),
            vec![
                [1, 2, 3],
                [1, 2, 3],
                [4, 5, 6],
                [1, 2, 3],
                [1, 2, 3],
                [4, 5, 6]
            ]
        );
        assert_eq!(compose(l, h, &bounds, &[]), vec![[1, 2, 3]; 6]);
    }
}
