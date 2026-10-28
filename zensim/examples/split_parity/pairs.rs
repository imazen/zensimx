// Deterministic sRGB8 image pairs for the repository-split parity gate.
//
// This file is `include!`d by `zensimx/zensim/examples/split_parity_golden.rs`
// (the research reference) and by `zensim/examples/split_parity_check.rs` in
// imazen/zensim (the slim crate). Both copies must stay byte-identical; the
// checker compares the two crates' scores bit for bit on these pairs.

/// One generated pair: id, width, height, source RGB8, distorted RGB8.
pub struct SplitPair {
    pub id: String,
    pub width: usize,
    pub height: usize,
    pub src: Vec<u8>,
    pub dst: Vec<u8>,
}

struct XorShift(u64);

impl XorShift {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn byte(&mut self) -> u8 {
        (self.next() >> 56) as u8
    }
    fn signed(&mut self, amp: i32) -> i32 {
        (self.next() % (2 * amp as u64 + 1)) as i32 - amp
    }
}

fn clamp8(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

fn content(kind: &str, w: usize, h: usize, seed: u64) -> Vec<u8> {
    let mut rng = XorShift(seed | 1);
    let mut img = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            let (r, g, b) = match kind {
                // smooth gradients plus fine texture
                "photo" => {
                    let t = rng.signed(12);
                    (
                        (x * 255 / w.max(1)) as i32 + t,
                        (y * 255 / h.max(1)) as i32 - t,
                        (((x + y) * 128) / (w + h).max(1)) as i32 + 64 + t / 2,
                    )
                }
                // hard edges: checkerboard with coloured tiles
                "edges" => {
                    let tile = ((x / 7) + (y / 5)) % 3;
                    match tile {
                        0 => (240, 30, 30),
                        1 => (20, 200, 60),
                        _ => (30, 40, 220),
                    }
                }
                // text-like thin dark strokes on a light ground
                _ => {
                    let stroke = (x % 9 == 2) || (y % 11 == 4 && x % 23 < 15) || ((x + 2 * y) % 31 == 0);
                    if stroke { (25, 25, 30) } else { (245, 242, 235) }
                }
            };
            img[i] = clamp8(r);
            img[i + 1] = clamp8(g);
            img[i + 2] = clamp8(b);
        }
    }
    img
}

fn distort(kind: &str, src: &[u8], w: usize, h: usize, seed: u64) -> Vec<u8> {
    let mut rng = XorShift(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
    let mut out = src.to_vec();
    match kind {
        "identity" => {}
        "noise" => {
            for v in out.iter_mut() {
                *v = clamp8(*v as i32 + rng.signed(10));
            }
        }
        "blur" => {
            for y in 0..h {
                for x in 0..w {
                    for c in 0..3 {
                        let mut acc = 0i32;
                        let mut n = 0i32;
                        for dy in -1i32..=1 {
                            for dx in -1i32..=1 {
                                let (xx, yy) = (x as i32 + dx, y as i32 + dy);
                                if xx >= 0 && yy >= 0 && (xx as usize) < w && (yy as usize) < h {
                                    acc += src[(yy as usize * w + xx as usize) * 3 + c] as i32;
                                    n += 1;
                                }
                            }
                        }
                        out[(y * w + x) * 3 + c] = (acc / n) as u8;
                    }
                }
            }
        }
        "posterize" => {
            for v in out.iter_mut() {
                *v = (*v & 0xF0) | 0x08;
            }
        }
        "shift" => {
            for p in out.as_chunks_mut::<3>().0 {
                p[0] = clamp8(p[0] as i32 + 6);
                p[2] = clamp8(p[2] as i32 - 4);
            }
        }
        "blocks" => {
            for by in (0..h).step_by(8) {
                for bx in (0..w).step_by(8) {
                    if rng.byte() < 140 {
                        continue;
                    }
                    for c in 0..3 {
                        let (mut acc, mut n) = (0u32, 0u32);
                        for y in by..(by + 8).min(h) {
                            for x in bx..(bx + 8).min(w) {
                                acc += src[(y * w + x) * 3 + c] as u32;
                                n += 1;
                            }
                        }
                        let m = (acc / n) as u8;
                        for y in by..(by + 8).min(h) {
                            for x in bx..(bx + 8).min(w) {
                                out[(y * w + x) * 3 + c] = m;
                            }
                        }
                    }
                }
            }
        }
        // a few isolated single-pixel changes (near-identity)
        _ => {
            for _ in 0..3 {
                let i = (rng.next() as usize) % out.len();
                out[i] = out[i].wrapping_add(1);
            }
        }
    }
    out
}

/// Every pair the parity gate scores, in a fixed order.
pub fn split_parity_pairs() -> Vec<SplitPair> {
    let sizes = [
        (8, 8),
        (16, 16),
        (31, 17),
        (64, 64),
        (65, 63),
        (128, 96),
        (200, 150),
        (256, 256),
        (333, 251),
        (512, 384),
        (1024, 768),
    ];
    let contents = ["photo", "edges", "text"];
    let distortions = ["identity", "noise", "blur", "posterize", "shift", "blocks", "pixels"];
    let mut out = Vec::new();
    let mut seed = 1u64;
    for &(w, h) in &sizes {
        for c in contents {
            seed += 1;
            let src = content(c, w, h, seed);
            for d in distortions {
                seed += 1;
                let dst = distort(d, &src, w, h, seed);
                out.push(SplitPair { id: format!("{w}x{h}-{c}-{d}"), width: w, height: h, src: src.clone(), dst });
            }
        }
    }
    out
}
