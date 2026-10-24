//! Shared pieces of the loop tools: the wasm codec's view of a reconstruction
//! (sRGB8), PNG input, and the external steering scorers they attach to
//! jxl-encoder's perceptual loop: a zensim-workbench map ([`MapScorer`]) and a
//! map served by another process ([`ServerScorer`], for metrics this crate
//! cannot link).

use std::io::{BufRead, BufReader, Read, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::{Arc, Mutex};

use jxl_encoder::ExternalLoopMetric;

/// sRGB8 of a linear value in `[0, 1]`, as the wasm codec hands
/// reconstructions to the scorer.
pub fn srgb8(v: f32) -> u8 {
    let v = v.clamp(0.0, 1.0);
    let e = if v <= 0.003_130_8 {
        12.92 * v
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    };
    (e * 255.0 + 0.5) as u8
}

/// A reconstruction (planar linear RGB, rows `stride` values apart) as sRGB8
/// RGBA.
pub fn recon_rgba(
    r: &[f32],
    g: &[f32],
    b: &[f32],
    stride: usize,
    width: usize,
    height: usize,
) -> Vec<u8> {
    let mut rgba = Vec::with_capacity(width * height * 4);
    for y in 0..height {
        for x in y * stride..y * stride + width {
            rgba.extend([srgb8(r[x]), srgb8(g[x]), srgb8(b[x]), 255]);
        }
    }
    rgba
}

/// An 8-bit PNG as packed RGB8.
pub fn read_png(path: &str) -> (Vec<u8>, u32, u32) {
    let mut decoder = png::Decoder::new(std::io::BufReader::new(
        std::fs::File::open(path).unwrap_or_else(|e| panic!("{path}: {e}")),
    ));
    // Palette and sub-byte images expand to 8-bit RGB(A); 16-bit strips.
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder.read_info().expect("png header");
    let mut buf = vec![0; reader.output_buffer_size().expect("png size")];
    let info = reader.next_frame(&mut buf).expect("png frame");
    assert_eq!(info.bit_depth, png::BitDepth::Eight, "8-bit PNG expected");
    let rgb: Vec<u8> = match info.color_type {
        png::ColorType::Rgb => buf[..info.buffer_size()].to_vec(),
        png::ColorType::Rgba => buf[..info.buffer_size()]
            .as_chunks::<4>()
            .0
            .iter()
            .flat_map(|p| [p[0], p[1], p[2]])
            .collect(),
        png::ColorType::Grayscale => buf[..info.buffer_size()]
            .iter()
            .flat_map(|&v| [v, v, v])
            .collect(),
        other => panic!("unsupported PNG color type {other:?}"),
    };
    (rgb, info.width, info.height)
}

/// Packed RGB8 as opaque RGBA8.
pub fn rgba(rgb: &[u8]) -> Vec<u8> {
    rgb.as_chunks::<3>()
        .0
        .iter()
        .flat_map(|p| [p[0], p[1], p[2], 255])
        .collect()
}

/// The first error a scorer met. The loop answers an error by ending the
/// seed with the previous iteration's result, so a tool must check this after
/// every encode.
pub type ErrorSlot = Arc<Mutex<Option<String>>>;

fn record(slot: &ErrorSlot, e: &str) {
    let mut s = slot.lock().unwrap_or_else(|p| p.into_inner());
    if s.is_none() {
        *s = Some(e.to_string());
    }
}

/// A zensim-workbench map at a placement on butteraugli's scale, through the
/// adapter zenmetrics' wasm codec uses (`web/codecs/jxl/src/lib.rs`): each
/// reconstruction as sRGB8 RGBA, the map's values placed, the score their
/// largest value.
pub struct MapScorer {
    pub metric: String,
    pub map: String,
    pub scale: f64,
    pub exponent: f64,
    /// The source the encode was given, sRGB8 RGBA.
    pub reference: Vec<u8>,
    pub error: ErrorSlot,
}

impl ExternalLoopMetric for MapScorer {
    fn set_reference(
        &mut self,
        _r: &[f32],
        _g: &[f32],
        _b: &[f32],
        _width: usize,
        _height: usize,
    ) -> Result<(), String> {
        // As in the workbench, the scorer already holds the source the encode
        // was given.
        Ok(())
    }

    fn compare(
        &mut self,
        r: &[f32],
        g: &[f32],
        b: &[f32],
        stride: usize,
        width: usize,
        height: usize,
        diffmap: &mut Vec<f32>,
    ) -> Result<f64, String> {
        let rgba = recon_rgba(r, g, b, stride, width, height);
        let (values, score) = zensim_workbench::loop_map_values(
            &self.metric,
            &self.map,
            &self.reference,
            &rgba,
            width as u32,
            height as u32,
            width * 4,
            self.scale,
            self.exponent,
        )
        .inspect_err(|e| record(&self.error, e))?;
        diffmap.clear();
        diffmap.extend_from_slice(&values);
        Ok(score)
    }
}

/// A steering-map server: another process that computes a metric's loop map,
/// for metrics this crate cannot link (a private module, say).
///
/// Protocol on the child's stdin and stdout, little-endian. On start the
/// server writes one line of JSON describing itself (at least the metric, the
/// map and the placement it applies). Then each message from the client is a
/// tag byte, `width` and `height` as `u32`, and `width × height` RGBA8 bytes:
/// tag `R` sets the reference (no reply); tag `D` scores a distorted image, and
/// the server replies with the score (`f64`, the loop's) and `width × height`
/// steering values (`f32`, already placed on butteraugli's scale). The server
/// exits when its stdin closes.
pub struct MapServer {
    child: Child,
    stdin: Option<ChildStdin>,
    stdout: BufReader<ChildStdout>,
    /// The server's self-description, its first line.
    pub header: String,
}

impl MapServer {
    /// Starts `command[0]` with the remaining arguments and reads its header.
    pub fn spawn(command: &[String]) -> Self {
        let mut child = Command::new(&command[0])
            .args(&command[1..])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .unwrap_or_else(|e| panic!("map server {:?}: {e}", command[0]));
        let stdin = child.stdin.take();
        let mut stdout = BufReader::new(child.stdout.take().expect("server stdout"));
        let mut header = String::new();
        stdout.read_line(&mut header).expect("server header");
        assert!(
            header.ends_with('\n'),
            "map server {:?} exited before its header",
            command[0]
        );
        header.pop();
        Self {
            child,
            stdin,
            stdout,
            header,
        }
    }

    fn send(&mut self, tag: u8, rgba: &[u8], width: usize, height: usize) -> std::io::Result<()> {
        assert_eq!(rgba.len(), width * height * 4, "RGBA8 of width × height");
        let w = self.stdin.as_mut().expect("server stdin");
        w.write_all(&[tag])?;
        w.write_all(&(width as u32).to_le_bytes())?;
        w.write_all(&(height as u32).to_le_bytes())?;
        w.write_all(rgba)?;
        w.flush()
    }

    /// Sets the reference.
    pub fn set_reference(
        &mut self,
        rgba: &[u8],
        width: usize,
        height: usize,
    ) -> std::io::Result<()> {
        self.send(b'R', rgba, width, height)
    }

    /// Scores `rgba` against the reference: fills `values` and returns the score.
    pub fn compare(
        &mut self,
        rgba: &[u8],
        width: usize,
        height: usize,
        values: &mut Vec<f32>,
    ) -> std::io::Result<f64> {
        self.send(b'D', rgba, width, height)?;
        let mut score = [0u8; 8];
        self.stdout.read_exact(&mut score)?;
        let mut raw = vec![0u8; width * height * 4];
        self.stdout.read_exact(&mut raw)?;
        values.clear();
        values.extend(
            raw.as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b)),
        );
        Ok(f64::from_le_bytes(score))
    }
}

impl Drop for MapServer {
    fn drop(&mut self) {
        // Closing stdin ends the server.
        drop(self.stdin.take());
        let _ = self.child.wait();
    }
}

/// The loop scored by a [`MapServer`]: each reconstruction as sRGB8 RGBA, as
/// [`MapScorer`] does.
pub struct ServerScorer {
    pub server: Arc<Mutex<MapServer>>,
    /// The source the encode was given, sRGB8 RGBA.
    pub reference: Vec<u8>,
    pub error: ErrorSlot,
}

impl ExternalLoopMetric for ServerScorer {
    fn set_reference(
        &mut self,
        _r: &[f32],
        _g: &[f32],
        _b: &[f32],
        width: usize,
        height: usize,
    ) -> Result<(), String> {
        let mut server = self.server.lock().unwrap_or_else(|p| p.into_inner());
        server
            .set_reference(&self.reference, width, height)
            .map_err(|e| format!("map server reference: {e}"))
            .inspect_err(|e| record(&self.error, e))
    }

    fn compare(
        &mut self,
        r: &[f32],
        g: &[f32],
        b: &[f32],
        stride: usize,
        width: usize,
        height: usize,
        diffmap: &mut Vec<f32>,
    ) -> Result<f64, String> {
        let rgba = recon_rgba(r, g, b, stride, width, height);
        let mut server = self.server.lock().unwrap_or_else(|p| p.into_inner());
        let score = server
            .compare(&rgba, width, height, diffmap)
            .map_err(|e| format!("map server compare: {e}"))
            .inspect_err(|e| record(&self.error, e))?;
        if !score.is_finite() || diffmap.iter().any(|v| !v.is_finite()) {
            let e = "map server returned a nonfinite value".to_string();
            record(&self.error, &e);
            return Err(e);
        }
        Ok(score)
    }
}
