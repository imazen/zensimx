//! zensim research bakes as a zenmetrics workbench module (the workbench's
//! `extra-modules.json` hook): every embedded bake as a metric, with its score,
//! its spatial maps and JPEG XL loop steering maps.
//!
//! The bakes are named at build time (`ZENSIM_WORKBENCH_BAKES=name=path;…`, see
//! `build.rs`), by convention `zensim-<arm>@rev<N>`: the arm and the formula
//! revision it serves. The menu group names the zensim version and commit the
//! module was built from (`ZENSIM_WORKBENCH_LABEL`). A revision-6 bake
//! (`ringhpm` + `globaff`) serves through the same `BakeScorer` owner as every
//! other bake. The maps are the bake's attribution map
//! (`BakeScorer::compute_with_ref_and_attribution`) under the research painting
//! modes of `benchmarks/byconstruction_2026-10-22.md` ("Maps"); each value is
//! the map's density on its cell grid, negative values (score-raising features)
//! clamped to zero for display.
//!
//! The module and its weights are private when the bakes are: serve them on
//! the LAN only. Exploration record: `benchmarks/byconstruction_2026-10-22.md`.

use std::cell::RefCell;

use wasm_bindgen::prelude::*;
use zensim::{BakeScorer, Fused944Session, RgbSlice};

include!(concat!(env!("OUT_DIR"), "/bakes.rs"));

/// The maps every metric publishes: `(name, window mode, paint mode, split,
/// smoothing σ in source pixels, bin)` (`zensim`'s research map modes).
/// `diffmap` (first) is the one the workbench shows by default: centred
/// window, guided paint and the exact split, the best balance in the map
/// study, integrated over σ = 3 px so a row-dithered gradient's error does
/// not show as one stripe per row (`benchmarks/byconstruction_2026-10-22.md`,
/// "Map study after the fixes" and "Streaking").
pub const MAPS: &[(&str, &str, &str, &str, f32, usize)] = &[
    ("diffmap", "center", "guided", "exact", 3.0, 1),
    // The same map per pixel, unintegrated.
    ("diffmap_px", "center", "guided", "exact", 0.0, 1),
    // Today's map: removal split, uniform window spread.
    ("removal", "box", "nearest", "repair", 0.0, 1),
    // 8×8 blocks for the JPEG XL loop.
    ("blocks8", "center", "nearest", "exact", 0.0, 8),
    // For steering: `diffmap` with the amplitude split (every nonlinear pool
    // as a mean pool of its member), so a block's value moves with its own
    // error as it does with the image's ("JPEG XL reachability with Tga").
    ("diffmap_amp", "center", "guided", "amplitude", 3.0, 1),
];

/// The module's menu group: the zensim version and source this module was
/// built from (`ZENSIM_WORKBENCH_LABEL` at build time).
pub const MODULE_LABEL: &str = match option_env!("ZENSIM_WORKBENCH_LABEL") {
    Some(label) => label,
    None => "zensim candidates (private module)",
};

/// `"zensimx"`.
#[wasm_bindgen]
pub fn module_name() -> String {
    "zensimx".into()
}

/// Threads scoring a pair: one (this module runs on the calling worker).
#[wasm_bindgen]
pub fn thread_count() -> usize {
    1
}

/// Every metric name (one per embedded bake), as a JSON array.
#[wasm_bindgen]
pub fn available_metrics() -> String {
    serde_json::json!(BAKES.iter().map(|b| b.0).collect::<Vec<_>>()).to_string()
}

/// Every metric with spatial maps (all of them), as a JSON array.
#[wasm_bindgen]
pub fn available_spatial_maps() -> String {
    available_metrics()
}

/// The workbench's cancellation and progress words. This module runs
/// synchronously and has no shared memory, so the words are never read
/// during a call.
#[wasm_bindgen]
pub struct JobControl {
    state: Box<[u32; 4]>,
}

#[wasm_bindgen]
impl JobControl {
    /// A control.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            state: Box::new([0; 4]),
        }
    }
    /// Offset of the four words in this module's memory.
    pub fn state_ptr(&self) -> usize {
        self.state.as_ptr() as usize
    }
    /// Begin a phase.
    pub fn begin(&mut self, phase: u32, total: u32) {
        self.state[1] = phase;
        self.state[2] = 0;
        self.state[3] = total;
    }
}

impl Default for JobControl {
    fn default() -> Self {
        Self::new()
    }
}

/// RGBA8 rows (opaque, as the workbench requires) to RGB8 pixels.
fn rgb(rgba: &[u8], width: u32, height: u32, stride: usize) -> Result<Vec<[u8; 3]>, String> {
    let (w, h) = (width as usize, height as usize);
    let row = w.checked_mul(4).ok_or("width overflow")?;
    if w == 0 || h == 0 || stride < row || rgba.len() < (h - 1) * stride + row {
        return Err("invalid dimensions, stride or buffer length".into());
    }
    let mut out = Vec::with_capacity(w * h);
    for y in 0..h {
        for p in rgba[y * stride..y * stride + row].as_chunks::<4>().0 {
            if p[3] != 255 {
                return Err(
                    "opaque sRGB8 required; choose an explicit alpha composite before scoring"
                        .into(),
                );
            }
            out.push([p[0], p[1], p[2]]);
        }
    }
    Ok(out)
}

fn model(name: &str) -> Result<zenpredict::Model, String> {
    let (_, bytes, _) = BAKES.iter().find(|b| b.0 == name).ok_or_else(|| {
        format!(
            "{name} is not a metric of this module ({})",
            available_metrics()
        )
    })?;
    zenpredict::Model::from_bytes(bytes).map_err(|e| format!("{name}: {e:?}"))
}

/// [`score_rgba`] with a plain error, callable natively.
pub fn score_json(
    metric: &str,
    reference: &[u8],
    distorted: &[u8],
    width: u32,
    height: u32,
    stride: usize,
) -> Result<String, String> {
    let m = model(metric)?;
    let (r, d) = (
        rgb(reference, width, height, stride)?,
        rgb(distorted, width, height, stride)?,
    );
    let (w, h) = (width as usize, height as usize);
    let mut scorer = BakeScorer::new(&m)
        .map_err(|e| e.to_string())?
        .with_parallel(false);
    let s = scorer
        .compute(&RgbSlice::new(&r, w, h), &RgbSlice::new(&d, w, h), None)
        .map_err(|e| e.to_string())?
        .score();
    let v = if s.is_finite() {
        serde_json::json!(s)
    } else {
        serde_json::json!(s.to_string())
    };
    Ok(
        serde_json::json!({"metric": metric, "backend": "native-cpu-wasm",
        "values": [{"column": "score", "value": v}]})
        .to_string(),
    )
}

/// Score a pair (the worker's `score_rgba`): JSON `{metric, backend, values}`.
#[wasm_bindgen]
pub fn score_rgba(
    metric: &str,
    reference: &[u8],
    distorted: &[u8],
    width: u32,
    height: u32,
    stride: usize,
    display: &str,
) -> Result<String, JsError> {
    let _ = display;
    score_json(metric, reference, distorted, width, height, stride).map_err(|e| JsError::new(&e))
}

/// One map, as the worker reads it.
#[wasm_bindgen]
#[derive(Clone)]
pub struct SpatialMap {
    name: String,
    names: Vec<String>,
    width: u32,
    height: u32,
    input: (u32, u32),
    cell: u32,
    values: Vec<f32>,
}

#[wasm_bindgen]
impl SpatialMap {
    /// The map's name (`diffmap`, `box`, …).
    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.name.clone()
    }
    /// Every map name of the metric, as a JSON array.
    #[wasm_bindgen(getter)]
    pub fn names(&self) -> String {
        serde_json::json!(self.names).to_string()
    }
    /// Always `"difference"`: the density of score loss.
    #[wasm_bindgen(getter)]
    pub fn polarity(&self) -> String {
        "difference".into()
    }
    /// Width in cells.
    #[wasm_bindgen(getter)]
    pub fn width(&self) -> u32 {
        self.width
    }
    /// Height in cells.
    #[wasm_bindgen(getter)]
    pub fn height(&self) -> u32 {
        self.height
    }
    /// `false`: a difference map.
    #[wasm_bindgen(getter)]
    pub fn similarity(&self) -> bool {
        false
    }
    /// Cell ownership, as the workbench's geometry JSON.
    #[wasm_bindgen(getter)]
    pub fn geometry(&self) -> String {
        serde_json::json!({"inputWidth": self.input.0, "inputHeight": self.input.1, "originX": 0, "originY": 0,
            "cellWidth": self.cell, "cellHeight": self.cell, "cellsX": self.width, "cellsY": self.height})
        .to_string()
    }
    /// Row-major cell values.
    #[wasm_bindgen(getter)]
    pub fn values(&self) -> Vec<f32> {
        self.values.clone()
    }
}

/// The last evaluation's inputs and the maps computed from them.
struct Last {
    metric: String,
    width: u32,
    height: u32,
    reference: Vec<[u8; 3]>,
    distorted: Vec<[u8; 3]>,
    maps: Vec<SpatialMap>,
}

thread_local! {
    static LAST: RefCell<Option<Last>> = const { RefCell::new(None) };
}

/// One map of a pair: the bake's attribution map under the map's modes.
pub fn compute_map(
    metric: &str,
    map: &str,
    reference: &[[u8; 3]],
    distorted: &[[u8; 3]],
    width: u32,
    height: u32,
) -> Result<SpatialMap, String> {
    let &(name, window, paint, split, smooth, bin) = MAPS
        .iter()
        .find(|m| m.0 == map)
        .ok_or_else(|| format!("no map named {map} (maps: {})", map_names().join(", ")))?;
    let m = model(metric)?;
    let (w, h) = (width as usize, height as usize);
    let (rs, ds) = (
        RgbSlice::new(reference, w, h),
        RgbSlice::new(distorted, w, h),
    );
    let mut scorer = BakeScorer::new(&m)
        .map_err(|e| e.to_string())?
        .with_parallel(false)
        .with_map_paint(window, paint, split)
        .and_then(|s| s.with_map_smooth(smooth))
        .map_err(|e| e.to_string())?;
    let pre = scorer
        .precompute_reference(&rs)
        .map_err(|e| e.to_string())?;
    let mut session = Fused944Session::new();
    let att = scorer
        .compute_with_ref_and_attribution(&rs, &pre, &ds, None, &mut session, bin)
        .map_err(|e| e.to_string())?;
    let a = att.attribution();
    Ok(SpatialMap {
        name: name.into(),
        names: map_names(),
        width: a.grid_width() as u32,
        height: a.grid_height() as u32,
        input: (width, height),
        cell: bin as u32,
        values: a.density().iter().map(|&v| v.max(0.0)).collect(),
    })
}

/// Every map name, `diffmap` first.
pub fn map_names() -> Vec<String> {
    MAPS.iter().map(|m| m.0.to_string()).collect()
}

/// The map named `map` of `metric` (empty: `diffmap`). The pair stays cached
/// for [`cached_spatial_map`].
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn spatial_map(
    metric: &str,
    map: &str,
    reference: &[u8],
    distorted: &[u8],
    width: u32,
    height: u32,
    stride: usize,
    display: &str,
) -> Result<SpatialMap, JsError> {
    let _ = display;
    let run = || -> Result<SpatialMap, String> {
        let (r, d) = (
            rgb(reference, width, height, stride)?,
            rgb(distorted, width, height, stride)?,
        );
        let want = if map.is_empty() { "diffmap" } else { map };
        let chosen = compute_map(metric, want, &r, &d, width, height)?;
        LAST.with(|s| {
            *s.borrow_mut() = Some(Last {
                metric: metric.into(),
                width,
                height,
                reference: r,
                distorted: d,
                maps: vec![chosen.clone()],
            })
        });
        Ok(chosen)
    };
    run().map_err(|e| JsError::new(&e))
}

/// Another map of the last [`spatial_map`] pair, computed on first request.
#[wasm_bindgen]
pub fn cached_spatial_map(name: &str) -> Result<SpatialMap, JsError> {
    LAST.with(|s| {
        let mut slot = s.borrow_mut();
        let last = slot
            .as_mut()
            .ok_or_else(|| JsError::new("no evaluation has published maps yet"))?;
        if let Some(m) = last.maps.iter().find(|m| m.name == name) {
            return Ok(m.clone());
        }
        let m = compute_map(
            &last.metric,
            name,
            &last.reference,
            &last.distorted,
            last.width,
            last.height,
        )
        .map_err(|e| JsError::new(&e))?;
        last.maps.push(m.clone());
        Ok(m)
    })
}

/// A map for jxl-encoder's loop: one value per pixel, and its maximum.
#[wasm_bindgen]
pub struct LoopMap {
    values: Vec<f32>,
    score: f64,
}

#[wasm_bindgen]
impl LoopMap {
    /// Row-major per-pixel values.
    #[wasm_bindgen(getter)]
    pub fn values(&self) -> Vec<f32> {
        self.values.clone()
    }
    /// The largest value.
    #[wasm_bindgen(getter)]
    pub fn score(&self) -> f64 {
        self.score
    }
}

/// [`loop_map`] with a plain error, callable natively: `metric`'s map `map`
/// read as local error (cell density times the image's pixel count, so the
/// image mean is the map's total at any size; cells spread over their
/// pixels), then `scale · e^exponent`.
#[allow(clippy::too_many_arguments)]
pub fn loop_map_values(
    metric: &str,
    map: &str,
    reference: &[u8],
    distorted: &[u8],
    width: u32,
    height: u32,
    stride: usize,
    scale: f64,
    exponent: f64,
) -> Result<(Vec<f32>, f64), String> {
    if !(scale.is_finite() && scale > 0.0 && exponent.is_finite() && exponent > 0.0) {
        return Err("scale and exponent must be finite and positive".into());
    }
    let (r, d) = (
        rgb(reference, width, height, stride)?,
        rgb(distorted, width, height, stride)?,
    );
    let want = if map.is_empty() { "diffmap" } else { map };
    let m = compute_map(metric, want, &r, &d, width, height)?;
    let (w, h) = (width as usize, height as usize);
    let n = (w * h) as f64;
    let mut values = vec![0.0f32; w * h];
    let cell = m.cell as usize;
    for y in 0..h {
        for x in 0..w {
            let v = m.values[(y / cell) * m.width as usize + x / cell] as f64 * n;
            values[y * w + x] = (scale * v.max(0.0).powf(exponent)) as f32;
        }
    }
    let score = values.iter().copied().fold(0f32, f32::max).into();
    Ok((values, score))
}

/// The steering map for one loop iteration (see [`loop_map_values`]).
#[wasm_bindgen]
#[allow(clippy::too_many_arguments)]
pub fn loop_map(
    metric: &str,
    map: &str,
    reference: &[u8],
    distorted: &[u8],
    width: u32,
    height: u32,
    stride: usize,
    display: &str,
    scale: f64,
    exponent: f64,
) -> Result<LoopMap, JsError> {
    let _ = display;
    let (values, score) = loop_map_values(
        metric, map, reference, distorted, width, height, stride, scale, exponent,
    )
    .map_err(|e| JsError::new(&e))?;
    Ok(LoopMap { values, score })
}

/// Each map's placement on butteraugli's scale for one bake,
/// `(bake SHA-256 prefix, map, scale, exponent, r)`: fitted by zenmetrics'
/// method on its calibration encodes of the CID22-512 training images, with
/// `r` the log-space correlation of block distances with butteraugli's on the
/// held-out gb82 screenshots (`examples/loop_calibrate.rs`, 2026-10-24, file
/// `loop_calibration_2026-10-24c.tsv`; record
/// `benchmarks/byconstruction_2026-10-22.md`). A bake or map without an entry
/// gets scale 1, exponent 0.5 and no `r`.
pub const LOOP_DEFAULTS: &[(&str, &str, f64, f64, f64)] = &[
    // The revision-6 Tga candidate (maps with the globaff term).
    ("5ece9e45", "diffmap", 1.329536e-1, 0.6, 0.6532),
    ("5ece9e45", "diffmap_px", 1.411546e-1, 0.4, 0.4948),
    ("5ece9e45", "removal", 1.499584e-1, 0.5, 0.5125),
    ("5ece9e45", "blocks8", 1.553179e-1, 0.6, 0.6112),
    // M64a, the strictly monotone max-pair arm (its KADID-held-out fold cell, revision 6).
    ("b8555871", "diffmap", 1.372091e-1, 0.6, 0.6810),
    ("b8555871", "diffmap_px", 1.449550e-1, 0.4, 0.4570),
    ("b8555871", "removal", 1.506346e-1, 0.5, 0.3960),
    ("b8555871", "blocks8", 1.590695e-1, 0.6, 0.6422),
    // B (diff+frag-820, revision 5).
    ("89cb4861", "diffmap", 1.302887e-1, 0.6, 0.4909),
    ("89cb4861", "diffmap_px", 1.748248e-1, 0.35, 0.2805),
    ("89cb4861", "removal", 1.740943e-1, 0.45, 0.3308),
    ("89cb4861", "blocks8", 2.027928e-1, 0.5, 0.4378),
];

/// The loop defaults, as `[{metric, map, scale, exponent, r}]`, one per
/// embedded bake and map.
#[wasm_bindgen]
pub fn loop_map_defaults() -> String {
    serde_json::json!(
        BAKES
            .iter()
            .flat_map(|b| MAPS.iter().map(move |m| {
                let d = LOOP_DEFAULTS
                    .iter()
                    .find(|d| b.2.starts_with(d.0) && d.1 == m.0)
                    .map_or((1.0, 0.5, f64::NAN), |d| (d.2, d.3, d.4));
                serde_json::json!({"metric": b.0, "map": m.0, "scale": d.0, "exponent": d.1,
                    "r": if d.2.is_finite() { serde_json::json!(d.2) } else { serde_json::Value::Null }})
            }))
            .collect::<Vec<_>>()
    )
    .to_string()
}

/// The workbench manifest entry for this module (`extra-modules.json`).
pub fn manifest_entry(url: &str) -> serde_json::Value {
    let metrics: Vec<_> = BAKES
        .iter()
        .map(|b| serde_json::json!({"name": b.0, "maps": map_names(), "bake_sha256": b.2,
            "cli": format!("score_pair_with_bake --bake <{}> --ref {{reference}} --dist {{distorted}}", b.0)}))
        .collect();
    serde_json::json!({
        "name": "zensimx",
        "label": MODULE_LABEL,
        "url": url,
        "metrics": metrics,
        "loopDefaults": serde_json::from_str::<serde_json::Value>(&loop_map_defaults()).expect("defaults json"),
    })
}
