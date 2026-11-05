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

use wasm_bindgen::prelude::*;
include!(concat!(env!("OUT_DIR"), "/bakes.rs"));

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

// Pages needs raw detectors and compact-bank inference only. Legacy attribution
// exports link the built-in B profile through their generic fallback paths.
#[cfg(not(all(feature = "pages-viz", target_arch = "wasm32")))]
mod legacy;
#[cfg(not(all(feature = "pages-viz", target_arch = "wasm32")))]
pub use legacy::*;

#[cfg(feature = "pages-viz")]
mod viz;
#[cfg(feature = "pages-viz")]
pub use viz::*;
