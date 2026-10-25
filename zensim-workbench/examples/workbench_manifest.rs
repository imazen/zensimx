//! Print the zenmetrics workbench's `extra-modules.json` entry for this module:
//! `{"modules": [entry]}`, the module glue at `<url>` (relative to the
//! workbench's worker.js; default `pkg-zensimx/zensim_workbench.js`).
//!
//! `ZENSIM_WORKBENCH_BAKES=... cargo run --release --example workbench_manifest -- [url]`

fn main() {
    let url = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "pkg-zensimx/zensim_workbench.js".into());
    let entry = zensim_workbench::manifest_entry(&url);
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({ "modules": [entry] })).unwrap()
    );
}
