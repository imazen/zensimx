use crate::Result;
use sha2::{Digest, Sha256};
use std::path::Path;
pub(crate) fn write(out: &Path) -> Result<()> {
    let sources = [
        ("main", include_str!("main.rs")),
        ("provenance", include_str!("provenance.rs")),
        ("audit", include_str!("audit.rs")),
        ("bank", include_str!("bank.rs")),
        ("pages", include_str!("pages.rs")),
        ("constraints", include_str!("constraints.rs")),
        ("function", include_str!("function.rs")),
        ("model", include_str!("model.rs")),
        ("lens", include_str!("lens.rs")),
        ("compare", include_str!("compare.rs")),
        ("atlas", include_str!("atlas.rs")),
        ("pixels", include_str!("pixels.rs")),
        ("manifest", include_str!("../Cargo.toml")),
        ("build", include_str!("../build.rs")),
        ("research", include_str!("../../../zensim/src/research.rs")),
        (
            "maps",
            include_str!("../../../zensim/src/feature_v2_viz.rs"),
        ),
    ];
    let mut hash = Sha256::new();
    for (name, text) in sources {
        hash.update(name.as_bytes());
        hash.update([0]);
        hash.update(text.as_bytes());
        hash.update([0]);
    }
    crate::json(
        &out.join("provenance.json"),
        &serde_json::json!({"binary_sha256":crate::hash(&std::env::current_exe()?)?,"compiled_metric_source_sha256":env!("ZENSIM_VIZ_METRIC_SOURCE_SHA256"),"compiled_cargo_lock_sha256":env!("ZENSIM_VIZ_LOCK_SHA256"),"compiled_inspection_source_sha256":hash.finalize().iter().map(|b|format!("{b:02x}")).collect::<String>(),"source_hash_scope":"compiled exporter modules, manifest, research entry and private map exporter; dependency revisions are pinned by Cargo.lock","command":std::env::args().collect::<Vec<_>>(),"formula_revision_environment":std::env::var("ZENSIM_FORMULA_REV").ok(),"data_scope":"no AIC tables admitted; explicit bank path and SHA256 in each result"}),
    )
}
