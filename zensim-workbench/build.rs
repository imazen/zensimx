//! Embed the bakes named by `ZENSIM_WORKBENCH_BAKES` (`name=path;name=path`):
//! each is copied into `OUT_DIR` and listed, with its SHA-256, in `bakes.rs`.
//! The bakes may be private; the built module then is too.

use sha2::{Digest, Sha256};
use std::fmt::Write as _;

fn main() {
    println!("cargo:rerun-if-env-changed=ZENSIM_WORKBENCH_BAKES");
    println!("cargo:rerun-if-env-changed=ZENSIM_WORKBENCH_LABEL");
    let spec = std::env::var("ZENSIM_WORKBENCH_BAKES").unwrap_or_default();
    let out = std::path::PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    let mut table = String::from("pub static BAKES: &[(&str, &[u8], &str)] = &[\n");
    for (i, entry) in spec.split(';').filter(|e| !e.trim().is_empty()).enumerate() {
        let (name, path) = entry.split_once('=').unwrap_or_else(|| {
            panic!("ZENSIM_WORKBENCH_BAKES entry {entry:?}: expected name=path")
        });
        println!("cargo:rerun-if-changed={path}");
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let sha = Sha256::digest(&bytes);
        let hex = sha.iter().fold(String::new(), |mut s, b| {
            let _ = write!(s, "{b:02x}");
            s
        });
        let file = out.join(format!("bake{i}.bin"));
        std::fs::write(&file, &bytes).expect("copy bake");
        let _ = writeln!(
            table,
            "    ({name:?}, include_bytes!(concat!(env!(\"OUT_DIR\"), \"/bake{i}.bin\")), {hex:?}),"
        );
    }
    table.push_str("];\n");
    std::fs::write(out.join("bakes.rs"), table).expect("write bakes.rs");
}
