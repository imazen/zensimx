use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Path, PathBuf},
};
fn collect(dir: &Path, files: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).expect("metric source directory") {
        let path = entry.expect("source entry").path();
        if path.is_dir() {
            collect(&path, files);
        } else if path.extension().is_some_and(|v| v == "rs") {
            files.push(path);
        }
    }
}
fn main() {
    let root = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let metric = root.join("../../zensim");
    let mut files = Vec::new();
    collect(&metric.join("src"), &mut files);
    files.push(metric.join("Cargo.toml"));
    files.sort();
    let mut digest = Sha256::new();
    for path in files {
        println!("cargo:rerun-if-changed={}", path.display());
        digest.update(
            path.strip_prefix(&metric)
                .unwrap()
                .to_string_lossy()
                .as_bytes(),
        );
        digest.update([0]);
        digest.update(fs::read(&path).expect("metric source bytes"));
        digest.update([0]);
    }
    let value = digest
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
    println!("cargo:rustc-env=ZENSIM_VIZ_METRIC_SOURCE_SHA256={value}");
    let lock = root.join("Cargo.lock");
    println!("cargo:rerun-if-changed={}", lock.display());
    let value = Sha256::digest(fs::read(lock).expect("standalone Cargo.lock"))
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>();
    println!("cargo:rustc-env=ZENSIM_VIZ_LOCK_SHA256={value}");
}
