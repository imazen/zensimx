fn main() {
    println!("cargo:rerun-if-env-changed=MARGARINE_SOURCE");
    let source = std::env::var("MARGARINE_SOURCE")
        .expect("set MARGARINE_SOURCE to the pinned zenmetrics/crates/margarine directory");
    let kernel = std::path::Path::new(&source)
        .join("kernel.rs")
        .canonicalize()
        .unwrap();
    println!("cargo:rustc-env=MARGARINE_KERNEL={}", kernel.display());
    println!("cargo:rerun-if-changed={}", kernel.display());
}
