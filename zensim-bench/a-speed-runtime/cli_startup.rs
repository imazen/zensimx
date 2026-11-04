//! Measure the unchanged margarine command's no-input startup separately.
#![forbid(unsafe_code)]
use std::{
    path::Path,
    process::{Command, Stdio},
    time::Duration,
};

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let command = args[1].clone();
    assert!(Path::new(&command).is_file());
    let result = zenbench::run(|suite| {
        suite.compare("margarine/command-help-startup", |group| {
            let cfg = group.config();
            cfg.min_rounds = 32;
            cfg.max_rounds = 64;
            cfg.auto_rounds = false;
            cfg.min_iterations = 1;
            cfg.max_iterations = 1;
            cfg.warmup_time = Duration::from_millis(100);
            group.bench("margarine_help", move |b| {
                b.iter(|| {
                    let status = Command::new(&command)
                        .arg("--help")
                        .stdout(Stdio::null())
                        .status()
                        .expect("start pinned margarine command");
                    assert!(status.success());
                    zenbench::black_box(status)
                });
            });
        });
    });
    result.save(Path::new(&args[2])).unwrap();
    assert!(!result.unreliable);
    assert!(
        result.comparisons[0]
            .samples
            .iter()
            .filter(|r| r.gate_clean == Some(true) && r.iterations == 1)
            .count()
            >= 32
    );
}
