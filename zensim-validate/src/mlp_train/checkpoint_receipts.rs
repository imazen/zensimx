//! Private shared decoder for native checkpoint write ownership receipts.
use std::path::PathBuf;

pub(crate) fn paths(log: &[String]) -> Vec<(PathBuf, usize)> {
    #[derive(serde::Deserialize)]
    #[serde(untagged)]
    enum CheckpointPath {
        Utf8(PathBuf),
        Native(std::ffi::OsString),
    }
    #[derive(serde::Deserialize)]
    struct WrittenCheckpoint {
        path: CheckpointPath,
        epoch: usize,
    }
    log.iter()
        .filter_map(|line| line.strip_prefix("  checkpoint dump: "))
        .map(|receipt| {
            let written: WrittenCheckpoint =
                serde_json::from_str(receipt).expect("checkpoint write receipt");
            let path = match written.path {
                CheckpointPath::Utf8(path) => path,
                CheckpointPath::Native(path) => PathBuf::from(path),
            };
            (path, written.epoch)
        })
        .collect()
}
