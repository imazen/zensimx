//! Bounded atomic storage for private full-state trainer snapshots.
//! A run reserves 64 MiB including both retained states and a pending write.
//! The shared store reserves at most 4 GiB; completed runs still count.
use serde::{Serialize, de::DeserializeOwned};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{self, Read, Write},
    path::{Path, PathBuf},
};

const RUN_BYTES: u64 = 64 * 1024 * 1024;
const SHARED_BYTES: u64 = 4 * 1024 * 1024 * 1024;
const METADATA_BYTES: u64 = 4096;
const STATE_BYTES: u64 = (RUN_BYTES - METADATA_BYTES) / 3;

pub(super) struct Store {
    directory: PathBuf,
    _lock: File,
    generation: u64,
}

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}
fn regular(path: &Path) -> io::Result<()> {
    if !fs::symlink_metadata(path)?.file_type().is_file() {
        return Err(invalid("snapshot path is not a regular file"));
    }
    Ok(())
}

fn free_reserve(path: &Path) -> io::Result<()> {
    let output = std::process::Command::new("df")
        .args(["-B1", "--output=avail"])
        .arg(path)
        .output()?;
    if !output.status.success() {
        return Err(invalid("cannot resolve snapshot filesystem headroom"));
    }
    let free = String::from_utf8_lossy(&output.stdout)
        .split_whitespace()
        .last()
        .unwrap_or("")
        .parse::<u64>()
        .map_err(|_| invalid("invalid snapshot filesystem headroom"))?;
    if free < 30 * 1024 * 1024 * 1024 + SHARED_BYTES {
        return Err(invalid(
            "snapshot admission would threaten 30 GiB reserve plus shared pending writes",
        ));
    }
    Ok(())
}

impl Store {
    pub(super) fn open(root: &Path, run: &str) -> io::Result<Self> {
        if run.len() != 64
            || !run
                .bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
        {
            return Err(invalid("snapshot run identity must be lowercase SHA-256"));
        }
        fs::create_dir_all(root)?;
        free_reserve(root)?;
        if !fs::symlink_metadata(root)?.file_type().is_dir() {
            return Err(invalid("snapshot root is not a directory"));
        }
        let lock_path = root.join(".lock");
        let global = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&lock_path)?;
        regular(&lock_path)?;
        global.lock()?;
        let directory = root.join(run);
        let mut runs = 0_u64;
        for entry in fs::read_dir(root)? {
            let entry = entry?;
            if entry.file_name() == ".lock" {
                continue;
            }
            if !entry.file_type()?.is_dir() {
                return Err(invalid("unknown file in shared snapshot store"));
            }
            runs += 1;
        }
        if !directory.exists() {
            if (runs + 1)
                .checked_mul(RUN_BYTES)
                .and_then(|n| n.checked_add(METADATA_BYTES))
                .is_none_or(|n| n > SHARED_BYTES)
            {
                return Err(invalid(
                    "shared snapshot reservation exhausted; archive completed states before admission",
                ));
            }
            fs::create_dir(&directory)?;
            File::open(root)?.sync_all()?;
        }
        if !fs::symlink_metadata(&directory)?.file_type().is_dir() {
            return Err(invalid("snapshot run path is not a directory"));
        }
        let lock_path = directory.join(".lock");
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&lock_path)?;
        regular(&lock_path)?;
        lock.try_lock()
            .map_err(|e| io::Error::other(e.to_string()))?;
        drop(global);
        let mut store = Self {
            directory,
            _lock: lock,
            generation: 0,
        };
        store.check_files()?;
        for slot in 0..2 {
            if let Ok((generation, _)) = store.read_slot(slot) {
                store.generation = store.generation.max(generation);
            }
        }
        Ok(store)
    }

    fn check_files(&self) -> io::Result<()> {
        let mut total = 0_u64;
        for entry in fs::read_dir(&self.directory)? {
            let entry = entry?;
            let name = entry.file_name();
            if ![".lock", "state-0.snap", "state-1.snap", "writing.snap"]
                .iter()
                .any(|n| name == *n)
            {
                return Err(invalid("unknown file in snapshot run directory"));
            }
            regular(&entry.path())?;
            let bytes = entry.metadata()?.len();
            if name != ".lock" && bytes > STATE_BYTES {
                return Err(invalid("snapshot exceeds per-state ceiling"));
            }
            total = total
                .checked_add(bytes)
                .ok_or_else(|| invalid("snapshot size overflow"))?;
        }
        if total > RUN_BYTES {
            return Err(invalid("snapshot run exceeds 64 MiB ceiling"));
        }
        Ok(())
    }

    fn read_slot(&self, slot: usize) -> io::Result<(u64, Vec<u8>)> {
        let path = self.directory.join(format!("state-{slot}.snap"));
        regular(&path)?;
        let mut file = File::open(path)?;
        if file.metadata()?.len() > STATE_BYTES {
            return Err(invalid("oversized snapshot"));
        }
        let mut header = [0_u8; 40];
        file.read_exact(&mut header)?;
        let mut body = Vec::new();
        file.take(STATE_BYTES).read_to_end(&mut body)?;
        if Sha256::digest(&body)[..] != header[8..] {
            return Err(invalid("snapshot checksum differs"));
        }
        Ok((u64::from_le_bytes(header[..8].try_into().unwrap()), body))
    }

    pub(super) fn load<T: DeserializeOwned>(&self) -> io::Result<Option<T>> {
        let mut states = Vec::new();
        let mut existed = false;
        for slot in 0..2 {
            existed |= self.directory.join(format!("state-{slot}.snap")).exists();
            if let Ok(state) = self.read_slot(slot) {
                states.push(state);
            }
        }
        states.sort_by_key(|(generation, _)| std::cmp::Reverse(*generation));
        for (_, body) in states {
            if let Ok(state) = serde_json::from_slice(&body) {
                return Ok(Some(state));
            }
        }
        if existed {
            Err(invalid("no intact reusable snapshot"))
        } else {
            Ok(None)
        }
    }

    pub(super) fn save<T: Serialize>(&mut self, state: &T) -> io::Result<()> {
        // Serialize into a bounded buffer before creating/truncating any file.
        struct Buffer(Vec<u8>);
        impl Write for Buffer {
            fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
                if self.0.len() as u64 + bytes.len() as u64 + 40 > STATE_BYTES {
                    return Err(invalid("snapshot exceeds bounded serialization allowance"));
                }
                self.0.extend_from_slice(bytes);
                Ok(bytes.len())
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        let mut body = Buffer(Vec::new());
        serde_json::to_writer(&mut body, state)?;
        self.check_files()?;
        free_reserve(&self.directory)?;
        let generation = self
            .generation
            .checked_add(1)
            .ok_or_else(|| invalid("snapshot generation overflow"))?;
        let pending = self.directory.join("writing.snap");
        if pending.exists() {
            regular(&pending)?;
        }
        let mut file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&pending)?;
        file.write_all(&generation.to_le_bytes())?;
        file.write_all(&Sha256::digest(&body.0))?;
        file.write_all(&body.0)?;
        file.sync_all()?;
        fs::rename(
            pending,
            self.directory
                .join(format!("state-{}.snap", generation % 2)),
        )?;
        File::open(&self.directory)?.sync_all()?;
        self.generation = generation;
        self.check_files()
    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;
    fn directory(label: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "zensim-resume-store-{label}-{}",
            std::process::id()
        ));
        fs::create_dir(&path).unwrap();
        path
    }
    #[test]
    fn rotation_crash_fallback_and_exclusive_writer() {
        let root = directory("rotation");
        let run = "a".repeat(64);
        let mut store = Store::open(&root, &run).unwrap();
        assert!(Store::open(&root, &run).is_err());
        for step in 0..20_u64 {
            store.save(&vec![step, step + 1]).unwrap();
        }
        assert_eq!(store.load::<Vec<u64>>().unwrap().unwrap(), vec![19, 20]);
        assert_eq!(fs::read_dir(root.join(&run)).unwrap().count(), 3);
        fs::write(root.join(&run).join("state-0.snap"), b"torn").unwrap();
        fs::write(root.join(&run).join("writing.snap"), b"unfinished").unwrap();
        drop(store);
        let mut store = Store::open(&root, &run).unwrap();
        assert_eq!(store.load::<Vec<u64>>().unwrap().unwrap(), vec![18, 19]);
        store.save(&vec![21_u64]).unwrap();
        assert_eq!(store.load::<Vec<u64>>().unwrap().unwrap(), vec![21]);
        drop(store);
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn count_reservations_bound_completed_runs_too() {
        let root = directory("quota");
        for n in 0..63 {
            drop(Store::open(&root, &format!("{n:064x}")).unwrap());
        }
        assert!(Store::open(&root, &format!("{:064x}", 63)).is_err());
        assert!(63 * RUN_BYTES + METADATA_BYTES <= SHARED_BYTES);
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn oversized_state_refuses_before_overwriting_recoverable_bytes() {
        let root = directory("oversize");
        let run = "b".repeat(64);
        let mut store = Store::open(&root, &run).unwrap();
        store.save(&vec![3_u8]).unwrap();
        assert!(store.save(&"x".repeat(STATE_BYTES as usize)).is_err());
        assert_eq!(store.load::<Vec<u8>>().unwrap().unwrap(), vec![3]);
        assert!(!root.join(&run).join("writing.snap").exists());
        drop(store);
        fs::remove_dir_all(root).unwrap();
    }
}
