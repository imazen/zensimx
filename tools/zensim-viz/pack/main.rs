//! Existing native IMIJ-set writer/reader, with full byte verification.
#![forbid(unsafe_code)]
use imij::planes::SeekReader;
use imij_set::{EntryMetadata, IndexPlacement, Reader, Summary, Writer};
use sha2::{Digest, Sha256};
use std::{
    fs::{File, OpenOptions},
    io::BufWriter,
};
fn hash(b: &[u8]) -> String {
    Sha256::digest(b)
        .iter()
        .map(|v| format!("{v:02x}"))
        .collect()
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let output = args.next().ok_or("output archive required")?;
    let mut files = Vec::new();
    for arg in args {
        let (name, path) = arg.split_once('=').ok_or("NAME=PATH required")?;
        if name.starts_with('/') || name.contains("..") || name.contains('\\') {
            return Err("relative entry name required".into());
        }
        files.push((name.to_owned(), std::fs::read(path)?));
    }
    if files.is_empty() {
        return Err("no entries".into());
    }
    let sink = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&output)?;
    let mut writer =
        Writer::new(BufWriter::new(sink)).with_index_placement(IndexPlacement::AtCommit)?;
    for (key, (name, bytes)) in files.iter().enumerate() {
        let mut metadata = EntryMetadata::default();
        metadata.filename = Some(name.clone());
        writer.insert_with_metadata(
            key as u64,
            bytes.len() as u64,
            0,
            Summary::default(),
            &metadata,
            &mut bytes.as_slice(),
        )?;
    }
    let (_, snapshot) = writer.finish()?;
    let mut reader = Reader::open(SeekReader::new(File::open(&output)?))?;
    let mut records = Vec::new();
    for (key, (name, expected)) in files.iter().enumerate() {
        let entry = reader.get(key as u64)?.ok_or("missing entry")?;
        let mut decoded = Vec::new();
        reader.copy_to(&entry, &mut decoded)?; // verifies native XXH3 too
        if &decoded != expected || entry.metadata().filename.as_deref() != Some(name) {
            return Err("archive roundtrip differs".into());
        }
        records.push(serde_json::json!({"name":name,"offset":entry.offset(),"length":entry.len(),"sha256":hash(&decoded)}));
    }
    println!(
        "{}",
        serde_json::json!({"bytes":std::fs::metadata(output)?.len(),"generation":snapshot.generation(),"entries":records,"native_roundtrip_verified":true})
    );
    Ok(())
}
