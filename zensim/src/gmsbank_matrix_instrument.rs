// Explicit, private qualification instrument shared byte-for-byte with the
// frozen corrected parent. No alternate feature extractor or model surface.
#[test]
fn gmsbank_chroma_prefix_dump() {
    use std::io::Write;
    use sha2::{Digest, Sha256};
    let root = std::path::PathBuf::from(std::env::var("GMSBANK_MATRIX_DIR").unwrap());
    let output = std::path::PathBuf::from(std::env::var("GMSBANK_MATRIX_OUT").unwrap());
    let tier = std::env::var("GMSBANK_MATRIX_TIER").unwrap();
    match tier.as_str() {
        "native" => (),
        "v3" => { archmage::X64V4Token::dangerously_disable_token_process_wide(true).unwrap(); }
        "scalar" => { archmage::X64V2Token::dangerously_disable_token_process_wide(true).unwrap(); }
        _ => panic!("unknown tier"),
    }
    let on = std::env::var("GMSBANK_MATRIX_ARM").unwrap() == "on";
    let parallel = std::env::var("RAYON_NUM_THREADS").unwrap() != "1";
    let records: Vec<serde_json::Value> = serde_json::from_slice(
        &std::fs::read(root.join("population.json")).unwrap()).unwrap();
    assert_eq!(records.len(), 144);
    std::fs::create_dir(&output).unwrap();
    let mut inputs = Vec::new();
    for set in ["cid22", "safesyn", "kadid"] {
        let mut file = std::io::BufWriter::new(std::fs::OpenOptions::new()
            .write(true).create_new(true).open(output.join(format!("{set}.csv"))).unwrap());
        write!(file,"ref_basename,pair_key").unwrap();
        for i in 0..if on { 1502 } else { 1322 } { write!(file,",f{i}").unwrap(); }
        writeln!(file).unwrap();
        for row in records.iter().filter(|r|r["set"] == set) {
            let read = |side: &str| {
                let expected = row[format!("{side}_pixels_sha256")].as_str();
                let (bytes,w,h) = if let Some(digest) = expected {
                    (std::fs::read(root.join("rgb").join(format!("{digest}.rgb"))).unwrap(),
                     row["width"].as_u64().unwrap() as usize,
                     row["height"].as_u64().unwrap() as usize)
                } else {
                    // A few frozen KADID gate rows are absent from key parquet.
                    // Decode their RGB8 PNG through the same zenpng owner;
                    // refuse any format/conversion requiring interpretation.
                    let name = row[format!("{side}_path")].as_str().unwrap();
                    assert_eq!(std::path::Path::new(name).extension().unwrap(), "png");
                    let data = std::fs::read(root.join("encoded").join(name.trim_start_matches('/'))).unwrap();
                    let decoded = zenpng::decode(&data, &zenpng::PngDecodeConfig::default(), &enough::Unstoppable).unwrap();
                    let pb = decoded.pixels;
                    assert_eq!(pb.format().layout, zenpixels::SampleLayout::RGB8);
                    let (w,h) = (pb.width() as usize,pb.height() as usize);
                    let slice = pb.as_view();
                    let mut bytes = Vec::with_capacity(w*h*3);
                    for y in 0..h { bytes.extend_from_slice(&slice.row(y as u32)[..w*3]); }
                    (bytes,w,h)
                };
                assert_eq!(bytes.len(), w*h*3);
                let digest = Sha256::digest(&bytes).iter().map(|b| format!("{b:02x}")).collect::<String>();
                if let Some(expected) = expected { assert_eq!(digest,expected); }
                (bytes.as_chunks::<3>().0.to_vec(),w,h,digest)
            };
            let (src,w,h,a) = read("ref");
            let (dst,dw,dh,b) = read("dist");
            assert_eq!((w,h),(dw,dh));
            let mut scratch = V2Scratch::new();
            let values = compute_folded720_streaming_impl(
                &RgbSlice::new(&src,w,h), &RgbSlice::new(&dst,w,h), None,parallel,
                V2NewFeatureToggles { gmsbank: on, ..rev4_all_toggles() },
                &mut scratch,None).unwrap().into_features();
            assert_eq!(values.len(),if on { 1502 } else { 1322 });
            let name = std::path::Path::new(row["ref_path"].as_str().unwrap()).file_name().unwrap().to_str().unwrap();
            write!(file,"{name},{}",row["pair_key"].as_str().unwrap()).unwrap();
            for value in values { write!(file,",{value:.17e}").unwrap(); }
            writeln!(file).unwrap();
            inputs.push(serde_json::json!({"pair_key":row["pair_key"],"width":w,"height":h,"ref_sha":a,"dist_sha":b}));
        }
    }
    std::fs::write(output.join("pixels.json"),serde_json::to_vec_pretty(&inputs).unwrap()).unwrap();
    println!("matrix_rows={} tier={tier} parallel={parallel} on={on}",inputs.len());
}
