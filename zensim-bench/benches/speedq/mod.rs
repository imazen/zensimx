//! SPEEDQ score-parity preflight for the existing speed matrix.
//! Formula revisions and dispatch ceilings are isolated in separate processes.
//! Timing remains blocked until every required score-parity cell passes.
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::cell::RefCell;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use zenpredict::Model;
use zensim::{BakeScorer, RgbSlice, Zensim, ZensimProfile};

#[cfg(feature = "costcmp")]
mod costcmp;

const SOURCE_SHA: &str = "f803b74c4252952f337abdc0234c2930839d45dddc32ae9b8b5296d6c840f400";
const ID_SHA: &str = "0a6a20dc356acef3bef9deffc411f03189813e8b924fddcf7b22f7efea6b9f17";
const ID_BYTES: &str = include_str!("../../../benchmarks/costset2_2026-10-03.candidate_ids.json");

fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|v| format!("{v:02x}"))
        .collect()
}

fn geometry() -> (usize, usize) {
    let text = std::env::var("ZEN_S2_GEOMETRY").expect("explicit geometry");
    let (w, h) = text.split_once('x').expect("WIDTHxHEIGHT");
    let pair = (w.parse().unwrap(), h.parse().unwrap());
    assert!(pair.0 > 0 && pair.1 > 0);
    pair
}

#[cfg(target_arch = "x86_64")]
fn tier() -> String {
    use archmage::{SimdToken, X64V3Token, X64V4Token, X64V4xToken};
    let requested = std::env::var("ZEN_S2_TIER").expect("explicit tier");
    match requested.as_str() {
        "v4x" => assert!(X64V4xToken::summon().is_some(), "v4x unavailable"),
        "v4" => {
            X64V4xToken::dangerously_disable_token_process_wide(true).unwrap();
            assert!(X64V4Token::summon().is_some() && X64V4xToken::summon().is_none());
        }
        "v3" => {
            X64V4Token::dangerously_disable_token_process_wide(true).unwrap();
            assert!(X64V3Token::summon().is_some() && X64V4Token::summon().is_none());
        }
        "scalar" => {
            X64V3Token::dangerously_disable_token_process_wide(true).unwrap();
            assert!(X64V3Token::summon().is_none() && X64V4Token::summon().is_none());
        }
        _ => panic!("unknown tier"),
    }
    requested
}

#[cfg(not(target_arch = "x86_64"))]
fn tier() -> String {
    panic!("SPEEDQ dev grid requires x86_64");
}

fn worker(arm: &str) {
    let _guard = archmage::testing::lock_token_testing();
    let tier = tier();
    let (w, h) = geometry();
    let (src, dst) = super::test_pair(w, h);
    let mut hash = Sha256::new();
    hash.update(bytemuck::cast_slice::<[u8; 3], u8>(&src));
    hash.update(bytemuck::cast_slice::<[u8; 3], u8>(&dst));
    let input_sha = hash
        .finalize()
        .iter()
        .map(|v| format!("{v:02x}"))
        .collect::<String>();
    let src: &'static [[u8; 3]] = Box::leak(src.into_boxed_slice());
    let dst: &'static [[u8; 3]] = Box::leak(dst.into_boxed_slice());
    let mut model_info = Value::Null;
    let feature_values = Rc::new(RefCell::new(Vec::<f64>::new()));
    let keep_features = std::env::var_os("ZEN_S2_PARITY_FEATURES").is_some();
    #[cfg(feature = "costcmp")]
    let extra = costcmp::action(arm, src, dst, w, h);
    #[cfg(not(feature = "costcmp"))]
    let extra: Option<(Box<dyn FnMut() -> f64>, Value)> = None;
    let mut action: Box<dyn FnMut() -> f64> = if let Some((action, info)) = extra {
        model_info = info;
        action
    } else {
        match arm {
            "by_v2fy" => {
                let path = std::env::var("ZEN_S2_SPEEDQ_BAKE").expect("pinned bake path");
                let original = std::fs::read(path).unwrap();
                assert_eq!(
                    digest(&original),
                    SOURCE_SHA,
                    "frozen production model changed"
                );
                let revision = std::env::var("ZENSIM_FORMULA_REV").unwrap();
                assert!(matches!(revision.as_str(), "3" | "4" | "5"));
                // Existing metadata owner preserves weight sections without requantization.
                let bytes = zenpredict_bake::append_metadata_utf8(
                    &original,
                    "zentrain.formula_revision",
                    &revision,
                )
                .unwrap();
                let stamped_sha = digest(&bytes);
                let model = Box::leak(Box::new(Model::from_bytes(&bytes).unwrap()));
                assert_eq!(model.layers().next().unwrap().out_dim, 128, "H128 shape");
                assert_eq!(model.n_outputs(), 1);
                let mut scorer = BakeScorer::new(model).unwrap().with_parallel(true);
                assert_eq!(digest(ID_BYTES.as_bytes()), ID_SHA);
                let canonical: Value = serde_json::from_str(ID_BYTES).unwrap();
                let expected: Vec<u16> = canonical["candidates"]["by_v2fy"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap() as u16)
                    .collect();
                assert_eq!(expected.len(), 420);
                assert_eq!(scorer.consumed_feature_ids().unwrap(), expected);
                model_info = json!({"source_sha256": SOURCE_SHA, "stamped_sha256": stamped_sha,
                "revision": revision, "hidden":128, "consumed_ids":expected,
                "source_bytes":original.len(), "revision_metadata_only":true,
                "scope":"frozen seed-0 packed f16 production weights; metadata-only revision comparison"});
                let retained = Rc::clone(&feature_values);
                Box::new(move || {
                    let result = scorer
                        .compute(&RgbSlice::new(src, w, h), &RgbSlice::new(dst, w, h), None)
                        .unwrap();
                    if keep_features {
                        *retained.borrow_mut() = expected
                            .iter()
                            .map(|id| result.features()[usize::from(*id)])
                            .collect();
                    }
                    result.score()
                })
            }
            "zensim_B" => {
                assert_eq!(
                    std::env::var("ZENSIM_FORMULA_REV").unwrap(),
                    "1",
                    "B is serving Rev1"
                );
                let z = Zensim::new(ZensimProfile::B).with_parallel(true);
                Box::new(move || {
                    z.compute(&RgbSlice::new(src, w, h), &RgbSlice::new(dst, w, h))
                        .unwrap()
                        .score()
                })
            }
            "fast_ssim2" => Box::new(move || {
                fast_ssim2::compute_ssimulacra2(
                    imgref::Img::new(&*src, w, h),
                    imgref::Img::new(&*dst, w, h),
                )
                .unwrap()
            }),
            "butteraugli" => Box::new(move || {
                let rs: &[rgb::RGB8] = bytemuck::cast_slice(&*src);
                let ds: &[rgb::RGB8] = bytemuck::cast_slice(&*dst);
                butteraugli::butteraugli(
                    imgref::Img::new(rs, w, h),
                    imgref::Img::new(ds, w, h),
                    &butteraugli::ButteraugliParams::default(),
                )
                .unwrap()
                .score
            }),
            "ssimulacra2_rs" => {
                // Same boundary as the existing peer: sRGB widening is untimed.
                let rs = super::make_f32_srgb(src);
                let ds = super::make_f32_srgb(dst);
                Box::new(move || {
                    let s = ssimulacra2::Rgb::new(
                        rs.clone(),
                        w,
                        h,
                        ssimulacra2::TransferCharacteristic::SRGB,
                        ssimulacra2::ColorPrimaries::BT709,
                    )
                    .unwrap();
                    let d = ssimulacra2::Rgb::new(
                        ds.clone(),
                        w,
                        h,
                        ssimulacra2::TransferCharacteristic::SRGB,
                        ssimulacra2::ColorPrimaries::BT709,
                    )
                    .unwrap();
                    ssimulacra2::compute_frame_ssimulacra2(s, d).unwrap()
                })
            }
            _ => panic!("unknown SPEEDQ arm"),
        }
    };
    let score = action();
    assert!(score.is_finite(), "nonfinite score");
    let bits = score.to_bits();
    // RSS-only runs record the pool size too for by_v2fy, whose walk has
    // already built the global pool; other arms keep the untouched pool.
    let actual_threads = if std::env::var_os("ZEN_S2_RSS_ONLY").is_none()
        || arm == "by_v2fy"
        || std::env::var_os("ZEN_S2_VERIFY_THREADS").is_some()
    {
        let actual = rayon::current_num_threads();
        assert_eq!(
            actual,
            std::env::var("RAYON_NUM_THREADS")
                .unwrap()
                .parse::<usize>()
                .unwrap()
        );
        Some(actual)
    } else {
        None
    };
    let ready = json!({"pid":std::process::id(),"arm":arm,"tier":tier,"width":w,"height":h,
        "score":score,"score_bits":format!("{bits:016x}"),"input_sha256":input_sha,
        "model":model_info,"threads":std::env::var("RAYON_NUM_THREADS").unwrap(),
        "ssim2_rayon":cfg!(feature="ssim2-rayon"),"feature_values":*feature_values.borrow(),"actual_rayon_threads":actual_threads});
    if std::env::var_os("ZEN_S2_RSS_ONLY").is_some() {
        // Explicit profiler-only repetitions reuse the serving action and its
        // score-bit guard; normal parity/RSS and paired timing are unchanged.
        if let Ok(calls) = std::env::var("ZEN_S2_PROFILE_CALLS") {
            for _ in 0..calls.parse::<usize>().expect("profile call count") {
                assert_eq!(action().to_bits(), bits, "profile score changed");
            }
        }
        println!("{ready}");
        return;
    }
    for _ in 0..2 {
        assert_eq!(action().to_bits(), bits, "same-worker warmup score changed");
    }
    let listener = UnixListener::bind(std::env::var("ZEN_S2_SOCKET").unwrap()).unwrap();
    println!("{ready}");
    std::io::stdout().flush().unwrap();
    let (mut stream, _) = listener.accept().unwrap();
    writeln!(stream, "{ready}").unwrap();
    let reader = BufReader::new(stream.try_clone().unwrap());
    for line in reader.lines() {
        let line = line.unwrap();
        if line == "quit" {
            break;
        }
        assert_eq!(line, "call");
        let start = Instant::now();
        let score = action();
        let elapsed = start.elapsed().as_nanos() as u64;
        assert_eq!(score.to_bits(), bits, "same-worker timing score changed");
        writeln!(stream, "{elapsed}").unwrap();
    }
}

struct Worker {
    input: UnixStream,
    output: BufReader<UnixStream>,
    ready: Value,
}
impl Worker {
    fn new(name: &str) -> Self {
        let paths: Value =
            serde_json::from_str(&std::env::var("ZEN_S2_WORKER_SOCKETS").unwrap()).unwrap();
        let input = UnixStream::connect(paths[name].as_str().unwrap()).unwrap();
        let mut output = BufReader::new(input.try_clone().unwrap());
        let mut line = String::new();
        assert!(
            output.read_line(&mut line).unwrap() > 0,
            "worker READY missing"
        );
        let ready = serde_json::from_str(&line).unwrap();
        Self {
            input,
            output,
            ready,
        }
    }
    fn call(&mut self) -> u64 {
        writeln!(self.input, "call").unwrap();
        let mut line = String::new();
        assert!(
            self.output.read_line(&mut line).unwrap() > 0,
            "worker stopped"
        );
        line.trim().parse().unwrap()
    }
}
impl Drop for Worker {
    fn drop(&mut self) {
        let _ = writeln!(self.input, "quit");
    }
}
pub(super) fn run() {
    if let Ok(arm) = std::env::var("ZEN_S2_SPEEDQ_WORKER") {
        worker(&arm);
        return;
    }
    let (w, h) = geometry();
    let dest = std::path::PathBuf::from(std::env::var("ZENBENCH_RESULT_PATH").unwrap());
    assert!(!dest.exists(), "fresh raw result required");
    let declarations = [
        ("by_v2fy_r3", "by_v2fy", "3"),
        ("by_v2fy_r4", "by_v2fy", "4"),
        ("by_v2fy_r5", "by_v2fy", "5"),
        #[cfg(feature = "costcmp")]
        ("by_v2fy_r5_before", "by_v2fy", "5"),
        #[cfg(feature = "costcmp")]
        ("by_v2fy_r5_b64", "by_v2fy", "5"),
        #[cfg(feature = "costcmp")]
        ("by_v2fy_r5_b128", "by_v2fy", "5"),
        #[cfg(feature = "costcmp")]
        ("by_v2fy_r5_b256", "by_v2fy", "5"),
        #[cfg(feature = "costcmp")]
        ("by_v2fy_r5_floor3", "by_v2fy", "5"),
        #[cfg(feature = "costcmp")]
        ("by_v2fy_r5_floor17", "by_v2fy", "5"),
        ("zensim_B", "zensim_B", "1"),
        ("fast_ssim2", "fast_ssim2", "1"),
        ("butteraugli", "butteraugli", "1"),
        ("ssimulacra2_rs", "ssimulacra2_rs", "1"),
        #[cfg(feature = "costcmp")]
        ("zensim_A", "zensim_A", "1"),
        #[cfg(feature = "costcmp")]
        ("fast_ssim2_main", "fast_ssim2_main", "1"),
        #[cfg(feature = "costcmp")]
        ("e33_a", "e33_a", "5"),
        #[cfg(feature = "costcmp")]
        ("e33_c", "e33_c", "5"),
        #[cfg(feature = "costcmp")]
        ("zensim_D", "zensim_D", "1"),
        #[cfg(feature = "costcmp")]
        ("e33_a_map", "e33_a_map", "5"),
        #[cfg(feature = "costcmp")]
        ("e33_c_map", "e33_c_map", "5"),
        #[cfg(feature = "costcmp")]
        ("e33_seed0_map", "e33_seed0_map", "5"),
        #[cfg(feature = "costcmp")]
        ("e34_control", "e34_control", "5"),
        #[cfg(feature = "costcmp")]
        ("e34_candidate", "e34_candidate", "5"),
        #[cfg(feature = "costcmp")]
        ("e34_candidate_cached", "e34_candidate_cached", "5"),
        #[cfg(feature = "costcmp")]
        ("e34_candidate_prepare", "e34_candidate_prepare", "5"),
        #[cfg(feature = "costcmp")]
        ("e34_candidate_serial", "e34_candidate_serial", "5"),
        #[cfg(feature = "costcmp")]
        ("e34_candidate_map_serial", "e34_candidate_map_serial", "5"),
        #[cfg(feature = "costcmp")]
        ("e34_candidate_map", "e34_candidate_map", "5"),
        #[cfg(feature = "costcmp")]
        ("e34_candidate_map_before", "e34_candidate_map_before", "5"),
        #[cfg(feature = "costcmp")]
        ("e34_candidate_map_bin2", "e34_candidate_map_bin2", "5"),
        #[cfg(feature = "costcmp")]
        ("e34_candidate_map_bin8", "e34_candidate_map_bin8", "5"),
        #[cfg(feature = "costcmp")]
        ("e34_candidate_map_bin16", "e34_candidate_map_bin16", "5"),
        #[cfg(feature = "costcmp")]
        ("e34_candidate_baseline", "e34_candidate_baseline", "5"),
        #[cfg(feature = "costcmp")]
        (
            "e34_candidate_map_retained",
            "e34_candidate_map_retained",
            "5",
        ),
        #[cfg(feature = "costcmp")]
        ("e34_butter_scalar", "e34_butter_scalar", "5"),
        #[cfg(feature = "costcmp")]
        ("e34_butter_map", "e34_butter_map", "5"),
        #[cfg(feature = "costcmp")]
        ("e34_butter_baseline", "e34_butter_baseline", "5"),
        #[cfg(feature = "costcmp")]
        ("zensim_B_rev5", "zensim_B_rev5", "5"),
        #[cfg(feature = "costcmp")]
        ("r7_general", "r7_general", "7"),
        #[cfg(feature = "costcmp")]
        ("r7_codec", "r7_codec", "7"),
        #[cfg(feature = "costcmp")]
        ("r7_general_cached", "r7_general_cached", "7"),
        #[cfg(feature = "costcmp")]
        ("r7_codec_cached", "r7_codec_cached", "7"),
        #[cfg(feature = "costcmp")]
        ("r7_general_map", "r7_general_map", "7"),
        #[cfg(feature = "costcmp")]
        ("r7_codec_map", "r7_codec_map", "7"),
    ];
    let only = std::env::var("ZEN_S2_ARMS").expect("explicit arm inventory");
    let mut owners = Vec::new();
    let mut metadata = serde_json::Map::new();
    for (name, arm, revision) in declarations {
        if !only.split(',').any(|s| s == name) {
            continue;
        }
        let worker = Worker::new(name);
        assert_eq!(worker.ready["arm"], arm);
        if arm == "by_v2fy" {
            assert_eq!(worker.ready["model"]["revision"], revision);
        }
        metadata.insert(name.into(), worker.ready.clone());
        owners.push((
            name,
            Arc::new(Mutex::new(worker)),
            Arc::new(Mutex::new(Vec::<u64>::new())),
        ));
    }
    assert_eq!(
        owners.len(),
        only.split(',').count(),
        "unknown/duplicate arm"
    );
    let traces: Vec<_> = owners
        .iter()
        .map(|(name, _, calls)| (*name, Arc::clone(calls)))
        .collect();
    let rounds = super::env_usize("ZEN_S2_ROUNDS", 32);
    assert_eq!(rounds, 32, "SPEEDQ retains exactly 32 clean rounds");
    let mut gate_clean = Vec::new();
    let mut inner: serde_json::Map<String, Value> = traces
        .iter()
        .map(|(name, _)| ((*name).into(), json!([])))
        .collect();
    let mut batches = Vec::new();
    let mut unreliable = false;
    let mut gate_waits = 0;
    let mut timer_resolution_ns = 0;
    let attempt_start = Instant::now();
    loop {
        let clean = gate_clean.iter().filter(|v| **v == Some(true)).count();
        if clean == rounds || gate_clean.len() == 64 || unreliable {
            break;
        }
        // A batch cannot reach the target before its last round: it requests
        // only the number still missing, bounded by the remaining attempt cap.
        let batch_rounds = (rounds - clean).min(64 - gate_clean.len());
        let remaining = Duration::from_secs(3600).saturating_sub(attempt_start.elapsed());
        if remaining.is_zero() {
            break;
        }
        let warmup = if batches.is_empty() {
            Duration::from_millis(20)
        } else {
            Duration::ZERO
        };
        let result = zenbench::run(|suite| {
            suite.compare(format!("speedq_{w}x{h}"), |group| {
                group
                    .config()
                    .max_rounds(batch_rounds)
                    .min_rounds(batch_rounds)
                    .max_wall_time(remaining)
                    .warmup_time(warmup)
                    .auto_rounds(false);
                group.config().min_iterations = 1;
                group.config().max_iterations = 1;
                for (name, owner, calls) in &owners {
                    let owner = Arc::clone(owner);
                    let calls = Arc::clone(calls);
                    group.bench(*name, move |b| {
                        b.iter(|| {
                            let ns = owner.lock().unwrap().call();
                            calls.lock().unwrap().push(ns);
                            zenbench::black_box(ns)
                        })
                    });
                }
            });
        });
        let comp = &result.comparisons[0];
        assert!(
            comp.samples.iter().all(|s| s.iterations == 1),
            "one call per round"
        );
        assert!(comp.samples.len() <= batch_rounds);
        for (name, trace) in &traces {
            let calls = trace.lock().unwrap();
            assert!(calls.len() >= comp.samples.len());
            // Each batch has untimed zenbench estimation calls. Its final N
            // calls are exactly the N completed, paired measurement rounds.
            inner[*name].as_array_mut().unwrap().extend(
                calls[calls.len() - comp.samples.len()..]
                    .iter()
                    .map(|v| json!(v)),
            );
        }
        let completed = comp.samples.len();
        gate_clean.extend(comp.samples.iter().map(|s| s.gate_clean));
        unreliable |= result.unreliable;
        gate_waits += result.gate_waits;
        timer_resolution_ns = timer_resolution_ns.max(result.timer_resolution_ns);
        let path = dest.with_file_name(format!("zenbench.batch-{}.json", batches.len()));
        assert!(!path.exists());
        result.save(&path).unwrap();
        batches.push(json!({"path":path.file_name().unwrap().to_str().unwrap(),
            "rounds_requested":batch_rounds,"rounds_total":completed,
            "round_offset":gate_clean.len()-completed,"warmup_ms":warmup.as_millis()}));
        if completed < batch_rounds {
            break;
        }
    }
    let raw = json!({"schema":"speedq-parent-batches-v1","batches":batches,
        "rounds_total":gate_clean.len(),"round_cap":64});
    let mut parent = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&dest)
        .unwrap();
    writeln!(parent, "{}", serde_json::to_string_pretty(&raw).unwrap()).unwrap();
    let report = json!({"schema":"speedq-inner-rounds-v3","workers":metadata,
        "paired_rounds":inner,"zenbench_gate_waits":gate_waits,
        "zenbench_unreliable":unreliable,"gate_clean":gate_clean,
        "timer_resolution_ns":timer_resolution_ns,"parent_rounds_include_ipc":true,"worker_rounds_exclude_ipc":true,
        "sample_alignment":"per-batch final N calls after untimed estimation; one shared gate flag per complete paired round"});
    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(dest.with_extension("inner.json"))
        .unwrap();
    writeln!(f, "{}", serde_json::to_string_pretty(&report).unwrap()).unwrap();
}
