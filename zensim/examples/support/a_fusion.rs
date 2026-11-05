//! Two-binary extraction experiments using the cost study's frozen RGB pairs.
//! The persistent-worker/inner-timer protocol follows SPEEDQ. Only serving calls
//! are timed; parent zenbench samples provide shared paired-round admission.
use super::*;
use serde_json::{Value, json};
use std::io::{BufRead, BufReader};
use std::os::unix::net::{UnixListener, UnixStream};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Instant;

const BAKE_SHA: &str = "185f43f7e5e1327c34ad18f0b73b96b872663726a0fac6ff524430d0d8430088";

// Report the token selected by the same incant ordering as the metric kernels.
#[archmage::magetypes(v4x, v4, v3, neon, wasm128, scalar)]
fn fusion_dispatch_token(token: Token) -> &'static str {
    std::any::type_name_of_val(&token)
}

fn selected_token() -> &'static str {
    archmage::incant!(
        fusion_dispatch_token(),
        [v4x, v4, v3, neon, wasm128, scalar]
    )
}

fn full_feature_hash(features: &[f64]) -> String {
    hash(
        &features
            .iter()
            .flat_map(|value| value.to_bits().to_le_bytes())
            .collect::<Vec<_>>(),
    )
}

fn worker(args: &[String]) {
    assert_eq!(args.len(), 4, "worker PAIRS BAKE SIZE THREADS");
    let threads: usize = args[3].parse().unwrap();
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .unwrap();
    let bytes = std::fs::read(&args[1]).unwrap();
    assert_eq!(hash(&bytes), BAKE_SHA);
    let model = zenpredict::Model::from_bytes(&bytes).unwrap();
    let mut scorer = BakeScorer::new(&model).unwrap().with_parallel(threads > 1);
    let ids = scorer.consumed_feature_ids().unwrap();
    let loaded = load(Path::new(&args[0]), &args[2]);
    assert_eq!(loaded.len(), 20);
    let mut records = Vec::new();
    for item in &loaded {
        let p = &item.pair;
        let result = scorer
            .compute(
                &RgbSlice::new(&item.reference, p.width, p.height),
                &RgbSlice::new(&item.distorted, p.width, p.height),
                None,
            )
            .unwrap();
        records.push(json!({"pair":p,"score":result.score(),"score_bits":result.score().to_bits(),
            "consumed_bits":ids.iter().map(|&i| result.features()[usize::from(i)].to_bits()).collect::<Vec<_>>()}));
    }
    let ready = json!({"pid":std::process::id(),"threads":threads,"bake_sha256":BAKE_SHA,"selected_token":selected_token(),
        "build_commit":option_env!("ZENSIM_BUILD_COMMIT"),"consumed_ids":ids,"records":records});
    let listener = UnixListener::bind(std::env::var_os("A_FUSION_SOCKET").unwrap()).unwrap();
    println!("{ready}");
    std::io::stdout().flush().unwrap();
    let (mut stream, _) = listener.accept().unwrap();
    writeln!(stream, "{ready}").unwrap();
    for command in BufReader::new(stream.try_clone().unwrap()).lines() {
        let command = command.unwrap();
        if command == "quit" {
            break;
        }
        assert_eq!(command, "all");
        let mut times = Vec::with_capacity(20);
        for (index, item) in loaded.iter().enumerate() {
            let p = &item.pair;
            let r = RgbSlice::new(&item.reference, p.width, p.height);
            let d = RgbSlice::new(&item.distorted, p.width, p.height);
            let start = Instant::now();
            let result = scorer.compute(&r, &d, None).unwrap();
            let ns = start.elapsed().as_nanos() as u64;
            assert_eq!(
                result.score().to_bits(),
                records[index]["score_bits"].as_u64().unwrap()
            );
            std::hint::black_box(result);
            times.push(ns);
        }
        writeln!(stream, "{}", serde_json::to_string(&times).unwrap()).unwrap();
    }
}

struct Launcher {
    child: Child,
    ready: Value,
}

impl Launcher {
    fn new(binary: &str, args: &[String], out: &Path, arm: &str) -> Self {
        let stderr = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(out.join(format!("{arm}.stderr")))
            .unwrap();
        let wrapper = PathBuf::from(std::env::var_os("HOME").unwrap())
            .join("work/claudehints/scripts/run-heavy");
        let memory = std::env::var("A_FUSION_WORKER_MEM").unwrap_or_else(|_| "16G".into());
        assert!(matches!(memory.as_str(), "16G" | "24G"));
        let mut child = Command::new(wrapper)
            .args([
                "--mem",
                &memory,
                "--jobs",
                "16",
                "--",
                "/usr/bin/time",
                "-v",
                "-o",
            ])
            .arg(out.join(format!("{arm}.time.txt")))
            .arg(binary)
            .env("A_FUSION_SOCKET", out.join(format!("{arm}.sock")))
            .args(["fusion", "worker"])
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(stderr)
            .spawn()
            .unwrap();
        let mut output = BufReader::new(child.stdout.take().unwrap());
        let mut line = String::new();
        assert!(
            output.read_line(&mut line).unwrap() > 0,
            "worker READY missing"
        );
        Self {
            child,
            ready: serde_json::from_str(&line).unwrap(),
        }
    }
}
impl Drop for Launcher {
    fn drop(&mut self) {
        assert!(self.child.wait().unwrap().success(), "worker failed");
    }
}

struct Worker {
    input: UnixStream,
    output: BufReader<UnixStream>,
    ready: Value,
}
impl Worker {
    fn new(out: &Path, arm: &str) -> Self {
        let input = UnixStream::connect(out.join(format!("{arm}.sock"))).unwrap();
        let mut output = BufReader::new(input.try_clone().unwrap());
        let mut line = String::new();
        assert!(output.read_line(&mut line).unwrap() > 0);
        Self {
            input,
            output,
            ready: serde_json::from_str(&line).unwrap(),
        }
    }
    fn call(&mut self) -> Vec<u64> {
        writeln!(self.input, "all").unwrap();
        let mut line = String::new();
        assert!(self.output.read_line(&mut line).unwrap() > 0);
        let times: Vec<u64> = serde_json::from_str(&line).unwrap();
        assert_eq!(times.len(), 20);
        times
    }
}
impl Drop for Worker {
    fn drop(&mut self) {
        let _ = writeln!(self.input, "quit");
    }
}

fn write_new(path: &Path, value: &Value) {
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .unwrap();
    writeln!(file, "{}", serde_json::to_string_pretty(value).unwrap()).unwrap();
}

fn compare(args: &[String]) {
    assert_eq!(
        args.len(),
        7,
        "compare BEFORE AFTER PAIRS BAKE SIZE THREADS OUT"
    );
    let out = Path::new(&args[6]);
    std::fs::create_dir_all(out).unwrap();
    if std::env::var_os("A_FUSION_OWNED_PEERS").is_none() {
        let launchers: Vec<_> = ["before", "after"]
            .iter()
            .enumerate()
            .map(|(i, arm)| Launcher::new(&args[i], &args[2..6], out, arm))
            .collect();
        let mut pids = std::env::var("ZENBENCH_LAUNCHER_PIDS").unwrap_or_default();
        for launcher in &launchers {
            if !pids.is_empty() {
                pids.push(',');
            }
            pids.push_str(&launcher.ready["pid"].as_u64().unwrap().to_string());
        }
        let status = Command::new(std::env::current_exe().unwrap())
            .args(["fusion", "compare"])
            .args(args)
            .env("A_FUSION_OWNED_PEERS", "1")
            .env("ZENBENCH_LAUNCHER_PIDS", pids)
            .status()
            .unwrap();
        assert!(status.success(), "paired bench subprocess failed");
        drop(launchers);
        return;
    }
    let owners: Vec<_> = ["before", "after"]
        .into_iter()
        .map(|arm| {
            (
                arm,
                Arc::new(Mutex::new(Worker::new(out, arm))),
                Arc::new(Mutex::new(Vec::<Vec<u64>>::new())),
            )
        })
        .collect();
    let ready: Vec<_> = owners
        .iter()
        .map(|(_, owner, _)| owner.lock().unwrap().ready.clone())
        .collect();
    write_new(
        &out.join("workers.json"),
        &json!({"before":ready[0],"after":ready[1]}),
    );
    assert_eq!(ready[0]["consumed_ids"], ready[1]["consumed_ids"]);
    for (index, before) in ready[0]["records"].as_array().unwrap().iter().enumerate() {
        assert_eq!(before["pair"], ready[1]["records"][index]["pair"]);
        if std::env::var_os("A_FUSION_ALLOW_DRIFT").is_none() {
            assert_eq!(
                *before, ready[1]["records"][index],
                "exact candidate changes bits"
            );
        }
    }
    let id = &args[4];
    let cell = out;
    let mut flags = Vec::new();
    let mut inner = [Vec::new(), Vec::new()];
    let mut batch = 0;
    let mut unreliable = false;
    while flags.iter().filter(|v| **v == Some(true)).count() < 32 && flags.len() < 64 && !unreliable
    {
        let rounds =
            (32 - flags.iter().filter(|v| **v == Some(true)).count()).min(64 - flags.len());
        let result = zenbench::run(|suite| {
            suite.compare(format!("fusion_{id}"), |g| {
                let config = g.config();
                config
                    .min_rounds(rounds)
                    .max_rounds(rounds)
                    .auto_rounds(false)
                    .warmup_time(if batch == 0 {
                        Duration::from_millis(100)
                    } else {
                        Duration::ZERO
                    })
                    .max_time(Duration::from_secs(1800))
                    .max_wall_time(Duration::from_secs(1800));
                config.min_iterations = 1;
                config.max_iterations = 1;
                config.min_sample_ns = 0;
                config.stack_jitter = false;
                g.baseline("before");
                for (arm, owner, calls) in &owners {
                    let (owner, calls) = (Arc::clone(owner), Arc::clone(calls));
                    g.bench(*arm, move |b| {
                        b.iter(|| {
                            let times = owner.lock().unwrap().call();
                            let total: u64 = times.iter().sum();
                            calls.lock().unwrap().push(times);
                            std::hint::black_box(total)
                        })
                    });
                }
            });
        });
        result
            .save(cell.join(format!("batch-{batch}.json")))
            .unwrap();
        let samples = &result.comparisons[0].samples;
        assert!(samples.iter().all(|s| s.iterations == 1));
        for (i, (_, _, calls)) in owners.iter().enumerate() {
            let calls = calls.lock().unwrap();
            assert!(calls.len() >= samples.len());
            inner[i].extend_from_slice(&calls[calls.len() - samples.len()..]);
        }
        flags.extend(samples.iter().map(|s| s.gate_clean));
        unreliable |= result.unreliable;
        batch += 1;
        if samples.is_empty() {
            break;
        }
    }
    let selected: Vec<_> = flags
        .iter()
        .enumerate()
        .filter_map(|(i, f)| (*f == Some(true)).then_some(i))
        .take(32)
        .collect();
    let accepted = selected.len() == 32 && !unreliable;
    write_new(
        &cell.join("inner.json"),
        &json!({"before_ns":inner[0],"after_ns":inner[1],
            "gate_clean":flags,"selected":selected,"unreliable":unreliable,"accepted":accepted,
            "warmup_ms_per_arm":100,"images_per_round":20,
            "timer_boundary":"BakeScorer::compute per image only; input views, IPC, IO and result drop excluded"}),
    );
    println!("completed {id} accepted={accepted}");
    assert!(
        accepted,
        "fewer than 32 clean rounds; preserve attempt and retry separately"
    );
}

// Freeze one decode per distinct encoded source. Both binaries later consume
// these exact RGB bytes; decoder-era differences are upstream of this gate.
fn prepare_eval(args: &[String]) {
    assert_eq!(args.len(), 2, "prepare-eval TABLE NEW_DIRECTORY");
    let out = Path::new(&args[1]);
    std::fs::create_dir(out).unwrap();
    let text = std::fs::read_to_string(&args[0]).unwrap();
    let mut lines = text.lines();
    let header: Vec<_> = lines.next().unwrap().split('\t').collect();
    let column = |name| header.iter().position(|&key| key == name).unwrap();
    let (rc, dc, kc) = (column("ref_path"), column("dist_path"), column("pair_key"));
    let mut sources = std::collections::BTreeMap::<String, Value>::new();
    let mut table = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(out.join("pairs.tsv"))
        .unwrap();
    writeln!(
        table,
        "ref_path\tdist_path\tpair_key\twidth\theight\tref_pixels_sha256\tdist_pixels_sha256"
    )
    .unwrap();
    let mut count = 0;
    for line in lines {
        let row: Vec<_> = line.split('\t').collect();
        for index in [rc, dc] {
            if !sources.contains_key(row[index]) {
                let path = Path::new(row[index]);
                let (pixels, width, height) = zen_io::decode_rgb8(path);
                let rgb = bytemuck::cast_slice(&pixels);
                let sha = hash(rgb);
                let file = out.join(format!("{sha}.rgb"));
                if file.exists() {
                    assert_eq!(hash(&std::fs::read(&file).unwrap()), sha);
                } else {
                    let mut f = std::fs::OpenOptions::new()
                        .write(true)
                        .create_new(true)
                        .open(&file)
                        .unwrap();
                    f.write_all(rgb).unwrap();
                }
                sources.insert(
                    row[index].to_string(),
                    json!({"source":row[index],
                    "source_sha256":hash(&std::fs::read(path).unwrap()), "pixels_sha256":sha,
                    "width":width,"height":height,"raw_rgb_path":file}),
                );
            }
        }
        let (r, d) = (&sources[row[rc]], &sources[row[dc]]);
        assert_eq!(
            (r["width"].clone(), r["height"].clone()),
            (d["width"].clone(), d["height"].clone())
        );
        writeln!(
            table,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}",
            r["raw_rgb_path"].as_str().unwrap(),
            d["raw_rgb_path"].as_str().unwrap(),
            row[kc],
            r["width"],
            r["height"],
            r["pixels_sha256"].as_str().unwrap(),
            d["pixels_sha256"].as_str().unwrap()
        )
        .unwrap();
        count += 1;
        if count % 100 == 0 {
            table.flush().unwrap();
            eprintln!("frozen EVAL pixels {count}/4678");
        }
    }
    assert_eq!(count, 4678);
    table.flush().unwrap();
    write_new(
        &out.join("manifest.json"),
        &json!({"input_table_sha256":hash(text.as_bytes()),
        "rows":count,"jpeg_decoder":"zenjpeg 0.8.4", "sources":sources,
        "pairs_sha256":hash(&std::fs::read(out.join("pairs.tsv")).unwrap())}),
    );
}

fn pairs(args: &[String]) {
    assert_eq!(args.len(), 3, "pairs TABLE BAKE OUTPUT.jsonl");
    let bytes = std::fs::read(&args[1]).unwrap();
    assert_eq!(hash(&bytes), BAKE_SHA);
    let model = zenpredict::Model::from_bytes(&bytes).unwrap();
    let mut scorer = BakeScorer::new(&model).unwrap().with_parallel(false);
    let consumed = scorer.consumed_feature_ids().unwrap();
    let text = std::fs::read_to_string(&args[0]).unwrap();
    let mut lines = text.lines();
    let header: Vec<_> = lines.next().unwrap().split('\t').collect();
    let column = |name| header.iter().position(|&key| key == name).unwrap();
    let (rc, dc, kc) = (column("ref_path"), column("dist_path"), column("pair_key"));
    let mut output = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[2])
        .unwrap();
    let mut count = 0;
    for line in lines {
        let row: Vec<_> = line.split('\t').collect();
        let cached = header.contains(&"ref_pixels_sha256");
        let read = |path: &str, expected: &str, w: usize, h: usize| {
            let bytes = std::fs::read(path).unwrap();
            assert_eq!(bytes.len(), w * h * 3);
            assert_eq!(hash(&bytes), expected);
            bytes.as_chunks::<3>().0.to_vec()
        };
        let (r, d, w, h) = if cached {
            let w = row[column("width")].parse().unwrap();
            let h = row[column("height")].parse().unwrap();
            (
                read(row[rc], row[column("ref_pixels_sha256")], w, h),
                read(row[dc], row[column("dist_pixels_sha256")], w, h),
                w,
                h,
            )
        } else {
            let (r, w, h) = zen_io::decode_rgb8(Path::new(row[rc]));
            let (d, dw, dh) = zen_io::decode_rgb8(Path::new(row[dc]));
            assert_eq!((w, h), (dw, dh));
            (r, d, w, h)
        };
        let result = scorer
            .compute(&RgbSlice::new(&r, w, h), &RgbSlice::new(&d, w, h), None)
            .unwrap();
        writeln!(
            output,
            "{}",
            json!({"pair_key":row[kc],"width":w,"height":h,
            "reference_file_sha256":hash(&std::fs::read(row[rc]).unwrap()),
            "distorted_file_sha256":hash(&std::fs::read(row[dc]).unwrap()),
            "reference_pixels_sha256":hash(bytemuck::cast_slice(&r)),
            "distorted_pixels_sha256":hash(bytemuck::cast_slice(&d)),
            "score_bits":result.score().to_bits(),"score":result.score(),
            "full_feature_count":result.features().len(),
            "full_feature_bits_sha256":full_feature_hash(result.features()),
            "consumed_ids":consumed,"consumed_feature_bits":consumed.iter()
                .map(|&id|result.features()[usize::from(id)].to_bits()).collect::<Vec<_>>()})
        )
        .unwrap();
        count += 1;
        if count % 100 == 0 {
            output.flush().unwrap();
            eprintln!("completed EVAL pair {count}");
        }
    }
    output.flush().unwrap();
    assert_eq!(count, 4678, "complete frozen EVAL feature fixture required");
    eprintln!("completed EVAL pairs {count}");
}

// Era experiments must record changed identity scores rather than weakening
// the existing E33 gate's pin, feature, or cross-tier assertions.
fn identity(args: &[String]) {
    assert_eq!(args.len(), 3, "identity PACKET BAKE OUTPUT.jsonl");
    let bytes = std::fs::read(&args[1]).unwrap();
    assert_eq!(hash(&bytes), BAKE_SHA);
    let model = zenpredict::Model::from_bytes(&bytes).unwrap();
    let mut scorer = BakeScorer::new(&model).unwrap().with_parallel(false);
    let ids = scorer.consumed_feature_ids().unwrap();
    let packet: Value = serde_json::from_slice(&std::fs::read(&args[0]).unwrap()).unwrap();
    let sources = packet.as_array().unwrap();
    assert_eq!(sources.len(), 62);
    let loaded: Vec<_> = sources
        .iter()
        .map(|source| {
            let path = Path::new(source["ref_path"].as_str().unwrap());
            assert_eq!(
                hash(&std::fs::read(path).unwrap()),
                source["file_sha256"].as_str().unwrap()
            );
            let (pixels, w, h) = zen_io::decode_rgb8(path);
            if let Some(expected) = source["ref_pixels_sha256"].as_str() {
                assert_eq!(hash(bytemuck::cast_slice(&pixels)), expected);
            }
            (source["ref_group"].clone(), pixels, w, h)
        })
        .collect();
    let mut output = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[2])
        .unwrap();
    let _guard = archmage::testing::lock_token_testing();
    let shard = std::env::var("A_FUSION_IDENTITY_SHARD").ok().map(|s| {
        let (i, n) = s.split_once('/').unwrap();
        let i = i.parse::<usize>().unwrap();
        let n = n.parse::<usize>().unwrap();
        assert!(n > 0 && i < n);
        (i, n)
    });
    let mut permutation = 0;
    let mut count = 0;
    let report = archmage::testing::for_each_token_permutation(
        archmage::testing::CompileTimePolicy::WarnStderr,
        |perm| {
            let index = permutation;
            permutation += 1;
            if shard.is_some_and(|(i, n)| index % n != i) {
                return;
            }
            if std::env::var_os("A_FUSION_IDENTITY_FIRST_ONLY").is_some() && count != 0 {
                return;
            }
            for (group, pixels, w, h) in &loaded {
                let view = RgbSlice::new(pixels, *w, *h);
                let result = scorer.compute(&view, &view, None).unwrap();
                let score = result.score();
                assert!(score.is_finite());
                writeln!(
                    output,
                    "{}",
                    json!({"permutation":perm.to_string(), "reference":group,
                    "candidate_score_bits":[score.to_bits()], "score":score,
                    "full_feature_count":result.features().len(),
                    "full_feature_bits_sha256":full_feature_hash(result.features()),
                    "consumed_ids":ids,"pixels_sha256":hash(bytemuck::cast_slice(pixels)),
                    "consumed_feature_bits":ids.iter().map(|&id|result.features()[usize::from(id)].to_bits()).collect::<Vec<_>>()})
                )
                .unwrap();
                count += 1;
            }
            output.flush().unwrap();
            eprintln!("fusion identity {count} scores: {perm}");
        },
    );
    eprintln!("{report}");
    writeln!(
        output,
        "{}",
        json!({"summary":true,"rows":count,"sources":62})
    )
    .unwrap();
}

pub(super) fn run(args: &[String]) {
    match args[0].as_str() {
        "dispatch" => {
            assert_eq!(args.len(), 1);
            println!(
                "{}",
                json!({"selected_token": selected_token(),
                "avx512_build": cfg!(feature = "avx512")})
            );
        }
        "worker" => worker(&args[1..]),
        "compare" => compare(&args[1..]),
        "pairs" => pairs(&args[1..]),
        "identity" => identity(&args[1..]),
        "prepare-eval" => prepare_eval(&args[1..]),
        _ => panic!("fusion dispatch|worker|compare|pairs|identity|prepare-eval"),
    }
}
