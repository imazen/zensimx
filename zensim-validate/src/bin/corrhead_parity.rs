//! **The corruption-head parity gate** — the Rust evaluator against the
//! sklearn model it was exported from, on the same rows, in f64.
//!
//! Pre-registered as gates G1 + G2 of
//! `docs/PLAN_CORRHEAD_SERVING_2026-09-06.md`; record:
//! `benchmarks/corruption_head_serving_2026-09-06.md`.
//!
//! This is a BINARY rather than a `#[test]` on purpose. The vectors it needs
//! are a 66 MB npz on `/mnt/v`, and a test that quietly passed when that file
//! was absent would be exactly the "graceful skip" the workspace rules forbid.
//! The skip decision belongs to the caller, so the caller runs the binary and
//! it fails loudly when the data is missing.
//!
//! ## What is compared, and why in two stages
//!
//! * **`decision_function`** — `baseline + sum of tree outputs`. The tree walk
//!   is exact arithmetic (comparisons and f64 additions in a fixed order), so
//!   this must agree at **0 ulp**. A difference here is a real defect: a wrong
//!   bracket, a wrong child, a mis-parsed node.
//! * **`probability`** — after `1/(1+exp(-raw))` and the isotonic
//!   interpolation. `exp` is the one inexact step (numpy uses its own SIMD
//!   kernel; Rust's `f64::exp` calls the platform libm), so this is reported
//!   as a measured `max |delta|` rather than asserted at 0.
//! * **the FIRE SET** — `{p > deadband}` — compared as an exact set equality,
//!   because that, not the probability, is what the deploy composition reads.
//!   Two implementations can differ by 1e-16 on `p` and still be identical
//!   where it matters, and this is the check that says so.

use std::io::Write;
use std::path::PathBuf;

use zensim::corruption_head::CorruptionHead;
use zensim_validate::npz::Npz;

fn usage() -> ! {
    eprintln!(
        "corrhead_parity --head <head.zcth> --parity <parity.npz> [--set test|gate|both] \
         [--raw-ulp-max N] [--prob-tol F]\n\n\
         --predict-raw <out.tsv> --set test runs inference on test_X only.\n\n\
         Compares zensim::corruption_head against the sklearn vectors the exporter wrote.\n\
         `--raw-ulp-max` defaults to 0: the tree walk is exact and any drift is a defect.\n\
         `--prob-tol` defaults to 1e-12; the achieved max is always printed."
    );
    std::process::exit(2)
}

/// Distance in representable f64 steps. Reported instead of a bare abs delta
/// because "0 ulp" is a claim about the arithmetic and "2e-16" is a claim
/// about a magnitude — only the first one gates a tree walk.
fn ulp_distance(a: f64, b: f64) -> u64 {
    if a == b {
        return 0;
    }
    if a.is_nan() || b.is_nan() {
        return u64::MAX;
    }
    let key = |x: f64| -> i64 {
        let bits = x.to_bits() as i64;
        if bits < 0 { i64::MIN - bits } else { bits }
    };
    key(a).abs_diff(key(b))
}

struct Args {
    head: PathBuf,
    parity: PathBuf,
    sets: Vec<String>,
    raw_ulp_max: u64,
    prob_tol: f64,
    predictions: Option<PathBuf>,
}

fn parse_args() -> Args {
    let mut head = None;
    let mut parity = None;
    let mut sets = vec!["test".to_string(), "gate".to_string()];
    let mut raw_ulp_max = 0u64;
    let mut prob_tol = 1e-12f64;
    let mut predictions = None;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--head" => head = it.next().map(PathBuf::from),
            "--predict-raw" => {
                predictions = Some(it.next().map(PathBuf::from).unwrap_or_else(|| usage()))
            }
            "--parity" => parity = it.next().map(PathBuf::from),
            "--set" => {
                let v = it.next().unwrap_or_else(|| usage());
                sets = if v == "both" {
                    vec!["test".into(), "gate".into()]
                } else {
                    vec![v]
                };
            }
            "--raw-ulp-max" => {
                raw_ulp_max = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or_else(|| usage())
            }
            "--prob-tol" => {
                prob_tol = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or_else(|| usage())
            }
            "-h" | "--help" => usage(),
            other => {
                eprintln!("unknown argument {other:?}");
                usage()
            }
        }
    }
    Args {
        head: head.unwrap_or_else(|| usage()),
        parity: parity.unwrap_or_else(|| usage()),
        sets,
        raw_ulp_max,
        prob_tol,
        predictions,
    }
}

fn main() {
    let args = parse_args();
    let bytes =
        std::fs::read(&args.head).unwrap_or_else(|e| panic!("read {}: {e}", args.head.display()));
    let head = CorruptionHead::from_bytes(&bytes)
        .unwrap_or_else(|e| panic!("{}: {e}", args.head.display()));
    let npz = Npz::open(&args.parity).unwrap_or_else(|e| panic!("{}: {e}", args.parity.display()));

    println!(
        "head   {}\n  caller width {}, reads {}, {} trees / {} nodes, schema {:#018x}\n  \
         deadband P>{} (score < {})",
        args.head.display(),
        head.caller_input_width(),
        head.declared_feature_ids().len(),
        head.n_trees(),
        head.n_nodes(),
        head.schema_hash(),
        head.deadband(),
        head.deadband_score(),
    );
    println!("parity {}", args.parity.display());

    // Explicit inference mode is separate from the unchanged parity gate.
    // Its NPZ needs only <set>_X; it never invents reference predictions.
    if let Some(path) = &args.predictions {
        assert_eq!(
            args.sets.len(),
            1,
            "raw prediction requires one explicit --set"
        );
        let set = &args.sets[0];
        let x = npz.get(&format!("{set}_X")).expect("prediction matrix");
        let [n_rows, n_cols] = x.shape.as_slice() else {
            panic!("prediction matrix must be 2-D");
        };
        let values = x.f64s().expect("prediction matrix f64");
        let ids = head.declared_feature_ids();
        assert_eq!(*n_cols, ids.len(), "prediction matrix read set");
        assert_eq!(values.len(), n_rows * n_cols, "prediction matrix length");
        assert!(
            values.iter().all(|value| value.is_finite()),
            "nonfinite prediction input"
        );
        let mut row = vec![0.0f64; head.caller_input_width()];
        let output = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .unwrap_or_else(|error| panic!("create {}: {error}", path.display()));
        let mut output = std::io::BufWriter::new(output);
        writeln!(output, "row_idx\traw").expect("prediction header");
        for i in 0..*n_rows {
            for (j, &id) in ids.iter().enumerate() {
                row[usize::from(id)] = values[i * n_cols + j];
            }
            let raw = head.decision_function(&row).expect("prediction width");
            assert!(raw.is_finite(), "nonfinite native prediction at row {i}");
            writeln!(output, "{i}\t{raw:.17e}").expect("prediction row");
        }
        output.flush().expect("prediction flush");
        println!("raw predictions: {n_rows} rows -> {}", path.display());
        return;
    }

    let mut failed = false;
    for set in &args.sets {
        let x = npz
            .get(&format!("{set}_X"))
            .unwrap_or_else(|e| panic!("{set}_X: {e}"));
        let raw_ref = npz
            .get(&format!("{set}_raw"))
            .unwrap_or_else(|e| panic!("{set}_raw: {e}"));
        let p_ref = npz
            .get(&format!("{set}_p"))
            .unwrap_or_else(|e| panic!("{set}_p: {e}"));
        let (n_rows, n_cols) = match x.shape.as_slice() {
            [r, c] => (*r, *c),
            other => panic!("{set}_X has shape {other:?}, expected 2-D"),
        };
        let xs = x.f64s().unwrap_or_else(|e| panic!("{set}_X: {e}"));
        let raw_ref = raw_ref.f64s().unwrap_or_else(|e| panic!("{set}_raw: {e}"));
        let p_ref = p_ref.f64s().unwrap_or_else(|e| panic!("{set}_p: {e}"));
        assert_eq!(raw_ref.len(), n_rows, "{set}_raw length");
        assert_eq!(p_ref.len(), n_rows, "{set}_p length");

        let ids = head.declared_feature_ids();
        assert_eq!(
            ids.len(),
            n_cols,
            "{set}_X has {n_cols} columns; the head reads {}",
            ids.len()
        );

        // The parity matrix is the head's READ SET, column j = declared id
        // ids[j]. Scatter it into a caller-width row so the evaluator gathers
        // through the same declaration a real caller would — testing the
        // gather, not bypassing it. Slots the head does not declare are left
        // at zero and are never read; if they ever were, the raw comparison
        // below would catch it at 0 ulp.
        let width = head.caller_input_width();
        let mut row = vec![0.0f64; width];
        let mut worst_raw_ulp = 0u64;
        let mut worst_raw_at = 0usize;
        let mut worst_p = 0.0f64;
        let mut worst_p_at = 0usize;
        let mut fire_mismatch = 0usize;
        let mut fire_rust = 0usize;
        let mut fire_py = 0usize;
        let mut score_disagrees = 0usize;
        let t = head.deadband();
        let t_score = head.deadband_score();

        for i in 0..n_rows {
            for (j, &id) in ids.iter().enumerate() {
                row[usize::from(id)] = xs[i * n_cols + j];
            }
            let raw = head.decision_function(&row).expect("width");
            let d = ulp_distance(raw, raw_ref[i]);
            if d > worst_raw_ulp {
                worst_raw_ulp = d;
                worst_raw_at = i;
            }
            let p = head.probability_f64(&row).expect("width");
            let dp = (p - p_ref[i]).abs();
            if dp > worst_p {
                worst_p = dp;
                worst_p_at = i;
            }
            let fr = p > t;
            let fp = p_ref[i] > t;
            fire_rust += usize::from(fr);
            fire_py += usize::from(fp);
            if fr != fp {
                fire_mismatch += 1;
            }
            // The score-space and probability-space deadbands must select the
            // SAME rows. They are two spellings of one decision and a
            // disagreement would mean `bake_verdict` (score space) and the
            // Python record (probability space) silently gate different rows.
            let s = head.score_f64(&row).expect("width");
            if (s < t_score) != fr {
                score_disagrees += 1;
            }
        }

        let raw_ok = worst_raw_ulp <= args.raw_ulp_max;
        let p_ok = worst_p <= args.prob_tol;
        let fire_ok = fire_mismatch == 0;
        let score_ok = score_disagrees == 0;
        failed |= !(raw_ok && p_ok && fire_ok && score_ok);
        println!(
            "\n[{set}] n = {n_rows} rows x {n_cols} declared features\n  \
             G2 decision_function  max {worst_raw_ulp} ulp (row {worst_raw_at})  {}\n  \
             G1 probability        max |delta| {worst_p:.6e} (row {worst_p_at})  {}\n  \
             G1 fire set  P > {t}  rust {fire_rust} / python {fire_py}, \
             {fire_mismatch} disagreements  {}\n  \
             .. score < {t_score} selects the same rows  ({score_disagrees} disagreements)  {}",
            if raw_ok { "PASS" } else { "FAIL" },
            if p_ok { "PASS" } else { "FAIL" },
            if fire_ok { "PASS" } else { "FAIL" },
            if score_ok { "PASS" } else { "FAIL" },
        );
    }

    if failed {
        eprintln!("\ncorrhead_parity: FAIL");
        std::process::exit(1);
    }
    println!("\ncorrhead_parity: PASS");
}
