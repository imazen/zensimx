//! A linear stub bake over a feature-id list, for timing the walk that a read set plans (by-construction perf).
//!
//! Exploration record: `benchmarks/byconstruction_2026-10-22.md` ("Which cells can revision 7 skip"). The bake
//! declares the ids (dense contract: one input per id, ascending) and a formula revision, and weighs every input
//! −1e-3, so `BakeScorer` plans and computes exactly the features those ids need. Its score means nothing.
//!
//! Usage: byc_stub_bake [--ids-file PATH (whitespace-separated ids)] --revision 5|6|7 --out PATH
//! Without --ids-file, revision 7 declares the registry's entire basic + peaks + v2 + globaff scope.

fn main() {
    let mut ids_file = None;
    let mut revision = None;
    let mut out = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = || args.next().unwrap_or_else(|| panic!("{a} needs a value"));
        match a.as_str() {
            "--ids-file" => ids_file = Some(val()),
            "--revision" => revision = Some(val()),
            "--out" => out = Some(val()),
            other => panic!("unknown argument {other}"),
        }
    }
    let (revision, out) = match (revision, out) {
        (Some(r), Some(o)) => (r, o),
        _ => {
            eprintln!("usage: byc_stub_bake [--ids-file PATH] --revision 5|6|7 --out PATH");
            std::process::exit(2);
        }
    };
    let mut ids: Vec<usize> = if let Some(path) = ids_file {
        std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {path}: {e}"))
            .split_ascii_whitespace()
            .map(|t| t.parse().unwrap_or_else(|_| panic!("bad id {t}")))
            .collect()
    } else {
        assert_eq!(revision, "7", "--ids-file is required below revision 7");
        use zensim::feature_set_id::ComputeToken;
        [
            ComputeToken::Basic,
            ComputeToken::Peaks,
            ComputeToken::V2,
            ComputeToken::GlobAff,
        ]
        .into_iter()
        .fold(zensim::feature_set_id::SlotSet::default(), |ids, family| {
            ids.union(&zensim::research::family_slots(family))
        })
        .iter_slots()
        .collect()
    };
    ids.sort_unstable();
    ids.dedup();
    assert!(!ids.is_empty(), "no feature ids");
    let n = ids.len();
    let id_text = ids
        .iter()
        .map(|i| i.to_string())
        .collect::<Vec<_>>()
        .join(" ");
    let recipe = serde_json::json!({
        "schema_hash": 1,
        "scaler_mean": vec![0.0f32; n],
        "scaler_scale": vec![1.0f32; n],
        "metadata": [
            {"key": "zentrain.feature_ids", "type": "utf8", "text": id_text},
            {"key": "zentrain.formula_revision", "type": "utf8", "text": revision},
        ],
        "layers": [{"in_dim": n, "out_dim": 1, "activation": "identity", "dtype": "f32",
                    "weights": vec![-1e-3f32; n], "biases": [0.0f32]}],
    });
    let bytes = zenpredict_bake::bake_from_json_str(&recipe.to_string())
        .unwrap_or_else(|e| panic!("bake: {e:?}"));
    std::fs::write(&out, &bytes).unwrap_or_else(|e| panic!("write {out}: {e}"));
    println!("{out}: {n} ids, revision {revision}, {} bytes", bytes.len());
}
