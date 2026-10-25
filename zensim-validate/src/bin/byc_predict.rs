//! Forward a compact derived-input bake through zenpredict on a wide table (by-construction exploration,
//! `benchmarks/byconstruction_2026-10-22.md`). For research bakes that read ids past zensim's registry (the byc_globaff
//! descriptor), which `bake_dial_refit predict --score-units` cannot load. Output is the bake's raw output (score
//! polarity), row order kept: `row_idx<TAB>pred`.
//!
//! Usage: byc_predict --bake PATH --decl DECLARATION.json --corpus TABLE.parquet --out PRED.tsv
use std::path::PathBuf;

use zensim_validate::derived_inputs::Fx1Declaration;

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let (mut bake, mut decl, mut corpus, mut out) = (None, None, None, None);
    while let Some(a) = args.next() {
        let v = args.next().map(PathBuf::from);
        match a.as_str() {
            "--bake" => bake = v,
            "--decl" => decl = v,
            "--corpus" => corpus = v,
            "--out" => out = v,
            other => return Err(format!("unknown argument {other}")),
        }
    }
    let (bake, decl, corpus, out) = (
        bake.ok_or("--bake")?,
        decl.ok_or("--decl")?,
        corpus.ok_or("--corpus")?,
        out.ok_or("--out")?,
    );
    let (decl, _) = Fx1Declaration::load(&decl)?;
    let bytes = std::fs::read(&bake).map_err(|e| format!("{bake:?}: {e}"))?;
    let model = zenpredict::Model::from_bytes(&bytes).map_err(|e| format!("parse bake: {e:?}"))?;
    let width = decl.direct.len() + decl.products.len();
    if model.n_inputs() != width || model.feature_transforms().is_some() {
        return Err(format!(
            "bake has {} inputs, the declaration builds {width}",
            model.n_inputs()
        ));
    }
    let g = zensim_validate::parquet_loader::load_parquet(&corpus, "predict", "human_score", 1.0)?;
    let mut predictor = zenpredict::Predictor::new(&model);
    let mut x = Vec::with_capacity(width);
    let mut s = String::from("row_idx\tpred\n");
    for (i, row) in g.feature_rows.iter().enumerate() {
        decl.compact_row(row, &mut x);
        let p = predictor
            .predict(&x)
            .map_err(|e| format!("forward: {e:?}"))?;
        s.push_str(&format!("{i}\t{:?}\n", p[0] as f64));
    }
    std::fs::write(&out, s).map_err(|e| format!("{out:?}: {e}"))?;
    eprintln!("byc_predict: {} rows -> {out:?}", g.feature_rows.len());
    Ok(())
}
