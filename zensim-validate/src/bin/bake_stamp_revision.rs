//! Stamp `zentrain.formula_revision` onto a copy of a bake.
//!
//! Bakes trained before the admission metadata existed (e.g. the v2c featpot
//! cells trained on Rev4-stamped tables whose `table_admission.formula_revision`
//! resolved "unknown") carry no stamp and load only as the shipped legacy era.
//! Revision A/B measurement and Rev4 serve qualification both need the SAME
//! weights loadable under a pinned process revision — this splices the
//! metadata section in via `zenpredict_bake::append_metadata_utf8`, which
//! leaves the weight sections byte-identical (no requantization).
//!
//! Usage: bake_stamp_revision <in.bin> <1|2|3|4|5|6|7> <out.bin>
//!
//! Revision 6 (`ringhpm` + `globaff`) is the arithmetic a bake trained on the
//! by-construction `ringhpm` roots was extracted with; stamping declares it so
//! the bake serves without `ZENSIM_RING_FORM`.

fn main() {
    let mut args = std::env::args();
    let bin = args.next().unwrap();
    let (src, rev, dst) = match (args.next(), args.next(), args.next(), args.next()) {
        (Some(s), Some(r), Some(d), None) => (s, r, d),
        _ => {
            eprintln!("usage: {bin} <in.bin> <revision 1|2|3|4|5|6|7> <out.bin>");
            std::process::exit(2);
        }
    };
    if !matches!(rev.as_str(), "1" | "2" | "3" | "4" | "5" | "6" | "7") {
        eprintln!("revision must be one of 1, 2, 3, 4, 5, 6, 7");
        std::process::exit(2);
    }
    let bytes = std::fs::read(&src).unwrap_or_else(|e| panic!("read {src}: {e}"));
    let stamped = zenpredict_bake::append_metadata_utf8(&bytes, "zentrain.formula_revision", &rev)
        .unwrap_or_else(|e| panic!("splice zentrain.formula_revision: {e:?}"));
    std::fs::write(&dst, &stamped).unwrap_or_else(|e| panic!("write {dst}: {e}"));
    println!(
        "{dst}: {} -> {} bytes, zentrain.formula_revision={rev}",
        bytes.len(),
        stamped.len()
    );
}
