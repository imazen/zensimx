//! Print the canonical feature-set id for explicit slots, for naming by-construction derived roots
//! (`benchmarks/byconstruction_2026-10-22.md`). The hash comes from `zensim::feature_set_id`, the one owner.
//!
//! Usage: byc_fsid COMPUTE LAYOUT ERA SLOTS, e.g. `byc_fsid basic+peaks+v2 1891 rev5_ringhpm_globaff 0-227,372-719,1867-1890`
use zensim::feature_set_id::{ComputeParts, FeatureSetId, SlotSet};

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let [compute, layout, era, slots] = a.as_slice() else {
        panic!("usage: byc_fsid COMPUTE LAYOUT ERA SLOTS");
    };
    let compute = ComputeParts::parse(compute).expect("compute tokens");
    let layout: usize = layout.parse().expect("layout width");
    let ids = slots.split(',').flat_map(|r| match r.split_once('-') {
        Some((lo, hi)) => lo.parse::<usize>().unwrap()..hi.parse::<usize>().unwrap() + 1,
        None => r.parse::<usize>().unwrap()..r.parse::<usize>().unwrap() + 1,
    });
    let set = SlotSet::from_slots(ids);
    let id =
        FeatureSetId::from_slots_with_layout(compute, layout, era, &set).expect("valid era token");
    println!("{id} slots_hash8={:08x} n_slots={}", set.hash8(), set.len());
}
