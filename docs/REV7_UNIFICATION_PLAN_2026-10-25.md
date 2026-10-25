# Revision 7: one engine, ship-ready (plan, 2026-10-25)

Owner, 2026-10-25: "I want to get rev 7 factored so that it is unified and ship-ready." Earlier the same day: "For
rev 7, we can restructure to our heart's content for speed; we will eventually be shifting zensim stuff into
zenmetrics and pruning zensim back to JUST rev 7 + legacy A support."

## Target

- **One engine.** zensim computes features one way: the revision-7 fold walk, one arithmetic per leaf kernel
  (revision 7's f32 row lanes into f64 pools, the cancellation-free forms, the Newton cube root), dispatched to v4x,
  v4, v3, NEON (scalar canon until the NEON bodies land), wasm128 and scalar, identical bits on every tier and thread
  count. One exact f64 oracle, test-only.
- **Legacy A, isolated.** Profile A (the 372-input v47-strict MLP on the revision-1 buffered walk with extended and
  masked/IW features) keeps every score bit-identical. Its walk and per-tier kernels move under one legacy module
  compiled only with `deprecated-profiles`.
- **No process switches in the product.** A bake declares revision 7 (or is A's embedded bake); `ZENSIM_FORMULA_REV`,
  `ZENSIM_RING_FORM` and the `ZENSIM_MAP_*` research toggles leave product code. Map modes the product needs become
  API options.
- **Training extraction stays.** `research::extract` computes the revision-7 families (basic, peaks, v2, globaff) for
  training tables. Families revision 7 does not compute (palette, the Rev4 banks, gmsbank, CSFW, append, CVVDP/LMS
  extractors, the probe arms) leave the product crate: behind a non-default `research` feature first, then to
  zenmetrics with the rest of the research surface.
- **The default profile serves revision 7.**

## Where it stands (measured 2026-10-25)

- Revision 7 is on `explore/byconstruction` and merges into main as `6d146e15` (main's only source change since the
  branch point, the attribution fix `bab88fde`, was a port of the branch's `ec9289df`).
- `zensim/src`: 137,917 lines in 60 files. Uses of `FormulaRevision::RevN`: Rev1 26, Rev2 26, Rev3 64, Rev4 41,
  Rev5 115, Rev6 15, Rev7 18.
- Profiles: A (deprecated, revision 1, buffered walk), PreviewV0_1 and PreviewV0_2 (linear, revision 1), B (default,
  revision 5, fold), BHdr (HDR, revisions 1–4, buffered PU walk), C, CHdr, D (`candidate-profiles`, on by default).
- Walks: the fold (`feature_v2` + `feature_v2_stream`), the buffered walk (`streaming.rs` + `fused.rs`),
  `research::extract`, attribution pass B, the steering session, `local_refine`. Leaf arithmetic forms: per-tier
  revision 1–3 bodies, canon f32 (revision 4), canon64/rec64, the revision-5 local window, revision 7's row lanes,
  the exact f64 oracle (`benchmarks/feature_paths_inventory_2026-10-22.md` has the per-leaf table).
- Revision 7 against revision 5 (B's weights, 55 dumps): 4,438 of 7,920 feature values move, median 1.1e-6 relative,
  largest 4.0e-3; scores by at most 5.4e-5 points. Retrained at revision 7, T reproduces T and M stays within seed
  spread (`benchmarks/byconstruction_2026-10-22.md`, "Revision 7 with raw ringing").

## Phases, each gated

0. **Land revision 7 on main.** The merge, verified locally (workspace clippy, the default and all-features test
   suites, zensim-validate, the minimal-feature clippy set, script lint, the API snapshot), pushed with
   `scripts/safe_push.sh`, then the CI workflow dispatched.
1. **Oracles before any deletion.** (a) Revision 7: the 55-dump bit harness (`scripts/byconstruction/perf_bits.sh`)
   for B's revision-7 stamp, plus a full-width research extraction (every revision-7 slot) on the same 11 inputs,
   at every x86 tier and eight threads. (b) A: score and all 372 features on the same inputs at every tier.
   (c) The existing golden tests (`v1_golden_bytes`, `imageflow_checksums`, `cross_platform`). Every later commit
   reproduces (a) and (b) bit for bit and passes the suite, or it does not land.
2. **Default profile on revision 7** (owner decision D1).
3. **Delete, one family of paths per commit:** the research switches; the removed profiles and their weights (D2,
   D3); revisions 2–6 arithmetic and every revision branch (`FormulaRevision` keeps only what A and revision 7
   need); the canon f32, canon64/rec64 and local-window leaf forms revision 7 does not run; the buffered walk's
   non-A paths, with A's path moved into the legacy module; research-only families behind `research`.
4. **API and release.** The public removals as one reviewed PR (assigned to lilith); `CHANGELOG.md` QUEUED BREAKING
   CHANGES; the API snapshot (`just api-doc`); README; semver checks. zensim 0.3.0 is unpublished, so the break
   lands in 0.3.0.
5. **Verification.** Paired speed against today's B (one binary, v4x and v3, the focus cases), peak RSS, wasm and
   i686 tier identity, the full CI matrix (windows-11-arm, macOS ARM64, i686 via cross).

## Owner decisions (my recommendation first)

- **D1, default model.** B's weights stamped revision 7: byte-identical weights, calibration and identity behaviour;
  scores move by at most 5.4e-5 points; revision 7's speed. Alternative: full-data M retrained at revision 7, if its
  gates pass (different scores; better mixture behaviour in the cells).
- **D2, HDR.** BHdr runs revisions 1–4 in the buffered PU walk and no revision-7 HDR model exists. (a) Keep BHdr as a
  second legacy profile until a revision-7 HDR model is trained and qualified (recommended: HDR is part of the
  mission); (b) drop HDR from the shipped API now; (c) serve HDR through revision 7's PU fold with the SDR model,
  unqualified.
- **D3, public removals in 0.3.0.** PreviewV0_1, PreviewV0_2, C, CHdr, D and their weights; the process switches
  above; the research-only families (behind `research`). A stays behind `deprecated-profiles`.
