# Final sibling dependency cohort (2026-11-05)

Dependency convergence only. The six sibling mains are remotely verified;
selected immutable revisions are in [the cohort record](../scripts/sibling-cohort.json).
The primary shared checkouts were left untouched. All four former external
path consumers now use Git revisions; repository-internal paths stay local.
No model, metric kernel, fixture, assertion or threshold changed in this step.

## Tool qualification

`just sibling-tools-check` compiles all targets for target (all features),
picker-prep, UltraHDR probe, benchmark, visualization (real-bake-tests), and
shipgates-target. Benchmark features are training, zen-decode,
zen-decode-tests, dvifm-judge, gmsd-arm and m3-fixtures. Its existing explicit
`ZENSIM_BENCH_SKIP_CPP_FFI=1` flag excludes C++ comparison arms: those are not
qualified here. No benchmark, fixture regeneration or historical encoder gate
was run. Encoder revisions changed, so older gate results retain their old
provenance and are not restated as current results.

All six compile: `rc=0 45s | peak-RSS 0.87GiB | min-avail 7251MiB | peak-load 3.61`.
The probe uses the moved `zenpixels::hdr::DiffuseWhite` name with the same
value. The benchmark's bitmap adapter now uses `ffc78413`; precise lock
resolution removes the older zenpixels copy admitted by its broad range.
Its unchanged stage-isolation unused import and obsolete AIC smoke example's
unreachable-code warnings remain visible; they are not strict-lint passes.

The UltraHDR probe intentionally retains the public F model at `473a4cfb`
and its own zenpredict `f0903489`, replacing its external local checkout.
Its mix model Git blob is `303306d3cc6225f1009fd595464a29f9e1963c90`, also
present in the inspected public checkout. This is a compile qualification,
not a replay of the old UltraHDR encoder study. Other research roots use
zenpredict/bake `38dc80f7`. Root/visualization statistical source pins are
unchanged; formerly local benchmark statistics use `8d6bb1e`.

`just sibling-source-check <receipt>` checks nine manifests and all seven
resolved roots, including source identity, one shared core/pixel source,
selected codec revisions and no external local-package paths. Missing locks
or mismatched sources fail. The explicit public-F runtime is separately pinned.
The final tool/source/script scope passes after unused-patch removal:
`rc=0 29s | peak-RSS 0.17GiB | min-avail 8264MiB | peak-load 3.59`.
The three already-ignored standalone lockfiles are archived with the evidence.

## Owner-visible sibling limitations

- AVIF has seven existing production/backend failures, reproduced on untouched
  `252baa40`: malformed EXIF policy versus its assertion, five animation tests
  with a 13-pixel canvas refused by the selected backend (minimum 16), and the
  `anim_mdhd_after_stbl` inventory sentinel. Default is 525 passes/one baseline
  failure/11 existing ignores; production 872/seven/11; backends 572/seven/11.
  Zero new failures. Decoder policy and every assertion remain unchanged.
  Both [baseline CI](https://github.com/imazen/zenavif/actions/runs/42022900813)
  and [converged CI](https://github.com/imazen/zenavif/actions/runs/42048092775)
  fail the same malformed-EXIF assertion in seven jobs; no other failed
  assertion appears in the converged failed-job logs. Converged clippy,
  feature permutations, WASM, MSRV, queue determinism, fuzz and link-u pass.
- WebP and AVIF lack checked metadata retention. Existing legacy setter
  callers have narrowly scoped deprecation allowances; assertions remain.
  WebP's empty/ICC/CICP checked-setter probes all return MetadataRetention.
- UltraHDR retains and polls the owned Pulse at existing cancellation points;
  phase/progress reporting is not implemented by this compatibility step.
- Analyzer uses a private compatibility conversion preserving its existing
  HDR/linear feature values; it does not invent source-peak metadata. Direct
  legacy parity is not claimed for cases where the old converter panicked.

## Final frozen qualification

Root qualification: 904 runtime passes across 49 targets (25 existing ignores),
10 doctests pass (three existing ignores). Regression tooling: 408 runtime
passes across five targets and eight doctests pass (eight existing ignored
doctests). CI-exact clippy and the 1,576-script lint pass. Combined scope:
`rc=0 992s | peak-RSS 3.21GiB | min-avail 4459MiB | peak-load 6.39`.

Built source: `91f29ad04000f63e7d3d3f7f8e4ccf98171247df`, preserved with
the resolved locks and complete source archive. Release build:
`rc=0 157s | peak-RSS 1.34GiB | min-avail 7179MiB | peak-load 4.05`.
Binary SHA-256: `a_family_cost`
`d34e71a1cd72189485fc126f372063b2d10a3ffbdf5cf9eacb68c1c09bab34ad`;
`split_parity_golden`
`155b7a934aed027f2466be59c8dfec61a843f1e4984198d285fa59b84a51dc3c`.

Fresh frozen replay on i270 (`oracle-final-run01`) passes: 4,678/4,678
predictions exactly equal the frozen table, zero differences. Source file and
decoded-pixel hashes, all 504 consumed feature values, full-feature hashes and
score bits are unchanged. The complete capture is byte-identical, SHA-256
`aa1810c58b3615b57cb32fc21978111cbc6776566af791de8d1f1d3e8cd9deef`.
The complete split golden is byte-identical: 145,327 lines, 18,586,466 bytes,
SHA-256 `93c94bee1e518f3e105ebcc75f7a4e06a9367bc626f418a5fc7fdee6346b8627`.
Combined native replay, split, frozen-prediction and strict pixel/feature checks:
`rc=0 234s | peak-RSS 0.52GiB | min-avail 24425MiB | peak-load 7.44`.
All eight input/model hashes and both binary hashes pass again after the run.
The first orchestration launch omitted two unchanged just imports and exited
before reading any pairs; that log is retained. The completed retry changed
no scientific inputs or gates. Post-build changes contain receipts only.

Evidence, checked on dev: `/mnt/v/output/zensim-siblings-20261105/zensimx-final/`.
Mirror: `/mnt/tower/output/zensim-siblings-20261105/zensimx-final/`.
Scientific inputs, released binaries, study cache and final-read checkout stay frozen.
