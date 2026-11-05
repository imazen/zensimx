# Preparatory sibling pixel graph (2026-11-05)

Coordinator approved doing this dependency step before the remaining siblings:
JPEG's deliberately paired zensim/zensim-regress pin at `85577588` cannot
consume current PNG buffers while its workspace requires zenpixels 0.2.16.
The JPEG test build produced two E0308 errors at pixel_ops lines 228 and 577.

Base: `c746a409dd8aa1d6339974b5faef6e3c92992618`.
Built source: `c7875d8d89efa638f7de06bd70eeb04db2a88a46`.
The resolved root lock has one zenpixels/convert 0.3.1 source (`fa836abf`),
one zencodec 0.2.0/zenmeta source (`1816b4d0`), PNG `02a7f0df`,
UltraHDR `85cc309c`, and PNG's existing zenflate strip source `a0ea0363`.
Zenpredict and its baker remain pinned at `f0903489`; metric, model and
formula code are unchanged. E34's deprecated EXIF method calls now use
zenmeta free functions, which delegate to those same retained implementations.
Its existing policies and assertions are unchanged.

Validation on a Ryzen 9 7950X under WSL2, isolated source/target and shared lock:

- CI-exact workspace/all-target/all-feature clippy, excluding only the existing
  wasm test package: `rc=0 36s | peak-RSS 1.04GiB | min-avail 7471MiB | peak-load 4.69`.
- E34 metadata: two tests pass. zensim-regress: 416 tests/doctests pass,
  zero fail, eight existing ignored doctests. The combined test/initial-build
  scope: `rc=0 434s | peak-RSS 1.42GiB | min-avail 6675MiB | peak-load 6.35`.
- Final source-stamped release examples: `rc=0 99s | peak-RSS 1.23GiB | min-avail 7337MiB | peak-load 5.12`.
  No target-cpu=native flag; training, custom-profiles and classification features;
  thin LTO, debug=0, force-frame-pointers=yes, two build jobs.

Binary SHA-256:

- a_family_cost: `49d8da31a4ce884fe816a6983a71f1e2b051e5df4740a63ee315389b7bb96a89`
- split_parity_golden: `7e5beda99519834e9d6acc68fd168a58d46c87bff0a04cb2709572ab7a4c3b5b`

Raw logs, source archive, resolved Cargo.lock and binaries, checked on dev:
`/mnt/v/output/zensim-siblings-20261105/zensimx-prep/`.
Tower mirror: `/mnt/tower/output/zensim-siblings-20261105/zensimx-prep/`.
Frozen source replay and split gates pass on i270 (new run `oracle-prep-run02`):
4,678/4,678 predictions equal the frozen table, zero differences; all source
file hashes, decoded pixel hashes, 504 consumed feature values per pair, full
feature hashes and scores equal the retained source capture. The complete
JSONL is byte-identical: `aa1810c58b3615b57cb32fc21978111cbc6776566af791de8d1f1d3e8cd9deef`.
The split golden is byte-identical: 145,327 lines, 18,586,466 bytes,
SHA-256 `93c94bee1e518f3e105ebcc75f7a4e06a9367bc626f418a5fc7fdee6346b8627`.
Native capture + split + the initial parser dispatch:
`rc=1 253s | peak-RSS 0.43GiB | min-avail 23883MiB | peak-load 7.80`;
the nonzero status is a missing parser module after both native gates completed.
Parse-only retry (system Python with existing pyarrow, complete imports):
`rc=0 1s | peak-RSS 0.52GiB | min-avail 24710MiB | peak-load 5.07`.
Input and model hashes pass again after replay. No scoring or split computation
was repeated to repair parsing. Runtime/shared-pixel tests remain required
before landing.
Initial orchestration failures and a 210-row capture with unavailable corpus
access are retained; none count as a passing oracle. The original fixtures,
source image bytes, released binaries and study cache are unchanged.
