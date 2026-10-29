# Provenance of the 2026-10-29 native revision-7 HDR feature banks

Each bank directory's `_MANIFEST.json` (assembler `scripts/byconstruction/hdr_native_bank.py`, `hdr_upiq_bank.py`, zensimx main) embeds the plan
manifest it was built from (copied here as `plan_*._MANIFEST.json`). Mapping of every cited identifier to a pushed commit:

| Cited | Meaning | Where it lives (pushed) | Check |
|---|---|---|---|
| executor_sha256 `3cc3fce197544045…` | `hdr_features_exec.py` baked into the fleet image | imazen/zenmetrics master `6b0a7f1c19b7` (`scripts/jobsys/hdr_features_exec.py`, `docker/hdr-native/Dockerfile`, tests) | `jj file show` of that commit hashes to 3cc3fce197544045…; the image recipe is the Dockerfile there |
| image base | `ghcr.io/imazen/zenfleet-worker:5a99a89` = `sha256:a890b058b84e081bcdcd92d6c00972c8c1d9490ced1f3ca3c55f146d5ecee424` | zenmetrics master `5a99a89e` | published tag |
| extractor_sha256 `63bed9a64ed265d2…` (`hdr944_extract`, ladders/4Kdtb/teacher) | bookworm build of the `hdr944_extract` example | source = zensimx jj commit `a86520ca68dc` (local-only, parent main `cd69b9e0`); its `zensim-bench/examples/hdr944_extract.rs` (blob sha256 prefix adc39c569737), `zensim-bench/Cargo.toml` (5fb4fd7613d9) and `Cargo.lock` (b8728693d2b5) are byte-identical in pushed main `297a9fe5`; nothing under `zensim/src` or `zensim-bench/src` differs between the two | `jj diff --from a86520ca68dc --to 297a9fe5 --summary` lists only scripts, tests, docs, examples, admission code |
| build_commit `a86520ca…` (ladders, 4Kdtb, teacher) | ZENSIM_BUILD_COMMIT stamped into the extractor | as above | as above |
| producer_sha256 `7614fcb7…` (4Kdtb, teacher train/val) | `scripts/rev4_featpot/rev5_bank.py` | zensimx main `297a9fe5` and later (identical file) | sha256 of the file starts 7614fcb7 |
| producer_sha256 `a7c5a909…` (ladders-r1) | rev5_bank.py before `--hdr-path-map` was added | `rev5_bank.py.a7c5a909` here (byte-identical copy; equals the pushed file minus the two `--hdr-path-map` hunks) | sha256 starts a7c5a909 |
| producer_sha256 `fa0b682b…` (UPIQ recovery-r2) | rev5_bank.py of jj commit `a86520ca68dc` (before the ladder planner) | `rev5_bank.py.fa0b682b` here | sha256 starts fa0b682b |
| UPIQ extractor `upiq_pu_score` sha `73d1374f…`, build_commit `d86b3b44383433e56ebc6161ee6c8db12639c5af` | native UPIQ-380 extractor and its build snapshot (the stored-byte recovery never executes it; the 380 rows were computed by it on 2026-10-29) | snapshot = zensimx main `bf94ecbd` + `upiq_pu_score_build_d86b3b44.patch` (4 files: justfile, `zensim-bench/Cargo.toml`, `zensim-bench/examples/hdr944_extract.rs`, `zensim-validate/src/bin/upiq_pu_score/ingest.rs`). The commit itself exists only in the build host's checkout (`~/work/zen/zensim` on the Ryzen 5900XT agent host), a local wip snapshot | applying the patch to `bf94ecbd` reproduces the four blob ids in `d86b3b44.blobs.txt` exactly (checked) |

## Not resolved
- `d86b3b44` is not on any remote; the patch above is the only pushed record of its changes (`ingest.rs` edits live nowhere else). The binary `upiq_pu_score` (73d1374f…) was not rebuilt from the patch, so source→binary identity is by the build host's recorded pin, not by a fresh build.
- The `hdr944_extract.rs` inside `d86b3b44` (blob 41d87dbf87…) differs from the one in `a86520ca68dc` (adc39c569737); only the latter produced the HDR banks.
- Sibling path dependencies used for the bookworm build (not in this repo): zenanalyze `edb1be9d`, zenavif `a284ec4e`, zenjpeg `1a886208`, zenwebp `ca238d00`, zenmetrics `7ff78221`, dvifmish working tree; rest git-pinned in `Cargo.lock`.
- views.spec.json of each plan (30–300 KB) are not committed; they are in the tower mirror `byc-hdr-20261029/` and their sha256 is in each plan manifest (`spec_sha256`).
