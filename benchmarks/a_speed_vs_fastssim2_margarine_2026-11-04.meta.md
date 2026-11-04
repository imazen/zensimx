# Public A peer speed comparison: identities and reproduction

The measurement host is **i270**, Intel Core Ultra 7 270K Plus, Ubuntu 26.04 LTS, Linux x86-64.
Rust is `1.99.1` (`58cf8f9c3`, LLVM `23.1.1`); the kernel is `7.0.0-38-generic`.
All metric arms use runtime v3/AVX2 dispatch. Every measurement uses the same
CPU affinity, `0-15`: eight P cores and eight E cores, without SMT or a CCD
boundary. The single-thread process may migrate within that same set; the
sixteen-thread configuration has sixteen eligible logical CPUs. Results do
not establish performance on a homogeneous core set or on Zen 5.

## Frozen sources

- Public [`imazen/zensim`](https://github.com/imazen/zensim/tree/bebc6ea45d2b6adb6ab70affc3e289707ef74fb0),
  branch `profile-f` as resolved before building:
  `bebc6ea45d2b6adb6ab70affc3e289707ef74fb0`, package `zensim 0.3.0`.
  `ZensimProfile::A`, RGB8 sRGB, ordinary `Zensim::compute`; no cached reference,
  maps or research scorer. `with_parallel(false)` at one thread and `true` at
  sixteen. The branch may advance; this measurement never follows it. The embedded mix
  bake SHA-256 is `d5e2a56db35c32620d9f9d9c19ba3bc78bd8fea2b8cd933cde8447428d85e62a`.
- Registry [`fast-ssim2 0.9.1`](https://docs.rs/fast-ssim2/0.9.1/fast_ssim2/fn.compute_ssimulacra2.html),
  exact version `=0.9.1`, `rayon` enabled in both configurations, default
  `compute_ssimulacra2` on `PixelSlice::RGB8_SRGB`. No strip-mode change.
- [`margarine`](https://github.com/imazen/zenmetrics/tree/097b5e4de9f8e5322b0f0abd4efb2b7055e05931/crates/margarine),
  package `0.0.0`, zenmetrics commit
  `097b5e4de9f8e5322b0f0abd4efb2b7055e05931`. The selected `simd-row-malta`
  kernel uses 128-row/512-column geometry and primary max pooling; lower means
  less distortion. The package has no public library target. This instrument
  includes its original private `kernel.rs` and modules unchanged at the
  benchmark crate root and calls the exact `row_psycho::compute_geometry`
  entry used by its command. This avoids CLI, decode and process-start cost;
  end-to-end CLI scoring/decode cost is **not measured**. A separate no-input
  command-startup baseline is described below. All 41 staged Rust source hashes match the
  clean source tree; the hash inventory is retained with the artifacts.
- [`zenbench`](https://github.com/imazen/zenbench/tree/82ff70a3f0c212cf88f53d6e9c3e02de13f32eec),
  package `0.1.10`, commit `82ff70a3f0c212cf88f53d6e9c3e02de13f32eec`.
  Explicit launcher registration excludes the benchmark's own Rayon workers
  from foreign-load accounting while preserving foreign processes and global
  RAM/temperature checks. This pin includes the existing Linux worker and
  process-leader accounting fixes. No zenbench source or threshold is changed.
- [`zenresize`](https://github.com/imazen/zenresize/tree/e3975fb9d6d6b7baa96038a0eb8e27febb37c012)
  `0.3.1`, commit `e3975fb9d6d6b7baa96038a0eb8e27febb37c012`, Mitchell,
  RGB8 sRGB. Central square crops for the three square sizes; uncropped native
  aspect ratio for large images. No input is enlarged.
- [`zenjpeg`](https://github.com/imazen/zenjpeg/tree/4933461f2b6d760b7b112dc94397718fdadb5119)
  `0.9.0`, commit `4933461f2b6d760b7b112dc94397718fdadb5119`: q40 YCbCr
  4:2:0 encode and RGB8 decode. Independent `image 0.25.10` JPEG readback
  verifies format acceptance/dimensions; only zenjpeg's decoded bytes enter
  the metric arms. PNG decoding uses registry `zenpng 0.1.4` and
  `zenpixels-convert 0.2.16` on the specified SDR sources.
- [`codec-corpus`](https://github.com/imazen/codec-corpus/tree/8e10d4d765667c1c49d74413878fc4bfb46dcf8d):
  commit `8e10d4d765667c1c49d74413878fc4bfb46dcf8d`. Ten lexicographically
  first CID22 TRAIN PNGs and twenty lexicographically first CLIC-1024 files
  whose basenames occur in original CLIC TRAIN. Tiny/small use the ten CID22
  and first ten CLIC sources; 1024² uses all twenty CLIC sources. No labels,
  quality-based selection, fitting of metric weights or quality qualification.
- Canonical `imazen-26-png-v3` checkout, commit
  `b1faae550eba4d29efee1de1a32954ef11a0e1ed`: seven PNG-v3 SDR renders each
  from the `1000-lilith-photos-general` and `1200-lilith-interiors` folders,
  six from `1400-lilith-nature`. Fourteen source files required selective
  canonical Git LFS hydration; each payload matches its original LFS OID.
  The twenty files and actual PNG SHA-256 values are listed in the report.
  No quarantined image copy is used.

All fifty staged sources have verified PNG IHDR depth 8 and color type 2
(RGB). The retained `SOURCE_HEADERS.json` records those checks. No alpha
removal or precision conversion is needed for this SDR matrix.

The complete resolved graph is in
[`Cargo.lock`](../zensim-bench/a-speed-runtime/Cargo.lock). Runtime dispatch
uses `archmage`/`magetypes 0.9.30`; Rayon is `1.12.0`. All release builds use
thin LTO and one codegen unit, with no `target-cpu=native` or custom `RUSTFLAGS`.
Cargo's JSON `compiler-artifact.executable` selects the actual executable,
rather than an old filename left in `target/release/deps`.

## Commands

Run on the eligible host. `source-provenance.json` binds every staged source
to its original filename and SHA-256; `sources.json` fixes order. The artifact
root below is the actual i270 run directory. The benchmark source is
[`instrument.rs`](../zensim-bench/benches/a_speed_peers/instrument.rs); its separate
[`manifest`](../zensim-bench/a-speed-runtime/Cargo.toml) avoids unrelated
research and sibling-codec dependencies.

```sh
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
export TMPDIR="$HOME/tmp/zensim-bench/scratch"
export MARGARINE_SOURCE="$HOME/tmp/zensim-bench/margarine"
mkdir -p "$TMPDIR"
cd "$HOME/tmp/zensim-bench/harness/a-speed-runtime"
flock "$HOME/tmp/heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 16G --jobs 8 -- cargo update
flock "$HOME/tmp/heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 16G --jobs 8 -- cargo clippy --release --locked --bin a-speed-prepare --bench a_speed_peers -- -D warnings
flock "$HOME/tmp/heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 16G --jobs 8 -- cargo build --release --locked --bins --benches
flock "$HOME/tmp/heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 16G --jobs 8 -- cargo build --release --locked --bench a_speed_peers --message-format=json
cd "$HOME/tmp/zensim-bench"
flock "$HOME/tmp/heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 16G --jobs 8 -- ./harness/a-speed-runtime/target/release/a-speed-prepare sources.json fixtures
BENCH_CPUS=0-15 bash harness/a-speed-runtime/run.sh "$PWD" "$PWD/bin/a-speed-peers"
```

The collector invokes one latency process and three separate RSS processes
per size/thread cell, serialized through the shared heavy lock and a 16 GiB
hard cgroup cap. Latency processes register the launcher PID, enable gate
tracing, set `RAYON_NUM_THREADS`/`OMP_NUM_THREADS` to 1 or 16 and use
`taskset -c 0-15`. GNU `time -v` records exact process peak RSS in KiB;
run-heavy additionally reports rounded GiB, minimum host available RAM and
peak load. RSS processes score all twenty pairs sequentially with one pair
resident at a time; the latency process holds the twenty input pairs.

Before latency, load1 must be below 2.0. Every saved round retains the
zenbench gate result and randomized arm execution order. At most 64 rounds
are measured per image; reporting selects the first 32 clean, single-call
rounds with one shared index set for all arms. Fewer than 32 clean rounds,
an unreliable suite, or any traced foreign compiler/trainer causes refusal.
Warmup is 100 ms per arm and image; estimation and gate waits are excluded.
No batched throughput mean is presented as individual-call p95 latency.

The initial attempt was interrupted when an unrelated C++ build started
outside the shared lock. Its logs and partial results remain archived as
invalid. The first completed tiny-image retry also contains foreign Rust
compiler traces and is excluded in full. No timing from either attempt is
admitted. The corrected fixture
run also preserves the earlier failed attempt that encountered LFS pointers.

For analysis from the repository checkout:

```sh
just a-speed-report /path/to/copied-artifacts benchmarks/a_speed_vs_fastssim2_margarine_2026-11-04.md
```

OLS fits use all eighty per-image median times and actual pixel counts,
equally weighted by image, separately for each arm/thread configuration.
The reported p50/p95 use equally pooled individual calls, 640 per cell/arm,
with type-7 quantiles. A/peer ratios divide those observed p50 values; the
16-thread speedup is p50 at one thread divided by p50 at sixteen. Negative
OLS intercepts are permitted regression terms, not physical startup times.
Changing content at the large size limits the interpretation of the fits.

The running instrument was frozen from the initial flat bench-source path.
The tracked source was subsequently moved byte-for-byte into a nested file
to avoid parent-package auto-discovery; the matrix continues using the same
archived executable. This path change alters no scoring or timing code.

Frozen matrix executable SHA-256: `489a3ceabf0b85abbf58129309d922129f99fa94309aae73fdcf77b6b2284c3b`. The unchanged
instrument source SHA-256 is
`909a10de480398a3902a2c18e44d75937d2ab9cf1ea4d32c25249ca01d5ba456`.
The initial harness commit is `446f9e7b`; the archive retains Cargo’s full
`compiler-artifact` record and build logs for that executable.

## Final validation and separate command startup

The contaminated tiny cell was preserved in `results-foreign-build-run02.bak`,
then replaced in full under the owner's reserved quiet window on i270:

```sh
BENCH_CPUS=0-15 BENCH_THREADS=1 BENCH_SIZES=64 bash final-source/a-speed-runtime/run.sh "$PWD" "$PWD/bin/a-speed-peers"
export CARGO_TARGET_DIR="$PWD/harness/a-speed-runtime/target"
flock "$HOME/tmp/heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 16G --jobs 8 -- cargo clippy --release --locked --manifest-path final-source/a-speed-runtime/Cargo.toml --bins --bench a_speed_peers -- -D warnings
flock "$HOME/tmp/heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 16G --jobs 8 -- cargo build --release --locked --manifest-path final-source/a-speed-runtime/Cargo.toml --bins --bench a_speed_peers
cp "$CARGO_TARGET_DIR/release/a-speed-cli-startup" "$PWD/bin/a-speed-cli-startup"
flock "$HOME/tmp/heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem 16G --jobs 8 -- bash final-source/a-speed-runtime/build_cli.sh "$PWD" "$PWD/final-source/a-speed-runtime"
bash final-source/a-speed-runtime/startup.sh "$PWD" "$PWD/bin/a-speed-cli-startup"
python3 -B final-source/a-speed-runtime/report.py "$PWD" "$PWD/REPORT.md" --command-startup
```

Final path-layout builds and strict clippy passed on i270. The scoring source
remains byte-identical, and all eight latency cells and 24 isolated RSS
processes use the original frozen executable. Rebuilt validation artifacts
are preserved separately. All 41 margarine source hashes were rechecked
after the original CLI build. Eight reporter admission checks pass; the
production report validates all 80 fixture identities, paired clean samples,
absence of compiler traces, and complete per-image RSS score output. All 240
printed-score comparisons between thread counts match.

`build_cli.sh` uses the unchanged original `margarine.rs` command, its original
private modules, and the instrument's locked dependency graph and release
settings. It adds no public library API. CLI SHA-256 is
`c576c23242d81c71675743bba535d99a46811a5d012d42e4e4a80775d644aa12`;
startup instrument SHA-256 is
`9f82248de9ce62e60c36b80804ddfa541f387c5aaef326fd39b9af887898c76c`.
The `margarine --help` baseline measures spawn, executable loading, argument
handling and exit, with 100 ms warmup and 32 retained clean single-call
zenbench rounds. It does not initialize a scoring pool or decode images;
these values are not subtracted from the scoring matrix. Original encoded
PNG/JPEG command smoke checks also pass at both thread settings; they are
not end-to-end latency benchmarks or decoder-equivalence tests.

The owner-reserved i270 window was released after all measurements and
validation, before artifact copying. `FINAL_VALIDATION.json` records the
frozen identity, completed matrix, clean process state and release.
