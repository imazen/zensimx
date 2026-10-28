# Pure-Rust WASM+CubeCL trainer plan

> **⚠ BANNER 2026-07-18:** Phase 1 (pure-Rust trainer core) landed as `zensim-train-core/`. The WASM+CubeCL browser frontend is NOT built — still open.

**Status**: scoping doc, 2026-05-11 (post-V0_5 ship). Per user directive
"work toward a pure Rust training pipeline that can run in WebAssembly
on background workers with CubeCL acceleration to allow interactive
exploration and plotting of different weights and adjustments to targets."

## Goal

Replace the Python `train_v_next_mlp.py` (and its Rust sibling
`zensim-validate/src/mlp_train.rs`, 982 LOC) with a **pure-Rust trainer
that compiles to WebAssembly** and runs in a browser background worker.
The trainer uses **CubeCL** (cross-platform GPU compute via `wgpu` /
WebGPU) to accelerate the hot loops (forward, backward, RankNet pairs).

User-facing flow:
1. Browser loads the zensim weight slot UI (HTML + WASM zensim-train).
2. User adjusts target curve / band weights / TV coefficient / etc.
3. Background worker retrains the MLP (CubeCL on WebGPU) — typically
   10-100 seconds for a few thousand iterations.
4. Live plots stream back: per-band SROCC, scatter, residual,
   monotonicity, calibration curve.
5. User commits the new weight file (download) or rejects and tweaks
   again.

## Why this matters

- **Interactivity**: Python trainer requires a workstation + 5-15 min
  per experiment. Browser trainer is 10-100 seconds, no setup.
- **Reproducibility**: weights produced this way ship with the exact
  hyperparameter trace in the URL/manifest, so anyone can re-run.
- **Audit trail**: per-band visualizations land in the same UI; no
  separate plot-generation step.
- **Removes platform skew**: Rust is what we ship; the Python trainer
  introduced a known -0.03 CID22 SROCC drift vs the Rust trainer (Tick
  15). Browser path keeps everything in Rust.

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│ Browser tab                                                     │
│ ┌────────────────────┐   ┌────────────────────────────────────┐│
│ │ UI (Yew/Leptos)    │←→ │ Background Worker                  ││
│ │ - Weight knobs     │   │ ┌──────────────────────────────┐  ││
│ │ - Target curve     │   │ │ zensim-train-wasm (WASM)     │  ││
│ │ - Live plots       │   │ │  - load features.bin         │  ││
│ │ - Manifest/export  │   │ │  - RankNet + Adam loop       │  ││
│ └────────────────────┘   │ │  - CubeCL on WebGPU          │  ││
│         ↑                │ │  - per-band SROCC eval       │  ││
│         │  postMessage   │ │  - emit progress events      │  ││
│         │                │ └──────────────────────────────┘  ││
│         └────────────────│ ↓ GPU                              ││
│                          │ ┌──────────────────────────────┐  ││
│                          │ │ WebGPU (browser-native)      │  ││
│                          │ └──────────────────────────────┘  ││
│                          └────────────────────────────────────┘│
└────────────────────────────────────────────────────────────────┘
```

## Crates / dependencies

- **`cubecl` 0.10+** (`wgpu` feature, WebGPU support). Already on
  crates.io, well-maintained, multi-platform.
- **`wgpu`** as the backend; works in browsers via WebGPU.
- **`wasm-bindgen`** + **`wasm-bindgen-rayon`** for `Worker` integration.
- **`zensim`** crate (existing) — for feature extraction. Already
  compiles to WASM via `zenwasm-abi` infra.
- **`zenpredict`** crate — for ZNPR v2 read/write at session boundary.
- **`linfa` / `ndarray`** as compatibility layer if CubeCL doesn't
  support all ops we need (RankNet pair selection is the candidate
  for compute-on-CPU since it's small ranking work).

## New workspace member: `zensim-train-wasm`

Path: `~/work/zen/zensimx/zensim-train-wasm/`

```
zensim-train-wasm/
  Cargo.toml          # cdylib = wasm32-unknown-unknown
  src/
    lib.rs            # wasm_bindgen entrypoints
    trainer.rs        # RankNet + Adam (CubeCL kernels)
    eval.rs           # per-band SROCC, monotonicity audit
    storage.rs        # IndexedDB persistence
    progress.rs       # postMessage emitters
  www/
    index.html        # UI shell
    worker.js         # Worker boilerplate
    style.css
```

## Phased implementation

### Phase 0 (this tick) — scoping doc
✓ This doc.

### Phase 1 — Rust-native trainer port
- Port `zensim-validate/src/mlp_train.rs` (982 LOC) to a library crate
  with NO WASM-incompatible deps (no rayon by default, no std::fs).
- Maintain feature parity with current Rust trainer including TV
  regularizer, RankNet pair sampling, Adam, group weighting.
- Verify bit-exact equivalence on a small fixed seed.
- Time budget: 1-2 days.

### Phase 2 — CubeCL kernel for forward/backward
- Replace ndarray matmul with CubeCL kernels.
- Implement: Linear layer, LeakyReLU, MSE+RankNet loss, Adam updates.
- Validate against Phase 1 output (bit-exact-ish, within fp32 noise).
- Time budget: 2-3 days.

### Phase 3 — WASM build + wasm-bindgen API
- Add `wasm32-unknown-unknown` target.
- Expose `train(features_bin, hyperparams) -> bake_bytes` to JS.
- Test in a minimal HTML page.
- Time budget: 1 day.

### Phase 4 — Background Worker integration
- Move trainer call to a Web Worker via wasm-bindgen-rayon.
- Stream progress events (loss, val SROCC, current epoch).
- Test end-to-end: load features.bin → train → return bake.
- Time budget: 1 day.

### Phase 5 — UI
- Yew or Leptos for the interactive panel.
- Knobs: TV weight, learning rate, group weights, target curve points.
- Live plots: matplotlib-rs / plotly-rs equivalent in WASM
  (or fall back to canvas-based primitives).
- Time budget: 2-3 days.

### Phase 6 — Polish + ship
- Wire to imageflow demo page.
- Add weight export with reproducibility manifest.
- Document in zensim README.
- Time budget: 1 day.

**Total estimate**: 8-12 days of focused work.

## Open design questions

1. **Where do features come from?** The features.bin format already
   exists (ZSFC v3); browser worker can fetch it from a CDN or load
   from user upload. The 3.7 GB training set won't fit in IndexedDB —
   use streaming reads from a hosted parquet.

2. **Compute budget**: WebGPU on consumer laptops is ~1 TFLOP fp32
   (Apple M1/M2: 2.6 TFLOPS; mid-range NVIDIA: 5-10 TFLOPS). The
   current trainer takes ~8 min on 16-core CPU for one bake. With GPU
   acceleration: ~30-60 seconds expected. Interactive.

3. **Reproducibility**: every training run needs an associated
   `manifest.json` with: features.bin md5, hyperparameter values,
   random seed, library versions. The browser session URL should be
   restorable.

4. **Cross-browser support**: WebGPU is in Chrome stable, Edge, and
   Safari Tech Preview. Firefox is behind a flag. Fall back to WebGL2
   for breadth? Likely no — too much complexity, GPU compute on WebGL2
   is painful.

## What's NOT in scope (yet)

- New feature extraction (must reuse existing 228 features from
  zensim's analyze_features_rgb8).
- New training data corpus (must reuse `safe_synth_218k_features.csv`
  + the KonJND-aligned mix).
- Replacing the Python `score_unified_with_bake.py` — that lives in
  the offline analysis flow.
- Multi-target loss (per CLAUDE.md long-term goal #2, CID22-paper
  methodology reproduction) — separate workstream.

## CID22 paper methodology reproduction — interleaved

Per CLAUDE.md long-term goal #2, the WASM trainer must support
reproducing CID22 paper methodology:
- Per-codec SROCC (Table 3) on TSBPC/DSBQS-derived MCOS
- Pairwise SROCC (Table 6) within same source
- Per-band stats (Table 5 cutoffs B0/B1/B2/B3 at 50/65/90)
- PJND calibration check (Table 4 anchor at ssim2 ≈ 63)

These should land as runtime evals in the WASM UI's "metrics"
sidebar so the user can compare against published numbers.

## Related infrastructure

- `zenwasm-abi` — host-side cdylib loader (existing). Pattern for
  wasm32-unknown-unknown library packaging.
- `zenpredict` — ZNPR v2 reader/writer. Already pure-Rust + no_std.
- `archmage` — SIMD dispatch. Not needed for WASM (CubeCL replaces it
  on the GPU path; CPU fallback can use scalar Rust).

## Next concrete tick after this doc

Phase 1: start porting `mlp_train.rs` into a new
`zensim-train-core/` library crate that has zero WASM-incompatible
deps. Bit-exact reproduction of the existing trainer is the first
milestone.
