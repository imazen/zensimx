# Adam subnormal reproduction (2026-10-26)

Source before the numerical change: `e8704f52`. Tool:
`tools/adam-denormal-probe`, which includes the actual trainer's `adam_simd.rs`.
Dependencies match the root lockfile: archmage/magetypes 0.9.30. Rust 1.99.0,
release, thin LTO, no target-cpu override. This is an engineering counter
experiment, not model calibration or a training-data sweep.

Intel Core Ultra 7 265K (`i265`), CPU 2, nice 19 and idle I/O priority. One existing
control trainer occupied a different core. Command, with `STATE=decay|normal`:

```sh
perf stat -e cycles,instructions,cpu_core/assists.fp/ -- \
  taskset -c 2 nice -n19 ionice -c3 \
  tools/adam-denormal-probe/target/release/adam-denormal-probe fused "$STATE" 2000 v3
```

Both cases execute 2,000 updates of 104,960 parameters, zero pair inputs/gradients,
no L2, beta1=0.9, beta2=0.999, epsilon=1e-8, lr=0.005 and unit bias corrections.
`decay` starts both moments at twice the smallest normal f64; `normal` starts at
0.01. The probe prints final moment bits to make dead-code elimination observable.

| Before state | Elapsed seconds | P-core cycles | P-core instructions | FP assists | Final m bits | Final v bits |
|---|---:|---:|---:|---:|---|---|
| decay | 7.412144902 | 39,667,957,482 | 1,860,428,458 | 268,105,485 | `0000000000000005` | `0004538ecb401ffc` |
| normal | 0.104186039 | 551,691,308 | 1,821,210,454 | 0 | `2c846471c6e4004f` | `3f5626b2251f5bd9` |

One observation per arm; no dispersion or full-cell speedup claimed by this
initial reproduction. The persistent nonzero subnormal moment state and the
assists are absent from the normal-state control. Existing Adam unit/tier tests
and strict probe Clippy pass on this Intel host. Workspace formatting passes.

The optimizer fix and matched complete-cell comparisons are pending in this
initial record. Existing study binaries and result directories remain frozen.
