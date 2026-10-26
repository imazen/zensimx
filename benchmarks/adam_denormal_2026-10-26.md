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

## Residual-assist investigation

A moments-only repair flushes moment inputs and outputs using safe magetypes
exponent masks (signed zero preserved; no MXCSR writes). The same decay probe
then takes 0.161684931 s, with 860,596,132 cycles, 2,976,281,834 instructions and
368,007 assists. Its final moment bits are both zero. The existing parity tests
and new cross-boundary decay tests pass.

The first complete-cell attempt with that repair was retained as an incomplete
diagnostic and stopped before completion. A 10-second sample reports
53,080,394,583 cycles, 97,670,128,778 instructions and 56,276,665 assists. Sampling
the assists attributes all 565 samples to the fused Adam owner, immediately
after the first-moment FMA. A separate contemporaneous sample of the frozen
trainer reports 53,570,029,752 cycles, 10,180,642,198 instructions and 339,779,789
assists over 10.001831809 s. These are mechanism samples, not matched epochs or
full-cell speed measurements. A competing compilation also occurred during the
incomplete attempt, so its elapsed time is excluded from speed evidence.

Extend the fixed-work probe with `tiny-l2` (w=1e-304) and `subnormal-weight`
(w=1e-310), initially zero moments, zero input gradients and L2 scale 1e-5.
Both use the same 2,000 steps, 104,960 parameters and P-core 2:

| Repair / state | Elapsed seconds | FP assists |
|---|---:|---:|
| Moments only / tiny-l2 | 3.254515048 | 104,969,618 |
| Moments only / subnormal-weight | 3.251834918 | 104,969,632 |
| Also flush incoming gradients / tiny-l2 | 2.093459258 | 52,487,305 |
| Also bypass underflowing L2 products / tiny-l2 | 0.283669212 | 0 |
| Also bypass underflowing L2 products / subnormal-weight | 0.281937528 | 0 |

The complete repair shares a safe L2 row owner between fused and unfused
updates. It bypasses products below the normal range before multiplying,
classifies the product at the boundary, and keeps multiply/add separate.
Boundary tests check 64 adjacent representable inputs on each side for seven
scales, all reachable dispatch permutations, plus fused/unfused parity. The
existing trainer L2 rounding and shorter-slice tests pass unchanged.

The full trainer's Adam tests pass on i265. Resource records:
`run-heavy: done rc=0 83s | peak-RSS 1.63GiB | min-avail 18629MiB | peak-load 3.30`
for the release Adam test build/run; the subsequent trainer rebuild reports
`run-heavy: done rc=0 29s | peak-RSS 1.32GiB | min-avail 20293MiB | peak-load 3.20`.

Raw evidence is retained on dev under
`/mnt/v/output/zensim/adam-denormal-20261026/` and on the Intel measurement hosts
under the task's scratch directory. Full-cell before/after timing, complete
artifact byte comparisons, AVX-512/NEON execution and final workspace lint are
still pending. A fresh isolated full baseline is running on i134; a corrected
full replay is running on i265. Frozen study binaries remain unchanged.

## Repeated first-moment underflow from a normal gradient

The L2-only repair's real-cell profile still attributed all 262 assist samples
to the first-moment FMA. A separate ten-second sample measured 52,331,398 assists.
That incomplete replay is preserved as a diagnostic, not a complete fit.

`tiny-moment` starts w=1e-302, zero moments and L2 scale 1e-5. The gradient is
normal, but `(1-beta1)*g` is subnormal. With zero m, every first-moment FMA
underflows and its flushed output stays zero, so clearing stored moments alone
cannot stop the repeated assists. For the same 2,000 steps / 104,960 parameters:

| First-moment behavior | Elapsed seconds | User seconds | FP assists |
|---|---:|---:|---:|
| Compute then flush | 1.930437168 | 1.927052000 | 52,487,710 |
| Bypass guaranteed flushed zero | 0.395777772 | 0.338942000 | 0 |

The bypass requires m=0, a nonzero gradient, and a conservative magnitude
bound derived from the actual `(1-beta1)` coefficient. It selects the same
signed zero that a subnormal result would flush to. Adjacent-value and signed
zero tests compare it against the independently rounded FMA. Nonpositive or
nonfinite coefficients retain the original arithmetic. L2 scales above one
likewise retain products that can turn a subnormal weight into a normal result.

The corrected real-cell diagnostic sample on i265 reports 53,693,626,562 cycles,
119,215,927,819 instructions and 636,657 assists in 10.002780896 s. It is not an
epoch-matched speed comparison. Native NEON decay, boundary, fused/unfused and
dispatch checks pass, as does strict all-target probe Clippy on ARM. Resource
record: `run-heavy: done rc=0 20s | peak-RSS 0.37GiB | min-avail 26854MiB | peak-load 6.57`.
The full i134 baseline, corrected i265 replay, queued dev verification and
complete byte comparisons remain in progress.

## Completed numerical replay before the inactive-state optimization

The versioned repaired trainer completed the registered `Mw025/without_kadid_s0`
control on i265 CPU 2: 120 epochs, 50,000 draws per epoch, initialization seed
1101, sampling seed 101, v3 dispatch, one thread. It used the frozen image,
original argv and authority; only the executable and fresh output paths changed.
The frozen study's control continued on another P-core. This is an intermediate
complete replay, not the final matched-host before/after speed comparison.

| Measurement | Repaired replay v3 |
|---|---:|
| Perf elapsed seconds | 3081.861411501 |
| Perf user seconds | 3075.294343000 |
| P-core cycles | 16,124,958,389,993 |
| P-core instructions | 35,777,423,536,219 |
| FP assists | 26,369,672 |
| `/usr/bin/time -v` maximum RSS (KiB) | 2,131,976 |
| Swaps | 0 |

The output directory is `intel/intel-after-v3/` under the artifact root above.
`intel/comparison-v3/COMPARISON.json` records native comparisons, SHA-256s and
paths to complete zero-based byte-offset listings. Both original and replayed
predictions contain 7,869 rows / 184,397 bytes and have SHA-256
`9333a13dff53fa4200166e723a9c33f0a48cf32e6beacd40856163571d536eff`.
All prediction bytes match; zero rows differ as text or parsed f64 values.

Raw model files differ, because they embed run provenance. Native `strip`
removing only the `zentrain.repro` key makes the entire files identical:

| Artifact | Original bytes | Replay bytes | Different common offsets | Extra replay tail bytes | Bytes after removing only repro |
|---|---:|---:|---:|---:|---:|
| refit best | 37,304 | 37,541 | 16,154 | 237 | 31,496, identical |
| refit last | 34,808 | 35,064 | 16,331 | 256 | 28,983, identical |
| native dense | 34,808 | 35,064 | 16,331 | 256 | 28,983, identical |
| freshly packed f16 | 30,444 | 30,700 | 16,539 | 256 | 24,619, identical |
| best spec JSON | 11,992 | 13,136 | 11,390 | 1,144 | not stripped |

The spec's changed fields are argv paths, cwd, input paths and timestamp; its
numerical settings match. Packing uses the existing canonical production packer,
the frozen CID22 TRAIN anchor and the original E33 output stage; it adds no
model selection. The frozen predictor itself reproduces the original prediction
TSV exactly before comparing the candidate. The native model reader confirms
`optimizer_arithmetic=adam-f64-normal-moments-v1` in the candidate's embedded
`zentrain.repro`. Raw model bytes are not claimed identical.

The next exact optimization bypasses arithmetic/stores for chunks whose +0
gradient and moments are a fixed point, including conservatively bounded
positive weights whose L2 contribution cannot start a normal moment. The full
scalar composition remains its oracle. Boundary, signed-zero, exceptional
hyperparameter and ordinary tier-parity tests pass on native Intel; final
complete timing and AVX-512 execution remain separate gates.
