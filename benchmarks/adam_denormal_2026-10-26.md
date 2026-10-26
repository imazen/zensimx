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

## Final source verification on Intel

The final source passes CI-exact workspace Clippy with all targets/features
(excluding the browser-only crate), with warnings denied. The trainer Adam and
L2 unit tests, both existing Adam integration suites and the standalone
release probe tests pass unchanged expectations on i265. Strict all-target
release probe Clippy and formatting pass. Native NEON execution and the clean
x86 cross-compile check also pass; native AVX-512 execution is scheduled for the
quiet dev window.

Combined build/check resource record:
`run-heavy: done rc=1 340s | peak-RSS 1.35GiB | min-avail 18555MiB | peak-load 6.55`.
Every Rust check passed. That command's final script lint fails because the
isolated remote source copy lacks the pre-existing `zensim--rev7main` sibling
referenced by `candidate_registry.py`; the same script lint passes in the local
workspace (1,287 scripts runnable). No script expectation or path was changed to
accept the remote layout. Full output is in
`intel/results/verification-final2.log` under the artifact root. The standalone
native model reader also verifies the new arithmetic version in the completed
replay's embedded provenance.

## Final fixed-work probe

Final source `286fd9e4`, preserved trainer SHA-256
`cef30445eaf804969145f89e59282c1e009668462a1234c8f93c68d4053d8159`.
The kernel file hash is
`2430fb5324bc8f06d99e5fb553de30bdd681d003173d533171286bc69790b86a`.
The final rebuild resource record is
`run-heavy: done rc=0 33s | peak-RSS 1.30GiB | min-avail 19252MiB | peak-load 4.16`.
The following single observations use the same i265 CPU 2, 2,000 steps,
104,960 parameters and v3 tier as the original probe:

| Final state | Elapsed seconds | P-core cycles | P-core instructions | FP assists |
|---|---:|---:|---:|---:|
| normal | 0.187329725 | 966,317,142 | 3,682,696,729 | 0 |
| decay | 0.128464481 | 663,815,065 | 2,545,934,359 | 368,270 |
| tiny-l2 | 0.807562376 | 4,151,191,942 | 2,009,511,391 | 0 |
| tiny-moment | 0.807628885 | 4,150,042,174 | 2,009,406,363 | 0 |
| subnormal-weight | 0.807697956 | 4,150,957,465 | 2,009,883,025 | 0 |

Final normal-state moment bits match the original probe. All other final
moments are +0. The decay arm still pays for transitions into the flushed
state; repeated tiny-gradient states have zero assists. The normal control
has added instruction/time cost, and the tiny-weight fixed-point shortcut is
slower than the earlier L2-only probe despite fewer instructions. These are
reported costs, not a claim of improvement on every input. Full-cell timings
use the final preserved executable. Probe resource record:
`run-heavy: done rc=0 3s | peak-RSS 0.02GiB | min-avail 20641MiB | peak-load 3.34`.

Final native NEON verification uses the exact kernel hash above on
Neoverse-N1: all applicable decay, signed-zero, L2-boundary, fused/unfused and
dispatch checks pass, as does strict all-target release probe Clippy including
the native provenance reader dependency. Full output:
`arm/neon-final-versioned-probe.log`. Resource record:
`run-heavy: done rc=0 14s | peak-RSS 0.32GiB | min-avail 27152MiB | peak-load 6.37`.

## Completed final Intel replay and section-preserving byte proof

Final v4 completes all 120 registered epochs on i265 CPU 2 with the preserved
`cef30445…8159` trainer (full SHA above). Perf elapsed: **2779.769636014 s**;
user: 2775.920338000 s; system: 1.605729000 s. P-core counters:
14,849,340,987,154 cycles; 32,993,011,393,865 instructions;
26,369,672 FP assists. GNU time reports 2,132,120 KiB maximum trainer RSS and
zero swaps. Docker imposes a separate 12 GiB no-swap cap. The run-heavy record
covers the Docker client, not container RSS:
`run-heavy: done rc=0 2780s | peak-RSS 0.03GiB | min-avail 18540MiB | peak-load 3.52`.

The final replay again matches all 7,869 / 184,397 prediction bytes and all
native provenance-stripped bakes. Raw final comparisons:

| Artifact | Original bytes | Final bytes | Different common offsets | Extra final tail bytes |
|---|---:|---:|---:|---:|
| refit best | 37,304 | 37,537 | 16,133 | 233 |
| refit last | 34,808 | 35,060 | 16,326 | 252 |
| native dense | 34,808 | 35,060 | 16,326 | 252 |
| fresh packed f16 | 30,444 | 30,696 | 16,534 | 252 |
| best spec JSON | 11,992 | 13,136 | 11,390 | 1,144 |

The existing native `strip` command re-emits a model. A stronger independent
check uses the pinned native `append_metadata_utf8` section-splice owner to
replace only `zentrain.repro` with `{}`. It preserves packed weight bits,
permutations and all other sections without rebaking them. All four resulting
whole files are byte-identical:

| Constant-provenance artifact | Bytes | SHA-256, both arms |
|---|---:|---|
| best | 31,514 | `69a935ed93acb673e62f46b9086de22a334c9ed743d69aa3f6984e128ccc88aa` |
| last / dense | 28,998 | `bc4a61a35af4c040264a96b588828fffe6afd9272294be4a1fff498143333f97` |
| packed | 24,634 | `aa31427404cc43922185ad212b7c238e72efba39c51c81eec72dd2ef42c4f11b` |

Evidence: `intel/comparison-splice-v4/COMPARISON.json`, with every raw differing
byte and complete native JSON provenance differences. There are 34 changed
embedded provenance fields (paths, container hostname, source identity,
timestamp and the new arithmetic version) and 28 spec fields (paths, cwd,
timestamp); the numerical settings and input hashes match. This is the frozen
control only, not a promise that all possible future fits keep their outputs.
The section-splice tool SHA is
`6bf5dbb3b24adfcfa9c744f4cf27f115a84d6027176838a0c02e0da6ef7a5bf5`.
Its strict all-target Clippy/build resource record:
`run-heavy: done rc=0 8s | peak-RSS 0.25GiB | min-avail 22448MiB | peak-load 1.09`.
The comparison completes with
`run-heavy: done rc=0 3s | peak-RSS 0.03GiB | min-avail 22540MiB | peak-load 1.10`.
The frozen i134 baseline and quiet dev before/after measurements remain pending.

## Synthetic zero-moment memory setup

A cycles profile of the final probe's tiny-moment case attributes 99.97% of
samples to fused Adam and 86.76% to the v/m load and following instruction.
Retired-load/cache, alias, branch, TLB and machine-clear counters do not identify
FP assists as its remaining cost. One stall-counter observation reports
31,646,139,457 load-bound cycles out of 41,873,664,125 cycles; the preceding
zero-bypass implementation reports 376,145,669 out of 18,295,135,619. These
counter passes use 20,000 updates, not the 2,000-update tables above. Their
complete raw files remain in `intel/results/`.

Add an optional `lazy|touched` probe setup. `touched` writes the initial value
into every m/v element through safe `black_box` references before the update
loop. Both modes have identical numerical state and use the same kernel. This
controls lazy zero allocation, rather than changing trainer arithmetic. With
one preserved executable (`d54009a05a8d8d9bad0d4b2abc052d183c71100fd8504175cea1564b0fd432ba`),
interleaved on i265 CPU 2, tiny-moment, 104,960 parameters, 20,000 updates:

| Moment setup | Elapsed seconds | P-core cycles | Instructions | Load-bound cycles | FP assists |
|---|---:|---:|---:|---:|---:|
| lazy 1 | 7.806803397 | 41,849,258,162 | 20,002,718,251 | 31,407,674,113 | 0 |
| touched 1 | 0.781310919 | 4,189,468,006 | 19,969,671,555 | 334,002,354 | 0 |
| lazy 2 | 7.800425164 | 41,844,218,992 | 20,001,028,695 | 31,511,402,513 | 0 |
| touched 2 | 0.777104722 | 4,165,663,381 | 19,969,653,732 | 313,368,591 | 0 |

All final moment bits are +0 in both modes. Writing the initial moment pages
removes this probe's load stall. Lazy shared-zero-page mappings are the likely
cause; the precise hardware mechanism is not established. The normal and decay
probes already initialize moments to nonzero values; the real fit also updates
moments before they decay. At this investigation stage the trainer is unchanged; the store-preserving
successor below addresses the observed cost. No original lazy timing is
replaced or scaled to another size.
The new default remains lazy so earlier commands retain their setup.

The updated tool passes its native Intel tests and strict all-target Clippy.
Build record:
`run-heavy: done rc=0 5s | peak-RSS 0.30GiB | min-avail 22485MiB | peak-load 1.25`.
Page-control record:
`run-heavy: done rc=0 17s | peak-RSS 0.02GiB | min-avail 22437MiB | peak-load 1.48`.
Cycles profile record:
`run-heavy: done rc=0 8s | peak-RSS 0.08GiB | min-avail 22650MiB | peak-load 1.19`.
Cache/alias/TLB passes:
`run-heavy: done rc=0 34s | peak-RSS 0.02GiB | min-avail 22362MiB | peak-load 1.57`.
Stall passes:
`run-heavy: done rc=0 11s | peak-RSS 0.02GiB | min-avail 22718MiB | peak-load 1.21`.

The final section-splice/page-control tool also passes native Neoverse-N1
strict all-target Clippy and its applicable Adam tests with the unchanged
kernel hash. Evidence: `arm/neon-final-splice-page-tool.log`.
`run-heavy: done rc=0 28s | peak-RSS 0.31GiB | min-avail 26923MiB | peak-load 6.65`.

## Store-preserving successor

The zero-state shortcut introduced a load-bound regression for initially lazy
zero-moment pages. Retain identity m/v stores while skipping FP arithmetic.
A single opaque +0 value per call prevents LLVM from deleting those stores
after the zero-state proof. This changes memory materialization, not arithmetic,
and retains `adam-f64-normal-moments-v1`. The preceding v4 timings above remain
intermediate measurements with their original preserved binaries.

For the same 20,000-update tiny-moment probe on i265 CPU 2:

| Store-preserving setup | Elapsed seconds | P-core cycles | Instructions | Load-bound cycles | FP assists |
|---|---:|---:|---:|---:|---:|
| lazy | 0.840776747 | 4,334,262,154 | 21,543,699,161 | 430,284,647 | 0 |
| touched | 0.846275218 | 4,545,693,223 | 21,544,264,911 | 575,981,667 | 0 |

Both final m/v bit patterns remain +0. Preserve the 0.840776747 s observation's
user time (0.806617000 s) separately; the wall/user gap is not concealed. Strict
workspace all-target/all-feature Clippy and the existing Adam/L2/precision and
probe tests pass on Intel with unchanged expectations. Resource records:
`run-heavy: done rc=0 5s | peak-RSS 0.31GiB | min-avail 22298MiB | peak-load 1.23`
(probe build/check),
`run-heavy: done rc=0 2s | peak-RSS 0.02GiB | min-avail 22558MiB | peak-load 1.13`
(counter pair), and
`run-heavy: done rc=0 44s | peak-RSS 1.30GiB | min-avail 20628MiB | peak-load 2.42`
(full trainer checks/rebuild).

The preserved successor trainer SHA-256 is
`c82bdf978fe05c99f8e0d1a7f8788b15408def271d2145e86c2c3e1325a5e44b`.
Its probe SHA is
`5459d9f22685e9f5ddb9e9fd976dcdabbd635dcaf8f841e83e5fcf746d88bbb6`.
Full successor control timings and byte comparisons are measured separately.
The dev frozen-binary baseline is already running; its subsequent candidate
phase explicitly validates the successor's native test binary before starting.

## Quiet dev baseline and native successor gate

The frozen `acfb2a9f…fe28` trainer completed all 120 control epochs on dev CPU 2
in 1766.748224283 s (1763.822494000 user, 0.697226000 system).
Counters: 9,390,750,362,071 cycles; 27,774,659,110,690 instructions;
4,995 `fp_disp_faults.all`. GNU time maximum RSS: 2,132,008 KiB; swaps: 0.
Intel `assists.fp` is not exposed by this AMD PMU (the actual perf rejection
and full event inventory are retained under `dev/counter-support/`). AMD
floating-point dispatch faults are a different event and are not presented as
Intel FP assists.

The source-identical successor gate on dev executes all native x86 dispatch
permutations, including AVX-512, with unchanged expectations. Its test executable
SHA is `28ed1b3d3e7b4de68ecbdc61be2561b511203e430365c6769ef63fd3db848944`;
its command, checksum and complete output are retained in
`dev/results/dev-after-v4.native-test-command.json` and
`dev/results/dev-after-v4.native-tests.log`. The after directory's original
iteration label is retained; its INVOCATION records the actual successor
trainer `c82bdf97…e44b`. The source hash on Intel and native ARM is
`4dc9e41f303cd021950936484d7e1a81f537f9282e5301394b6eb13edd06b91e`.
Final ARM source verification:
`run-heavy: done rc=0 9s | peak-RSS 0.32GiB | min-avail 26796MiB | peak-load 6.23`.
The same shared heavy-lock scope covers both quiet dev fits; its final resource
line and successor counters are recorded after completion.

The original control folder has best/last refit bakes and a dense bake, with no
pre-existing packed bake. Packed comparisons therefore freshly pack both the
frozen reference and candidate through the same canonical owner and TRAIN
anchor. Comparison subprocesses explicitly use one thread and v3 dispatch.

## Store-preserving full replays and normal-path profiling

The store-preserving successor completes the i265 control in 2721.960114026 s
(2718.716656000 user, 1.198070000 system), with 14,575,609,544,173 cycles,
33,168,445,500,348 instructions and 26,369,672 FP assists. Maximum RSS is
2,132,212 KiB, with no swaps. Resource wrapper:
`run-heavy: done rc=0 2722s | peak-RSS 0.03GiB | min-avail 19211MiB | peak-load 3.14`.
The wrapper measures its client; GNU time measures the actual container trainer.

On quiet dev, the same successor takes 2140.277556482 s (2137.630308000 user,
0.606442000 system), 11,349,578,767,010 cycles, 33,177,262,625,347 instructions
and 8 AMD FP dispatch faults. Maximum RSS is 2,131,308 KiB; no swaps. This is a
21.1% elapsed-time regression from the frozen 1766.748224283 s baseline.
Both trainers exit successfully and write COMPLETE records. The outer resource
wrapper subsequently exits 2 with an unexpected-EOF error in its footer. Its
source modification time is 04:53:43 UTC, during the pair, and its current syntax
check passes. No wrapper peak-RSS footer is available for this pair; the trainer
GNU time records and Docker's 12 GiB cap remain available. Later long runs use a
checksum-recorded private snapshot of the wrapper to avoid live script changes.

All three new comparisons (Intel successor, dev frozen baseline, dev successor)
reproduce all 7,869 prediction rows exactly, including their serialized bytes.
Best/last refit, dense and freshly packed bakes become byte-identical after
the native section-splice tool replaces only `zentrain.repro` with `{}`.
This preserves every other raw section and does not rebake the model. Raw bake
and specification differences remain recorded, with every offset and JSON path.

The new `normal-l2` probe covers the common nonzero L2/moment path. On i265 CPU 2,
20,000 updates over 104,960 parameters take 3.731870611 s, 19,811,587,703 cycles,
55,145,586,719 instructions and zero assists. Final w/m/v bits are
`3dc1e0912e0ca552` / `3cbabd0460f81372` / `3db676e45977c4b5`. Cycle sampling
attributes 99.89% to the fused Adam owner. The preserved annotation shows repeated
classification/blend work around the canonical sqrt/divide chain. This motivates
removing redundant classification without changing any arithmetic result.
Probe build/check resource record:
`run-heavy: done rc=0 2s | peak-RSS 0.21GiB | min-avail 22259MiB | peak-load 1.05`.
Counter/profile pair:
`run-heavy: done rc=0 8s | peak-RSS 0.05GiB | min-avail 21975MiB | peak-load 1.15`.

## Remove redundant normal-path classification

Use integer ordering of positive f64 encodings for the L2 magnitude cutoff,
avoiding both the preliminary weight flush and an FP comparison on its original
value. Skip first-moment blends when no lane is tiny. The fused +0-gradient,
zero-input path reuses the L2 helper's classified result. Arithmetic ordering,
cutoffs and rounding remain unchanged. The arithmetic tag stays
`adam-f64-normal-moments-v1`; each measurement preserves a distinct executable.

| i265 CPU 2 probe | Steps | Elapsed seconds | Cycles | Instructions | FP assists |
|---|---:|---:|---:|---:|---:|
| normal L2, touched | 20,000 | 2.475792475 | 13,277,077,206 | 48,525,337,601 | 0 |
| tiny moment, lazy | 20,000 | 0.820526093 | 4,402,860,530 | 21,748,161,218 | 0 |
| decay, lazy | 2,000 | 0.118682449 | 635,605,127 | 2,644,673,705 | 367,360 |

The normal-L2 final weight and both moment bits exactly match the preceding
probe. The decay probe's remaining assists are transient boundary crossings;
final m/v are +0, and repeated tiny-state updates incur zero assists.
All 22 native Intel probe tests and all 15 applicable native ARM tests pass,
including unchanged tier parity expectations and boundary-neighbor checks.
Strict probe Clippy passes on both. Kernel source SHA-256:
`60600a314bad60a8cee80c95d5c1233492a616a71e576271a7eeb5f944f8505d`.
Resource records:
`run-heavy: done rc=0 5s | peak-RSS 0.30GiB | min-avail 22062MiB | peak-load 1.30`
(Intel probe checks/build),
`run-heavy: done rc=0 3s | peak-RSS 0.02GiB | min-avail 22235MiB | peak-load 1.29`
(Intel counters), and
`run-heavy: done rc=0 10s | peak-RSS 0.32GiB | min-avail 29388MiB | peak-load 1.21`
(native ARM checks).
The canonical workspace Clippy recipe and existing trainer Adam/L2/rsqrt tests
also pass with unchanged expectations. Release trainer SHA-256:
`1106eb3d3c5b58d1bc8ef067f88d97d2733793c798c241ae3cdd39376413cea1`;
native x86 test executable:
`604e100d921e735cc16bc62b7d8f39f2707fe8b7f781b1cd5e10bf95cfc56ea5`.
`run-heavy: done rc=0 51s | peak-RSS 1.30GiB | min-avail 20671MiB | peak-load 6.07`.

## Complete optimized Intel replay and auxiliary metadata finding

The `1106eb3d…cea1` successor completes the same i265 control in
2820.563446415 s (2817.506403000 user, 1.144824000 system), with
15,154,811,841,658 cycles, 32,872,884,536,270 instructions and 26,369,672
FP assists. Maximum trainer RSS is 2,132,044 KiB, with no swaps.
`run-heavy: done rc=0 2821s | peak-RSS 0.03GiB | min-avail 19968MiB | peak-load 4.32`.
This full observation is slower than the preceding 2721.960114026 s candidate;
the normal-L2 microbenchmark improvement does not establish a full-cell gain.
These i265 observations share the host with another frozen control. The fully
matched before/after measurement uses i134's owned input snapshot.

All 7,869 prediction rows remain byte-identical. Unlike previous replays,
replacing only `zentrain.repro` does not make these model files identical:

| Raw artifact | Different common offsets | Unmatched tail bytes |
|---|---:|---:|
| best refit | 16,480 | 214 |
| last refit / dense | 16,145 | 233 |
| freshly packed | 16,352 | 233 |
| best specification | 11,390 | 1,144 |

The native model owner confirms identical layer dimensions, activations,
weight-storage bits (including packed values and scales), bias bits and scaler
bits. Every other inspected numeric view matches. The sole non-repro metadata
difference, in best/last/dense/packed, is the absent `zentrain.feature_set_id`,
whose reference value is `basic+peaks+v2@w1825/rev7_rowlanes#36c3f3af`.
The recorded input hashes are identical, and every table resolved its feature
identity at initial admission. At the final metadata pass, the shared coverage
file and sidecar no longer exist at their original scratch path. The trainer
therefore reports an unresolved leg and omits that stamp. This is retained as
an artifact difference; no missing bytes are described as equal. Dev and i134
use owned coverage copies, including the original manifest.

The diagnostic `--inspect` reads the native model views and emits integer bit
patterns for all weight formats, biases and scalers, plus each other metadata
entry's exact bytes. Its SHA-256 is
`6c912be62a4b6d395e016790eea810157a60355d0c59e69abc667900bf8b7b0e`.
The comparison helper records component differences independently of the raw
byte and provenance-splice comparisons. Original failed splice reports remain
preserved alongside the stronger native component audit.
Strict inspector Clippy/build and real four-bake audit pass:
`run-heavy: done rc=0 2s | peak-RSS 0.22GiB | min-avail 24602MiB | peak-load 0.63`,
`run-heavy: done rc=0 3s | peak-RSS 0.03GiB | min-avail 24038MiB | peak-load 0.82`.
