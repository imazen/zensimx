# Paired shaper gradient reductions

The joint shaper computes two dot products against the same first-layer row.
Interleaving their independent accumulators reads that row once while retaining
the original order of additions in each sum. The helper also retains the
standard iterator sum's signed-zero identity. No approximate arithmetic is used.

The standalone probe at `73aef910a0bf` measured a bank of 984 rows with 128
hidden units on r3500 through zenbench. The first 32 gate-clean paired rounds
across two fixed 32-round processes were retained; rejected rounds excluded both
arms. Both original batch results, including their incomplete status, remain
unchanged. The report uses 10,000 paired bootstrap draws with seed 20261107.

| Kernel | Median time per bank | 95% bootstrap interval |
|---|---:|---:|
| Two passes | 151.370 microseconds | 151.165–151.562 |
| Paired pass | 88.541 microseconds | 88.222–88.822 |

The paired round time ratio is 0.585263, with interval 0.583528–0.586430.
This measures the synthetic dot-product bank only. Whole-training speed has not
been measured. Exact arithmetic checks cover empty and uneven widths, scaled
signed inputs and signed zero.

The production helper builds with warnings denied. Its gradient probe and all
eight input-shaping library tests pass. The trainer build's run-heavy peak RSS
was 1.70 GiB; the separate library-test compilation peaked at 1.20 GiB.
The candidate trainer SHA-256 is
`d68d5d443ba6fb14d99855084a7442edc13e728fb492bb19be2ea070987cb2de`.
Both original
240-epoch controls, joint-product and joint-separate identity shaping, now
passed every checkpoint comparison: 480 of 480 checkpoints matched after
removing only the reproduction metadata. The native phase inputs, settings,
seeds and exposure remained frozen. Other families keep the existing qualified
runtime; this evidence does not authorize substituting the candidate into them.

The full qualification archive passed durable checksum readback for 1,937 files
and 592,364,916 bytes under
`/mnt/tower/output/zensim/shaping-completion-20261107/paired-gradient-native-20261109/qualification`.
It retains the original references, candidate checkpoints, canonical bakes,
registration, reports and raw logs. These two engineering runs add no scientific
conditions to the registered study. Whole-training speed remains unmeasured.

`shaping_gradient_pretrained.py` separately registers corrected joint-product and
joint-separate pretrained controls at inner-0-0, setting 0, seed 0. Each check
imports that original fit's exact learned input bank and retains its complete
120-epoch, 50,000-pair-per-epoch second phase. The original pretraining prefix
was not recomputed. Both checks passed all 120 finalized checkpoints after
removing only reproduction metadata: 240 additional comparisons, or 720 across
all four controls. These are engineering qualifications, separate from the
registered scientific fit count. Their complete durable archive contains
977 files and 417,811,300 bytes under the corresponding
`paired-gradient-pretrained-native-20261109/qualification` suffix.

After all six workers validated the certificate and the registration/collection
owners were upgraded, the candidate was activated for unstarted cells in these
four classes. Existing scientific fits retain their original runtime. The
certificate preserves actual native build identities, original corrected-control
qualification and every scientific selection gate. See
[runtime substitution policy](../docs/SHAPING_RUNTIME_EQUIVALENCE.md).
The deployment's 61-file, 87,970,261-byte archive passed complete durable readback
at `runtime-equivalence-deployment-archive-20261109` beside the study archives.
It includes prebuilt tools, compressed source, qualification proofs and activation
bindings. Whole-training speed remains unmeasured.

## Live trainer CPU profiles

Two 15-second, 199 Hz captures observed running native trainers without sending
training signals or changing their commands. These are different families,
settings and CPUs; their percentages cannot establish a speed ratio.
Parent frames include unresolved addresses, so only leaf self cycles are used.

The candidate product-pretrained joint check on r7900x recorded 3,026 samples
with none lost. Fused AVX-512 Adam accounted for 38.78%, input-shaping gradient
updates 24.04%, max-pair projection 9.64%, shared-head backward 8.87% and paired
AVX-512 forward inference 7.98% of sampled cycles.

An original-runtime scientific Softplus rescue on i134 recorded 2,997 samples
with none lost. Keep its hybrid CPU events separate: the performance-core event
attributed 44.01% to fused AVX2 Adam, 37.77% to max-pair projection and 4.79% to
paired AVX2 forward inference. The efficiency-core event had only 48 samples;
its separate Adam/projection percentages were 52.20%/37.21%.

Both capture processes reported `done rc=0 16s | peak-RSS 0.05GiB` through
run-heavy. Minimum available memory and peak load were 13,507 MiB / 2.39 on
r7900x and 23,761 MiB / 8.59 on i134. These receipts measure the profiler,
not the target trainer's memory. The complete 20-file, 50,880,538-byte archive
passed durable readback at `trainer-gradient-profiles-20261109` beside the study
archives. Adam and constraint projection remain concrete optimization targets;
these windows do not measure the study's overall evaluation/checking fraction.

Raw evidence and complete durable checksum readback:
`/mnt/v/output/zensim/shaping-completion-20261107/gradient-kernel-probe-20261109/`,
mirrored under the same suffix in `/mnt/tower/output/zensim/`.
Candidate build and component-test logs are on r7900x under
`/mnt/v/output/zensim/shaping-completion-20261107/paired-gradient-native-20261109/`.
