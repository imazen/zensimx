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
It has not been deployed to registered scientific fits. Both original
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
is not recomputed. Every one of the 120 finalized checkpoints must match after
removing only reproduction metadata. These are two engineering qualification
runs, separate from the registered scientific fit count. Registration grants
no scientific runtime substitution; the existing downstream gates stay in force.

Raw evidence and complete durable checksum readback:
`/mnt/v/output/zensim/shaping-completion-20261107/gradient-kernel-probe-20261109/`,
mirrored under the same suffix in `/mnt/tower/output/zensim/`.
Candidate build and component-test logs are on r7900x under
`/mnt/v/output/zensim/shaping-completion-20261107/paired-gradient-native-20261109/`.
