# Fixed external shaping comparison — November 8, 2026

Incomplete: six corrected jointly trained pretrained models are not yet reported here.
The first 28 models (24 neural, three tree and one deterministic additive model)
have both external datasets reported, with every registered paired seed retained. These are fixed
setting-0 / inner-0-0 comparisons, not tuned finalist results.

Mean signed SROCC across the three paired seeds, computed from the preserved
native per-fit panels:

| Approach | CSIQ866 | LIVE clean10/267 | LIVE full779, overlap |
|---|---:|---:|---:|
| extra_trees | 0.959952 | 0.944171 | 0.940379 |
| product_joint_identity | 0.953635 | 0.962849 | 0.965796 |
| product_pretrained_frozen | 0.955031 | 0.963537 | 0.965663 |
| relu | 0.942569 | 0.962992 | 0.964200 |
| separate_joint_identity | 0.955053 | 0.956459 | 0.965584 |
| separate_pretrained_frozen | 0.955550 | 0.954860 | 0.964601 |
| shaped_additive | 0.935884 | 0.941939 | 0.928206 |
| silu_diagnostic | 0.893703 | 0.797161 | 0.838427 |
| silu_monotone | 0.939651 | 0.963340 | 0.963649 |
| softplus | 0.940742 | 0.963019 | 0.963212 |

Product and separate identity-initialized shaping improve these fixed-condition
CSIQ point means relative to ReLU. The clean LIVE subset does not show a consistent
gain: product is close to ReLU and separate shaping is lower. This does not
establish universally better per-feature shapes. Ordinary SiLU's unconstrained
diagnostic is weaker in these conditions; this is not its selected best-setting
comparison. Differences between the anchored activations and ReLU are small on
LIVE clean10. Corrected frozen product shaping also improves the CSIQ point
mean; its LIVE clean10 difference is small. Corrected frozen separate shaping
has a lower LIVE clean10 mean than ReLU. No significance claim is made from
these means. The six corrected frozen models retain all paired seeds and the
same prespecified setting; they were not selected on these external responses.

The interactive external page retains each fit's signed SROCC, Kendall,
increasing-logistic PLCC, raw PLCC, 95% whole-source SROCC interval and raw
scatter. Intervals use 1,000 draws and seed20261108, with the same source order
for paired seeds. Their descriptive interpretation and small-source limitation
remain visible in the native report.

Registration and exact model bindings precede target parsing; see
[the protocol](../docs/SHAPING_EXTERNAL_VALIDATION.md). Features use the exact
Rev7 extraction build and feature-set identity admitted by training. No external
outcome influences tuning, rescues, selection, calibration or serving promotion.

Artifacts: `/mnt/v/output/zensim/shaping-completion-20261107/external-validation-20261108/`.
The initial neural batch preserved 35 successful dataset jobs, two failed additive
adapter invocations and the earlier successful ReLU/CSIQ pilot. The additive bake
requires its frozen 2,952-term basis; the ordinary dense-feature predictor refused
it. No training was repeated. The dedicated native basis adapter subsequently matched all15,374 frozen
responses exactly, including the existing identity/damage probe rows. Tree
inference matched all15,109 frozen raw responses exactly; its unchanged parity
gate also passed. No tolerance was relaxed. Qualification SHA is
`4d2ff83252ef37f570f1b1582b58ff4d203f57c858dadc6d8c8a7633de389ff5`.
The additive model remains an opaque-vector model requiring its frozen basis,
not a directly pixel-servable bake.

Native adapter qualification and the eight external control jobs completed:

```text
run-heavy: done rc=0 8s | peak-RSS 0.20GiB | min-avail 10512MiB | peak-load 24.43
run-heavy: done rc=0 8s | peak-RSS 0.26GiB | min-avail 25580MiB | peak-load 2.32
```

All 12 corrected frozen-model dataset jobs completed on an approved worker,
reusing the existing frozen data payload and adding 5,225,822 bytes of bound
models/reports. No new training or feature extraction ran for this comparison:

```text
run-heavy: done rc=0 18s | peak-RSS 0.26GiB | min-avail 18217MiB | peak-load 20.76
```

The cold helper build ran at one CPU thread alongside active fits, with no
training changes. Scoped clippy passed with warnings denied. Compiler wrapper
accounting:

```text
run-heavy: done rc=0 1542s | peak-RSS 1.35GiB | min-avail 9807MiB | peak-load 25.94
run-heavy: done rc=0 200s | peak-RSS 0.65GiB | min-avail 10106MiB | peak-load 24.30
```

Neural batch wrapper accounting, including an SSD admission wait:

```text
run-heavy: done rc=1 244s | peak-RSS 0.26GiB | min-avail 23679MiB | peak-load 6.43
```

The pilot independently completed with:

```text
run-heavy: done rc=0 1s | peak-RSS 0.25GiB | min-avail 24310MiB | peak-load 6.24
```

Browser verification passed for CSIQ866, LIVE clean267 and LIVE full779 scatter
points, with no page errors or mobile overflow. Build and adapter qualification
are separate from training-fit counts. The first-18 archive has been cold-read and checksum verified: 448 artifact
files /280,955,068 bytes, plus its receipt (449 final inventory files). Receipt
SHA-256 `32ba7786c28a66d5edd34b55d434ea1f1c15d0c88aad0de43dbda28f6ffbacc5`;
manifest SHA-256 `d6e1d7037c2d169b12b918d55a1373baef81f1cdfeef30daceec12a66f475a55`.
Destination: the run's durable `external-neural-first18-archive-20261108/` mirror.
Browser PNG witnesses are preserved in a verified tar inside this table archive;
the original witnesses remain available. Mirror wrapper accounting:

```text
run-heavy: done rc=0 15s | peak-RSS 0.02GiB | min-avail 45276MiB | peak-load 2.00
```

The four external controls and their native parity evidence also have a verified
durable snapshot: `external-controls-native-archive-20261108/`, 177 artifacts /
231,062,836 bytes, plus its receipt. Receipt SHA-256
`4e53b9a73070068d9d9df04aa2bbce54a69cf9f55bc9f9bf6f7636c71e44d9fa`;
manifest SHA-256
`29f80a56246153e7154cff0fb616cc7ff462bc98800d913714ea73c6ae445f35`.
All artifacts were reread from the destination with cache-eviction requests.
The immutable snapshot includes both failed additive adapter attempts alongside
the successful native-basis outputs. Its binaries, raw arrays, predictions,
bootstrap inputs and browser evidence remain inspectable without fitting.
`external-controls-reproduction-20261108/` separately preserves reproduction
commands and the exact executed qualification source; the first snapshot was
already sealed before those explanatory files were added.

```text
run-heavy: done rc=0 5s | peak-RSS 0.03GiB | min-avail 44497MiB | peak-load 0.70
```

The six corrected frozen models have a separate durable snapshot,
`external-corrected-frozen-archive-20261108/`: 216 artifacts /93,678,019 bytes
plus its receipt, all destination checksums reread with cache-eviction requests.
Receipt SHA-256
`59d17318df7b481334af356bb0d77c5d7664e17808a4c1519e38a011dc51494a`;
manifest SHA-256
`e03eea7c8e5dd56e5678a7e9cef9f086696d331eb8202f31b8bb353ff9a7ad86`.
Its `REPRODUCE.sh` executed on an approved worker: all 12 prediction TSVs matched
byte for byte, and all signed-statistic and source-interval commands completed.
This replays inference only. Frozen bakes, inference corpora, native helpers,
targets/source IDs and command lines are included; shared training data is not
duplicated per scoring batch. Browser evidence covers all 84 current
model/population plots and the delayed-response selection regression.

```text
run-heavy: done rc=0 3s | peak-RSS 0.03GiB | min-avail 44608MiB | peak-load 2.35
run-heavy: done rc=0 2s | peak-RSS 0.11GiB | min-avail 18458MiB | peak-load 20.41
```
