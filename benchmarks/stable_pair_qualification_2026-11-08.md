# Stable pair export qualification

This is numerical engineering evidence, not a quality or generalization result.
No fits, tuning, outer evaluation or final-holdout consumption occurred.
Protocol: [qualified exports](../docs/SHAPING_STABLE_PAIR_QUALIFICATION.md).

Four original diagnostic checkpoints failed the unchanged 1e-5 damage-order gate.
Their direct-max native exports have zero damage-order violations, exact identity
100 and finite extreme rays. The original failed artifacts remain preserved.
Four fixed setting-zero, seed-one pretrained checkpoints passed the same gates.
All eight comparisons reproduced stored original predictions; unmarked V3
predictions were bit-identical to those originals on every compared row.

| Checkpoint | Family | Epoch | Compared TRAIN + inner-validation rows | Maximum absolute export score delta |
|---|---|---:|---:|---:|
| 1a22976d73e29a7c2217 | Monotone SiLU | 219 | 15,109 | 0.00018501281738636521 |
| d20acb0b9441fc96aab8 | Product joint identity | 119 | 15,109 | 0.00007200241088867188 |
| 7e28ac0fc83682e66022 | Separate joint identity | 119 | 15,010 | 0.00006079673767089844 |
| 46c5fe4370e1e47e9b4a | Product joint identity | 99 | 15,051 | 0.00016880035400390625 |
| c0d9aaabcb8d92973557 | Product pretrained frozen | 119 | 15,109 | 0.0000324249267578125 |
| 0be5b8631033a77025da | Product pretrained joint | 119 | 15,109 | 0.0001220703125 |
| 06e6913fc8a8e1873ab5 | Separate pretrained frozen | 119 | 15,109 | 0.0000324249267578125 |
| 928af6bf2d152f2e1ae9 | Separate pretrained joint | 119 | 15,109 | 0.00004863739013671875 |

Total: 120,715 row comparisons. Native structural admission preserves all stored
parameters, scalers and unit-order bits. Stable export changes evaluation order;
the measured score deltas are not quality improvements. Every paired cell in an
admitted family must use the same representation, including scheduled diagnostics.

The proof root is `remaining-order-gates-20261108` under the study output and its
durable mirror. Its 804 artifacts total 257,035,039 bytes before final inventory.
Cold destination readback and executable-bit verification passed. Its deduplicated
code-free replay closure contains 33 opaque Parquet tables, eight original bakes
and two native executables; `REPRODUCE.sh` prepares eight conversions and 248
prediction comparisons. The prepared full replay was not redundantly rerun.

Native wrapper records: four initial probes 1 s / peak RSS 0.12 GiB; four complete
population comparisons 11 s / 0.22 GiB; four additional comparisons 13 s / 0.22 GiB.
The qualification archive took 8 s / 0.02 GiB. These are the corresponding
`run-heavy` peak-RSS records, not RSS extrapolations or runtime benchmarks.

The first uniform evaluation wave completed 271 models and 1,948 report positions.
Its collection verified 62,607 artifacts / 7,874,163,404 bytes and every candidate's
original provenance before publication. Collection took 234 s / peak RSS 0.10 GiB,
minimum available RAM 42,769 MiB. These collection bytes are mainly inspection
JSON, converted bakes and scalar predictions; no pixel diagnostic maps were made.

Evidence pointers, receipts and input hashes remain in the study's sealed
qualification, wave registrations and collection directories. Completion of this
engineering wave does not complete the registered downstream scientific study.
