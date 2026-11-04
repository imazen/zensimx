# R14 artifact pointer

Checked on dev, 2026-11-04. This experiment is still running; no archive
completion or model/read result is asserted here.

Local root: `/var/tmp/byc-transforms-20261104/`.
Planned final mirror: `/mnt/tower/output/zensim/r14-transforms-20261104/`.
R2 mirror: not created.

The `screen/` directory contains the TRAIN-only caller pool, target,
`INPUTS.json`, three frozen transform TSVs, `SUMMARY.json` and `_MANIFEST.json`.
The manifest owns dtypes, shapes, exact source-group hashes and output hashes.
Producer commit: `bfa87dd790f8122dd91ddc3c68f59f9d4a3b360c`.
Registration: `5b5a013482db4944042883995ad6892f92244c69`.

| Artifact | SHA256 |
|---|---|
| screen/TRAIN.f32 | `f60750b5cb0cdb284feb50ee39e0f53067cadfd25d4ab0a9caa33d376238b1da` |
| screen/human_score.f64 | `0bfd1b746e91f4e62bfc8bb71b8132ce0b48e4a5d7caa7e23cb46dc7b2345dce` |
| screen/T1a.tsv | `3612b34e4c074a35437b94d7789dac531209f6e1927bf78115cacb795c0723b8` |
| screen/T1b.tsv | `678bfbeb7a90165cf243ceffd52601ee12a2c542502fc0f9ce1bfc636e6bc523` |
| screen/T1c.tsv | `ba0020c411aaefb5cac84f6e03bc135c318cb5a94ca9263d5125c166d807df2a` |

`arm/` retains build/test logs and both byte-identical A0/A1 no-op goldens.
Both goldens have SHA256
`15641f4bbd3ebd26ab1f5a06fb08e7dca306b4dafb28df575cd1b4d6b83b1a90`.
Read [the report](r14_input_transforms_2026-11-04.md) for measured findings
and remaining work.
