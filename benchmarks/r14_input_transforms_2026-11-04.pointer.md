# R14 artifact pointer

Checked on dev, 2026-11-05. All required scientific stages are complete.
The canonical mirror passes full checksum comparison and three random SHA256
readbacks; its current receipt and inventory are named below.

Local root: `/var/tmp/byc-transforms-20261104/`.
Verified canonical mirror: `/mnt/tower/output/zensim/r14-transforms-20261104/`.
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
and measured limits.

The canonical mirror contains the full root except rebuildable `target/`.
`SHA256SUMS` owns per-file hashes; `ARCHIVE_VERIFIED.json` owns the complete
checksum comparison, three readback hashes and verification timestamp.
`record/` preserves the report, read tables, receipts and R14 execution owners
with a committed source identity in its `_MANIFEST.json`.

| Final packed model | SHA256 |
|---|---|
| T1a seed 0 | `6a28893f896696fde6832782631e0af0582e9a317478fc5dc621198716935487` |
| T1b seed 0 | `7a02cd54ba692f5995477657d9ec5ec125602678e1f397dcde9af46a53c75065` |
| T1c seed 0 | `de187fbabbdb1a1e999fed21c19f4589281e3e6184893c4ba559921922b71d98` |

`FROZEN_FITS.json` owns packed/stripped identities. `SOURCE_PINS.json` owns
serving binaries (`912609aa`); `ACTIVE_TRAINER.json` owns the T1 successor
(`fc41da56`), and `T2_TRAINER.json` owns the final projected-knee trainer
(build-only; T2 was not released). `source-capsules/NATIVE_SOURCE.json`
records the exact final-trainer git archive and its SHA256.
`READ_SOURCE_PINS.json` and `READ_COHORT_VERIFIED.json` bind the complete
24-stage frozen read cohort. `T2_RELEASE.json` and `SEED_ELIGIBILITY.json`
record the preregistered no-run/no-repeat decisions.

R13 comparisons are captured from completed width-owner artifacts only, with
exact source/model hashes in `R13_COMPARISON.json`; older snapshots are retained.
All of the above local paths were checked on dev. No R2 mirror was created.
