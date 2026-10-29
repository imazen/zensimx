# Human HDR source memberships, 2026-10-29

Assignments are frozen; TRAIN admission is pending source-screen review and
native photometric contracts. No subjective payload or image pixels read.
Producer: `944bbbd5c03a`; `scripts/byconstruction/dataset_keys.py hdr-membership`.

Local directory (checked on dev): `/var/tmp/byc-hdr-20261029/membership-v1/`.
Mirror: `/mnt/tower/output/zensim/byc-hdr-20261029/membership-v1/`.
Both files were compared byte for byte after copying. No R2 mirror created.

- `memberships.json`: `8f8cba8f408e4ac7899c3b88c257809e3249205f7880859356749d6e9b656a44`
- `_MANIFEST.json`: `1c5a726b421f2d2ed0ad27cf111f4d5189c41b34c033e80dd370573cc771bf36`

The manifest pins source-only release files, the producer and every original
reference/distortion file by the release inventory's SHA-256. The image hashes
are inventory assertions until native reading verifies the bytes. All 204
conditions are retained, including distinct rated conditions with identical
encoded bytes. Training labels remain forbidden until admission completes.

See [DATA_SPLITS](../docs/DATA_SPLITS.md) for roles and historical exposure.

## Active membership-v2: Ghent fallback

Owner light-contract decision supersedes Ghent's provisional TRAIN assignment.
All108 Ghent conditions are EVAL; 4Kdtb retains48TRAIN/48EVAL. No new labels
were read. Native source screening still gates the 4Kdtb TRAIN admission.

Producer `a245e1e5a4d3`; local (checked on dev)
`/var/tmp/byc-hdr-20261029/membership-v2/`; byte-identical mirror
`/mnt/tower/output/zensim/byc-hdr-20261029/membership-v2/`.

- `memberships.json`: `946058e666c2a4f83bff9b1c1de78b8c4c195c2898e9762ee52bca9ba9d39f29`
- `_MANIFEST.json`: `c98ffdd3b44466f4bb468ad25f1eb7f6ebf3619aeea043adbe71467c73e1c7b3`

v1 is retained as historical exposure evidence. No R2 copy created.
