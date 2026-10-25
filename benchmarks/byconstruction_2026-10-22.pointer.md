# Pointer: by-construction exploration data (2026-10-22)

Record: [byconstruction_2026-10-22.md](byconstruction_2026-10-22.md). Paths checked on **dev** only
(`/mnt/v` differs between hosts). Mirrored to tower at `/mnt/tower/output/zensim/byconstruction-2026-10-22/`
(265 MB, 2026-10-22; spot-checked sha256 equal for the mixture manifest, B's mixture scores, M's G-STEER result and
the all-family probe).

| What | Path (dev) | Size |
|---|---|---|
| Damage-mixture images, manifest, per-bake scores | `/mnt/v/output/zensim/byconstruction-2026-10-22/mixture/` | 137 MB |
| Label-free gate outputs for full-data T (NEARID, verdicts, G-STEER, identity) | `/mnt/v/output/zensim/byconstruction-2026-10-22/gates/` | 55 MB |
| Gate outputs for full-data M | `/mnt/v/output/zensim/byconstruction-2026-10-22/gates-M/` | |
| Feature probe (global edits vs visible damage, dumped feature vectors) | `/mnt/v/output/zensim/byconstruction-2026-10-22/feature-probe/` | |
| LODO cells (train.log, bakes, held-out predictions, `byc_record.json`) | `/mnt/v/output/zensim/byconstruction-2026-10-22/cells/<arm>/without_<fold>_s<seed>/` (working copy `~/tmp/byc/cells/`) | 64 MB total |
| Best-dev-epoch predictions (`predict-best`) | `/mnt/v/output/zensim/byconstruction-2026-10-22/best/` | 28 MB |
| Full-data T and M fits with packed bakes | `/mnt/v/output/zensim/byconstruction-2026-10-22/full/` | |
| Pinned trainer/predictor builds, with `SHA256SUMS` and `BUILD_COMMIT` | `/var/tmp/byconstruction-bin/<commit>/` | |

The mixture images are browsable at `http://dev.lan:3300/zensim/byconstruction-2026-10-22/mixture/img/`.

sha256 (dev, 2026-10-22):

```
2ba1dd7f6b2ac20e753ae21dc1c35ce444075ca6a4a632334cfde45eef43ecac  mixture/manifest.json
29f3e82b32e010885e310ea8c70b6ecc43c8fd6d35d68597bf964216dc722ee4  mixture/scores_B.json
21ebe54bd4a42cda527450574a2d4a1ed8e3c09c1ff99f1b595b99f463922b2b  gates/steer135/tfull/result.json
e785bb6d4b794ab6bcddbdcc057b7d6ea810e7bc42a7256a21df601fd71e9d0d  gates/verdict-s0-tfull/standard/gaddr.json
81fc8fdfe9de79ad98168be9534a670b9b220c254cbf93a0cc46dbc3195fefaf  gates/verdict-s0-tfull/ladder/gaddr.json
50f8aef948e0f2af7ecb3b1dead821e1f3eedc4704f7af8b5b190dc947dedc44  gates/nearid-s0/candidate-tfull.GATES.json
```

Inputs it reads (unchanged, not copied): E34 evaluation `panel-input.json` and stored `base-c`
predictions/dense bakes (`/mnt/v/output/zensim/e34-launch-2026-10-15/evaluation/`), the v40r4
`v2e29` root (`/mnt/v/output/zensim/v40r4-2026-10-08/v2e29/`), E34's reference inventory
(`/mnt/v/output/zensim/e34-prep-2026-10-15/part5/p4-inventory-r2/references/`).

`~/tmp` and `/var/tmp` copies are working state; the record cites hashes from `byc_record.json`
for anything a result depends on.
