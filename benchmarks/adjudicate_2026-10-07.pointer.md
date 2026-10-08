# Frozen seed0 adjudication evidence

Source / build review: `d8d426be132a31f581ea85d77174bb4f59e3cd4c`;
base `dcf7e3552b06e1bc0b36520f945d77d62d60295b`. Local-only lane;
no push and no shared-main movement. See [measured report](adjudicate_2026-10-07.md).

Local: `/mnt/v/output/zensim/adjudicate-2026-10-07/`.
Verified mirror: `/mnt/tower/output/zensim/adjudicate-2026-10-07/` (observed 2026-10-08T10:24:34.556445+00:00).
Resolve the mount with `findmnt -T /mnt/tower/output/zensim -o TARGET,SOURCE,FSTYPE`.
R2 mirror absent. Manifest build_commit binds the committed review source;
executable SHA256s are in FINAL_PROVENANCE.json, not inferred from filenames.

- `_MANIFEST.json`: `e0e217215437bea492d90810fdee8a0381b358d8adaf0d307cb300f0d4e43e6f`.
- `FINAL_VERIFICATION.json`: `eab9561d42b5bc530a3c713af110737c705f3963593cbdfee5cd1802e8801bc7`.
- `MIRROR_VERIFICATION.json`: `1e748df0a2070c3dd46123c1219e66329887cc213db931ef7112a2e1d2ae6c73`.
- `CARGO_CLEANUP.json`: `6bfadfb0a7e9958ce377b7f1c72ad4519ff5b906433256563dc13fd536279c27`.

All 374 archived inventory files (364505364 bytes)
SHA256-match the mirror. Three randomly selected files independently match:

- `replay/B-refusal-broad-48.log`: `10880e54fcdfabc64485e257a30f0262abc281612f42dd831ccebdb6d7b063c4`.

- `clippy-serve.log`: `4726a63c632d3a436880d67010bb175af7ed3e5c445f1faa560ba520db346d04`.

- `replay/B-refusal-m3-21.log`: `105b3a27ed60dd08404f7c1996d025daa36a1fcbba7686ea4aa5b890af8dccd7`.

Retained: pinned models, canonical/native audit programs and historical versions,
exact engineering inputs/pixels and their original-to-archive path map, raw scores,
per-rung classifications, steering maps, original B refusals, source/patches,
full logs and program/input/model provenance. No label or calibration payloads.
Two receipt files are explicitly outside the immutable inventory; they also
byte-match the mirror. Only this lane's inactive Cargo target was removed:
`/home/lilith/tmp/adjudicate/target`, 1702256640 allocated bytes reclaimed.
No other target or generated artifact was deleted. Unlanded workspace retained
for coordinator review; cleanup after landing remains coordinator-owned.

Coverage control receipt: `rc=0 22s | peak-RSS 0.06GiB | min-avail 37779MiB | peak-load 0.76`.
Final verification: `rc=0 1s | peak-RSS 0.67GiB | min-avail 37701MiB | peak-load 2.84`.
Mirror: `rc=0 5s | peak-RSS 0.02GiB | min-avail 38408MiB | peak-load 2.46`.
These are run-heavy process receipts, not scorer speed or per-size memory qualification.
