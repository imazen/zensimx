# Chroma-only JPEG pilot — Stage A

Registration: [by-construction record](byconstruction_2026-10-22.md#chroma-only-jpeg-ladders-with-an-ssimulacra2-teacher-registration-2026-10-28-owner-request-before-any-encode).

Observed on dev and i270: `/var/tmp/byc-chroma-20261028/pilot-run04/`.
Tower mirror: `/mnt/tower/output/zensim/byc-chroma-20261028/`.
The local coordinator report is `~/tmp/handoff/chroma-ladders/report.md`.
Use `_MANIFEST.json`, `SOURCE_PROVENANCE.json`, `SOURCE_REPLAY.json`,
`CONTROL.json`, `ORACLE.json`, `SUMMARY.json` and `LADDERS.json` for exact
pins and coverage. `LADDERS.json` retains all rung scores, stop reasons and
content-addressed JPEG hashes; generated tables and images stay outside git.

- Generation owner: `6c9ce8b89f247a2e44c6c72cb3b33e19c2b4bea4`.
- Teacher binary: SHA256 `7c7ffbbfa033e8ca1a8f103d472b61ccde061c2394b03d519af2852ee8eeda87`;
  the preserved SafeSyn baseline-recovery executable, exact four-cell replay.
- Sweep binary: SHA256 `37c3bc112d8bdc86b949005f8fcafd419510b88b7dd7c7a8938ba59251c4e50f`;
  `ghcr.io/imazen/zenfleet-worker:exec-cvvdp-0a61830d`.
- Preparation binary: SHA256 `cb23096d02f5696d9517442ab3d8ce6dcafa6868390dd0f2fc50dcbd1de65947`;
  existing imazen Mitchell fixture owner.
- 81 TRAIN sources, 162 views, 12,636 encoded/scored cells; maximum raster
  512×512. The complete 21-class imazen-26 pilot has only one texture source.
- All-ones Y: 140/162 ladders admitted for each YCbCr subsampling family.
  XYB: 80/162 admitted; median endpoint score loss 7.929322 over scales 0.1–5.0.
- All 12,636 JPEGs decode with the independent oracle, all CAS hashes match,
  and independent PNG RGB bytes match the frozen teacher's inputs exactly.

Incidental Rev3 feature CSVs emitted by the old teacher are diagnostics only.
They are not a training leg and must not be mixed with the pinned Rev7 banks.
Stage B generation, diagnostic teachers, Rev7 extraction and fitting have not
started. The full-grid proposal requires coordinator review and zenfleet.
