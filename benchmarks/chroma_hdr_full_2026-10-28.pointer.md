# HDR chroma-only teacher generation

Status: predeclaration qualification passed; full generation/admission pending. No feature bank or model.

The committed PNG-v3 corpus tree `b1faae550eba4d29efee1de1a32954ef11a0e1ed`
selects 38 TRAIN HDR origins from 76 total, across four photographic classes.
Source LFS hashes and original provenance links are retained in SOURCE_CENSUS.json.
Working-copy metadata is excluded from selection.

Native PQ16 references use Mitchell linear-light downscales at long edges
32, 64, 128, 192, 256, 384, 512, 640, 768 and 960, plus native central crops
256 and 512 where supported. No upscale; every view stays below 1,000,000 pixels.
Tiny views remain a minority. The pinned zenavif SVT encoder emits 10-bit
PQ BT.2020 4:2:0, speed 6, one thread, at luma qualities 95, 85, 75 and 60.
Shared U/V qindex deltas are 0, 8, 16, 24, 32, 40, 48, 56 and 63, matching
HDR-A3. The adapter signals full range and matrix 6; this is explicit BT.601
matrix coding with BT.2020 primaries, not BT.2020-NCL coding.

Target: natural SSIMULACRA2, standard cube-root path, fast-ssim2 0.8.2 source
`b7c2b4b33f3a26b753caabd5264a7fef9c4f4545`, zenmetrics helper owner
`1caf1848e2aee04da6354fd8fc94646c428d3bbf`. Native PQ decodes to unclipped
absolute linear BT.709, then nits/100. No PU21 or tone mapping. All 208 stored
HDR-A3 doubles replay exactly with the original helper executable
`4e34a176419a4bf38c2ba209a42ddc3e1bf2fcd06a38a42a50e0f485f1689e55`.
The reference-reuse build must replay every double and native bitstream before
production declaration.

Admission reuses `chroma_pilot.admission`: stop at the first adjacent tie or
increase, keep steps at least 1.0 below the last kept rung, reject fewer than
three kept rungs. Negative scores remain valid. No thresholds change.

Comparison is fixed before declaration: first TRAIN origin per class at
Mitchell edges 256 and 512. Only the 512 view also carries the 20-point luma
control (q95 through q5 step5, plus q1, delta0). These views receive E26's
HDR-VDP-3 comparison: `10*q_jod`, common native BT.709 ingress, 60 ppd,
quality task, rgb-bt709, LED-LCD-sRGB, age24, no surround. This diagnostic
never selects target rungs. Its frozen executable SHA256 is
`1278c6935ad4ebbb624e6213c56ad672602b05433b288ea1f9274dbf02dee721`.

The executor is `zenmetrics/scripts/jobsys/hdr_chroma_exec.py`.
`chroma_hdr_full.py` owns committed-source admission and scientific readback;
`zenfleet-ctl declare-views` and existing workers own declarations, claims,
retries, CAS and ledger transport. One job renders a whole view once, scores
with one bounded-reference teacher process, and persists every AVIF plus its
exact PQ16 PNG rendition before DONE. Scores are hash-keyed float64 Parquet.
Frozen natural SSIMULACRA2 exposes no per-pixel map; none is approximated.

Worker data root: `/var/tmp/byc-chroma-20261028/hdr-full/`.
Mirror: `/mnt/tower/output/zensim/byc-chroma-20261028/hdr-full/`.
Full run/host/hash and resource receipts are kept with the data and in the
operator report. Final source/geometry, per-host rates, admission by quality
and class, comparison and mirror qualification remain incomplete here.

Qualification receipts: 456 prepared views (76 tiny,76 native crops,zero
duplicates,max746880 pixels); all536 input mirror files/2431917988 bytes
read back by SHA256. A real52-cell job matches A3 and publishes every native
AVIF/PQ16 object before DONE. The16-view640-cell sample covers all four
HDR classes at64/256/512/960. All sixteen primary receipt hashes match
across i270/i265/r3500/r3800x; the sample's entire media graph and float64
Parquet targets pass admission readback. An independent FFmpeg/libdav1d
whole-view decode/header check passes36 cells, including fixed luma/base
quantizers and shared U/V deltas. Full coverage remains a separate gate.

Serial sample measurements (cells divided by measured Docker-worker elapsed):
i270640/164.7548229941167s; i265640/172.32136089692358s;
r3500640/370.7443624588195s; r3800x640/273.35085832607s. These measurements
include startup, encoding, scoring and CAS; they are not production throughput
or an uncontended CPU comparison. Full logs and resource receipts remain with
the data. The16 completed sample jobs are reused by the production declaration.

Verified image:
`ghcr.io/imazen/zenfleet-worker@sha256:17c02021ca80c667aacd12d2ced8873657f387f5dea0a8f6f2e9139ae6f6ccfd`.
Renderer `7e0a8236d974e86ade3a0c21af47da70a54011d9c7c547522fa96cf4913b58d5`;
reference-reuse helper `f8e5c1c7e74ebebe1884df51b27c6827fa70c0613c5fe2cf99102da4fccb0ad1`;
executor `9ad4012c958530f4042f0d0fae72059878384fcdb75e110ffef8d39a7561e3ac`.
These are zenmetrics commits5b4f5373,2d2a3194 andc375a124 on master.
