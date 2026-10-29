# HDR chroma-only teacher generation

Status: full generation, media/rung admission and independent decoding pass. Final archive receipt is being sealed. No feature bank or model.

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

Full generation/admission:456 views,16480 native AVIF/PQ16 cells,16480 unique
AVIF hashes. The1824 chroma ladders admit1461, retaining10422 chroma rungs.
By fixed luma quality: q95434/456 (3467 kept), q85398/456 (2774), q75301/456
(1933), q60328/456 (2248). All4 luma controls admit, retaining77 diagnostic
rungs. Keep control-only cells outside the chroma training leg; no feature
extraction or fitting is part of this record.

The frozen eight-view352-cell comparison admits28/32 natural chroma ladders
and0/32 HDR-VDP-3 ladders; both admit all4 luma controls. E26's stored
9.579838179794153 control replays exactly. Its native ingress dispatches by
filename extension, so comparison copies carry `.pq.png` names and are
hash-identical to their CAS objects. The frozen executable is unchanged.

Every one of the16480 native AVIFs independently decodes with libdav1d and
passes full header checks:10-bit PQ BT.2020 full-range420, matrix6, fixed
luma qindices per quality and every requested shared U/V delta.
All target doubles survive Parquet readback exactly. The raw ledger audit
finds507 DONE records,456 unique jobs,51 extra completions over41 repeated
jobs (33 cross-host),zero conflicting outputs andzero failures. Repeated
executions are retained in the raw archive.

The observed261.473512092-second production window measures unique cells/s:
i27015.144937505588198, i26513.07971875482617,
r35001.927537500711225, r3800x6.471018752387684. These are first-wins
ledger observations under concurrent fleet load, not isolated CPU timings.
The complete result/log/command headers remain in PRODUCTION_RATES.json
and the run logs. Admission run-heavy: rc0,511s,peak-RSS0.03GiB,
min-avail22070MiB,peak-load15.47 (Docker client only; validation has an
explicit2GiB container cap). Independent native oracle: rc0,240s,
peak-RSS0.14GiB,min-avail24767MiB,peak-load21.60.

All34781 immutable CAS objects/9341015837 bytes were copied from the mirror
to the primary worker and read back by SHA256; the copy receipt is
CAS_MIRROR.json. Copy run-heavy: rc0,162s,peak-RSS0.21GiB,
min-avail12138MiB,peak-load15.54. Full input mirror:536 files/2431917988
bytes. CAS and per-view receipts retain the exact PQ16 renditions. Encoded
CICP/depth fields describe10-bit matrix6 AVIF; rendition PNGs are16-bit
RGB with CICP9/16/0/full-range.
