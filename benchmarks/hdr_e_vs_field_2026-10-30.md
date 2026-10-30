# E against CVVDP, butteraugli and HDR-VDP-2 on HDR (2026-10-30)

Question (owner, 2026-10-30): how does E do on HDR against butteraugli and CVVDP?
E is the finalized revision-7 general profile (bake SHA-256 `7642eff4c965d9b7…`) scored through the HDR front end
(PU21, 100 cd/m2 at 1.0) from the native revision-7 feature banks of the HDR lane (`/var/tmp/byc-hdr-20261029/banks`,
dev). E never trained on UPIQ or on HDR-VDP-3 labels, so every row below is held out for E. No fit, no selection.

Reference metrics are the stored per-pair predictions in `/mnt/v/output/zenmetrics/upiq-pu/panel_*.tsv` (dev; the
June 2026 UPIQ runs cited in `bhdr_improvement_split_lineage_2026-07-12.md` §8.1, which these numbers reproduce),
joined by condition id; the join is asserted against the stored JOD for every metric. HDRTEACH CVVDP is the raw
`cvvdp_jod` column of `hdrteach_val.parquet`, not the historic mixed label.

Commands: `scripts/byconstruction/hdr_panel_upiq_all.py --models E=<bake>:fx1 ECodec=<bake>:fx1 --byc-predict
/var/tmp/zensim-bin-10d087cd/byc_predict --out <dir>` (with `PYTHONPATH=/var/tmp/byc-chroma-20261028/a2/cluster-libs`),
then `scripts/byconstruction/hdr_e_vs_field.py <dir>`. Check: E on the 50 UPIQ development rows gives 0.8387, the
HDR lane's recorded value.

## UPIQ HDR (380 pairs, 30 references; target JOD)

| metric | pooled | Narwaria | Korshunov | within-reference |
|---|---:|---:|---:|---:|
| E | 0.792 | 0.900 | 0.947 | 0.958 |
| ECodec | 0.636 | 0.501 | 0.941 | 0.939 |
| CVVDP 10k nits | 0.831 | 0.781 | 0.969 | 0.958 |
| CVVDP default | 0.758 | 0.664 | 0.965 | 0.954 |
| butteraugli p3 | 0.628 | 0.354 | 0.940 | 0.908 |
| butteraugli max | 0.630 | 0.400 | 0.917 | 0.887 |
| HDR-VDP-2 | 0.812 | 0.886 | 0.948 | 0.957 |
| PU-PieAPP | 0.875 | 0.778 | 0.925 | 0.922 |

Paired bootstrap over the 30 references (2,000 draws), E minus metric:

| vs | population | mean | 95% CI |
|---|---|---:|---|
| CVVDP 10k nits | pooled | -0.037 | [-0.089, +0.028] |
| CVVDP 10k nits | narwaria | +0.110 | [-0.007, +0.274] |
| CVVDP 10k nits | korshunov | -0.022 | [-0.041, -0.008] |
| butteraugli p3 | pooled | +0.160 | [+0.008, +0.329] |
| butteraugli p3 | narwaria | +0.497 | [+0.156, +0.744] |
| butteraugli p3 | korshunov | +0.007 | [-0.010, +0.026] |
| HDR-VDP-2 | pooled | -0.019 | [-0.052, +0.010] |
| HDR-VDP-2 | narwaria | +0.014 | [-0.015, +0.045] |
| HDR-VDP-2 | korshunov | -0.002 | [-0.014, +0.009] |

HDRTEACH VAL (3,900 rows, 300 references; target HDR-VDP-3 Q_JOD):

| metric | pooled | within-reference | inversions / pairs |
|---|---:|---:|---:|
| E | 0.941 | 0.9995 | 28 / 23400 |
| CVVDP | 0.821 | 0.9821 | 602 / 23400 |

## Reading

- UPIQ pooled SROCC mostly measures how two studies' JOD scales line up (§8.1 of the July record); the per-study
  columns are the fairer read.
- Against CVVDP (10k nits): E is better on Narwaria (+0.110, CI just touches zero), slightly worse on Korshunov
  (-0.022, CI excludes zero), and not separable pooled. Within-reference ties (0.958 each).
- Against butteraugli: E is clearly better pooled (+0.160) and on Narwaria (+0.497); tied on Korshunov.
- Against HDR-VDP-2: indistinguishable on every population.
- ECodec is weaker on HDR than E (Narwaria 0.501).
- HDRTEACH: E tracks HDR-VDP-3 far more closely than CVVDP does (28 vs 602 within-reference inversions).
  This is agreement with another model, not with people.

## Not measured

Butteraugli on HDRTEACH; CVVDP and butteraugli on 4Kdtb (no stored scores; CVVDP here is the GPU port and dev has no
GPU); HDR-VDC/AVT (no stored revision-7 features).
