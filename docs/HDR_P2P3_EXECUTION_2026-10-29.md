# HDR paths P2/P3: execution registration (lane fhdr)

Registered 2026-10-29 after the native feature banks below were published and verified and **before any HDR-leg fit,
dev-fold score or checkpoint selection**. It adds to, and weakens nothing in,
[HDR_TRAINING_PATHS_2026-10-29.md](HDR_TRAINING_PATHS_2026-10-29.md).

## Native extractor and banks (all revision 7, `basic+peaks+v2@w1825/rev7_rowlanes#36c3f3af`, 576 measured ids)

- `hdr944_extract` rebuilt in `rust:1-bookworm` from zensimx jj commit `a86520ca68dc` (WIP patch of `c1c386c8`'s lane, applied on
  main `cd69b9e0`); SHA-256 `63bed9a64ed265d2e8b8bad4aabaae5d952215afda935ec911a0c82d1c204aa7`. Qualification: Rev5 mode reproduces the
  stored E26 HDR val bank bit for bit (24 rows, 10,080 measured values, 0 mismatches); Rev7 mode equals the glibc-2.43 build bit for bit on
  6 ladder cells; a published ladder-bank row equals a fresh re-extraction (3,456 values, 0 mismatches); 3 example tests pass.
- Worker: zenmetrics 5a99a89e image (`hdr:` publication fix); every shard ledger row `done`; every CAS object hashes to its name; every
  sidecar named by a result exists (`scripts/byconstruction/hdr_native_bank.py` refuses otherwise).

| Bank | Rows | Producer | `_MANIFEST.json` SHA-256 prefix |
|---|---|---|---|
| UPIQ-380 (stored-byte recovery, native `upiq_pu_score` rows) | 380 (330 fit / 50 development) | `d86b3b44` / `73d1374f…` | 4755536aa30099f4 |
| HDR chroma ladders, all kept rungs (consumer TRAIN 7,177 + EVAL 3,245) | 10,422 | `a86520ca` / `63bed9a6…` | 37713a36113021f9 |
| 4Kdtb (Bike TRAIN 48, Regatta EVAL 48) | 96 | same | c3b97fdaed77414f |
| HDRTEACH train (agree rows) | 7,390 | same | e16ea551f20753b5 |
| HDRTEACH val (report-only) | 3,900 | same | c23e9415e7730b19 |

4Kdtb source screen (limits stated): byte-identical references vs every library inventory (no match, prior record); in addition a
24x24 log-luminance thumbnail correlation of the eight 4Kdtb references against the eight HDdtb source images found a maximum of 0.70
(Bike vs Regatta 0.24); scene names are disjoint. This is not a perceptual screen across all collections; no clearance beyond
that is claimed.

## Legs (TRAIN rows only; EVAL, VAL, UPIQ development rows and the 50 UPIQ development labels never enter a root)

| Leg | Rows / groups | Group | Target | Mode | Base mass |
|---|---|---|---|---|---|
| `hdr_upiq380` | 330 | UPIQ reference scene | 100+10·JOD | withinref,rank | 0.5 |
| `hdr_4kdtb_train` | 48 | 4Kdtb source sequence frame | MOS | withinref,rank | 0.5 |
| `hdr_ladder_chroma` | 7,177 | view × luma quality ladder | natural SSIMULACRA2 | withinref,both | 1.0 |
| `hdr_teacher_hdrvdp3` | 7,390 | reference image | 10·q_jod (HDR-VDP-3) | withinref,both | 0.5 |

`hdr_teacher_hdrvdp3` and any UPIQ evidence share HDR-VDP-3 lineage (calibrated on UPIQ): UPIQ is not independent for models trained with it.
Admission schema `hdr-native-leg-v1` (Python `admit_hdr_leg`, Rust `hdr_admission.rs`) checks producer/extractor pins, the registered row count,
ordered keys, family reservation and the nominal-weight correction; it reads no target table.

## Paths

- **P2** = Mlic root (frozen `a7fd4af0…`, fx1 inputs) + the four legs. **P3** = the fchroma fx2+chroma root (frozen `0f2f8923…`) + the same
  four legs under fx2. Each: E21 cells (4 folds × seeds 0-2), three full seeds, then the E recipe's general continuation
  (x256 + rehearsal, e59). The codec counterpart (AIC-only e29) only if the general path completes.
- **Scale grid** s ∈ {0.5, 1.0} multiplies every base mass. Selection is TRAIN-side only: (1) reject any s whose E21 cells fail the
  inherited SDR no-regression rule (paired mean ΔSROCC ≥ -0.002, each source population ≥ -0.005, W2 > -2 SE, CID22-A bias-spread +0.005);
  (2) among eligible, maximise the unweighted mean over legs of the within-reference Spearman on held-out HDR TRAIN families, over two
  source folds `int(sha256("hdr-p2p3-dev-fold-v1:"+family),16)%2` (a leg with fewer than two held-out families is report-only for that fold);
  (3) ties go to the lower s. EVAL, VAL, T0 UPIQ, the video sets and the 50 UPIQ development rows never select.
- Reproduction control first (a stored Mlic cell with the trainer binary used here; the HDR admission is an additive code path).

## Not claimed

No HDR-eval number exists at registration. HDR-VDC/AVT stored Rev7 features do not exist: those rows stay INCOMPLETE.
