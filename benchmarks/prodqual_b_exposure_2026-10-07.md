# PRODQUAL-B exposure ledger proposal — 2026-10-07

Frozen composition: production D1 seed 0 only, packed SHA-256
`f803b74c4252952f337abdc0234c2930839d45dddc32ae9b8b5296d6c840f400`,
base `ddf375af`. No tune, refit, recalibration, threshold change or ensemble.
The preread registrations in this directory precede numerical reads. This
proposal records this lane's actual exposure; it does not change admission policy.

No KADID TERMINAL, sealed, T0 or AIC-family payload was read, hashed or enumerated.
No human opinion label was opened. Model bytes, manifests, registered instrument
metadata, source code and previous gate reports are distinguished from new
numerical instrument and pixel reads below. Fields named `human_score` in these
registered engineering tables are ordinal keys, not opinion labels.

| Read population | Purpose and exposure | Binding/evidence |
|---|---|---|
| Explicit Rev5 standard grid, 4,424 rows | Canonical bake_verdict dial and addressability; already exposed engineering instrument | Main preread registration; `standard/COMMAND.json`, `gaddr.json`, `fulleval.json` |
| Explicit Rev5 ladder grid, 9,593 rows | Same owner, denser engineering diagnostic; no human ranking | Main preread registration; `ladder/COMMAND.json`, `gaddr.json`, `fulleval.json` |
| Explicit negative-tail probe, 2,000 rows | Read with both grids; synthetic distortion tail, no five-codec truth | Registration; both gaddr reports |
| Explicit feature identity probe, 38 rows | Read with both grids, retained separate from actual native pixel identity | Registration; C5/C6 owner receipts |
| broad/owner/jpeg8/m3 keys | Admission geometry, source role and file/pixel hash metadata; read for seed0 and incumbent attempts | Five initial key reads in `EXPOSURE_READS.json`, complete proposal in `EXPOSURE_FINAL.json` |
| 59 delivered pairs, 118 paths | 135 seed0 steering cases; identical inputs attempted with B. TRAIN, already exposed KADID SELECT design, or historical M3 diagnostic fixtures only | `STEERING_PREREAD{,_B}.json`; 135 seed0 maps and 135 B refusal logs |
| Identity keys and 38 source pixels | Full registered native identity through seed0 Rev5 and serving B Rev1, isolated processes | `pixel-identity/PREREAD.json`, `pairs.tsv`, `RESULT.json` |
| CHROMAQ 32 CLIC tuning references and 1,760 registered distortion cells | Native frozen seed0 scoring; unchanged owner analysis also reads already stored serving B and independent objective metric columns from the two sweep TSVs | CHROMAQ preread registration, exact four TSV paths, `chromaq/PIXEL_INPUT_HASHES.json`, `summary.json` |
| STEERCODEC eight CLIC crops and 72 existing native codec decodes | Forty-eight registered quality swaps, quarter-block selection; three codecs, q30→60 and q60→85; no human labels | STEERCODEC preread registration and `steercodec/INPUT_HASHES.json` |
| STEERCODEC own generated composites | Native zenpng writes; independent reader checks every output pixel against selected low/high rectangles; frozen seed0, serving B and independent objective judges score the same resulting pixels | `COMPOSE.json`, 192 PNGs, 264 pairs, judge commands and score outputs |
| Seed0 and B model bytes; SHIPPATH6 v4 head | Hash verification, scorer input; compatibility-only head probe. Companion not silently attached to the frozen single-model evaluation | `PROGRAM_MODEL_PINS.json`, `companion-probe.log` |
| RD guard synthetic fixtures | Wrong/malformed/matching pin routing tests only, no real intervention measurement packet | `rd-guard-tests-final.log` |
| SPEEDQ reports | Existing report metadata only; no new timing or stress population read | Archived base report and DONE marker; timing weights differ from seed0 |

The complete external manifest retains exact paths, input hashes, roles and
repeated owner invocation counts. Table payload reads are the four explicit
standard/ladder/negtail/identity tables only; the remaining instrument parquets
were metadata-hashed before use, while their key tables and delivered pixels
provided native steering inputs. No numerical rank board was evaluated with an
empty corpus set: any zero-valued aggregate placeholder in raw instrument-only
fulleval output is not a ranking measurement.

Human G-RANK remains blocked under DATA_SPLITS. D1 TRAIN/SELECT design permission
does not turn design scores into an independent release qualification or admit
CID22-49/NITS/LIVE/CSIQ/MCIQA T0 payloads. AIC holdout-family rows stay excluded.
Five-codec resolvable floors need registered new-grid receipts and per-cell mentor
truth. RD needs a registered four-codec frozen-model intervention packet; G-TARGET
needs matching composition/configuration/TRAIN calibration and bound witnesses.
Integrity class/source strata require a reviewed admission and rule before any
numerical corruption read. No human HDR accuracy evidence was opened.

Canonical evidence: `/mnt/v/output/zensim/prodqual-b-2026-10-07/`.
Mirror: `/mnt/tower/output/zensim/prodqual-b-2026-10-07/`.
