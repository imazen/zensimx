# Final confirmatory read of the shipped model A (zensim 0.3.0) on the sealed sets — registration, 2026-11-05

Owner, 2026-11-05 (verbatim): "score on the final sets". Nothing in this read selects, tunes or
changes any model; failure or success is final and is reported as measured. Written before any label is opened by this assessment; historical exposures are disclosed below. To be committed to the zensimx record (`benchmarks/final_read_a_2026-11-05.md`)
and appended to `docs/DATA_SPLITS.md` exposure ledger before the lane reads a label.

## Model under read

* Public `zensim` 0.3.0 line, profile `A` (`zensim-a`), bake
  `weights/rev7_mix_h128_2026-11-03.bin`, SHA-256
  `d5e2a56db35c32620d9f9d9c19ba3bc78bd8fea2b8cd933cde8447428d85e62a` (research `F`, recipe
  `mixF3`, seed 0, continuation epoch 59). Frozen at imazen/zensim PR #2 head `bebc6ea4`
  (profile-f `70469822` adds only the parallel cutoff, scores unchanged).
* Scoring surface: the research crate at a pinned commit whose `split_parity_golden` is byte-identical
  to the public crate's `split_parity_check` (1,386 rows, 2026-11-04). The lane records the commit,
  the binary SHA-256 and the parity receipt. Serial extraction, `ZENSIM_FORMULA_REV=7`.
* Comparators, reported beside A, never selecting: SSIMULACRA2 and butteraugli 3-norm (the R10
  amendment 5 reference-metric tooling, `scripts/byconstruction/e36_exp56/refmetrics_run.sh`);
  for KADID TERMINAL additionally research profile `B` as its 2026-10-07 registration requires.

## Sets (approved frozen-A assessment; prior exposure disclosed per population)

| Population | Rows / references | Label | Pixels | Statistic |
|---|---|---|---|---|
| CID22-B | 24 references, their validation pairs | MCOS (quality) | library `cid22` PNG/JPEG decodes | signed SROCC/KROCC/PLCC(4PL), RMSE on the collection scale, within-reference SROCC Prior zensim read: October 4 R7/R7a and October 18 candidate assessment (2,100 rows); `docs/DATA_SPLITS.md:1613–1617,2761–2791`. |
| CSIQ | 866 distorted / 30 references | DMOS (distortion; sign disclosed, reported as quality = 1−DMOS) | library `csiq` | same panel Prior zensim read: October 4 R7/R7a and October 18 candidate assessment (865 rows); `docs/DATA_SPLITS.md:1613–1617,2761–2791`. This registration retains the approved 866-row population. |
| LIVE release 2 (open external since 2026-10-03) | 779 / 29 references; also the 10 content-disjoint references (267 pairs) fixed by the 2026-10-03 audit | `1 − dmos_new/100` | library `live-iqa` | same panel, both populations Prior external zensim reads since October 3; October 18 clean10 assessment (267 rows); `docs/DATA_SPLITS.md:2761–2791` and `scripts/rev4_featpot/external_sets.py`. |
| KonJND JPEG TERMINAL | the registered TERMINAL split only | PJND (threshold) | library `konjnd-1k` JPEG, decoded by zenjpeg main `762fa712` | pooled signed SROCC vs PJND with reference bootstrap; within-reference SROCC and interval undefined (100 pairs / 100 references), as approved by coordinator. Prior zensim read: October 4 R7/R7a, SELECT and TERMINAL labels; `docs/DATA_SPLITS.md:1613–1617`. |
| KADID TERMINAL | 2,000 pairs | MOS | local `kadid10k` | the 2026-10-07 registration exactly: signed SROCC, KROCC, PLCC, within-reference SROCC, per-distortion-type SROCC, `zenstats::scatter` geometry; comparators the shipped research profile B (`codec_target()` of the research crate at the pinned commit, bake SHA recorded) and the Rev4 research by_v2fy bake (resolved from `benchmarks/release_gate_map_2026-10-07.md` / `releasegate5_admission_2026-10-07.md`, SHA recorded); confirmation rule: A's signed SROCC not worse than B's by more than 2 SE (paired bootstrap over references, 10,000 draws, seed fixed in the pin) and not worse than by_v2fy's by more than 0.005; failing either is a failed confirmation, recorded as such. If the by_v2fy bake cannot be located, the read still happens, the B rule is applied, and the missing comparator is recorded as a deviation, never silently dropped No prior zensim model scored this population; third-party-metric/reliability reads October 24–26 are disclosed at `docs/DATA_SPLITS.md:2892–2924`. |
| AIC-4 sample | 300 pairs / 5 references; SDR25 = its 50-row JPEG-AI subset, reported as a subset, not independently | `q_jnd` (distortion-oriented; reported signed, sign disclosed) | library `jpeg-aic4-sample` | signed SROCC, within-reference SROCC Prior zensim read: October 4 R7/R7a; `docs/DATA_SPLITS.md:1613–1617`. Prior JND-unit study: `docs/DATA_SPLITS.md:2886–2889`. SDR25 inherits this exposure as its subset. |
| MCL-JCI | all rated conditions | JND-based quality | library `mcl-jci` | signed SROCC, within-reference SROCC. Prior exposure, disclosed: zenpapers read every subject's JND chain and 243 library score rows on 2026-10-24/25 for JND-unit and observer studies with third-party metrics only, without a preregistration (ledger, "Exposure ledger — 2026-10-24/25"); no zensim model was scored, fitted or selected on them. D3 made its images training sources (KonFiG-IQA crops ten of them); its ratings never trained any zensim model Prior zensim read: October 4 R7/R7a, including the 40-source subset excluding KonFiG crops; `docs/DATA_SPLITS.md:1601–1617`. Additional October 24/25 exposure: `docs/DATA_SPLITS.md:2867–2899`. |
| CLIC 2022 perceptual (TERMINAL) | 57,300 2AFC questions / 57,243 reference crops; every question retained | human preference | library `clic-2022-perceptual` | agreement with the majority choice, as the CLIC 2021 read |

Excluded and why: AIC-2026 `public_test` (its one read is spent, 2026-10-23; AIC-2026 numbers never
leave private files); CID22-A25, AIC-3, KADID TRAIN/SELECT, TID2013, KonFiG (training sources);
EVAL's nine collections (the mix was chosen on them; already reported); the threshold sets
JND-MID-MA, VLIC and Konstanz PJND (different instrument, no registered statistic).

## Reading rule

* One read per population, ledgered in `LABEL_READS.json` before any label column is parsed
  (`mos_confirm.py confirm` pattern); the DATA_SPLITS exposure ledger is appended under `flock`
  before the first label open. A failed or partial read is still spent and is reported as such.
* Resampling unit: references (whole-reference bootstrap, 10,000 draws, fixed seed recorded).
  Every statistic is reported with its interval; no threshold decides anything except KADID
  TERMINAL's registered rule, which is reported pass/fail.
* Orientation: `--signed-quality`; distortion-oriented targets (CSIQ DMOS, AIC-4 `q_jnd`) are
  reported with the sign and the orientation declared, never flipped silently.
* Disclosures carried into every report of these numbers: LIVE's 19 Kodak references contain a
  TID2013 reference (training source); MCL-JCI shares sources with KonFiG-IQA (training); SDR25 is
  inside AIC-4; CID22-B is the second half of a reference set whose first half trained A; CLIC 2022
  shares CLIC's protocol with the CLIC 2021 confirmation read.
* Prior label exposure, disclosed rather than denied: the exposure ledger records that MCL-JCI and
  Konstanz PJND labels (2026-10-24/25, zenpapers JND-unit and observer studies) and KADID TERMINAL
  labels (2026-10-24 to 26, reliability and metric-fit analyses) were read by third-party-metric
  studies before this registration, with no zensim candidate scored, fitted or selected on them, and
  that no preregistration preceded those reads. Earlier zensim reads are also on the ledger and
  are disclosed here: the R7/R7a set comparison of 2026-10-04 scored four frozen models on CID22-B,
  the AIC-4 sample, CSIQ, MCL-JCI and KonJND JPEG (SELECT and TERMINAL), and the 2026-10-18
  candidate reads covered CID22-B (2,100 rows), CSIQ (865) and LIVE's 10 content-disjoint references
  (267 rows). This read is therefore a frozen-A assessment on populations that earlier frozen
  models were already assessed on, with the one exception of KADID TERMINAL, which no zensim model
  has been scored on; the lane records the exact ledger lines beside each population.
* Publication: results go into the zensimx record and the public `zensim` write-up
  (`docs/`), with the disclosures above. Nothing about AIC-2026 appears in public files.

## Pixels and provenance

Library content read in place from `/mnt/tower/input/iqa/datasets/<id>/` (dev; local copies under
`/mnt/v/dataset/` only where DATA_SPLITS already names them). PNG/BMP decodes through the research
crate's existing decode path (zenbitmaps/zenpng); JPEG stimuli through zenjpeg main `762fa712`
(the pin that reproduced the frozen EVAL table bit for bit, 2026-11-05). Every decode map records
`rgb8_sha256`; feature banks record the extractor SHA; no third-party decoder.

## Execution pin and pre-read status

- Research source pinned to `bd384ffb408833231a3a91158b879c401340836a` (the parent of the registration workspace).
- Public parity source: imazen/zensim `70469822`, example `zensim/examples/split_parity_check.rs`.
- Bootstrap: 10,000 whole-reference draws, seed `20261105`; percentile 95% intervals. KADID comparisons use paired draws.
- Registration is committed before any label column is parsed. Scorer/extractor binary hashes, label-file hashes and parity receipts will be appended in a committed pre-read pin before any label parse.
- At registration: no labels parsed or sets scored by this assessment. Each actual read requires its own pre-open LABEL_READS.json and exposure append.
- Coordinator amendments: preserve both KADID comparators/limits; if by_v2fy cannot be located, proceed with B and disclose the missing comparator. October 4/18 prior zensim exposures replace the earlier unread/first-zensim claims; approved populations unchanged.

## Label-free preflight — 2026-11-05, coordinator clarification required

Registration landed at `f3e5dc3dc3759356cb632568c005ec5a64248951`. No label values have been parsed and no set has been scored; no read has been spent by this assessment.

- CLIC library `pairs.tsv`: 57,300 physical rows, 57,300 distinct canonical `(reference, unordered candidate pair)` questions, 57,243 distinct reference crops. The registered 57,243 question count is therefore a reference-crop count; population correction awaits coordinator direction.
- Frozen KonJND JPEG TERMINAL keys: 100 pairs, 100 reference groups. A within-reference SROCC and its interval are undefined with one observation per reference; reporting it as undefined awaits coordinator direction.
- MCL-JCI library `labels.tsv`: 243 threshold rows, 50 references, all 243 rows have `item == ref`. Its README describes JND locations on the JPEG-QF axis, rather than a per-distorted-image quality target. The historical instrument is 5,000 pairs (`benchmarks/mcl-jci.pointer.md`), with `jnd_dist` targets anchored at QF100. Its recorded `/var/tmp/datasets/mcl-jci/mcljci_labels.csv` is absent; expected SHA256 `36f17dd8bac2d085f718e388fd5db307d887338c9c2b296349bb2ef20f184802`. No substitute target or mapping has been invented.

Research parity generator built from the pinned source, SHA256 `47d0709139552e47941faad740e28a4c08c06154011b86f00787768a69bfbd9f`. Build passed: `run-heavy: done rc=0 61s | peak-RSS 1.32GiB | min-avail 36513MiB | peak-load 5.69`. Public generator build and actual parity comparison remain pending; this is not a parity pass.

Raw-file hashes below were computed without parsing label values:

| Library table | SHA256 |
|---|---|
| cid22/labels.tsv | `128a7b1a9a2a7846f02f9253bebe062c8a26b91c39dcd0a711853cfee39a4b9c` |
| csiq/labels.tsv | `495ffe5f9982766c06c024f09512f4339f32f6dac57b0e0d74b083d251eeec49` |
| live-iqa/labels.tsv | `6673c707bc9a5be7de9185c3abfe793a8424777988428034dfb1dff7b452574f` |
| konjnd-1k/labels.tsv | `9c27f7153c7efbae29ef67eb7a2b9e2ea59f7e612c35c73a61d5eab2e6633fd9` |
| jpeg-aic4-sample/labels.tsv | `a64cabbdf47bc77c3692267e30c7db55ea6197114697e33c3dbb81c887bbf498` |
| mcl-jci/labels.tsv | `6dd9668c421eda6546359e6311214965d7f43bc36a2281bb7656556432a55e4a` |
| clic-2022-perceptual/pairs.tsv | `03300f61963c2fc96e5daf3cad3a0e8a43438637959875b904d7a944db295db1` |

## Coordinator instrument decisions — 2026-11-05, before reads

CLIC uses every one of the 57,300 questions across 57,243 reference crops. KonJND JPEG TERMINAL reports pooled signed SROCC with reference bootstrap; within-reference SROCC and its interval are undefined (100 pairs / 100 references). MCL-JCI is on hold until the coordinator supplies the historical 5,000-row `jnd_dist` table or its documented construction rule; this lane must not construct a target. Public parity and all other populations proceed in registration order, each label read after its ledger entry. The earlier preflight notes above retain their historical pending status; these decisions resolve CLIC and KonJND only.

## Coordinator target and label-file amendment — 2026-11-05, before reads

The historical MCL target and builder are lost. Before opening any subject chain, register the supplied construction: for each of 50 sources, pair the pristine BMP with every QF-q JPEG, q=1..100 (5,000 rows). `jnd_dist(source,q)` is the mean across its 30 subject chains of the number of chain points with QF >= q; the QF100 anchor has target0 by definition. This target is distortion-oriented. Report all50 sources and the R7a clean40 subset excluding source numbers1,3,6,7,9,17,28,31,45,50. Record every raw subject-chain SHA and the constructed target table SHA. Byte identity with the lost target table cannot be verified. Construction reference: zenpapers `scripts/jnd_ladder_agreement.py` and `benchmarks/mcl-jci.pointer.md`.

The coordinator named `/var/tmp/datasets/mcl-jci/extracted/MCL-JCI/`; pre-read inspection found0 chain files and0 source-image files there. The in-place library contains50 chain files under `labels/JND_samples/` and50 BMP files under `content/MCL-JCI/source_images/`. A path-substitution question is pending; no chain has been opened, no MCL target constructed and no MCL read performed. Other populations continue.

The October4 pin's exact label files govern any difference from library label tables. This changes label-file authority, not the approved population: CID22-B remains24 references/2100 pairs; the historical October4 pin itself had a23-reference clean allowlist, while the current approved24-reference B bank also includes `3316926_opo25u`. KonJND uses only the100 TERMINAL pairs and preserves the pinned SRC0437 JPEG058 path. LIVE remains its fixed external pair list. Hash checks on all original files below passed before reading labels.

| Population | Governing label file | SHA256 |
|---|---|---|
| cid22_b | `/mnt/v/dataset/cid22/CID22_validation_set/cid22val_pairs_ab.tsv` | `3ce0f7438ea02277102217f504a1a680541e2a599be068f83fc49bb8fb1dfa8b` |
| aic4 | `/mnt/v/output/zensim/v2-backfill-2026-07-20/aic4_pairs.tsv` | `955a9601e94c0877a00c822dbbbd17ad4ae11544dd6d550b9ed79d1ad9d983ff` |
| konjnd_jpeg_terminal | `/mnt/v/output/zensim/v2-backfill-2026-07-20/konjnd_jpeg_val_pairs.tsv` | `70148a39c90d97d35899a171204d3996034cf0019fbe0c496ac51ae4c4d40856` |
| csiq | `/mnt/v/dataset/csiq/csiq_pairs.tsv` | `78b1dac5f74e6f02198f9011eb12f655c358c567d6007497aeac9aa7e5f5a288` |
| LIVE | `/mnt/v/datasets/LIVE/live_r2_pairs.tsv` | `49fcd7d23c9e35b147b52e3361103dbbf6a81aa8ae9d7b16f4763c7e7051fe5d` |

### MCL library path approved — 2026-11-05, before chain reads

Coordinator explicitly approves using the library files in place: chains under `/mnt/tower/input/iqa/datasets/mcl-jci/labels/JND_samples/`, pixels under `content/MCL-JCI/`. All50 raw chain files are hash-pinned without parsing them; aggregate `MCL_RAW_SHA256.json` SHA256 `9f49d76445161e46e61978ced5ca03482ac65d37505373b2ca57ba8d83b2b948`, retained in study `pins/`. The registered5000-row construction and full50/clean40 populations remain unchanged. No subject chain opened for label parsing or target construction yet.

## Final instrument pin — before the first label parse

Public/research parity passed the repository's documented split-parity normalizer: 135,625 records matched, zero missing, extra or different. Both canonical outputs SHA256 `3ab036b8559e819a54feb5795ea6d498a3b2a09aaa34c41312bebc0365a98eb1`. The normalizer maps research E to public A and compares Rev1 preview/configuration records and Rev7 A/HDR records. Research uses `ZENSIM_POW_FORM=libm`; the public default supplies the matching arithmetic. The unnormalized outputs include research-only records and therefore differ in shape. Official comparison receipt: `run-heavy: done rc=0 238s | peak-RSS 0.32GiB | min-avail 38135MiB | peak-load 4.16`.

Frozen binaries:

| Instrument | SHA256 |
|---|---|
| bake_dial_refit | `4174bb026a64abb15e402adb16a33f905da0864edcd538f026f3a1a8db733429` |
| jnd-ladder-score | `7a1cf243895b958891ee83efe2c32828634da0e209305004dbfac3caeb2e8346` |
| panel | `54901b0cda291117a108aa4f0c3f2b481d26dce62f9c1c6e68ec208a3820533c` |
| serve_custom_bake | `3eefd0d45b8888bbc3464d3529179f75a937fe7747097d1ad4c335f62c3fe94b` |
| split_parity_check | `2609d89b6b07305e15f9997d561cb285ef00867de62e3227f8175feed3078c9b` |
| split_parity_golden | `47d0709139552e47941faad740e28a4c08c06154011b86f00787768a69bfbd9f` |
| trust-decode | `47dc7d599febbee982e83545528494abc119b7f2de0f41999e111047097a213b` |

Extractor SHA256 `ed4957417339779dd8ed25fe59a6166326ba9db86d5c1530b1e1c788c9d614d8`, source `b77883be18db5933a9db4b9e2670a2ba88a9729b`, serial Rev7 `basic+peaks+v2@w1825/rev7_rowlanes#36c3f3af`. Raw f64 CSVs are retained beside each bank. Predictor source is the registered research revision; the signed statistical tool is commit `a6c23628`, native serving comparator `cc6169df`, decoder including opaque RGBA support `332371bd`. All48 panel tests, strict workspace clippy and script lint passed; synthetic signed controls exercised all10,000 reference draws without real labels.

KADID research B is `codec_target()` at the registered source, formula5, bake SHA256 `89cb4861e2651ff8d4a98bb879feed83a0f9eb3f22b83e644e42eb2291d74b2e`. The coordinator explicitly approves by_v2fy seed0, epoch119 (`epoch_rule:last`), formula4, bake SHA256 `d34f53111fa8baff4fc6003947768facfedc123e5bb1fd97aef99302297bd6d5`, original `full_s0/refit/last.bin` under the historical confirmation recipe. Both comparators will execute natively on the same decoded pixels. Model manifest SHA256 `088a81c1685ecd99a57ec8f85b41f7ca74d606a1a229a9bf22fc18dfb1b6bc46`. KADID label file `/mnt/v/dataset/kadid10k/dmos.csv` SHA256 `573e2ed98fdaa2a5aed7b50ad716906e125e1001ed0e97c8e93e79d7b997ba99`. Its `dmos` field is quality MOS; all2,000 original TERMINAL rows govern over the older1,952-row feature bank.

Targets and score orientations: CID's pinned `human_score` is MCOS/100, restored to MCOS by multiplying100 for collection-scale RMSE (construction: `scripts/canonical_corpus/build_fr_corpus_pairs.py`). CSIQ is already1−DMOS; LIVE is already1−dmos_new/100; AIC4 `human_score` is distortion q_jnd; KonJND is its pinned PJND threshold; MCL is the registered distortion count. A/SSIM2/B/by_v2fy are quality, and butteraugli quality is explicitly −p3. Signed statistics preserve that relationship. CLIC uses every question's majority, with half credit for tied majority or prediction. Intervals use10,000 whole-reference PCG64 draws seed20261105, percentile95% with linear/type7 endpoints; paired KADID draws are shared.

Coordinator staging amendment supersedes in-place content reads for LIVE, MCL and CLIC: one sequential rsync from each library content folder to `/var/tmp/final-read-a-20261105/stage/<id>/content`, every file verified against its dataset SHA256SUMS before decoding. No large data under home or the local dataset volume. Every scorer consumes the native normalized PNGs; independent Pillow readback verifies geometry and every RGB SHA. No third-party pixels enter feature extraction or scoring.

CID22-B label-free completion: 2,100 pairs,24 references,2,124 stimuli. Keys SHA256 `b699066295048de539ba988f86c5435b327a33cce612acb334374f43d1b1415e`; decode-map SHA256 `bf77ceb8e9080102726ae0fa837bb504eda4ed3d5e7734d5a3d16ff32f56ecfa`. All4,200 feature RGB checks and all2,124 independent stimulus readbacks passed. Zero pixel-identical pairs; the cached prediction CLI's lack of identity evidence does not affect this population. Later populations are guarded before label parsing.

Receipts: native decode `rc=0 71s | peak-RSS 0.06GiB | min-avail38427MiB | peak-load1.07`; feature extraction `rc=0 37s | peak-RSS0.28GiB | min-avail38316MiB | peak-load0.88`; A prediction `rc=0 1s | peak-RSS0.19GiB | min-avail38397MiB | peak-load0.47`; peers `rc=0 76s | peak-RSS0.13GiB | min-avail38019MiB | peak-load1.75`; independent readback `rc=0 10s | peak-RSS0.07GiB | min-avail31080MiB | peak-load1.03`. The initial opaque-alpha decoder refusal and peer scorer's missing model-directory environment failure occurred before label reads; their logs and intermediate artifacts are retained.
