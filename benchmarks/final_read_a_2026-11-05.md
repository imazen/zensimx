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

## CID22-B result — frozen-A assessment, read spent

2,100 pairs /24 references; quality MCOS, butteraugli quality=−p3. 95% whole-reference percentile intervals,10,000 valid draws for every cell, seed20261105. Within-reference mean defined on all24 references. Prior exposures: DATA_SPLITS lines1613–1617,2761–2791. CID22-B is the second half of the reference set whose first half trained A.

| Metric | Signed SROCC | KROCC | PLCC4PL | RMSE (MCOS) | Within-reference SROCC |
|---|---|---|---|---|---|
| A | 0.908926706 [0.876866846, 0.939159319] | 0.734654357 [0.689769001, 0.783003589] | 0.913389974 [0.883148694, 0.941129940] | 5.452548481 [4.532380746, 6.245206491] | 0.959433709 [0.950752289, 0.967297550] |
| SSIMULACRA2 | 0.913094224 [0.883263150, 0.940670082] | 0.739625436 [0.695978572, 0.785353720] | 0.917160646 [0.889563468, 0.941858485] | 5.337787846 [4.507927732, 6.113361596] | 0.958154147 [0.946529619, 0.967491009] |
| butteraugli3_quality | 0.781058047 [0.698243342, 0.861063845] | 0.598548287 [0.528366873, 0.676012534] | 0.776074628 [0.709731641, 0.855133028] | 8.446872954 [6.859664576, 9.519745951] | 0.903902685 [0.864125963, 0.935446761] |

Read receipt: `rc=0 0s | peak-RSS0.07GiB | min-avail31355MiB | peak-load0.75`. Statistics: `run-heavy: done rc=0 81s | peak-RSS 0.89GiB | min-avail 30410MiB | peak-load 6.43`. Pre-open exposure commit `a02598e15b37443faf91f8c50ae8b300eedca6cb` was verified on origin/main before parsing. Target-cache SHA256 `a8b0caa3a1a9f238a3cab88580097874332fee0ec4dcfcabbdc7e483295fcc3f`; RESULTS.json SHA256 `5003d8fe0f20187569ec7fbada9144917feca955935766d09679eee7cdf64c5b`; completed LABEL_READS.json SHA256 `7b72c7050e58c81d9cf4f61d33eeef0e4ae7dd1bbcbb0838ac0e22b4c77d4c2f`. No model selected or changed.

## CSIQ result — frozen-A assessment, read spent

866 pairs /30 references; quality1−DMOS; butteraugli quality=−p3. 95% whole-reference intervals,10,000 valid draws for all cells, seed20261105. All30 within-reference correlations defined. Prior exposures: DATA_SPLITS lines1613–1617,2761–2791; the earlier October18 candidate read had865 rows. Governing pinned label SHA256 `78b1dac5f74e6f02198f9011eb12f655c358c567d6007497aeac9aa7e5f5a288` differs bytewise from library SHA256 `495ffe5f9982766c06c024f09512f4339f32f6dac57b0e0d74b083d251eeec49`; the pinned file governs.

| Metric | Signed SROCC | KROCC | PLCC4PL | RMSE | Within-reference SROCC |
|---|---|---|---|---|---|
| A | 0.946058167 [0.938141640, 0.953728089] | 0.795676331 [0.781180043, 0.811851468] | 0.942352010 [0.933710083, 0.951353980] | 0.087848457 [0.080513655, 0.094598213] | 0.951751419 [0.945748743, 0.957371023] |
| SSIMULACRA2 | 0.904756667 [0.893626292, 0.915689934] | 0.732740967 [0.717562476, 0.749165240] | 0.887734990 [0.876762161, 0.899472346] | 0.120856721 [0.112244940, 0.129125137] | 0.908420311 [0.898231794, 0.918019836] |
| butteraugli3_quality | 0.814449907 [0.797829260, 0.830706233] | 0.635796157 [0.620981491, 0.651910146] | 0.808457237 [0.795328985, 0.822168439] | 0.154513019 [0.145254183, 0.163196645] | 0.815148643 [0.798092187, 0.831281077] |

All896 decoded stimuli independently verified,1,732 feature RGB checks passed,zero identical pairs. Receipts: keys `rc=0 0s | peak-RSS0.07GiB | min-avail31356MiB | peak-load6.43`; decode/readback `rc=0 7s | peak-RSS0.07GiB | min-avail31344MiB | peak-load5.91`; features `rc=0 16s | peak-RSS0.21GiB | min-avail31124MiB | peak-load5.52`; A prediction `rc=0 1s | peak-RSS0.17GiB | min-avail31419MiB | peak-load3.22`; peers `rc=0 32s | peak-RSS0.13GiB | min-avail31124MiB | peak-load3.22`; read `rc=0 0s | peak-RSS0.07GiB | min-avail31501MiB | peak-load1.72`; statistics `run-heavy: done rc=0 94s | peak-RSS 0.35GiB | min-avail 30848MiB | peak-load 7.34`. Pre-open commit `c37ad1e3561361088dc388f4c0447cef0b796128` verified remotely before parse. Target cache SHA256 `ef0a1ea87d0672bb4b4f1b3bd8368fcd728c3bae4c9502e4ddffdbb5d09efa0c`; result SHA256 `e4cf91e3e972a024f528d56614eca627a8b0b2e0601dea7a50c9ca4e650dbc07`; completed journal SHA256 `8fb4c1747e32a65ba8c6e471669659ad2793a142d2d153cd46ba97cd3a36707a`.

## LIVE release2 — frozen-A read spent, full and fixed clean10

Quality1−dmos_new/100; butteraugli quality=−p3. All intervals95% whole-reference percentile,10,000 valid draws, seed20261105. Prior exposure at DATA_SPLITS lines2761–2791 includes October18 clean10. LIVE19 Kodak references contain TID2013 training content; fixed clean10 excludes them per the October3 audit.

### all: 779 pairs /29 references

| Metric | Signed SROCC | KROCC | PLCC4PL | RMSE | Within-reference SROCC |
|---|---|---|---|---|---|
| A | 0.967369417 [0.961121429, 0.972128864] | 0.840643847 [0.828768091, 0.852753627] | 0.966337129 [0.961342200, 0.971113515] | 0.070294207 [0.065279943, 0.075160488] | 0.965753259 [0.959953828, 0.970693398] |
| SSIMULACRA2 | 0.959925801 [0.949924550, 0.967667174] | 0.829245353 [0.813585678, 0.843694491] | 0.954484868 [0.946026068, 0.961766968] | 0.081490884 [0.074692866, 0.088431477] | 0.956554273 [0.946893059, 0.964607821] |
| butteraugli3_quality | 0.916378056 [0.899526499, 0.931583005] | 0.754247622 [0.730527016, 0.778688485] | 0.908102453 [0.891745565, 0.923425759] | 0.114410998 [0.104747730, 0.123731020] | 0.916418915 [0.897548271, 0.932379095] |

RESULTS.json SHA256 `16e60bd182430a65111b48025aa8b8eb3975693ab5d12eae774b8938976ea954`.

### clean10: 267 pairs /10 references

| Metric | Signed SROCC | KROCC | PLCC4PL | RMSE | Within-reference SROCC |
|---|---|---|---|---|---|
| A | 0.966511023 [0.960109572, 0.972365728] | 0.840273186 [0.832851828, 0.858804726] | 0.966413098 [0.961852258, 0.973010408] | 0.070229439 [0.062332842, 0.074687031] | 0.967286021 [0.958513579, 0.974320467] |
| SSIMULACRA2 | 0.960528006 [0.944989557, 0.970799008] | 0.834471595 [0.817053304, 0.855720577] | 0.957617506 [0.943237948, 0.969154883] | 0.078714370 [0.066430848, 0.090158154] | 0.953676979 [0.934651520, 0.968049588] |
| butteraugli3_quality | 0.910527257 [0.894009505, 0.924940264] | 0.747898335 [0.728049214, 0.775646538] | 0.907636635 [0.892209265, 0.924049681] | 0.114708222 [0.103741393, 0.122959783] | 0.908387086 [0.890141813, 0.924478728] |

RESULTS.json SHA256 `be96c8c8e2a95596cb0be65bfe33201a0dca3ec2176d51193a24c0ff59755a06`.

Staging:1,020 files /1,083,889,744 bytes, every file verified; dataset SHA256SUMS SHA256 `5c46005ca25c5e35e24ed4c844759d82565bd32311b72baef19116051fb3f09d`. All808 source stimuli /792 distinct RGB buffers independently verified; all1,558 feature RGB hashes matched; no identical pairs. Governing LIVE label SHA256 `49fcd7d23c9e35b147b52e3361103dbbf6a81aa8ae9d7b16f4763c7e7051fe5d`; library SHA256 `6673c707bc9a5be7de9185c3abfe793a8424777988428034dfb1dff7b452574f` differs bytewise.

Receipts: staging `rc=0 43s | peak-RSS0.07GiB | min-avail31281MiB | peak-load7.34`; keys `rc=0 0s | peak-RSS0.06GiB | min-avail31420MiB | peak-load3.54`; decode/readback `rc=0 161s | peak-RSS0.07GiB | min-avail31257MiB | peak-load3.34`; bank `rc=0 18s | peak-RSS0.22GiB | min-avail31075MiB | peak-load1.72`; prediction `rc=0 0s | peak-RSS0.17GiB | min-avail31321MiB | peak-load1.66`; peers `rc=0 78s | peak-RSS0.17GiB | min-avail30935MiB | peak-load2.13`; read `rc=0 0s | peak-RSS0.07GiB | min-avail31404MiB | peak-load1.59`; statistics `run-heavy: done rc=0 102s | peak-RSS 0.35GiB | min-avail 30976MiB | peak-load 7.39`. Pre-open commit `6cf232ec519f` verified remotely before parse. Target cache `340d4c9f276179b1ea5ce74d80f5889694180e13d367cab5e6ff2e609c43cd10`; completed journal `65eb0b35f147ebfbb7c392d176be595c40a8051acc8a00e1edeaf97134d2ada3`.

## KonJND JPEG TERMINAL — frozen-A read spent

100 pairs /100 references, pooled signed statistics versus pinned raw PJND threshold. Signs retained, butteraugli quality=−p3. Within-reference SROCC and its interval undefined: one pair per reference. 10,000 valid draws for every pooled statistic, seed20261105,95% whole-reference percentile intervals. Prior zensim exposure: DATA_SPLITS lines1613–1617.

| Metric | Signed SROCC | KROCC | PLCC4PL | RMSE (PJND) | Within-reference SROCC |
|---|---|---|---|---|---|
| A | -0.368034324 [-0.533382696, -0.180963615] | -0.257073586 [-0.383895537, -0.124410400] | -0.394729406 [-0.608187606, -0.264247975] | 7.477532075 [6.081121784, 8.377865945] | undefined |
| SSIMULACRA2 | -0.400978113 [-0.567517238, -0.208665174] | -0.283346830 [-0.408301261, -0.148547697] | -0.449908665 [-0.612844212, -0.319472454] | 7.268189518 [5.975707621, 8.058430663] | undefined |
| butteraugli3_quality | 0.052950089 [-0.162600740, 0.260998445] | 0.022635410 [-0.123201474, 0.163489972] | 0.405172900 [0.245605173, 0.626847834] | 7.440442665 [5.797174062, 8.452195856] | undefined |

All200 stimuli independently verified and all200 feature RGB comparisons passed. The pinned target file SHA256 `70148a39c90d97d35899a171204d3996034cf0019fbe0c496ac51ae4c4d40856` governs over byte-different library table SHA256 `9c27f7153c7efbae29ef67eb7a2b9e2ea59f7e612c35c73a61d5eab2e6633fd9`; pinned SRC0437 JPEG058 retained. Pre-open journal was committed and verified on origin/main in `87200234d546` before parse.

- keys: `run-heavy: done rc=0 1s | peak-RSS 0.06GiB | min-avail 31253MiB | peak-load 6.96`
- decode: `run-heavy: done rc=0 16s | peak-RSS 0.07GiB | min-avail 31258MiB | peak-load 6.96`
- bank: `run-heavy: done rc=0 2s | peak-RSS 0.17GiB | min-avail 31185MiB | peak-load 5.71`
- predict: `run-heavy: done rc=0 0s | peak-RSS 0.14GiB | min-avail 31320MiB | peak-load 5.33`
- peers: `run-heavy: done rc=0 8s | peak-RSS 0.12GiB | min-avail 31167MiB | peak-load 5.33`
- ledger: `run-heavy: done rc=0 0s | peak-RSS 0.06GiB | min-avail 31286MiB | peak-load 4.67`
- read: `run-heavy: done rc=0 1s | peak-RSS 0.06GiB | min-avail 31367MiB | peak-load 2.43`
- stats: `run-heavy: done rc=0 8s | peak-RSS 0.13GiB | min-avail 31308MiB | peak-load 3.37`

Target cache `ca83376cfab13f34285d5326589b9ad54b211449bf7ef510977547f5b4c6a6c3`; result `614576af02fe66d5b2de476dec8d5e0d7c8a9ce56f622bf5fc1ce02b7ecf0776`; completed journal `a3e2cc761c67dc55ca6dc5bdc7683c62f7d2a512741619e8c88bb8ca5a6806f4`.
