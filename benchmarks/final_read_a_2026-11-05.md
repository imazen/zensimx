# Final registered frozen-A assessment (zensim 0.3.0) — 2026-11-05

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
| MCL-JCI | 5,000 rows /50 sources; also registered clean40 subset | mean subject-chain JND count (distortion) | library `mcl-jci` | signed SROCC, within-reference SROCC. Prior exposure, disclosed: zenpapers read every subject's JND chain and 243 library score rows on 2026-10-24/25 for JND-unit and observer studies with third-party metrics only, without a preregistration (ledger, "Exposure ledger — 2026-10-24/25"); no zensim model was scored, fitted or selected on them. D3 made its images training sources (KonFiG-IQA crops ten of them); its ratings never trained any zensim model Prior zensim read: October 4 R7/R7a, including the 40-source subset excluding KonFiG crops; `docs/DATA_SPLITS.md:1601–1617`. Additional October 24/25 exposure: `docs/DATA_SPLITS.md:2867–2899`. |
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

### KADID pre-label identity-surface check

All2,000 original TERMINAL pairs /16 references retained. All2,016 stimuli independently decoded; all4,000 feature RGB checks passed. Keys SHA256 `5cf3d221f3905f041caab2fed07a4462b62a5e3bf668cf242941dcf23096d5a6`; decode-map SHA256 `4b5e0abd37ac2d43727480a9cbbb743db4a844d349728bc7c6232379425b776a`. There are64 byte-identical pairs. The cached predictor does not pass an identity flag, so all64 were checked before labels against the native pixel surface:64/64 cached/native scores exactly100.0, zero differences. Frozen A, predictor and native scorer unchanged. This verifies the registered cached path on every identity row without overriding any prediction or dropping any pair. The pre-read orchestration guard now requires a bank/model/binary-bound exact parity receipt whenever identity rows occur.

Cached64-score TSV SHA256 `408071c6659cc576a746a9b765ad3430010d38bb17611c50d00fa7cbd2e136bd`; native64-score TSV SHA256 `57249289e4acc401cbbbcf252904bfd4d8e02ee50f5505ec2f96d0ec2facae9c`. Receipts: cached `rc=0 0s | peak-RSS0.06GiB | min-avail31454MiB | peak-load1.00`; native `rc=0 1s | peak-RSS0.02GiB | min-avail31394MiB | peak-load1.00`. The preliminary single-row Snappy control was refused by the frozen loader, retained as failed; its Zstd replacement and all64-row controls passed. No human target was parsed in any control.

KADID preparation receipts: keys `rc=0 0s | peak-RSS0.07GiB | min-avail31349MiB | peak-load3.10`; decode/readback `rc=0 11s | peak-RSS0.07GiB | min-avail31293MiB | peak-load2.93`; bank `rc=0 31s | peak-RSS0.28GiB | min-avail30483MiB | peak-load2.71`. KonJND's completed label parse receipt was `rc=0 1s | peak-RSS0.06GiB | min-avail31367MiB | peak-load2.43`.

### KADID comparator serving blocked — labels remain unopened

A and reference peers cover all2,000 rows. B natively served all2,000 rows at process revision5. The approved seed0/epoch119 by_v2fy artifact (`d34f53111fa8baff4fc6003947768facfedc123e5bb1fd97aef99302297bd6d5`) refuses its first pair at process revision4: `ModelLoadFailed { reason: "formula revisions 4 and 5 cannot be mixed with another revision in one process: other formula gates follow ZENSIM_FORMULA_REV, so the request and the process must carry the same revision when either is revision 4 or later" }`. Its preserved output has a header and zero predictions; B has2,000 predictions. No KADID LABEL_READS.json was created and no KADID target parsed. The model was located, so the coordinator's missing-artifact fallback does not automatically authorize omission of this unservable comparator. No revision switch, metadata rewrite, substitute model or threshold change was made.

The native `read-meta` diagnostic for `zentrain.formula_revision` reports `MetadataTypeMismatch expected Utf8, got Bytes`. The metadata owner's `get_utf8` returns this same diagnostic for an absent key; it therefore does not establish that the artifact declares revision5. The generic serving refusal above likewise does not print the requested revision. The historical result manifest still binds seed0/last epoch119 to the approved SHA. Coordinator direction is required before altering the interpretation or producing a metadata-stamped derivative.

Peer receipt: `run-heavy: done rc=0 64s | peak-RSS 0.13GiB | min-avail 30930MiB | peak-load 1.18`. Combined comparator invocation (B complete, by_v2fy refused): `run-heavy: done rc=1 9s | peak-RSS 0.13GiB | min-avail 30973MiB | peak-load 1.40`. All failure logs and partial outputs retained. Remaining populations AIC4, MCL and CLIC have no target reads by this assessment.

Blocker receipt SHA256 `4a6ca17d58a9d1bef33173b7d902cd6d166ad2c39684a30be369c1a1ca60c759`; B predictions SHA256 `b2dc1fab177fcfa3cd030f1288c9d4fa9738fa435ab5624bdbce0604c936360f`; by_v2fy header-only TSV SHA256 `424a269ef5aed4e649943a99a85f93904fa069e1728cc1c9f5a24fc2a112749b`.

### Coordinator-approved by_v2fy Rev4 metadata derivative — before labels

Coordinator approved the conditional metadata-only copy and validation. The original key was absent: canonical `bake_dial_refit append-meta` explicitly checks every existing key and refuses any overwrite before writing. The original had only `zentrain.repro`; its training record is the registered Rev4 historical-replay recipe, selected seed0 /last epoch119. Only UTF8 `zentrain.formula_revision = 4` was appended to a separate copy. Original SHA256 `d34f53111fa8baff4fc6003947768facfedc123e5bb1fd97aef99302297bd6d5` remains unchanged; the derivative SHA256 is `130ddf713c37551f8ec204b49d25cd4f61eb4cb858bf3c055ee07242dbad2017`.

Numerical preservation: stripping metadata from canonical comparison copies yielded byte-identical numerical serialization SHA256 `dd453ffa91406a559b0eb9d047899e5d7dea8f6047bd918a4e0ef2fb47b27f09` (weights, scalers, transforms and numerical sections). The original repro metadata is byte-identical, SHA256 `9135e72859fea469986d5e70e917f52cca7ef93f2b62e078009d8bd4cb07b6be`. All256 finite cached-forward synthetic probes matched byte-for-byte, TSV SHA256 `3d324970703a3d4652f5eaea55fd2794218e8897a73b9a1d1085d8aadfa718a8`. Probe matrix:256×1853, PCG64 seed20261105 uniform[0,1), with first four rows replaced by zero,one,linspace(−1,1),100. No human target used. Original and derivative cached receipts: `rc=0 0s | peak-RSS0.09GiB | min-avail33598MiB/33548MiB | peak-load1.52`.

The derivative natively served all2,000 original TERMINAL rows at processRev4, all finite. Receipt: `run-heavy: done rc=0 124s | peak-RSS 0.04GiB | min-avail 32962MiB | peak-load 1.78`. Failed original output is preserved separately. No model selection, threshold change, numeric weight/calibration change or human-label parse occurred. The amended comparator identity is this metadata derivative of the approved artifact; B and A are unchanged.

Validation receipt SHA256 `9869ae54ddaccb35250433c079b4eaf29f259a08267afc3e6bb78393a9abf7fb`; amended MODELS.json SHA256 `f350016a9fdc7626d3a575523f87d4f1896abe6a35bb18d04a8fff58398c3a95`; native by_v2fy predictions SHA256 `cc556b01639daa2fef3fd69765248f86536edde22431c5922faa4c37664fd07b`.

## KADID TERMINAL result — FAILED confirmation; read spent

All2,000 original pairs /16 references. Quality MOS, butteraugli quality=−p3. Whole-reference95% percentile intervals,10,000 valid draws for every statistic, seed20261105. Within-reference SROCC defined on all16 references. No prior zensim scoring of this population; prior third-party label studies October24–26 at DATA_SPLITS lines2892–2924 remain disclosed.

| Metric | Signed SROCC | KROCC | PLCC4PL | RMSE (MOS) | Within-reference SROCC |
|---|---|---|---|---|---|
| A | 0.892838812 [0.883444091, 0.901577301] | 0.708078600 [0.695952302, 0.721110899] | 0.891139627 [0.881768468, 0.900902122] | 0.499342720 [0.478590049, 0.517265888] | 0.897371218 [0.887072563, 0.906736372] |
| B | 0.941618888 [0.937373776, 0.946677087] | 0.788619351 [0.781637102, 0.798223979] | 0.940842116 [0.936376741, 0.946345510] | 0.372909490 [0.355672726, 0.385805458] | 0.947274690 [0.942929472, 0.951460179] |
| by_v2fy | 0.933327560 [0.925634631, 0.941471833] | 0.775573063 [0.763691673, 0.789402967] | 0.932774187 [0.924550105, 0.941999933] | 0.396698545 [0.368596137, 0.420334893] | 0.940828085 [0.933730830, 0.947201406] |
| SSIMULACRA2 | 0.800193337 [0.778928913, 0.822535024] | 0.604825222 [0.583410966, 0.628967958] | 0.796059333 [0.777433125, 0.816340739] | 0.666061413 [0.636599357, 0.689099380] | 0.813745969 [0.792226851, 0.833588779] |
| butteraugli3_quality | 0.571754058 [0.536981650, 0.605816189] | 0.405244249 [0.378841685, 0.432870012] | 0.583479809 [0.558264621, 0.609477487] | 0.893771107 [0.867990462, 0.915513485] | 0.573801564 [0.538404499, 0.607781774] |

Both original rules failed: A−B=-0.048780075448, paired-reference SE=0.004090569222, required≥-0.008181138444 (−2SE); paired95% interval[-0.057570507085,-0.041547035624]. A−by_v2fy=-0.040488748392, required≥−0.005; paired95% interval[-0.049274191331,-0.032347297759]. Both paired intervals have10,000 valid draws. This is a failed frozen confirmation, not a selection or retuning opportunity. The approved metadata-only by_v2fy derivative above is used; all numerical content is unchanged.

### Every distortion type: signed SROCC [95% reference interval]

Each type has80 rows /16 references (verified from the cached targets),10,000 finite draws, seed20261105. Canonical `zenstats::resample::spearman_interval`; small-source percentile intervals are descriptive and may under-cover.

| Type | A | B | by_v2fy | SSIMULACRA2 | Butteraugli3 quality |
|---|---|---|---|---|---|
| 01 | 0.959068858 [0.940918428, 0.975193989] | 0.963731248 [0.946315863, 0.977882959] | 0.962577424 [0.947393922, 0.976108657] | 0.948778634 [0.928643325, 0.968126803] | 0.945175879 [0.925462836, 0.965978050] |
| 02 | 0.899328186 [0.853925108, 0.945395010] | 0.909213620 [0.876497752, 0.942617866] | 0.897557486 [0.859636473, 0.934306626] | 0.868827579 [0.818655082, 0.924944280] | 0.869965048 [0.817590808, 0.922764734] |
| 03 | 0.965239578 [0.948109423, 0.979411556] | 0.957471571 [0.940052097, 0.972491247] | 0.953987738 [0.931190762, 0.973039914] | 0.945372312 [0.921493906, 0.969374047] | 0.946972992 [0.929745082, 0.967410048] |
| 04 | 0.945178539 [0.920944549, 0.962611121] | 0.910100279 [0.847393941, 0.957492881] | 0.886054320 [0.810474298, 0.938963792] | 0.910475448 [0.865309150, 0.946002799] | 0.871223439 [0.819029507, 0.920985871] |
| 05 | 0.789797537 [0.609764113, 0.918385823] | 0.807177117 [0.648363889, 0.916469452] | 0.808408463 [0.640706615, 0.928808532] | 0.743721714 [0.581084208, 0.878555104] | 0.646621227 [0.495066031, 0.813150972] |
| 06 | 0.901139663 [0.853655587, 0.938691097] | 0.882638513 [0.810589035, 0.936276710] | 0.885323407 [0.811200166, 0.935827244] | 0.909686772 [0.868582867, 0.938537188] | 0.849376306 [0.775206328, 0.907986209] |
| 07 | 0.725359333 [0.581664446, 0.830181186] | 0.555461156 [0.372806962, 0.712923691] | 0.506399524 [0.299039380, 0.709629327] | 0.689688807 [0.524253191, 0.822904612] | 0.600465587 [0.402665820, 0.761562346] |
| 08 | 0.924671718 [0.896579328, 0.955610484] | 0.924424561 [0.879028247, 0.961872776] | 0.918539880 [0.869254107, 0.958778580] | 0.897296181 [0.850558082, 0.948127262] | 0.854361547 [0.787849199, 0.928147690] |
| 09 | 0.939509490 [0.907629889, 0.962717924] | 0.931688745 [0.899335372, 0.958165960] | 0.933130951 [0.901772636, 0.957325646] | 0.932837820 [0.900815057, 0.958038873] | 0.943472626 [0.914212407, 0.964003843] |
| 10 | 0.896027387 [0.850670529, 0.939062067] | 0.885216194 [0.838952922, 0.931493846] | 0.875296163 [0.830707277, 0.921046163] | 0.878520760 [0.834826528, 0.923162455] | 0.904962450 [0.866810781, 0.936831184] |
| 11 | 0.855794545 [0.744293653, 0.940848766] | 0.922263455 [0.890591164, 0.954906912] | 0.919179785 [0.895095008, 0.947603570] | 0.878283056 [0.803876402, 0.940535823] | 0.923869777 [0.883548548, 0.960814113] |
| 12 | 0.899029351 [0.834663659, 0.956149005] | 0.942170193 [0.925518903, 0.961962365] | 0.940552411 [0.915403406, 0.963549301] | 0.915887104 [0.871968775, 0.960060482] | 0.955804106 [0.932100109, 0.976283509] |
| 13 | 0.896211363 [0.841323514, 0.947417161] | 0.911127112 [0.876852143, 0.949167130] | 0.891309804 [0.860335781, 0.927442361] | 0.705965208 [0.632061712, 0.811061033] | 0.774504950 [0.707814549, 0.863297219] |
| 14 | 0.936318622 [0.915793591, 0.956560848] | 0.960454836 [0.944033607, 0.973784177] | 0.953444901 [0.932402215, 0.971336521] | 0.940866874 [0.917440991, 0.961943221] | 0.933669383 [0.875162645, 0.972410300] |
| 15 | 0.945683939 [0.915496606, 0.971396123] | 0.922213007 [0.886738602, 0.958412606] | 0.912822290 [0.873769604, 0.952044280] | 0.938825550 [0.915927020, 0.962488046] | 0.843382565 [0.789540944, 0.902316695] |
| 16 | 0.932486160 [0.903208421, 0.960381638] | 0.950528153 [0.938446306, 0.962631206] | 0.941958500 [0.929964685, 0.955435156] | 0.898078591 [0.852819464, 0.941473572] | 0.927515527 [0.903062823, 0.953015249] |
| 17 | 0.882039880 [0.840113178, 0.925779834] | 0.928508700 [0.902480959, 0.953027135] | 0.916548519 [0.891868337, 0.941997031] | 0.889731918 [0.850441576, 0.929335707] | 0.909337234 [0.883522796, 0.936315647] |
| 18 | 0.740248668 [0.631594302, 0.839244675] | 0.784831304 [0.712627684, 0.851862494] | 0.799126054 [0.726212928, 0.863405435] | 0.701834476 [0.619993620, 0.789417578] | 0.710361790 [0.592218147, 0.820237742] |
| 19 | 0.942007425 [0.927000438, 0.958757528] | 0.959685377 [0.943621046, 0.973974760] | 0.957071198 [0.942276002, 0.970534752] | 0.848401028 [0.800113958, 0.907987593] | 0.802377394 [0.725652979, 0.866886570] |
| 20 | 0.500430869 [0.243183230, 0.748037580] | 0.575580497 [0.387413127, 0.763785886] | 0.417649040 [0.184053125, 0.683388100] | 0.204322577 [-0.091758071, 0.553664364] | 0.195459376 [-0.108128764, 0.466965978] |
| 21 | 0.576921739 [0.477188843, 0.680690164] | 0.638902256 [0.524065584, 0.740673638] | 0.583088138 [0.479391287, 0.693574067] | 0.506922557 [0.387398638, 0.638844573] | 0.535984351 [0.413318803, 0.632257802] |
| 22 | 0.847752332 [0.786151105, 0.908664505] | 0.841468517 [0.771511716, 0.900008368] | 0.860659945 [0.799021311, 0.920228234] | 0.834680589 [0.754343625, 0.914550349] | 0.799158276 [0.721618909, 0.872259593] |
| 23 | 0.488774954 [0.297451617, 0.696025602] | 0.609810082 [0.466646316, 0.752604004] | 0.622387598 [0.494339610, 0.756293389] | 0.551815061 [0.407317703, 0.687725339] | 0.374896807 [0.221631809, 0.530394292] |
| 24 | 0.888594529 [0.839273060, 0.942326002] | 0.912908429 [0.875626951, 0.958323125] | 0.883037735 [0.844883630, 0.927285498] | 0.844245692 [0.776368159, 0.919054094] | 0.852334695 [0.787239907, 0.920375878] |
| 25 | 0.720772151 [0.612507818, 0.839551250] | 0.833708678 [0.772054077, 0.883449219] | 0.736966247 [0.629118567, 0.838278331] | 0.716972811 [0.613948307, 0.817270293] | 0.718720038 [0.615698076, 0.828923340] |

### Full-population native scatter diagnostics

All2,000 rows, `zenstats::scatter`, schema scatter-v2; raw tails and rank-normalized geometry are retained. No plot sample, label drop or clipping of raw scores. These are descriptive geometry measurements, not additional confirmation thresholds.

geo

| Metric | clampHi | clampLo | clump | cov | mad | maxd | out4 | p99d | span | target_clump | target_cov |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 0.032 | 0.0005 | 0.078 | 1 | 0.44478 | 0.475274725275 | 0 | 0.370934065934 | 3.64 | 0.083 | 1 |
| B | 0.032 | 0.0005 | 0.078 | 1 | 0.29652 | 0.475274725275 | 0.0065 | 0.31043956044 | 3.64 | 0.083 | 1 |
| by_v2fy | 0.032 | 0.0005 | 0.078 | 1 | 0.340998 | 0.53021978022 | 0.005 | 0.340714285714 | 3.64 | 0.083 | 1 |
| SSIMULACRA2 | 0.032 | 0.0005 | 0.078 | 1 | 0.59304 | 0.678571428571 | 0.001 | 0.541208791209 | 3.64 | 0.083 | 1 |
| butteraugli3_quality | 0.032 | 0.0005 | 0.078 | 1 | 1.082298 | 0.824175824176 | 0 | 0.659340659341 | 3.64 | 0.083 | 1 |

raw

| Metric | clump | cov | mad | max | p99 | pred_max | pred_min | residual_max | residual_p99 | slope |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 0.1335 | 0.8 | 21.5287742548 | 5.3397586906 | 3.4480288977 | 100 | -152.571829718 | 114.958459425 | 74.2318357624 | 36.0702065151 |
| B | 0.1105 | 0.85 | 13.0548393128 | 5.30785103369 | 2.87973263039 | 100 | -97.6971710788 | 69.2931423411 | 37.5944467535 | 29.5076260257 |
| by_v2fy | 0.112 | 0.9 | 12.1685264846 | 5.02909278054 | 3.17158988116 | 100 | -86.4011535645 | 61.1966486932 | 38.593575467 | 28.1957698341 |
| SSIMULACRA2 | 0.124 | 0.6 | 28.8745437512 | 5.20399273661 | 2.72189974116 | 100 | -211.62195 | 150.262915954 | 78.5936131626 | 36.1462437593 |
| butteraugli3_quality | 0.28 | 0.7 | 8.4255636493 | 8.47261252934 | 5.13571900041 | -0 | -89.671973 | 71.3865361418 | 43.2713273229 | 5.86666866817 |

Scatter receipt SHAs:

- A: `62323433d33cf6f2309cc89149400676d4ce2813018ea3dc7bfaed3bb6604856`
- B: `bcb0c8f70b7993a03a8ed9f50551b6e1c6ab8837e8d89855e632b3bd8e10d769`
- by_v2fy: `c83fa4cc5e5e388a02c6f9841295f75a6b30bc8afbf56cd5810fb4738c30f5b9`
- SSIMULACRA2: `836d5814b3903dc27f166328f7d9a218c1f2270334ba3a9589165d03e0e44030`
- butteraugli3_quality: `eff87a25f6f92c8b9cefaa006766364260f3b5c2f3e271c0c029f1ea97fd97b1`

Statistics receipt: `run-heavy: done rc=0 256s | peak-RSS 1.23GiB | min-avail 32055MiB | peak-load 9.18`. Read: `rc=0 0s | peak-RSS0.07GiB | min-avail33542MiB | peak-load0.85`. Pre-open commit `85a08787d7d3` was verified remotely before parse. Target cache `d22ae1e1cd492864c264b6830671d9219c0271e0099518452c157e1460267a5c`; result `51fb82ce177212804149072488ba604954f4264d516455bcc9f5d99dd627cf43`; completed journal `ff4d4858c6aef4043fe69f82ab3a63ae82443243eb9562a2ea66fc3cdd4ec724`. All earlier failed serving/control artifacts preserved; no second raw-label open.

## AIC4 and SDR25 — frozen-A read spent

Distortion-oriented q_jnd; quality metric signs retained; butteraugli quality=−p3. SDR25 is the50-row JPEG-AI subset of AIC4, not independent. Prior zensim read: DATA_SPLITS lines1613–1617; later JND study lines2886–2889. 95% whole-reference percentile intervals,10,000 valid draws per statistic, seed20261105. Only5 reference groups; intervals are descriptive with limited source coverage.

### all: 300 pairs /5 references

| Metric | Signed SROCC | KROCC | PLCC4PL | RMSE (q_jnd) | Within-reference SROCC |
|---|---|---|---|---|---|
| A | -0.921776464 [-0.961732777, -0.892515873] | -0.755897436 [-0.830807065, -0.710320125] | -0.910638304 [-0.959857060, -0.891894102] | 0.386207256 [0.245807099, 0.448532622] | -0.939560989 [-0.967924423, -0.904706863] |
| SSIMULACRA2 | -0.912548806 [-0.959555734, -0.877435814] | -0.745507246 [-0.835328402, -0.693583725] | -0.899948565 [-0.958510363, -0.873821943] | 0.407509811 [0.240421312, 0.487357708] | -0.940727980 [-0.969958322, -0.908385663] |
| butteraugli3_quality | -0.896924188 [-0.965041346, -0.839286818] | -0.726376812 [-0.845220506, -0.653610552] | -0.883370624 [-0.955630164, -0.841214621] | 0.438053705 [0.252501529, 0.490460035] | -0.937171437 [-0.963200889, -0.895248680] |

Result SHA256 `6e8304cc63f71a14cbc1d4e0c022955f9557c9e2e0548450b2778dbd40693dc1`.

### SDR25: 50 pairs /5 references

| Metric | Signed SROCC | KROCC | PLCC4PL | RMSE (q_jnd) | Within-reference SROCC |
|---|---|---|---|---|---|
| A | -0.855078031 [-0.991147462, -0.739966354] | -0.732244898 [-0.949367089, -0.611618257] | -0.894268530 [-0.989661933, -0.835314319] | 0.331906627 [0.095107431, 0.406639013] | -1.000000000 [-1.000000000, -1.000000000] |
| SSIMULACRA2 | -0.841344538 [-0.989891697, -0.706615348] | -0.702857143 [-0.931330472, -0.581742739] | -0.882005799 [-0.982791365, -0.814631236] | 0.349489255 [0.125999723, 0.428981303] | -1.000000000 [-1.000000000, -1.000000000] |
| butteraugli3_quality | -0.841632653 [-0.991705669, -0.708303249] | -0.712653061 [-0.951931330, -0.594770404] | -0.887265545 [-0.986331112, -0.817885716] | 0.342088014 [0.108171851, 0.414872472] | -1.000000000 [-1.000000000, -1.000000000] |

Result SHA256 `bc6564cfd50b307bd2fd2c578351b4e7a4d765ea18c067cddc49feeb28de1ade`.

All305 stimuli independently verified and all610 feature RGB checks passed. Frozen pinned target file SHA256 `955a9601e94c0877a00c822dbbbd17ad4ae11544dd6d550b9ed79d1ad9d983ff` differs bytewise from library SHA256 `a64cabbdf47bc77c3692267e30c7db55ea6197114697e33c3dbb81c887bbf498`; pinned file governs. Pre-open journal `97e947cf9d986798d3c98bb1ce629a7f8adbd7114ba36cad2c9b2a6febcdf5dd` verified remotely in `11527604aa32` before parse.

- keys: `run-heavy: done rc=0 1s | peak-RSS 0.06GiB | min-avail 33498MiB | peak-load 8.39`
- decode: `run-heavy: done rc=0 4s | peak-RSS 0.07GiB | min-avail 33471MiB | peak-load 4.85`
- bank: `run-heavy: done rc=0 11s | peak-RSS 0.19GiB | min-avail 33383MiB | peak-load 4.54`
- predict: `run-heavy: done rc=0 0s | peak-RSS 0.15GiB | min-avail 33549MiB | peak-load 3.83`
- peers: `run-heavy: done rc=0 49s | peak-RSS 0.19GiB | min-avail 33224MiB | peak-load 3.83`
- ledger: `run-heavy: done rc=0 1s | peak-RSS 0.06GiB | min-avail 33461MiB | peak-load 2.39`
- read: `run-heavy: done rc=0 0s | peak-RSS 0.06GiB | min-avail 33590MiB | peak-load 1.75`
- stats: `run-heavy: done rc=0 23s | peak-RSS 0.18GiB | min-avail 33342MiB | peak-load 3.69`

Target cache `06ec295c3af79c12e86625b053572e928f8aab4b9a99bb2e242ddac3badf10fc`; completed journal `d2885fc4ffe3403c482c7ffc438a7c242f783e745456e7aff1e81b3d29ce6611`.

## Durable study location and approved temporary-PNG effort — before MCL labels

Owner instruction: do not keep needed artifacts in temporary storage. The full study was atomically moved onto durable root-volume storage at `/var/lib/zensim-final-read-a-20261105`; `/var/tmp/final-read-a-20261105` is only a compatibility symlink for immutable earlier receipts. Reusable source is committed in this repository. The frozen extractor is now also retained under the durable study's `bin/`, SHA unchanged (`ed495741…`). The final tower mirror remains `/mnt/tower/output/zensim/final-read-a-2026-11-05/`.

Coordinator approved switching only temporary lossless PNG output compression from zenpng0.1.4 Balanced effort13 to Fastest effort1. JPEG decode remains zenjpeg `762fa712`; pixel conversion, models, populations and statistics are unchanged. New decoder SHA256 `1813afa0c45b469021775dfd9933735ce1c10ae0db35b27112e2c0b9da4a9e39`; old decoder SHA256 `47dc7d599febbee982e83545528494abc119b7f2de0f41999e111047097a213b` retained outside Cargo output. The earlier MCL decode was interrupted only after preserving580 completed records and every completed PNG; all artifacts retained. Existing pixel-hash-named PNGs are reused; remaining files use lossless Fastest compression. No MCL subject chain or target parsed.

Independent Pillow readback of the same1920×1080 BMP and JPEG through both encodings matches RGB bytes exactly; input file SHAs, geometry and RGB hashes match the previous tool. All previous580 records will be checked against the resumed map, followed by the full independent readback. Native decoder tests2/2 and strict standalone clippy pass. Build receipt `rc=0 0s | peak-RSS0.18GiB | min-avail33167MiB | peak-load0.54`; test/clippy receipt `rc=0 0s | peak-RSS0.16GiB | min-avail33064MiB | peak-load0.61`; real-input lossless control `rc=0 0s | peak-RSS0.04GiB | min-avail33056MiB | peak-load0.56`. No runtime benchmark or speedup estimate is claimed.

Durable location receipt SHA256 `41268403b87a5d343b37d736fdd1839d199c1cbeee348d20a8e7588592359f95`; Fastest parity receipt SHA256 `7959965b55652fae06ef867ca5ad75bf42b68114c84fa8327d90fc8164c24ef4`; relocated MODELS.json SHA256 `f566d1412d617a1535c0e58ded4b2bfdb9a99f1bab0b7170b94bb808d4ee366c`. Current model paths point to durable storage; original artifact SHAs are unchanged.

## MCL-JCI final pre-open pin — no subject-chain target parsed

All5,000 pairs /50 sources and5,050 decoded stimuli retained; all10,000 feature RGB hashes and all5,050 independent readbacks passed. All580 records from before the approved lossless PNG effort change match exactly after durable-prefix remapping. Feature extraction remained serial, one process. Reference metrics use the existing R10 frozen binary at8 outer threads under24GiB: all64 uniformly spaced label-free control pairs match1-thread emitted SSIM2/BA scores exactly, as do all600 preserved serial-prefix rows against the full5,000-row output. No metric or statistical arithmetic changed.

Receipts: staged5,202 files /1,716,532,646 bytes, every file verified, dataset SHA256SUMS `2edb91675c9155937553e11ff138125d13cea873d0bdb569ae273de8355686ad`; stage `rc=0 86s | peak-RSS0.07GiB | min-avail32585MiB | peak-load4.03`. Resumed decode/readback `rc=0 347s | peak-RSS0.10GiB | min-avail32555MiB | peak-load2.00`; feature bank `rc=0 621s | peak-RSS0.45GiB | min-avail46002MiB | peak-load2.88`; A prediction `rc=0 0s | peak-RSS0.25GiB | min-avail46566MiB | peak-load2.47`; peers including controls `rc=0 320s | peak-RSS6.49GiB | min-avail38702MiB | peak-load9.44`. Interrupted serial/decode artifacts are retained; no label read occurred in those attempts.

Canonical keys SHA256 `4b3dbf770bd0eaf00ef49d560f175565de08d3470dead9c7189e83f71402033b`.

Decode map SHA256 `31ff9311b398632e79459011b19c5a5b080668b53bf46abc11b1f05a9d39161b`.

Prefix parity SHA256 `789c8a6a4409794dd34962475aea8510274dd39ab77fbc24b4ad49c957d95aee`.

Thread parity SHA256 `8dea93ef6d6c7597532ff5e563d6060ae903155ab14a68dd7f1b526b3c063dac`.

Serial-prefix parity SHA256 `a338cccfcf0c06abd7ebbd14c7767dfe48e648616df2a58319d53d00a41b1d61`.

Pre-open journal SHA256 `afdfc753c574988a3e3122df000a3f27e7a24ef81ac63b2703d37b5e5d3e5a59`.

Raw-chain pins (raw bytes hashed, target values not parsed):

| File | SHA256 |
|---|---|
| JND_image_01.txt | `ecef89e08312c6cfba21d53503688092d34347bf15d3c968267f56b6143bbb97` |
| JND_image_02.txt | `0ea2c68fa3e4c08010ec02657fea23d55ff16f7dfb81a9af214c5167bfeeca2e` |
| JND_image_03.txt | `38b685ccb1498b587b2bf67987e47f27b0d1299ac1e6c3337af822588d329775` |
| JND_image_04.txt | `6d5d43344062a139892f425729cba93a49e59f343678977dfa7e830b68d0f937` |
| JND_image_05.txt | `d7f01fca3ebe911409db3956fa833ae980a8054a1a20ea7f9409f8441057101b` |
| JND_image_06.txt | `229ee1a4c7de09d62d949a7c04434d8aa9b2787444e39875457c83fb395c2def` |
| JND_image_07.txt | `ee07a8e75bf98347fbc8c0ec34a14b8ae4dcd2d02c8ac9bc93abfe46d9021cd4` |
| JND_image_08.txt | `d98a25438feb7c7a2d223bf6751cec9c81691138ed130745264cfc5e83055b94` |
| JND_image_09.txt | `db45ee865876a854cd6d1b732598bf432e0931e9d0d4bd8d534b1a6f8bd3564d` |
| JND_image_10.txt | `15ca5d3336a8eb105a81fb6dbff627f4edce3edf58a9183b03a4ea97f909f1eb` |
| JND_image_11.txt | `7b1d6300d52fd49bab2db2dd78cd760e4620d47d398d7a59e4face770979c9ee` |
| JND_image_12.txt | `1ce985134044f35168a48439f999438150e53a9b0ccce43a781a7b4c322992fe` |
| JND_image_13.txt | `35d425cc0b92c2888668d77eb514a4b78449ab097b18501ead4b0c0eac2bd766` |
| JND_image_14.txt | `4e9dae3bea4329b2003917c5ff96bed6ec4e6ee207cf15dff7eaa8b740aca218` |
| JND_image_15.txt | `52f80c4e34d9dfd2ec0e3dbfefef3abbd6cd39bdeb9fc3f5041b5084e95661c2` |
| JND_image_16.txt | `683a8515c6f6fe68068c51d706c478ba1cbd6171e21db14a6ec10922df005267` |
| JND_image_17.txt | `465602ba34f762ccda38e6303204c30eb665075ce09c3b6cf9a6f58ac2c4ecba` |
| JND_image_18.txt | `ab73baf4a1370fe2d85daca1ddc2c8c6d7d401abd33a16f866aaff740cd2a075` |
| JND_image_19.txt | `da9e6df2d781e9ee3f40d656d2b139eaca284be7eedba2980405f7f025e8e2c9` |
| JND_image_20.txt | `b8432ca1517b1e108b4ea0ea76477c8b53ab348b65d009301896b4e3405a8b94` |
| JND_image_21.txt | `46a212d8e3b1165e712fbe9aeaff7f829262cb6f2ea3190f277a703237e8a963` |
| JND_image_22.txt | `f9658ed8fd09ea1de18afd0afeecdf2c7d85f4cb63f6d78f73e33a5903b28008` |
| JND_image_23.txt | `caf447fa35d909c0e7682649be4177ecd0cacfda36d8e2a201d6fba5863e674b` |
| JND_image_24.txt | `42f3f6eb9aef7a02b892ec6250071890170f5ce0c6ad2af33f14f1bb5fab9253` |
| JND_image_25.txt | `c72b4fd9818f579ee2160e38ac0f753f8c4768bf9bac0aac12d79cbbc89e819e` |
| JND_image_26.txt | `c89fbcc28e4122099a166b7f168621cb1d3ad4320f147f9f07b942abeea90023` |
| JND_image_27.txt | `bff3b848169ce109926a8c98711f59aff5dd5abd87dd76be78af58bef82720c8` |
| JND_image_28.txt | `b3b651881217ab99618c3430c2ef39aa5ac78889a6ba65d3e8a4bfd6d82c6076` |
| JND_image_29.txt | `01268faf2cc6113d0401d5c72d487f0992456caef624d5edcbc03ff23bc8ef8a` |
| JND_image_30.txt | `f4c7e60cd798484bcb50266063ef55f909a1e81df55ad7b927cf4e4690688883` |
| JND_image_31.txt | `30adda37b7060c84c06367f31c1199d9006d5c74949c64ba676232bdda147379` |
| JND_image_32.txt | `2e2b16d0e3655e04ef360638231d76285eed4bb2a4de3d3ed9281638a4764633` |
| JND_image_33.txt | `770936e7df193b2c2fa9927dbe489f6e5170b3f4116651b724d234b0ffb60057` |
| JND_image_34.txt | `fde2663c9af90980e7e9b80d88b1eb1a65f38be1a10eec93c21e7d611574c633` |
| JND_image_35.txt | `eaf5f79c0b843b15298aa21f8acf30dd3fe4f1ef1ad7dfc0bb182cf539e01cca` |
| JND_image_36.txt | `0c313f45221becab8176c7b570871978e00dc20d3cf7801e3ba7ee909ff3fc94` |
| JND_image_37.txt | `d9eb279614c00555b60fc29c6d9dbd6503bfabf75e1073ae192c2c13b17b6e2f` |
| JND_image_38.txt | `aa62177a62e014ffac220d7bceb913147c792edf9389d95d4ae50b934ba854c9` |
| JND_image_39.txt | `f474ce5535266562582ab36655b6cef17e5d0b8564d2e7ff9f2653e34c3d89c4` |
| JND_image_40.txt | `d1f40cbbe6ed33669b023a34dfeaa723ba83fc7b027aec203bc2648568aa60dd` |
| JND_image_41.txt | `e412f999cc4ab92003a7356faf535f5915cad7c247725a00e2f0c9e761dee854` |
| JND_image_42.txt | `d45250ec06a6f7bf45907c4fd35c50110978d617b5762298a92005a097019a17` |
| JND_image_43.txt | `2f47c91ae1f1b2e6bb60a729e35d60c85c02a25a70bee28094e2222b0b3f8a6e` |
| JND_image_44.txt | `5e7796bb775a269fa2f1fef8341ea979acdc71b9cf3d93ba41eba3b4f230b7af` |
| JND_image_45.txt | `32a8d73acce001445e197735c38d78250e0d84fe100fa67311d51e7d3ee24818` |
| JND_image_46.txt | `98c0173d1aca015e9fd782b214d544000f3e052ff43ff5a76bd586b5815b6add` |
| JND_image_47.txt | `c03b71ef93c0402420629eaabcbcf46334149a56e95ba1d3c575d911d8cd942e` |
| JND_image_48.txt | `2f0eb20bb9a76091da27ad3ae74064a57e55736fda82b5af8f79ed8822c278df` |
| JND_image_49.txt | `1fb2457557486137e4235857cc9f106b0fc2a8ecdd88101938edade986d00685` |
| JND_image_50.txt | `747b392eecea1787a306e73e0961cff575677851e77147634738222c0961341d` |

The registered construction is mean count of chain points QF≥q across30 subjects, q1..100, with q100 target0 by definition. Full50 and fixed clean40 are reported. Excluded clean40 sources:1,3,6,7,9,17,28,31,45,50. The lost original CSV SHA256 `36f17dd8bac2d085f718e388fd5db307d887338c9c2b296349bb2ef20f184802` cannot be byte-verified against the reconstruction. No labels, weights or subset membership were selected on these scores.

## Governing versus library files: key-only comparison

No additional target column was parsed. Reference filenames identify the approved references; candidate keys retain the codec parent for CID22/LIVE to avoid basename aliases, and use the collision-free filenames for the other sets. Every governing row has a unique projected pair key. All files differ bytewise in schema/path representation; numeric library targets were not compared or substituted.

| Set | Library total rows | Library rows on approved refs | Governing rows | Common pixel keys | Governing only | Library only |
|---|---:|---:|---:|---:|---:|---:|
| cid22_b | 4341 | 2124 | 2100 | 2100 | 0 | 24 |
| csiq | 866 | 866 | 866 | 866 | 0 | 0 |
| live | 779 | 779 | 779 | 779 | 0 | 0 |
| konjnd_terminal | 1008 | 100 | 100 | 0 | 100 | 100 |
| aic4 | 300 | 300 | 300 | 300 | 0 | 0 |

CID22 library adds24 reference/self rows on these references; the governing2,100 distorted pairs are all present. KonJND library entries pair each source with itself (threshold records), while the governing file pairs the source with the coded threshold JPEG. The pinned100 coded pairs, including SRC0437 JPEG058, govern. CSIQ, LIVE and AIC4 projected pixel-key populations match completely; their original target conventions and scales govern. MCL library243 threshold records remain distinct from the coordinator-approved5,000-row reconstruction.

Key-comparison receipt SHA256 `8bb3467fc4f7d456703c8bd07d3afd4a606d5c8530ebddc5c15abe4dbc877f51`.

## MCL-JCI results — full50 and fixed R7a clean40, read spent

Distortion-oriented mean JND counts; quality scores retain their negative correlations; butteraugli quality=−p3. Every cell has10,000 valid whole-reference draws, seed20261105,95% percentile intervals. Within-reference SROCC defined for every source in each panel. Prior R7/R7a exposure: DATA_SPLITS lines1601–1617; later third-party label studies lines2867–2899. Full50 includes KonFiG training-source overlaps; clean40 excludes1,3,6,7,9,17,28,31,45,50, fixed before this read.

### all: 5000 pairs /50 sources

| Metric | Signed SROCC | KROCC | PLCC4PL | RMSE (JND count) | Within-reference SROCC |
|---|---|---|---|---|---|
| A | -0.896932387 [-0.914725059, -0.878509286] | -0.756792691 [-0.784911078, -0.730030419] | -0.927140774 [-0.953719618, -0.902912019] | 0.594454193 [0.451273061, 0.712595707] | -0.933867048 [-0.944413949, -0.923026856] |
| SSIMULACRA2 | -0.884864711 [-0.906228055, -0.862671564] | -0.739921139 [-0.773287783, -0.710009652] | -0.916680294 [-0.949120950, -0.887649519] | 0.633969112 [0.472066047, 0.769775762] | -0.933821645 [-0.944384537, -0.923009973] |
| butteraugli3_quality | -0.850526852 [-0.885806843, -0.814714611] | -0.697018155 [-0.742589342, -0.655656259] | -0.896875551 [-0.932629994, -0.861288320] | 0.701648558 [0.542900212, 0.841639468] | -0.933647916 [-0.944281491, -0.922797792] |

RESULTS.json SHA256 `92c24cd8c5327043790c4a71f6de33e1f0ed46aac93c24ecdee57eec18409c6b`.

### R7a_clean40: 4000 pairs /40 sources

| Metric | Signed SROCC | KROCC | PLCC4PL | RMSE (JND count) | Within-reference SROCC |
|---|---|---|---|---|---|
| A | -0.904456237 [-0.921112372, -0.886363586] | -0.768631575 [-0.795093528, -0.741298705] | -0.929185931 [-0.959368633, -0.901898714] | 0.587412021 [0.417331234, 0.720339628] | -0.931893492 [-0.943422400, -0.920078666] |
| SSIMULACRA2 | -0.890712810 [-0.912269204, -0.867320796] | -0.749285011 [-0.784230052, -0.717230935] | -0.916786016 [-0.955874442, -0.882832455] | 0.634718251 [0.433415739, 0.788568999] | -0.931839621 [-0.943325555, -0.920044075] |
| butteraugli3_quality | -0.855604715 [-0.892608889, -0.815794403] | -0.702685420 [-0.752000445, -0.656116781] | -0.896019620 [-0.938157495, -0.854432631] | 0.705656142 [0.518376013, 0.867636989] | -0.931655515 [-0.943247060, -0.919788154] |

RESULTS.json SHA256 `a249f97b2ce7ac201ce8a5a73510c891bc42f659f4e1f3e3db155720e3b43782`.

Pre-open commit `fd75539f6b4e` was verified remotely before the single subject-chain parse. Reconstructed table SHA256 `8d9568e9b897b82106843b88fa59c59c14f9f11ea080e084a827cf63db17be1e`; target cache `371e5af9d441270929859fa8f2d7c9c9b144910e852107c3b287fa34170a7b34`; completed journal `76eae51230975beb02c97232a8515836ab73d9f4091fad883f35d6803f7b8769`. The QF100 target0 is an anchor definition, not pixel identity with the pristine BMP. Byte identity with the lost historical table cannot be verified. Read receipt `rc=0 1s | peak-RSS0.07GiB | min-avail46261MiB | peak-load1.72`; full50+clean40 statistics `run-heavy: done rc=0 871s | peak-RSS 1.95GiB | min-avail 43817MiB | peak-load 9.00`. No second raw-chain parse, tuning or model selection.

The approved CID22-B population-key input is now retained in durable study `pins/CID22_B_APPROVED_POPULATION_KEYS.parquet`, SHA256 `fa1fce412e8a4ffabc43e809216e3eaf0d4114b04307643cfd517e59d959b003`, exactly matching the input pinned before the read. The key-generation owner uses this retained copy; it no longer needs the original temporary bank. The frozen extractor and every scorer/bake are already retained in the durable study. The synthetic-control panel alias is relative and portable; its earlier alias is preserved. No label target was parsed or prediction changed by this preservation.

Coordinator source documents are retained under durable study `pins/`: `COORDINATOR_BRIEF.md` (5,520 bytes, SHA256 `5fd5388ed6a33a402dc5f24f675dafd188bfa875faab4d4f2323f63b0aec3242`) and `COORDINATOR_REGISTRATION_DRAFT.md` (7,422 bytes, SHA256 `52fc61f1b4d06328e8b505dfb1049aec8e633fd052cbf8cb6a73b8d43f06c694`), both byte-identical to the supplied originals. `just final-read-progress <root>` records live process arguments and artifact-log observations in `LIVE_OBSERVATION.json`, separately from the stage owner's progress record. No preference values are opened by supervision.

## CLIC 2022 staging and key pin — before preferences

The single sequential rsync completed all170,537 content files,145,439,278,184 bytes. Every file matched the dataset SHA256SUMS (manifest SHA256 `317f69a49a012a613534bef5cd3900be5c72eeaa4fd877732de8ae497e49907c`) before native decoding. Staging receipt: `run-heavy: done rc=0 3928s | peak-RSS 0.15GiB | min-avail 43944MiB | peak-load 8.79`.

All57,300 questions over57,243 reference crops are retained:114,593 unique reference–candidate pairs and170,533 required stimuli. The four other manifest files are preserved and verified, but unused by those questions; exact names are in durable `inputs/clic22/MANIFEST_KEY_COVERAGE.json`. Key-table SHA256 `b6a4f07075b328a63fae7557001e3ba3e30e6b706b4d94b876c8923ded60435f`; key receipt `rc=0 2s | peak-RSS0.42GiB | min-avail44268MiB | peak-load1.33`. No preference values parsed.

The registered label file still has SHA256 `03300f61963c2fc96e5daf3cad3a0e8a43438637959875b904d7a944db295db1`. Before key generation, the owner now enforces that fixed digest instead of accepting a newly computed digest and correctly names the registered target as majority(n_a,n_b), tied majority0.5. No numerical or population rule changes. Orchestration SHA256 `5a93871a8911266b5f762d78c1a342dfc7dcfefcf168471bdfc6aef7a787337f`, source commit `ecdaf24544f1`.

Final script lint:1575 scripts checked,all runnable; `rc=0 16s | peak-RSS0.03GiB | min-avail44261MiB | peak-load1.57`. All5 synthetic parser controls pass,including every tied-majority CLIC question retained and refusal of a second open after a failed parse; `rc=0 0s | peak-RSS0.07GiB | min-avail44361MiB | peak-load1.44`. The receipt audit confirms all7 completed populations' target-cache hashes and result row counts match their spent read journals, with zero raw-label reopens; durable `pins/COMPLETED_RESULT_RECEIPT_AUDIT.json` SHA256 `69c26e2e6ab4e5f7c71791ae1bf7ee0a98aedb060f7f52a5049c49c78d4456f4`.

### CLIC protocol clarification requested — before labels

The coordinator draft's “agreement with the majority choice, as the CLIC2021 read” combines two different rules. The final-read parser and pre-read pin implement literal majority(n_a,n_b), with tied majority0.5. The historical CLIC owner `scripts/byconstruction/e36_exp56/clic.py:126–130` instead uses `(n_a+0.5*n_tie)/(n_a+n_b+n_tie)` and soft-vote agreement via the canonical forced-choice owner. They need not yield the same agreement. No CLIC preference values parsed and no CLIC LABEL_READS.json exists. The coordinator must resolve this before the target read: retain majority and disclose the historical difference (recommended), amend to historical soft votes, or explicitly register both with majority primary. All57,300 questions,57,243 reference crops,10,000 whole-reference bootstrap draws and seed20261105 remain fixed. Label-free pixel verification continues while the target read is held. Durable question/evidence: `pins/CLIC_PROTOCOL_HOLD.json`.

### Coordinator resolution — both CLIC agreements registered before labels

The coordinator answers: “Report both; majority is the registered primary.” Primary: majority n_a versus n_b, tied majority or prediction receives half credit. Supplement: historical soft-vote agreement with target `(n_a+0.5*n_tie)/(n_a+n_b+n_tie)`. All57,300 questions over57,243 reference crops remain; each result uses10,000 whole-reference draws,seed20261105,type7 percentile95% intervals. The single target parse retains both targets and tie-only questions (majority and soft target0.5); no second label open. The original label-free key table/population stays unchanged. The per-read ledger receipt additionally names both agreement rules and vote columns. Durable resolution: `pins/CLIC_PROTOCOL_RESOLUTION.json`. The earlier CLIC2021 number is a soft-vote comparison; the majority primary is explicitly distinguished from it. No CLIC preference values parsed at this amendment.

CLIC native decode and independent Pillow12.3.0 readback completed all170,533 source stimuli/scoring PNGs, every geometry and RGB hash matched. Decode-map SHA256 `44144bd38c83a0e366c5f15ffea6e86e445d019714742e4566f8c72ac2c98b83`; receipt `run-heavy: done rc=0 2565s | peak-RSS 0.48GiB | min-avail 43571MiB | peak-load 4.39`. Feature extraction and the one-time preference read remain pending.

The approved CLIC dual-target owner passes all5 synthetic parser controls, including majority/soft values and tie-only questions in one retained parse. Receipt `rc=0 0s | peak-RSS0.07GiB | min-avail45910MiB | peak-load3.12`. Updated orchestration SHA256 `e6831e067b9313b564abd869e3b72122c3938d68b775d21dd0f011a3428b89e0`. Label-free serial feature extraction has started; no CLIC preference values parsed.

Coordinator execution update: public zensim main now contains the merged0.3.0 development README and `docs/MODEL_A_REPORT.md`. Fetch/rebase before publishing the completed CLIC update to this final-read doc, report section6.5, the known-limits line if needed, and README pending sentence. The durable study stays at `/var/lib/zensim-final-read-a-20261105`; notify the coordinator when it is complete for the coordinator's tower mirror and move. Mirror verification remains pending. This transfers archival execution to the coordinator; scientific pins and required checksum verification remain unchanged.
