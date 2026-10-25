# Zensim Dataset & Experiment History — accurate record

**October 25, confidence side tables:** Every key, role, membership state, family ID
and row order is equal to the original 21-collection admission, across 498,216 rows.
Published confidence metadata matches 2,444 rows; missing data stay explicit. The
source flags do not amend membership or authorize a fit. No source quality target or
MCL-JCI rating payload was read. The [confidence pointer](../benchmarks/dataset_confidence_2026-10-25.pointer.md)
binds every output and the independent identity/role verification. New subset source
admission is a separate operation; IMI remains deferred.

**October 25, complete provided-library registration receipts and Rev7 features:**
The following table repeats the canonical [membership registration](DATA_SPLITS.md#october-25--provided-library-membership-registration-labels-unopened) so this history directly binds every collection's roles, counts and receipts. Membership SHA-256 binds `<id>/membership.parquet`; receipt SHA-256 binds `<id>/SHA256SUMS`, under `/mnt/v/zensim/dataset-admission-20261025/` on dev. These are frozen identifier/role memberships, without subjective targets.

| Collection | FROZEN roles / membership rows | EXCLUDED roles / rows | Membership SHA-256 | Receipt SHA-256 |
| --- | --- | --- | --- | --- |
| bapps | TRAIN 146003; EVAL 36344 | TRAIN 5397 | `96e5acc0d01c412d81e9a5259df5a6af903550a2d85138cc3b45fd6f75f9b669` | `7802eafc860c507fa065e010ea17640496ed34f2b82381e296fc586e7164cbd4` |
| cid-iq | EVAL 210; TRAIN 330; TEST 150 | 0 | `e8f1b6dadf7c6c72ffc8ce11cfde8de4f077e1b1daa224b5b8714c54998d90c7` | `e53c6de0d5757fa1d418c53556f8cbba0eaf413359aa10090131d6707741de0e` |
| cid-oa | EVAL 91; TRAIN 143; TEST 65 | 0 | `d179d060bd9d7f1df64da48ecdefae005367c9437ca6a0196d2fd22e4c53fcb2` | `f4e4881a1c0c30738ab08cb99c38aaaab860484ed7c1ca960f0ac02d0e6b297b` |
| cid-pps | TEST 12; TRAIN 12; EVAL 6 | 0 | `51ad10649359d7cc3fed42e9c97c374349c4d9b7b2787a089d8ab38f5c86a7b6` | `e393ec09455c13fbe89264e27b40272dcf323117f6a474e1ec212490fde96521` |
| colourlab-chroma-masking | TEST 192; EVAL 288 | CALIBRATION 480 | `71a01b5081e97c5a4e7100b967d0d5408aa35559ebaacf9fd3e4d831bac35f93` | `09c50d740349efb0100a07c9c7ea0ae8e259f0d3e8be3edf3826f82325014fb3` |
| colourlab-rating-variability-2022 | TRAIN 320; TEST 128; EVAL 192 | 0 | `58f741e9a6fd1d49b569768f93a2450b369d2f0eaf6b4d0ad9fba65d3b1cc91d` | `8fabe17c0dc33dfcd36e31387e16175665e00df7868431637b9cc5b95a2e0cdd` |
| diffiqa | TRAIN 7272; EVAL 17916; TEST 35844 | TRAIN 118176 | `09132713d1410b4b989f95caceec8d5e516cd88332480ae252140d51dbedc57d` | `66df1f3bbd9de640a3f605e3feb4306031f244544ca1ce228cff8aa9883eb591` |
| fg-iqa | TRAIN 900; TEST 360; EVAL 540 | 0 | `f99b45d5d959bea80bd9dea06ad6f846c8df3dea218ea4efd08b54f28f0af5be` | `857ac9d309c7415f98f9646ae96710618f61d024222408cf3dfb8a67f04a662d` |
| ie-bench-db | TRAIN 523; TEST 1000 | TRAIN 2461 | `aeb82b6431a57ae2ad14d01a5f51663e6893ce89d341a7527efbdbbffc4c9943` | `be659703c3977725cf361d7678c08878b4ab180eca87cb8bfb0e9834c19e90d6` |
| iqa-odi | TEST 191; EVAL 48; TRAIN 72 | TRAIN 648; TEST 1 | `42082bdc79764a1f93097b868c65c80d0d9d85bbc800360e419586ae04ff62df` | `047c0e5d1ccdbc3427655fd9b1d20fc712850524c7626add10df271b59fd7d76` |
| irccyn-ivc | EVAL 60; TRAIN 105; TEST 45 | 0 | `c7f3fc9e910563cd91ddd753addf00f2e1661c613a8e88cf499d1f12cc9fa2a5` | `618af1cd5633266dca70f80bc0a41b0e987f753cf0d1aea026b48a1a60c838b2` |
| live-md | EVAL 180; TRAIN 210; TEST 60 | 0 | `839ccb8387783baa44b12919ebf1dbfeebc5fdf925e310ce7a23748a70f393a8` | `451a813d75d27a111661a5c79a6d3b88b83f7a158f1af6c38653048a2bc5843b` |
| locvis | none | TRAIN 451; TEST 106 | `e7ceba92684c78721faa496ceb8549bdf0750dbe18e23cda9981f68e745dafba` | `cb9ee78d49f16371cfb62d584afc2f69b2f18d63fe4b9ef0f93477611e46ba20` |
| locvisvc | TRAIN 337; TEST 119; EVAL 30 | TRAIN 401; EVAL 28; TEST 52 | `fd1d3791d96643bd0c2820b8480b2e44128e72b63570cb9368a31720765bb5dd` | `7d2114fd4665ace4bc6962f54a33f2a46bd1cd89ab1a42f8471d6d7b4a3d2337` |
| mdid | TRAIN 800; EVAL 480; TEST 320 | 0 | `6b8c1e5744e9e5b078463898a3d71d72e65175cca4baad5604403fe574739710` | `5db038d67b026f1805420da7b188a880563fc0b81a2aca15eab73f8645423ff8` |
| nits-iqa | EVAL 405 | 0 | `58b0dada3dc29fdbd06fdd624aaf0a585ed352e79e295dc9a63f49ff61c4c9c5` | `3de846f7681e234a803145f295c528ee09621c618cff9d2ca9913ba18a48756f` |
| pieapp | TRAIN 51000; EVAL 7260; TEST 4800 | 0 | `1d8e0e7d4f18cfeb345b14b60fbdc0922458f6c1c97823bc7ee423ebb46d6b99` | `9cb3ab1d1c54d22976babc3b17743fe579bb373e9ee45b78b9803944e33264b7` |
| pipal | TRAIN 23200 | 0 | `d79e5ae3ebfa8412a3fe9c30fba4c314c59b051ef941fb3a03afd6f3b0df595e` | `c094ea4d8e8c60cc1b29bb136b5de8c27fb979907a781822652ed4f675df252f` |
| spcd | CALIBRATION 9947; SOURCE_EXPOSED_TEST_DIAGNOSTIC 20000 | CALIBRATION 53 | `fb893c96fca33e6dff1ef9c30f30b6d5543d025bf621dfb0ee7d2d13b0c3483b` | `a27316dfd282ea4e29c9494d04482f81861fcc407268325b9a0e389feb4c1a7b` |
| vcl-fer | EVAL 192; TRAIN 264; TEST 96 | 0 | `1a412c4d0c525a87126ac78fcd543f2c8d4fbeb9c1ee0cd3eec684e7f6c78d2b` | `9a958598a710e5fdf2c639480ac83538f2aef1aee6bca5a0a8862c695f375d37` |

Historical accepted validation receipt: `a4013c0bb1a72b5b4990cc78eb7003acb0e87e2374a01c7d5c8a47bf1d82f9d8`; frozen source-family registry: `9c5ba4733a2935931b0567b3c7b72fdce6deff730ea9b49afaa6594ef81ac65f`. The historical receipt includes LS2N IPI SR, which the owner excludes from phase 2 together with ESPL because both were acquired after the library was provided. Their files remain untouched. The holdings/acquisition-log audit found no other successful post-library collection acquisition. SCID, SIQAD, QACS and JPEG-AI EPFL are closed as not in the library, with no retry. No dataset downloads, probes or web searches followed the correction.

Protected CID22 gold, CIQA, MCL-JCI/D3, AIC, KonJND JPEG, CSIQ, LIVE and CLIC 2022 keep their roles. NITS remains T0 EVAL-only; CLIC 2024 remains source-exposed diagnostic. SPCD stays CALIBRATION / SOURCE_EXPOSED_TEST_DIAGNOSTIC. LocVis duplicates, teacher/automatic maps, temporal flicker, inherited held-out families, ambiguous IQA-ODI ID233 and Colourlab chroma's unresolved CSIQ-parent calibration rows remain excluded. The registration retains all per-row exclusion gates; no source role is rewritten.

The pinned Rev7 assessment owner delivered 417,060 distinct file pairs across 16 supported banks, covering all required sides for 278,053 of 369,062 FROZEN memberships. DiffIQA's 61,032 rows and SPCD's 29,947 rows skip because there is no pristine reference; all 30 CID:PPS rows fail pinned TIFF decode (TRAIN 12, EVAL 6, TEST 12), with exact native errors retained. There are zero unresolved memberships. All 375 final checksum inventory entries were verified on tower, with three random readbacks. Only owned BAPPS unpack scratch was removed afterward; originals and hash/native receipts remain intact. The [complete feature pointer](https://github.com/imazen/zensim/blob/4aec1ab6a147912c7f5dff3a79e6c33e857b025e/benchmarks/dataset_features_rev7_2026-10-25.pointer.md) gives per-role feature counts, exact build/owner/binary pins, inventory and mirror hashes, and the adjacent removal receipt.

No response table, workbook, participant payload or label column was opened; no model trained. Published illustrative documentation examples were seen during protocol reading, violating the literal no-label-values-read condition; they were not used numerically. Identifier-only disclosure binds two FROZEN TRAIN, one FROZEN EVAL and four FROZEN TEST rows, plus six already-excluded TRAIN rows; FG-IQA/PieAPP illustrations remain unresolved to a unique membership. All roles remain unchanged, and later assessment must disclose this exposure. The separate 17-population leg proposal awaits coordinator approval; this registration grants no label-access, fitting or assessment authority.


**October 25, provided-library frozen memberships; response tables unopened:**
The [registration](DATA_SPLITS.md#october-25--provided-library-membership-registration-labels-unopened) records all 20 in-scope collections, per-role counts, membership and receipt SHA-256s, protected populations and conservative fit exclusions. The accepted historical receipt includes LS2N IPI SR; the owner excludes it and ESPL from phase 2 because they were acquired afterward, leaving their files untouched. SCID, SIQAD, QACS and JPEG-AI EPFL are closed, absent from the library, never retry. No subjective response table or label column was opened and no model trained. A subsequent protocol README read exposed embedded illustrative DMOS/vote examples; those were not used numerically, and the phase-2 report records the exception. DiffIQA and SPCD have no pristine reference and skip full-reference feature extraction. Native splits and all earlier seals remain in force.


The registration also binds published documentation examples to their existing TRAIN/EVAL/TEST identifiers. Their values did not enter features or fitting; later assessment discloses the exposure and preserves all native roles.

**October 18, completed TRAIN repair screens:**
The conservative8794-reference auxiliary union has zero exact current RGB matches in all five external panels and all six admitted D1 human banks. Current raw-file hashes are retained; crop/resize and historical raw-source limits remain.
[The comparison](../benchmarks/color_noise_repair_2026-10-18.md) records all12 packed fits, exact three-seed control reproduction, the unchanged single-model choice and the separate fixed fine-chroma ensemble choice. Both choices precede new external scoring. Every assessed population retains its original role/exposure; no protected input is admitted. The first LIVE severity inversion remains in all repair compositions.

**October 18, prospective TRAIN-only color/noise repair:**
The [registered 2×2 comparison](../benchmarks/color_noise_repair_registration_2026-10-18.md)
reuses the receipt-approved four-source D1 recipe, TRAIN oracle legs and pinned
ordinal pool. Its TRAIN-development decision population is2387 rows: KADID1210,
TID480, KonFiG84, CID22-A613. Registered noise subsets contain200 KADID and140
TID rows. No new labels, source roles or feature arithmetic are introduced.
All504 read inputs are finite across all seven admitted TRAIN/development
legs; the restored chroma declaration adds82 difference inputs and82 products.
The exact original E33 native program is reused. Seed0 control checkpoint and
packed model match the original candidate byte-for-byte after native removal
of reproduction metadata. Four training and four serving smokes pass. A twelve-pair TRAIN pixel replay
across the six admitted human-bank members matches all504 consumed features
exactly at both f32 and f64 precision, with matching decoded RGB hashes.
This bounded replay does not resolve the retained extractor source commit.
Selection is fixed before any new external scoring. CSIQ/PIPAL/LIVE inherit
the original gauntlet exposure and overlap limits; LIVE motivated this repair
and is an exposed diagnostic, never selection data. Original qualification
requirements and protected-set exclusions remain unchanged. Artifact identities
and role receipts live under the registered dev output root.

**October 4, Rev5 entry/refinement engineering audit:**
[Serving report](../benchmarks/rev5_entries_2026-10-04.md) records the owner-directed
48-case KADID local-refinement panel: I01/I21/I41/I61, JPEG distortion 10 at
levels 03/05, seeds 5101/5103/5107, by_v2fy and v2basic. These are repeated
engineering diagnostics against frozen model weights; the six prior-revision
bakes are explicitly restamped for Rev5 timing/serving experiments. They are
not newly trained Rev5 models, quality qualification, or training labels.
Corpus files remain read-only under `/mnt/v/dataset/kadid10k/images`.
The CLI smoke fixture is twelve independently audited Rev5 feature vectors
with synthetic ordered targets and engineering row names, generated under
`/var/tmp/rev5/owner-tool-tests2`. It is never admitted for fitting or human
quality assessment; its Parquet uses Zstd and preserves the basic/peaks/v2
canonical IDs during densification. Source paths, formulas, binary/model/input
hashes, exact environment, and raw result locations are retained in the report.
No old feature table is relabeled, merged, or overwritten.

**October 4, Rev5 review fix receipts:**
The same twelve engineering audit vectors were regenerated under
`/var/tmp/rev5/fix-vector-audits`; supported Rev5 values are unchanged, as are
all 192 Rev1–Rev4 archived vectors. The synthetic CLI fixture is now also
retained at `/var/tmp/rev5/fix-tool-entries`. Its source/dense Parquet bytes,
verdict rows and bake bytes match the earlier engineering fixture exactly;
only the manifest's source paths change (SHA-256
`5b50065cbefe4cfce2a23c21a8ef6d2de8f3a1ae681de39889740a69055e1ab2`). Labels remain synthetic,
with no fitting, data admission or new human/secret-holdout exposure. The
serving report records fresh native/foreign, identity, refinement, historical
steering, CLI and work-census receipts. No existing data was relabeled.

**September 15, frozen recovery and corrected HDR:**
[Complete results](../benchmarks/recovery_completion_2026-09-15.md). Ten Rust
TRAIN-only fits, two fixed five-member compositions; complete full-composite,
scatter, ladder, 27-cell JPEG, bounded scalar/native JXL, native input/HDR and
strict latency assessment. Published TEST only for frozen evaluation where no
EVAL exists, historical exposure linked; secret holdouts untouched. All old
TRAIN reservations preserved. Corrected declared-primary PQ extraction has
7,425 pairs / 495 reference variants / 33 source families, all live peak
columns, fresh native CVVDP and PU-SSIM2 labels. Old mislabeled/all-zero-peak
or 8-bit-shell HDR products are not reusable under this contract. Native UPIQ
uses zenexr and fails the shared-weight hypothesis against BHdr. Model bytes
remain fixed after evaluation; neither candidate is promoted.


**September 14, complete native JXL ensembles:**
[Native evidence](../benchmarks/jxl_ensemble_native_2026-09-14.md) retains the
preceding corrected nine-family TRAIN/eight-family validation admission and all
frozen model bytes. Seven compositions complete 4,914 target cases; four refuse
unsupported spatial terms. A TRAIN-discovered discrete quantizer bug is fixed;
2,646 neutral ladder pairs are byte/pixel/score exact. Two independent judges
cover all 8,442 output pairs. No human TEST/secret access, model fitting or
EVAL-adaptive tuning. Native RD remains diagnostic and all previous product
qualification states remain unchanged.


**September 14, scalar targeting:**
[Complete-composition comparison](../benchmarks/rev3_targeting_2026-09-14.md)
uses nine TRAIN calibration families and eight original validation families.
The first attempted twelve-family calibration missed later reservations and
was stopped during evaluation; it is retained as inadmissible. The corrected
batch excludes 6068/9066/8462 without replacement, preserving the frozen models,
policies, target grids and gates. All 9,648 admitted steering cases complete;
no human TEST or secret holdout is used. Native spatial qualification remains open.

**September 14, clarified public TEST assessment:**
[Complete frozen comparison](../benchmarks/rev3_public_human_eval_2026-09-14.md)
adds CID22 gold 4,292, AIC-3 600 and AIC-4 300 under the later user ruling:
TEST can assess frozen candidates when no EVAL exists; secret holdouts remain
untouched. Original roles and exposure are recorded. All prior EVAL predictions
are unchanged. Eleven model rows now have nine panels and 6/6 composite coverage;
all nine MT913 models fail necessary ranking and dial gates. Fresh 9,593-cell
ladder features in both revisions and all 297 ensemble JPEG repair cells complete.
No training or adaptive selection. Native RD/targeting/corruption/HDR remain open.

**September14, native integrity admission:**
[Evidence](../benchmarks/native_integrity_admission_2026-09-14.md) reuses the
same214 TRAIN-fit pairs and215 files. The trainer's native scoring command
verifies explicit color/precision identities and all214 canonical f32 row
hashes. Its native input era uses Rev3/sqrt and round-trip CSV parsing.
All earlier feature CSV bytes and score/spatial fields remain unchanged.
Thirty-three identities are valid;181 nonidentities stay unresolved, with one
head activation. This diagnostic packet is rejected for fitting before feature
reads. No new fits, EVAL, protected data access or model promotion.

**September 14, complete canonical-feature audit correction:**
[Replay](../benchmarks/complete_feature_audit_2026-09-14.md) uses the same
214 admitted TRAIN-fit pairs /215 files with fresh outputs. The earlier native
ensemble's zero consumed-feature maximum was not measured; its score and
scalar/spatial parity were valid. The new complete-read-set audit actually
checks all80 ensemble IDs, and all228 base/tree IDs, with zero feature deltas.
All prior audit fields and feature CSVs remain unchanged. New scope/ID fields
distinguish these receipts. No fitting, EVAL, source relabelling or qualification.

**September 14, native SDR extraction/serving contract:**
[Results](../benchmarks/native_sdr_contract_2026-09-14.md) retain the preceding
native-color admission's214 TRAIN-fit pairs /215 files. Native372 base/head and
native944 ensemble audits complete with exact consumed-feature, cached/f32 and
spatial parity; legacy before/after CSV/audit bytes are unchanged. Source color
and native depth now reach the public score surface under opt-in audit-v2.
The TSV's historical human_score column is a row-ID join key in this diagnostic,
not a training target. No new fit, EVAL or clean-negative labels are admitted.
Original P3 encoding semantics and q85 remain unresolved; downstream manifests
must explicitly bind the new input era. No protected population is accessed.


**September 14, native color/depth inspection:**
[Native admission](../benchmarks/native_color_admission_2026-09-14.md) retains
the existing product/honest-control TRAIN-fit assignments for214 pairs and215
distinct files, including the P3 sources and unresolved q85 control. The new
native decoder report separates file metadata from decoded samples. All215
legacy RGB8 hashes reproduce after syncing AVIF/JXL decoder updates; no scores,
labels, training, protected populations or previous receipts change. None of
180 P3-derived reconstructions retains the source ICC, but original encoder
conversion remains unproven. Native precision retention is now available;
the feature-extraction caller still uses its explicit legacy RGB8 contract.

**September 14, Rev3 integrity TRAIN pilot and color correction:**
[Registration/results](../benchmarks/integrity_rev3_train_2026-09-14.md) preserve
the product family partitions. Native corruption generation completed35 of36
requested sources, refusing ICC-tagged mobile content; the alternative mobile
source refused too. Source1556/8012/8014 have33 Display-P3 renditions affecting
213 fitting pairs. The current PNG flattening path does not apply that profile.
Keep these as a documented color limitation of the earlier product packet;
numerical replay equality does not establish correct sRGB interpretation.

The explicitly incomplete sRGB pilot excludes those213 fitting pairs plus the
previously unresolved AVIF q85 pair, without changing calibration/development.
One Rev3 D228/HGB fit on11,472 distinct binary pairs passes its known-label
numerical gates, but coverage blocks EVAL. Six admission tests and exact public
Rust parity pass. All25,025 generated catalog attempts and all35,582 scored
candidate attempts remain retained; ambiguous labels are not binary negatives.
No protected source segments were accessed or reassigned.

**September 14, broad honest integrity TRAIN audit:**
[Results](../benchmarks/corruption_honest_train_2026-09-14.md) reuse all10,499
admitted product pairs and272 original TRAIN-development JXL interventions.
Frozen Rev1 D/HGB inference, unchanged threshold, fresh hashes and exact public
surface parity. Product and native source-family roles are preserved; no new
fit, calibration, encodes or protected data. All26 observed activations retain
their conservative valid labels; one q85 AVIF color cast needs investigation
before future clean-negative admission. All26 reject prepared steering despite
unchanged scalar scores; all272 native attempts remain accepted. All active
A/Bs are bound to the original pixels. No model qualifies.

**September 14, quarter-resolution chroma TRAIN fit:**
[Results](../benchmarks/chroma_scale_train_2026-09-14.md) reuse the admitted
product human/codec fitting and development tables, plus native development
only. Nine H32 fits compare y60, additional quarter B, and additional quarter
XB. Explicit fitting-group checkpoint weights and disjoint sampler windows
preserve separation; one repeat is exact. No native fitting supervision, new
extraction, sources, encodes, calibration, EVAL or TEST. Both treatments fail
registered quality continuation despite rank/own-map improvements. Full native
feature/cache parity passes. All four timing suites retain noisy-round failures.

**September 14, native filter feature comparison:**
[Results](../benchmarks/native_filter_2026-09-14.md) reuse the native phase
packet's eight TRAIN-development cells and130 translations with four neutral
60-feature Rust instruments: cascade box and direct triangle/Mitchell/
RobidouxSharp at1/2/4/8. No fitting, calibration, new sources, encodes or
protected populations. Exact box-feature and pixel/peer reproduction passes.
Direct contracts include binary16 input quantization. Focal L4 span falls,
mean span rises, and no trained-model quality improvement is established.
The same f14 period-control failures persist; all focal-pair controls pass.

**September 14, native sampling-phase diagnostic:**
[Results](../benchmarks/native_phase_2026-09-14.md) reuse the same four TRAIN
development families and eight native cells, with frozen plain/constrained
y60/H32 ensembles. Both sides receive a joint RGB translation in a fixed black
canvas; all original pixels are retained. These are diagnostic transforms,
not new codec encodes, fitting data or EVAL. Full64-phase coverage is restricted
to the predeclared screen failure; other cells use six residues plus period
controls. Original native outputs reproduce exactly. Twelve of48 whole-cell
period controls fail narrowly at full-resolution Y SSIM-L4, while all focal-pair
controls pass. Scalar preference and coarse-B contributions vary substantially
with phase. No new model, calibration, sampling contract, protected population,
or native RD qualification. The gallery retains original A/Bs and adds bound
phase figures without replacing their unpadded native evidence.

**September 14, native signed contribution diagnosis:**
[Results](../benchmarks/native_contributions_2026-09-14.md) reuse exactly the
preceding eight TRAIN development cells and frozen six-model/D compositions.
No new admission, fit, calibration, encode, extraction regime, EVAL or TEST.
The existing native public replay retains baseline sensitivities under its
diagnostic flag; ordinary results remain byte-identical. All 22 peer conflicts
are present in local feature linearization. Positive sensitivities dominate
local120 conflict contributions; a separate y60 failure follows coarse B
feature changes. Neither observation proves a feature arithmetic bug or a
retraining/ablation improvement. Existing model verdicts remain unchanged.

**September 14, frozen constrained-model native replay:**
[Comparison](../benchmarks/native_constraints_2026-09-14.md) reuses six existing
product TRAIN ensembles, all 812 native-local outputs and 408 earlier diagnostic
outputs. The same four development families remain the primary comparison;
fitting-role and earlier diagnostic content are separately labelled. No new
admission, fitting, calibration, encoding, extraction, EVAL or TEST. All prior
decoded pixel/peer rows and overlapping plain-model scores/maps reproduce.
All three constrained profiles fail the registered advancement screen despite
improved own-score map consistency. Peer-response associations are now visible
alongside internal consistency in the existing gallery. No model is promoted.

**September 14, explicit robust native-pair comparison:**
[Results](../benchmarks/native_robust_train_2026-09-14.md) reuse the preceding
eight TRAIN fit/four TRAIN development families and exact Rev3 features.
Two-row pair buckets preserve family assignments: 500 all-pairs versus 214
two-mentor robust pairs, using existing native pixels. No new admission,
encodes, extraction, calibration, EVAL or TEST. Both SSIM2 and Butteraugli
select training pairs, so their agreement is mentor consistency evidence.
Twelve initial fits are retained with an initialization-only interpretation:
nearby legacy sample seeds overlap. Twelve amended primary fits use recorded
disjoint sampler windows. All primary members and ensembles receive complete
canonical TRAIN panels/scatter; basic156 fails original-control rank tolerance,
y60 retains unresolved tail/map tradeoffs. Neither is promoted.

**September 14, native local TRAIN recipe comparison:**
[Protocol/results](../benchmarks/native_local_train_2026-09-14.md) admits12existing
product TRAIN families (8fit/4development), with original family assignments.
The prior replay's three families and all historical protected families remain
excluded. Canonical full944 Rev3 extraction on812 new fixed native JXL outputs
retains516fit/264development local rows after exact pixel deduplication. One
ICC-bearing source fails; a documented metadata-only replacement uses the next
eligible fitting-family hash, without color stripping or role reassignment.
All1,016 encoder attempts including204 from that failure are retained.
Twelve Rust fits with matched seeds compare unchanged profiles/control recipes
against10% local ranking supervision. Both treatments fail preregistered
advancement. No EVAL/TEST, calibration or release qualification. Sparse canonical
panel statistics are explicitly null where undefined; no fabricated passes.



**September 14, native local-quantizer replay:**
[Protocol/results](../benchmarks/native_map_replay_2026-09-14.md) reuses only
origins2010/7066/8206 from the saved September8 coarse intervention packets.
Canonical TRAIN family admission excludes document6068 under the later suffix8
reservation before pixels. Both256/512 renditions and distances1/3 yield408
retained bitstreams, freshly decoded with the pinned canonical decoder. Old
pixels and peers are not reused across decoder eras. Six frozen Rev3 ensembles
plus frozen Rev1 D are measured through public Rust scoring, with fresh native
same-buffer peers and exact final-source reproduction. This may overlap fitting
content; it is TRAIN mechanism development, not EVAL or new independent families.
No fits, new encodes, calibration, TEST/terminal or model promotion. It reveals
native map and scalar-preference failures; all earlier failed gates remain.

**September 14, newest real TRAIN precision checks:**
[Protocol/results](../benchmarks/precision_rev3_2026-09-14.md) reuse report,
screenshot and photo pairs from the existing admitted TRAIN development packet.
Five selected block probes compare native Rev3 SSIM/basic-edge features with
independent f64 arithmetic on unchanged XYB pyramids. No fitting, new source
admission, calibration, EVAL/test access or production arithmetic change.
Reference contribution changes are small relative to the failed maps, while
all five real probes exceed the existing synthetic locality tolerance.
No model qualification changes; original reports remain immutable.


**September 14, newest spatial family diagnosis:**
[Protocol/results](../benchmarks/spatial_diagnosis_2026-09-14.md) retain the
existing five TRAIN development origins and all frozen model bytes. Sixty-nine
saved analyses plus six matched family diagnostics separate max response from
other approximation failures. Three already-trained family subsets then add
69 wider spatial calls on the same 23 pairs, with no new admission, fitting,
calibration, EVAL/test access or scalar formula. All pixel/cache witnesses pass;
these controls still have twelve failed cases. Oracles are not deployable
models. No shipping or qualification status changes.


**September 14, newest spatial coverage:**
[Protocol/results](../benchmarks/spatial_coverage_2026-09-14.md) reuse existing
admitted TRAIN development origins and frozen models; no new source admission,
fit, calibration or EVAL/test access. A preregistered 30-cell content/size/JXL-q
grid has 23 available pairs and seven explicit gaps. All 69 model/pair checks
and 4,878 repairs pass native input and scalar parity; twelve spatial gate
failures remain. Six bin1 diagnostic calls on the two M3f-failing images add
210 repairs and reproduce all ranks unchanged. The previous five-case spatial
pass does not generalize for basic228/H32. No model qualification changes.


**September 14, newest aligned-bin optimization:**
[Protocol/results](../benchmarks/aligned_bins_2026-09-14.md) replay the same
forty admitted TRAIN finite-moment cases exactly (1,176 repairs). No source
admission, training/calibration, EVAL/test read or feature arithmetic change.
Complete prepared timing accepts the private aligned-fold optimization;
quality/steering verdicts are unchanged. An admission-metadata-only dimension
census identifies existing larger renditions for the next registered spatial
screen, not new held-out evidence. No model qualifies from this change.


**September 14, newest row-fusion replay:**
[Experiment and profile](../benchmarks/moment_rows_2026-09-14.md) reuse the same
forty admitted TRAIN finite-moment cases without new training/calibration,
source admission or EVAL/test access. Every report field matches exactly.
The optimization is rejected after complete-path timing shows modest enabled
benefit with slower disabled controls; production attribution stays at the
preceding implementation. Synthetic hardware profiling locates bin folding,
max retention and SSIM work as larger costs. No model qualification changes.


**September 14, newest base-image finite moments:**
[Protocol/results](../benchmarks/finite_moments_2026-09-14.md) apply the frozen
four basic/peak ensembles to the same five admitted TRAIN JXL pairs at bins
1 and 8 (forty cases, 1,176 repairs). Only original base signals enter the new
opt-in predictor; repaired images measure outcomes. Scores/features/gradients,
old density and disabled refinement remain exact. Basic228/H32 passes all five
spatial cases at both bins; other heads retain M2 failures. Paired runtime
shows substantial added cost, so the option stays off by default. No training,
calibration, new source admission or EVAL/test access. Artifact root:
`~/work/zensim-validation-2026-09-14/finite-moments/`. No model is qualified.

**September 14, newest moment-curvature diagnosis:**
[Protocol/results](../benchmarks/moment_curvature_2026-09-14.md) reuse twenty
immutable TRAIN reports from the basic/peak campaign, without new decoding,
scoring, fitting, calibration or EVAL/test access. The existing Rust analysis
owner separates finite root curvature from remaining map error. For basic228/H32,
the curvature-only oracle passes all five M3f cells and removes 92.9% of aggregate
squared error against observed feature linearization. Repaired-image features
make this an unavailable-to-runtime oracle; scalar/map verdicts are unchanged.
All prior fields and the guarded replay are exact, with independent algebra
tests and discriminating input controls. Artifacts:
`~/work/zensim-validation-2026-09-14/moment-curvature/`.

**September 14, newest basic/peak decomposition:**
[Protocol/results](../benchmarks/product_peaks_2026-09-14.md) reuse the same
admitted native product-TRAIN packet without new extraction, calibration or
EVAL/test access. Twelve paired fits and four full ensembles separate HF,
L8, hard-max and head-capacity effects. All 31,548 raw/packed predictions are
bit-exact, twenty original-bitstream native audits pass, and 588 finite pixel
repairs locate a remaining non-max approximation failure. All 23 codec
development sources are distinct source families for descriptive paired
bootstrap. An old rounded control export was refused; the prior verified
round-trip export is used instead. Full panels/scatter, source uncertainty,
failed gates and matched timing are retained. No model qualifies; no inference
formula changed. Artifacts: `~/work/zensim-validation-2026-09-14/product-peaks/`.

**September 14, later product scale frontier:**
[Protocol/results](../benchmarks/product_scales_2026-09-14.md) reuse the exact
preceding product-TRAIN tables and source-family roles. No new source admission,
feature extraction formula, calibration, EVAL or TEST read. Eighteen fits over
six declared subsets, three paired seeds and the corrected mixture yield six
complete served ensembles. All 47,322 raw/packed development predictions match
bitwise; canonical panels retain every seed and complete composition.

A synthetic supported coarse-IW-MSE model reproduces NaN attribution from
unrequested zero-count cells. The retained pass now skips zero-sensitivity
cells before normalization. Ten real TRAIN y346 map calls become finite;
20 other calls' complete block records remain exactly unchanged. All six
ensembles have exact consumed-feature/native-score audits on five original
JXL pairs. All still fail some registered spatial cases. Basic228 gives a
useful inexpensive scalar control, but its prepared map cost and coherence
remain inadequate evidence for shipping. Timings and memory receipts cover
1MP/4MP synthetic inputs; p95 and native encoder RD remain unmeasured.
Artifacts: `~/work/zensim-validation-2026-09-14/product-scales/`.

**September 14, newest TRAIN product packet and sampler correction:**
[Protocol/results](../benchmarks/product_train_2026-09-14.md) and
[hashed summary](../benchmarks/product_train_2026-09-14.results.json).
Original W-LIN7 TRAIN key/URI metadata admits 10,499 original-bitstream pairs
across 148 sources and 141 families, after canonical validation/test, historical
test families, and eval8 reservations are excluded. Original file/pixel hashes,
Rev3 formula/schema, fresh same-pixel SSIMULACRA2 targets (575 negative), and
family-separated internal fit/development/calibration are preserved. No decoded
duplicates or identical reference pixels cross the internal partitions. Native
human TRAIN development re-extraction matches all 944,000 reused feature cells.

Legacy stratified sampling ignores relative group weights and excludes singleton
band cells: 194 human / 2,002 codec strata yield only 8.88% human pairs and 86.90%
row coverage. Eight initial fits are retained as a stopped, mis-specified mix.
Analytically collision-corrected uniform weights yield 49.881–50.198% usable
human pairs and full coverage; all three actual training digests match replay.
All 24 replacement fits and eight complete ensembles have Rust API predictions
and canonical TRAIN panels. Round-trip output proves 63,096 raw/packed predictions
bit-exact. Matched B/D are freshly extracted at Rev1 on identical development
pixels; their unknown historical training exposure remains a limitation.
Five small TRAIN JXL cases per ensemble expose the scalar/spatial tradeoff;
full944 has stronger proxy rank but incomplete refinement support. No new EVAL,
TEST, calibration use, native RD, or shipping qualification. Artifacts are under
`~/work/zensim-validation-2026-09-14/product-train/`.

**September 14, frozen Rev3 matched evaluation:**
[Protocol/results](../benchmarks/rev3_qualification_2026-09-14.md) add native
Rev3 full944 KonJND JPEG SELECT (404 references/pairs) and KonFiG validation
(436 pairs, four origins) to the existing 3,125-pair KADID SELECT evidence.
Original admitted view hashes, round-half-up JPEG pairing, KonFiG content
deduplication, source/file/pixel identities, formula revision, and frozen model
bytes are retained. No terminal payload or model fit. B/D are re-extracted at
Rev1 and compared on the same pixels; their absent historical era metadata is
explicitly unqualified, with complete current native prediction parity.
All nine candidates trail B/D on KonJND despite higher KADID rank, fail measured
outlier clauses, and remain unshippable. Selected619 fails 836/840 new spatial
parity cases. Some wide models exceed identity. Source-cluster intervals,
complete scatter diagnostics and keyed CPU SSIMULACRA2 comparisons are retained;
KonFiG's four origins and SSIMULACRA2 tuning exposure limit interpretation.
Native artifacts: `~/work/zensim-validation-2026-09-14/rev3-qualification/`.

**September 13, later user split ruling:** new work uses train only for all
fitting/calibration/checkpoint decisions, eval only for gates/evaluation, and
never opens test/terminal segments. This overrides prior touch-once plans and
the internal test-role usage described in the dated studies below. Their
artifacts remain immutable; no test-to-eval relabeling is performed. The active
feature-screen v2 requires separate admitted manifests, refuses mixed caches,
and moves prediction/eval out of the training stage. Thirteen synthetic boundary
checks and a real Rust fixture smoke verify the new path; no scientific corpus
was read or a competitive model trained by this implementation.

**September 14, newest strict integrity prototype — bounded evidence only:**
[Protocol/results](../benchmarks/steering_integrity_2026-09-14.md) and
[hash-bound summary](../benchmarks/steering_integrity_2026-09-14.results.json).
Original corruption-manifest metadata admits eight fit, two training-calibration
and eight eval origins; 8462/9066 are excluded because of later test roles.
Fresh Rust Rev1/libm/f32 D228 extraction reads explicit source/native-codec paths;
no historical mixed feature tables. New decoded-pixel hashes bind native inputs
whose original records only pinned bitstreams. Original feature/decoder identity
and role receipts remain in `TRAIN_ADMISSION_NATIVE.json` and the extraction
audits; source manifest SHA is
`1ec469fe694284f029cf769258fe7bd91e231f336825686699ea371be6c5f433`.
Final training admission SHA:
`d61262d19865a2abc6464317eb1090a9c6729aca073f189e58f342174b270eae`.

Severity rules and visual layout amendment were frozen before fitting. Inert
and weak/ambiguous operations are excluded from binary training, not relabelled
as valid negatives. There are 900 unique fit and 124 calibration pairs. One
HGB100 head plus a determinism replay (identical predictive bytes) gives zero
calibration honest activation; frozen eval detects 175/180 admitted severe
proxies, 109/113 real-bug proxies, with 0/320 honest activation. Full eval scores
6,024 attempts / 5,679 pixel-unique pairs, retaining 1,086 activations among 5,173
unlabelled weak/ambiguous pairs as unresolved specificity. A known binary label
on duplicate pixels supersedes missing labels, with all catalog provenance
retained; contrary binary labels refuse the report. No eval-tuned threshold,
no test reads, no product qualification. See results for the five misses,
prepared-map audit, frozen tools, complete input hashes and measured cost.

**Purpose.** An honest ledger of *what training data and experiments were tried, what
actually helped, and — where it differs — the **real** reason a thing failed vs the
reason we thought at the time.* Several "verdicts" in this project were later found to
rest on a hidden bug, confound, or invalid measurement. This doc exists so we don't
re-add a dataset a confound already poisoned, re-run a dead-end, or trust a retracted
claim.

**September 13, later runtime profiles — development evidence:**
[Runtime study](../benchmarks/runtime_profiles_2026-09-13.md) reuses the exact
admitted native Rev3 tables below with their input/table/extractor identities;
no new source, split, feature formula or corruption label is admitted. The
existing owner verifies hashes before reuse. There are 48 screening MLP fits
and eight additional human finalist seeds, plus two deterministic linear fits.
Human and codec-proxy objectives remain separate; no new corruption classifier
is trained under the old binary labels. Mixtures add no fits. The 96-cell broad
repair panel is now used to select mixture weights and is not independent
evidence for those weights. Final weighted compositions are replayed through
the Rust surface. No terminal human panel or native target/RD qualification
was run, and retained artifact scores still block product qualification.

**September 13, newest steering-subset study — STANDS within development panels:**
[Protocol and results](../benchmarks/steerable_subset_2026-09-13.md) retain the
same 19,958-row/source-family admission and native Rev3 formula era below.
Fresh extraction and reused tables are hash-bound; no new source or terminal
data was admitted. There are 195 unique MLP fits (117+60+18), with 27 initial
fits reused in the 45-fit five-seed finalist directory; nine additional linear
controls are deterministic. Do not count inherited artifacts as fresh fits.
The human test split is now feature-selection evidence, not a new terminal
holdout. Broader spatial work reuses the September 8 max-attribution registration,
verifies all 24 original image hashes and current train admission, and executes
four block sizes through the prepared Rust API. The linear local120 control
passes 96/96 but under-penalizes severe corruption. More accurate uniform MLP
ensembles fail some spatial cells. No model qualifies, and native RD/targeting
claims remain unavailable. Local extraction optimizations preserve consumed
feature arithmetic; corrected HF-gain map derivatives supersede earlier audits.
All results, failures, model identities and intermediate binaries remain under
`~/work/zensim-validation-2026-09-13/steerable/`.

**September 13, latest scale-selective944 study — STANDS within these panels:**
[189-fit results](../benchmarks/scale_selective_944_2026-09-13.md) reuse the exact
19,958-row admission below, with fresh native and direct Triangle extractions.
Native feature CSV is byte-identical to the prior producer. New explicitly
versioned direct sampling eras cover 1/2/4/8, 1/3/5/7 and 1/2/3/5; do not mix
these feature tables or retag old bakes. Each coarse level comes directly from
original XYB with zenresize binary16 source rounding, independently verified.
Legacy/v2 masked+IW additions are tested separately at every scale. Three seeds,
H128, 32 epochs, every-epoch dev selection. Prime effects are mixed; wide
features remain useful for corruption. Actual v2 scale dispatch now reduces
native619 cost by 60% ST/58% MT8 on the bounded benchmark, correcting the earlier
no-compute-savings limitation. Spatial failures and limited source diversity
still prevent qualification. No protected data or deployed model changed.

**September 13, later capability study — STANDS within the stated panels:**
[Protocol/results](../benchmarks/feature_ceiling_2026-09-13.md) and
[exact read sets](../benchmarks/feature_ceiling_2026-09-13.json).
Fresh full944 Rev3 extraction through BakeScorer: 11,125 KADID/TID human rows,
620 codec-proxy rows (30 origins/29 families), 8,213 deduplicated corruption
rows (12 admitted origins). TID is train-only; KADID terminal/T0 and fixed
corruption validation origins are untouched. The later source-family manifest
excludes 8414/8434 and co-locates 7004/7058; digit-only admission was insufficient.
Old tables provide membership/labels, never feature values. Producer
`basic+peaks+masked+iw+v2+append+append2@w944/ceiling_rev3#b782e349` includes
all 944 slots, unlike older wide producers that omitted v1 pooled features.
Inputs, byte hashes, family/row IDs, tools, bakes and original codec binary are
retained under `~/work/zensim-validation-2026-09-13/ceiling/final/`; codec commit
coverage limitations are explicit in `CODEC_PROVENANCE.json`.

162 fits show data dependence, a checkpoint-cadence confound, useful wide
corruption features, and weak v2 spatial repair prediction. No reduced layout
passes every family/error bar. Restricting v2 read sets does not yet save their
extraction cost. This native-box study does not extend the fractional-plane
training evidence. No release qualification or universal feature ceiling.

**September 13 sampling follow-up (earlier):**
[20-layout/60-fit screen](../benchmarks/sampling_serving_2026-09-13.md),
using only the fixed T2 JXL proxy packet and reused training corruption PNGs.
Explicit sampling contracts are served through Rust BakeScorer. All layouts
have failed spatial cells; no qualification or protected-data evaluation.
[Legacy coarse-feature audit](../benchmarks/coarse_legacy_features_2026-09-13.md)
separates September shipped read sets from August measured contribution tables.

**Status flags** on each verdict:
- **STANDS** — verified, still correct.
- **RE-FRAMED** — the finding survived but the *stated reason* was wrong/incomplete.
- **INVALIDATED** — the evidence was a bug/confound; conclusion does not hold as stated.
- **UNVERIFIED** — asserted but never independently confirmed; treat as provisional.

**How this was built (2026-07-15).** Synthesized from a 4-reader survey of `benchmarks/*.md`,
`docs/*.md`, `CLAUDE.md`, the memory store, and `git log`, plus direct verification of the
two most load-bearing claims (the kadid/tid `ssim2_gpu` misjoin — reproduced this session;
the V0_8-leak doc conflict — confirmed in `CLAUDE.md`). **Every row carries a citation so it
can be re-verified, not venerated** (per the project's "docs almost always lie — trace to
source" rule). Where the survey and a shipped doc disagree, that conflict is called out in §4.

---

## 0. The five recurring "actual why" families — READ THIS FIRST

September 8 later kernel experiment: [padded SSIM rows](../benchmarks/padded_ssim_rows_2026-09-08.md)
retain every saved feature/score/map byte while reducing measured complete
scalar cost. Row pitch changes physical storage, not horizontal tile boundaries
or sum order, and differs from the historical plane-base staggering failure.
The cache/code-generation mechanisms are not separately identified. Non-trigger
controls bound a small regression; formal performance and model qualification
remain incomplete. No dataset, feature era or model changed.

September 8 later measurement: [A/D scalar cost and benchmark repair](../benchmarks/model_blend_speed_2026-09-08.md).
The first run's four rounds/240 waits came from detecting its own Linux
lock-heartbeat task as a competing benchmark. An observed failing regression
and canonical repair enable 40 rounds per geometry. Complete blend costs
essentially the same as A alone; common extraction/full-pool work determines
the next optimization. Background activity prevents strict release admission;
observed latency inequalities are diagnostic. No model or dataset changed,
and no validation or terminal images were consumed.

The dominant class of dead-end in this project was **never a bad mechanism — it was an
invalid selection or verdict instrument.** Almost every entry below is an instance of one
of these five. The guard each produced is the scar tissue; §6 collects them.

1. **The ssim2-target selection trap.** The synthetic training corpora use an *ssim2-derived*
   `human_score`. So any verdict scored by *rank-agreement with an ssim2-shaped signal*
   rewards ssim2-shaped bakes and *penalizes deliberately-different ones* — even when the
   different bake is better on human MOS. Recurs ≥4×: the IW-SSIM "falsification" (§3.3),
   the TV/cvvdp_w1 panel reversal (§3.7), the w11 corpus-refit (§3.4), and `cid22_train`
   anti-correlating with real CID22 as a *selection axis*. **Guard:** never pick candidates
   on an ssim2-derived axis; use held-out human MOS + the full panel.
2. **SROCC-only blindness to dynamic range.** SROCC is rank-invariant, so it is *totally
   blind* to a collapsed/clamped dial. V0_5 bakes were "panel-best" while 85% of predictions
   clamped to 0; V39 had SROCC 0.88 with a dial that ran to −128 at *high* quality.
   **Guard:** mandatory **two-panel** eval (rank *and* dial: G1 dynamic-range + monotonicity).
3. **Post-selection / holdout-fishing on small n.** Picking the best-of-N on the same small
   test set and then reporting its p-value. Burned UPIQ-380 (~21 looks; the BHdr "p=0.005"
   became p=0.221 corrected) and nearly mis-read AIC-4 (n=300). **Guard:** Westfall-Young
   family correction; a holdout is never a recipe-search axis.
4. **Join / feature-extraction integrity.** The metric/model was fine; the *data plumbing*
   silently broadcast or zeroed features. Ref-only joins (kadid/tid misjoin), GPU odd-dim
   garbage (dial-grid corruption), empty→0.0 imputation (picker all-zero features).
   **Guard:** `_validate_metric_columns()`, grid quarantine, fail-loud joins, assert
   features non-constant.
5. **Regime / measurement mismatch.** Comparing across mismatched feature regimes or
   drifted code. u8-shell vs PU-linear grams, validate-vs-runtime spline caps, trainer code
   drift, f32 finite-difference floors. **Guard:** `--features-name` regime tags,
   runtime-parity caps, `[training].trainer_commit` pin.

---

## 1. Training-corpus ledger — what was fit, what actually helped

Shipped today (verified in `zensim/src/profile.rs`): **A** = `v47_strict_qat_native_2026-05-27.bin`
(MLP, deprecated-but-compiled); **B** = `b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07.bin`
(linear SDR default); **BHdr** = `bhdr_linear_shaped_cvvdpmix_2026-07-12.bin` (linear HDR, sha `7d7f2123`).

| Dataset | What it is | Used to fit | Verdict | Why (stated) | Cite |
|---|---|---|---|---|---|
| **hdr_v3mix** | 7,410 HDR-JXL renditions; target = cvvdp-mix `0.5·clip01(ssim2)+0.5·clip01((JOD−6)/4)` | B cid head (80%), BHdr, B kon head, B winsor corpus; **@944 (hdr944-leg): the Appendix Q phase-1 MLP co-train leg (w 1.2 = 17.0% pair share) + hdr-only linear control (2026-08-05)** | **HELPED — most load-bearing**; @944 co-train: **SDR-free** (full wave-11 SDR panel held) + korshunov-parity, but **narwaria-blind** (all Q candidates significantly behind BHdr there — zenjxl-only supervision suspected; Q.R7) | the cvvdp-mix TARGET is the driver (+0.039 CID22 vs pure ssim2); transfers to SDR linearly. @944 the corpus trains in-domain 0.93+ MLPs whose UPIQ transfer varies 0.05–0.73 across seeds — in-domain val is BLIND to transfer collapse | `profile_b_methodology_2026-07-12.md §2`; `linear_projections_2026-07-03.md` (commit `1b2bdb9b`); `sota944_campaign_2026-08-03.md` Appendix Q/Q.R |
| **safesyn** | 196,086 synthetic tiles (CLIC/CID22/kodak/gb82-sc crops), ssim2-derived target, CID22-49-purged | A (v11→v47), B kon head | **HELPED (foundational)** | base training mass; supplies the BVLS kon head its bulk | `CLAUDE.md` "Safe synthetic dataset"; `DATA_SPLITS.md §3` |
| **cid22_train** (201-ref) | 17,611 rows, **ssim2-anchored** (`human_score`=`ssim2_gpu/100`, NOT MCOS) | A, B kon head | **HELPED (train-legal)** | 201 refs disjoint from the 49-ref holdout; human MCOS never trains | `DATA_SPLITS.md §7.1` |
| **kadid / tid** | 10,125 / 3,000 rows, DMOS/MOS; ~95% analytic (non-compression) distortions | A, B kon head (low weight) | **NEUTRAL / integrity guard** | ssim2 tuned in-sample → never a scoreboard; train==val | `DATA_SPLITS.md §3` (T1) |
| **konjnd-dense** | 20,160 rows (active-mix target) + 1,008-ref val (PJND) | A (modest), B kon head (**≈wash**) | **HELPED (A) / NEUTRAL (B)** | for B the KonJND skill is the **BVLS sign-mask, not the corpus** (`canonkjhdr15`≈`canonhdr15`) | `linear_projections_2026-07-03.md` round-2 |
| **inclusive-winsor corpus** | 10,810 = hdr_v3mix + zenjxl near-lossless SDR sweep; fits B's winsor bounds only | B calibration | **HELPED (user-caught fix)** | frees 245/372 clamped-constant features → near-lossless dial 91.5→96.1 at ~0 MOS cost | `profile_b_methodology §3` (commit `aaa1ecac`) |
| **cvvdp_iwssim_LARGE** | 73,300 rows, `mix_cv40_iw60` target, **300 feats** | PreviewV0_5Balanced (V_22, not a current Profile) | **HELPED (V_22 era)** | +0.068 KADID/+0.087 TID; "marginal value is in content types not in training" | `v22_mix_LARGE_iwssim_methodology_2026-05-18.md` |
| **bigcodec / imazen-26** (canonical-picker) | 2.9–5.7M ssim2-normalized real-codec cells, 414 imazen-26 origins | MLP recipes (w≈0.25, later 0.5); **EXCLUDED from linear B** | **⚠ RE-FRAMED 2026-08-04 — the exclusion was RIGHT and its MLP carve-out was NEVER VALIDATED.** Original: "EXCLUDED from linear — poisons CID22 (0.65–0.76); MLPs absorb it via capacity" | a 372→1 linear head gets pulled to the ssim2 multi-codec regime. **"MLPs absorb it" was tested on CID22 ONLY. Measured 2026-08-04: at ~78% of row mass (bigcodec + its distillation teachers) the 944-era recipe collapses classic-IQA breadth AND KADID at EVERY feature width — 372-width arms on the 944 data score CSIQ 0.681/0.438, LIVE 0.666/0.300, KADID 0.546/0.641 vs B 0.934/0.897/0.809 and winner_dial 0.958/0.960/0.946. MLPs did not absorb it; they inherited it on the axes nobody re-checked.** **⚠ KADID FIGURES CORRECTED 2026-08-04 (§3.20): the ext-lineage tables store KADID's target INVERTED, so every KADID number in this row was published as an unsigned magnitude of a SIGNED quantity. Against KADID's real human MOS the 372-width-on-944-data arms are **−0.546 / −0.641** (anti-correlated), B is **+0.809** and winner_dial **+0.946**. The row's CONCLUSION is unchanged and in fact understated — the 944 recipe does not merely weaken KADID, it INVERTS it. CSIQ/LIVE/CID22 in this row are unaffected (those corpora are correctly oriented on every root).** Why it stayed invisible: KADID/TID were classified "integrity guard, never a scoreboard" (row above), and CSIQ/LIVE did not exist as corpora until `ac787382` (2026-07-18), AFTER the MLP recipes were set — so no gate in force at the time could fail. | `linear_projections_2026-07-03.md` Falsified #1; commit `c7900d53`; falsification: `sota944_campaign_2026-08-03.md` width-discriminator (k=2/arm — seed spread too wide to rank widths; the breadth/KADID collapse is robust to that limitation) |
| **KADIS-700k** | 700k analytic-distortion cells on 140k Pixabay refs; GPU-metric targets, no human labels | negative-region calibration; the §8.32-8.33 MLP-neg prototype (not shipped) | **EXCLUDED for shipped fit; diagnostic/negatives only** | cvvdp-scalar target falsified (V41); analytic + more-photographic bias | `DATA_SPLITS.md §2b`; `project_profile_b_hdr.md §8.31-8.33` |
| **kadis-hdr** | 11,400 PQ-domain synthetic HDR cells, 1,140 imazen-26 refs | BHdr breadth retrain | **EXCLUDED — falsified on UPIQ** | synthetic HDR breadth does not transfer to human-MOS HDR (linear family) | `bhdr_improvement_split_lineage §8.8-8.15` |
| **CID22-49 / AIC-3 / AIC-4 / UPIQ-HDR / JPEG-AI-SDR25** | human-MOS holdouts | **HOLDOUT-ONLY** | **EXCLUDED (sacred)** | the only held-out human generalization gates; UPIQ-380 now BURNED (~21 looks) | `CLAUDE.md` "CID22 is VALIDATION-ONLY"; `DATA_SPLITS.md §3` |

**Superseded / unverified-ship:** hdr_v3 (pure-ssim2, prior BHdr `anchored2`); e1-fill/zenjpeg-420-e1 (V0_7 line, built ~85%, **no promoted-bake trail** → UNVERIFIED); KonFiG-IQA (v53 group, **no shipped-bake trail** → UNVERIFIED); cvvdp-only LARGE (falsified, craters KonJND); dssim co-train (falsified, see §2). Pre-`VALIDATION-ONLY`-era weights (Nelder-Mead `ae28074`, CMA-ES) trained on {KADIK, CID22, TID} *including CID22* — any CID22 SROCC from that era is **contaminated**, archaeology only.

**AIC-HDR2025: UNOBTAINABLE (user ruling 2026-08-05)** — the dataset was never publicly released (README-only clone at `/mnt/v/datasets/aic-hdr2025/`) and the authors are unresponsive; STOP live-checking `github.com/jpeg-aic/AIC-HDR2025`. Removed from the HDR anchor plans; it can never enter train or eval.

---

## 2. What didn't work — falsification ledger (verdict strength matters)

Ranked by consequence. **Verdict strength** distinguishes a rigorous multi-stat/multi-seed
falsification from a single-run or SROCC-only call (the latter are re-flagged in §4).

| Approach | Why it failed (stated) | Strength | Re-opened? | Cite |
|---|---|---|---|---|
| **CVVDP-scalar-target** (predict CVVDP's pooled JOD) | "emulating CVVDP's OUTPUT ≠ having its CSF mechanism"; strictly worse ranker (CID22 0.66/0.70) | ⚠ **RE-FRAMED 2026-07-15** — a TARGET-SHAPE confound, not a cvvdp limit (§3.17). RAW `cvvdp_score` trains fine (CID22 0.85); only `cvvdp_log_norm`'s near-lossless-tail expansion craters it (0.58). | RE-OPENED — raw-scalar cvvdp is a viable HF lever | `benchmarks/…§8.36`; `v47_cvvdp_target_FALSIFIED_2026-05-27.md` |
| **IW-SSIM direction** | "wins KADID+TID, craters CID22 (FRIQUEE transfer)" | ⚠ MIXED — the CID22 crater is the ssim2-SROCC trap (§0.1); IW won TID best-ever (Z-RMSE 0.231) | created "SROCC-only BANNED"; single-MLP high-k re-confirmed on full panel | `CLAUDE.md` "SROCC-only BANNED"; `falsification_reeval_results_2026-05-15.md` |
| **Su-2023 distortion-manifold pretrain** | wins KADID+TID, loses held-out CID22 −0.027 (synth→authentic transfer) | multi-stat, multi-corpus | no | `docs/v0_20_path_evaluation_2026-05-14.md §B` |
| **bigcodec mass in a linear fit** | poisons CID22 (0.65–0.76 at ≥0.25 w); MLPs absorb it | deterministic refits | durable rule "keep bigcodec OUT of linear" | `linear_projections_2026-07-03.md` |
| **BHdr kadis-hdr breadth retrain** | synthetic HDR breadth ⊄ human-MOS HDR | ⚠→ RIGOROUS *after* a regime confound was fixed (§3.9) | withdrawn then re-confirmed | `bhdr_improvement_split_lineage §8.9-8.15` |
| **BHdr cvvdp-mix "significant UPIQ win"** | promotion basis was POST-SELECTION (p=0.005→0.221 corrected) | RIGOROUS retraction | shipped anyway; UPIQ-380 burned | `bhdr_improvement_split_lineage §7` |
| **hdranch3 BHdr ramp-proxy** | optimized a proxy, never ran UPIQ; UPIQ 0.606 vs 0.7536 | RIGOROUS multi-stratum | dead; installed "UPIQ-panel-before-ship" | `project_bhdr_hdranch3_falsified.md` |
| **G5 KonJND-HF ≥0.70** (agg head + regime ensemble) | Pareto limit — clearing 0.70 craters CID22; regimes overlap in feature space | falsified ×2 architectures | dead; needs better HF *representation* | `CLAUDE.md` V39-learning #9 |
| **dssim co-training** | regressed CID22 0.04–0.07 | ⚠ single-cycle, **confounded** (61% was recipe drift, 39% dssim) | retry never funded (bakes stuck ZNPR v2) | `cycle_7_dssim_outcomes_2026-05-12.md` |
| **Dynamic-range-floor regularizer (V40)** | overshoots >100 | ⚠ weak/single-config | no | `CLAUDE.md` V39-learning #7 |
| **Axioms-only residual metric** | A3 (no inversions) not gained; degenerate range (CID22 0.50-0.62) | 2 parameterizations, full panel | no | `axioms_only_residual_falsified_2026-05-27.md` |
| **Cascade / NNLS-all-pinned ensembles** | dominated by direct convex blends | deterministic 44/44 | no | `linear_projections_2026-07-03.md` |
| **D1/D2/D3 input-shaping** | V_20 IS is a B3 specialist (−0.014 aggregate); shaping is intrinsic | train+eval per variant | D2 shipped as niche PreviewV0_4 | `CLAUDE.md` "V_20 learnings" |
| **Recipe knobs** (low/mid-q boost, cosine LR, h=64/256, NiN-off, per-group std, P²) | none broke the V_18 CID22 ceiling | mixed (cycle-9/12 re-confirmed) | ~120 cycle-7-13 bakes are ZNPR v2, retry deprioritized | `falsification_reeval_results_2026-05-15.md` |

---

## 3. The confounds — "we thought X, the actual why was Y"

The most important section. Each: **thought-why → actual-why**, when discovered, status,
shipped-bake impact. All shipped bakes were verified *unaffected* except where noted.

### Tier 1 — highest consequence

**3.20 The ext-lineage KADID TARGET IS INVERTED — `(5−dmos)/4` instead of `(dmos−1)/4`
(discovered 2026-08-04, Tier 1).**
*Thought:* KADID SROCC magnitudes were comparable across the 372/canonical era and the
720/924/944 ext era, so "era 0.946 → 944 0.423" read as a competence regression.
*Actual:* the two lineages store **opposite orientations of the same label**, and every
KADID number ever published from an ext root is **sign-flipped**.
`scripts/canonical_corpus/build_fr_corpus_pairs.py` `build_kadid()` (line 113) emits
`human_score = (5 − dmos)/4` — the standard invert-a-DMOS reflex that CSIQ (`1−DMOS`) and
LIVE (`1−dmos_new/100`) correctly need — but **KADID's `dmos` column is a MOS in
disguise**: raw crowdsourced DCR (349,800 ratings, `raw_crowdsource_data.csv`) falls
4.0789 → 2.0072 across severity levels 1–5, so it was already quality-oriented and the
flip inverts it. The canonical lineage (`build_canonical_parquets.py:288`,
`fix_kadid_tid_build_pairs.py:15`) uses the correct `(dmos−1)/4`. Both residuals are
**exactly 0.0** — two transforms, not drift.
**Consequences.** (i) Every `ext720`/`ext924`/`ext944` KADID SROCC is the negative of the
true-quality value; **110 of 188 board bakes are anti-correlated with KADID's real human
MOS** while the board renders all 188 as positive magnitudes. (ii) Every 944-era model
**trained** on the flipped column (`--group kadid:…ext_kadid.parquet`), so its KADID
inversion is real and inherited — dose-response over 111 fullevals: train weight 0.50 →
mean −0.457, weight 1.50 → mean −0.925. (iii) The wave-8 **E1 gate ("KADID ≥ 0.70")** was
passed by the three most-inverted arms (−0.93) and failed by `W8C_s3101`, the only
wave-8 arm whose KADID was correctly oriented (+0.358). (iv) The pre-ext era models are
**unaffected in behaviour** — winner_dial is +0.9464 and shipped **B** +0.8201/+0.8085,
positive on **25/25** KADID distortion types; only their reported sign was wrong.
**TID is CLEAN on every root** (verified against the same raw ratings, +0.9168, n=960);
so are CSIQ and LIVE (their natives genuinely are distortion-oriented).
**Status: eval reporting INVALIDATED for ext-lineage KADID; the 944 models' KADID
behaviour is a real defect requiring a retrain on a fixed table if KADID competence is
wanted. Shipped bakes (B/BHdr/A) unaffected — all pre-date the ext lineage and trained on
the correct column.** Cite: `benchmarks/sota944_campaign_2026-08-03.md` REGISTERED
APPENDIX F + F.R1..F.R8; `benchmarks/wave9/kadid_orientation_2026-08-04.md` (which
measured the orientation and flagged the anomaly one day earlier);
`benchmarks/eval_annotations.json` entries `kadid-ext-root-inverted` +
`kadid-ext-trained-inverted-model`.

**3.1 kadid/tid canonical `ssim2_gpu` = ref-vs-ref MISJOIN; `iwssim` = human-score copy.**
*Thought:* KADID/TID carried real IW-SSIM + SSIMULACRA2 columns → multi-target training was
legitimate. *Actual:* two committed bugs — `iwssim` was a literal copy of `human_score` (a
val-only mock whose "mock" qualifier was stripped upstream → training on the eval label);
`ssim2_gpu` was joined on `ref_basename` alone (codec/q keys dropped) → every ref's ~125
distortions averaged to one ≈99.99 value → corr ≈0.01 with MOS. **The metric was sound; the
Python join collapsed it.** Discovered 2026-05-25. **Status: RE-FRAMED/fixed** — re-joined
positionally (post-fix real SROCC KADID ssim2 0.013→0.813); `_fixed_2026-05-25.parquet`
siblings + a `_validate_metric_columns()` guard. **Shipped bakes: NONE affected** (all train
on `human_score`/real cvvdp). **INVALID:** ~10 multi-target experiments with kadid/tid
train_weight>0 on the contaminated column. *Verified this session* — training on the current
canonical `ssim2_gpu` ranks kadid **backwards** (val SROCC −0.07..−0.13). Cite:
`DATA_INTEGRITY_kadid_tid_metric_columns_2026-05-25.md`, `DATA_INTEGRITY_root_cause_2026-05-25.md`.

**3.2 dHash d≤16 "contamination" cleanups = mostly FALSE POSITIVES.**
*Thought:* V0_8's CID22 0.8948 was inflated by 361 near-dup training sources leaking 22/49
CID22 refs (drove a 30.6 GiB purge + V0_15 retrain); and V0_18's CID22 was inflated by KADID
overlap at d≤16 (drove the V0_18→V0_19 ship). *Actual:* **d≤16 is the LOOSE screening
threshold, not a contamination cutoff.** User montage review confirmed NONE of the flagged
matches are the same image (flat-region/UI/composition false positives); at d≤10 CID22↔KADID
and CID22↔TID have ZERO overlap. Discovered 2026-05-14. **Status: V0_19 REVERTED** (V0_18
restored), "V0_18 inflated" RETRACTED. **⚠ OPEN:** the 2026-05-12 V0_8 361-source purge used
the *same* loose d≤16 threshold, was flagged "re-audit at d≤10," and **never was** — so the
"V0_8 was inflated by leakage" claim is itself **UNVERIFIED** under the same false-positive
caveat. See §4.1. Cite: `dhash_threshold_revert_2026-05-14.md`, `v0_19_REVERTED_2026-05-14.md`.

**3.3 IW-SSIM "falsified on CID22 SROCC" = the ssim2-target selection trap (§0.1).**
*Thought:* IW bakes are falsified — CID22 SROCC craters to 0.4632. *Actual:* the corpus's
`human_score` is ssim2-derived, so CID22 SROCC rewards ssim2-shape and penalizes IW's
deliberately-different surface; the same bakes WIN TID human MOS + PWRC + Z-RMSE (best TID
ever). Discovered 2026-05-15. **Status: RE-FRAMED** — created the project-wide **SROCC-only
BANNED** rule; all prior SROCC-only "falsified" verdicts marked provisional. Cite:
`CLAUDE.md:408-488`.

**3.4 2026-05-29 DIAL GRID corruption — masked/IW features were GPU odd-dim garbage.**
*Thought:* shipped B scored −80/−81 on a "webp trio" → B-specific OOD content blindness →
proposed corpus-refit / knob-floor. *Actual:* the −80s came from the **grid's stored
features**, not the model — on 9/115 ladders (8/24 webp) the masked/IW features (f228..f371)
were extraction garbage, **bit-constant across each ladder's 40 q-values** (a `zensim-gpu`
odd-dim pyramid pathology emitting non-NaN garbage). Fresh CPU extract → sane + ordered; the
webp knob-blocker **never existed** (quarantined grid p10 83.7 vs 9.4). Discovered 2026-07-05.
**Status: grid QUARANTINED** (`dial_grid_372col_2026-05-29_quarantined.parquet`) — any dial
number on those 9 ladders since 2026-05-29 by ANY bake is garbage-input. The corpus-refit
"fix" was run anyway and independently falsified (ssim2-target trap again). **Shipped bakes:
none affected.** Cite: `MEMORY.md:30`, `provenance_best_results_2026-07-04.md §w11` (commit `ae4209a8`).

### Tier 2 — significant

**3.5 V0_5 "panel-best" bakes hid a BROKEN DIAL (SROCC-only, §0.2).** Distance-shaped raw
(≈[−30,30]) → runtime `clamp(0,100)` pinned 84.6% of predictions to 0; SROCC blind to it.
**Status: fixed** via PCHIP spline retrofit (rank-invariant) + the mandatory two-panel
scorecard. Cite: `dial_bug_audit_2026-05-20.md`.

**3.6 BHdr "UPIQ +0.0223 significant p=0.005" = POST-SELECTION (§0.3).** λ picked best-of-7
on the same 380 UPIQ pairs; Westfall-Young maxT → **p=0.221 NS**, "6/7 corpora" → 4W/2L/1-ns.
Discovered same day as ship (`f5704efb`). **Status: improvement NOT established (non-inferior
only); ship kept as-is.** The λ-robust out-of-domain SDR gains (CID22/TID/KonJND) still stand.
Cite: `project_profile_b_hdr.md §7`.

**3.7 TV/KADIS "A is Pareto-optimal" = 3 data misreads.** (1) TV pairs were POISONED — signed
U-shaped distortion types 7/18/25 excluded from the safety grid but not the TV pairs → 6.68%
taught wrong order → every "dirty-TV KADID crater" is contaminated. (2) the oracle monotonicity
ceiling is 0.980 (cvvdp's own step-inversion on real reversals) → chasing 0.99 flattened real
reversals. (3) full panel *reverses* the aggregate-SROCC story (cvvdp_w1 beats A on 4/6). 
**Status: clean pairs rebuilt; A stays (CID22 gap real ~4 SEM).** Cite: `project_tv_monotonicity_pareto.md`.

**3.8 AIC-3 "0.80 vs 0.96 CVVDP gap" = 5-image-subset ARTIFACT.** CVVDP's 0.96 was 5 images;
on the full 600-pair set CVVDP is 0.79 pooled / 0.93 per-ref and our bake is 0.9475 per-ref
(ABOVE raw CVVDP). **Status: SUPERSEDES "need CSF features to close the CVVDP gap"** — prevented
a wasted feature-engineering investment. Cite: `aic3_cvvdp_feature_spike_2026-05-25.md`.

**3.9 kadis-hdr breadth falsification — FIRST confounded (regime), THEN clean (§0.5).** §8.12
failed the UPIQ gate → "breadth falsified," but the kadis features were u8-shell while the
family is PU-linear (a "regime chimera"). §8.13 WITHDREW it; re-extracted PU-linear, re-ran →
§8.15 clean re-falsification (fails both UPIQ strata). **Status: conclusion SURVIVED after the
confound was removed; the original numbers were invalid.** Regime-consistency now enforced.
Cite: `project_profile_b_hdr.md:370-390`.

**3.10 B near-lossless "intrinsic to linearity (falsified 3 ways)" → MISCALIBRATED WINSOR.**
*Thought:* B's linear projection can't rank near-lossless; falsified 3 ways → "A is the nl knob."
*Actual:* B's shipped winsor bounds turned 245/372 features CONSTANT on nl content — self-inflicted.
Clean bounds: nl per-img SROCC 0.286→0.886, CID22 +0.049, at ~0 MOS cost. **Status: RE-FRAMED,
fixed + shipped (inclusive-winsor B).** NUANCE: a later decomposition (`98e9f395`) found the
dominant cause was ~80% feature-vanishing + ~15-20% winsor, so the "winsor is THE cause" framing
was itself partially walked back — but the fix stands. Cite: commits `a6edbced`→`aaa1ecac`→`98e9f395`.

**3.17 CVVDP-scalar "dead end" = the `cvvdp_log_norm` TARGET SHAPE, not cvvdp (2026-07-15).**
*Thought:* training toward CVVDP's scalar output is a dead end (V41: CID22 0.66 vs ssim2 0.88) —
"emulating CVVDP's output ≠ its CSF mechanism." *Actual:* on safesyn cvvdp is 100% present,
learnable (feat→cvvdp 0.987), and agrees with ssim2 (0.984). Same de-poisoned pipeline: **raw
`cvvdp_score` → CID22 0.85** (fine), **`cvvdp_log_norm` → CID22 0.58** (craters). log_norm is
rank-identical to score (SROCC 1.0000) but **exponentially expands the near-lossless tail** (37%
of pairs in cvvdp [9.9,10] → log_norm [27.75,100], std 20), so MSE loss over-weights the top and
under-fits the rest. **Status: RE-FRAMED** — the blanket "cvvdp_* scalar is bad" is too broad; raw
`cvvdp_score` is viable, and per Mohammadi 2025 (cvvdp best in HF/near-lossless) it's the indicated
lever for the HF regime ssim2 can't rank. **Guard: never MSE-regress a log-expanded target; use
raw or a rank loss.** Shipped bakes unaffected (all ssim2-trained). Caveat: V41's exact target
column not re-confirmed from its log; the de-poisoned probe isolates the shape effect regardless.
Cite: `§8.36`.

### Tier 3 — contained (caught at/near ship)

- **3.11 V39 output-spline upper-extrapolation** — validate-side extrapolated uncapped above the
  top knot while runtime caps ≤100 → dial-p95 artifacts of 321-504; "bit-exact" claim false above
  the knot. **RESOLVED** (`5d4978db`, capped for parity). Validate-side only; no bake affected.
- **3.12 V39 "universally better than V0_3" = actually 5/6** — V0_3 significantly wins AIC-4
  (Δ+0.023, p=0.001, invisible under single-SROCC CI). **RE-FRAMED**; V39 shipped anyway (right
  call for the 5 compression holdouts). Cite: `v5_vs_v03_comparison_2026-05-25.md`.
- **3.13 Profile-A "reproduction regression" = TRAINER CODE DRIFT.** Recipe re-run on current
  main gave a collapsed 57KB bake; at the pinned tree `e9442678` it reproduces byte-identically.
  **RESOLVED** — landed a `[training].trainer_commit` gate. No bake affected.
- **3.14 konjnd-agg 2-layer gradient "bug" = MALFORMED FD TEST.** f32 forward floored the central
  difference (ε=1e-6) + a pure-relative gate unbounded as grad→0. Gradients were correct.
  **RESOLVED** (`fix #35`, ε=1e-2 + atol/rtol gradcheck). A *separate* real ~3% pool-head issue
  found + fixed same investigation; no bake affected. Cite: `CLAUDE.md:20-38`.
- **3.15 zenjpeg picker "SROCC 0.341" = trained on the DIAL ALONE.** All 108 `feat_*` were 0.0
  (empty→0.0 join imputation); the picker never saw image content. **RE-FRAMED** as the zq-only
  baseline; re-joined fail-loud. (zenjpeg codec, not core zensim.) Cite: `project_picker_retrain_zenpredict_caps.md`.
- **3.16 konjnd val "negative polarity bug" = STRUCTURAL anti-correlation + naming collision.**
  val/konjnd SROCC is structurally negative (at-PJND pairs) → `|SROCC|` is correct; and
  `human_score` means TWO quantities (train active-mix [0,1] vs val mean-PJND [22,70]) → a default
  `--target-column human_score` fits the wrong one. Interpretation trap, not a code bug. Cite:
  `feedback_konjnd_human_score_two_columns.md`.
- **3.18 kadid/tid `iwssim`/`ssim2_gpu` leaked/misjoined in canonical — RESOLVED 2026-07-15.** The
  V39-#8 bug — `iwssim` = a byte-identical copy of `human_score` (target leak), `ssim2_gpu` =
  ref-vs-ref misjoin — was documented as "fixed in `*_fixed_2026-05-25.parquet` siblings" but those
  were never promoted; verified still-live 2026-07-15 (`iwssim==human_score` max|Δ|=0.00e+00; `ssim2`
  pinned 93.5/100/100). **FIXED by promoting the verified siblings** (`promote_fixed_kadid_tid_2026-07-15.py`,
  user directive): `canonical-2026-05-21/train/{kadid,tid}.parquet` now carry real iwssim
  (SROCC-vs-MOS +0.850/+0.779) + real ssim2 (spans [−367,100]/[−96,90], SROCC +0.813/+0.846); all
  376 preserved cols (372 features + human_score/cvvdp_*/pjnd_target) byte-identical → **zero feature
  drift**. Corrupt originals preserved at `<c>.CORRUPT-v39bug.pre-2026-07-15.bak.parquet` (+ R2
  `_archive/` + Tower). Manifest entries [5]/[9] updated (new sha256). R2 + Tower re-synced. Shipped
  bakes were always SAFE (trained on `human_score`, never kadid/tid iwssim). Cite:
  `benchmarks/column_audit_2026-07-15.md`, `…§8.37 A`, `scripts/canonical_corpus/fix_kadid_tid_apply_scores.py`.
  **INDEPENDENTLY CONFIRMED 2026-07-15 (same day, different route) — BOTH corpora:** from-scratch GPU
  re-extractions (`zenmetrics batch --metric ssim2-gpu`, no reference to the parquet) reproduce the
  promoted columns exactly — **KADID +0.8133** vs documented +0.813 (10,125 pairs, range
  **[−367.2, 100.0]** vs documented [−367,100]) and **TID +0.8460** vs documented +0.846 (3,000 pairs).
  On KADID, row-for-row: **SROCC(fresh, canonical) = 1.000000**, mean |Δ| **1.4e-5**, 10,062/10,125 rows
  identical to <1e-4 (the rest ≤0.001 = f32/GPU noise). The promotion is correct, not merely
  self-consistent. Byproduct worth knowing: **kadid/tid ssim2 + cvvdp + iwssim already exist in the
  canonical train parquets** — recomputing them is redundant; only **butteraugli** is genuinely absent
  from the canonical schema. Fresh reference-metric panel (for the dashboard, `refmetrics/`):
  KADID ssim2 +0.8133 / cvvdp +0.8339 / butteraugli↓ +0.5431; TID ssim2 +0.8460 / cvvdp +0.8531 /
  butteraugli↓ +0.6622 — butteraugli is weakest on both, as expected for a compression-tuned metric on
  KADID/TID's ~95% non-compression distortions (a live confirmation of the §2 "KADID/TID are integrity
  guards, not compression signal" framing). Gotcha for re-runs: TID's `reference_images/*.BMP` are not
  decodable by zenmetrics — use `reference_images_png/` (pixel-identical, max|Δ|=0).
- **3.19 IW/masked HF-moment features explode on non-photographic content — unbounded energy +
  a `1/n`-vs-`Σw` normalization bug (2026-07-15).** bigcodec IW/masked reached 5.8e6 (vs photographic
  p99.9=0.48) — traced to (a) `iw_art4`/`iw_det4` being **unbounded, un-per-image-normalized** edge
  energies in XYB (sharp synthetic edges → 7 orders above photo; the masked block blows up
  identically → energy, not weight, is the primary driver), and (b) a genuine secondary bug: the
  extractor normalizes IW moments by `1/n` (`streaming.rs:424`) while the reference `iw_pool.rs:399`
  uses `1/Σw` (`shipped = ref·mean_w^0.25`, ~1.5–2× inflation on high-activity). Not a decode
  corruption (`o_9292.png` is valid in-range). Not a pixel-ZERO-TOLERANCE issue (streaming↔full-image
  agree; both diverge from the unused reference). Remedy: winsor guard (shipped, §8.35) is load-bearing
  and sufficient; the `Σw` switch + per-image energy normalization is a scheduled full-retrain fix.
  Cite: `…§8.37 B`.
  > **⟳ CORRECTED + QUANTIFIED 2026-07-16 (`benchmarks/ssim_moment_explosion_2026-07-16.md`).**
  > Two of this row's claims are wrong; the headline 5.8e6 is right. (i) The
  > exploding features are **`iw_ssim_4th`/`masked_ssim_4th`** (the SSIM-map
  > higher moments, XYB **ch2**, finest scale), NOT `iw_art4`/`iw_det4` — the
  > edge features measured 0.02–0.09 on the shipped extractor. (ii) The `1/n`-
  > vs-`Σw` weight bug is NOT even a "secondary" driver: the shipped weight is
  > `1+4a` and `mean_w` spans only **1.03–1.27×** (measured, full instrument),
  > so fixing it does ~nothing. The ACTUAL cause is the per-pixel SSIM
  > `d = (1 − num_m·num_s/denom_s)·mask` having a `.max(0)` floor but **no upper
  > cap**, and `num_m = 1 − (mu1−mu2)²` having **no C1** → unbounded-negative on
  > high-magnitude chroma → `d` to millions, L4-amplified (worst row: 4th/mean =
  > 630×). Denominator-cancellation was tested and FALSIFIED (f32 floors it at
  > ~C2, 1.2×). Fix = add a C1 to the luminance term (principled) or cap `d`
  > (cheap); both need a re-extract + full panel + sign-off. The winsor guard
  > clamps the symptom; it is why B ships.
- **3.20 The `mix_cv*` (cvvdp×iwssim) target columns are POISON; raw cvvdp is ceiling-saturated
  (2026-07-15, completes 3.17).** `mix_cv50_iw50` → CID22 0.705 (poison): it inherits cvvdp's
  near-lossless saturation AND is log-expanded (safesyn tail 2.06), and is absent on cid22_train
  (0% cov). Raw `cvvdp_score` is SATURATED on codec pairs (safesyn 37%+ in [9.8,10], tail≈0.0004) →
  MSE can't discriminate the top → 0.849; log_norm re-expands it → 0.597. cvvdp is damned as an MSE
  target either way, but rank-fine (SROCC-to-raw 1.0) → use rank/pairwise on raw cvvdp for the HF
  band only. `ssim2_log_norm` (tail 0.04) trains fine (0.880) → the confound is saturation-specific,
  not "log_norm." Cite: `benchmarks/column_audit_2026-07-15.md`, `…§8.37 A`.

- **3.21 raw-cvvdp-rank auxiliary = an AIC-3 lever, NOT a G5/KonJND lever — MEASURED, FALSIFIED
  (2026-07-15).** The §3.17 reframe left open "raw cvvdp is the indicated lever for the HF regime
  ssim2 can't rank (G5)." Tested directly (`cvvdp_hf_probe.py` — deleted 2026-07-29 with the
  falsified campaign, in git history; `ssim2-MSE + lambda*rank(raw cvvdp)`,
  lambda 0->0.6): AIC-3 |SROCC| rises monotonically (+0.005 to +0.017) but **KonJND |SROCC| FALLS
  monotonically (-0.024 to -0.101)** with a small CID22 cost. The two HF holdouts pull opposite —
  cvvdp saturates (~10) where KonJND needs fine PJND discrimination, so its rank signal orders the
  mid-band and de-aligns from PJND. G5 (KonJND floor) is NOT closed by cvvdp supervision — consistent
  with the standing "G5 needs HF *representation*, not supervision" limit. The reframe stands for
  AIC-3/mid-band only. Untested: a separate gated HF head. Cite: `bhdr_improvement_split_lineage §8.38`.

- **3.22 JXL near-lossless encoder bug — training corpora NEVER contaminated; 33 eval-grid cells
  were — AUDITED + PURGED (2026-07-15).** *Thought-why:* "the jxl encoder had a bug at the highest
  qualities and we never cleaned the parquets — there should be replacement rows somewhere," implying
  a training-data contamination needing a purge + re-join. *Actual-why:* **the training corpora never
  sampled the broken zone at all.** The bug (jxl-encoder, two layers: quantized DC stored as `i16`
  saturated at fine distances → corrupt encode at source, content-dependent on `|DC|>32767`; then the
  header lied `modular_16bit_buffer_sufficient=true` so conformant decoders truncated the widened DC
  and desynced the DC ANS stream) only fires at **butteraugli distance ≤0.02** (≈JXL-native quality
  ≥99.7 — the "0.3 quality points" phrasing; `(100−q)/10 = d`). **Distance ≥0.03 is byte-identical /
  hash-proven at EVERY date**, so it needs no date bound and no re-encode. Fixed `008499e1` (i32 DC +
  interim 0.03 floor, zenjxl#18) → `eeb52735` 2026-07-06T06:09Z (`force_modular_32bit`, floor removed,
  jxl-encoder#94); re-verified 2026-07-14 to d=0.001 across jxl-rs/jxl-oxide/zenjxl (`a0f7e870`, CI
  green) — nothing remains broken. MEASURED clean, three independent ways: `safesyn` 26,362
  `zenjxl-e7` rows q5–q100, q=100 → ssim2 med **95.13** / butteraugli med **0.229** (monotone across
  all q, never below the 0.03 floor); `cvvdp_iwssim_LARGE` zenjxl distances are only **{0.5,1,2,5}**
  (16–250× above the boundary; LARGE stayed 73,300 rows); score sidecars JXL q=10..90 → d≥0.5. Also
  verified clean by direct R2/duckdb query: `canonical-picker-2026-06-27/zenjxl_lossy` +
  `canonical-picker-2026-07-01-zensimA/zenjxl_lossy` (q∈[5,90], generic-quality-only — structurally
  never resolves to native d<0.03), `zenjxl_lossy_hqfill_2026-07-01.parquet` (62,958 rows, d∈[0.05,1.3]),
  `hqfill_7metric_sidecar_2026-07-02.parquet` (62,173 rows, d∈[0.05,1.3]).
  **⚠ CORRECTION (2026-07-15, same day):** an earlier revision of this entry said LARGE's 2026-05-18
  "fresh jxl" 8-distance sweep "was PLANNED but BLOCKED and never landed (`c17447f5`)" — that was read
  off the commit message and is **WRONG**. Direct R2 verification (`aws s3 ls --recursive --summarize`
  on `s3://zentrain/multi-codec-2026-05-18/omni/`) shows the sweep **COMPLETED**: 112/112 objects dated
  2026-05-19, 24,800 cells; its 6,400 zenjxl rows span distance **[0.1, 10.0]**. It was never *merged*
  into canonical LARGE — "blocked" described the merge, not the sweep. The clean verdict is unchanged
  (clean either way, by two independent routes), but do not repeat the "never landed" claim.
  **Near-miss worth knowing:** `jxl-dense-20260530` (2k k-means sources × 44-distance ladder **starting
  at 0.025**) is the one sweep that WOULD have mass-produced contaminated rows — **confirmed never
  launched** (`s3://coefficient/jobs/jxl-dense-20260530/` holds only the 2 prep files from 2026-05-30;
  no `chunks/`, `done/`, `features/`, `variants/` ever appeared — blocked on red sweep-image CI,
  `9d8f73a5`, never resumed).
  **Shipped bakes are UNAFFECTED — no purge of training data was
  needed or done.** What *was* contaminated: `dial_grid_372col_2026-05-29` (eval-only) carries 33 JXL
  cells at d=0.025, built ~5 weeks pre-fix — mean feature-L2 **4.011** / max|feat| **59.29** vs the
  healthy d=0.05..0.35 ceiling **1.56** (L2 0.109→0.246), a **37× distortion explosion at the LOWEST
  distance**, backwards from the monotone trend; 4 of 33 unambiguously broken, all high-DC graphic
  content exactly as the DC>i16 mechanism predicts. Purged into
  `dial_grid_372col_2026-05-29_quarantined_v2.parquet` (4,424 rows, sha256 `6546c43e…`, R2
  `s3://zentrain/eval-grids/`), which drops both this and the §w11 webp garbage; originals preserved.
  *Caveat:* LARGE's extreme feature values are the **separate** §3.19 IW-explosion on `gen-chart`
  content — NOT this bug; do not conflate. Cite:
  `benchmarks/jxl_nearlossless_contamination_2026-07-15.md`, `benchmarks/jxl_nearlossless_dial_2026-07-05.md`.

- **3.23 konjnd-dense as a training corpus = a KonJND MEMORIZATION lever, NOT a G5 lever — MEASURED,
  FALSIFIED (2026-07-15).** *Thought-why:* "we have an HF/near-threshold corpus (konjnd-dense, 20,160
  rows) sitting unused in canonical — adding it to the blend should close G5 (the KonJND ≥0.70 floor)."
  *Actual-why:* it lifts KonJND only by turning KonJND into **train==val**. Measured on the honest
  blend (`{safesyn:1, bigcodec:1.5, kadis:0.3}`, ZERO human eval-corpus data), sweeping
  konjnd-dense weight 0→0.5→1→2 (`blend_search.py --round 5`, seeds 1,7,13,17,23, deterministic —
  reproduced byte-for-byte across two runs): **KonJND |SROCC| rises 0.5143 → 0.5411 → 0.5467** (+0.027
  / +0.032) **while AIC-3 — the one HONEST JND holdout — falls monotonically 0.7908 → 0.7858 → 0.7840**
  (−0.005 / −0.007), CID22 flat-to-down (0.8850 → 0.8854 → 0.8827). So the KonJND gain does not
  transfer to held-out JND: it is memorization of the corpus it now trains on. Even at its best, KonJND
  0.5467 is far under the 0.70 G5 floor.
  **Symmetry with §3.21 — two opposite supervisions, same verdict:** §3.21 pushed raw-cvvdp rank and
  got AIC-3 UP / KonJND DOWN; §3.23 pushes KonJND data and gets KonJND UP / AIC-3 DOWN. The two HF
  holdouts pull against each other under *either* supervision, which is a second independent line of
  evidence for the standing limit: **G5 needs an HF feature REPRESENTATION, not more supervision and
  not more data** (consistent with the already-falsified single-MLP agg-head and regime-routed-ensemble
  attempts). Ship implication: the honest champion (no konjnd) stands; konjnd-dense does not earn a
  slot. Note the composite ranks `r5-hon+kon0.5` first (1.404) **only because the konjnd leg is now
  cheat** — a live example of why the honesty-per-(bake,corpus) matrix is load-bearing. Cite:
  `benchmarks/blend_search_r5_2026-07-15.tsv`.

- **3.24 the weak near-lossless (high-tail) rank is NOT the winsor bound — MEASURED, FALSIFIED
  (2026-07-15).** Enabled by the tail stat added the same day (§ the range-restriction fix): the honest
  champion's CID22 **high-tail SROCC 0.434 sits well below its low-tail 0.642** → near-lossless rank is
  the genuinely weak end. The width-10 B9 band could not show this (range-restricted, see
  `[[project-band-srocc-range-restriction]]`). *Thought-why:* "`winsor_pct` fits lo/hi percentiles on the
  TRAINING distribution; bounds too tight clamp near-lossless features to constants (feature-vanishing)
  — exactly the Part-7 mechanism, where recomputing clean p1/p99 bounds lifted B's near-lossless per-img
  SROCC 0.286→0.886 AND CID22 +0.049." *Actual-why:* **flat.** Sweeping `winsor_pct` 0 → 0.02 → 0.05 →
  0.1 → 0.25 → 0.5 → 1.0 (`blend_search.py --round 6`, seeds 1,7,13) moves high-tail only
  0.426/0.438/0.442/0.437/0.433/0.439/0.446 — a **0.020 spread with no monotone trend across a 50×
  change**, and **winsor=0 (no winsorization at all) gives 0.426**, i.e. no better. Low-tail equally flat
  (0.628–0.642). The pre-registered falsification fires.
  **Why this does NOT contradict Part 7:** Part 7's defect was a **fit-corpus/eval-regime MISMATCH** —
  B's *shipped* bounds were fit on a corpus that EXCLUDED near-lossless, so 245/372 features went
  constant there and a recompute recovered them. The honest MLP's bounds are fit on its own training
  data, which already contains near-lossless content (safesyn q100 → butteraugli 0.229), so there is
  nothing for a looser bound to recover. Winsorization is not a near-lossless lever when the fit corpus
  already covers the regime — it is only a lever for repairing a mismatch. (Winsor still slightly helps
  CID22: 0.8862 at the 0.1 default vs 0.8802 at 0 — outlier guarding, not near-lossless.)
  **⚠ SELF-CORRECTION (same day, measured within the hour):** the "next suspects" first written here —
  (a) near-lossless feature-vanishing, (b) training-target saturation, (c) CID22 top-end MOS noise, with
  (b) called "most likely" — were **two wrong inferences stacked on an unverified premise.** Both are now
  MEASURED and (a)+(b) are refuted; (c) is right, and is better described as an intrinsic difficulty floor:
  - **The premise "high-tail = near-lossless" is FALSE.** CID22's high-tail (top 30% by MOS, MCOS
    81.2–91.9) has ssim2 p10 **75.4** / p50 **81.7** / p90 **88.1**, and **0.0000 of it is above ssim2 95**.
    It never enters the near-lossless region at all — it is simply the better half of CID22's ordinary
    compression range, which training covers densely. So (a) cannot apply.
  - **(b) is REFUTED: the target is NOT ceiling-saturated.** safesyn's actual target is `ssim2_gpu`
    (not `human_score`): p50 **68.73**, p90 **91.15**, p99 **95.89**, only **1.86%** above 95 and **0.35%**
    within 2 points of max. That is nothing like the cvvdp saturation §3.20 measured (37% piled in
    [9.8,10]). There is gradient at the top; `ssim2_log_norm` is not indicated here.
  - **(c) is the answer, and it is a DIFFICULTY FLOOR, not our defect.** **ssim2 itself** shows the same
    asymmetry on the same split: ssim2's own low-tail SROCC **+0.649** vs high-tail **+0.463** — versus our
    honest champion's **+0.642 / +0.434**. Our bake tracks an independent, well-established metric to
    within **0.007 (low-tail) / 0.029 (high-tail)**. Ranking stimuli humans rated MCOS 81–92 is
    intrinsically harder than 28–64 (subtler differences, more MOS noise per unit of signal) — the tail
    gap is a property of the task and the corpus, not of the model.
  **Standing lesson:** a low tail/band number is only evidence of a model weakness *after* you check what
  the tail actually contains and what an independent metric scores on the same split. Both checks are
  cheap; skipping them produced a plausible, well-argued, wrong causal story that survived one commit.
  Cite: `benchmarks/blend_search_r6_2026-07-15.tsv`.

---

## 4. Live doc conflicts + inaccuracies to fix (flagged for user, NOT yet edited)

Per the "batch doc corrections, ask before editing" rule, these are surfaced, not changed:

1. **`CLAUDE.md:302-318` "⚠️ V0_8 CID22 SROCC IS INFLATED" box is UNVERIFIED and conflicts with
   the post-revert rules at `CLAUDE.md:2036-2057`.** The box asserts the V0_8 leak (22/49) as
   fact; but it used the same loose d≤16 threshold that §3.2 shows mostly-false-positives, was
   flagged "re-audit at d≤10," and never was. **Proposed fix:** mark the box UNVERIFIED / re-audit
   at d≤10, or run the d≤10 audit to settle it.
2. **`profile_b_methodology_2026-07-12.md §1` names the shipped BHdr as `anchored2` (`373eac56`)**,
   but `profile.rs` + `reproduce_bhdr.sh` ship `7d7f2123` (cvvdp-mix). The methodology §1 table is
   **stale** (its own §3b admits the promotion). **Proposed fix:** update §1 to `7d7f2123`.
3. **The BHdr commit `6b4aae03`, CHANGELOG, and `profile.rs` comment still assert the retracted
   "significant UPIQ win"** (§3.6). **Proposed fix:** annotate with the §7 audit correction (claim
   is non-inferiority, not improvement).

---

## 5. imazen-26 + the diversity gap + the informed retrain plan

**imazen-26** = the imazen-org codec-corpus (`~/work/codec-corpus/imazen-26/`): **2,160 images
across 21 content categories** deliberately spanning the hard cases — real photos (own
phone/camera), Unsplash stock, museum artwork (CC0), **born-digital gov documents (text/tables/
charts)**, **document/manuscript scans incl. bilevel CCITT patents**, **synthetic charts +
line/polygon patterns**, **402 UI screenshots (mobile+web)**, and **910 AI-generated graphics**.
Split = **LSD-origin** (last digit of origin id: {0,2,4,6,8}→train, {1,3,5}→val, {7,9}→test;
`zenmetrics/scripts/picker/origin_split.py` is the one source of truth). dHash-clean vs CID22-49
at d≤10 (min d=12) → adding it does not contaminate the gold holdout. ("26" reads as the 2026
corpus, not a category count — inference, not documented.)

**The gap (why this matters).** The current MLP training set is **photographic-dominant with
synthetic/analytic distortions**: safesyn = synthetic tiles from photo sources; cid22_train =
CID22 photos; KADIS = Pixabay photos with *analytic* distortions. Under-represented: **screen/UI,
line-art/vector/charts, document/scan/bilevel, AI-generated graphics, artwork — and REAL
modern-codec distortions** (the RD the dial must actually rank).

**Best gap-filler:** `canonical-picker-2026-07-01-zensimA` (5 SDR datasets, 5.74M cells) — or
the ready-made LSD-split **`bigcodec_hqdedup_traindigits_2026-07-02.parquet`** (2.32M deduped
rows; verified: `ref_basename` + `human_score`=ssim2/100 ∈[0,1] + 372 `f`-feats, 608+ imazen-26
origins). It fixes both gaps at once (content diversity + real codec distortions). **NOT** KADIS
(more photographic/analytic bias) and **NOT** kadis-hdr (HDR domain — BHdr track).

**Retrain plan that HONORS this history (task #17):**
- **MLP, not linear** — bigcodec poisons a linear CID22 (§2) but MLPs absorb it (§1). This is the
  scientific rationale for doing diversity as the §8.33 MLP. **Still MEASURE** CID22 with it in.
- **Target = ssim2** across all groups (normalize the [0,1] bigcodec/safesyn/cid22 targets ×100 to
  KADIS's ssim2 scale, then standardize) — *not* `score_zensim` (that would distill profile A).
- **Honor the HQ-saturation confound** — ssim2 labels saturate above 0.85 (cvvdp-agreement 0.48);
  densifying HQ-ssim2 *made bakes worse*. Downweight the HQ band or lean on the fill4 cvvdp column
  there; use ssim2 only in its reliable ~30-85 band.
- **Valid selection instruments (§0)** — training-side selection (never peek CID22, never the
  `cid22_train` axis which anti-correlates), full Mohammadi panel + two-panel dial, paired
  significance. Collapse-gate the seed fan.
- **Content-dedup mandatory** — 22.2% of raw canonical rows are byte-identical knob-no-ops (bigcodec
  is already hqdedup, but verify C10 <1%).

---

## 6. Standing guards this history produced (the scar tissue)

- **Two-panel eval mandatory** (rank + dial; G1 dynamic-range + monotonicity) — from §3.4/§3.5.
- **Full Mohammadi panel, SROCC-only BANNED** (SROCC + PLCC + KROCC + OR + PWRC + Z-RMSE) — from §3.3.
- **Paired significance + Westfall-Young family correction; a holdout is never a selection axis** — §3.6/§3.12.
- **Pre-registered gates + one holdout look** — from the post-selection incidents.
- **`_validate_metric_columns()`, grid quarantine, fail-loud joins, assert-features-non-constant** — §3.1/§3.4/§3.15.
- **Regime tags (`--features-name`), runtime-parity spline caps, `[training].trainer_commit` pin** — §3.9/§3.11/§3.13.
- **`|SROCC|` for konjnd; `pjnd_target` not `human_score`** — §3.16.
- **CID22-49 / AIC / UPIQ / SDR25 are holdout-only; UPIQ-380 is BURNED for the BHdr linear family.**
- **UPIQ-panel every BHdr candidate before ship** — §3.13 (hdranch3).
- **Keep bigcodec/imazen-26 OUT of *linear* mixes; MLPs only.** — §1/§2.
- **Non-photo standing gate** (added 2026-07-15): `bake_verdict` scores a held-out imazen-26
  diverse-content axis (`nonphoto` corpus) by default + a **G-NP gate** (SROCC <0.88 = content-weak,
  <0.50 = crash). Every photographic-only bake (B/A/§8.33) flags content-weak (~0.86) — the
  blindness the 6 photographic corpora can't see. Dashboard: `scripts/v_next/bake_dashboard.py`.
- **NO GRACEFUL SKIPS in eval** — a missing corpus fails loud with an R2 fetch hint; all 7 eval
  corpora are mirrored to `s3://zentrain/eval-corpora/` (the exact `2026-05-15-full-features`
  files `bake_verdict` reads). Never silently drop an axis.
- **Winsorize before the scaler on any bigcodec/imazen-26 train** (combined `[p0.1,p99.9]`,
  baked as `WinsorP99` transforms) — de-poisons the IW-block extraction garbage (§3.4-family). §8.35.
- **Never MSE-regress a log-expanded target** (e.g. `cvvdp_log_norm`) — use raw or a rank loss. §3.17.
- **Run `check_table_integrity.py` on any mix before training it** (added 2026-08-04) — the
  structural gate the project went two years without. Covers target finiteness/range/degeneracy,
  feature finiteness + constant columns + unguarded tails, duplicate rows, teacher-twin row
  correspondence + agreement, and eval leakage by reference identity. `--mix-from-spec <bake>.
  bin.spec.json` audits a whole recipe from its embedded repro. §3.18.

### §3.18 — the SOTA-944 mix data-integrity audit (2026-08-04)

Report: `benchmarks/data_integrity_audit_2026-08-04.md`. Plan:
`benchmarks/sota944_campaign_2026-08-03.md` REGISTERED APPENDIX G.

**Thought-why:** after the KADID inversion (Appendix F) — found by the project's FIRST
orientation gate, on its first run, six weeks late — the open question was whether the rest of
the mix hid similar defects, and whether the mix had ever been chosen on evidence.

**Actual-why / what was found:** the mix is clean where it would have been catastrophic and
dirty where nobody was looking.

- **Eval leakage is ZERO** — 10 training legs × 10 eval corpora, every pair empty. CID22
  explicitly: `cid22_train`'s 201 references ∩ the `cid22val` 49-reference holdout = ∅, and
  `cid22_train`'s target is `ssim2_gpu`, never human MOS. The VALIDATION-ONLY rule holds at both
  the reference and the target level.
- **Content-dedup WAS applied** — max duplicate-row mass 3.66% (bigcodec), far below the 22.2%
  documented pre-dedup rate. The mandate in the content-dedup section is being honored.
- **`tkadis` is a training-signal conflict** — the kadis teacher twin ranks its own rows at
  signed SROCC **+0.2485** vs the base kadis target, with a systematic +0.579 median offset, at
  **7.87% of sampling mass against the base leg's 2.36%**. The clip/affine explanation is
  FALSIFIED (0.05% clipped; SROCC identical on the unclipped subset; SROCC is affine-invariant).
  It is a genuine teacher generalization failure on the KADIS distortion distribution.
- **Effective sampling mass is ~uncorrelated with row count** and had never been written down.
  The trainer picks a group by `train_w / Σ train_w` then draws rows uniformly *within* it, so
  pair share is **independent of row count**: `konjnd_bpg` is 1.03% of rows → **18.90%** of
  pairs; `bigcodec` is 26.71% of rows → 7.87%. **The weights ARE the mix.** Table:
  `benchmarks/data_integrity_sampling_mass_2026-08-04.tsv`.
- **9 of 11 legs cannot be orientation-checked against humans at all** — they carry metric
  (ssim2) or teacher targets. KADID's inversion was catchable *because KADID is one of only two
  legs where an external check was ever possible*. `check_target_orientation.py --provenance`
  now reports this explicitly so the gap cannot be mistaken for a clean bill of health.
- **The canonical promotion drops the quality key**, so ladder monotonicity is unauditable for
  safesyn / cid22_train / konjnd_bpg (17.5% of rows). Where the key survived, monotonicity
  PASSES cleanly (bigcodec: 9,228 ladders, median SROCC(q, target) +0.963; kadis: per-source
  median −1.000 against severity). **Carry the quality key into every canonical table.**
- **The 39 never-populated feature slots are an EXTRACTOR property, not a data gap** — zero
  slots are constant in only *some* groups. Independently reproduces `bake_contrib`'s count and
  classifies it. Prune candidates only (the `n_inputs()` vs `caller_input_width()` hazard, E.9).

**Two audit-side errors caught before publication, recorded so they are not repeated:**
`pair_tie_prob ≠ row_tie_rate` (KADID reads 99.60% by the wrong one, 0.876% by the right one —
the trainer drops *pairs*), and `signed_cbrt` is a tail guard just like `winsor_p99` (f38's 776×
excursion is guarded, not unguarded). Both are now encoded in the gate.

**Registered non-conclusion:** no check disqualified the mix, therefore **the mix-composition
question is UNANSWERED**. Absence of defects is not evidence of optimality; establishing an
optimum needs a weight sweep with held-out scoring, which this audit did not run. Do not cite
this audit as evidence that the mix is good — only that it is not broken.

---

*Maintenance: update this doc whenever a verdict is added, re-framed, or a confound is found.
Prefer editing here in-place over a handoff note. Every claim keeps its citation.*

### §3.24 — imazen-26 root-source correction: wrong estate copy, provenance-derived sharing, manifest split-column fix (2026-08-27)

**Thought-why:** `/mnt/v/imazen-26` was "the imazen-26 SDR estate", so the
2026-08-27 dHash audit indexed it as its refs; `imazen26_manifest.tsv`'s
`split` column was assumed meaningful; content sharing between synthetic-v2
and the eval slices was assumed to need dHash to find.

**Actual-why:** that directory is the INSPIRATION/collection copy (the wsl box
had already quarantined its copy as `imazen-26-inspo`; this box's rename had
never happened). The canonical estate is the **`imazen/imazen-26` git repo**
(CORPUS-MANIFEST membership oracle + canonical split manifests + sha-pinned
variant-set registry) with official png-v3 derivatives; eval/picker origin
`o_NNNN` IS the canonical 4-digit id. The manifest TSV's `split` column was
stale bookkeeping (wrong on 1,239/2,157 rows; header mislabeled `sha256`) —
corrected 2026-08-27 to the canonical last-digit rule (consumers verified;
pre-fix bytes in `.pre-splitfix.bak`). And sharing is **provenance-derivable**:
generator tokens live in the filenames on both sides, and cross-id duplicate
families follow the name grammar — dHash is only the verifier (it also
false-positives on flat/line content and misses non-picker-covered ids).

**Outcome:** chain byte-verified (repo → registered picker set → local
mirrors); sharing = 68 exact-token ids + 166 split-piercing family ids;
realized eval inflation MEASURED ≈0 at certain tier AND the 25–32% upper bound
(median Δ +0.0043, max 0.0143, nonphoto positive). Annotation
`imazen26-nonphoto-sharing-provenance-2026-08-27` in force; exclusion = board
D1. Records: `benchmarks/imazen26_dhash_audit_2026-08-27.md`, imazen-26
`benchmarks/split_crossid_dupes_2026-08-27.md` + `derive_sharing_provenance.py`.

**Rules:** join imazen-26 data by the 4-digit id, never by dHash; derive
sharing/duplicates from provenance first, verify with dHash second; never
index `/mnt/v/imazen-26*`; a by-design content family crosses the split iff
its id offsets flip parity (family-aware split is the structural fix).

### §3.25 — the family-aware purity program EXECUTED (2026-08-28, user decisions)

User calls (recorded in the audit md ★REGISTERED section): structural
family-aware re-slice + full-board rescore (done — 280 rows grafted, 11
ensembles + 2 wrong-regime rows annotated); measure hfnlproxy/instruments
(done — SDR instrument clean at d≤2; HDR instrument: hdr_v3mix carries 7/9
census scenes — judge/training overlap annotated, freeze gates unaffected);
training policy = **purge + family-aware**: `benchmarks/
synthv2_channelA_purge_2026-08-28.tsv` lists the synthetic-v2 files that
leave the metric training lineage for all FUTURE trainings, and future
picker/bigcodec/HDR train views bucket by `split_map_family.tsv` + hold the
HDR instrument scenes out entirely. Existing bakes stand, era-tagged.
D3 one-shot wirings executed (zenjpeg/jxl/svt). D4 decided: hdrgrid stays
era-B. **Rules for every future corpus build:** consume the family split,
apply the purge list, and keep instrument scenes out of training views.

### §3.26 — v1's 372-feature vector width: RESOLVED 2026-08-30 (it was SIZE, not batch)

**STATUS: RESOLVED.** Root-caused, fixed in the owner, and gated —
commit `f9fac41e` (`fix(v1-372 width): reflect-pad at EVERY pyramid entry`) on
`main@origin`; gate `zensim/tests/v1_feature_width_pure_function.rs`; full
record with every measurement `benchmarks/v1_width_defect_2026-08-30.md`.
**The BATCH framing below is retracted in turn** — this entry keeps both
retractions because two different wrong explanations were registered here and
the second one is the more misleading of the two.

**What was seen.** Re-extracting the R1b eval slices at the v1-372 regime
produced RAGGED CSVs: 453 of 6,953 imazen26 rows, 422 of 6,142 nonphoto, 493 of
7,717 hfnlproxy (~6.5 % each) carried **279** feature columns instead of 372.
279 = 3 scales × 3 channels × 31; 372 = 4 scales × 3 × 31 — one whole scale
missing. The rows are not empty: a short row carries ~268 non-zero values, i.e.
a real 3-scale computation.

**Retraction 1 (the original entry).** "v1's vector length is size-dependent —
a rendition too small for the 4th scale emits 3 scales" was rejected on
2026-08-30 because `512x384` appeared among the short rows, the same size
appeared in both sets, and 259 of 957 references carried BOTH widths.

**Retraction 2 (what replaced it, also wrong).** "The width is a function of
the BATCH, not of the pair" — evidenced by the same 453 pairs re-run as their
own batch giving only 33 short and 5 run alone giving 0. **Both numbers fail to
reproduce.** A binary built from the pre-fix tree (`6d0a393a`) on the exact
pair lists gives **5 short of 5 alone, 453 short of 453 alone, 453 short of the
6,953-row batch** — the width never moves, and the row VALUES are byte-identical
across all three compositions (pre-fix and post-fix alike). The likeliest source
of the 33/0 reading is a re-run routed through a `Zensim::compute*` entry (which
reflect-pads, so 0 short) and/or a pair list rebuilt from `ref_basename`, which
§8.5(a) of the R1b doc itself records as **not row-unique**.

**The truth: it is a pure function of `(W, H)`.** The scale walk starts at
`w = simd_padded_width(width)` but plain `h = height` and stops at `w < 8 ||
h < 8`, so a 4-scale pyramid needs **`simd_padded_width(W) ≥ 64 AND H ≥ 64`**
(i.e. `W ≥ 49 && H ≥ 64` below 497 px). The predicate `2 + n_scales(W,H)·3·31`
reproduces the field count of **all 20,812 stored rows with ZERO errors**; every
short row's min side is in `[36, 55]`; the short and full size sets are
**disjoint**. The width asymmetry is why "too small" looked falsified: `54x96` is
FULL (54 → 64 by SIMD alignment) while `96x54` is SHORT, and `62x96` is FULL
while `48x64` is SHORT. The size figures quoted above do not describe the short
rows under any parse: across all three slices they span **13 distinct `(W, H)`
classes**, min side `{36…55}` and max side `{64, 96}` — not 168 sizes spanning
36…1024 — and `512x384` is **not** among them.

**Mechanism.** `compute_with_config_inner` (`metric.rs:3145`, behind every
`Zensim::compute*`) reflect-pads any sub-64 side before the walk. Three entries
did not: `compute_zensim_with_config` (`metric.rs:4800`, `training`) returned a
**silent short vector** (93/186/279 wide, no error) — and **both** v1-372
extractors call it (`extract_features_372col.rs:195`,
`v2_ab_extract.rs:319`), which is exactly why the identical short set came from
both tools and both flows; `compute_zensim_with_ref_and_config`
(`metric.rs:706`, `training`) and `Zensim::compute_with_ref_into`
(`metric.rs:2271`, a **product** API) **panicked** `scale 0 width mismatch`.

**Fix.** One owner for the decision — `metric::needs_pyramid_pad(w, h,
num_scales)` + `min_pyramid_dim_for_scales` + `reflect_pad_for_scales` — used at
all seven pyramid entries. `MIN_PYRAMID_DIM` stays 64; the threshold is now
`num_scales`-aware so `--num-scales 5/6` cannot truncate either.

**Consequence for existing data — MEASURED, and it is benign.** Every canonical
372 parquet is full width with real values; nothing is NaN- or zero-padded:

| table | rows | last-93 block all-zero | non-finite |
|---|---|---|---|
| `cid22_features_372col_2026-05-15` | 4,292 (complete corpus) | 0 | 0 |
| `kadid_features_372col_2026-05-15` | 10,125 (complete corpus) | 0 | 0 |
| `konjnd_features_372col_2026-05-15` | 1,008 (complete corpus) | 0 | 0 |
| `imazen26_test_120k_2026-07-16` | 7,844 | 0 | 0 |
| `nonphoto_features_372col_2026-07-15` | 8,241 | 8 (0.10 %) | 0 |

A ragged CSV cannot become a parquet silently — the builder raises
(`ArrowInvalid: Column ... expected length 6953 but got length 6500`) — so a
short row is always either a build failure or an explicit drop, never a padded
row. The 8 nonphoto all-zero rows are the identical-input signature, not
truncation. **Confirmed 2026-08-30 by direct measurement:** a header-level
dimension scan of all **149,195** pairs across the 11 canonical legs finds
**0 pairs** that could have truncated; re-extracting cid22val (250), kon504
(504) and safesyn (250) gives 372 on every row, alone == batch byte-identical,
and the **pre-fix and post-fix binaries agree BYTE-FOR-BYTE** on all 1,004 rows.
(Unrelated and pre-existing: a fresh extraction differs from the *stored*
2026-05-15 / 2026-08-29 tables on most slots — that is the §8.5(b) extractor
drift, not this defect; the two binaries agreeing byte-for-byte is the proof.)

**The 944 regime does not have this problem.** The folded/append path emits a
fixed width, so R1b's 944 tables are ragged-free by construction. Measured:
`foldapp2pools` on the 453 sub-64 pairs emits 946 columns on every row, and the
pre-fix and post-fix CSVs are BYTE-IDENTICAL.

**Where it did land, still open.** The three R1b eval slices are the only place
sub-64 renditions met the defective extractor. `build_r1b_samepair_roots.py`
dropped the 1,368 short rows (counted, never silent), so
`r1b-samepair372-2026-08-30` is a size-correlated 6.5 % restriction of its
population, and `r1b-372root-2026-08-30/` carries **three dangling symlinks**
into `r1b-372slices-2026-08-30/`, which only ever received
`ext_konjnd_jpeg_val.parquet`. Full-width replacements now exist as CSVs at
`/mnt/v/output/zensim/v1width-fix-recheck-2026-08-30/` (sha-manifested,
`build_commit` stamped, with the exact affected-pair lists); promoting them to
parquet and re-cutting the same-pair roots at 6,953 / 6,142 / 7,717 rows belongs
to the keyed-rebuild lane and is REGISTERED, not executed.

**Guard.** `v2_ab_extract` refuses to write a table whose row count differs from
the pairs count and exits 3 (`ZENSIM_AB_ALLOW_MISSING=1` is the caller's visible
opt-out). That guard catches a DROPPED row; the new width gate catches a SHORT
one. Record: `benchmarks/r1b_keyed_rebuild_2026-08-30.md` §8.5(d) +
`benchmarks/v1_width_defect_2026-08-30.md`.

### §3.27 — the STORED 372-col masked/IW block was a function of the thread count (2026-08-30)

**Thought-why:** "the 2026-05-15 canonical 372 tables and today's extractor
disagree on masked/IW — three and a half months of extractor evolution."
(§8.5(b) of `benchmarks/r1b_keyed_rebuild_2026-08-30.md`, priced there at
+0.0060 SROCC for B on CID22 and left as an unexplained cross-era gap.)

**Actual why:** ONE commit, and the stored side of the comparison was never
reproducible in the first place. `2dab8f30` (2026-05-17,
*"principled per-channel H-blur activity for masked/IW features"*) replaced the
activity-map reference — which read `bufs.mu1` at strip-**overlap** rows that
the fused V-blur never writes, i.e. whatever the buffer-reuse cascade left there
— with a per-channel `H_blur(src_c)`. Its own message names the blast radius:
*"Affects masked (228..300) and IW (300..372) feature blocks. Basic 228 features
are unchanged."* Combined with `6af83b60`'s pre-fix band layout
(`num_bands = rayon::current_num_threads().min(total_strips)`, made geometry-only
2026-06-09), the thread count chose where those overlap rows fell — so the
pre-fix masked/IW block was **machine-dependent**.

**Measured** (`benchmarks/v1_extractor_drift_2026-08-30.md`; artifacts +
`_MANIFEST.json` at `/mnt/v/output/zensim/v1-extractor-drift-2026-08-30/`):

- **The stored table does not reproduce at its own build commit.** A probe built
  at `58e6f8d8` — the commit `_MANIFEST.md` records for the 2026-05-15 tables —
  run at `RAYON_NUM_THREADS` 1 / 2 / 8 / 28 produces **four different**
  504×372 outputs. T=1 vs T=28 moves **100 % of rows on all 144 masked/IW slots,
  up to |Δ| 0.086**, while basic + peaks have **zero** cells outside the golden
  tolerance. At HEAD the same four runs give **one** md5.
- **Blast radius is exactly masked+IW.** stored-vs-HEAD on cid22val (4,292
  pairs): `f0..155` and `f156..227` **bit-identical** (max_abs 0), `f228..371`
  differ on 100 % of rows (max_abs 0.0374 masked / 0.1235 IW). Pixels are
  therefore identical, and HEAD's with-ref path == HEAD's plain path
  bit-for-bit, so neither decode nor entry path is a confound.
- **Nothing since.** `2dab8f30` → HEAD is **0 cells over tolerance** on
  4,292 × 372 (residual 5.55e-17 on 18 scale-0 IW slots).
- **Era map:** `2026-05-15-full-features` is PRE-fix.
  `ext720-canonical-2026-07-22` and HEAD are the SAME post-fix era (they differ
  only on the 439 rows where the zen_io decoder disagrees with the `image`
  crate, and those differ in basic too). `r1b-pools944-2026-08-30` is post-fix
  but on the FOLD path (the documented `folded720_*` padded-width class).
- **`canonical-2026-05-21/train/{kadid,tid}.parquet` carry the SAME pre-fix
  values** — row-order identical to the 2026-05-15 root on `f0`, `f228`, `f300`,
  `f353`. So the whole training lineage of the 372 era inherits them.

**Product consequence (the reason this is a §3 entry and not a footnote):**
shipped **Profile B** has 23 of its 95 live inputs in `f228..371` and its single
largest-magnitude input is `f353 = iw/s2/c2/iw_mse` (L2 norm 182.4, 2× the next).
Same bake, same pairs, same pixels, matched row sets, two feature tables:
CID22 SROCC **0.87638 → 0.88212**, KonJND **0.54665 → 0.64967**, AIC-3
0.77743 → 0.79410, TID 0.78866 → 0.79691, KADID 0.82008 → **0.80426** (KADID is
B's train==val CHEAT corpus, so a memorization score falling is the expected
sign). Per-pair the **dial** moves by mean **−4.98** (CID22) / **−5.86**
(KonJND) zensim points, 99.9 % / 100 % of pairs by more than 0.5 points, max
17.4. The product API (`Zensim::compute` at `codec_target`) matches the
fresh-root prediction to 8 decimals on 10/10 sampled pairs — so **the runtime B
is not the evaluated B**, and the published B numbers are stored-root values.

**Why nobody caught it in May.** The 2026-05-20 canonical-build audit
(`~/work/zen/_ml-inventory-2026-05-20/10-canonical-build-audit.md`) that
`zensim/CLAUDE.md` cites as "bit-equivalent … no build drift; trustworthy as-is"
**sampled only `f0..f99`** — its own §1 says *"emits f0..f99"* and its tolerance
is `max_abs_diff(extracted_f0..f99, parquet_f0..f99)`. `f0..f99` is entirely
inside the basic block, the one block that did NOT drift. It ran at `fdd1b8f6`
(2026-05-19), already past `2dab8f30`, so a single masked slot in the sample
would have caught this three months earlier. Its §5 softening of the
`DATA_PROVENANCE.md` "semantically incompatible" warning rests on the same
100-column sample and does not extend to `f156..371`.

**Guard.** `zensim/tests/v1_feature_width_pure_function.rs` gained
`v1_372_is_bit_identical_across_rayon_pool_sizes` and
`v1_masked_and_iw_blocks_are_thread_invariant` — rayon pools of 1/2/3/5/8 (and
1/2/4/7/16) on both `compute_zensim_with_config` and `Zensim::compute`, at sizes
spanning several `STRIP_INNER = 32` strips. The extractor was NOT changed and no
tolerance was widened: the pre-fix values were undefined buffer contents, not a
different-but-valid definition.

**Registered, not executed:** rebuild the canonical 372 root at HEAD as a NEW
dated root (cheap — cid22 8.8 s, kadid 14.9 s, tid 4.2 s, konjnd 2.9 s, aic3
14.2 s at 8 jobs) rather than overwriting in place; re-verdict every
372-input `uses_f156_371` board cell; re-extract B's training legs (safesyn
196,086 + cid22_train 17,611 + kadid 10,125 + tid 3,000 + `hdr_v3mix`) and
consider a retrain — a fleet wave, not a step. `aic4` cannot be refreshed: its
source CSV under `/mnt/v/backups/...` no longer exists on this box.

**Separate defect found in passing, NOT fixed:** `zensim-validate --extract-only
--format tid2013` yields **2,880 of 3,000** TID pairs today — 120 rows silently
dropped on decode/extract failure, surfaced only as a `2880 valid pairs` count.
Same "silent skip" class already documented for `dataset_metric_baseline`.

### §3.28 — the DATED current-extractor 372 eval root, and the era shift priced per model (2026-08-30)

**Thought-why:** "§3.27 declared the stored 372 tables stale; rebuild the root
and re-verdict the lineage." (Registered follow-ups (a) + (b) of
`benchmarks/v1_extractor_drift_2026-08-30.md` §4c.1–2.)

**Actual why + what shipped:** a NEW dated root at
`/mnt/v/zen/zensim-training/2026-08-30-full-features-372/` (`_MANIFEST.json`,
`build_commit ea16c7ee`, per-file sha256 + row accounting + per-corpus ERA +
per-slot drift-vs-stored), never an in-place overwrite — the 2026-05-15 root is
untouched and every published 372-regime number keeps its substrate. Eight of
the fourteen default `bake_verdict` corpora were re-extracted through the SAME
tool the stored table used; six (aic4, nonphoto, imazen26, sdr25, hfnlproxy,
hf_nearlossless) are byte-copies because their distorted material is
bigcodec/R2 encodes or, for aic4, a source CSV that no longer exists — so an
era delta of zero on those is a **structural identity, not evidence**. Plus a
current-era `kon504` ruler + a `kon504/` one-file side root.

**Measured (`benchmarks/eval372_current_root_2026-08-30.md`):**

- **The era shift is MODEL-SPECIFIC, not a constant** — 11 bakes × 2 roots ×
  the default 14-corpus list, same instrument: from **exactly 0.00000** (three
  basic-block-only bakes, all 15 corpora) to **|Δ| 0.489** (`cl_tfm_LQ_MLP`,
  KonJND). No correction factor exists for a published 372-era number.
- **41 ordering flips.** Shipped **B goes 4th → 1st on CID22** among
  {B, `blend_2L_H128`, `cl_tfm_LQ_MLP`, `Ebothg_scr05`}; `cl_tfm_LQ_MLP` goes
  **1st → last** on KonJND (0.761 → 0.272) and on AIC-3; the composite leader
  changes `cl_tfm_LQ_MLP` → `blend_2L_H128`.
- **The 2-layer blend's headline CID22 win over B (+0.004) is an era artifact** —
  on the current extractor it is **−0.0002**, i.e. B is ahead. Its TID +0.062 and
  nonphoto +0.088 survive; its KonJND deficit deepens (−0.038 → −0.145).
- **csiq / live / pipal (built 2026-07-18) are BIT-IDENTICAL to a HEAD
  re-extraction** — 0 slots differing, max_abs exactly 0 on 866 / 779 / 21,800
  rows. §3.27's commit-level era map is now a direct measurement on three
  corpora, and it survives six weeks of extractor work including the blur
  rewrite. Same check on cid22 + konjnd vs the drift lane's own `f9fac41e`
  extraction: 0 cells over tolerance — the `714da506` / `8a98a286` BIT-EXACT
  claims hold on corpus data.
- **The 372 dial + corruption grids are themselves pre-fix-era** (extracted
  2026-05-29 / 05-28, i.e. after `2dab8f30` but before `6af83b60`) and are NOT
  re-extractable: their pairs TSV names the `q<X>.png` decode cache deleted
  2026-06-22 (**2,560 of 2,560 dist paths missing**). Both eras of a
  stored-vs-current re-verdict read the same grid file, so the identical dial
  panel is an identity, not a clean bill of health.
- **kon504 is two files under nearly one name.** R1b's keyed 504 rebuild is
  post-fix (HEAD reproduces it bit-for-bit — W-LIN r7, `df931814`); the 372
  root's `konjnd_jpeg504_372_2026-08-29.parquet` is a byte-exact subset of the
  **pre-fix** konjnd table and carries the full §3.27 signature (masked 34,525 /
  IW 35,254 cells over tolerance on 100 % of rows). Both lanes are right about
  different files.

**CORRECTION to §3.27's source doc (§3b of `v1_extractor_drift_2026-08-30.md`):**
its KADID / TID / AIC-3 rows are invalid — the study's `mkroots.py` aligned
stored↔fresh on `(ref_basename, human_score)`, a key that is **not unique** on
those corpora (stored rows in repeated groups: KADID 64.8 %, AIC-3 100 %, TID
24.2 %), so a whole group collapsed onto one fresh row. Its own `freshroot/`
tables carry the evidence: **aic3 100 distinct rows of 600**, kadid 6,227 of
10,125, tid 2,505 of 2,880; **cid22 and konjnd 0 duplicated**, which is why the
headline CID22/KonJND numbers stand. Positionally aligned: KADID 0.82008 →
**0.80847**, TID **0.78683 → 0.77852**, AIC-3 0.77743 → **0.76501** — the TID
and AIC-3 deltas **change sign**, so "on every genuine holdout the runtime B is
better" is falsified (AIC-3 goes DOWN). Mechanism and decision unchanged.

**Loader fix that came with it (`2d94890c`):** `load_tid2013` forced the
reference stem upper-case while TID2013 ships `i25.bmp` LOWERCASE, so 120 of
3,000 rows named a nonexistent path and were dropped with the loss visible only
as a row count (§3.27's "separate defect found in passing"). Both sides now
resolve through a case-insensitive index and an unresolved label row is FATAL
unless the caller sets `ZENSIM_ALLOW_MISSING_PAIRS=1`. The new root's TID is
**3,000/3,000**, and the recovered 120 rows are bit-identical to the stored
table in basic+peaks — they were absent, not different.

**Registered, NOT executed:** the board is NOT regenerated (ready-to-promote
verdicts at `/mnt/v/output/zensim/eval372-roster-2026-08-30/json/<label>_new.json`);
the dial/corruption grid rebuild needs a decode pass first; aic4 stays pre-fix;
B's training legs stay pre-fix (the retrain is still a fleet wave); BHdr's own
PU-linear HDR route is still unmeasured — the `BHdr_sdr_route` row bounds only
the SDR-route sensitivity (≈0.002 SROCC).

**Registry:** `benchmarks/eval_annotations.json` gains
`eval372-stored-root-thread-dependent-2026-08-30` (invalidated),
`eval372-basic-only-bakes-era-independent-2026-08-30` (annotated — measured
Δ 0.00000, the invalidation does NOT apply) and
`dial372-grid-thread-dependent-era-2026-08-30` (annotated).

**SAME-DAY FOLLOW-UPS (2026-08-30).**
1. **The default `--features-root` WAS flipped** — user directive, executed by
   the board lane in `a25d1b80`: `--regime 372` now defaults to the
   current-extractor root through the single constant
   `zensim_validate::eval_roots::DEFAULT_FEATURES_ROOT_372`, and every verdict
   prints its ruler. A **stored**-era read now needs the explicit flag
   (`STORED_FEATURES_ROOT_2026_05_15`) — the reverse of the discipline in force
   when the roster above was run.
2. **Board-row attribution CORRECTED: 7 of the 9 rows were never stored-root
   reads.** They are `regime:"720"` ext720 reads
   (`board372-row-read-on-ext720-root-2026-08-30`), which this lane reproduced
   independently: board `cl_tfm_corruption_LQ_MLP_s13` vs a fresh
   `bake_verdict --regime 720 --corpora cid22` is **BIT-EXACT on 4,292 pairs**
   and **96.4 points** from a stored-372 read; board **B** and
   **`T_appT_b372_lam1e-3`** are **BIT-EXACT against this lane's stored-root
   re-verdicts** (and 17.4 / 6.09 from the current-root ones), so they are the
   only genuine stored-era board rows and **B's pair is the board's only clean
   era A/B**. The three basic-only rows differ from the stored and current 372
   runs by the SAME amount (0.294/0.294, 0.797/0.797, 0.502/0.502) — an
   era-independent offset, i.e. the folded-720 space, not an era. The two
   affected registry scopes were NARROWED in place to the 2 genuine rows (the
   `kadid-ext-root-inverted` precedent), so no post-fix row carries a false
   era-stale badge. **§3.28's roster science is untouched — it never used a
   board row**; every number came from this lane's own paired runs on the two
   roots.
3. **The provenance gap that forced that archaeology is CLOSED**:
   `bake_verdict --full-json` now records a `features_root` block (path,
   registered era label, root `_MANIFEST.json` sha256 + declared regime, and the
   per-corpus file sha256s it actually read), so "which ruler produced this row?"
   is answerable from the artifact instead of by re-running and diffing
   predictions.

### §3.29 — the default eval root IS the current-extractor root, and what the board rows turned out to be (2026-08-30)

**Thought-why:** "§3.28 left two follow-ups open for governance: flip
`bake_verdict`'s default `--features-root`, and put the current-era verdicts on
the board." (User decision; both executed the same day.)

**Actual-why / what shipped.**

1. **Default flipped, and it now has ONE owner.** The 372 root path was a string
   literal in ten `.rs` files. It is now
   `zensim_validate::eval_roots::DEFAULT_FEATURES_ROOT_372` =
   `/mnt/v/zen/zensim-training/2026-08-30-full-features-372`, with
   `STORED_FEATURES_ROOT_2026_05_15` naming the previous default (which the probe
   and trainer bins now reference BY NAME, so their era choice is deliberate and
   visible rather than accidental). `bake_verdict` and `bake_compare` read the new
   constant; `bake_dial_refit gate`'s default corpus moved to the same-named file
   under the new root. **Nothing was rewritten and nothing was deleted** — the
   2026-05-15 root stays on disk and stays a valid STORED-ERA read; the flip only
   changes what a *flagless* invocation means going forward.
2. **Every verdict is now self-describing.** `bake_verdict` prints
   `bake_verdict: features-root era — <label> :: <path>` before it loads a corpus
   (`eval_roots::era_of`, which labels the four registered roots and reports an
   unregistered one as UNKNOWN rather than guessing). Two tests pin the default
   (`default_features_root_is_the_current_extractor_372_root`,
   `explicit_features_root_overrides_the_default_and_relabels_the_era`) plus two
   in `eval_roots`.
3. **VERIFIED:** a flagless `bake_verdict --bake <shipped B>` and the same run
   with `--features-root <new root>` produce a **byte-identical** `--full-json`
   (sha256 `9596f1bd9f3b2166612866f830e36079675f470870162b73a9ecd2c6d756b2c7`;
   the markdown differs only in the wall-time line), and the flagless numbers
   reproduce §3.28's current-era column exactly (CID22 0.8821166166, KonJND
   0.6496694639, KADID 0.8084738650, TID 0.7785195153, AIC-3 0.7650123966,
   composite 0.8407364995733521).
4. **NOT in the flip:** the 372 dial + corruption grids. They are their own
   pre-fix files (2026-05-29 / 2026-05-28), not re-extractable without a decode
   pass, and stay annotated `dial372-grid-thread-dependent-era-2026-08-30`.
   Training legs are untouched and still pre-fix.

**The board, and a MEASURED surprise.** The 11 current-era verdicts are promoted
as `<stored name>@cur372` rows (`benchmarks/board_era_rows_2026-08-30.md`), the
stored rows kept and gate-verified byte-identical. Pairing them exposed that
**7 of the 9 "stored-era" 372-class board rows were never reads of the 2026-05-15
root**: each is stamped `regime: "720"`, and a fresh `bake_verdict --regime 720
--corpora cid22` reproduces its per-pair predictions bit-exactly, i.e. they came
from `ext720-canonical-2026-07-22`, whose masked/IW block is POST-FIX. Those rows
already agree with the current-extractor read to ≤2e-4 on CID22 while differing
from the true stored-root read by up to 0.0153 SROCC. Only
`b_sdr_linear_cid80_inclwinsor_dense_dial` and `T_appT_b372_lam1e-3` (stamped
`regime: "372"`) are genuine stored-root reads. §3.28's era table is unaffected —
it compared `_old` vs `_new` from one instrument and never used a board row.

**Registry:** `benchmarks/eval_annotations.json` gains
`board372-row-read-on-ext720-root-2026-08-30` (annotated, 7 cells — the scope
correction above), `eval372-current-root-copied-corpora-2026-08-30` (annotated,
11 cells — six of the new root's fourteen corpora are byte-copies, and 39.5 % of
`product_composite`'s weight rides on them) and
`dial372-grid-thread-dependent-era-current-rows-2026-08-30` (annotated, 11 cells).

### §3.30 — ERA-3: v1 stopped pooling phantom columns (option C, 2026-08-30)

**Thought-why:** the fold's `f0..371` disagreed with the buffered v1 path, and
the working assumption (twice) was that the FOLD was the odd one out — first
that it was a tolerance question, then that the fix was to make the fold
reproduce v1 by pre-padding its input.

**Actual-why.** Measured, it is the reverse. **Buffered v1 pooled columns that
are not in the image.** `simd_padded_width` rounded the width up to a multiple
of 16 and added a *further* 16 whenever that landed on an even multiple of 16
at or above 512 (an L1d set-aliasing dodge); `mirror_pad_columns` filled the
extras by reflect-101; and the scale walk then pooled them. The fold never
padded. So the fold was already computing the correct statistic and v1 was not.

Consequences, all measured:

* Divergence up to **81.6 % relative** on a pool slot (200×150) and 17.4 %
  (127×93) — not a tolerance question.
* **Every common production width was in the divergent class**: 512→528,
  576→592, 768→784, 1024→1040, 1152→1168, 2304→2320. Tight widths (multiples
  of 16 below 512, and the non-bumped alignments above) were bit-identical
  across eras and do **not** change era. It is a per-ROW property of the
  width, not a per-table one.
* The h=93 "residual" reported mid-investigation was an artifact of the
  rejected pre-pad workaround, not of either walk. **Under C it does not
  exist.**

**What shipped (era-3, `56bbcda2`).** `blur::pyramid_plane_stride` returns the
width and is the single greppable owner; `mirror_pad_columns` and its three
call sites are deleted (with no padding they could never fire). The fold
needed no change, and that is verified structurally rather than numerically:
its production path never references the owner — every occurrence in
`feature_v2.rs` is a doc comment or test code — so **the 944 regimes are
unchanged by construction and no 944 table or model is invalidated.**

**It is also cheaper.** Buffered v1-372 instruction counts: **−9.02 % at 576,
−7.37 % at 1152, +0.00 % at the tight-width control 592**. The aliasing
rationale in the old doc is stale — on the current kernels the bump cost 7-9 %
at exactly the sizes it applied to. That is why C ships as "no padding" rather
than "padded buffers with pool-width exclusion": the latter would still blur
the phantom columns and is strictly slower.

**The golden gate was structurally blind, and that is the headline.**
`GOLDEN_SYNTHETIC` (64×64) and `GOLDEN_REAL` (96×96) are both stride-invariant
widths, so the golden set sat entirely in the tight class — it would have
passed however wrong the padded class was. **Their values are UNCHANGED by
era-3; they were re-verified, not re-pinned**, and that they still pass is the
evidence. A third fixture `GOLDEN_NONTIGHT` (200×150, procedural) now covers
the class. Negative control run: reintroducing era-2 padding fails **only**
the new fixture — synthetic and real both still pass.

**Gate meanings that inverted** (sanctioned re-pins, each a tightening — no
tolerance was widened anywhere):

| gate | era-2 meaning | era-3 meaning |
|---|---|---|
| `v1_padded_width_divergence_is_column_padding` → `v1_372_bit_exact_to_fold_at_every_width` | assert the paths DIFFER at non-tight widths, bounded | assert they are BIT-IDENTICAL at all 19 geometries incl. h=93 |
| `folded720_v1_basic_matches_v1_path` | `expect_bit_exact` flag; 127/200 in a "divergence class" | flag deleted; every geometry asserts bit-exactness |
| `folded720_v1_pools_match_v1_path` | same | same |
| *(new)* `pyramid_stride_has_no_phantom_columns` | — | pins the decision at its owner over 24 widths |

**Prior-era artifacts** (registered in `benchmarks/eval_annotations.json` as
`v1-372-era2-phantom-column-pooling`, `v1-golden-tight-width-blind-pre-era3`,
`eval-root-2026-08-30-372-prior-era`): every 372 table extracted through the
buffered extractor at a **non-tight** width, which includes the day-old
`2026-08-30-full-features-372` root and the 2026-05-15 STORED root. **No 944
table and no 944-trained model is affected.**

**NOT flipped, listed for the user:** `zensim_validate::eval_roots::
DEFAULT_FEATURES_ROOT_372` stays pointing at the era-2 root until the
era-3 re-extraction lands and the shipped-B-under-C delta is reviewed.

### §3.31 — the era-3 eval root, and shipped B priced under C (2026-08-30)

**Built.** `/mnt/v/zen/zensim-training/2026-08-30-era3-full-features-372` —
eight corpora genuinely re-extracted under option C (cid22 4292, kadid 10125,
tid 3000, pipal 21800, konjnd 1008, aic3 600, csiq 866, live 779) plus the
`kon504` fresh subset. **Six are copied and remain prior-era**: aic4,
nonphoto, imazen26, sdr25, hfnlproxy, hf_nearlossless (registered as
`eval-root-era3-2026-08-30-mixed-era-copies`).

**A silent-loss bug in the build script, found and fixed.** `zv cid22` pointed
at the dataset PARENT rather than `CID22_validation_set`. The failure reports
`4292 pairs in 0.0s (12427864/s)` — no work done — then `0 valid pairs`, writes
a **34-byte empty cache**, and **exits 0**. The era-2 build hit the same thing
(its cid22 cache is also 34 bytes, and its CSV is timestamped a minute later
from a hand re-run). Now fixed with the symptom documented at the call site.

**The era predicate validated itself, for free.** Seven of eight era-2 vs
era-3 CSVs differ; **`pipal` is byte-identical**, because its images are
288×288 and 288 is stride-invariant — it sits entirely in the tight class where
C is a no-op. That is a control nobody had to build, landing exactly where
§3.30's predicate says it must.

**Shipped B under C** (same bake, same corpora, only the features root
changed — `b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07`):

| corpus | era-2 | era-3 (C) | delta | n |
|---|---:|---:|---:|---:|
| cid22 | 0.882117 | 0.882141 | **+0.000024** | 4292 |
| konjnd | −0.649669 | −0.654272 | −0.004603 | 1008 |
| tid | 0.778520 | 0.778883 | +0.000363 | 3000 |
| csiq | 0.934208 | 0.934929 | +0.000721 | 866 |
| live | 0.897026 | 0.898517 | +0.001491 | 779 |
| kadid | 0.808474 | 0.808505 | +0.000032 | 10125 |
| aic3 | 0.765012 | 0.763666 | −0.001347 | 600 |
| **pipal** | 0.564971 | 0.564971 | **+0.000000** | 21800 |

**C does not meaningfully move B anywhere.** Largest is konjnd 0.0046, and
konjnd is read as a magnitude so that is B improving. cid22 moves +0.000024.
`pipal` at exactly zero is the instrument confirming itself. The features
change materially at non-tight widths (up to 81.6 % on a pool slot); the
pooled rank statistics barely move, which is what removing ≤16 columns of ~576
should do.

**NOT FLIPPED — the user's item.** `eval_roots::DEFAULT_FEATURES_ROOT_372`
still points at the era-2 root. The flip is now supported by the table above;
the caveat to weigh is that six corpora in the era-3 root are copied prior-era
rows, so flipping makes the default root era-mixed until those can be
re-extracted.


## §3.29 — the AVIF autotune training view: what we thought we were building vs what the data allowed (2026-09-04)

**Thought-why.** Union every scored AVIF DOE wave into one view, train a picker
that predicts `{backend, knob tuple, chroma, bit depth, speed}` plus expected
`{bytes, wall-time}` from source features + a target quality, and validate it on
held-out ODD origins.

**Actual-why, four ways the data said otherwise — every one MEASURED, not
inferred:**

1. **There are no odd origins, and there cannot be.** The 32 AVIF DOE references
   were k-means-selected under `--parity 0` precisely so no val/test-origin
   content could reach a training artifact. `origin_split.split_of` returns
   `train` for all 32, so the canonical validate/test buckets are structurally
   empty and `train_hybrid.py` correctly refuses the corpus with *"0 validation
   rows … not a train-biased even-only set"*. The fix was NOT to weaken that
   guard: `train_hybrid` gained a declared `SPLIT_RULE` hook implementing the
   already-registered even-only sub-split (`DATA_SPLITS.md` L158, the
   `avifgen-2026-08-06` precedent), which subdivides the TRAIN bucket only and
   hard-errors on any origin that is not canonical-train. 26 train / 6 `eval8`.
   **`eval8` is a leg-side holdout; the canonical one does not exist for AVIF.**
2. **`chroma` is not an axis.** Backend and chroma are perfectly collinear in
   every row (svt 4:2:0, zenrav1e 4:4:4; 1,114 `av1C` boxes, zero exceptions),
   because no chroma knob is wired for AVIF at all. It ships as a DERIVED
   attribute of backend. A picker fitted here learns the pair, and no
   re-analysis of existing bytes can split them.
3. **`wall-time` is a model of a model.** No fleet path persists a duration for
   any DOE cell, so the view's `encode_ms` comes from the speed instrument's
   `alpha + beta*MP` fits — single-threaded, q45-anchored, per-source fits on 5
   of 32 sources, pooled fits flagged `linear_model_failed` on 20/20 arms with
   beta spreading 24.3x. The trained time head fails the trainer's own
   `TIME_HEAD_R2` gate on both bakes; the shipped answer is the LUT, and even
   that is labelled modelled.
4. **The backend head does not work.** 54.0 % agreement with the measured
   per-image winner against a **67.7 % always-`zenav1-svt` baseline** — worse
   than the constant, and it never recovers a zenrav1e win (0/25, 3/27). The
   mechanism is legible in the corpus: the entire zenrav1e arm ran on the 1024²
   budget corpus, so the model sees that backend at exactly one size.

**What DID work.** The bytes/knob head: held-out mean regret 13.4 % (48-cell
cross-size-verified bake) / 14.5 % (143-cell full bake) against the per-row
oracle, and on screen content the core bake is **0.7 % mean, exactly optimal on
92 % of decisions**. Both bakes carry `safety_report.passed = false` and were
baked `--allow-unsafe`; those thresholds assume a corpus two orders of magnitude
larger than 32 references, and the violations are recorded rather than tuned
away.

**Two corrections made inside this pass, before publishing.** (a) The per-image
backend reference is a BUDGET-corpus verdict (`pixels` reads 1,048,576 for every
cropped ref); an earlier pass scored all 293 decisions and read 58.0 %, which
applied a crop verdict to native pixels — the scoped 161-row read is 54.0 % with
132 rows counted NOT-COMPARABLE. (b) `build_commit` came back `null` because a
`jj workspace` has no `.git`; it now falls back to the colocated primary repo.

**Three era-scoped facts were re-derived from the bytes rather than inherited:**
cross-era byte identity (12,000/12,000 shared cell identities, 0 conflicts), the
dead-knob census (which settles a contradiction between the Stage-A and
era-delta records — `scm3`/`tn0` inert at speeds 4 and 6, `scm3` LIVE at speed
7), and svt speed-dial aliasing (presets 8/9/10 ≡ 7).

Record: `zenmetrics/benchmarks/avif_autotune_v1_2026-09-04.md`. Contract:
`zenmetrics/benchmarks/avif_autotune_contract_2026-09-04.md`. Data:
`/mnt/v/zen/avif-autotune-2026-09-04/` (triple-mirrored;
`~/work/zen/DATA_PROVENANCE.md`).

---

### §3.32 — shipped B's training legs: the re-extraction is BLOCKED, and the defect is the DIAL (2026-09-04)

**Thought-why.** §3.27 established that the stored 372-col masked/IW block is pre-fix and
that shipped **B** "was fit AND calibrated on pre-fix masked/IW and is serving post-fix" —
"a genuine train/serve skew". §4c.3 of `benchmarks/v1_extractor_drift_2026-08-30.md`
registered the remedy as *re-extract B's ~227k training pairs and retrain*, priced at
"6–7 min of single-box CPU". The implicit model was: B is contaminated wholesale; a
re-extraction plus retrain repairs it.

**Actual-why.** Both halves of that model are wrong, measured.

1. **B is only ~13.5 % pre-fix by weight-fitting mass, but 100 % pre-fix in its dial
   anchor.** B = `0.8·cid + 0.2·kon`. The **cid head — 80 % of B — is fit on `hdr_v3mix`
   alone** (7,410 rows, extracted 2026-07-03, i.e. after both fixes), so it is already
   current-era. Only `kon` carries pre-fix legs, and within `canonhdr15` the post-fix
   `hdr_v3mix` is up-weighted ×15 = 32.7 % of that head's mass. Pre-fix share of the kon
   head 67.3 %; of B, **13.47 %**. Meanwhile `multiband_anchor_dial100.parquet` — B's
   *entire* dial calibration — joins into `safesyn` at **2,000/2,000** on
   `(ref_basename, f0)` and is therefore 100 % pre-fix. That is exactly the symptom shape
   §3.27 measured: rank barely moves (CID22 +0.0057) while the dial shifts −4.98/−5.86.
   The weights were never the problem.

2. **The re-extraction cannot be run** — ⚠ **numbers CORRECTED 2026-09-04, see §3.34.**
   `safesyn` (57.6 % of the kon head's mass, and the source of the anchor) was extracted
   from the `q<X>.png` decode cache, which is **0 % present** (0 of 3,000 sampled rows, all
   six codec families) — that part stands. The surviving bitstreams do re-decode to
   *different pixels*, but the magnitude first recorded here was measured through the
   third-party `image` crate, which reads an **XYB JPEG as an ordinary JPEG** and cannot
   decode AVIF or JXL at all. ~~worst `0.659 → 2875.0`, 69 % of cells over tolerance,
   roughly 10^4× the correction~~ — **RETRACTED.** Re-run with imazen decoders on all six
   families (360 rows, alignment gate 360/360): basic worst cell **5.481e+1**, the XYB
   family's own worst **30.31**, `|Δ| > 1.0` on **14 cells = 0.025 %**. The conclusion
   survives on a different number: re-decoding shifts shipped B's dial by **mean −3.658
   points, 73 % of the −4.98 era defect**, so a fresh safesyn still confounds the fix with
   a decoder-era term of comparable size. The compute estimate in §4c.3 was fine; **the
   inputs do not exist**, and that was never checked.

3. **A dial re-anchoring corrects the era term at ZERO rank cost.** `kadid`+`tid` exist in
   both eras with row-order-identical refs and byte-identical `human_score`, giving a
   matched-era anchor pair (13,005 rows, `target_score = max(ssim2_gpu,0)`; only the
   feature era differs). `bake_dial_refit shared-anchor` on each, scored on the current-era
   root: **SROCC identical to 5 dp on all five corpora** (spline is rank-invariant), while
   the per-pair dial moves **+6.196 (CID22, sd 2.246) / +6.235 (KonJND, sd 1.929)**,
   100 % of pairs > 0.5 pt. Against the defect's −4.977 (sd 2.301) / −5.857 (sd 2.299):
   same order, opposite sign, matching spread.
   **⚠ But anchor CONTENT moves the dial just as much** (−6.44 / −7.78 for a stored-era
   kadid+tid anchor vs the shipped safesyn one — a comparison itself confounded by
   procedure, `extend-top`/30 knots vs `shared-anchor`/12). So B's absolute dial is
   anchor-dependent at the same ±6–8 pt order as the era defect, and swapping anchors is
   not a drop-in fix.

**Also corrected here.** `zensim/CLAUDE.md` "Safe synthetic dataset" says *"the CSV
`decoded_path` PNGs no longer exist"*. That is true of
`2026-05-16/safesyn_with_iwssim.csv` — the actual extraction input — and **false** of
`synthetic-v2/training_safe_synthetic.csv`, whose `decoded_path` is the bitstream and is
present. The two files are row-identical and differ only in that column; conflating them is
what makes safesyn look re-extractable.

**Also found, not fixed.** `extract_features_372col` decodes via `image::open()` and
returns `None` on failure, so AVIF and JXL rows vanish **silently**: 240 of 360 probe rows
scored, i.e. 30.8 % of safesyn would be dropped without a word. Third instance of the
silent-row-drop class (`dataset_metric_baseline`, `zensim-validate --extract-only`,
now this). The decode helpers already exist at
`zensim-bench/examples/extract_features_372col_omni.rs:266-293` behind `extract-omni`.

4. **The fit chain is NOT byte-reproducible — and now the cost is priced.** Re-running
   `legs → gram → fit → ensemble → f16 bake → anchored bake` from the same stored legs
   breaks byte-identity in two named places: (a) `canonhdr15-bvls-raw` is an **iterative
   active-set (BVLS) solve**, so its `w` reproduces to **1.19e-5** relative against the
   closed-form lasso head's 2.25e-12, and 0.2× of that reaches the ensemble where one
   weight (`f83`) straddles an f16 tie — **371/372 f16 lanes identical**, bake sha
   differs; (b) the 823 B anchored-bake step's exact invocation is not recovered by
   `bake_dial_refit shared-anchor` at its defaults (`88a57447` vs the committed
   `7b326ac5`; the producer `shared_anchor_refit.py` was deleted 2026-07-29). The
   campaign's *"44/44 refits byte-identical"* determinism claim holds for a re-run **from
   cached grams**; it does not survive re-accumulating the grams from parquet.
   **Priced end-to-end**: rank reproduces to **≤ 3e-5 SROCC** (exactly 0.00000 on four of
   five corpora) and the dial to **≤ 0.071 points**, of which only ≤ 0.013 is the fit
   re-run itself — i.e. the pipeline's own noise floor is **~70× below** the −4.98-point
   defect. A fresh-legs comparison is therefore interpretable *functionally*, never on
   bytes, and any retrain moving the dial < ~0.1 pt has measured nothing.
   (Also: `ens-Pline-cid80.npz` was never emitted by the committed `cmd_ensemble`, which
   only produces `Pline-cid{30,50,70}` — a missing commit, not a lost recipe; the
   arithmetic reproduces the stored npz to 1.4e-14 with an exact bias.)

5. **The executable subset was RUN, and it falsified this lane's own registered prior.**
   §5 of the record predicted the launchable partial — kadid + tid swapped to current-era
   features, **1.94 % of the kon head's weighted mass, 0.39 % of B's** — was "very unlikely
   to clear" the 0.071-pt floor. Against a **matched control** (same pipeline, same day,
   stored legs), it clears it by ~13×: the dial moves **+0.838 (CID22) / +0.920 (KonJND)**
   mean, 99.5 % of pairs above floor, **in the correcting direction**, recovering 17 %/16 %
   of the defect's magnitude from 1.94 % of the mass; and rank moves on **all five**
   corpora — CID22 **−0.00087**, KonJND **+0.01244 |SROCC|**, AIC-3 +0.00106, TID +0.00156,
   KADID +0.00515 (⚠ KADID is B's train==val corpus *and* one of the swapped legs, so its
   gain is partly memorization, not generalization; the two genuine holdouts, CID22 and
   AIC-3, **disagree in sign**). Mechanism: BVLS is an **active-set** solve, and a
   1.94 %-mass perturbation moved three features (285, 336, 357 — all masked/IW) across the
   active boundary, `max |Δw|` 0.299 (89 % rel), support 85 → 86.
   **Do NOT extrapolate to the full re-extraction** — the effect runs through a non-linear
   active-set boundary, and the remaining 98 % is safesyn + cid22_train with safesyn
   BLOCKED. The arm is an instrument, not a ship candidate.

**Nothing was rewritten, retrained, or flipped.** Record:
`benchmarks/b_reextract_wave_2026-09-04.md`. Instruments + `_MANIFEST.json`:
`/mnt/v/output/zensim/bfresh-2026-09-04/`.

### §3.33 — class-C free slots: no new root was needed, and the free-40's route parity does not hold on real pixels (2026-09-04)

**Thought-why.** The class-C lane was scoped to emit the next tranche of "already in a
register, never emitted" 944 slots from a v1-basic-only walk, and then to **extract a new
dated fast-class root** (`/mnt/v/zen/zensim-training/2026-09-04-fastclass-classC/`) on the
fleet so the distillation lane would have tables carrying them. The implicit model was:
new slots ⇒ new columns ⇒ new extraction.

**Actual-why.** The model is wrong for this tranche, and measuring it first saved a fleet
wave.

1. **The class-C slots are EXISTING 944 positions, so every non-folded 944 root already
   carries them.** The 24 landed slots are the v2-348 `MSE` cell per (scale, channel) and
   the append `LUM_{DARK,MID,BRIGHT}_ERR` trio at Y per scale — positions a FULL 944 walk
   has always computed. What the class-C route changes is the COST of producing them, not
   their existence. MEASURED on
   `/mnt/v/zen/zensim-training/r1b-pools944-2026-08-30/ext_cid22val.parquet` (regime
   `folded720append2pools`, `build_commit ced6f52a`): **4,292/4,292 rows non-zero on every
   one of the 24**. **No new root was built.** The distillation lane trains on the existing
   root with `scripts/sota944/slice_basic156_free_classc.txt` (289 coordinates).

2. **Real-pixel route parity: the class-C 24 hold, the previous lane's free-40 do not.**
   `v2_ab_extract` gained mode `foldapp2fast` (the cheap route) beside `foldapp2pools` (the
   full walk that built the root); both were run over the SAME 773 real pairs (1-in-9
   stride of `pairs_imazen26_png.tsv`, real zenavif/zenwebp/zenjpeg output, 64×48 …
   1024×1024). Against the free-features lane's own 2e-5 bar:

   | set | cells | over bar | worst \|Δ\| |
   |---|--:|--:|--:|
   | class-C (24) | 18,552 | **0 (0.00 %)** | **9.81e-8** |
   | free-40 (raw moments) | 28,601 | **2,607 (9.12 %)** | **3.63e-3** |

   The misses are entirely `GLOBAL_CLOSS` (1,467) + `GLOBAL_CGAIN` (1,132), with
   `GLOBAL_DMEAN` 8 and `LUMA_MEAN_REF` **0** — i.e. exactly the slots
   `global_stats_from_raw_moments` derives through `Σs²/n − (Σs/n)²`, a
   catastrophic-cancellation form, where the two routes' f32→f64 staging differs and the
   true value on a near-identical pair sits at the f32 accumulation floor (worst relative
   error ~55×). It grows with plane size (3.63e-3 in the largest-25 % bucket vs 1.60e-3 in
   the smallest). The previous lane's gate is synthetic-image-only, which is why it read
   5.35e-6 and passed. **Reported, NOT fixed — those slots belong to that lane.** Consequence
   for anyone shipping a fast-class model: it would be trained on one route's
   `GLOBAL_CGAIN`/`GLOBAL_CLOSS` and served the other's, disagreeing by up to 3.6e-3.
   Basic + peaks are BIT-identical between the routes (worst \|Δ\| exactly 0.0 over
   773 × 228 cells).

3. **A v1-only 944 walk does not leave every unreached slot at zero.** All twelve
   `PJND_FRAGILITY` slots (`f393 422 451 / 480 509 538 / 567 596 625 / 654 683 712`) read a
   constant **1.0** on 773/773 rows — a `finish_channel_scale` formula artifact on zeroed
   accumulators, pre-existing and identical on the `RawMoments` route. A training lane must
   slice to the free set explicitly; "every non-zero column" hands a model twelve constant
   columns that do not exist in the stored 944 tables.

**Cost of the tranche, published honestly:** +1.3–1.5 % of the 156 walk at 1T on the
native AVX-512 tier and +2.0–2.3 % on the v3/AVX2 tier (CI-excludes-1.0 at every 1T cell,
both tiers); threaded cells mostly straddle 1.0 and are reported, not asserted. The lane
brief expected "~zero marginal"; the measurement is what is published.

Record: `benchmarks/free_features_classC_2026-09-04.md`. Artifacts:
`/mnt/v/output/zensim/classc-2026-09-04/{native,capv3,routeparity}/`. Code: `a8b24c8e`.

### §3.34 — the safesyn re-decode probe, run with OUR decoders: a 52× retraction, and decoder era is a −3.66-point term on every corpus (2026-09-04)

**Thought-why.** §3.32 point 2 concluded that safesyn is not re-extractable because
re-decoding its surviving bitstreams moves the basic `f0..155` block by *"roughly 10^4×"*
the correction the wave existed to apply — worst cell `0.659 → 2875.0`, 69 % of cells over
golden tolerance. The implicit model was: the stored pixels are unrecoverable by a wide
margin, so the whole question is closed.

**Actual-why.** That probe decoded with the third-party **`image` crate**, in a
measurement whose purpose was tuning an imazen model — an IMAZEN-ONLY rule violation
(`~/work/zen/CLAUDE.md`), and the violation *is* the result. `image` reads an **XYB JPEG
as an ordinary YCbCr JPEG** and never applies the inverse XYB→sRGB transform, so the
`zenjpeg-420-xyb-e2` family (14.4 % of safesyn) was compared in the wrong colour space;
and it has **no AVIF and no JXL decoder at all**, so `zenavif-s5-e6` (34,001 rows) +
`zenjxl-e7` (26,362 rows) = **30.8 % of safesyn** was silently dropped and never measured.

Re-run through `zencodec` magic-byte detection + zenjpeg / zenpng / zenwebp / zenavif /
zenjxl — **360 rows, 60 per family, all six families, q5..q100 across 16 q values,
alignment gate 360/360 on `(ref_basename, cpu_ssimulacra2)`**, extraction
`360/360 scored, 0 failed`:

| quantity | retracted (`image`) | **corrected (imazen)** | ratio |
|---|---:|---:|---:|
| basic `f0..155` max abs | 2.874e+3 | **5.481e+1** | **52×** |
| XYB family worst cell | 0.659 → 2875.0 | **29.84 → 60.16** | **95×** |
| peaks / masked / IW max abs | 1.613 / 1.034 / 1.246 | **0.166 / 0.0411 / 0.132** | 9.7× / 25× / 9.4× |
| basic cells with \|Δ\| > 1.0 | not reported | **14 of 56,160 (0.025 %), 6 of 360 rows** | — |
| families measurable | 4 of 6 | **6 of 6** | — |

The corrected numbers land **inside the drift bounds `zensim/CLAUDE.md` had already
recorded on 2026-06-22** from an independent measurement (plain JPEG `max_abs ≤ 5` —
measured ≤ 0.48; XYB `≤ 42` — measured 30.31; JXL differs by decoder lineage — measured
the worst family at 54.81). The retracted 2,875 was outside all three, which is the tell.

**The conclusion survives, on a number that is now the interesting one.** Forwarding both
matrices through **shipped B** on the canonical runtime: re-decoding shifts the dial by
**mean −3.658 points (median −3.181, sd 2.589, 94.4 % of rows > 0.5, max 16.0)** against
an era defect of **−4.977 / −5.857**. **Decoder era is 73 % of the extractor era, same
sign.** So a fresh safesyn still cannot isolate the extractor fix — but as a confound of
*comparable* size, not of four orders. Median basic cell moves 1.29e-5; the retracted
doc's own median (1.09e-5) was the part that was never wrong. **What was wrong was the
tail, and the tail was the argument.**

**What this changes for every corpus, not just safesyn.** Any corpus whose pixels were
decoded once and stored is pinned to that decoder's era, and re-reading it today costs
~3.7 dial points *before* any model change. imazen-26 is not exempt. The remedy is not to
hunt for a corpus that escaped decoder drift; it is to choose a **deliberate** era,
re-extract through it, and record the decoder per format in the manifest — which is what
the imazen-26 anchor build (§3.35) does.

**Owner fix that made the probe possible at all.** `extract_features_372col` decoded with
`image::open(..).ok()?`: a third-party decoder behind a `?` that turned failure into a
dropped row. It now goes through `zensim-bench/examples/shared/zen_decode.rs` (one owner,
shared with `verify_bitstream_decode`) and returns `Result`; a row that cannot be decoded
aborts the run unless the caller passes `--allow-failures N` (default 0). First contact
found a further gap the pre-existing `extract-omni` helper shared: **every probed
`zenavif-s5-e6` row decodes to `Rgb16`**, which both helpers rejected — now flattened by
the canonical `zenpixels_convert::RowConverter`, not a hand-rolled `v >> 8`.

Record: `benchmarks/safesyn_zencodec_probe_2026-09-04.md`. Gate:
`zensim-bench/tests/zen_decode_formats.rs` (13 tests; the AVIF, JXL and XYB-JPEG ones are
failing-first against the old path). Artifacts:
`/mnt/v/output/zensim/im26anchor-2026-09-04/probe/`.

### §3.35 — the imazen-26 dial anchor, and B's dial split cleanly into ERA and CONTENT (2026-09-04)

**Thought-why.** §3.32 established that 100 % of shipped B's dial calibration comes from a
2,000-row safesyn subset, and §9d of `benchmarks/b_reextract_wave_2026-09-04.md` concluded
that swapping it was not a fix: *"B's absolute dial is anchor-dependent at the ±6–8 point
level, which is the same order as the era defect"*, so any re-anchor "trades one
uncontrolled ±6-point dial shift for another". The implicit model was that anchor CONTENT
and anchor ERA are comparably large, leaving no clean move.

**Actual-why.** §9d's estimate was **confounded** — it varied content (safesyn multiband →
kadid+tid) *and* procedure (30-knot `extend-top` → 12-knot `shared-anchor`) at once. Varied
cleanly, the two terms are an order of magnitude apart. The instrument is a 2×2 with a
matched middle arm: the **same 2,000 safesyn anchor rows, same targets, re-decoded and
re-extracted today**, sitting between the shipped anchor and a new imazen-26 anchor.

| term | held fixed | CID22 | KonJND | AIC-3 |
|---|---|---:|---:|---:|
| procedure floor | anchor + era; chain rebuilt | +0.031 | +0.028 | +0.028 |
| **ERA** | **content exactly fixed** | **+3.892** | **+4.798** | **+3.864** |
| **CONTENT** | era fixed, both current | **−0.395** | **−0.989** | **−0.233** |
| total | — | +3.528 | +3.837 | +3.659 |

**Era is 4–10× content on every holdout, and rank is identical to 5 dp across all five
arms on all five corpora** (CID22 0.88212, KonJND −0.51938, AIC-3 0.76501, TID 0.77852,
KADID 0.80847). TID and KADID show large positive content terms (+2.0/+2.3) — they are the
corpora B's kon head was fit on, KADID being its documented train==val corpus, so that is a
train/serve corpus shift, not a generalization signal.

A `shared-anchor` refit on the same content read today recovers **78 % (CID22) / 82 %
(KonJND)** of the −4.977 / −5.857 era defect, in the correcting direction, at **126× the
procedure floor** and zero rank cost.

**A design property worth writing down: `extend-top` alone cannot fix an era skew.** It
keeps the bottom and in-distribution knots VERBATIM and only extends above the top knot;
CID22's dial tops out at 90.41, below that domain. Swapping only the `extend-top` anchor
moved the human corpora by **0.000**. Only `shared-anchor` refits the in-distribution
spline, which is where the skew lives.

**The new anchor.**
`/mnt/v/zen/zensim-training/2026-09-04-imazen26-anchor-372/imazen26_multiband_anchor_dial100_2026-09-04.parquet`
(sha256 `b2e8ead6…`, 4,000 rows × 382 cols; LAN
`s3://zentrain/anchors/2026-09-04-imazen26-anchor-372/`). Built from **imazen-26 bigcodec
TRAIN encodes**, 4 lossy codecs × 10 decile bands × 100 rows, 192 distinct origins, 1,224
distinct refs, `target_score = max(score_ssim2, 0)` — the shipped anchor's own rule.
**Nothing was re-encoded**: distorted bytes were byte-range-read from the canonical run
tars, reference bytes are the local `clean-picker-corpus-2026-06-26` renditions (0 of 1,224
missing). It doubles the shipped anchor's rows and adds a **codec axis the shipped anchor
does not have at all** (that file has no codec and no q column).

**The 924 tables could not supply it.** Their `f156..f371` are STRUCTURAL ZEROS (0 of 5,000
sampled rows nonzero); shipped B reads 49 of those 216 slots, so an anchor cut from those
columns would feed it real zeros — the `--regime 944` mis-scoring hazard in another
costume. Every feature is freshly extracted at 372.

**DECODER ERA IS NOW A RECORDED PROPERTY.** `_MANIFEST.json` names the decoder crate,
version and commit **per format** (zenpng 0.2.0 `00d6deb`, zenjpeg 0.9.0 `fad6a0af`,
zenwebp 0.5.0 `20898b7`, zenavif 0.2.0 `6dfdf6f`, zenjxl 0.3.0 `f0efd6d`), because §3.34
measured that re-reading a stored-pixel corpus costs ~3.7 dial points. **Known
mixed-era caveat, stated rather than hidden:** `target_score` is bigcodec's stored ssim2
from *its* decode era while the features are decoded today. The shipped safesyn anchor has
the same shape of property, so this is not a regression — but it is not single-era either.

**Two owner corrections this forced.** (1)
`scripts/canonical_corpus/resolve_bigcodec_pair_uris.py` listed `zenjpeg_lossy` and
`zenwebp_lossy` as fetch-mode `object`; re-measured, `canonical/2026-06-27/<ds>/encodes/`
is **empty for all four lossy datasets**, so both are now `tarrange` — as written, every
zenjpeg/zenwebp `dist_uri` 404s, which fails as a *fetch* error and slips past that
script's own 100 %-resolution gate. (2) `fetch_bigcodec_bytes.py` hard-required a
`human_score` column and now auto-detects (`--score-col`), and carries a numeric
`row_index` through so a corpus cut can rejoin its rows.

**NOT SHIPPED, and the honest reason.** All gates pass — monotonicity **0.9770** (better
than shipped's 0.9740), tied 0.0000, G-RANGE PASS — but the dial **compresses**: reach
96.85 → 85.74, p5 13.73 → 22.91, dynamic range −10.5. About half of that is already in the
current-era safesyn arm (reach 94.23), so it is part era, part content. A Profile-B swap is
a ship-default flip and belongs to the user.

**The registered top-densification was RUN, and it half-worked.**
`imazen26_anchor_topdense_2026-09-04.parquet` (sha `0d5d27a3eb5be4f9`, 3,785 rows: 300 per
decile to 90, then 800 in [90,95) plus every one of the 285 rows the corpus has above 95 —
1,085 above 90 against the uniform cut's 400) moves `extend-top`'s saturation `k`
1.325 → **1.885** and recovers **reach 85.74 → 88.96** and **p95 98.44 → 99.12**. But
**`p5` does not move (22.91 → 22.99)**. So the top-end loss was a top-density artifact and
is fixable there, while the **larger remaining term is the FLOOR**, which `shared-anchor`'s
percentile-edge fit sets and `extend-top` structurally cannot touch. A low-band
densification is the next lever — registered, not run.

**ARM 2 — replacing safesyn INSIDE the kon head — is a measured NEGATIVE.** The leg
`imazen26_konleg_40k_2026-09-04.parquet` (sha `d2bbb218914cd2de`; 40,000 imazen-26 TRAIN
rows, disjoint from the anchor, `human_score = score_ssim2/100` UNCLAMPED so its 2,781
negative rows survive as safesyn's do) enters `canonhdr15` at weight **196086/40000 =
4.90215**, so the mix's weighted mass is unchanged and only the leg's CONTENT differs.
Result, against a same-day control that is **byte-identical to the b_reextract lane's armC**
(`f08b3c8052e13e37`): **CID22 0.88212 → 0.85100 (−0.03112, ~1,000× the 3e-5 floor)** and
**AIC-3 −0.00321** — both genuine holdouts DOWN — while TID (+0.0157) and KADID (+0.0179),
the corpora the kon head is fit on, go up. Dial monotonicity 0.9740 → 0.9566.
**Mass-matching is not diversity-matching:** safesyn's leg spans 1,495 references and six
codec families (including two XYB-JPEG variants that exist nowhere else); the imazen-26 leg
spans **212 origins and four codecs**, up-weighted 4.9× to equal influence. The corpus's
212 train origins are a diversity ceiling, so more rows would not obviously fix it.
**Replacing safesyn is safe where it is a CALIBRATION input (the anchor: rank-invariant)
and expensive where it is a FITTING input (the kon head: −0.031 CID22).**

**Provenance correction to `b_reextract_wave_2026-09-04.md` §10a:** it describes the
uncommitted `ens-Pline-cid80` step as *"anchor-normalised 0.8\*cid + 0.2\*kon"* and omits
the **per-head taus**. Without them the reconstruction misses the stored npz by max |Δw|
**1.8e+2** (tau on both) or **1.5e+0** (tau on neither). The owner's `HEAD_POOL` has
cid `tau = 0.0`, kon `tau = 0.005`; with those it reproduces to **2.84e-14** with an
identical 95-feature support. This lane's arm-2 driver proves the reconstruction against
the stored artifact before using it.

Record: `benchmarks/imazen26_anchor_2026-09-04.md`. Artifacts:
`/mnt/v/output/zensim/im26anchor-2026-09-04/{build,arms,probe}/`.

### §3.36 — three 944-POOLS eval instruments built, and the 944 identity vector is NOT zero (2026-09-05)

**What was needed and did not exist.** Grading a bake that reads `f156..371` (any model
using the free set's 72 peaks) at 944 width required a dial grid, a negative-tail probe and
an identity probe all in the **`folded720append2pools`** era of
`/mnt/v/zen/zensim-training/r1b-pools944-2026-08-30/` (`build_commit ced6f52a`). Only the
grid existed (`wlin7b-2026-08-30/dial_grid_944col_POOLS_2026-08-30.parquet`). **The two
stored 944 negative-tail probes are the wrong era** — `negtail_probe_944_2026-08-01era` and
`…_era2r4_foldapp2` both carry `f156..371` STRUCTURALLY ZERO, which is exactly the block a
peaks-using bake reads, so serving either would have mis-predicted silently. **No 944
identity probe existed at any era.**

**What was built** (all in `/mnt/v/output/zensim/dfree-2026-09-05/`, registered append-only
in `benchmarks/dial_addressability_floor_2026-09-04.json` as `(instrument, peer_ssim2)` rows):

| instrument | rows | sha256 | rule |
|---|--:|---|---|
| `probes/identity_probe_944pools_2026-09-05.parquet` | 39 | `31fc4403…` | the 39 dial-grid references paired with themselves, extracted by the PINNED pools-era binary |
| `probes/negtail_probe_944pools_2026-09-05.parquet` | 2000 | `bafc8994…` | `cut_gaddr_negtail_probe.py`'s registered rule (20 equal-count quantile bins over the negative population, lowest 100 row indices per bin) applied to the pools safesyn leg's `human_score × 100` |
| `anchors/anchor944_r1b_dial_{clamped,negrich}[_id21].parquet` | 1976 / 1997 | `235dd65c…` / `5742aa7b…` / `2871bb4e…` / `7af3e25e…` | stride 55 (the era-2 `anchor944_pools_dial` manifest's own rule) minus the 44 rows the negtail probe also selects, so anchor and probe are **disjoint by construction** |

**The era control, run before anything rested on it.**
`wlin7b-2026-08-30/v2_ab_extract_PREFIX_PINNED` at `ZENSIM_AB_MODE=foldapp2pools`
reproduces **both** the r1b root (12 `ext_imazen26` rows × 944 = 11,328 cells, max |Δ| 0.0)
**and** the POOLS dial grid (7 matched cells × 944 = 6,608, max |Δ| 0.0) **bit-exactly**.
Separately MEASURED and REFUSED: `ext944-era2r4-2026-09-01/` declares the same regime and
ships an `anchor944_pools_dial.parquet` and a pools-regime `ext_kadis.parquet`, but is a
**different extractor era** — same rows, same order, `human_score` bit-identical, features
NOT (max |Δ| 7.46e-2 on `f0`, 1.96e-2 on `f160`, 3.13e-3 on `f930`).

**The finding: at 944 width the identity feature vector is NOT the zero vector.**
`dial_addressability.rs` carries a named constant asserting *"ref == dist yields all-zero
features for every image; identity dial = dial(0-vector)"*, and the 372 probe's registry row
records that as a measured property. On the 39-row 944-POOLS probe, **286 of 944 slots are
non-zero on at least one row and they VARY BY IMAGE** — the fold's own basic block
(`f0`/`f1`/`f2` at 4.7e-6 … 8.1e-4), every peak triple, the whole `PJND_FRAGILITY` family
(the formula artifact `free_features_classC_2026-09-04.md` §6.3 documents), and
`LUMA_MEAN_REF`, which is REFERENCE-ONLY and therefore cannot be zero on a ref==dist pair.
C5 and C6 still measure correctly (`IdentityMeasure` counts rows outside the band and
compares grid cells against `dial_max`), but **the note string is wrong at this width and is
emitted verbatim into every `--gaddr-json`**; the new registry row states the correction in
its `measured_property` field. Making the constant conditional on the probe is REGISTERED,
NOT DONE.

**Round-trip control:** `peer_ssim2` graded against its own three freshly-appended rows
reads **SHIPPABLE, 15 pass / 0 fail / 0 not-measured**, every bar tied bit-exactly; and the
four grid scalars (`min −55.354544, max 99.993468, p5 12.0158692, p95 95.4858354`) reproduce
the parked `gaddrinst` lane's independently-computed full-944 values EXACTLY. ssim2's
identity is **100.000000** on all 39 (38 carried from the 372 lane's `zenmetrics batch
--metric ssim2` run; the 39th, `9059ec43b26aa167_769x513`, measured here by the same tool).

**⚠ These bars are NOT the 372 bars.** Most sharply on A8: this probe's ssim2 p1 is
**−64.23** against the 372 kadis probe's **−187.13**, because safesyn's negative population
is shallower than KADIS's. A number from the two instrument sets must never be compared.

Record: `benchmarks/d_free_id100_2026-09-05.md` §§1.4, 2, 3. Artifacts + shas:
`benchmarks/d_free_id100_bakes_2026-09-05.pointer.md`.

### §3.37 — the v1-372 eval roots predate option C, so every 372 verdict is one era behind the runtime (2026-09-05)

**Thought-why:** the ship-flip lane needed to know which extraction era the *product
runtime* emits, so it could say whether a candidate bake's published numbers describe what
users get. The expectation, from the roots' own manifests and from `CLAUDE.md`, was
"HEAD == the 2026-08-30 root": that root is the DEFAULT `--regime 372` features root, it
was built the same week, and `CLAUDE.md`'s EXTRACTION PERF section said option C — the one
known pending v1 semantics change — was *"**Not flipped** — default untouched pending the
era rollout"*.

**Actual-why:** option C has been flipped since `56bbcda2` (2026-08-30 **15:43**), whose own
message calls it *"STAGE 1 of the C rollout"*. The default root was built at `ea16c7ee`,
2026-08-30 **13:21** — **two hours earlier**. `CLAUDE.md` was stale by six days.

**MEASURED.** `extract_features_372col --corpus pairs-tsv --path
/mnt/v/dataset/csiq/csiq_pairs.tsv` at HEAD — the *same tool* on the *same input file*
`scripts/canonical_corpus/build_eval372_root.sh` used — against the root's own build output
`~/tmp/eval372root/csiq.csv`. Row alignment established first, not assumed: `ref_basename`
order identical and `human_score` **bit-identical positionally on all 866 rows**.

| block | cells differing | max \|Δ\| |
|---|--:|--:|
| basic `f0..155` | **120,804 / 135,096** | **4.536785** |
| peaks `f156..227` | 34,566 / 62,352 | 0.326375 |
| masked `f228..299` | 62,346 / 62,352 | 0.067955 |
| iw `f300..371` | 62,346 / 62,352 | 0.079387 |

Identical results against BOTH v1-372 roots (`2026-08-30-full-features-372` and
`2026-05-15-full-features`) — consistent, since those two agree on basic+peaks. Every row
differs, on 285–341 of 372 slots; no row and no slot is clean. CSIQ is 512×512, exactly the
padded-width divergence class C removed (the commit: *"it put 512/576/768/1024/1152/2304
all in the divergent class"*).

**Two alternatives ruled out by measurement, not by argument.** `ZENSIM_ERA2_DENSE=0`
reproduces HEAD's output **byte-for-byte**, so `515001dc`'s era-2 dense flip is not the
cause (it moves only `f372+`). `v1_golden_bytes` passes **5/5** at HEAD, because every
golden fixture is 64×64 / 96×96 / 128×128 / 200×150 — tight class or below the tile — which
is the same blindness `CLAUDE.md` had already flagged for the goldens and which the C
rollout's own non-tight fixture only partly covers.

**What this does and does not invalidate.** It does **not** touch a same-root A/B (both arms
read the same bytes) and it does not reorder 372 cells relative to each other. It does mean
every ABSOLUTE `--regime 372` number — shipped D's CID22 **0.86338** included — is an era-1
read of a model the product now feeds era-3 features. **The 944 roots are NOT affected**:
`56bbcda2` verified structurally (not just numerically) that the fold's production path
never references the padding owner, so the 944 regimes are unchanged by construction.

**Registered, not run:** re-extract the 372 root at HEAD
(`scripts/canonical_corpus/build_eval372_root.sh` + `pack_eval372_root.py` — a decode-bound
corpus pass, no code change) and re-verdict the D/B lineages on it. Registry:
`benchmarks/eval_annotations.json` → `v1-372-eval-roots-predate-option-c-2026-09-05`.
Record: `benchmarks/d_ship_flip_2026-09-05.md` §3.

---

### §3.38 — the RUNTIME-era 372 root exists; the canonical DIAL GRID's pixels do not (2026-09-05)

**Thought-why.** §3.37 registered "re-extract the 372 root at HEAD" as a decode-bound corpus
pass with no code change, and `benchmarks/d_ship_flip_2026-09-05.md` §5 registered the peaks
refit that would be gated on it. Both assumed the three G-ADDR instruments could be rebuilt
the same way.

**Actual-why.** The ROOT rebuilt cleanly. Two of the three instruments did too. The dial
grid could not, and the reason is a data loss nobody had connected to it.

**The root.** `/mnt/v/zen/zensim-training/2026-09-05-full-features-372-postC`, era token
`v1postc`, `build_commit 4fbd8ff8`, `feature_set_id basic+peaks+masked+iw@w372/v1postc#d16a1091`,
named once in `zensim_validate::eval_roots::POSTC_FEATURES_ROOT_372`. Eight corpora
re-extracted at HEAD (cid22 4,292 · kadid 10,125 · tid 3,000 · csiq 866 · live 779 · pipal
21,800 · konjnd 1,008 · aic3 600), six byte-copied from the `v1cur` root and era-stamped per
file (aic4, nonphoto, imazen26, sdr25, hfnlproxy, hf_nearlossless), `kon504` derived plus its
side root. **`pack_eval372_root.py` now REFUSES** a fresh table whose `human_score` is not
BIT-identical positionally to the superseded root's — the assumption every era comparison
rests on, previously unchecked. It passed 8/8. It is **NOT the default**; which root a
flagless `bake_verdict` reads stays a user decision.

**And the era it exposes is mild on RANK.** MEASURED across five shipped 372-class bakes,
`v1cur` → `v1postc` moves CID22 SROCC by **≤ 6.8e-4** (shipped D −5e-5, previous D −5e-5,
Profile A −5.9e-4, shipped B +2e-5, the new peaks arm −6.8e-4). That is two orders below the
`v1pre` → `v1cur` step's |0.489| worst case (§3.28), and the reason is structural: that flip
moved `f228..371`, which only some models read, while this one moves `f0..227`, which every
372 model reads — but smoothly, so a rank statistic absorbs most of it. **The DIAL moves
much more**, so this is not licence to read a stored 372 dial number as current.

**⛔ The canonical dial grid is UNREBUILDABLE.** `dial_grid_372col_2026-05-29_quarantined_v2`'s
own build list (`eval_panels_2026-05-29/qsweep_372_grid.tsv`) points every distorted cell at
`/mnt/v/input/zensim/images/<ref>/<codec>/q<N>.png` — the decode cache deleted 2026-06-22 —
and **0 of its 2,560 rows' paths exist**. Its jpeg leg was `mozjpeg-rs-420-e4`; the surviving
`dial-grid-pixels-2026-07-27/` set is a `zenjpeg` re-encode of the same (image, codec, q)
lattice. Confirmed numerically before it was confirmed from the build list: a PRE-option-C
binary (`27cfde15`) does not reproduce the stored grid from those pixels either — basic
max |Δ| **0.703** on 2,168 of 2,340 control cells, against **0.055** for the entire
preC→HEAD era step.

**The surviving pixels are the right instrument anyway, and this is the sharp part.** The
registered `peer_ssim2` G-ADDR grid pins were measured ON the 2026-07-27 pixels:
`ssim2-bar-2026-08-31/dialcells_ssim2_qv2grid.tsv` **is**
`dialcells_ssim2_944grid.tsv` restricted to these 4,424 keys (4,424 of 4,424 exactly equal),
and the 944 grids were built from them. Spot-check through `zenmetrics batch --metric ssim2`
on the new pixels: `(00b13be94a4867dd_1022x818, jpeg, q0)` reads **−8.0345** against the pin
table's **−8.0450**. So the canonical instrument has ALWAYS paired ssim2 truth from one pixel
set with features from another; the rebuild removes that, it does not introduce it. Derived
through the owner, `peer_ssim2` on the new grids reproduces its canonical registry row on all
eight scalars (min −55.354544 · max 98.376644 · p5 10.26332105 · p95 95.45929934999998 ·
reach 153.731188 · DR 85.19597829999998 · mono 0.99235757295044 · tied 0.0000).

**The identity probe is ERA-INVARIANT**: a HEAD rebuild of the same 38 `ref == dist` pairs is
**byte-identical** to the registered 2026-09-04 probe (sha `e6f9096b8e0ebd97…`, 0 nonzero
cells of 14,136). At w372 the identity feature vector is the zero vector in both eras — which
does NOT extend to w944 (§3.36 measured 190 of 944 slots non-zero there).

**The negative-tail probe IS recoverable, and its provenance is now pinned.** All 2,000 rows
join UNIQUELY (0 ambiguous) to `kadis700k_canonical_gpu_2026-07-01.parquet` on
`(feat_0, feat_1, feat_2, score_ssim2_gpu)` → `distorted_url` (R2) + `source_filename` (local,
2,000/2,000). Control: a PRE-C extraction of the recovered pixels reproduces the stored probe
to ≤ 7.66 absolute on a ~1.2e5-scale slot (6.2e-5 relative), while the preC→HEAD era step on
the same rows is 18,953 on that slot.

**The frozen safesyn Gram is `v1pre`, measured not assumed.**
`linear-probe/grams/safesyn.npz`'s first and second moments match
`canonical-2026-05-21/train/safesyn.parquet` exactly on every column checked (f0
`s=3688.568709 S=272.523676`; f1, f155, f156, f200 likewise, n = 196,086 both sides). Every
bake in the ADD156 / D lineage is therefore trained two extraction eras behind what the
runtime serves it.

**A silent-corruption class worth its own line:** `extract_features_372col` sorts its output
by `ref_basename` before writing (`:216`), so positional re-attachment of key columns is
UNSOUND for any input not already in that order — and it fails with the right row count and
every key present exactly once. It scrambled the first build of this grid across its 38
references (shipped D's ladder monotonicity 0.9847 → 0.5611). Carry a numeric `row_id` and
invert the permutation; do not change the sort, because the root build's positional
`human_score` gate depends on both sides having it.

Artifacts: `/mnt/v/output/zensim/dpeaks372-2026-09-05/`. Record:
`benchmarks/d_peaks_372_postC_2026-09-05.md`.

---

### §3.39 — the floor-dense LADDER instrument: jpeg's "three lowest settings" were one setting sampled three times (2026-09-05)

**Thought-why.** §3.38 rebuilt the dial grid at the runtime era from the surviving
2026-07-27 pixels and treated the instrument as fixed. The remaining ladder problem was
assumed to be extraction era.

**Actual-why.** It was the **encoder axis**. `A7r` asks whether a codec's three lowest
configurable settings still resolve, and on every grid built before this one that question
was unanswerable for `jpeg`: **zenjpeg emits ONE bitstream for every q in 0..10** —
identical bytes AND identical ssim2 to 6 dp, on every reference tested — so the grid's
q 0 / 5 / 10 are one setting sampled three times. The mentor's jpeg bar was therefore a
vacuous `0.0000`, which anything passes, and the incumbent's own grading shows it as jpeg
`bottom_medians` **22.22 / 22.22 / 22.22**.

**Per-codec floors, measured** (same setting == same encoded bytes AND same ssim2):

| codec | plateau | first DISTINCT |
|---|---|---|
| `zenjpeg` | q 0..10 — **eleven settings, one output** | q = 11 |
| `zenwebp` | q = 0 | q = 1 |
| `zenavif`/`zenravif` | q 0..1 | q = 2 |
| `zenavif`/`svt-rs` | q 0..1, then **pairwise ties** (quality 0..100 onto QP 0..63) | q = 2 |
| `zenjxl` | **distance >= 25** (26/30/40/50 byte-identical to 25.0) | d = 24 |

**So the rule is DEDUP BY ENCODE HASH, never a per-codec step table.** `avif-svt` is
**36.4 %** duplicate settings against `avif-rav1e`'s **3.0 %** on the same quality axis —
a 12x difference no fixed step could express. The instrument keeps DISTINCT settings only
(which is what lets `dial_addressability.rs` stay unchanged: its "bottom K" become the
bottom K *configurable* settings by construction); the archive keeps every step flagged.

**What it changed.** jpeg's mentor bar `0.0000` -> **0.5385**, and **shipped Profile D —
a clean A7r pass on every older grid — FAILS**, on jpeg, by one ladder (20/39 vs 21/39).
Profile B fails all five codecs with a POSITIVE `dial_min` on each. Nothing installed.

**And the fix cannot be calibration.** All 19 of D's failing jpeg ladders are inversions
in the RAW pre-spline model (raw-vs-dial ordering verdicts agree **39/39**), and the two
shipped D bakes — same ADD156 weights, different output spline — have **identical A7r on
all five codecs** while `avif-rav1e` `dial_min` moves −13.49 → −59.81. A monotone spline
moves range, never rank. **The A7r lever is the weights.**

**A codec defect this surfaced, filed as `imazen/jxl-encoder#101`:** at butteraugli
distance **>= 10.0 exactly** (9.9 is fine), the encoder writes a `SizeHeader` rounded UP
to even, so a 513x769 source declares 514x770 and cannot round-trip. Read from the
codestream's own header, so encoder-side. Pre-existing — the 2026-07-27 sweep shows the
identical signature on the same 13 odd-dimensioned images. It removes exactly those
ladders' FLOOR cells, so they are excluded as truncated-floor rather than graded several
steps up the curve; jxl carries 26 ladders, not 39. **It was diagnosable with no re-encode
only because this run persisted encoded bytes** — the discipline §3.38's own grid was
built in the absence of.

**Two process failures worth recording as data-integrity lessons.** (1) Editing a shell
script while bash is executing it: bash reads scripts incrementally by byte offset, so a
mid-run edit made it resume at a stale offset and die with `unexpected EOF` *after* the
avif leg and *before* jxl, with no COMPLETE marker and no error in the leg logs. Runs now
launch from a frozen copy. (2) Two long runs died silently when nohup'd inside a
backgrounded shell — rc=137 and rc=143, the latter at 786 s with 13.7 GiB still available,
so NOT memory: they were killed with their task's process group. Long jobs now launch
under `setsid`. Both failures are invisible in the output data; only the exit code and a
missing marker distinguish them from success.

Artifacts + provenance: `~/work/zen/DATA_PROVENANCE.md` ★ ladder-instrument-2026-09-05.
Record: `benchmarks/ladder_instrument_2026-09-05.md`.

**§3.39 addendum — the ANCHOR arm (2026-09-05).** A current-era, **UNCLAMPED**,
CID22-disjoint dial anchor was built alongside the instrument
(`ladder_anchor_372col_anchor.parquet`, sha `4683ed51ac6acf8c`, 4,552 rows = 4,520
distinct cells + 32 identity rows at 100). 32 k-means imazen-26 representatives x 4
ladders; `check_holdout_overlap --threshold 10` against the 49 CID22 validation
references returns **0 hits** (closest d = 19). It closes both defects of the shipped
anchor: the target keeps its sign (min **−69.97**, **404 negative rows — 8.9 %** — where
`max(ssim2, 0)` collapses that whole run onto one bottom knot), and targets and
features come from ONE fresh sweep instead of a stored-era score read against today's
decode.

**It is a registered asset and is NOT installed, for a measured reason.** Re-anchoring
shipped D on it (`shared-anchor`, 19 knots) leaves **A7r identical on all five codecs**
and CID22 identical to 5 dp — confirming, on a real anchor rather than by argument,
that a monotone spline cannot move a rank-level failure — while **cutting dial reach
194.97 -> 160.07** and flipping A2/A5/A6 from PASS to FAIL. Under the standing rule that
a model limiting dial range cannot ship, that disqualifies it on its own.

**This is the SECOND independent imazen-26-based anchor to lose reach** (§3.35's
2026-09-04 proposal cut 96.85 -> 85.74 and was likewise not shipped). Two builds,
different codec sets, different eras, same direction — enough to treat "imazen-26
anchors lose dial reach" as a pattern wanting a named cause before a third attempt.


### §3.40 — the postC corruption-head corpus (2026-09-05)

`/mnt/v/output/zensim/corruption-head-2026-09-05/` (LAN mirror
`s3://zentrain/corruption-head-2026-09-05/`), `_MANIFEST.json` with `build_commit`
+ per-file sha256. Four tables, all extracted at HEAD (post-`56bbcda2`, option C):

| file | rows | what |
|---|--:|---|
| `im26_corruption_372_postC.parquet` | 117,276 | 116,928 structural corruptions + 348 matched q10/q20 anchors; 173 ref_ids, 44 families, **0 skips** |
| `negrich_372_postC.parquet` | 60,000 | seed-0 sample of the 280,384 KADIS severe-honest rows, PNGs re-fetched from R2 and re-extracted |
| `corruption_grid_372col_postC_2026-09-05.parquet` | 2,016 | the persisted `gb82_dog` gate triples, re-extracted |
| `d{156,228,228nb}/corruption_head_*.bin` | — | the heads, ZNPR v3, at caller widths 372 and 944 |

**Why it exists.** Every pre-existing corruption table is one extraction era behind
the runtime. Re-extracting the same pixels at HEAD moves **51.8 %** of basic cells
on the corruption corpus (max |Δ| 0.336), **73.7 %** on the gate grid (max |Δ|
4.35, 2013/2016 rows) and **83.3 %** on negrich (100 % of rows). All three are the
padded-width class option C removed.

**Provenance hazards, recorded so they are not re-discovered:**

* The corruption sources are the 2026-07-24 set with `/mnt/v/imazen-26/` rewritten
  to `/mnt/v/imazen-26-inspo/` — the **quarantined** tree (renamed 2026-08-27).
  Kept deliberately so the era comparison is an era comparison; rebuilding on the
  canonical `imazen-26-png-v3` is REGISTERED, NOT RUN.
* The corruption pixels are **not stored** and never were — the builder discards
  them because they are a pure function of `(ref_id, seed, params)`. The generator
  (`codec-corpus` @ `3e7a8a2`, `--seed` default 1) is what makes that safe.
* negrich's `distorted_url` lives on the **GPU** canonical (2026-07-01), not the
  zensim-only one the subset was cut from; they join exactly on
  `(source_id, dist_type, severity_level)`, 280,384/280,384. The PNGs are on **R2
  only** — the LAN store carries `kadis-700k-gpu/canonical/` and no `distorted/`.
* Every distorted basename embeds `..._zenjpeg_q<N>_...` **regardless of the actual
  distortion** — a chunk-pipeline naming artifact. `dist_name` is the truth (24
  KADIS types; `color_saturate_hsv` is the mode). Do not read those rows as codec
  output.
* The reference downscale is PIL Lanczos and the q10/q20 anchors come from the
  `image` crate, both inherited from the 2026-07-24 instrument and deliberately
  unchanged (swapping either changes the PIXELS and confounds the era A/B).
  Replacing them with `zenresize`/`zenjpeg` per IMAZEN-ONLY is its own change.

Record: `benchmarks/corruption_head_d_2026-09-05.md`; ledger ROUND 97.

---

## §3.30 — fastclass2: what the fast class's blocker turned out to be, and two defects found on the way to measuring it (2026-09-05)

**Thought-why.** *"Find a 156-or-156+cheap model whose RANK is competitive with
the 944-class MLPs."* The premise was that the fast class had a rank gap to
close.

**Actual-why, measured before the campaign's own arms landed.** At k = 3 and on
a **closed era**, the fast class already clears two of the three bars: the 944
leaders re-scored on the era2r4 era read composite 0.8636 / 0.8626, CID22
0.8877 / 0.8908, KonJND 0.4783 / 0.4782, and the fast-class incumbent `FC_D3`
reads **0.8645 / 0.8863 / 0.4322** — past the composite bar, within 0.0014 of
the CID22 bar (inside the ~0.0069 per-model CI half-width), and −0.046 on
KonJND. **The rank problem is one axis wide.** Meanwhile the SHIP problem is a
different quantity entirely: A7r floor representability, which the fast class
fails on 5 of 5 codecs and which no dial chain can touch.

**Era note that matters for any future comparison.** The 944 leaders train on
`ext944-canonical-2026-08-01`, whose registered set is
`basic+v2+append+append2@w944/ext944` — **pools ZEROED**. Their `f156..371`
weights are therefore untrained initialisation values sitting on structurally
zero inputs (measured: 13.5–15.8 % of layer-0 L2 mass on peaks, 12.2–12.9 % on
masked, 8.1–8.3 % on IW, and **0.00 % on append/append2**). Scoring such a bake
on a pools-LIVE root multiplies those by real features. **Read a leader on
`ext944-era2r4-2026-09-01/foldapp2_views`, never on the pools root.**

**Data facts established, each reusable on its own:**

* **The fast class's identity contamination is FOUR slots** —
  `LUMA_MEAN_REF` at f926/931/936/941, the free set's only reference-absolute
  statistic. On the 39-row 944-pools identity probe they carry max |v| 0.688 /
  spread 0.261 while every other slot in the 265 set stays under 4.8e-3, all 33
  other raw-moment slots are identity-ZERO, and so are all 24 class-C slots.
  New reader slice `scripts/sota944/slice_basic156_free_nolumaref.txt` (261).
  Their layer-0 weight mass is **0.45–0.64 %** (ranks 238–260 of 944), so the
  fix is predicted to cost about half a percent of mass.
* **The id100 anchor chain works unchanged on this class**: one
  `bake_dial_refit pack --anchor <anchor ∪ 21 identity rows @ 100>` takes the
  contract 5/6 → **6/6** with CID22 bit-unchanged at 0.8863.
* **`Zensim::compute` short-circuits byte-identical input** to `(100, 0, zeros)`
  before the model (`metric.rs:3509/5225`, `.mark_identical()`), so a C5 failure
  is never a claim that `zensim(x, x) != 100` in production — C5 governs
  NEAR-identity, which is the regime a near-lossless dial lives in.
* **Servability, measured not inferred**: a 372-layout bake that READS THE PEAKS
  BLOCK serves today; every 944-declared bake is refused with
  `ModelForwardFailed`. So of the campaign's five sets only S156/S228 are
  servable, and only at the v1-372 layout.
* **The slice, isolated**: at fixed class (sparse additive), fixed layout (372),
  fixed anchor chain and one instrument, going 156 → 228 costs **three of five
  codecs** on A7r and 0.0315 of dial monotonicity.

**Two defects found, both in the silent-no-op class, both fixed at their owner:**

1. **`--coarse-decay` was silently discarded on the per-sample-α head.** It is
   applied by `apply_post_adam_penalties`, called at seven sites all inside
   `train_mlp_strategy`'s plain loop; `train_mlp_per_sample_alpha_head` never
   calls it. `--group-l1`, which rides the SAME function, had been guarded
   against exactly this since it landed, and says so in its own doc comment.
   Now `coarse_decay_unsupported_flag` fails loud, with the default plain-path
   case pinned as still allowed.
2. **The 372 lane's first draft named the un-normalised targets.** The recipe
   convention is `human_score ∈ [0,1]` with `--target-scale 100`, and the 372
   directory carries both forms: `cid22_train` [3.0102, 94.1532] vs
   `cid22_train_norm` [0.0301, 0.9415], `konjnd-dense` [−65.7108, 96.1549] vs
   `konjnd-dense-norm` [0, 1]. Two groups at ~100× the others' scale, and
   nothing would have crashed. Caught in pre-flight; 15 fits unspent.

**Naming.** Five feature sets registered and owner-hash-verified:
`basic+peaks@w944/era2r4` (228, `3fb78648`),
`basic+peaks+moments@w944/era2r4#0b476506` (261, the identity-clean READER
subset of its own producer — the designed "name is a handle, hash is the
identity" case), `basic@w944/era2r4#3ffe8670` (156 at 944 layout),
`basic+peaks@w372/v1pre` and `basic@w372/v1pre` (the servable forms), plus the
root `canonical-2026-05-21/train`. **The hash is layout-independent by design**,
so the 228 reader set carries `#3fb78648` at both 372 and 944 — and that
distinction is precisely what decides servability.

### §3.30a — RESULTS (all 57 cells landed; 3 failed, diagnosed)

* **YES, a 156-plus-cheap model is 944-competitive — on BOTH layouts.** The
  selected `S372_S228_H128_p` (k=3) reads composite **0.8732** / CID22
  **0.8896** / KonJND **0.4999** against the era-closed leader bars of 0.8626 /
  0.8877 / 0.4782, and the 944 lane's own `S228` H32 (0.8666 / 0.8887 / 0.4543)
  clears composite and CID22 too. The **rulers are verified identical** — 504
  KonJND refs and 4,292 CID22 pairs for every row — so the era difference
  changes each model's inputs and not the question.
* **CAPACITY IS NOT A LEVER.** H32 vs H128 moves composite by −0.0038…+0.0021
  across six set×width pairs, inside every seed spread, at 30–47 % fewer bytes.
* **THE COMPUTE CEILING IS BELOW THE RESTRICTED SETS.** `SORACLE` — the same
  recipe reading all 944 coordinates — reads composite **0.8581**, the lowest
  non-degenerate cell, and KonJND 0.4191 against `S228`'s 0.4543. **The fast
  class's KonJND gap is NOT a compute gap**; the campaign's standing hypothesis
  is falsified by its own registered instrument.
* **The per-sample-α head is an INVERTED ranker on this recipe** — raw CID22
  **−0.8921** at depth 2 (a *better* ordering in magnitude than the plain
  path's +0.8863) — which is why its pack died on a monotone spline it could not
  fit. `--monotonicity-reg` / `--konjnd-aggregation-*` / `--pjnd-passthrough-*`
  are all reachable only through that head, so they stay UNMEASURED behind a
  named prerequisite.
* **Ship: PROPOSE, not install.** Contract fails C5 (identity dial 90.9368, band
  [97.5, 100]) and C6 (1,642 of 9,593 cells above identity) — the registered
  either/or, a **weights** defect, worse here than shipped B's 6.01 % — and A7r
  fails 5/5 codecs. `ZensimProfile::D`, `zensim/weights/`, `profile.rs` and the
  manifests are untouched.
* **W4 MEASURED 2026-09-06 (ROUND 99), and both candidates PASS.** On an idle
  box (load 3.9 → 0.7), 8 cells × 10 starts, **80 starts, 0 discarded**, every
  control in family. Against the registered `add156_156basic` bar the selected
  `S228` maxes at **1.2202** and `S156` at **1.2121** (bar 1.25); against
  **`zensim_D`** — Profile D through the standard production path — **`S228` is
  faster in all 8 cells** (max 0.9733) and 1.43–2.15× faster than shipped B.
  The exam's own clause passes at **3.73–3.97× `fast_ssim2`** at 1T.
  **The forward pass is below the noise floor** (extract-only arms read *slower*
  than their full siblings), and the cell nearest the bar is the least resolved
  (native/t8/1152, control spread 16.6 %) while its well-resolved twin reads
  1.1510. Also confirms the kernel lane's DEFECT 2 from data: the bar arm's
  `V1PoolsMode::Off` walk is **slower than production's `Peaks`** at
  capv3/t1/576 (7.660 vs 6.280 ms).
  **A collation defect was found and fixed at the owner first**: `w4_report.py`
  globbed a filename pattern that predated the tier suffix (zero files matched,
  hence a header and no rows) and its arm-name regex captured zenbench's
  box-drawing prefix and the `±0.0` mad instead of the arm and its time — the
  second would have produced a plausible wrong answer had the first not masked
  it.

Record: `benchmarks/fastclass2_campaign_2026-09-05.md`, registration
`docs/PLAN_FASTCLASS2_2026-09-05.md`; ledger ROUND 98.

### §3.41 — the kernel lane 2 front-end levers move NO bytes, so no table and no bake is invalidated (2026-09-05)

**Ledger ROUND 99.** Recorded here precisely because a "the extraction kernel
changed" line in `git log` is the shape that makes a future session ask whether
its parquets are stale. **They are not.** Three levers landed in the fast-class
front end — `blur::downscale_2x_into`'s de-interleave (`3fde060b`),
`RollingPlane::from_pooled`'s buffer growth, and the `color.rs` de-interleave
bounds checks — and all three are **bit-exact by construction**, not by
tolerance:

* the downscale's per-lane add order `(((a+b)+c)+d) * 0.25` is unchanged and
  nothing sums across lanes, so f32 non-associativity has nothing to bite;
* `from_pooled` hands back a buffer whose contents are overwritten before any
  reader can name it (and now always zeros, where `Vec::resize` left a prefix
  of the previous walk's pixels — strictly *more* deterministic);
* the bounds-check change moves no load and no operation, only where the range
  is checked.

**Evidence, not assertion:** a `to_bits()` A/B of the full feature vector,
pre-lane binary vs post-lane binary, over **160 cells** = 20 geometries × 4
arms (`156`/`15c`/`15f`/`944full`) × {serial, 3-thread} — **0 differing bits**
on 320 dumps of 924/944/956 features each — plus `cargo test --workspace`
1,548 passed / 0 failed including `v1_golden_bytes` and the 11-test
`fold_engine_parity`, and `cargo public-api` at ZERO delta on 1,280 items.

**So: no era break, no re-extraction, no re-training, no feature-set id
change.** Every 372/720/924/944 root, every canonical parquet, every bake and
every published verdict remains readable and comparable across this change.
The one thing that DID change is speed (1T wall −14.9…−18.0 % on the `156`
arm), and speed is not a column in any table.

Registered and **NOT flipped** in the same lane: **E3**, dropping the
front-end cube root from two Halley iterations to one, which removes 3 of the
6 `vdivps` per 16 pixels and is worth ~4–7 % of the fast walk. That one *is*
an era break — it moves every v1 byte on every corpus — and it is batched with
E1/E2 per the lane's era policy rather than taken.

**Amended same day — a FOURTH lever landed after this section was first
written, and it does not change the conclusion.** **L18** hoists the H blur's
30 strided `(src, dst)` gathers to one range check each
(`fused_blur_h_ssim_inner_v4x` **2,634 → 1,683** static instructions,
callgrind Ir **−15.2 %** on the kernel, walk-side **−6.48 %**; cumulative for
the lane **−15.44 %**). It moves WHERE a bound is proven, not what is read, and
it was re-gated by the SAME 160-cell `to_bits()` A/B **on the exact pushed
binary** — 0 differing bits, 0 empty dumps — plus `public-api` ZERO delta and
the lane-1 ratchets unchanged (175.0 allocations/walk, linearity exactly 2×).
So the "no era break, no re-extraction" conclusion above covers all four
levers, not three.

Record: `benchmarks/kernel_fastclass_2026-09-05.md` §"LANE 2"; registration
`docs/PLAN_KERNEL_FASTCLASS_2026-09-05.md` §"LANE 2".

### §3.42 — the feature-defect audit: the engines are clean, the servability is not (2026-09-05)

**Thought-why.** The user asked, verbatim, *"do we have bugs in feature
calculations?"*, and mid-lane added *"also make sure everything can be served"*.
The implicit model behind the first question was that the answer lives in the
committed record — read the campaign docs, collate the known defects, done.

**Actual-why.** Collating was necessary and not sufficient. The record holds 27
distinct defects, but it could not answer whether the CODE is correct **today**,
because almost every entry was measured against a stored table, and a stored
table cannot distinguish "the extractor is wrong" from "the extractor changed
after this table was built". Nine invariant probes at HEAD separate them, and
the separation is the finding: **the extraction engines are clean and the DATA
is one era behind them.**

**MEASURED at `3376baee`** (`zensim/examples/feature_invariant_probe.rs`, new;
gates `zensim/tests/feature_invariants.rs`, 10 tests, all passing; full suite
green 27/27 binaries):

| invariant | result |
|---|---|
| determinism (5 repeats; rayon pools 1 and **28**, the rung `fold_engine_parity` stops short of) | BIT-IDENTICAL, 12/12 geometries |
| engine parity (buffered v1-372 ↔ fold v1-only ↔ fold944-full ↔ both product engines) | **BIT-EXACT, 33 of 33** |
| tier parity v4x vs v3 (`ZEN_FIP_CAP_V3`, 22,397 cells) | max abs **3.48e-8**, max rel 8.75e-6, **0 cells over the golden tolerance policy**; `f372..943` bit-exact |
| degenerate inputs (5 pathological families × 4 geometries × 3 routes) | **0 NaN, 0 Inf**, 0 non-finite scores |
| width independence (tight / non-tight / odd / sub-64 / past `H_TILE_WIDTH`) | no width-class effect |

**Three data-side corrections that change what published numbers mean.**

1. **The 372 identity all-zero vector is FABRICATED and has never been
   measured.** Both product-facing entries — `metric.rs::identical_result`
   (behind every `Zensim::compute*`) and the free function
   `compute_zensim_with_config` (behind BOTH v1-372 extractors) — short-circuit
   `source == distorted` and synthesise `(100, 0, zeros)`, flagged
   `mark_identical()` so the model forward is skipped. So the property
   `dial_addressability`'s constant states as measured is, at 372, a property of
   the short-circuit. Computed on the same pixels the v1 block populates **144
   of 372** slots (max |v| **1.12e-3**); at 944, **286 of 944**, independently
   reproducing §3.36's count on a synthetic rather than real population and
   resolving it into exactly three classes: **15 reference-only**
   (`GRAD_SRC_MEAN` append-local-16, `LUMA_MEAN_REF` append2-local-2 — correct,
   `∂f/∂dist ≡ 0`), **12 `PJND_FRAGILITY`** (v2-local-21: 0.395 on the full
   walk, exactly **1.0** on a v1-only walk — a formula artifact both ways), and
   **259 fp residue** ≤ 1.12e-3. Now a gate.

2. **`FEATURE_SET_IDS.md` §1 failure #9 is an ERA artifact, not a code claim.**
   It reads *"the v1-372 `f0..155` is NOT the 944 fold's `f0..155` — 156 of 156
   slots differ, max abs 1.0214"*, measured across two STORED instruments of
   different eras. In one process at one commit on the same pixels they are
   **bit-exact on 372 of 372 slots at 11 geometries and both SIMD tiers**,
   through the public free function both v1-372 extractors actually call. The
   row remains a true warning about the instruments; read as a claim about the
   code it sends a reader hunting a divergence option C already closed.

3. **`V1FreeExtras` is silently inert unless `append_block` is also declared.**
   `append_block` does double duty — it declares the LAYOUT (720 → 924, with
   `append2_block` → 944) *and* enables the append COMPUTE — and every
   raw-moment slot lives at `f720+`. A `v1_only` walk asking for
   `V1FreeExtras::RawMoments` **without** it emits a 720-wide vector whose
   populated-slot count is identical to `V1FreeExtras::Off` (**228 vs 228**),
   with no error: the same failure shape as the fixed `==`-vs-`!=` emission-gate
   defect, reached by a different route. Class C is only partly affected (its
   twelve v2-348 `MSE` cells are inside the 720 layout and survive, 228 → 240;
   its twelve `LUM_*_ERR` append cells are not). Now a gate.

**THE SERVABILITY CENSUS (the mid-lane addition), measured through
`Zensim::compute` on the committed golden real pair.** Bake probing used the
fastclass2 campaign's `zensim/examples/serve_custom_bake.rs`; the profile-enum
sweep needed this lane's probe because that one takes a bake path.

| population | n | SERVED | REFUSED |
|---|--:|--:|--:|
| shipped bakes (`zensim/weights/*.bin`) | 11 | 8 | **3** |
| board bakes (distinct files behind the 467 fullevals, all on disk) | 433 | 32 | **400** |
| registered feature sets (`feature_sets_registry.json` → `sets`) | 14 | 3 | **11** |
| selectable shipped profiles (`ZensimProfile`, default features) | 10 | 8 | **2** |

Every refusal in all four populations is the identical
`ModelForwardFailed { "bake declares more input features than the caller
supplied" }`. One cause: `Zensim::compute` emits a **372-layout** vector with
`free_extras: Off`, so the rule is exactly `caller_input_width() <= 372` serves.
Declared widths across the 433 board bakes: 4 at 156, 28 at 372 (the SERVED
set), 1 at 504, 8 at 720, 2 at 924, **389 at 944**.

**The sharp end is `ZensimProfile::C` and `CHdr`.** Both are REFUSED on every
image; `candidate-profiles` is **ON by default** and
`c_sdr_mlp944_corrmix_2026-08-05.bin` is in the crates.io `include` list, so it
ships. And the short-circuit hides it at exactly the input a smoke test would
use: both still return **`IDENTITY (ref vs ref) score = 100.000000`**, because
`mark_identical` fires before the model is consulted. A health check passes for
a profile that cannot score anything else.

**Every SERVED case is also served-but-MISMATCHED, with no new mechanism.** The
runtime side is self-consistent (the bit-exact result above); the table side is
one extraction era behind (§3.37 / F3b — the default 372 root was built at
`ea16c7ee` 13:21, option C flipped at `56bbcda2` 15:43 the same day). So the
mismatch is temporal, and closing it is the registered-not-run 372 re-extraction
with no code change. **Consequence for the architecture lane:** when 944 bakes
become servable they will be served CURRENT-extractor vectors against verdicts
read on stored roots of a declared era — the servability fix must land with the
feature-set-id match check wired into the SERVING path, not only the verdict
path, or it converts a loud `ModelForwardFailed` into a silent wrong number.

**A method lesson worth more than any single number.** A first monotonicity pass
reported 176 / 127 / 109 amplitude-real non-monotone slots across three ladders.
Adding a plain **MSE(ref, rung) stimulus control**, emitted through the same
violation counter, killed one ladder outright: repeated radius-1 box blur is
**non-monotone on 12 of 12 images** (`29.13 → 26.02 → 29.11 → 31.62 → 34.53 →
37.66`), and every feature that "violated" on it dips at the same rung. The
stimulus was the defect. On the two control-validated ladders (additive noise,
quantization) **40 and 55 slots are persistently non-monotone**, and most are
correct: 62 violating series contain an exact `0.0` beside non-zero values — the
signature of a rectified one-sided feature, e.g. the `GLOBAL_CGAIN`/
`GLOBAL_CLOSS` pair, where heavy quantization with a `+step/2` reconstruction
offset increases contrast so contrast-LOSS properly collapses to zero. **Do not
accept a monotonicity result that has no stimulus control**, and recipes
imposing a monotonicity penalty should exclude the persistent set by name from
`ladder3.tsv` rather than by guess.

Registry: five entries in `benchmarks/eval_annotations.json`
(`profile-c-chdr-unservable-2026-09-05`,
`identity-score-cliff-fabricated-2026-09-05`,
`feature-set-ids-row9-is-an-era-artifact-2026-09-05`,
`free-extras-inert-without-append-declaration-2026-09-05`,
`nonmonotone-feature-slots-are-by-design-2026-09-05`). Record:
`docs/FEATURE_DEFECTS_AUDIT_2026-09-05.md`. Artifacts + `_MANIFEST.json`:
`/mnt/v/output/zensim/feature-audit-2026-09-05/`.

### §3.43 — the fleet can recompute features at a named formula revision, and the rev1 output is BIT-EXACT to the postC 372 root (2026-09-06)

The rev2 record's §3 registers WHAT to recalculate and flags that
`JobKind::Feature` had **no executor** — feature tables were produced through
`ScoreFile` + a `zensim-foldapp2*` metric name. That vehicle cannot carry a
formula revision: `ZENSIM_FORMULA_REV=1` and `=2` are different work with
different output bytes but serialize to the identical `JobKind::ScoreFile`, so
they collide on ONE content-addressed `JobId` and a rev2 cell would read as
"already done" off a rev1 ledger. The executor was built into zenfleet rather
than bypassed (zenmetrics `7740637e`, plan
`zenmetrics/docs/PLAN_REV2_RECALC_2026-09-06.md`).

**MEASURED — the rev1 correctness gate.** The executor's output at revision 1,
regime 372, against the stored postC root
(`/mnt/v/zen/zensim-training/2026-09-05-full-features-372-postC/`,
`build_commit 4fbd8ff8` — the commit the executor's zensim is at), compared as
`to_bits()` with row alignment verified on `ref_basename` **before** any value
is read:

| corpus | producer of the stored table | rows | cells | differ |
|---|---|--:|--:|--:|
| csiq | `extract_features_372col` | 866 | 322,152 | **0** |
| tid | `zensim-validate --extract-only` | 3,000 | 1,116,000 | **0** |
| kadid | `zensim-validate --extract-only` | 10,125 | 3,766,500 | **0** |
| | | **13,991** | **5,204,652** | **0** |

Covering both producers is not a formality: they differ in **decoder crate AND
zensim entry point** — `extract_features_372col` uses `zen_decode`
(zencodec magic-byte detect → the five imazen codecs) with
`compute_zensim_with_config`, while `zensim-validate --extract-only` uses
`image::open` with `compute_zensim_with_ref_and_config` (the cached-ref path).
The fleet executor uses the imazen decoders with the cached-ref path — a third
combination — and reproduces both bit-exactly. This corroborates, from a
different lane and a different code path, R6's own C1 control
(`benchmarks/f4_arm_decision_2026-09-05.md` §1: the `ssim2` arm is `cmp`-clean
against the postC root's source CSVs at `ceb86c2d`).

**⛔ A GAP THE GATE FOUND: `live` cannot be re-extracted by the fleet today.**
The stored LIVE table was built from **`.bmp`** (measured: its `ref_basename` is
`bikes.bmp`, and the alternative `live_r2_pairs_png.tsv` misaligns against it,
so the PNG list is NOT what produced it). zensim's `zen_decode` handles BMP /
PNM / farbfeld via `zenbitmaps`; **zenmetrics' `decode_image_to_rgb8` has no BMP
arm**, so the executor fails loud with "could not detect image format" — which
is the fail-loud contract behaving correctly, not a silent wrong answer. LIVE is
779 rows of the 42,470 re-extractable rows in the 372 root (1.8 %). Fix, not
run: add a `zenbitmaps` BMP arm to `zenmetrics-cli/src/decode.rs`. Note also
that LIVE's stored row ORDER is not its pairs-TSV order, so it needs a key-based
join, not the positional compare the other three corpora use.
**✅ BOTH DONE — §3.45 corrects the "key-based join" framing** (it's a stable
sort, not a join) **and reports the fix + G-BITEXACT re-run at 0 differ.**

**The 6 byte-COPY corpora stay excluded from every count above** (aic4,
nonphoto, imazen26, sdr25, hfnlproxy, hf_nearlossless): they are copies in the
postC root and are not re-extractable on this box, so a zero delta there is an
identity, not evidence.

**Wave declaration is BLOCKED, and correctly so.** R6 has not published an F4
arm (`benchmarks/f4_arm_decision_2026-09-05.md` ends at its §2 correction, with
no verdict section), so no rev2 job may be declared — its rows would be at an
arm nobody chose. What is unblocked and done: the executor, its gates, and the
rev1 bit-exactness proof.

**Two R6 findings that change how the wave must be declared**, folded in here so
the next session does not re-derive them:

1. **F4's blast radius keys on POOL STATE (`feature_set_id`), not on width**
   (R6 §2). A zeroed-pool 944 table sees 36 moved slots; the 2026-09-05
   pools-live 944 instrument sees **132** — as sensitive as 372. A wave that
   assumes "944 ⇒ 36 slots" under-declares by 96 slots per pools-live table.
   The declare therefore carries `feature_set_id` per cell and the executor
   echoes it into every row; it is **declared and echoed, never re-derived**,
   because `zensim::feature_set_id` is that id's owner.
2. **Decoder era is an input.** Every emitted row records the decoder that read
   its pixels, because §3.34 measured decoder era at **73 %** of the
   extractor-era shift — a confound of comparable size, not a footnote.

Harness: `zenmetrics/scripts/jobsys/rev2_bitexact_gate.py` (refuses a sub-64
image and a misaligned row rather than reporting a number).

### §3.44 — R6's four-arm F4 tables, and the corpus fact that decides the arm (2026-09-05)

**What was built.** Four complete 372-col extractions of the same pixels, one
per F4 luminance arm (`ZENSIM_SSIM_LUMA` ∈ `{ssim2, c1, lorentz, clamp}`, one
binary at `ceb86c2d`, no rebuild between arms), at
`/mnt/v/output/zensim/rev2-2026-09-05/r6/`:

| leg | rows/arm | source |
|---|--:|---|
| safesyn | 196,086 | bitstreams (`.jpg` 111,068 / `.avif` 34,001 / `.jxl` 26,362 / `.webp` 24,655), pre-scanned 0 missing |
| cid22val / kadid / tid / konjnd / aic3 / csiq / live | 20,670 total | `build_eval372_root.sh`'s own datasets |
| ladder dial grid | 9,593 | the 2026-09-05 floor-dense instrument's pairs list |
| negative-tail probe | ≤2,000 | the registered `ssim2 < 0` rule on the arm's own safesyn |
| identity | 400 | self-pairs over 3,277 distinct references |

Each arm's `evalroot/<arm>/` is a drop-in `bake_verdict --features-root`, with a
`_MANIFEST.json` carrying `build_commit`, the arm token, and the decoder era per
format. **Six corpora are ABSENT, not copied** (aic4, nonphoto, imazen26, sdr25,
hfnlproxy, hf_nearlossless, plus pipal): they are byte-copies in the postC root,
and a copy inside an arm root would be a different arithmetic revision wearing
that arm's name.

**★ The corpus fact.** `clamp` differs from the shipped form only where
`(mu1−mu2)² > 1`, so it is a pathology DETECTOR. Across **217,756 rows** it moves
**0 cells**, and no row anywhere reaches `|f| > 2` — against the 5,814,302 that
motivated F4, which is a property of `bigcodec_hqdedup_traindigits_2026-07-02`
(2.3 M rows, **no local pixels**). **No corpus this box has pixels for enters
F4's pathological regime.**

**Two era facts a wave must carry forward.** (1) F4's blast radius keys on
`feature_set_id`, not width: `ext944`/`ext924` have `f156..371` all-zero (36
moved slots) while the 2026-09-05 pools-live ladder grid is 98.7 % nonzero there
(132). (2) These tables' decoder era is `shared/zen_decode.rs` at `ceb86c2d`;
§3.34 priced decoder era at 73 % of an extractor era, so a wave that decodes
elsewhere is measuring two changes at once.

**★ THE VERDICT: `SsimLumaForm::REV2_LUMA` = `Clamp`.** No arm won a rank
majority (`c1` met the 2-of-3 CI-excluding condition in 1 of 6 variants and fails
the healthy-cell gate at 29.4 M cells / worst 0.771 against a 1e-4 bar; `lorentz`
fails it at 24.0 M / 0.0901), so the pre-registered rule-4 branch fired and chose
the smallest healthy-cell perturbation — `clamp`, at **0 cells and 0 max**. It is
BIT-IDENTICAL to revision 1 through features, Gram, solve, spline and ZNPR bytes
(all 24 bakes; the six `clamp` bakes match their `ssim2` siblings sha for sha).
`SHIPPED_REVISION` stays `Rev1`.

**⛔ And the unbounded feature that DOES fire on these tables is `contrast_inc`,
not F4:** max rev1 `|f|` over all 372 slots is **36,465.7** (safesyn), 3,598.2
(LIVE), 927.9 (TID), 618.3 (KADID) — every one a `contrast_inc` slot
(`hf_energy_gain = max(0, hf_dst_L2/hf_src_L2 − 1)`), unbounded above by exactly
F4's flat-source mechanism, while `var_loss`/`tex_loss` are bounded at 1.0.
**122 of 779 LIVE rows (15.7 %) exceed 100.** No registered defect, no arm, no
gate — reported, not fixed.

Record: [`../benchmarks/f4_arm_decision_2026-09-05.md`](../benchmarks/f4_arm_decision_2026-09-05.md).

### §3.45 — §3.43's two prerequisites closed: the 372 corpora are on the LAN store, and `live` re-extracts bit-exact (2026-09-06)

Ops lane, independent of the F4 arm token §3.43/§3.44 are blocked on. Both
findings live in `~/work/zen/DATA_PROVENANCE.md` (`rev2-lan-stage-2026-09-06`)
and zenmetrics `docs/PLAN_REV2_RECALC_2026-09-06.md` §7.5 — this section is the
short version for readers of this ledger.

**LAN staging.** The 8 re-extractable 372 corpora (cid22, kadid, tid, csiq,
live, aic3, konjnd, pipal — 43,870 rows, ~25 GB) are synced to
`s3://codec-corpus/eval372-rev2-2026-09-06/<corpus>/`. Verified reachable with
NO executor code change: `zenmetrics-cli`'s `resolve_source` /
`resolve_feature_input` already handle `s3://…` for both sides of a Feature
job — `zenmetrics scripts/jobsys/verify_lan_stage_reachability.sh` proves
bit-identical features between the staged `s3://` paths and the local
originals on csiq/live/pipal/konjnd (0 cells differ), and `r7900x` read 3
staged files by sha256 over SSH (3/3 matched). **PIPAL is staged as the full
23,200-pair set** (200 refs x 116 distortions, `load_pipal` returns it
un-truncated) — 1,400 more than the postC root's 21,800; no per-reference
selection rule was found (not `--max-images`, which defaults to 0/all and
isn't passed by the build script), so this is registered as open, not solved.

**LIVE re-extracts bit-exact.** The BMP arm §3.43 registered landed in
zenmetrics (`crates/zenmetrics-cli/src/decode.rs`, commit `5be5da80`):
`zenbitmaps::decode_bmp` funnelled through every named `PixelLayout`, gated
behind a new `bmp` cargo feature, 4 unit tests round-tripping real BMP bytes
from `zenbitmaps`' own encoder. **§3.43's "needs a key-based join" was an
overstatement of the real fix**: MEASURED, a **stable sort of LIVE's pairs by
`basename(ref_path)`** (Python `sorted(key=…)`, matching the Rust
`rows.sort_by(|a,b| a.0.cmp(&b.0))` at
`zensim-bench/examples/extract_features_372col.rs:216` that
`extract_features_372col` runs before writing its CSV) reproduces the stored
table's exact row order — verified positionally against BOTH `ref_basename`
and `human_score` on all 779 rows, zero mismatches. No join key beyond the
reference's own filename was needed. Gate re-run
(`zenmetrics scripts/jobsys/rev2_bitexact_gate.py`) with the fixed binary and
the stable-sorted pairs:

```
compared 289788 cells (779 rows x 372 features)
RESULT: BIT-EXACT — 0 of 289788 cells differ
```

§3.43's G-BITEXACT table (csiq/tid/kadid) plus this result now covers all 4
of the postC root's non-byte-copy producers this box can re-extract: 14,770
rows, 5,494,440 cells, 0 differ. No arithmetic changed — this is entirely
fleet-side (decode dispatch) + data-side (staging, and a row-order finding).

### §3.46 — F17: the audit's "one live arithmetic defect" was not the one that fires (2026-09-06)

**Thought-why.** R6 chose F4's bounded luminance arm and, chasing its own
"max over ALL slots" column, reported an unbounded feature that is not F4 and
did not fix it. The rev2 lane's remaining job looked like recalculation.

**Actual-why.** `contrast_inc` = `max(0, var_dst/var_src − 1)` is unbounded by
exactly F4's mechanism, and on every corpus this box has pixels for **it is the
unbounded value that actually occurs**. MEASURED over 216,756 real pairs on
eight corpora, all 372 slots: its twelve slots are the **top twelve by maximum**
(worst **36,465.74**, safesyn) and the thirteenth slot in the whole vector is
**1.972**. The separation is a partition, not a tail judgement. Against the gold
holdout's own p99.9 over those slots (CID22, 0.34687) that is **×105,127**. F4's
5,814,302 belongs to a bigcodec sweep with no local pixels and moves **zero** of
these rows.

The family's other two members — `var_loss`, `tex_loss` — max at exactly
**1.000000**, because their numerators are `max(0, src − dst)` and their own
denominators bound them. The `var_src > 1e-10` guard is a threshold, not a
stabiliser.

**Data consequences.**

* **`FormulaRevision::Rev2` now batches THREE eras** — `v1ssimcap` (F4),
  `freecomp` (F5), `v1hfgain` (F17). One era boundary, one recalculation. A wave
  declaring only the first two under-declares every table by twelve slots.
* **Blast radius: the same twelve at every shape**, MEASURED at `944full`
  (pools live), `924` (pools zeroed), `372` and `156` — unlike F4, whose count
  is 132 at 372 / pools-live 944 and 36 at the zeroed roots. Combined v1-side:
  **144** at 372 and pools-live 944, **48** at `ext944`/`ext924`.
* **Exposure, per shipped bake, from each bake's OWN transform block:** A and B
  are winsor-guarded on their F17 slots; **D — today's SDR default — has no
  transform block at all** and reads f116 (measured max 1,380) and f155 (2,127)
  raw into a 28-input linear head; **CHdr reads all twelve at `identity`**; C has
  10 of 12 guarded; BHdr 4 of 6. A bake-side transform is therefore not the
  answer to F17: it is what is already deployed and what the default lacks.
* **Serving a rev1 bake rev2 features is ~4 orders cheaper than an era shift.**
  Shipped Profile D, NOT refitted, on the most disruptive arm: SROCC moves
  |≤6e-5| on all seven corpora and the dial moves mean −0.0003 (CID22) to −0.012
  (LIVE), worst pair −0.935, with 0.00–0.64 % of pairs past the 0.5-pt
  materiality bar — against −4.98 (extractor era) and −3.658 (decoder era) on
  record for shipped B. Does NOT transfer to CHdr, which reads all twelve raw.
* **⛔ A FOURTH hand-copy is in another repo.** `zenmetrics/crates/zensim-gpu/
  src/pipeline.rs:1305-1310` carries the same expression. A rev2 wave using the
  GPU oracle must land the matching change there first, or pin the oracle to the
  CPU walk. Its own `cpu_gpu_feature_sweep` / `cpu_parity` / `extended_parity`
  gates would catch the divergence.

**The arm is DECIDED: `HfGainForm::SaturatingExcess` = `g/(g+1)`**, by the rule
pre-registered before any table was extracted. Five arms, one binary, 216,756
rows, two slices x two solvers. It is the ONLY arm passing the structural gates
— `bexcess` fails order-preservation with **263,195** inversions (it reads the
MAGNITUDE, not the ratio), `cap` fails it with **67,224** new ties (F4's `Clamp`
analogue, free there and not here), `log1p` has no structural bound and reads
**10.504** — and it also wins a strict majority of {CID22, KonJND, AIC-3} with
CI-excluding paired-bootstrap deltas in 2 of 4 variants.

**What it buys, MEASURED: LIVE 0.7357 -> 0.9500 (+0.214)**, TID +0.033, KADID
+0.021, CID22 +0.0027..+0.0090, AIC-3 +0.0009..+0.0032. LIVE holds 122 of 779
rows with a cell above 100 — the unbounded slot was wrecking the fit there, not
decorating it.

**What it costs, MEASURED and NOT attributable to the arm: KonJND regresses
-0.013..-0.080**, CI-excluding on 3 of 4 variants — and every bounded arm shows
the same sign, so the near-threshold corpus is using the unbounded magnitude and
no bounded form returns it. If that axis matters it is an APPEND slot under the
append-only numbering, never a reason to keep an unbounded one. The dial does
not regress (monotonicity within 0.0011, tied 0.0000-0.0001, negative-tail
fraction within 0.022 and better on one variant) at a reach cost of 2.2-11.8
points.

**Nothing was rewritten.** `ssim_form::SHIPPED_REVISION` is `Rev1`, the F17
revision is `Proposed`, and every stored table and published verdict is
unaffected. The revision-1 control arm reproduces the prior wave's tables on all
ten legs AND its four bakes sha for sha, end to end. The lane's tables, 20
bakes and `_MANIFEST.json` are at `/mnt/v/output/zensim/rev2-2026-09-05/r6b/`.

### §3.47 — the corruption head's four theories, tested at rev1 (2026-09-06)

**Round row.** Pre-registered (`docs/PLAN_CORRHEAD_THEORIES_2026-09-06.md`, pushed
at `bdb46b75` before any result existed), executed against the 2026-09-05 rev1
tables with **no re-extraction and no fleet**. The incumbent `d228` split.tsv is
read VERBATIM — never re-derived — and the reconstruction is parity-gated against
that head's own `metrics.json` (subclass counts and fold sizes exact) before any
arm is fitted. Slice `f0..227` throughout, the slice free at D's
`V1PoolsMode::Peaks` walk.

**Data**: `im26_corruption_372_postC` (116,928 + 348), `negrich_372_postC`
(60,000), the 2026-09-05 ladder (9,593), the `gb82_dog` gate grid (2,016), and
Profile D's dial scored over every row via `predict_features_with_bake
--bake-post raw`. Cached at
`/mnt/v/output/zensim/corruption-head-2026-09-05/theories/dataset_rev1.npz`
(233 MB) with a `_MANIFEST.json` carrying `build_commit` and per-file sha256.

**The headline is a model-form result, and it dissolves the other three
questions.** pAUC over ladder-FP ∈ [0, 5 %]: **logistic 54.20 → `mlp64_32` 97.73
→ `hgb` 98.11**, train ≈ test for every arm, same ordering on the single-source
gate grid that removes the content degree of freedom. The near-lossless false
positives §3-era work called "a separability limit of the feature set" are
separable in the SAME 228 features (q ≥ 95 FP **50.00 % → 2.38 %** at higher
detection), and the miss profile (`whole`-region, low-amplitude edits; worst
family 17.2 %) lifts to a worst family of 82.8 %.

**Two era/measurement lessons worth carrying.** (1) *Never compare corruption-head
arms at a fixed threshold.* Removing positives shifts the balanced class prior, so
a leave-one-family-out at T = 0.9 raises FP **and** detection together and the real
effect is invisible; at matched FP the greedy 8-family cull reads −4.30 pt FP and
+12.88 pt detection, both CIs excluding zero. (2) *Isotonic plateaus break matched
operating points.* The first pass reported an arm at FP exactly 0.00 % for three
different targets because its top plateau exceeded the FP budget; the fix is a
rank tie-break inside the plateau (`eps * rank(p_raw)`), after which every arm
lands on the identical achieved FP at every target.

**ERA**: rev1 (`ssim_form::SHIPPED_REVISION = Rev1`, post-option-C `56bbcda2`).
Revision 2 changes 12 basic slots this head reads, so none of these numbers
survive that flip unmeasured. **Nothing was wired in, no bake replaced, no ZNPR
emitted** — the winning forms have no wire format (`emit_znpr` writes one identity
layer from `coef_`; the owner now refuses `--bake-out` for them). Record:
`benchmarks/corruption_head_theories_2026-09-06.md`.


## §3.46 — the rev2 372 eval root, and the two data defects the fleet gate found (2026-09-06)

**New root:** `/mnt/v/zen/zensim-training/2026-09-06-full-features-372-rev2/` —
8 corpora, **43,870 rows**, era `v1postc_rev2`, feature-set id
`basic+peaks+masked+iw@w372/v1postc_rev2#d16a1091`, build_commit `88477e38`.
Produced by the **zenfleet Feature executor** (run
`s3://zentrain/jobs/rev2feat372-20260906`, 3,907 jobs, one 24-core box, ~2.5 min),
not by a local extractor. Mirrored to `s3://zentrain/eval-roots/…` and Tower.
It is the first eval root at formula revision 2 and the first one whose cid22 leg
is decoded entirely by imazen codecs.

**Validity:** the SAME binary at `ZENSIM_FORMULA_REV=1` is bit-exact to the postC
root on csiq / tid / konjnd / live / aic3, and at `=2` it is bit-exact to the R6b
lane's `satexcess` arm on csiq / tid / konjnd / live / kadid / aic3
(**6,092,616 cells, 0 differ**), through a different code path from
`extract_features_372col`. That independently confirms what R6b could only
derive: at width 372, revision 2 IS the `satexcess` arm.

**Defect 1 — konjnd's staged pairs named the wrong file on 4 of 1,008 rows.**
The level was derived with round-half-to-EVEN; the stored root used round-half-UP,
so the four rows whose PJND is exactly `X.5` (`SRC0437` 58.5,
`SRC0823`/`SRC0904`/`SRC0993` 30.5) pointed at `_030`/`_058` instead of
`_031`/`_059`. Feature deltas to **0.068** — content, not rounding. Corrected in
this root; the correction is proved, not asserted (re-extracting the four with the
half-up file is bit-exact to the stored root).

**Defect 2 — 12.5 % of CID22 is decoded by a NON-IMAZEN decoder, in every root
before this one.** 536 of cid22's 4,292 rows differ from both the postC root and
the r6b rev2 root, and they are **exactly** the `.jpg` rows (3,756 png + 536 jpg;
100 % of jpg differ, 0 % of png), ~360 of 372 slots each, max |Δ| 9.5e-3. Cause:
the postC root's cid22/kadid/tid/pipal legs come from `zensim-validate
--extract-only`, which decodes with the third-party `image` crate
(`zensim-validate/src/main.rs:830, 854, 1720, 1744`). kadid (png) and tid (bmp)
agree because both decoders give identical pixels there; **JPEG is where they
part.** `extract_features_372col` was migrated to `zen_decode.rs` on 2026-09-04;
`zensim-validate` was not. CID22 is the gold human-MOS holdout, so every published
CID22 number is read partly through a foreign decoder, and the manifests'
`decoder_era` strings name only `zen_decode.rs`. **The price — what a bake's CID22
SROCC does across the two decode eras — is NOT measured; it needs one bake scored
on both tables.**

**Also registered, not landed:** `powf` makes the extractor libc-dependent on 144
of its 372 slots (`benchmarks/libm_pow_nondeterminism_2026-09-06.md`). The fleet's
musl build and the dev box's glibc build of the SAME source differ by one ULP on
~0.024 % of cells; glibc 2.36 and 2.43 are bit-identical. The wave was unblocked by
building the fleet binary against the base image's own glibc, not by changing any
arithmetic.

> **UPDATED 2026-09-06 — the OWNER and the ERA landed the same day; the FLIP did
> not** (`benchmarks/libc_determinism_2026-09-06.md`). `zensim::det_math`
> (`RootForm::{LibmPowf, NestedSqrt}` + `DetRoots`), defect `DEFECT_F18`, era
> **`v1detroot`** on `FormulaRevision::Rev2`, override `ZENSIM_ROOT_FORM=libm|sqrt`.
> `SHIPPED_REVISION` is still `Rev1` and the default form is still `LibmPowf`, so
> **no stored table and no shipped byte moves, and the glibc-rebuild workaround
> above is still the operative one for revision-1 extractions.** Three corrections
> to the sentence above, all measured:
> * **the era is 156 slots, not 144** — `(M4/n)^0.25` is also the v2 `ssim_dev4`
>   pool, in three finalizers the discovery record's v1-block table did not reach;
> * **the rest of the feature path was already libm-free**, established by reading
>   the sites rather than inferring from one corpus (sRGB→linear is `linear_srgb`'s
>   LUT, the opsin cube root is `cbrtf_fast`/`cbrt_midp`, the SIMD PU-XYB path is
>   magetypes' `log2_midp_precise`/`exp2_midp_precise`);
> * **the SCORE is exposed too and is NOT fixed** — `metric.rs` calls `powf` at
>   `0.5979 / 1.2244 / 0.6130 / b`, none a power of two, so a dial value is
>   libc-dependent on every profile. **CLOSED the same day by §3.51 (F19, era
>   `scorepow`), which also MEASURED that this bullet's exposure survives F18
>   exactly: 1 of 220 scores differ across libcs with `v1detroot` applied and
>   nothing else.**
>
> **⚠ Provenance consequence for anyone flipping rev2:** `v1detroot` invalidates
> any `ZENSIM_FORMULA_REV=2` table extracted BEFORE 2026-09-06 — the R6b lane has
> some. Reproduce those from a new binary with `ZENSIM_ROOT_FORM=libm`.


### §3.48 — the corruption head's BLAS/OpenMP thread-count nondeterminism is fixed at the owner (2026-09-06)

**Ledger ROUND 100.** §3.47's own theories lane found, and left open, that
`train_corruption_head.py`'s bake was a function of the ambient BLAS/OpenMP
thread count — the identical recipe at ambient 1/4/8/28 threads produced four
different `corruption_head_d228.bin` files, and the shipped 2026-09-05 `d228`
head is the 28-thread one, so a `run-heavy --jobs 8` re-run did not reproduce
it. This round fixes it at the owner.

**Fix:** force `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/`MKL_NUM_THREADS`/
`VECLIB_MAXIMUM_THREADS`/`NUMEXPR_NUM_THREADS`/`BLIS_NUM_THREADS` to `"1"`
(unconditionally) before `numpy` is imported, plus `threadpoolctl.
threadpool_limits(1)` immediately after import. Module-level, so every caller
of `make_classifier` — the trainer's own `main()`, and `corrhead_theories.py`/
`corrhead_tests.py`, which import it rather than subprocess it — gets the pin.

**MEASURED (this box, `main` at `478bc28e` + the fix):** the exact `d228`
recipe (`corrhead_arms.sh`'s argv) at ambient 1/4/8/28 threads now produces
byte-identical `corruption_head_d228.bin` (sha256 `6f97b653ba5fea2d…`),
`..._w944.bin`, `metrics.json`, and the persisted weights `.json` — all four
files, not only the artifact the bug was originally characterized by. Before
touching the file, the UNPATCHED script at ambient 8 threads was independently
re-measured on this box first and reproduced §3.47's recorded 8-thread sha
(`23ad9c5b…`) exactly, confirming this round's repro setup matches the
original finding byte for byte before claiming a fix.

**The fix does not, and structurally cannot, reproduce the historical
28-thread shipped bake** (`da411c8c9cd6a6e216c81515714fecf76b7e3d0dcf38c9be2e11dc2f390fd8b2`)
**byte-for-byte** — pinning to 1 thread reproduces the natural single-thread
reduction order, which is a different reduction order from the historical
28-thread ambient (unpinned) run; `6f97b653…` is in fact exactly §3.47's own
recorded "1T" value. **The shipped `corruption_head_d228.bin` was NOT
replaced** — every file under `d228*/` is untouched (verified: mtimes and
sha256 unchanged). Registered delta, measured through the existing evaluation
path on the canonical `gb82_dog` held-out gate grid (672 triples, 372-wide,
scored via `predict_features_with_bake` — i.e. the actual baked bytes, not the
trainer's `CalibratedClassifierCV`-based `metrics.json`, which is a known
separate discrepancy this round does not touch): detection at T=0.9 moves
**83.929 % → 84.077 % (Δ +0.149 pt)**; FP on both `q10` and `q20` matched
honest anchors is **unchanged at 0.000 %**. The trainer's own held-out
test-fold curve moves by a comparable amount (T=0.9 detection 89.527 % →
89.424 %, per-family recall up to 0.38 pt), consistent with §3.47's own
"up to 0.4 pt" characterization of the analogous 8-thread case — and the
`split.tsv` partition itself (905,503 bytes) is byte-identical across every
run, pre- and post-fix: the PRNG-driven source split never depended on BLAS,
only the fitted weights did.

A synthetic (20,000×50) smoke test of `make_classifier("hgb", seed=0)` — the
`HistGradientBoostingClassifier` candidate §3.47 introduced, whose histogram
building threads via `libgomp` rather than BLAS — found no thread-order
sensitivity at that scale, either before or after this fix. It is not shipped
(`can_bake` refuses `--bake-out` for non-`logistic` forms) and this round does
not claim its histogram building is provably order-invariant in general; the
fix's env-var mechanism covers it regardless, since OpenMP reads its
thread-count env var at first use rather than at library-load time, which is
also why the pin has to be env-var-based rather than only
`threadpoolctl.threadpool_limits` (the latter only clamps libraries already
loaded at call time, and `--model hgb` loads `libgomp` lazily, after the
module-level `threadpool_limits` call has already run).

**Status: FIXED, not narrowed.** The mechanism is closed for every future run,
on every box, at every thread count. Whether to retrain and ship a new
`d228`-class bake against the now-deterministic path (a documented <0.15 pt
detection / 0 pt FP change on the canonical gate) is a separate, ungated
product decision this round does not make.

New tool: `scripts/v_next/corrhead_determinism_gate.py` — runs the recipe at
N ambient thread counts, asserts byte-identical bakes (exit 0 = holds, exit 1
= regression, exit 2 = could not run), and reports the shipped-vs-new gate-grid
delta when they differ. Record: `benchmarks/corruption_head_theories_2026-09-06.md`
§11 (addendum to §9).

**Addendum (same day) — three more roots and a bigger decoder finding.** §3.46
above covers the 372 root. The same wave also produced:

* `/mnt/v/zen/zensim-training/2026-09-06-full-features-944-rev2/` (era
  `era2r4_rev2`) — the **first rev2 extraction at a width where F5/`freecomp` is
  reachable**. `f156..371` measured structurally zero across five corpora; its
  `f0..155` agrees with the 372 root to **1.96e-8** (walk order, not formula), so
  the two roots must not be column-mixed.
* `/mnt/v/zen/zensim-training/2026-09-06-full-features-372-rev1-fleet/` (era
  `v1postc`) — the **CONTROL**: same producer, decoders, image and libc at
  revision 1, so `rev1-fleet vs postC` isolates the DECODER and `rev1-fleet vs
  rev2-fleet` isolates the REVISION.
* `/mnt/v/zen/zensim-training/2026-09-06-safesyn-rev2/` — the 196,086-row rev2
  training leg, and **the finding that matters most**.

**Both confounds are now PRICED**, with the control root:

* **The CID22 foreign decoder costs ≤ 1e-4 CID22 SROCC** on all three shipped SDR
  bakes (D +0.8633→+0.8632, B +0.8821→+0.8821, A +0.8655→+0.8654). A provenance
  and reproducibility defect, **not** a ranking one; annotated, not invalidated.
* **An unrefitted rev2 flip costs the SDR default exactly 0.00000** across CID22,
  KonJND, AIC-3, CSIQ, LIVE and TID. A moves slightly up (+0.0003 CID22, +0.0009
  CSIQ, +0.0008 TID); B moves ≤0.0014 in mixed directions. This says nothing about
  a REFITTED bake, where the refit lane measured the opposite sign in the D chain.

**⛔ THE TRAINING LEG DOES NOT TRANSFER — the fleet's AVIF decoder is a different
era.** The fleet safesyn table vs the R6b lane's local one: same pairs, same
revision, row alignment 0 of 196,086 mismatched on both `ref_basename` and
`human_score` — yet **51.2 % of cells differ, worst 0.136**. Per container:
`.jpg` 63.8 % of rows at worst **1.11e-07**, `.jxl` 64.5 % at 1.04e-07, `.webp`
63.9 % at 8.57e-08 — all rounding — while **`.avif` differs on 99.3 % of its
34,001 rows at 0.0217**, five orders of magnitude larger. zensim's
`shared/zen_decode.rs` at `e34f937d` and `zenmetrics-cli/src/decode.rs` at master
do not resolve to the same AVIF decoder, and zenavif is under a backend-rewrite
hold. The eval corpora never saw this because their distorted sides ship as
decoded PNG/BMP. **Which decoder is right is NOT decided** — both are imazen — and
a fleet training leg needs the AVIF decoder **pinned** to the era of the tables it
will join, as a declared and checked property of the image rather than an accident
of which commit it was built from.

**PIPAL in all four roots is FEATURES ONLY and NOT SCOREABLE**: the staged pairs
TSV is the one of eight with no target column, so there is no `human_score`.
Deliberately not repaired by deriving an elo, which would risk a normalisation
that does not match `load_pipal`'s.

---

## §3.49 — the corruption head becomes servable: `ZCTH` v1, and one wrong reading of sklearn's own source (2026-09-06)

**Ledger ROUND 101.** §3.47's theories lane concluded the corruption head should
be a gradient-boosted tree and recorded that it could not be shipped, because
"a nonlinear head has no wire format here". This section records the format, the
gates, and the one thing that went wrong on the way — which is the part worth
re-reading.

**Thought-why vs actual-why, the entry that earns its place.** The evaluator has
to reproduce sklearn's isotonic calibration. `IsotonicRegression._build_f`
constructs a `scipy.interpolate.interp1d(kind="linear")`, so the obvious move is
to read `interp1d._call_linear` — leftmost `searchsorted` bracket,
convex-combination evaluation — and reimplement it. That was done, with a unit
test asserting the value that code produces, and **both were wrong**:
`interp1d.__init__` routes plain `linear` to `_call_linear_np`, a one-line call
to `np.interp`, which uses the **rightmost** bracket and the **slope** form.
MEASURED on a real 90-knot fit over 25,092 queries, the `np.interp` form is
**bit-identical** to `iso.predict` and the scipy-source form is off by up to
**1.11e-16** — small, and located exactly on the knots an isotonic fit is made
of. **Reading the constructor is not reading the call.** A dispatch inside
`__init__` can hand you a different algorithm than the method you read, and the
divergence lands precisely on the case your data consists of. Caught only by
comparing candidates numerically against the fitted object instead of trusting
the reimplementation.

**The format is not a ZNPR metadata blob, and the reason is a defect class this
ledger already carries.** `zenpredict` is frozen at the `zenanalyze-api`
contract (USER DIRECTIVE 2026-07-19), and — independently — every consumer that
holds a `zenpredict::Model` dispatches through `Predictor::predict`. A tree in a
`metadata[]` entry behind a plausible identity layer would be silently
mis-scored by anything that did not know to look, which is the `--regime 944`
shape. `b"ZCTH"` makes the confusion a refusal at byte 0.

**What is now MEASURED and should not be re-derived:**

* The Rust evaluator reproduces sklearn's `decision_function` at **0 ulp** over
  35,607 rows (the tree walk is exact arithmetic, so 0 was the bar), and the
  calibrated probability to **max |Δ| 3.330669e-16** — all of it `exp`, none of
  it moving a single deadband decision.
* Through `bake_verdict`, the tree head reproduces the theory lane's gate-grid
  row **exactly**: DEPLOY `pass_q20` **0.9985119047619048** = 671/672. That also
  settles, by measurement, that the theory lane's `rank_break` tie-break is a
  no-op at the `T = 0.9` deadband — the Rust path does not apply it and lands on
  the same 671.
* **`ZensimProfile::D` populates the peaks block `f156..228` at 72/72** on a real
  compare while zeroing only `f228..371`. "The `f0..f227` slice is free at D" was
  a claim about COST; this is the AVAILABILITY half, and it is what makes a
  372-wide head attachable to D at all. Both halves are now tests.
* **The incumbent LINEAR head is 2.7× more expensive to serve than the
  6,100-node tree** — 1.76 µs vs 659 ns, with the tree at 0.63× Profile D's own
  forward (95 % CI [−41.1 %, −32.0 %], zenbench 1T). The nonlinear head is not a
  speed tradeoff. The ZNPR path runs 372 feature transforms plus a 372-wide
  dense layer; the tree touches ~600 nodes and 228 standardisations.
* **The `hgb` bake is byte-identical across the determinism lane's BLAS thread
  pin**, at both caller widths — `HistGradientBoostingClassifier` never went
  through the lbfgs solve that made the logistic head thread-dependent. The
  logistic bake moves to `6f97b653…` under the pin, which is exactly the "1
  thread" value §3.48's table predicted, reproduced here independently.

**A property of the composition to know before using it.** `gate_score` floors a
flagged row to `min(perceptual, 0)`, so it can only sort a corruption **below**
an anchor whose own score is above zero. That holds for the gate grid's q20
anchors; it is not automatic, and a runtime test that used too severe an honest
anchor would have "passed" for the wrong reason. The G7 test asserts
`honest > 0` so the requirement is visible.

**Nothing shipped.** No weights replaced, no profile changed, no board cell, no
default-path behaviour, zero public API (`cargo public-api`, default build,
1284 items, zero delta). The `corruption-head` feature is off unless asked for
and the proposed public surface awaits the user's approval in
`docs/PLAN_CORRHEAD_SERVING_2026-09-06.md` §3. Everything is **rev1**; the REV2
WAVE's refit lane measured the same head at revision 2 separately (§8 of
`benchmarks/rev2_refit_2026-09-06.md`) and found the `hgb` candidate invariant.

Record: [`../benchmarks/corruption_head_serving_2026-09-06.md`](../benchmarks/corruption_head_serving_2026-09-06.md).
Artifacts + shas: `/mnt/v/output/zensim/corruption-head-2026-09-05/theories/_MANIFEST_ZCTH.json`.

---

## §3.50 — every F17 arm through the SHIPPED D chain: the bound costs rank, and the form the wide class chose costs the most (2026-09-06)

**Ledger ROUND 102.** §11.10a of the rev2 plan set `REV2_HFGAIN =
SaturatingExcess` from R6b's 147-coefficient lasso, where it wins CID22
**+0.00272** CI-excluding. The REFIT lane then ran the SHIPPED Profile-D chain
— same λ, [0, 1] target, 28 coefficients — on the same tables and measured the
same arm **losing 0.00456**, also CI-excluding, and named the cause: F17's
benefit is model-class-dependent. **A form chosen on one model class had never
been tested on the class that ships.** This section is that test: five arms ×
two slices + three winsor controls, pre-registered as plan §12 and pushed
(`1992505c`) before the first gram existed.

**Thought-why vs actual-why, the entry that earns its place.** The lane was
framed as "which bound is cheapest for a monotone sparse model" — i.e. as a
question about the VALUE the model reads. Measured first, from
`bake_block_profile` on the REFIT lane's own bakes, it is not that question at
all: **the revision-1 D reads ONE of the twelve F17 slots (f77) among 28 active
coefficients, and the revision-2 D reads FIVE (f12 f38 f64 f129 f155) among 33**
— at a228, one of 27 against seven of 39. An unbounded column with a 36,466 max
and a 2.6 %-above-1 tail has, after standardization, its variance carried by a
handful of rows; L1 drops it. Bound it and it becomes well-conditioned and L1
keeps it. **So "rev1 vs rev2 in the D chain" is a comparison of two different
SPARSE SUPPORTS, not of one model with a rescaled input** — which is exactly why
a single arm's delta cannot separate "bounding costs rank" from "*this* bound
costs rank", and why the per-arm sweep was the only instrument that could
answer it.

**The result, and it is one-signed.** Every arm that changes the feature loses
CID22 with a CI excluding zero, at BOTH slices, and loses CSIQ and LIVE at a156
as well. `satexcess` is the **WORST of the four** at both: **−0.00456** (a156) /
**−0.00406** (a228), against `log1p` −0.00088 / −0.00211 and `bexcess`
−0.00153 / −0.00206 — and the two cheaper arms are exactly the two R6b's
structural gates eliminate (`log1p` declares no bound; `bexcess` reads the
MAGNITUDE of `var_src`, not the ratio, and inverts the order 263,195 times).
Every arm also drops the `avif-rav1e` per-codec floor below revision 1's 0.6667
(`log1p` closest at 0.6154, three arms at 0.5897) while every arm IMPROVES
`jpeg` (0.7179 → 0.7692). Under the pre-registered rule NO arm is eligible at
a156; at a228 `cap` and `satexcess` are (floors equal on all five codecs) and
both fail the CID22 clause. **Step 3 fired. `REV2_HFGAIN` STAYS
`SaturatingExcess`, because step 4 forbids a per-class revision and no arm wins
both classes** — the trade is recorded for the user, not resolved here.

**The outlier read is the one that most changes how the arms should be
described.** On the 60 LIVE rows whose revision-1 `contrast_inc` exceeds 100 —
the rows the whole defect is about — `satexcess` at a156 is **worse** than
revision 1 (+0.93203 vs +0.93491); only `bexcess` and the winsor control beat
it there. At a228 the sign flips and every arm beats revision 1. **"Bounding
the feature fixes the ordering where it is pathological" is not a claim this
study can support.** *(A units correction on the way: `feature_rev2`'s "122 of
779 rows" is the CELL count of 9,348; the ROW count is 60.)*

**And the bake-side alternative is real, which the lane did not expect.**
`W-f17` — `winsor_p99 [p0.1, p99.9]` on the twelve F17 slots only, revision-1
features, clamp inside the fit — is the **ONLY** non-revision-1 model in the
study that keeps A7r 0 fails AND contract 6/6, and it has the **best
outlier-row ordering of all eight** (+0.94348). No era break, no recalculation,
no re-extraction. It costs CID22 −0.00348, **76 % of `satexcess`'s cost**.
Scope is the whole story: the SAME guard on all 372 slots — the shipped
Profile-B recipe — is the **worst** model in the study (CID22 −0.01536, KonJND
−0.02339, outlier-row SROCC **+0.87152**), because a p99.9 window fitted on
safesyn clamps LIVE's pathological rows onto a shared ceiling and deletes their
ordering. That is `cap`'s H5 new-ties failure arriving through the bake instead
of through the feature.

**A defect predicted from source and then measured.** `bake_dial_refit
add-winsor` carries *"everything the raw bake had (incl. its spline)
verbatim"*, so the literal `add-winsor → extend-top` chain leaves an
output-calibration spline fitted on the UN-clamped net — QUANTIZE-then-CALIBRATE
in another guise. Graded rather than argued: `W-all-carried` TIES revision 1 on
CID22 (−0.00002, CI [−0.00007, +0.00002]) and WINS CSIQ by **+0.04862**
CI-excluding — the largest single rank gain anywhere here, because it keeps
revision 1's weights and clamps only at inference — but it fails contract
**C6** (1 of 9,593 grid cells out-scores a perfect copy) and the `avif-svt`
floor. **A serve-time guard whose spline is REFIT would plausibly keep the CSIQ
gain without the C6 break; registered, not run.**

**Controls, because every number above rests on them.** This lane's `ratio` and
`satexcess` grams are **BYTE-IDENTICAL** to the REFIT lane's; its four
rev1/rev2 bakes are byte-identical modulo the tool-path-bearing
`zentrain.repro`; `extend-top` is a byte no-op on all ten arm bakes;
`densify`'s identity gate is BIT-identical 10/10; all five arms' identity
tables are all-zero on 372/372 slots so the 21-row identity anchor is
arm-independent by construction; and the `satexcess` a156 CID22 delta
reproduces **−0.00456 [−0.00549, −0.00364]** exactly from an independent gram,
fit, verdict and bootstrap. All sixteen bakes PASS the two-reference inversion
gate (G3 ≤ 0.07) — a pre-registered axis that turned out flat, reported as a
result rather than omitted.

**Nothing shipped.** `ssim_form::SHIPPED_REVISION` is `Rev1`,
`ZensimProfile::D` unchanged, `zensim/weights/` unchanged,
`benchmarks/feature_sets_registry.json` unchanged, the G-ADDR floor registry
NOT appended (these are R6b's arm probes, not the wave's canonical instruments),
no board cell, zero Rust changed.

Record: [`../benchmarks/rev2_d_arms_2026-09-06.md`](../benchmarks/rev2_d_arms_2026-09-06.md).
Pre-registration: [`PLAN_FEATURE_REV2_2026-09-05.md`](PLAN_FEATURE_REV2_2026-09-05.md) §12.
Artefacts + shas: `/mnt/v/output/zensim/rev2-d-arms-2026-09-06/_MANIFEST.json` (392 files).

## §3.51 — the SERVE-TIME F17 guard with a REFIT spline: a bounded D that costs nothing (2026-09-06)

**Lane:** REV2-D-GUARD. **Pre-registration:** `docs/PLAN_FEATURE_REV2_2026-09-05.md`
§12.7, pushed `a7a91263` before the first bake. **Record:**
`benchmarks/rev2_d_arms_2026-09-06.md` §11 addendum. **Artefacts:**
`/mnt/v/output/zensim/rev2-d-arms-2026-09-06/guard/` (`_MANIFEST.json`,
`build_commit`, 336 files). **Nothing installed.**

**What §3.50 left open.** Every arm that bounds `contrast_inc` in the FEATURE
costs the shipped D class CID22, and the bake-side alternative `W-f17` — which
refits the WEIGHTS on the clamped gram — costs −0.00348. The one combination
nobody had run was the one `add-winsor`'s own source made awkward: keep
revision 1's weights, clamp at SERVE time, and refit the output spline on the
guarded net. The tool CARRIES the pre-clamp spline, which is why
`W-all-carried` broke contract C6.

**The fix is ORDERING, not a patch.** `shared-anchor` forwards the anchor
through `build_fw_ops`, i.e. through the bake's own declared transforms, so a
refit that runs AFTER `add-winsor` is fitted on exactly what the runtime sees.
Two default-neutral owner extensions were still needed and landed at `e8e4ace2`:
`add-winsor --slots` (the guard could only cover all 372, and §3.50's own
finding is that SCOPE is what matters) and a REPEATABLE `shared-anchor
--anchor` (the shipped chain fits its spline over BOTH the 2,000-row negrich
dial anchor and the 21-row id100 identity anchor). Gate
`scripts/verify_winsor_scope_identity.sh`: omitting `--slots` re-emits two
stored PRE-change artefacts BYTE-IDENTICALLY, a scoped emit differs and declares
exactly 12 tokens over 372 lines, and two `--anchor` equals one physically
concatenated parquet — each with its own negative control.

**MEASURED.** Control `R-refit` (refit chain, no guard) is **rank-invisible**:
its spline differs from the fit-lasso one in 5 of 38 f32 knots by ONE ulp each
and every paired-bootstrap delta is **exactly +0.00000** on all seven corpora at
both slices, with identical G-ADDR, floors and inversion rate. Against that
control, **`W-f17-refit-p999`** — winsor `[p0.1, p99.9]` on the twelve
`contrast_inc` slots — reads **an EXACT CID22 TIE** (0.863666 vs 0.863666, CI
`[+0.00000, +0.00000]`) at both slices, holds **revision 1's per-codec floors
EXACTLY on all five codecs**, keeps **contract 6/6** and the inversion rate, and
**WINS LIVE** (+0.00032 a156 / +0.00022 a228, CI-excluding). Four arms clear all
four pre-registered criteria; p99.9 is recommended because **0 of CID22's 51,504
`contrast_inc` cells** exceed its window (831 at p99, 1,480 at p95), it clamps
0.100 % of training cells and 0.496 % of rows, and **p95 collapses f25's window
to `hi = 1e-9`** — `winsor_window`'s zero-constant fallback, which pins that
slot to a constant and is `cap`'s new-ties failure arriving through the bake.

**The registered follow-up is HALF right, and both halves matter.** `W-all-refit`
is `W-all-carried` with the spline refit and nothing else: the **C6 break IS the
carried spline** (above-identity 1 → 0, contract 5/6 → 6/6, `C3 frac<0`
0.9055 → 0.9100) and the **CSIQ +0.04862 survives intact** — but the
**`avif-svt` floor loss is NOT the spline**, surviving unchanged at 0.9744, so
that arm is still ineligible; **and the CSIQ gain belongs to the ALL-372 guard,
not to F17** (scoped to twelve slots CSIQ is a tie, −0.000004).

**The honest frame, and the next measurement.** On THIS bake the guard is nearly
inert: the shipped-D recipe re-fit on r6b's revision-1 tables reads **one** of
the twelve F17 slots (f77), so restricted to the support it clamps 197 of
196,086 safesyn rows, 64 of 779 LIVE rows and **0 of 4,292 CID22 rows**. The
result is that the bound is measured **FREE** in this class, not that it is a
rank win. `feature_defs::DEFECT_F17`'s exposure table names **f116 and f155**
for the SHIPPED ADD156 bake — a different support — so applying the same guard +
refit to the artefact that actually ships (which needs its own dial anchor and
instruments) is **REGISTERED, NOT RUN**.

## §3.52 — F19: the SCORE path stops being libc-dependent, and the fix is CHOSEN rather than derived (2026-09-06)

**Ledger ROUND 103.** §3.46's F18 entry named the score as a real, unmeasured
exposure and left it. This closes it, measures it, and states the one structural
difference that makes it a *different kind* of fix.

**The measurement that motivates it.** The F18 gate was extended from a
one-knob sweep to a 2×2 — one commit built twice (`x86_64-unknown-linux-gnu`
dynamic `libm.so.6`; `x86_64-unknown-linux-musl` `static-pie`), both era knobs
as runtime env vars on the same pair of binaries, 220 procedural cells:

| `ZENSIM_ROOT_FORM` | `ZENSIM_POW_FORM` | features differ | **score differs** |
|---|---|---:|---:|
| `libm` | `libm` (revision 1) | 21 / 81,840 | **1** / 220 |
| `libm` | `pure` | 21 / 81,840 | **0** / 220 |
| `sqrt` | `libm` | **0** / 81,840 | **1** / 220 |
| `sqrt` | `pure` (revision 2) | **0** / 81,840 | **0** / 220 |

Row 3 is the load-bearing one: **F18's fix left the score exactly as
libc-dependent as it found it.** The two defects are independent, which is why
there are two env vars and not one — and the table measures that rather than
asserting it.

**Why the arm is chosen, not derived.** `score_mapping_b` is `0.7` on every
shipped profile. F18's replacement was *unique* because IEEE-754 makes `sqrt`
correctly rounded, so `x^(1/4) = sqrt(sqrt(x))` is forced. **No such chain
evaluates `x^0.7`**, so F19's arm is an algorithm choice and the only property
purchasable is sameness: `libm::{pow, exp, log2}` — the pure-Rust fdlibm port,
already in the dependency graph twice, with no `fma` anywhere and — checked
against libm 0.2.16's own source — an arch `select_implementation!` on `exp`
alone, gated `x86_no_sse`, which no target this crate ships for selects.

**Error bound, derived over the score's own domain** (6,611 rows, priced
against `decimal` at 60 significant digits — no libm in the reference):
platform libm **max 1 ULP**, the port **max 1 ULP**,
`magetypes::nostd_math::powf_f64` **7.2e12 ULP**, a *perfectly rounded* f32 pow
**1.4e10 ULP**. The brief's suggested reuse of `log2_midp_precise` /
`exp2_midp_precise` is **not available at this width** — those exist only on
`f32x4`/`x8`/`x16`, and the f64 scalars that do exist are documented lowp
(`log2_f64(1.0)` returns `−1.87e−6` where the answer is exactly `0`). Even a
perfect f32 route destroys the head p-norm tail: at `p = 6`, `x = 1e−12`
underflows f32 to `0.0` where the true value is `1e−72`.

**★ NOT more accurate — the same correction F18 had to make.** The two arms
disagree on **523 of 6,611 rows (7.911 %)**, and the platform libm is nearer
the 60-digit truth on **520** of them, the port on **3**. The case is
determinism and a bounded error, never accuracy.

**A score era moves no feature slot, and the registry now says so.** Every other
era answers "which slots?" by derivation from the signal table; `scorepow`
answers "none". It is registered in `feature_defs::SCORE_PATH_REVISIONS`, and
`research::era_is_registered` consults **both** registries — otherwise "every
active era token is registered" would have silently broken the moment a score
era joined a revision, which is a registry gap and not a reason to weaken the
assertion.

**⚠ Provenance consequence for anyone flipping rev2**, in addition to
`v1detroot`'s: `scorepow` moves **SCORE bytes by up to a few ULP** and moves
**no stored feature table**, because it touches no feature.
`ZENSIM_POW_FORM=libm` reproduces any pre-era score exactly.

**⛔ BLOCKER on the flip, registered:** `zensim-validate::bake_runtime` (and its
`bake_compare` fork) re-implement the per-sample-α and hybrid head runtimes and
document themselves bit-exact with `metric.rs`. They do **not** follow
`PowForm`, and **no test holds them together** — the claim is prose, and a
prior lane recorded delegation as infeasible. It is true today (both defaults
`LibmPowf`) and false the instant `scorepow` activates, which would make a
VERDICT disagree with the score the product returns. Routing it needs a `pub`
surface on `det_math`, i.e. a public-API change. **One repo further out and not
fixable from here:** `zenpredict::feature_transform`'s `cbrt`/`powf`/`ln`/
`ln_1p`/`sin`/`cos` are on the PRODUCT path via `predict_transformed` and are
LIVE in Profiles A, BHdr and C — **B (the default) and D declare only
`winsor_p99`, a clamp, and are clean.** `zenpredict` already has
`#[cfg(not(feature = "std"))]` twins calling `libm::` explicitly, so the fix
there is to make that the `std` path too.

Record: [`benchmarks/score_path_libc_determinism_2026-09-06.md`](../benchmarks/score_path_libc_determinism_2026-09-06.md).
Gate: `scripts/verify_cross_libc_features.sh` (`just check-cross-libc`).

## §3.53 — the F17 guard on the bake that ACTUALLY SHIPS: READY, and it costs nothing (2026-09-06)

**Ledger ROUND 104.** §3.51 measured the serve-time guard on a **re-fit** of the
D recipe and closed with an explicit honesty note: that lineage reads **f77**
among the twelve `contrast_inc` slots, while `DEFECT_F17`'s exposure table names
**f116 and f155** for the artefact that ships, so *"the guard's effect on the
artefact that actually ships is REGISTERED, NOT RUN."* This is that run.

**Thought-why vs actual-why, the entry that earns its place.** The lane was
framed as "re-measure §3.51's recommended twelve-slot guard on the shipped
support". Measured first, from `zentrain.feature_ids` inside the shipped bake,
the twelve-slot scope is **not a scope at all here**: the shipped bake declares
28 ids and exactly **{f116, f155}** of the twelve, the other ten carry
exactly-zero weight, and after `densify` the twelve-slot arm and a two-slot arm
are **BYTE-IDENTICAL** (`c2db9e29…` both). So §3.51's headline
recommendation, transplanted, silently *is* the two-slot guard — and the two
lineages share **no F17 slot at all** (f77 vs f116/f155), which is why §3.51's
numbers could never have stood in for these. The thought-why was "does the
recommended scope transfer"; the actual-why is "the recommended scope has no
meaning on this support, and the question is whether guarding f116/f155 is
free."

**It is free, and that is the whole result.** Against the shipped bake, on the
runtime-era postC root with the FLOOR-DENSE ladder instruments: paired-bootstrap
CID22 an **exact tie with CI `[+0.00000, +0.00000]`**, every one of the twelve
board corpora inside `8.2e-5`, contract **6/6**, **every per-codec `A7r` floor
exactly equal** on all five codecs, identity **100.000000** with 0 of 9,593
cells above it, inversions **0.0053** unchanged, and the ZCTH-tree-head
corruption gate `pass_q10 = pass_q20 = 1.0000` unchanged. It perturbs **0 of
4,292 CID22 rows**, 0 KonJND, 0 AIC-3, and 262 of 196,086 training rows.

**Two facts recovered from artefacts rather than from docs.** (1) The shipped
4,222 B wide bake **IS** `fit-lasso`'s direct output (sha `921a8f67…` equals
`did100-2026-09-04/bakes/d_id100_negrich_raw.bin`), so **`extend-top` is NOT in
the D lineage** — a chain copied from §3.51 verbatim would have inserted a step
the shipped artefact never had. (2) `train_corruption_head.py::can_bake` returns
`name in ("logistic", "hgb")` — widened 2026-09-06, `hgb` emitting **ZCTH v1**
per §3.49 — so `rev2_refit_2026-09-06.md` §8.6b's *"admits only `logistic`"* is
**stale**: the tree head IS servable and `bake_verdict` loads it.

**What the guard does NOT buy.** It is a rank NO-OP, not a win, and it is a
small **LOSS** on the outlier ordering: −0.00987 on the 17 LIVE rows it actually
fires on (n = 17; two of those already sit at the dial's bottom extrapolation
floor −213.14861 and cannot move). The case for it is a bounded input at a
measured cost of zero — it must not be sold as a ranking improvement.

**Era confound, stated not measured away:** the windows are fitted on the
shipped bake's own training leg (`canonical-2026-05-21/train/safesyn.parquet`,
the source of the gram its repro names) while the eval root is post-option-C.
Windows and clamped rows are one extraction era apart, and this is not fixable
today — the postC root is an EVAL root and carries no safesyn leg.

**Staged, NOT installed.** `d_sdr_add156_id100_negrich_guard12_2026-09-06.bin`
(1,523 B, sha256
`c2db9e29ac97ad6f01f844628500b670ecab323d972169128d3b24daaf044aa8`) at
`/mnt/v/output/zensim/rev2-d-arms-2026-09-06/guard/shipped/` with a
`_MANIFEST.json`. `ZensimProfile::D`, `zensim/weights/`,
`ssim_form::SHIPPED_REVISION` and both registries are untouched. **The install
recipe forces a `dense_bake_flip_gate` change** — that gate asserts `to_bits()`
equality between D and a twin over the retired WIDE bytes, and the candidate is
deliberately unequal on a clamped input — so it must be re-pointed at the
guarded wide bake, never weakened. Board cells `D_shipped@dguard2` and
`D_guard12_p999@dguard2` (both SHIPPABLE, G-ADDR grafted).

Record: [`../benchmarks/rev2_d_arms_2026-09-06.md`](../benchmarks/rev2_d_arms_2026-09-06.md) §12.
Pre-registration: [`PLAN_FEATURE_REV2_2026-09-05.md`](PLAN_FEATURE_REV2_2026-09-05.md) §12.8
— **and its §12.8.0 order-honesty note**: that text was pushed after the arm
bakes and their rank/G-ADDR/perturbation/outlier tables existed, with the four
criteria copied verbatim from §12.7.1.

---

## §3.54 — the SCORE path's fork is gone; NO stored number moved, and the one divergence it uncovered was latent (2026-09-06)

**Ledger ROUND 105.** §3.52 (F19) landed the score-path owner in `zensim::metric`
and **registered, without fixing, the second copy** in `zensim-validate`'s
bake-evaluation runtime — calling it *"a **BLOCKER** on flipping
`SHIPPED_REVISION`"*. This closes it. **Status: STANDS. Nothing in this entry
invalidates a table, a bake, or a published verdict** — the point of recording it
here is the opposite: to say precisely which numbers were checked and found
unmoved, so a future session does not re-audit them.

**The fork, observed rather than argued.** `bake_verdict --full-json` on six
shipped/board bakes × `cid22,kadid,tid,konjnd,aic3`, each run twice:

| arm | A | board `v47_strict_qat_native` | B | BHdr | D | board `b_sdr…dense_dial` |
|---|---|---|---|---|---|---|
| **before**, `libm` vs `pure` | identical | identical | identical | identical | identical | identical |
| **after**, `libm` vs `pure` | **differs, 430 leaves** | **differs, 430 leaves** | identical | identical | identical | identical |
| **before vs after**, default arm | **byte-identical** | **byte-identical** | **byte-identical** | **byte-identical** | **byte-identical** | **byte-identical** |

Row 1 is the defect: the evaluation tooling was insensitive to the form the
product runtime obeys, so a `SHIPPED_REVISION` flip would have made every
published verdict describe arithmetic the product does not run. Row 3 is the
safety proof: **no number moved.**

**Row 2's four "identical" cells are correct and were measured, not assumed.**
`psa=false hyb=false pin=None` for B / BHdr / D / `board_Bwide`, so their whole
score path is `out[0]` → PCHIP spline, and the PCHIP basis is `powi` only — a
multiply chain that never reaches libm. Only Profile A and its wide twin carry
`zentrain.per_sample_alpha_head` **and** `zentrain.tanh_output_head`
(`scale = 30.0`), and only they can move.

**★ A fact that will otherwise be re-derived: A's HEAD is form-invariant; its
PIN is not.** On all 10,000 fixture rows A's per-sample-α head gives
bit-identical output under both `PowForm` arms, even though `|h|^6` disagrees on
**9.80 %** of random doubles, `x^(1/6)` on **14.07 %**, and `exp` on **9.74 %**
(1e6 samples each). Mechanism: A's hidden vector reaches **±2.6e4**, so
`alpha_logit` saturates the ±20 clamp, `α` is 1.0 to f64 resolution, and the
entire `y_pool` term — the only place the p-norm enters — is multiplied by
`(1 − α) ≈ 2e-9` and annihilated. What moves A's score is the **tanh pin's
`exp`**. Corollary: a head with `p_norm = 2` is form-invariant too, because
`x^2` and `x^0.5` are libm special cases (0 / 1,000,000 samples differ). So
"which bakes can this era move" is a property of the head's saturation and its
p, not of the bake's size or corpus.

**The one divergence uncovered, and why it is LATENT.** The validate-side PCHIP
evaluator capped its upper *extrapolation* at 100 (the 2026-07-04 audit) and left
the *interior* uncapped; `zensim::metric` caps both. The reachable trigger is a
knot whose `y` exceeds 100 — the wire format permits one and `parse_payload` does
not reject it. It is **not** Hermite overshoot: a draft gate tried to build an
overshoot fixture and failed its own vacuity guard, because the Fritsch–Carlson
rule keeps the interpolant inside its bracketing knots (now pinned in-tree).
**MEASURED over all 49 bakes on disk** (`zensim/weights`, its `archive/`,
`zensim-experimental/weights`): **0 declare a knot above 100.** No verdict ever
took that branch.

**Recorded and deliberately NOT changed** (it is not an owner divergence — both
implementations do it): the lower branch's `floor = ys[0] − (ys[n−1] − ys[0])` is
a floor only for an *increasing* spline. On a decreasing one it lands above
`ys[0]` and the `.max` makes it a hard value — seven `zensim-experimental` bakes
(`v_balanced_v2`, `v_balanced_v3`, `v_balanced_v3_per_codec`,
`v_compression_v2`, `v_compression_v3`, `v_compression_v3_per_codec`,
`zensim_b_phone_oled`) return exactly **200.0** at `x == xs[0]`, and
`v02_372feat_cell5` returns **188.05**. No shipped profile has a decreasing
spline, so this reaches no product number; changing it would move one.

**Still open, unchanged by this lane:** `zenpredict::feature_transform`'s
`cbrt`/`powf`/`ln`/`ln_1p` are on the PRODUCT path via `predict_transformed` and
live in Profiles A, BHdr and C (B — the default — and D are clean). It is in the
`zenanalyze` sibling repo, which this lane must not edit. It remains a blocker on
flipping `SHIPPED_REVISION`.

Record: [`benchmarks/score_owner_consolidation_2026-09-06.md`](../benchmarks/score_owner_consolidation_2026-09-06.md).
Gates: `zensim-validate/tests/{score_owner_parity,no_score_path_libm}.rs`
(both mutation-verified), `zensim::score_math`'s 9 unit tests.

---

## §3.55 — the dense-id gather was FEATURE-GATED, so four shipped profiles were silently mis-scored in every non-default build (2026-09-06)

**Ledger ROUND 106.** §3.53's cruft-purge increment 2A (`cb2f412d`) flipped `A`,
`B`, `BHdr` and `D` to the dense contract and proved, three ways, that **no
shipped score moved**. It did — on a DEFAULT build. This entry records the
configuration that proof did not cover, and closes it. **Status: FIXED. No
stored table, bake, board row or published verdict is invalidated** — every
number this repo has ever recorded came from a default-feature build, which was
always on the correct side.

**The defect.** Serving a dense bake means GATHERING its declared ids out of the
walk's identity-laid-out vector. The gather, and `feature_layout` behind it, was
`#[cfg(feature = "feature-regime-v2")]`. That feature is default-on — so
`cargo add zensim` was correct and every measurement in this ledger is safe —
but a consumer setting `default-features = false` fell through to
`prep_bake_input_f32`'s POSITIONAL PREFIX branch: `B` reads ids `f3..f369` and
was served positions `0..94`. Plausible numbers, wrong features, no error.

**MEASURED** (`zensim/examples/serving_matrix.rs`; reference = default, arm =
every default feature except `feature-regime-v2`, so tier and threading are
held constant). All 48 `A`/`B`/`BHdr`/`D` cells were wrong:

| profile | min \|Δ\| | max \|Δ\| | worst cell | default | without the gather |
|---|--:|--:|---|--:|--:|
| `A` | 2.4921 | 7.6714 | 256×256 checker_lsb | 94.114457 | 86.443061 |
| **`B`** (`codec_target()`) | 2.2580 | 34.6773 | 48×40 blur3 | 48.171502 | **13.494210** |
| `BHdr` | 3.3364 | 13.5977 | 127×93 checker_lsb | 95.117667 | 81.519927 |
| **`D`** | 4.5262 | **261.5804** | 576×96 blur3 | 48.431759 | **−213.148613** |
| `C` / `CHdr` | — | — | — | — | `ModelForwardFailed`, all 12 |

The instrument's own noise floor, same instrument, gather live in both arms,
`avx512`+`threads` against neither: **1.048e−5** over 24 cells. So the defect is
five orders of magnitude above the widest legitimate cross-tier movement.

**The fix is an UNGATE, and it is free.** `feature_layout` depends only on
`feature_set_id`, `feature_defs` and `mlp::Model` — none v2-gated — and every
shipped dense bake's highest declared id is < 372, which the legacy v1 buffered
walk already emits. So the gather works against the walk `--no-default-features`
already had. Same-parent A/B (a second workspace at `fc47b08e` with its own
target dir, best of 3 warm incremental rebuilds): `--no-default-features`
`.rlib` **3,138,964 → 3,273,736 B (+4.29 %)** and **2.60 → 2.74 s**; default
`.rlib` **7,492,468 → 7,496,008 B (+0.05 %)**, 5.85 → 5.89 s. The non-default
figure exceeds the module's own size because ungating it makes `feature_defs`'s
registry reachable in a build where nothing referenced it; an `.rlib` carries
metadata, so it upper-bounds shipped code growth rather than measuring it.
`candidate-profiles` now requires
`feature-regime-v2` (only the v2 walk emits the 944 `C`/`CHdr` declare), so
**every profile a build can name it can serve** — zero refusals in every
configuration.

**The generalizable lesson, and it is the reason this is in the ledger.** *A
gate that is `#[cfg]`-gated on the same feature as the code it protects is not a
gate.* The servability census — `every_shipped_profile_is_servable`, the
instrument whose entire purpose is this question — lived inside the
`feature-regime-v2`-gated `feature_plan`, so it was blind in exactly the builds
that were broken. It has moved to the always-compiled `zensim/src/serving.rs`.
The same shape produced a SECOND, independent instance in the same lane:
`zensim/Cargo.toml`'s `include` allowlist is only exercised by `cargo package`,
which no workspace build runs, and MEASURED with `cargo package --list` it was
missing **six** `include_bytes!` targets — a published 0.3.0 would not have
compiled for anyone. Both halves are now checked from a configuration that can
fail: a pinned score in the library's own unit tests (runs under every feature
permutation CI builds), and an allowlist check that reads the manifest via
`include_str!`.

Record: [`../benchmarks/dense_serving_ungate_2026-09-06.md`](../benchmarks/dense_serving_ungate_2026-09-06.md).
Gates: `zensim::serving`'s four tests, `scripts/serving_matrix.sh` (2
environments × 5-6 arms, with a vacuity guard so at least one arm is genuinely
v2-free), `zensim-wasm-tests::large_image_noise_distortion_matches_the_pinned_score`.

**AMENDMENT (2026-09-06, same day) — the gate's own tolerance was derived on one
architecture, and i686 CI caught it within the hour.** `every_shipped_profile_
scores_its_pinned_value` shipped at `TOL = 1e-2`, derived from an x86-64-only
population (max 1.048e-5). `Test (i686-unknown-linux-gnu)` went red on exactly
one cell — `PreviewV0_1/single-LSB` 98.378873 vs pinned 98.394763, delta
1.589e-2 — reproduced locally bit-for-bit. **Not a mis-serve and not this
entry's doing**: `PreviewV0_1` carries no MLP bake, and all four DENSE profiles
passed on i686. MEASURED over 160 cells (8 profiles x 4 geometries x 5
distortion strengths) on the four arms CI runs, **there are TWO arithmetic
classes, not four**: x86-64 AVX-512 vs AVX2 **2.1411e-5** (0 cells > 1e-2);
x86-64 vs i686 scalar and x86-64 vs wasm32 simd128 both **2.8221e-2** (9 cells);
**i686 vs wasm32 BIT-IDENTICAL on 160/160**. Every fused-`mul_add` backend
(AVX-512, AVX2, NEON) is one class; magetypes' scalar + wasm128 `mul_add` is
unfused and is the other — the same split `e1324192` measured as 1 ULP on the
ring tests, and the same "one shared alternative result set" structure the v1
golden policy already records for features. It becomes 1.6e-2 of SCORE because a
near-identical pair puts the dissimilarity features at the f32 cancellation
floor (193/228 features differ, worst **7.4x relative** at ~1e-6 absolute; raw
distance moves 1.42 %) and `100 - 18*d^0.7` has slope -35.5 there. The
kernel-side fusing fix was tried and rejected in `e1324192`. `TOL` is re-derived
as the geometric midpoint of two populations each measured on the population
that bounds it — noise **2.8221e-2** (160 cells; the gate's own 16 cells top out
at 1.5890e-2) and the smallest real mis-serve **on the gate's own cells**,
**0.945195**, measured directly by bypassing the gather in a local build (all 8
dense-profile cells move, `B`/single-LSB +0.945195 to `D`/quantize+shift
-172.261448, `D` inverting to -155.77) — giving **0.16**, 5.67x above one and
5.91x below the other; `zensim-wasm-tests` takes the same bar (its `1e-2` passed by luck on
its one cell). `scripts/serving_matrix.sh` still demands BIT-EXACT agreement
within one architecture. **Reusable: a cross-architecture score bar for this
metric cannot be tighter than ~3e-2 while that `mul_add` stays unfused; i686 and
wasm32-simd128 are ONE class; and a tolerance derivation must name the
population it was measured on.** Record: `../benchmarks/dense_serving_ungate_2026-09-06.md` §2d.

---

## §3.56 — output polarity had no owner, and it is why the campaign's best CID22 *ordering* arrived negative (2026-09-06)

**What a future session needs from this section:** no stored number moved, but
one CLASS of result is retrospectively explained, and two trainer flags now mean
something they did not before.

### The thought-why

The fastclass2 campaign (`benchmarks/fastclass2_campaign_2026-09-05.md` §11)
measured its 2-layer per-sample-α arm as **the campaign's single best CID22
ordering — `|−0.8921|`, beating the plain path's `+0.8863` — with the sign
backwards**, and could not ship it: `bake_dial_refit pack` reported
`packed tanh-pin range … corr=−0.8350` → `spline fit produced only 1 knots`,
because the output calibration spline is monotone increasing by construction.
The campaign's determination was "a property of the WEIGHTS … the right repair
is the head's output orientation at training time, not a post-hoc negation," and
it stopped there. All three seeds carry `.HARVEST_FAILED`; there is no fulleval
for them on the board.

### The actual-why, read from source and then MEASURED

`mlp_train/mod.rs:2088` has always derived `rank_target_sign` — `−1.0` when any
group carries an absolute/MSE term, else `+1.0` — with a correct comment and a
2026-07-15 measurement behind it (HF held-out per-ref SROCC **+0.6393 →
−0.3454** when rank supervision was *added* to a corpus). **It was used at
exactly one of eight polarity-sensitive sites** (`:2556`). The pool head, the
hybrid head, the per-sample-α head and BOTH plain-path mini-batch helpers each
recomputed a bare `signum(mos_a − mos_b)`; both TV within-ladder hinges assumed
DISTANCE in prose; the α head's monotonicity hinge assumed SCORE — *against its
own RankNet term, on the same path*.

The fastclass 372 recipe declares `:both` on 4 of its 6 legs. So the plain path
is SCORE-shaped and the α head stayed DISTANCE-shaped. Two terms, one output,
opposite directions.

**Reproduced in a 0.5-second unit test** (`tests/output_polarity.rs`): an α-head
recipe carrying an absolute term reads raw SROCC **−0.9970** at depth 1 and
**−0.9986** at depth 2 on a synthetic corpus where the plain path reads +0.99.
At `main@origin 0c6307a7`, **4 of 5 of those tests FAIL**; the fifth
("rank-only recipes are DISTANCE-shaped on every path") passes at both commits,
which is the control.

### What this does and does NOT invalidate

- **No stored bake's bytes move.** `OutputPolarity::Distance` is the default and
  reproduces the legacy convention. `tests/legacy_bake_sha.rs` pins five
  rank-only recipes to sha256 digests measured at the parent commit; all five
  reproduce byte-for-byte, **including both TV arms** (which is what proves the
  newly-reachable `--tv-margin` is the identical function at its `0.0` default). *(Corrected 2026-09-06, `9059b3cb`: those digests are a SAME-CLASS check — trained bakes are not bit-reproducible across SIMD tiers, so CI could never pass a Zen-4 digest on other platforms; `tests/legacy_bake_sha.rs` now asserts the invariant in-process — the polarity owner resolves every legacy recipe to `Distance`, whose multipliers are the IEEE identity, and each recipe retrains byte-identical to itself — while the pinned digests run only under `ZENSIM_ZEN4_GOLDEN_BAKE_SHA` / `just legacy-bake-zen4-golden`, an explicit opt-in, never a silent skip.)*
- **No board number is affected.** The α-head arms this explains have no
  fulleval, and no board bake sets `--minibatch-size > 1` or NiN with an
  absolute term.
- **Two flags now mean something new.** `--minibatch-size > 1` and
  `--norm-in-norm-weight > 0` with an absolute term used to silently DISCARD
  that term; they now refuse. `--n-hidden-layers >= 2` and `--skip-connection`
  used to be silent no-ops off the α-head path; they now refuse. A recipe that
  used either combination did not train what it said it trained.
- **`--leaky-alpha` was a train/serve divergence.** Every emitter hard-coded
  `Activation::LeakyRelu` while the runtime applies a hard-coded
  `zenpredict::LEAKY_RELU_ALPHA`. Any run with a different slope trained one
  function and served another. Now derived and refused when unrepresentable.
  Incidental measurement: the runtime constant is an **f32**, so
  `LEAKY_RELU_ALPHA as f64 = 0.009999999776482582`, not the f64 literal `0.01`
  the trainer defaults to — a ~2.2e-10 relative slope gap every LeakyReLU bake
  has always carried, below every gate's tolerance and worth naming once.

### The architecture this unblocked

`--nonneg-distance` makes the dial's identity (C5) and no-cell-above-identity
(C6) rows STRUCTURAL: `raw(x) = pin − g(x)` with `g ≥ 0` and `g(0⃗) = 0`
bit-exactly, so `raw(0⃗)` is the argmax over the whole input space — **provided
every active feature transform maps 0 to 0.** The canonical 372 screen's 28
positive-`lo` `winsor_p99` guards do not (they map `0 → lo`), and MEASURED on
this wave `raw(identity) = 99.6138` against a pin of 100.0. `raw(x) ≤ pin` ∀x
stays structural; identity being the argmax does not, so C6 = 0 is a measurement
on 9,593 cells across three seeds rather than a theorem. C5 passes regardless,
because the identity ANCHOR rows take the same forward and the spline maps that
raw to exactly 100. It exists because
`benchmarks/dial_addressability_gate_2026-09-04.md` §10.3 proves **no monotone
output spline can satisfy both C2 and C6** when real cells out-rank a perfect
copy in raw space — a weights defect, so it is fixed in the weights. It is
expressible in the SHIPPED wire format with **zero** runtime change, which is
why softplus / ReLU² / squared-norm were rejected. It does **not** claim C3/C4 or
A7r, and `g` is CONVEX at one hidden layer.

Record: [`benchmarks/best_of_all_2026-09-06.md`](../benchmarks/best_of_all_2026-09-06.md);
plan + deviations: [`docs/PLAN_BEST_OF_ALL_2026-09-06.md`](PLAN_BEST_OF_ALL_2026-09-06.md).

### One more owner defect found on the way

`bake_dial_refit densify`'s identity gate fed BOTH arms the pre-densify
caller-width row. A bake whose kept ids are a **contiguous prefix** densifies to
the IDENTITY layout, so `zensim::declared_feature_ids` returns `None` for it by
design and its gather is POSITIONAL — it must be fed its own narrower row.
Measured: `feature row has 372 values, bake expects 228`. Every shipped bake the
subcommand had been run on reads a SCATTERED id set (shipped D:
`6,8,11,14,…,155`), which is why the path had never been exercised. Fixed and
gated with the scattered control beside it.

**And a consequence worth carrying:** a DENSIFIED bake cannot be graded by the
pinned probes. Every registered negtail/identity probe is 372-wide and
`bake_verdict` scores a probe only when its column count equals the bake's
`caller_input_width`, so a densified 228-caller bake reads **C3, C4, C5 and C6
all NOT MEASURED** — the entire contract tier. Score the PACKED bake (372 caller
width via `Drop` transforms) and treat `densify` as a servability artifact.

### The best-of-all wave's answer (appended 2026-09-06)

27 cells, 9 arms, k=3, 0 failures. Record:
[`benchmarks/best_of_all_2026-09-06.md`](../benchmarks/best_of_all_2026-09-06.md).

**The dial contract is achievable in the weights, at a measured price.** C6 goes
**1,642 → 0 on every seed while `tied` goes 0.0017 → 0.0000**, so the gate
record's C2 ⊻ C6 either/or is dissolved rather than traded. Cost: **−0.0091
CID22** pooled, **−0.0052** per-ref — the constraint hurts between-image scale
more than the within-image ordering a dial walks.

**A7r's null is narrower than a loss/architecture sweep suggests, and this is the
part to carry forward.** It is unmoved by architecture, ladder loss and hinge
weight. It **does** respond to floor-reaching training DATA: the isolating A/B
(same architecture, hinge weight and band weights; differing only in whether the
ladder pairs reach the encoders' true lowest settings) moves **4 of 5 codec
floors** — `avif-svt` **+0.188**, `jpeg` +0.051, `jxl` +0.051, `webp` +0.026 —
and the single non-mover is **`avif-rav1e`, the one codec the anchor set contains
no ladders for**, which was pre-registered as a structural limit before the fit.
None reaches its bar. **Direction right, quantity insufficient** at 32 references
× 4 codecs; the named next lever is a larger, **rav1e-inclusive** anchor set.

**Two rulings a future session needs.** `freeze_check --select` **cannot see the
dial contract** — its floor set is
`{bandtail, breadth, cid22, dial, dialrange, hfnl, konjnd, nonphoto}`, so it chose
the CONTROL over four arms that take the contract from 4/6 to 6/6, on a CID22
hair (registry `select-rule-blind-to-dial-contract-2026-09-06`). And
`--nonneg-distance`'s identity pin is **conditional on zero-preserving feature
transforms**; the canonical 372 screen's 28 positive-`lo` winsor guards break it
(`raw(identity) = 99.6138`, not 100.0), so C6 = 0 is a measurement on 9,593 cells
rather than a theorem (registry
`nonneg-identity-pin-conditional-on-transforms-2026-09-06`).

**Nothing shipped.** `zensim/weights/` and `ZensimProfile::D` are untouched; the
ship rule failed on A7r **and** on a CI-clean KonJND regression (−0.06219).


---

## §3.57 — the fair board went stale at 508 fullevals; re-audited and rebuilt, cap fix scoped to two seed-duplicate rows (2026-09-06)

**Ledger ROUND 107.** Board hygiene lane, not a modeling result. The all-rows board
(`summer_gauntlet.html`) was rebuilt 2026-09-06 10:04 at 508 fullevals by a concurrent
lane; `summer_gauntlet_fair.html` was not, so it still reflected the 2026-09-05 19:58
population (97 of 433) and every fulleval promoted since — the 27 `BOA_*` best-of-all
cells, `D_guard12_p999@dguard2`/`D_shipped@dguard2`, and the `fc2_*` fast-class family —
was simply missing from the fair board's file.

**Checked before acting, not assumed**: the brief's framing was that these cells
carried `fair: {}`. They do not — `fair` is never a stored field in a source
`*.fulleval.json` (confirmed: none of the 508 files on disk has that key at all); it is
computed at render time by `gauntlet.fairness_of`, which was already correct for these
cells in the already-rebuilt all-rows board (`BOA_A_plain_s4004` -> `VERIFIED-FAIR,
k=3`). So the fix was a rebuild of the fair-only board, not a code or classification
change. Cross-checking the fresh 508-row audit against the *immediate* predecessor
snapshot (481 rows, generated 02:39 the same day) rather than the much older
2026-09-04 baseline (479 rows on disk, itself already grown past what its own
`.md` documents — flagged, not touched) showed the genuinely-new set was smaller than
the brief assumed: only the 27 `BOA_*` cells were new since 02:39; `D_guard12_p999@dguard2`
/ `D_shipped@dguard2` and all 12 `fc2_*` cells were already audited, just unrendered.

**Re-issued audit**: `fairness_tiers_2026-09-06.tsv` (508 rows, same mechanism as
`fair_gauntlet_2026-09-04.md` §2, nothing re-derived) —
`benchmarks/fairness_tiers_2026-09-06.pointer.md`. `fairness_tiers_2026-09-04.tsv` left
untouched (sha256 and mtime both verified unchanged before and after this pass).

**The cap fix, scoped as narrowly as the data allowed.** The first fair-only rebuild
landed at 12,655,576 B — 73 KB over the 12 MiB (12,582,912 B) cap. Per-cell scatter
measurement (not guesswork) found the entire overage traced to two **curated** rows
that are seed-duplicates of an already-fully-charted row in the same k-group:
`LSTAR_s4022_packed` (k=7 sibling of `LSTAR_s4021_packed`, same `seed_group`
`9facbc8a2223`) and `w11_s4014_e050` (an intermediate checkpoint of the exact same
seed 4014 as `w11_s4014_final` — the doc's own documented
`duplicate-promotion-same-seed-2026-09-04` pattern, "one training run promoted
twice"). Stripped their embedded `per_pair` at the source (mirroring
`promote_fulleval.py --strip-per-pair`'s own field names, `per_pair_stripped` +
a note), landing the rebuild at 12,419,271 B (11.85 MiB). Both cells keep every
scalar stat and their scoreboard row; neither tier nor k changed. **Nothing was
dropped silently**: the complete pre-strip JSON for both is preserved verbatim at
`fulleval/_perpair_full_backup/<name>.fulleval.full_backup.json` (these two fullevals
have no `source_verdict` to fall back on — they were not produced via
`promote_fulleval.py` — so a physical backup copy was made rather than relying on
that mechanism; their `bake` files also still exist on disk as a second, independent
recovery path via re-running `bake_verdict`).

**Verified, not asserted**: `scripts/v_next/gauntlet_gates.sh` PASS/rc=0 on both
rebuilt boards (fair: 171 of 508, all compare-fragment gates 4a-4e pass; all-rows:
491 of 508 after the unchanged 17-cell coverage-gate exclusion). `curl` 200 on the
served fair board via both the LAN gallery URL and localhost, byte-for-byte matching
the file on disk. A combined `#compare=BOA_F_nonneg32_s4004,D_guard12_p999@dguard2,
fc2_372_S228_H128_s4004` render check resolves all three with no missing-id banner
(no literal `S372_S228_H128_p` cell exists; substituted the fast-class family member
named above, noted rather than silently swapped).

**Nothing modeled, nothing shipped, no registry entry added.** This section exists to
record the board-hygiene mechanics and the one factual correction to the brief (the
already-classified cells), per the standing rule that a durable finding goes in the
ledger in the same pass it's made rather than only in chat.

---

## §3.50 — the board's operative dial ruler becomes the LADDER instrument (2026-09-06)

**Ledger ROUND 102.** Two lanes hit the same wall on the same day: the board's
`dial` / `dial.addressability` column is the **2026-05-29 CANONICAL** grid (4,424
cells, 106 ladders, one AVIF backend, jpeg's three lowest "settings" byte-identical),
while every current gate — per-codec floor representability under the operative
`resolvable` rule (`A7r`), the two-reference inversions, the contract rows on true
floors — is defined on the **2026-09-05 FLOOR-DENSE LADDER**
(`benchmarks/ladder_instrument_2026-09-05.md`). `promote_fulleval.py --graft-gaddr`
**correctly refused** to merge ladder readings into the canonical column (two
instruments in one column = silently wrong cross-cell comparisons), so the ladder
readings sat in a sidecar the board could not see and `freeze_check --select`'s new
CONTRACT veto could not act on them.

**Resolved by giving the ladder reading its OWN block, not by relaxing the refusal.**
`dial_ladder` + `dial_ladder_source`, written by `--graft-gaddr-ladder` whose gate is
the MIRROR of `--graft-gaddr`'s: where that proves *"this read was taken on the
board's own grid"*, this proves *"on a REGISTERED ladder grid"* (resolved from
`benchmarks/dial_addressability_floor_2026-09-04.json` at run time). **`dial` is not
in its write allow-list and 0 of 508 `dial` blocks changed.**

**Coverage: 450 of 508 cells graded, 58 NOT MEASURED with a recorded reason each.**
The 2026-09-04 as-run replay covered 97; `scripts/gaddr_board_ladder.py` reconstructs
an invocation for every cell from the fulleval's own fields when no log exists.

**The era gate is MEASURED, and it is the load-bearing decision.** Populated-slot
signatures: the ladder-944 grid and `dial_grid_944col_POOLS_2026-08-30` populate
**905** slots with `f0..719` LIVE (slot-set sha8 **`b6811ae0`**); `bake_verdict`'s
DEFAULT 944 grid (`dial_grid_944col_2026-08-01`) and `era2-rank`'s `r4` populate
**689** with **`f156..371` STRUCTURALLY ZERO** (**`026c0aba`**). A bake trained where
that block is always zero receives **zero gradient** on those 216 weights, so they
keep their INIT values, and a grid that populates them multiplies live data by
untrained weights — the `--regime 944` silent-mis-scoring bug class. So a 944 cell is
graded only when its own instrument is already pools-era **or**
`bake_block_profile`'s `uses_f156_371` is false. **359 of 381 admitted 944 cells are
provably immune** (weights exactly zero — reproducing this repo's "944 MLPs zero them
exactly, 216/216" on a 381-cell population), 22 are pools-era, **34 are refused**.

**Probe recovery is PROVEN, not guessed.** `measured.negtail` is a function of
`(bake, probe)` ONLY and never touches the dial grid, so the correct probe is the one
whose read reproduces the board's block **bit-for-bit** — verified identical on
`W10L9PH_s4007_packed` (`frac_below_zero` 0.0265, `min` −69.61934236224508). 14
recovered, 0 unrecovered. Two traps: **`measured.identity` mixes probe reads with GRID
counts** (`n_grid_cells_total` 4,817 on POOLS vs 9,593 on the ladder) so it can never
match across instruments — only its probe-scoped fields are compared; and **this build
emits C5/C6 as measured CHECKS while serialising no `measured.identity` dict**, so
"the identity probe was accepted" is read from the check states.

**WHAT FLIPPED — and the decomposition is the whole finding.** The NOT-SHIPPABLE badge
goes **63 → 75**, against a pre-registered expectation of a small DECREASE. Decomposed:
**21 of the +22 are NEWLY-GRADED cells that had no dial ruler at all** (171 of 508 cells
carried a G-ADDR block before, 450 do now); there is exactly **ONE true up-flip**
(`A2b_l0.002`, C1) against **TEN true down-flips** — a ruler effect of **−9**. On the 93
cells shared with the 2026-09-04 as-run set the contract-fail count goes **46 → 42**,
reproducing gate doc §17.7's 47 → 43 on a different code path. Every one of the 21 new
badges is on a LEGACY/FAIR-NOTED cell; every one of the 10 lost is VERIFIED-FAIR.

**The switch is DIRECTIONAL PER ROW, not uniformly stricter** — the most misreadable
fact here. **C1** (`mono`) gets **harder** (the ladder samples q 0..30 at step 1, so far
more near-flat adjacent rungs: `A2b_l0.002` 0.97852 → 0.91712 against 0.93); **C2**
(*flat/clamp dead-zone fraction*) gets **easier** (`W10L9PH_s4007_packed`
0.09081 → 0.03060 against 0.05). **Corrected 2026-09-07 against code and the
saved verdict:** these are `measured.grid.tied` fractions, affected by the
sampling population. The earlier “no cell out-scores a perfect copy” label
belongs to C6, which counts violations against a zero bar. That cell's C5/C6
remain NOT MEASURED on both instruments; no identity improvement was measured.

**`freeze_check --select`'s pick does NOT move** — `11e243eb0b86`
(`fc2_372_S228_H128_s4004/5/6`) under `--gaddr-block canonical`, `auto` and `ladder`
alike over the 125 VERIFIED-FAIR cells, with `canonical` **byte-identical to a pristine
`main@origin` binary** built from the parent revision's own source. But **10 fair cells
go `NO — CONTRACT FAIL` → `yes`** because their C2 genuinely passes on the operative
instrument: **the CONTRACT veto's removal-only property holds *within* a ruler, not
*across* one.**

**⚠ THE BOARD'S A7r FRACTIONS ARE `resolvable`, NOT the `distinct` ones the ladder
record's §9–§9.4 tables carry.** Operative mentor bars `avif-rav1e` **0.6410** /
`avif-svt` 1.0000 / `jpeg` **0.6667** / `jxl` 0.9615 / `webp` 1.0000 against
`distinct`'s 0.5385 / 1.0000 / 0.5385 / 0.9231 / 1.0000 — and the difference flips a
verdict: under `distinct` shipped Profile D fails jpeg by one ladder (0.5128); **under
the operative rule the ADD156/D lineage passes all five codecs** (0.6667 / 1.0000 /
0.6667 / 1.0000 / 1.0000) and is the only bake family on the board that does.
`peer_ssim2` is the only other five-codec pass; `peer_butteraugli` fails four of five,
a second independent argument against it as a mentor. Board-wide **10 of 450 clear all
five and 293 clear none**; every best-of-all and fast-class arm — including the
registered `--select` winner — collapses on `avif-rav1e` (0.13–0.21 against 0.641).

**Registry (append-only, +3):** `dial-addressability-canonical-instrument-2026-09-06`
(every pre-switch dial number is scoped to the canonical instrument — not wrong, not
rewritten, a retired-era ruler), `dial-ladder-not-measured-off-instrument-2026-09-06`
(the 58 cells, absent-not-failed, with their reasons),
`a7r-floor-rule-operative-resolvable-not-distinct-2026-09-06`.

**Board state:** fair 171 of 508 cells at **12,528,928 B (11.9485 MiB)** — margin under
the registered 12 MiB cap fell from ~164 KB to **53,984 B** because 279 cells gained a
first-ever G-ADDR block; all-rows 508 of 508 at 26,167,081 B. `gauntlet_gates.sh` rc=0
on both; both served 200; the named compare fragment
`#compare=d_id100_negrich@did100lane,BOA_F_nonneg32_s4004,D_guard12_p999@dguard2,peer_ssim2`
resolves 4/4 with no missing-id banner.

Record: `benchmarks/board_ladder_ruler_2026-09-06.md`; plan:
`docs/PLAN_BOARD_LADDER_RULER_2026-09-06.md`; gate doc §19; CLAUDE.md G-ADDR +
dashboard paragraphs corrected in place.


## September 7, 2026 — cleanup reproduction and rav1e floor-data control

**STANDS, bounded historical replay.** Three A_plain and three H_anchorlad
seeds exactly reproduce their September 6 verdict and ladder panels through
the cleaned Rust training/packing/surface path. The documented KonJND/KADID
source overlap remains an integrity/memorization guard, not a held-out result.
A_plain CID22 held-out SROCC is 0.889095 ± 0.002369 across seeds.

The preregistered eight-source rav1e control produced 280/280 encodes and
extractions, 272 nonidentity training rows, and 2,010 added TV pairs. The
4,520-row/four-codec and 204,746-pair old prefixes remain exact; combined
rows/pairs are 4,792/206,756. Source origin splits are train, original bytes
are preserved, and nearest CID22 gold dHash distances are 20–24. Normalization
uses unclamped SSIMULACRA2/100 and `ref_basename = image_id`.

Artifacts: `/mnt/v/output/zensim/cleanup-floor-control-2026-09-07/`.
`instruments/rav1e_train_norm.parquet` SHA-256
`d28cde382d02c5d0d7b227cbc6bdf1f811b6f0c62fa2212acbcfe6ecfe012a6f`;
combined `ladder_anchor_train_norm.parquet`
`0b2c11ad3efb534da98d8bafa71f2f9b6aed323deea499fc0c819d8e7f98a492`;
combined TV pairs `f92d47281c7f354cfb93053041da3f561e17c3e7beffa0b056a45f3e0041bb25`.
ENCODING_INPUTS, TRAINING_INPUTS, CONTROL_MANIFEST and INPUTS_AFTER pin the
actual sources, binaries, tables and pairing. The extractor's build commit
is unknown: its SHA does not establish a qualified new decoder/feature era.
The pre-launch A_plain manifest omitted H-only TV files; actual trainer input
hashes and byte-exact historical TV reconstruction provide the additional check.

Paired H versus H+rav1e CID22 delta is +0.000894 ± 0.002350 (sample SD, three
seeds). Both pass identity/negative-tail contract; all six fail all five codec
floor gates. Eight-source missing supervision is insufficient; no new model is
qualified and no loss/head/feature removal is justified. See the complete
[control record](../benchmarks/cleanup_scientific_controls_2026-09-07.md) and
[actual target-loop record](../benchmarks/cleanup_target_loop_2026-09-07.md).

## 2026-09-08 — bounds and train-calibrated steering replace the unbounded screen

**RE-FRAMED:** the September 7 fixed-target midpoint runs mixed attainable and
unwitnessed requests. Their legacy median-error screen cannot establish model
failure. User ruling: bound each image/codec first, train codec-specific seeds
only on imazen/imazen-26 training sources, compare 1/2/3 shots, and include JXL,
JPEG and WebP diffmaps before deeper AVIF expansion.

The new Rust instrument has 12 calibration and 8 validation origins, separated
by both canonical digit and family rules, with verified 256-longest-side
`cleanpicker-ladder11@2026-08-23` bytes. Corpus authority is the separate
imazen/imazen-26 repository; its August 27 family extension and August 30 URL
correction postdate the old codec-corpus guidance. These steering measurements
do not train a new metric, change historical splits, or erase old metric-model
sharing/provenance defects.

**STANDS, bounded measurement:** 2,730 actual scalar steering cells, 162 with
negative targets, run only where a bound-ladder output witnesses feasibility.
Of 360 fixed requests, 95 have witnesses, 151 lie outside the measured envelope
and 114 remain unwitnessed inside it. Unwitnessed is not synonymous with
impossible. Training calibration improves tail errors and ±1 hit rates at three
passes on all three codecs for B, while WebP's median gets worse. See the
[full record](../benchmarks/target_steering_bounds_2026-09-08.md).

Native diffmap qualification remains separate: existing JXL/JPEG/WebP/AVIF
owners and research branches were found, with stale scorer pins, conditional
map engagement and different pass accounting. The complete composed-model map
surface remains work; no native-map gain or product tolerance is established.

## 2026-09-08 — canonical native corruption input packet

**STANDS, generation complete; training admission incomplete.** The September 6
nonlinear HGB and Rust ZCTH results supersede the September 5 linear-head
separability/guard proposal. Historical detection does not certify the old
inspo-tree source split or foreign JPEG anchors.

The existing native fixture owner and Python corpus owner now generate a
separate canonical JSON-manifest packet. Generator math is pinned to
codec-corpus `8e10d4d7` (the old local `3e7a8a22` omitted later real bugs), with
zenpng IO and current zenjpeg q10/q20 4:2:0 anchors. The same canonical 12 train /
8 validation families yield 8,580 / 5,720 complete 372-feature rows, formula 1,
libm root form. All source/producer hashes and original pixels are retained.

Inert attempts are noncorrupt labels, with catalog identity preserved. The
raw tables retain 823 / 345 repeated source/pixel pairs and deliberately remain
**failed C10**; they are not approved training views. A separate 456 / 304-row
honest JXL/AVIF scalar-bound supplement re-decodes retained native bitstreams
with the same extractor. No new classifier fit or terminal-label evaluation.

See [registration](CANONICAL_CORRUPTION_2026-09-08.md) and
[complete input record](../benchmarks/canonical_corruption_2026-09-08.md) for
hashes, controls, original/final producer snapshots and remaining deduplication,
T0 content audit, current surface-feature parity and fit/calibration partition
work. Artifact root: `/mnt/v/output/zensim/canonical-corruption-2026-09-08/`.

## 2026-09-08 — canonical corruption public-surface parity and frozen-head screen

**STANDS, bounded development screen; no new fit.** All 15,060 canonical and
native honest rows now have exact complete Rust pixel/cache/stored-f32 score
agreement and exact original-table f32 feature parity. The first prototype's
678 identity disagreements exposed omitted cached identity context: pixel100
was correct, and raw zero features cannot prove identity. The new explicit
identity-aware cache surface preserves supported pixel and raw-feature behavior.

Deduplicated validation detection is 5,022/5,353, and strict below-q20 ordering
improves from 1,945 to 5,280/5,353. However eight of 304 honest native codec
outputs—all near-lossless JXL—are incorrectly lowered from 97.77–98.54 to zero.
Newer `real_bug` detection is 204/317. The frozen historical D+HGB is therefore
not shippable; these inputs do not establish independent held-out validation of
its old training recipe. T0 content admission, approved deduplicated training
views and exact fit/calibration identities remain incomplete. See the
[serving record](../benchmarks/canonical_corruption_serving_2026-09-08.md).

## 2026-09-08 — native source admission and canonical fit preparation

**STANDS, preparation only.** Twelve training origins have zero strict dHash
flags against 182 SDR reference entries, using a named native PNG/resample era;
all four distance-14..16 matches were reviewed as distinct content. LIVE's 29
BMP references now decode through the shared native owner with exact RGB/PNG
reproduction. Thirty UPIQ HDR EXR references remain pending; no exception to
the native-only rule or full content admission has been assumed.

The explicit fit/calibration/evaluation roles contain 5,504 / 2,709 / 5,679
source/pixel-unique rows. A single feature plus a binary label is an ambiguous
duplicate surrogate: the full-feature collision check finds no label conflicts.
The Parquet owner now supports a registered full-record key contract requiring
zero duplicates; the old sampled default and raw failures remain intact.
The trainer's canonical mode evaluates its exact exported single fit through
Rust and refuses fitting without complete admission. Numeric-fixture export
parity passes, but **no canonical image-data candidate has been fit**. See the
[preparation record](../benchmarks/canonical_corruption_refit_2026-09-08.md).


## 2026-09-08 — native PNG and coarse JXL intervention evidence

**STANDS, training-family mechanism screen only.** The same four canonical
training origins and their existing 256/512 variants are used without a fit
or new corpus split. The JXL instrument now has native PNG IO; all 272 previous
transform outputs reproduce byte/pixel/q/score/map-exactly. Coarse whole-transform
unions with ±20% raw-q changes yield expected D direction in 244/256 and 245/256
probes, but gain-per-byte prediction remains uneven. Area and amplitude change
together; no matched-RD or qualification claim. Source/PNG hashes, whole-transform
coverage, complete independent judges, final-code reproduction and negative
controls are retained in the [screen record](../benchmarks/zensim_coarse_interventions_2026-09-08.md).
The separate canonical corruption refit remains unfitted pending EXR admission.


## 2026-09-08 — actual coarse JXL allocation policy screen

**STANDS, failed training-family policy screen; no fit or qualification.** The
same four canonical families and existing 256/512 PNG variants feed a fixed
coarse-density policy. Before its map is applied, every distinct local globally
rescaled integer raw field in [2/3,3/2] is encoded and scored. This does not
exhaust other codec controls. All eight 256 cells improve D at budget, but
Butteraugli/SSIM2 noninferiority fails; 512 engineering coverage has further
losses. No separate validation or terminal family participates. The 249-encode
primary and all 498 judge values reproduce; 36 analysis and 11 CLI refusal
controls pass. [Failed policy record](../benchmarks/zensim_coarse_allocation_2026-09-08.md).


## 2026-09-08 — existing HDR PNG development inputs, user correction

The canonical imazen-26 `variant/png-v3` branch has all 76 HDR PNG companions;
all 76 local payloads match branch LFS SHA-256 OIDs and sizes. The family split
is 38 train / 20 validation / 18 terminal. The active August 23 15-scale set
(1,140 files) is also local; it retains June's pixels and membership and inherits
the current family split. No fitting, scoring or terminal pixel read occurred.
This is the HDR development source path. UPIQ's separate 30 EXRs serve protected
holdout-content admission and must not be described as needed to obtain HDR
training images. [Binding, exact revisions and chronology](TARGET_STEERING_PROTOCOL_2026-09-08.md#existing-hdr-png-inputs--user-correction-september-8).


## 2026-09-08 — EXR ownership correction, after the HDR PNG binding

The user explicitly selected `zenextras/zenexr` wrapping the Rust `exr` crate.
This resolves the earlier exception question and supersedes the unpushed custom
zenbitmaps EXR reader, whose verified source checkpoint remains archived.
The replacement preserves all 124,609,944 f32 samples across 98 saved independent
fixtures and 30 UPIQ HDR reference outputs. Source windows, chromaticities,
luminance metadata and associated alpha remain explicit.
[Decoder contract and parity record](../../zenextras/benchmarks/zenexr_validation_2026-09-08.md).
No human scores or distorted holdout images were read, no image-data model was
fit, and no content-admission decision follows from pixel parity. The remaining
reference fingerprinting/contextual audit still governs corruption-head fitting.

## 2026-09-08 — reference admission completed; first head fit hits serving guard

The remaining 12-source × 30-UPIQ-HDR screen completes through zenexr with zero
strict flags. Two looser matches are visually distinct paper/photograph pairs.
The prior 182-SDR-reference audit reproduces exactly. Full declared coverage
and its reviews are bound in `CONTENT_ADMISSION.json`; no source is excluded.
The existing source-family split, crop-blind limitation and protected labels
remain unchanged. HDR development still uses the existing imazen-26 PNGs.

The registered seed 4101 trains on 5,504 fit rows and calibrates on 2,709
separate-origin rows. The exported head matches Python in Rust on all 13,892
prepared rows (including the 5,679 separate evaluation rows), but the full
pixel audit correctly rejects unsupported companion feature coverage in D's
fast extraction plan. No pixel scorecard or model qualification is produced;
the next two seeds are not run. [Evidence and next serving task](../benchmarks/canonical_corruption_refit_2026-09-08.md).

## 2026-09-08 — later D-regime corruption refit and honest-cost screen

The user requests D's existing extraction regime instead of extending it for
the all-372 head. Canonical source tables and the completed admission receipt
are unchanged. The trainer explicitly selects f0..227 for the head while
preserving the full 372-column source contract and original source roles.

Three seeds serve through Rust on all 15,060 original attempts. Array export
parity and composed pixel/cache/stored-f32 scores pass; one honest row has a
non-firing confidence difference after f32 rounding. Validation detects all
tested non-inert RGB swaps and 5,330/5,353 corruptions, but lowers 20/304 honest
native outputs (22/326 honest rows overall). Strict below-q20 ordering is
4,877/5,353. The fixed screen fails; no model or default is promoted.

A preregistered follow-up changes honest fit weights by 4/16/64 and evaluates
only the twelve training origins, including the four separate calibration
origins. Cost 4 hits a pixel/stored-f32 fire disagreement; costs 16 and 64 fail
calibration protection/ordering bars across three seeds. No cost arm advances
to validation. No threshold is retuned or admission rule relaxed.
[Recipe, chronology, hashes and complete results](../benchmarks/canonical_corruption_d228_2026-09-08.md).

## 2026-09-08 — explicit corruption input precision, after the D228 cost sweep

The existing cost-4 recipe is reproduced once with a new ZCTH v2 contract:
round declared inputs to f32 before scaler arithmetic. Original source tables,
admission, roles, head IDs and fitted numeric sections are unchanged. All
9,036 training pixel comparisons and 8,213 unique array parity rows pass exact
raw/probability/fire/composed-score comparisons. Legacy v1 remains unchanged.
Calibration still lowers three honest near-lossless JXL outputs and fails the
fixed screen. No validation/terminal evaluation or new data admission occurs.
[Precision evidence and remaining accuracy work](../benchmarks/corruption_input_precision_2026-09-08.md).

## 2026-09-08 — honest native-map supplement and decoder-contract finding

Reuse 608 honest JXL/AVIF non-scalar bound outputs on the eight admitted fit
origins. Current native decoding/extraction adds 402 unique honest pixel pairs;
calibration and validation inputs remain unchanged. One fixed v2 cost-4 fit
passes 8,615-row array and 9,644-attempt pixel audits but lowers two honest
calibration outputs, still FAIL; no validation/terminal scoring.

All 336 JXL current decoded hashes differ from their old native-bound hashes,
with D score drift up to 2.425 points. All 272 AVIF decoded hashes agree. Keep
the two JXL decoder contracts distinct; the current training packet uses only
the canonical decoder. Source inspection suggests U8 dithering versus f32
plain rounding, but the decoder packages also differ and a controlled causal
check remains necessary. [Full record](../benchmarks/corruption_honest_map_2026-09-08.md).

## 2026-09-08 — native JXL delivered-pixel repair, after the honest supplement

The same-decoder on/off control now reproduces all 336 canonical and historical
hashes. Dithering changes 8,646,810 RGB samples by at most one code; only 178
samples differ between canonical undithered and historical f32-rounded output.
Native targeting now uses canonical U8 dithering and rejects older decoder-era
calibrations. On the unchanged twelve admitted train sources, 756 newly emitted
bitstreams independently reproduce every decoded hash and f32-reported D score.
Work: 756 full encodes, 1,512 internal reconstructions/map evaluations/native
comparisons, and 756 delivered-image comparisons, plus separate audit costs.
No model fit, validation/terminal scoring, or protected-content admission here.
The preceding honest-head fit already used canonical pixels; this decoder
repair does not invalidate or erase its two honest false positives.
[Controlled measurement and serving contract](../../jxl-encoder/benchmarks/zensim_decode_contract_2026-09-08.md).

## 2026-09-08 — base preferences after canonical JXL decoder repair

Reuse 252 canonical scalar JXL outputs and twelve identities on the same
admitted train origins. Nine packed/declared-ID models have complete Rust
pixel/cache audits; an additional D audit matches the PNG judge inputs to
bitstream pixels exactly. SSIM2/Butteraugli resolve 2,471 within-source pairs.
D has zero wrong directions; generation A/B have three each; three A_plain
seeds have 170–202 and H_anchorlad seeds 48–69. No network fitting or validation
occurs. The strict-improvement screen advances no model, and cannot distinguish
new zero-error candidates from D. Preserve its outcome and the separate spatial
RD failures. A subsequent A/D blend fit is registered on eight fit origins,
with four separate training-calibration origins and full Rust serving required.
[Complete record](../benchmarks/model_preferences_2026-09-08.md).

## 2026-09-08 — fitted A/D composition and actual spatial coverage

The preregistered hundredth-grid fit chooses D weight 0.41 on eight fit origins:
all 1,647 consensus pairs order correctly. All 824 pairs on the four separate
training-calibration origins also pass. Rust `BakeScorer::ensemble` reproduces
the complete calibrated blend exactly, with twelve exact identities and no
score above 100. No new network, encoding, validation or terminal work occurs.

Actual spatial serving on the same 264 pairs matches scalar scores/features,
but 252 nonidentity maps omit active pooled-feature terms (169 unique IDs in
f156..371 across the corpus). The D-only endpoint has complete coverage. The
scalar screen passes; native steering remains incomplete. Next implement the
missing v1 pooled integrands in the existing Rust attribution owner, preserving
feature arithmetic and explicit discontinuity/coverage limits.
[Model, controls and delivery](../benchmarks/model_blend_2026-09-08.md).


## 2026-09-13 — Rev3 feature subset development packet

Fresh canonical extraction of the September 8 model-preference packet:
264 pairs, twelve previously admitted T2 training origins, canonical decoded
PNG bytes and SSIMULACRA2 proxy labels pinned by the committed
`benchmarks/feature_screen_2026-09-13.json`. No new content or protected labels.
The existing eight fit origins remain fit; prior calibration origins1214/6064
become checkpoint dev,8462/9066 inner-test. All are still T2, previously
examined; this is not an unseen evaluation claim. Features are f64 Parquet
with source IDs, era `v1screen_rev3`, formula3, producer hash d16a1091.
Per-run manifests bind original/decoded bytes, splits, extractor/decoder,
trainer/panel binaries, cache tables and baked output hashes. No Rev1 table
was relabelled. Two H32 proxy models train and pass final Rust pixel/cache
audits; they do not qualify a competitive or production model.

Artifacts: `~/work/zensim-validation-2026-09-13/feature-screen-final-cold-complete/`
and `feature-screen-final-warm/`; earlier CSV-withinref and Snappy-format
attempts fail before training and remain preserved. A subsequent preliminary
fit exposed the trainer's historical automatic full-verdict launch: it refused
Rev3 before corpus scoring. The final recipe passes the new `--no-auto-eval`
flag; neither final run launches that evaluator. [Recipe and results](../benchmarks/fullres_y_subset_2026-09-12.md).


## September 14: strict train/eval minimal-wide study

See [registration, methods and results](../benchmarks/minimal_top_2026-09-13.md)
and its adjacent recipe/results JSON. Original canonical KADID TRAIN (5,000/40
refs) plus approved train-only TID (3,000/25 refs) train the frozen models.
Original canonical KADID SELECT (3,125/25 refs), admitted as eval, is re-extracted
with Rev3; no retired screen/test cache is read or relabeled. Source admissions,
canonical-view hashes, pixel hashes and row order live under
`~/work/zensim-validation-2026-09-13/minimal-top/`. All 8,125 KADID rows align
exactly on 12 unchanged MSE coordinates; a within-reference shifted-row negative
control rejects every row. Repeated signatures have identical decoded pixels.

The native Rust audit verifies 100 identity pairs in the 3,125-row eval. The
final identity-bearing view copies the same features/labels and adds verified
0/1 `pixels_identical` evidence. `ensemble_score_rows` and `bake_verdict` now
consume it through `BakeScorer::score_features_with_identity`. Earlier raw
feature-only eval artifacts are retained but superseded by `final-eval/`.
This correction changes no training input, checkpoint, calibration or model.

## September 14: native codec validation for frozen Rev3 candidates

[Protocol and results](../benchmarks/rev3_codec_eval_2026-09-14.md) admit the
hash-bound R1b D1 validation keys for imazen26/nonphoto/hfnlproxy, with canonical
source suffixes {1,3,5}. All 20,655 distinct original bitstream pairs produce
finite native Rev3 full944 features; 20,812 original slice memberships are
preserved. No size-dependent row dropping or old944 feature reuse. These are
overlapping historical SSIMULACRA2 proxy views, not independent human studies.
Original imazen26 targets are raw SSIMULACRA2; the other two are divided by 100.
New tables explicitly normalize all three to score/100 and retain original units.

The audit optionally calls existing fast-ssim2 on its exact decoded RGB8 buffers.
Historical/current teacher drift is recorded separately; high-band membership
is unchanged. Matched native B/D Rev1 extraction covers every pair and every
reported baseline prediction matches its image score exactly. Unknown legacy
bake training-era metadata remains unqualified. Nine frozen candidate verdicts
run through public BakeScorer on native features. Prior human panels are reused
only after manifest/hash checks and remain unchanged. All candidates remain
NO-SHIP, with complete product composite, targeting/RD and performance missing.

Artifacts: `~/work/zensim-validation-2026-09-14/rev3-codec-eval/`, including
original protocol/binary, admission, per-file/pixel hashes, full row predictions,
source bootstrap, codec panels and teacher drift. No test/terminal table or
historical internal test segment was opened or reclassified.

## 2026-09-14: explicit native SDR corruption delivery

[Color resolution](../benchmarks/integrity_color_resolution_2026-09-14.md) restores
original TRAIN mobile source8014 via full CMS and explicit sRGB clipping/u8
quantization before existing corruption operators. New manifest era
`native-corruption-fixtures-v2`; original/source-delivered/generated hashes and
input receipt are bound. TRAIN source1200 is byte-exact with v1 (718 files).
Frozen head assessment only: 716 rows, 705 distinct, 30/31 severe proxies,
0/3 known-valid activations. No fit/calibration/EVAL/public TEST access. Original
181 native nonidentity quarantines remain; dirty historical producer prevents
conclusive color admission. Do not mix old and v2 training packets.

## 2026-09-14: complete Rev3 recovery caches and corrected native input audit

Full TRAIN-only legacy-RGB8 Rev3 caches are verified: CID22 17,611 pairs/201
references; SafeSyn 196,086 pairs/3,218 source paths,944finite features each,
zero failures. Original oracles remain alongside fresh same-buffer fast-ssim2.
SafeSyn retains 11,591 negative targets. Parquet SHA256:
`6044fdc8cf4f646cda6457a9e73e54edd7d9fed57dfd95f1c47e1f4091220c68`.
Source-family fit/development/calibration partition admission is still pending;
no fitting, EVAL/TEST access or qualification occurred.

The214-pair native SDR TRAIN audit now supports existing fast-ssim2 on canonical
linear-float buffers with original and transformed hashes. Existing feature,
scalar and spatial results remain unchanged. This does not resolve historical
producer quarantines or make SDR peer scores an HDR oracle.

HDR input correction is `hdr-common-primaries-v2`: declared primaries transform
to the opsin linear-sRGB basis without SDR clipping; HLG uses source-basis
luminance. Affected old HDR caches/calibrations are incompatible. The datagen
extractor requires `hdr-common-primaries-v2-bt2020-pq10000`, preserves native16,
checks color metadata and refuses partial failures before producing output.
No corrected HDR training cache is claimed here. See [results](../benchmarks/rev3_native_optimization_2026-09-14.md).

## 2026-09-19: JPEG AIC2026 ingested as a metric-agreement panel (no human labels)

**What it is.** `AIC2026` (DaRUS doi:10.18419/DARUS-6156, v2.0, CC BY-SA 4.0;
Jenadeleh, Sneyers, Ascenso, Richter, Karabutov, Jia, Alshina, Watanabe,
Pinheiro, Ebrahimi, Saupe; arXiv:2607.22783) — 70 source images, 17
codec/configuration arms (5 base codecs on all 70: JPG, J2K, JXL, AVIF, JAI;
12 extended arms on 11–13 sources each), 20 distortion levels per arm (FTIC has
6), **9,618 distorted images**. Levels are placed to span roughly 0.2–4.0 JND
using CVVDP estimates. Ships full-resolution decodes, 840×944 `PTC_` (plain) and
`BTC_` (boosted/flicker) crops, codec-native bitstreams, and two score tables —
`metrics_fullres.csv` and `metrics_cropped.csv`, 9,618 rows each, 71 objective
IQA-method score columns (including `JND_*` calibrations for seven of them).
Local at `/mnt/v/datasets/aic2026/`.

**Why ingested.** It is the largest public population that holds *one content
set* against *17 codec arms* at *matched, finely-spaced fidelity* in the
high-quality range — the exact shape needed to ask whether a single zensim dial
value means the same thing across codecs, and whether our ladders are monotone
where a codec's operating points are only ~0.2 JND apart. It also carries peer
implementations of metrics we ship our own versions of (`SSIMULACRA2`,
`proposal-Butteraugli`), which makes an implementation-parity check possible
against a third party's numbers rather than our own.

**What it cannot show.** *It contains no human scores.* The subjective study
over these stimuli had not been released. Therefore:

- Nothing computed on it is accuracy against human judgment. Every number is
  **metric-vs-metric agreement** or **ladder behaviour**. Say so wherever it
  is cited.
- It must not be given a synthesised `human_score`, and it must not become a
  `rank.<corpus>` board axis that feeds the composite. If it ever becomes a
  board axis it is a clearly-labelled non-human agreement axis, excluded from
  the composite.
- **CVVDP placed the levels**, so CVVDP is monotone on these ladders by
  construction and is not a fair contestant on the monotonicity axis. A
  monotonicity table that ranks CVVDP first is reporting the selection rule.
- Column orientation is mixed (`JND_*` and the distance metrics rise with
  distortion; `SSIMULACRA2`/`CVVDP`/`PSNR`/`SSIM` families fall). Sign-normalise
  per column or every correlation sign is a coin flip.
- `BTC_*` crops are *boosted* stimuli (2× magnification and amplified
  artifacts, built for flicker presentation). They are not ordinary images and
  the shipped score tables do not cover them — `metrics_cropped.csv` is the
  `PTC_` population.

**Split role.** Registered in [DATA_SPLITS](DATA_SPLITS.md) §3 and §3d as a
member of `jpeg-aic-family-holdout-2026-09-01`: T0-family, **eval-only, never
trained on**, membership by content.

**Exposure.** First read by any zensim model on **2026-09-19**. All six models
scored in that first read (`PreviewV0_2`, `B`, `C`, `D`, and the two frozen
Rev3 ensembles `R915_y60_h32_ens5` / `R915_basic228_h128_ens5`) were frozen
before that date, so the first read is an honest out-of-sample observation for
all of them. Results: `benchmarks/aic2026_agreement_2026-09-19.md`; artifacts +
manifest pointer `benchmarks/aic2026_2026-09-19.pointer.md`.





**The split.** The 49 references of `CID22_validation_set.csv`
(`reference_img` membership only; zero overlap with the 201-ref
SSIMULACRA2-anchored train population — verified disjoint) sorted by
basename, Fisher–Yates shuffled with recorded seed **20260919**:

- **CID22-A (25 refs, FIT-ALLOWED for the Phase-2d constants fit only):**
  `1189261.png`, `1531677.png`, `159550.png`, `1624487.png`,
  `162520.png`, `164595.png`, `2079234.png`,
  `21169144185_3f7977cb5a_o.png`, `225228.png`, `2389166.png`,
  `2936831.png`, `3316926.png`, `3653963.png`, `373965.png`,
  `3762075.png`, `4215100.png`, `6078297.png`, `6292444.png`,
  `70497.png`, `7062219.png`, `844297.png`, `pexels-photo-2686358.png`,
  `pexels-photo-2802032.png`, `pexels-photo-4210863.png`,
  `ularapi_Semarang_City_Logo.png`
- **CID22-B (24 refs, SEALED):** `1025469.png`, `1044329.png`,
  `1279330.png`, `1418519.png`, `1420710.png`, `1475938.png`,
  `1544947.png`, `2190188.png`, `2253934.png`, `2670327.png`,
  `2736139.png`, `2775196.png`, `2887497.png`, `297394.png`,
  `3156482.png`, `3316926_opo25u.png`, `3637739.png`, `382297.png`,
  `5055743.png`, `5458393.png`, `7552578.png`, `792079.png`,
  `adriankierman-report-page.png`, `pexels-photo-1933873.png`





## 2026-09-23: Rev4 existing-family feature bank — content-addressed f32 sidecars, promoted with corrections

**What was built.** `/var/tmp/rev4-featbank/bank/` — the Rev4 feature bank,
existing-family Rev3 rows only: 18 sets, 249,227 source stimuli → 248,983
unique content-addressed `pair_key`s (244 pixel-identical stimuli collapsed,
features verified bit-identical before collapse), ≈696 MB of
`zstd-3/BYTE_STREAM_SPLIT` parquets. Feature set
`basic+peaks+masked+iw+v2+append+append2@w944/ceiling_rev3#b782e349`
(producer); the bank's populated-905 storage form is registered as consumer
`…/ceiling_rev3#e3db6aab` in `benchmarks/feature_sets_registry.json`.
Extractor: pinned `extract-native-admission`, sha256
`7c7ffbbfa033e8ca1a8f103d472b61ccde061c2394b03d519af2852ee8eeda87`,
`ZENSIM_FORMULA_REV=3`, `ZENSIM_ROOT_FORM=sqrt`. Report + per-file hashes:
`benchmarks/rev4_featbank_extract_2026-09-23.{md,json,pointer.md}`.

**The actual why / corrections worth remembering.**

- The ext944 caches were a *different era* (same decode, different formula
  revision) — proven by bit-compare, not assumed. Only baseline-recovery
  (cid22_train, safesyn) and the ceiling parquets were true Rev3 caches;
  everything else was fresh-extracted. Fresh re-extraction of all 11,125
  ceiling rows is bit-exact vs the ceiling parquets on all 13,872,080 cells.
- The extractor's `feats.csv` comes out sorted by `ref_basename`, NOT pairs
  order — positional joins silently corrupt. Every fresh pairs TSV carries a
  dense `row_id` echoed into audit JSONL and CSV; all joins are keyed.
- KADID no-op level-1 distortions decode to identical pixels: distinct
  source stimuli collapse onto shared `pair_key`s (train 5,000→4,880,
  SELECT 3,125→3,050, TERMINAL 2,000→1,952; CSIQ 866→865). Labels retain
  every stimulus (`source_row_id`); `keys.n_stimuli` is the multiplicity.
- `cid22_a25`'s `human_score` is **CID22 human MCOS/100**, not an SSIM2
  oracle — caught at review; `label_scale` is now a per-set manifest field.
- Held-out `human_score` replicas in extraction `pairs/`+`raw/` were moved
  (not deleted/regenerated) under `/var/tmp/rev4-featbank/_sealed/` per
  review correction 4a; the bank emits no labels for confirmation sets.
- **No feature here is shown to help.** Nothing was fitted, selected, or
  claimed; the bank is a storage/provenance artifact. Part B (new-family
  sidecars C1–C4 + C8 gmsbank) is gated on `featbank-impl` and
  `codex-gmsbank` reviews plus explicit user go.


## 2026-09-24 — quarantined C8 chroma calibration (gmsd-chroma)

C8's unconsumed f1322+ definition is revised in this lane. Width1502 alone
cannot identify it: native Y plus coarse XYZ/CS replaces the prior all-scale
XYZ gradient bank. Source prereg9016272c and implementationf38befae, design
`benchmarks/rev4_gmsbank_design_2026-09-23.md`, calibration
`benchmarks/gmsd-chroma_calibration_2026-09-24.md`. No f0–f1321 definition changes.
No human labels entered the pixel calibration or Part B gates.

The 652 previously frozen CID22 TRAIN/SafeSyn TRAIN records yield1520
native/box-derived pair/geometry records under the committed size/content
selection and inherited source groups. Original RGB8 decode snapshot
3306465d56d279a512b02c3a63de701e9b634d6aebda5e12fb164817ba491122;
this preserves the parent's legacy RGB8 decode era, including its recorded
AVIF route limitations. Do not mix with a changed decoder era by width.
Source manifests, exact commands, producer sources and resolved dependency
locks are retained under `/var/tmp/gmsd-chroma/`; large derivative/XYB planes
are in `/var/tmp/gmsd-chroma/remote-r5600g/` (14,566 files, 4.4 GB, copied from the former remote tree, which has since been removed) and its mirror on the tower at `output/zensim/gmsd-chroma-2026-09-24/remote-r5600g/` (sha256 spot-checked). Calibration report SHA256
`a6aecdf036d89b7afef10e9c8aaa6c935836fbc69ed12950e56f41eabcf9c633`.

Author MDSI maps are differential oracles only and enter no dataset/model.
The potential proposal updates P1 and proposes optional P2b (MDSI peer scalar);
its admission requires the coordinator's pre-label decision. No potential
fit, sidecar promotion, registry adoption or push is authorized by this record.


## 2026-09-25 — restored-cut sidecars (restore-cuts, quarantined)




## 2026-09-22: CID22-49 carries one picture under two names — CID22-B(24) corrected to CID22-B(23)



**Consequence for the A/B split.** The 2026-09-19 split was by reference
filename (25/24, zero shared names) and so leaked this picture: one of B's 24
references (89 of its 2,100 pairs) duplicates a fitted A reference. Any
split of CID22-49 by filename leaks it; **splits must be by content hash**.





**User direction (verbatim).** Work order §4: "Train the best configuration on
full CID22 human data (user direction: \"consider\")" … "fit it on all CID22
human validation pairs the way the author did, to get the like-for-like with
their in-sample 0.88289 / 0.69446" (the talk's reported CID22 figures). Variant selection (2026-09-22):
"Constants — THREE sources, each a preset: full CID22 human, author-style
(ledger the exposure first, per §4)". This entry is committed before any
CID22 label is read for these fits.





**Consequence.** Those presets have **zero held-out CID22 claim**: their
CID22 numbers are fit-domain only, labelled so wherever they appear, exactly
like the talk's reported in-sample CID22 figures. For every other model,
CID22-B(23) remains the held-out CID22 read, and only for models never
fitted or selected on B.



## 2026-09-22: NNCD-IQA registered as an EVAL-only corpus (zensim#62)



**Role.** EVAL-only, never trained on, never used for fitting, calibration or
selection. Target orientation: quality (MOS rises with quality).



**Exposure note.** During format inspection on 2026-09-22 the first four data
rows of `MOS_scores_sorted.csv` (bmshj2018-factorized, image 1) were printed
to the terminal before this registration. Nothing was fitted or selected with
them.

























## 2026-10-04 — R5INTEG canonical TRAIN revision refits

The explicit R5INTEG brief authorizes the historical canonical packet's12 TRAIN
origins and original8-fit/4-calibration split. Re-extract all9,036 attempts from
unchanged catalog PNG/native bitstreams through production research extraction
at Rev4 and Rev5. Pin original source bytes, order, two lockfiles, extractor SHA,
explicit populated IDs and producer feature-set ID in `/var/tmp/r5integ`.
Rev4 requests f0..719 and fits the existing all-372 regime; Rev5 requests
f0..227/f372..719 and fits the existing D228 regime. Wide storage columns beyond
the requested IDs are declared structural zeros, never computed families.

526 verified identity attempts/eight distinct identity pairs are retained in raw
extraction and excluded before fitting/weighting because public identity bypasses
model inference. The8,510 remaining attempts deduplicate to8,205 pairs:
5,499 fit/2,706 calibration,7,725 historical corruption positives/480 honest
nonidentity controls (456 native JXL/AVIF and24 q10/q20 anchors). No canonical
validation pair is decoded or scored, and no protected label or sealed data is
read. The existing admission owner does hash-only verification of protected
reference bytes; the worklog and ADMISSION_REHASH_SCOPE.json disclose those
reads. No label, threshold or seed is selected.

Use the existing canonical HGB/isotonic recipe, honest multiplier1, seed4101,
f32 input precision, fixed P>0.9. Export revision-bound ZCTH v3 through the
existing writer, evaluate the exact exported head through public Rust, and retain
failed activation gates. A predeclared Rev4 D228 diagnostic isolates the144-slot
masked/IW removal; it is not a selection arm. The retained c95bd5 Rev1 control is
the earlier September6 theory fit, not a same-training reproduction.

Calibration honest activations are1/160 (Rev4 all-372) and3/160 (Rev4 D228 and
Rev5 D228). The all-372 head fails zero-native activation; D228 also fails the
calibration <=1% activation limit. Historical catalog labels do not qualify
catastrophic recall. No profile/default is promoted. Full per-bake controls,
parity, low-quality strata, hashes and local artifacts are recorded in
[the worklog](../benchmarks/r5integ_WORKLOG.md).

## 2026-10-04 — R5INTEG2 registered basic plus v2 TRAIN development

Coordinator-authorized head reads are f0..227 and f372..719 (576 features),
registered before fitting in benchmarks/r5integ2_REGISTRATION.json. Reuse the
immutable R5INTEG Rev4/Rev5 research payloads at matching sqrt revisions,
with the same eight fit/four calibration TRAIN origins, no evaluation rows,
seed4101 and unchanged fixed HGB/weighted-isotonic/deadband recipe. Labels,
identities and duplicate handling are unchanged:8510 nonidentity attempts,
8205 unique pairs,5499 fit/2706 calibration,7725 positives/480 honest.
No extra extraction was used to prepare fit columns.

Width1825 exports unnecessarily enabled unrelated Rev4 runtime families;
benchmarks/r5integ2_LAYOUT_REGISTRATION.json registered bounded caller width720
before rerunning the same fits. Both8205-row NPZ feature, raw and probability
arrays are bit-identical to the first576 fits. The separate serving-extractor
pin keeps the immutable feature-producer receipt intact. The research audit
projects exactly the bounded prefix for head evaluation, preserving the full
consumed-ID coverage checks and original tolerances. Final Rev4/5 head hashes:
3368dd75296e09fa556debb1ed9ba7b7d65117a0c88d9fc45b2840edd4a37a47 /
ae789b5f2e8a598f4152caa2666be663e736561a5653a91e6a1b6a31c475c32b.

Both primary TRAIN calibration screens pass all seven existing gates:
2536/2546 positives detected,0/160 honest activations/lowering, real_bug
149/155 at Rev4 and150/155 at Rev5,2537/2546 below q20. This is TRAIN
development, not held-out or production qualification; no default is promoted.
The full six R5CONFIRM Rev5 bakes arrived during the follow-up and replace the
separate stamped smoke evidence in the final attach matrix. Their source
provenance is pinned from ~/tmp/rev5bakes/provenance.json; no fit data or
confirmation labels from that lane were opened here.

Origin6064 independent CPU zenmetrics score-pairs (requested exec-cvvdp Docker
image, pinned binary/image digest) judges retained honest pairs unchanged.
Native/delivered JXL d0.01 scores: SSIMULACRA2=96.65016051457665,
Butteraugli maximum=0.11993369460105896; RGB8 difference maximum1. AVIF160/176:
SSIMULACRA2=74.61134204762597/67.22642850810986, Butteraugli maximum=
4.041477203369141/4.670391082763672. Full precision is in external JUDGES.json.
Reference is196x256 RGB8, no alpha/profile/transfer chunks; legacy scoring
assumes sRGB. Source/native/delivered pixel hashes and CICP/ICC inspection
receipts are retained; no label or threshold changes follow from these checks.

The prepared handoff defect affected Rev1–4 (replayed against frozen7a9dbacb);
P1 now retains the complete union while detaching companion finite-difference
probes. P2 admits manifest roles and every origin before any audit/table/payload
read; historical evaluation behavior stays intact. Both fixes have direct
regressions. Scalar/steering correctness and measured additional X/B/peak work
are distinct from scientific specificity and qualification.

## 2026-10-04: R5INTEG3 exact by_v2fy read-set preregistration (TRAIN development)

`benchmarks/r5integ3_REGISTRATION.json` pins exactly420 by_v2fy semantic IDs
from the costset2 candidate JSON, verified by `zensim::declared_feature_ids`
against all six Rev4/Rev5 full by_v2fy bakes before fitting. Caller width720.
Reuse matching immutable Rev4/Rev5 research features and original TRAIN
fit/calibration origin split; unchanged fixed canonical recipe, seed4101 and
seven gates. No extraction for fitting, masked/IW reads, tuning or relabeling.
The existing trainer admits this separately registered exact gapped projection;
existing Rust exporter/parity and scorer owners remain responsible for serving.
Rev5 uses actual full R5CONFIRM bakes for the fit base and matrix. Register15
full compositions, unchanged five prepared probes,12-origin identity checks,
and exact private ComputeSet equality. This cannot qualify a production model.

R5INTEG3 measured result: fixed Rev4/Rev5 heads both pass all seven gates on
all15 actual full-bake compositions (9 Rev4 incl. supplemental U,6 Rev5).
Calibration detection2532/2546 and2534/2546; honest activation/lowering0/160;
real bugs147/155 and148/155; below q20 Rev4 2533–2537/2546, Rev5 2536–2537/2546.
These give4 and2 fewer positive detections than576-read heads on the same
TRAIN split. Exact private whole-ComputeSet equality passes all15, including
Rev5 by_v2fy full_res_xb=false and Peaks=Off; no added feature extraction.
Head inference remains additional work. Prepared75 rows (60 accept/15 typed
reject/3360 unchanged base-map queries), identity180 and cross-revision2 pass.
Full row, score and probability parity errors are zero. This remains TRAIN
research. Artifacts/pointers: `benchmarks/r5integ3_ARTIFACTS.json`;
`~/tmp/zensim-paper/rev4/R5INTEG3_assets/summary.json` owns detailed results.


## 2026-10-04 HDRTEACH — native HDR-VDP-3 metric teacher labels (completed 2026-10-05)

Corrected TRAIN 7425 pairs/495 references/33 origins and registered hdr_v3mix VAL 3900/300/20 retain exact original row IDs and roles. New schema 1 tables under `/mnt/v/output/zensim/hdrteach-2026-10-04/` add raw HDR-VDP-3.0.7 q_jod targets, original second-teacher JOD, actual producing host, binary/source/view/input provenance, within-reference averaged midranks and preregistered agree (absolute rank difference <=1 position). TRAIN second teacher is fresh 2026-09-15native CVVDP; VAL second teacher is historic registered JOD, with original mixed target separately renamed historic_cvvdp_mix. No feature columns copied or mixed, no training/bake/selection/default change, no UPIQ data or T0 opened.

Fixed ppd 60, quality, absolute BT.709 nits from native PQ/common-primary ingress, led-lcd-srgb/none/age 24/reference quality options, no added ambient reflection. zenmetrics348bde5bf96d635383965d80d9adb8f9c195fd14, binary1278c6935ad4ebbb624e6213c56ad672602b05433b288ea1f9274dbf02dee721. Input manifestSHA83f6efe96a6db66b1eff72cbcdbdd03d07a3144764140305a1cc92a90e26cca8 pins 12120 original files. Exact encoder source commits remain unknown; producer epochs/TSVs and native bytes are retained, never inferred from filenames. Full 3975-shard/11325-key admission and independent audit PASS. Signed pooled SROCC 0.8283190441/0.8211975658; within-reference mean 0.9988197386/0.9820512821; agree 7390/7425 and 3696/3900. All 795 reference correlations defined. This is metric agreement, not human accuracy.

HDR-VDP-3 paper sections 3 and 5 state UPIQ quality calibration/recalibration (>4000 SDR/HDR images). Consequently UPIQ cannot be an independent human test for any student trained on these targets. Publication text only read. Future fit/filter is separately registered and TRAIN-only; VAL remains validation-only. Original HDRCORR outputs/report unchanged. Report `zensim--hdrcorr/benchmarks/hdrteach_2026-10-04.md`; full provenance/input/view/manifest/audit and retained fleet stop/migration evidence are in HDRTEACH_assets.

- `hdrteach_train.parquet` SHA256 `deb70e775b043a578c77e0c3ff27960ebfa936e9d901497f9d74389e6ce9fbce`
- `hdrteach_val.parquet` SHA256 `4b0f39dea0255659232f248070c8683dd7edf5916c63621cbb099b4afb6c9057`
- `agreement.json` SHA256 `607ac4090a74c5f5537d096e1ed491b70b31f8ccc1b47c03752adaef305cb9db`

## 2026-10-05: SHIPPATH byte-preserving Rev5 TRAIN teacher admission view

The existing v2c builder now has a bounded `admit-teachers` route. Four copied
TRAIN teacher Parquets have the same SHA-256 as their frozen v2c5 source
files; no feature/target/key order changed and no original was overwritten.
New sidecars expose the recorded Rev5 producer identity and bind decoder era
to the actual c649e810… producing executable under legacy-rgb8. The shared
validation registry now knows Rev5's measured 576-slot subset. A canonical
Rust smoke fit embeds revision 5, that producer ID and qualified table
provenance without historical replay. It remains a bounded software smoke,
not a by_v2fy qualification or the full h32:H128:cv16:cf98 recipe.

Source hashes, exact commands, tests and all remaining gates are indexed in
[SHIPPATH worklog](../benchmarks/shippath_WORKLOG.md). The immutable research
receipts, provenance limitations and the Rev5 spec's 03:25 UTC correction
remain authoritative for their original artifacts.

## 2026-10-05 — SHIPPATH2 full recipe provenance view and strict export plumbing

Artifact authority `/var/tmp/shippath2/recipe-complete`, from frozen
`/var/tmp/rev4-featpot/v2c5` and `/var/tmp/rev5-featbank`. Four oracle teacher,
five design-human source and twelve human union/LODO fit/dev Parquets are copied
byte-identically. Generated label-free keys bind original key-file hashes,
ordered reference-disjoint development selections and row-key hashes. Human
role remains pending (SHIPPATH_decisions.md); preparing declarations does not
change scientific roles or create independent validation.

E15 pool remains42021 rows, SHA `6bf584ac70579bdf9a0242b7ccfd688ff0182cb5c75e35085ba71967207f8f5e`;
keys remain SHA `bc225a115ab8505738a5c17ced6d4fc592a9a38ac6d9ec98661f4e8f0898addf`.
New declaration binds actual extractor c649e810…/legacy-rgb8, Rev5 producer
basic+peaks+v2@w1825/rev5_localwin#36c3f3af, original selection-file/row-index/key
hashes and preserved derivation metadata. cf98 selects 17,600 existing rows with
zero target changes; original pool and all frozen sidecars remain unchanged.

A three-epoch TRAIN-only export smoke exercises the existing Rust trainer and
strict Python selection owner without historical replay. Epoch dumps preserve
revision/producer/repro and exact epoch-prefix sampler coverage. Canonical pack
needed an ID-indexed anchor width fix for declared dense inputs; a NaN-hole
regression and packed inference check cover it. No public API, arithmetic,
production weight/default or scientific qualification changed. Evidence/report:
`benchmarks/shippath2_WORKLOG.md`, `~/tmp/zensim-paper/rev4/SHIPPATH2_DONE.md`.


## 2026-10-05 — SHIPPATH3 review corrections (no new data)

Independent review reproduced stale checkpoint provenance relabeling and writes
under immutable source/bank roots. Rust now rejects nonempty checkpoint output
directories before admission/training. Fresh full-recipe receipts bind the bank
and original source paths; strict destination, scratch and derived-output guards
resolve symlink ancestry against all immutable inputs. Older views without that
binding require a fresh rebuild, preserving their frozen bytes. Regression data
is synthetic only; no human payload, protected label, scientific role, feature
value, trainer arithmetic or model qualification changed. See
`benchmarks/shippath3_WORKLOG.md` and `~/tmp/zensim-paper/rev4/SHIPPATH3_DONE.md`.


## 2026-10-05 — SHIPPATH4 checkpoint caller compatibility (synthetic only)

The SHIPPATH3 empty-directory rule rejected first fits whose inputs coexist
with dumps. The correction permits input/unrelated files, refuses surviving
trainer model artifacts, and stamps only invocation-local post-write receipts
from both CPU dump writers. Existing caller layouts and selected paths are
unchanged. Regressions use the real p2_mlp.train owner and synthetic rows; no
real labels or immutable inputs are read or modified. Before/after training
and packing compare all numerical/model/calibration bytes and canonical wire
bytes after eliding run identity; packing the same input is fully byte-identical.
The human-role decision stays pending. See benchmarks/shippath4_WORKLOG.md and
~/tmp/zensim-paper/rev4/SHIPPATH4_DONE.md.


## 2026-10-05 — SHIPPATH6 registered companion refit (unchanged TRAIN data)

Replayed the pinned R5INTEG3 Rev5/by_v2fy-420 preparation through the existing
HGB/scaler/isotonic factory and extended exporter: seed4101, fit5499/calibrate2706,
8205 unique source/pixel pairs. No new row/source selection, tuning, extraction,
EVAL or human fit. Fresh per-table admission views copy the two original TRAIN
Parquets byte-identically; original frozen roots remain immutable. The completed
canonical content-screen receipt is SHA-bound without reopening its protected
reference payloads. The executable-bound legacy-RGB8 decoder era records only
what the original extraction receipt proves.

ZCTH v4 preserves every numerical section of the registered v3 head, adds full
content/admission binding, and passes exact Rust and pixel/cache parity, all
seven original TRAIN gates and prepared/identity/revision/ComputeSet checks.
The composition owner verifies companion Table provenance; the historical
primary remains unqualified and the human-role decision remains pending.
See `benchmarks/shippath6_WORKLOG.md` and
`~/tmp/zensim-paper/rev4/SHIPPATH6_DONE.md`.

## 2026-10-05 — SHIPPATH7 Rev5 evaluation feature instruments

Features-only, same registered populations: ladder9593, standard replacement
4424, negative-tail2000, identity38, steering96/4/8/27, and seven original Rev5
confirmation/terminal banks totaling10721 rows. All 15 tables (26911 rows) declare
Rev5 `rev5_localwin`, exact populated by_v2fy420 IDs and the pinned c649 extractor/
legacy-RGB8 decoder. Steering uses59 exact canonical RGB8 deliveries. Source,
row/order, feature-vector, decoded-pixel and final table/declaration hashes are
retained; no protected/confirmation/T0 label attachment/read or model fit.

Frozen-input roots are preserved. Preparation grants no gate pass or exposure
approval; spent populations remain spent, the old May grid's pixels remain
unavailable, and the new table hash needs its own canonical floor-bar binding.
[SHIPPATH7 worklog](../benchmarks/shippath7_WORKLOG.md) describes artifact paths,
owner controls, pending exposure freeze and composition provenance limitations.

## 2026-10-05 — SHIPPATH8 metadata preflight and origin corrections

The pre-landing SHIPPATH7 review found omitted automatic metadata paths and eight
incorrect `pairs_origin` fields. The explicit identity route now checks/binds
complete root/table/model/companion metadata discovery before opening it. Both
reviewer symlinks into synthetic `_sealed` sentinels refuse before open/evaluator
launch; no real protected labels were read. No label access is granted.

Fresh metadata-only corrected bank/instrument views are under
`/var/tmp/shippath8/verified`. Every table/key byte and consumed 420 vector/order
hash is unchanged across 15 tables / 26,911 rows. Original extraction/producer/decoder
identities and SHIPPATH7 evidence stay immutable; the correction is explicitly
recorded rather than restamping the extraction. Actual Rev5 admission passes,
exposure remains pending and the human production-role decision is unchanged.
See [worklog](../benchmarks/shippath8_WORKLOG.md).

## 2026-10-05 — SHIPPATH9 companion holdout-boundary correction

The focused SHIPPATH8 review found that discovery and inventory omitted the
companion owner's existing holdout-ancestry protection. Companion provenance
and discovery now use one Rust predicate, mirrored by the Python preflight;
both lexical and resolved paths reject `_sealed`, `holdout` and `labels__`
components before candidate return or hashing. Four direct/symlinked source and
binding controls show zero sentinel opens/reads through the actual owners.
Only newly created synthetic sentinels were used; no real protected labels
were read. No model, table, role, exposure permission or qualification changed.
Original SHIPPATH7/8 indexed evidence is reverified unchanged. The allowed
composition identity is unchanged and exposure remains pending. See
[worklog](../benchmarks/shippath9_WORKLOG.md).

## 2026-10-05 E26 native Rev5 teacher leg (registered execution)

[Registration](../benchmarks/e26_hdr_teacher_registration_2026-10-05.md) fixes
by_v2fy, headN, hd4/hd16, ten seeds/five folds, unchanged E24 Rev5 controls.
TRAIN alone supplies7390 agreement rows from the immutable HDRTEACH TRAIN
table; target exactly10*q_jod, no clipping, fit-only withinref,rank with
coverage acceptance weighting. Transform committed before any fit at8e8f3170.
VAL retains all3900 rows/300 references, including disagreements, for the
frozen registered two-teacher assessment. No HDR dev, calibration, early
stopping or checkpoint search. Every fit selects zero-based epoch119.
Native PQ16 PNG/JXL, declared actual primaries and PQ10000 use the production
planned HDR fold through the existing research/extractor owners, exact420
IDs, f64 features, absent NaNs; fitting casts those measured slots to f32 as
the SDR owner does. Original file hashes, order, keys, extraction executable
and source are pinned in native producer manifests. Explicit IDs/per-slot
provenance identify the channel subset; its reconstructible family-token
FeatureSetId is null. Never replace that with a width-inferred identity.

[Native bank pins](../benchmarks/e26_native_features_2026-10-05.pointer.json).
Program v28=7e8ac08b6b97fc78057251bdfa4af2a605543d8a36c0b2e1be4aec458634d5a0;
TRAIN-only data pack=2eb85985a69a255a784c4b8f50a1a9180034c8ccc69d92f95d179c5ba723d3c2.
P2 admission fix checks record/sidecar contract before any candidate payload
opens; actual loader/packer forbidden-VAL read tripwires pass. Original failed
smoke retained. No UPIQ or sealed read. Encoder commits remain the inherited
HDRTEACH unknowns; new byte hashes cannot establish missing codec commits.

Coordinator-authorized serving binding, identical for50 controls and100 arms:
existing dense_bake BIT-IDENTICAL512-row gate then pinned bake_stamp_revision
to separate Rev5 outputs with per-bake source/dense/stamped/gate receipts.
Immutable cells remain untouched. v2c5 trainer logs formula_revision null and
qualified_provenance false are an explicit admission gap, to fix forward
separately; serving revision comes from the stamp, not retrospective trainer
qualification. Report-only external SDR, canonical corruption TRAIN and one
fixed HDR TRAIN steering panel cannot change the registered adoption rule.
[Execution worklog](../benchmarks/E26_WORKLOG.md).

Final registered E26 result: hd4 is the lowest passing weight; both arms pass
SDR as-good and the two within-reference HDR conditions over all50 matched
cells. Hd4 HDR-VDP-3 delta+0.0014655677655677746, SE0.0003411001471767929;
CVVDP delta+0.0015161172161172077, SE0.0003392474364166754. Hd16 deltas
+0.0014747252747252793 / +0.0015007326007325662, SEs0.0003378213879858726 /
0.00033559844516359565. Pooled HDR-VDP-3 falls(-0.044864262128246485 hd4,
-0.12476287879994075 hd16); it remains reported-only under the preregistration.
Human HDR, encoder RD/spatial and full product qualification remain MISSING.
All150 full VAL panels retain3900 rows/300 references and per-bake bindings;
whole native/cache control parity is bit-identical with0 pixel-identical rows.
Full records: output/zensim/e26-2026-10-05 (tower); large JSONs are external.
V28's actual packed Rust executables are the existing fleet-v2 set
6b28576f/81ec2207/c9c610b8, not the v25b hashes in copied binary_mix prose.
The embedded inventory is correct; supplemental provenance erratum records
actual hashes and bounded canonical parity evidence without rewriting the pack.


## 2026-10-05 E27 registered preparation on the E26 landing chain

Registration070b8247 fixes hp4 pooled rank and ha4 within-reference MSE+rank,
nominal HDR weight4 through E26's unchanged weighting convention. Every other
fit parameter and original TRAIN role/authority stays E26's:7390 agreement
rows, unclipped10*q_jod, Rev5 native420 by_v2fy IDs, headN, seeds0–9/five folds,
120 epochs/final119. Fresh v2e27 transport preserves all56 E26 payload hashes;
no HDR VAL/development/confirmation payload is added. A one-cell hp4 smoke
checks mechanics, not scientific adoption. E27 HDR VAL is not read for prep.

The assessment owner now supports the registered E27 pooled and within-reference
gates for both teachers, SDR E21, and larger passing pooled HDR-VDP-3 gain;
E26 hd4 and full raw/cell geometry remain report-only. Serving binding stays
approved dense prediction gate followed by receipt-bound Rev5 stamps, identically
for controls and new arms. Exact E26 executables/shared program data are reused;
merged runtime scripts and their v2c_wide dependency are explicitly pinned.
The old trainer's null revision/unqualified admission remains an inherited gap;
no stamp upgrades training qualification. Human HDR, original encoder commits,
encoder RD/spatial and product/runtime qualification remain MISSING.

E26 landing review found no defects. No E27 enqueue or source push; the launch
owner requires coordinator confirmation of E26 landing and matching local
zenmetrics profile push. No tail controller. A broad filename-only tool lookup
reached protected _sealed directory enumeration and child access was denied;
no label payload was opened and no data came from that attempt. Further tool
lookup is restricted to named tool directories. See the E27 worklog and
[preparation pointer](../benchmarks/e27_preparation_2026-10-05.pointer.json).

E27 preparation smoke completed: full hp4/kadid/s0, final119,7390 HDR TRAIN,
7869SDR predictions, zeroHDRdev; native/cache one-TRAIN-row score bit-identical
74.63024139404297 after dense gate+Rev5 stamp. Default-Rev1 native refusal was
retained and corrected by explicit Rev5 invocation, with no refit. No HDR VAL
assessment or fleet enqueue during preparation. Readiness is separately gated
on coordinator E26 landing and matching zenmetrics push confirmations.

## 2026-10-06 E27 registered assessment completed

All100 final119 arm cells and200 complete HDR VAL panels were verified. Neither
arm passes E27: hp4 passes SDR and pooled HDR-VDP-3 improvement but fails pooled
CVVDP noninferiority; ha4 fails SDR and HDR guards. Retain the control. E26 hd4,
all external categories and raw geometry remain report-only. Historical table
admission stays unqualified; approved Rev5 stamps supply serving binding. Human
HDR judgment, encoder provenance completion and shipping gates remain missing.
See [E27 worklog](../benchmarks/E27_WORKLOG.md),
[complete measured summary](../benchmarks/e27_result_summary_2026-10-05.json) and
[immutable Tower evidence](../benchmarks/e27_final_2026-10-05.pointer.json).

## 2026-10-19 — matched existing B/D gauntlet assessment

User-requested comparison on the exact previously exposed external populations: CSIQ865, PIPAL23200, CID22-B2100, KonJND JPEG SELECT404 and LIVE clean10/267. Existing B a96a5a66 and D cd1098b4 retain their Rev1 arithmetic and were scored on fresh default372 native features; Rev5 candidate banks are unchanged and incompatible columns were never substituted. All26836 B production predictions equal actual pixel/cache/literal/stored-f32 scores exactly; D has18 fixed exact native pixel audits spanning the five panels. Explicit integer row IDs restore extractor reference-grouped CSV ordering. Canonical native rank/scatter owners and10000 source-cluster interval draws remain unchanged. No fitting, selection, gate/threshold change, protected population access or qualification. Comparable B/D whole-dataset-withheld fits for the four D1 panels are unavailable, so those legacy entries remain absent. Historical source/crop lineage limits remain explicit. See [matched comparison](../benchmarks/gauntlet_legacy_comparison_2026-10-19.md) and [artifact pointer](../benchmarks/gauntlet_legacy_comparison_2026-10-19.pointer.md).

## 2026-10-19: supplied reconstructed-JND CSV, single frozen B assessment

Owner-authorized label overlay on AIC2026 public PTC crops: 5,768 pairs/42
references; primary curve labels cover all pairs and pointwise labels cover
2,050 pairs. The exact CSV and all 5,810 original RGB8 PNG hashes are bound in
ADMISSION.json. One named-profile native scoring run used frozen B `89cb4861…`,
source `8d6aa6cc`, with no head, fit or JND calibration. All predictions are finite.
Canonical signed quality SROCC is 0.9112672374317565 for curve labels and
0.8972082446190002 for pointwise labels; target orientation is −JND. This is one
assessment, without a claim of an untouched independent final test or propagated
observer-label uncertainty. Earlier objective-only release descriptions remain
historical; no earlier raw artifact or target was replaced.
[Results, native statistical ownership and pointer](../benchmarks/jnd_reconstruction_b_2026-10-19.pointer.md).


## 2026-10-20: owner-requested reconstructed-JND peer comparison

The owner requested DVIFM, SSIMULACRA2, butteraugli and GMSD comparisons
against the same supplied reconstruction. Exact 5,768 curve/2,050 pointwise
rows, 42 references, saved B predictions and supplied cropped metric columns
were joined by original distorted/reference IDs after all 5,810 image hashes
matched the original admission. All three supplied DVIFM variants are retained.
No pixel rescoring, training, model selection or JND calibration; zero dropped
rows. This extends the prior single-model exposure with explicit peer
assessment, without changing the historical first-read scope.
[Matched results and provenance](../benchmarks/jnd_reconstruction_b_2026-10-19.pointer.md).


## 2026-10-20: owner-selected reproduced Rust baseline tables

The owner subsequently selected retained reproduced metric TSVs for this
comparison. Exact rows, source/destination pixel identities and saved B
predictions remain unchanged. SSIMULACRA2, GMSD, butteraugli p-norm-3 and
butteraugli max are measured; DVIFM 0.3-rs is missing because the selected
directory contains no DVIFM table. No old DVIFM column is substituted.
No new pixel scoring or fitting.
[Results and bindings](../benchmarks/jnd_reconstruction_b_2026-10-19.pointer.md).


## 2026-10-20: native DVIFM 0.3-rs comparison completed

Following the owner's DVIFM0.3-rs request and reproduced-score directory
selection, one fresh native `port-lap-v0.3` batch (dvifmish `31c4e256`)
scored the exact5,768pairs/42references. All raw and JND scores exactly
match retained Rust `f24a08a` results. Native curve/pointwise SROCC is
0.95707746508388 /0.9330864427036173; no failures or drops. Frozen B and
other reproduced Rust metrics retain their prior predictions and populations.
No fitting, parameter selection or JND calibration. The frozen preset
provenance states its author constants were trained on AIC2026; this
comparison is not an unseen-corpus test for DVIFM. Original target and
image hashes remain bound.
[Complete results and runtime bindings](../benchmarks/jnd_reconstruction_b_2026-10-19.pointer.md).
