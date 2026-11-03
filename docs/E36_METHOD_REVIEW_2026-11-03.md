# E36 program: method review and a revised plan (2026-11-03)

Owner, 2026-11-03: "revisit the logic and experiments and zenpapers concepts leveraged so far in this process and
come up with a better set of experiments and methodology - question everything including prior directives."
Record under review: `benchmarks/e36_subset_test_2026-11-01.md` (experiments 1-6, amendments 1-7). Methods consulted:
zenpapers `PSYCHOVISUAL_LEARNINGS_FOR_ZENSIM.md` §A, `docs/iqa-methods/evaluation-statistics.md`,
`docs/jpeg-aic/metric-accuracy-and-jnd-mapping.md`, `docs/iqa-evaluation-novelty-2026-10-19.md`, and the library
catalogue. AIC2026 numbers stay in the private training repo.

## What the program established

- A within-ladder hinge on metric-consensus pairs (p30L) and a block-map term toward butteraugli (map 300) each move
  every registered measure in the intended direction, with G-STEER and G-CHROMA-L intact, and the map term's gain
  shows up on judges it did not train toward (SSIMULACRA2, CVVDP) in the JPEG XL loop.
- The human MOS mix matters more than any loss term: restricting the human leg to TID2013 + MDID lowered error against
  held-out human ratings by about 0.5 points at every seed, while the full KADID-dominated mix was the worst
  configuration on the collections it did not train on.
- Seed noise is of the same order as most effects (control spread 0.33 points on all-nine error; effects 0.3-0.8).
  The walk sampler cuts that spread about threefold at no cost.
- The private out-of-fold AIC2026 check moves the other way from every in-house measure for the map term, and not for
  the mix. The program's measures and the near-threshold JND regime disagree.

## What is wrong or weak, in order of consequence

1. **The truth is mostly other metrics.** The teacher legs are SSIMULACRA2; the E36 "primary" counts inversions of
   pairs that butteraugli and CVVDP agree on; the map term's target is butteraugli; three of the five loop judges are
   butteraugli, CVVDP and zensim itself. A gain on the primary or the butteraugli judge is partly a gain in agreeing
   with the teachers. Only the EVAL MOS sets, the SSIMULACRA2 and DVIFM loop judges, and AIC2026 are independent of the
   training signal, and only AIC2026 and the loop reflect the product's use.
2. **KADID and TID2013 carry 32x weight but are the wrong distortions.** 7,869 of the 10,590 human fit rows are KADID
   (synthetic blur, noise, colour, brightness). zenpapers' corpus hierarchy lists KADID/TID as integrity checks, "not
   for ranking", and DVIFM's authors fit only their JPEG and JPEG 2000 subsets. Experiment 5's result (every single
   source beats the full mix on untrained collections) is what that predicts. The D1 four-source decision was about
   licence and role, not about which rows deserve weight.
3. **The error-against-human readout is weaker than the data allows.** It fits a 4-parameter logistic per collection on
   the EVAL rows themselves (ITU-T P.1401's convention, so the residuals are in-sample), normalises by the target's 1-99% span (so a
   hard collection and an easy one count the same), takes a median (tails invisible), averages 48-row and 480-row
   collections equally, and carries no interval. KADID and TID2013 ship per-stimulus standard deviations, so Z-RMSE and
   the outlier ratio (the papers' decisive statistics) are computable on them and were never used. No comparison in the
   record has a paired, source-clustered interval except the AIC2026 one.
4. **EVAL is now selection data.** It chose the MOS mix, the map weight and the arm order across about forty readouts
   and seven amendments. The "three untrained collections" (IQA-ODI 48 rows, NITS-IQA 405, VCL-FER 192) were never
   registered as a confirmation set and two are small or off-domain (omnidirectional).
5. **G-STEER's pass/fail is brittle and partly off-target.** 135/135 at every seed hinges on single cases missing by
   0.004 (m2 0.986 vs 0.99). The novelty review already states m2 is a local-linearization diagnostic, not a steering
   ceiling. The mix arm "failing" and map 300 "passing" is a difference of one case per seed; the loop, which the owner
   ruled decisive, favoured the mix arm on every judge.
6. **Experiment 5 confounded source identity with weighting.** "Equal share of 36.035" gives a TID2013 row about four
   times the per-row weight a KADID row had, so "TID2013 transfers best" may mean "fewer, heavier rows transfer best".
   The experiment also used full retrains (20 minutes each) where a frozen-feature linear probe answers the transfer
   question in seconds, as the 2026-10-21 AIC-4 study did.
7. **Three seeds per arm, uniform sampler.** With the walk sampler available, every arm after experiment 6 should have
   used it; the comparison would have been about three times sharper for the same compute.
8. **The continuation's AIC2026 leg (x256) was never ablated.** Its own effect, and whether it fights the map term,
   is unknown.

## Revised methodology (register, then run)

**M1. Readout.** Replace the median-residual readout with the papers' panel per collection: SROCC, KROCC, PLCC after the
P.1401 4PL mapping, RMSE on the collection's own scale, and Z-RMSE and OR where per-stimulus sigma
exists (KADID, TID2013, and any library set with `std`). Aggregate with sources as the resampling unit (pigeonhole
bootstrap over source x codec where codecs exist) and report every arm-vs-control difference as a paired interval.
Weight collections by their row count or report them separately; never a flat mean over collections.

**M2. One registered confirmation set, read once.** Compression-centric human sets in the library that no training
leg or EVAL touches: NNCD-IQA (320 MOS, learned codecs + JPEG 2000; the set DVIFM's authors used), MIDD-CAP and
MIDD-GAR (near-lossless identification rates), MMSPG-JPEG-XR (186 MOS), CLIC 2021's 315 validation crops (2AFC),
CIQA (2AFC between JPEG qualities), mucped23 (2AFC, jpegli), Fidelity-LIC (learned codecs), and the threshold sets
JND-MID-MA and VLIC Cambridge. `docs/DATA_SPLITS.md` limits what each can claim: NNCD and both MIDD sets share every
source photo with TID2013, a training source, so they are source-exposed diagnostics; mucped23 uses CID22's
references; CLIC 2021's validation crops and CIQA are source-independent; CSIQ, LIVE release 2 and KonJND JPEG stay
reserved for the Rev4 confirmation and CLIC 2022 is TERMINAL. Screen the candidates with third-party metrics first
(which labels are reliable and agree with SSIMULACRA2 and its peers), then register one set and its statistic and
read it once for the final two candidates and p30L, after everything else.

**M3. Human training rows by distortion relevance, not by dataset.** Build the human leg from the compression rows of
KADID (09 JPEG 2000, 10 JPEG), TID2013 (10 JPEG, 11 JPEG 2000, 21 lossy compression of noisy images), KonFiG-IQA
(JPEG 2000 only; its other six types are synthetic) and all of CID22-A25, beside the dataset legs; keep the synthetic
distortions as a small integrity leg rather than the bulk. Compare at equal total weight, so weighting and identity
are separated.

**M4. Cheap screens before retrains.** For any "which data helps" question, fit ridge or 4PL heads on the frozen F3
features per source and score on the others (minutes), then retrain only the top two configurations.

**M5. Walk sampler for every experimental arm; three seeds remain; report seed spread beside every mean.**

**M6. G-STEER as a graded measure.** Keep the 135-case packet but report the paired distribution of m3f and m2 against
the control with an interval, and treat "135/135" as a flag to inspect, not a ship bar. The JPEG XL loop stays the
decision for steering, per the 2026-10-25 ruling.

**M7. Judges in the loop.** Add SSIMULACRA2-only and DVIFM-only decision columns; a steering change counts only if the
judges that did not train it agree. Report butteraugli and CVVDP as diagnostics when the arm trained toward them.

## Experiments, in priority order

1. **Ablate the AIC2026 leg** in p30L's continuation (weight 0 and x64 beside x256), measured by M1 and the private
   out-of-fold check. Cheap; it may explain the near-threshold cost.
2. **Human leg by distortion relevance (M3)**, three arms (compression rows only; compression rows + integrity leg;
   current mix as control), walk sampler, three seeds, M1 readout.
3. **Quality-band-gated map term:** apply the map term only to E36 views whose quality band is at or below 60 (the
   pool carries the band), weights 100 and 300; the hypothesis is that butteraugli's block map misleads near threshold.
4. **A second spatial target:** DVIFM 0.3's painted map (JND units, 5x5 blocks, an imazen port exists in dvifmish) as
   the map-term target, alone and as a consensus with butteraugli; judged in the loop by SSIMULACRA2 and CVVDP.
5. **Linear-probe transfer matrix (M4)** over every human source, including the compression subsets, to replace
   experiment 5's seed-0 single-arm table with intervals.
6. **Confirmation read (M2)** for the final two candidates.

## Directives questioned

- "The JPEG XL loop decides" holds, but the loop's judges must exclude the training targets for the decision column.
- "Datasets only from the tower library" holds; the library already has five untouched compression-centric sets.
- "Scale 0 mandatory", "no NEON", "p30L for now" are untouched by this review.
- The D1 human-role decision (four sources) does not say the sources' synthetic-distortion rows must carry the human
  leg's weight; M3 stays within it.

## Corrections (2026-11-03, after the review was committed)

KADID-10k's compression types are 09 (JPEG 2000) and 10 (JPEG), not 07 and 08 (DeepFL-IQA, arXiv:2001.08113, lists
the types). KonFiG-IQA's types are colour diffusion, high sharpen, jitter, JPEG 2000, lens blur, motion blur and
multiplicative noise, so only its JPEG 2000 rows are compression. P.1401 fits the mapping on the evaluated data
itself; the review had said the papers fit it on training data. M2 now carries `docs/DATA_SPLITS.md`'s exposure
limits for each candidate set.
