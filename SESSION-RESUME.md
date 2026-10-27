# Start here — one target score, one development path

## Dataset naming for the current October 27 comparison

**AIC-2026 public_train is not AIC-3.** The current Mlic/Neff/Vote/Order
continuations use D6-admitted AIC-2026 public_train: 3,920 PTC crops from
28 sources. Fold results split those TRAIN sources into 24 fitted and four
held-out sources. “Held-out” there means out-of-fold within public_train,
not AIC-3, AIC-2026 public_test, or an independent final evaluation.
Full-data results fit all 28 sources and are explicitly in sample, as the
owner requested. Do not present their correlation or error as test accuracy.

AIC-3 CTC is a separate 600-pair, 10-reference corpus and retains its holdout
role. This continuation study does not read it. Historical exposure recorded
in DATA_SPLITS remains recorded; “holdout role” is not a claim that AIC-3 has
never been inspected. AIC-2026 public_test (14 sources), AIC-4 and SDR25 are
also outside this new training/comparison read scope. Public_test's earlier
one-read exposure remains spent; these results do not reopen it.

Use **AIC-2026 public_train** in current result prose and tables. Frozen raw
artifacts retain their internal `aic` population key and historical filenames;
for this registered study that key maps to AIC-2026 public_train, with table
SHA256 `a2969c70f2e1c96f18348b0b8e66a28e41ee87ad839636e8329e63408b138b71`.
Do not rename or rewrite frozen artifacts to fix display terminology.

## Current work — October 27, 2026

This section and the October 26–27 owner updates in
[the production plan](docs/PRODUCTION_PRIORITIES_2026-09-15.md) control the active
work. The September chronology below is retained history, including its old
paused-goal statement. It is not the current queue. No new Mlic/AIC recipe has
been adopted as the serving model, and the seamless HDR frontend has not started.

### State at 19:56 UTC, 2026-10-27

Stage D is complete (control byte-identical): ×256 + the within-dataset rehearsal leg keeps the older sets in sample
through e59 at out-of-fold AIC 0.9390, 0.0021 above the Vote point (0.9369), so E21 out of sample decides
(byconstruction record, "Stage D: result"). ×1024 + rehearsal reaches 0.9458 but keeps at no checkpoint. Stage E
(Vote with the rehearsal leg at ×256 and ×1024) is registered (`e68f7d4a`) and planned (`5313ca25`). Dev queue, each
run waiting for its predecessor and taking `heavy.lock`: Mlic ×256 omission `mlic-aic-omission-aic256-…` (w21:pF,
running since 19:55) → Vote ×256 omission `mlic-aic-omission-voteaic256-…` (w21:pG) → D candidate omission
`mlic-aic-omission-xrefaic256-20261027-run01` (w21:p9) → stage E `mlic-aic-recipe-votexref-20261027-run01` (w21:pA).
Then: `~/tmp/claude-port-20261027/staged_summary.py /mnt/v/output/zensim/mlic-aic-omission-xrefaic256-20261027-run01`
settles D (copy in `/mnt/v/output/zensim/frontier-summary-20261027/`); `stagee_summary.py <E run> [<E omission run>]`
settles E (an E omission run is frozen only if E's candidate beats stage C's point by more than 0.002); then freeze
stage C for the final point with `stagec_launch.py` (add `--omission` only for a point without a completed omission
run), then near-identity and gates for its packed seed-0 model.

### State at 19:10 UTC, 2026-10-27

Stage B is complete (every recipe's replay control byte-identical). Under the registered B→C choice, **Vote ×256
epoch 29 (out-of-fold AIC 0.9369, keeps the older sets) is stage C's arm**; Order 0.9321 and Confidence 0.9320 sit on
Mlic's curve, Neff below (`/mnt/v/output/zensim/frontier-summary-20261027/STAGEB_SUMMARY.json`). The stage C recipe
variant (Vote's own full models, root and omission cells) is registered (`a29bb316`) and pushed (`667cea58`). Dev
queue: stage D `mlic-aic-xref-20261027-run01` fitting (w21:pE) → Mlic ×256 omission `mlic-aic-omission-aic256-…`
(w21:pF) → Vote ×256 omission `mlic-aic-omission-voteaic256-…` (w21:pG). Then: run
`~/tmp/claude-port-20261027/staged_summary.py <D omission run>` (D is judged against the Vote point; freeze a D
omission run first only if D's candidate beats 0.9369 by more than 0.002), then freeze stage C's `seeds-plan` and
`full-plan` for the chosen point with `stagec_launch.py`, then near-identity (`just dataset-nearid-candidate`) and
gates (`just candidate-gates-rev7`) for its packed seed-0 model.

### State at 16:10 UTC, 2026-10-27

Frontier: stage A2 done (no recovery point keeps the older sets); the registered rule over A+A2 advances
`MixedAIC256@e7` (out-of-fold AIC 0.9312; summary in `/mnt/v/output/zensim/frontier-summary-20261027/`). The dev
queue is chained (each run waits for its predecessor, then takes `heavy.lock`): stage B
`mlic-aic-recipe-{order,confidence,neff,vote}-aic256-20261027-run01` (herdr w21:p9/pA/pB/pC; order fitting since
16:03Z), stage D `mlic-aic-xref-20261027-run01` (w21:pE), stage C step 2 for the advanced arm
`mlic-aic-omission-aic256-20261027-run01` (w21:pF). Trainers: stage B `587e4764…` (main `216481a8`), stage D and
omission `b3b86f18…` (main `8b66cca2`), build records in each run's `TRAINER_BUILD.json`. `just mlic-priority-progress`
lists every run. After stage B: `~/tmp/claude-port-20261027/stageb_summary.py` (copy in the summary folder) applies the
registered B→C choice. After stage D and the omission run: compare D's candidate on E21 out of sample (needs a D
omission run for its arm) before freezing stage C's `seeds-plan`/`full-plan` for the chosen point.

### State at 15:25 UTC, 2026-10-27

Frontier stage A completed at 15:05Z (byconstruction record, "AIC-share frontier stage A:
result"): ×256 epoch 7 advances under the registered rule (out-of-fold AIC 0.9312); no
static mixture reaches 0.95 while keeping the older sets. Stage A2 recovery
(`mlic-aic-recovery-20261027-run01`, herdr w21:pD) is fitting since 15:06Z; its points join
the frontier. Stage B is ready: native Order/Confidence continuation admission
(`d7cad238`), coordinator `recipe-plan` (`216481a8`), trainer `587e4764…` built on i265
from `216481a8` (record and binaries in `~/tmp/claude-port-20261027/stageb/`), dry-run
plans for all four recipes and real-input admission smokes for Order and Confidence
(epoch 0 trained) in `~/tmp/claude-port-20261027/{dryrun-stageb,stageb-smoke}/`. After A2
completes: apply the rule (`~/tmp/claude-port-20261027/frontier_summary.py`), freeze one
`recipe-plan` per recipe at the advanced budget (chained by `--after-run`; Order,
Confidence, Neff, Vote), launch each `fold-queue` in an idle w21 shell pane.

### State at 14:10 UTC, 2026-10-27

All 15 seed-0 candidates pass the label-free gates (dial C1-C6 both grids, identity
at 12 SIMD permutations; G-STEER 129-135). D1 recommendation to the owner:
MlicVoteFT16x8 as the interim revision-7 default (alternative MlicFT16x8). Owner
directive 13:05Z: reach 0.95 on AIC-2026 while recovering the older sets. The
registered AIC-share frontier is running: stage A (`mlic-aic-share-frontier-20261027-run01`,
fitting since 14:07Z) and stage A2 recovery (`mlic-aic-recovery-20261027-run01`,
queued behind it). `just mlic-priority-progress` shows both.

### State at 11:45 UTC, 2026-10-27

The confidence follow-through finished at 11:25 UTC without failures; confidence
weighting does not improve on Mlic (byconstruction record, "MlicConfidence fold
and full-data results"). All 15 seed-0 candidates pass near-identity. The
matched-continuation diagnostic took `heavy.lock` at 11:26 UTC (mixed arm done,
AIC-only arm fitting); the gates queue (now 15 models) and the source-fold study
follow it. Native HDR controls and the f64 precision probe are on main
(`just hdr-precision`). Reports: the dev.lan session report (addenda 11:22 and
11:28 UTC, archived under the priority directory's `reports/`) and the
"AIC2026 monotone push" dashboard, both refreshed at 11:28 UTC. D1 waits for the
gates; the runtime-hardware and HDR-holdout decisions wait for the owner.

### Ownership and queue — 2026-10-27 09:21 UTC

The owner returned all zensim work to the original Claude session
(f9f1f552, herdr w21:p1) at 08:51 UTC; checkpoint tag
`zensim-claude-handoff-20261027`. Do not restart aic-finetune. Codex's
app-server is the parent of the running confidence followthrough: do not
stop it. The confidence run02 control matched all 7,869 original
predictions exactly; its 21 omission cells started at 08:53 UTC.

The registered matched-continuation diagnostic (mixed+AIC16 versus AIC-only
continuation of original Mlic full seed 0, 60 epochs) is implemented on main
(`fb3b0a79`) and queued at
`/mnt/v/output/zensim/mlic-matched-continuation-20261027-run01`. Its queue
waits, holding no lock, for the confidence followthrough to complete, then
takes `heavy.lock`. It runs in herdr tab `matched-continuation-queue`.
After it: label-free gates for the 12 seed-0 four-recipe candidates plus the
three seed-0 MlicConfidence models (`mlic-candidate-gates-20261027-run01`,
`QUEUE3.sh`, tab `candidate-gates-queue`), then the
registered source-fold study (`mlic-matched-continuation-folds-20261027-run01`,
tab `matched-folds-queue`; 42 continuations, out-of-fold AIC panels). Each
waiter holds no lock until its predecessor ends. `just mlic-priority-progress`
includes all three. Details, controls and hashes: the byconstruction record's
"Matched-continuation admission implemented and queued" and "Prospective
source-fold matched continuation" sections.

### Latest recovery — 2026-10-27 06:45 UTC

Active full-data root is now `/mnt/v/output/zensim/mlic-aic-full-20261027-run03`;
plan SHA256 `4684cbb8cd653f5409e292a4c6233ac74a4e797aa696f55be6a64a273a0fa4da`,
coordinator `d84cfdb235aee5d4e2f507235a9e2fcc324e461a`. Its
`CONTROL_VERIFIED.json` verifies the preserved full-budget run02 control using
exact native model bytes excluding only run-specific `zentrain.repro`, plus
matching recipe/admission/result fields. No numerical tolerance was introduced.
Matched schedules are complete. Six Neff/Vote seed trainers were verified live
in the single 24-GiB/6-job scope; Order's three seeds follow. Full coordinator
PID 2634178/start ticks 230811185; lock holder 2634157. These are dated
observations: re-resolve with `just mlic-priority-progress` before acting.

The existing full-panel source now waits on run03. Confidence's active queue is
`/mnt/v/output/zensim/mlic-confidence-20261027-run02`, with unchanged source
`d77c968934d08d34363ada295e04a6920ef33289`, trainer and corrected manifest index.
Its `QUEUED.json` binds the new full-panel dependency. It remains 0/21 and
requires its own full-budget unweighted control. Old full run02 and confidence
run01 are preserved failures; do not restart them. Completed fold evidence,
scientific inputs, strengths, seeds and all historical verdicts are unchanged.

### Full confidence manifest preparation — October 27, 2026

Three full-data manifests are verified under the shared admission run02
`full-contrast-bindings-run01/`. Source `2f9268f2`; manifest index SHA256
`d73a3422de00f6150f50e934dc6821ab7894f6eb85fe4a3e732fefbacc088bfc`;
original full-authority index SHA256
`7dbf9cb8deabdfc4b42b72637dd17112a41281e01094a5fe65ae987525f58923`.
`SOURCE.py`, `READ_EVENT.json` and `VERIFIED.json` preserve source/exposure and
all three seed bindings. Each full seed covers 8,339/10,590 human rows,
210/210 LIVE-MD and 320/320 Colourlab rows; uncovered references retain the
original loss. Complete-reference coverage checks pass. The 21 omission
manifests are byte-unchanged. Full native normalizers and fits have **not** run.

Eight binding/wrapper/dependency tests pass on i265 using the existing
confidence-admission Python environment (system Python lacks pyarrow).
Generator run-heavy: rc=0, 3s, peak RSS 0.36 GiB, minimum available 24,006 MiB,
peak load 0.06. This prepares the original full-data confidence fits; their
launcher must validate the separate full registry, use child-only native
confidence environment variables, verify normalizer receipts and final119,
then clear the confidence factors for the common AIC continuation. Do not feed
the full registry to the frozen 21-cell omission queue or modify that queue.
All full-fit/AIC steps retain the original seeds, strengths and panel policy.

### Confidence full-fit launcher — October 27, 2026

Committed owner `88155360` adds `just confidence-full-fit <plan>`. It requires
completed paired omission evidence and frozen source/binary/registry hashes;
it does not exclude confidence based on the historical SROCC verdict. It
launches all three original full seeds through the existing strict owner,
checks final119/packing and native normalizer receipts, then certifies each
packed model. Confidence environment variables are child-only, and inherited
ones are rejected. It does not perform common AIC continuation or promotion.

Four boundary tests pass on i265: complete evidence despite historical
rejection, missing plan pins, full-versus-omission/swapped-seed rejection,
and child environment/receipt isolation. Run-heavy rc=0, 1s, peak RSS
0.07 GiB, minimum available 24,285 MiB, peak load 0.21. The actual three full
manifests also pass the launch consumer's checks: rc=0, 0s, peak RSS 0.07 GiB,
minimum available 24,291 MiB, peak load 0.23. This is preparation, not an
executed full-confidence fit. A plan with real omission completion hashes
cannot be frozen until that study completes. The next implementation step is
the separate confidence AIC continuation/panel extension, preserving the
running four-recipe study and its frozen sources.

### Separate confidence AIC fold extension — registered October 27, 2026

The new `aic-confidence-adaptation-plan-v1` contract contains exactly Mlic and
MlicConfidence. It requires 48 continuation cells: 24 hash-verified original
Mlic reuses and 24 confidence fits (four original omissions × three seeds ×
two strengths). Its panel includes the 24 initial models plus 48 continuations,
72 total. The original four-arm contract still requires its exact four recipes,
96 continuations and 144 panel models. Neither contract can substitute the
other's smaller/larger inventory. Running source snapshots are unchanged.

The confidence extension reuses the same zero-step prediction parity,
strict native warm-start owner, original Mlic root/losses, fresh Adam at 1e-5,
50,000 draws/epoch, 8×4 and 16×8 strengths, source-fold rotation and final-epoch
checks. Only the starting model differs. It verifies each confidence manifest
and native normalizer receipt against the completed omission record and pinned
baseline registry. Inherited native or wrapper confidence environment variables
are rejected before any continuation or panel read; the common recipe cannot
accidentally retain its starting model's confidence factor.

The same panel owner preserves fitted/held-out separation, native statistics,
source/codec/ladder diagnostics, seed contrasts, whole-source bootstrap and
both mixture tests. It compares confidence against its own initialization and
matched Mlic continuation, with no SROCC selector. No extension plan is queued
until real confidence omission completion and artifact hashes exist. The
separate full-confidence AIC continuation/panel extension remains required.

All 28 AIC orchestration/panel tests pass on i265, including rejection of mixed
protocol inventories, an incomplete original study masquerading as an extension,
and inherited confidence factors. Run-heavy rc=0, 0s, peak RSS 0.12 GiB,
minimum available 24,299 MiB, peak load 0.12. This validates orchestration
boundaries; confidence adaptation has not run yet.

### Separate confidence AIC full-data extension — registered October 27, 2026

`aic-confidence-full-plan-v1` is exactly Mlic plus MlicConfidence. It requires
six initial models and twelve continuations: the three original Mlic full
models, three completed independent confidence full fits, and both common AIC
strengths for all seeds. The six Mlic continuations are reused from the complete
four-arm full-data study, with their completion-plan identity, three-seed
inventory, recipe/strength, result/input hashes and final model checks. Only
six new confidence continuations are needed. No best-seed selection or SROCC
eligibility cutoff is introduced.

Confidence initial reuse requires its completed full-fit plan, every pinned
prerequisite, exact recipe/final119/packed identity and per-seed manifest/native
normalizer receipt. It does not refit the base model inside the continuation
owner. Common continuation rejects inherited confidence factors and keeps the
same original Mlic root, fresh Adam, fixed budgets and zero-step prediction
identity controls. Order matching is performed only in the original four-arm
protocol. The original protocol remains 12 initial models plus 24 continuations;
its smaller confidence counterpart cannot satisfy those counts.

The same full panel owner covers 18 models × two representations × five
populations = 180 prediction outputs, plus both mixture checks and the existing
statistics/source/codec/seed/whole-source uncertainty panels. The original
four-arm panel remains 36 models / 360 predictions. Serving-f16 versus final-f32
still includes calibration and is not isolated packing error. Running plans,
source snapshots and results remain immutable. No confidence extension is yet
queued: exact completed omission/full-fit hashes must exist first.

All 32 AIC orchestration/panel controls pass on i265, including refusal of
missing/duplicate seeds, mismatched protocol inventories, wrong reuse strength,
changed reuse evidence and partial full-panel completion. Run-heavy rc=0, 0s,
peak RSS 0.12 GiB, minimum available 24,296 MiB, peak load 0.16. These are
orchestration controls, not evidence of completed confidence fits or adoption.

### Attached confidence followthrough supervisor — October 27, 2026

`confidence_followthrough.py` freezes its own source/config before waiting for
the independent confidence omission study. It checks the live PID/start time
while waiting, refuses failed/exited prerequisites, and verifies actual complete
paired evidence and source/trainer identities before constructing plans. A
historical rejection does not stop the new AIC comparison. Missing future
completion hashes are never invented: they are bound only when the corresponding
stage completes, under the already registered recipes and budgets.

Execution is serial: 48 matched fold continuation cells (24 Mlic reuses),
72-model fitted/held-out panels, three original full-confidence fits, twelve
full continuations (six Mlic reuses), then 180 full-panel prediction outputs
and both mixture tests. Each existing heavy owner acquires the shared lock and
run-heavy scope; the waiting supervisor holds neither. All commands/logs and
terminal failures are durable in its run directory. No detached agent or
background daemon is created. No automatic model adoption occurs.

Four supervisor tests pass on i265: live-wait versus PID reuse/death,
historical rejection with complete evidence, unchanged common fold recipe,
and correct full-control reuse/budgets. Run-heavy rc=0, 0s, peak RSS 0.07 GiB,
minimum available 24,297 MiB, peak load 0.03. Full native execution remains
pending the original AIC and confidence omission dependencies.

### Live serial confidence followthrough — October 27, 2026, 07:09 UTC

The attached supervisor is running at
`/mnt/v/output/zensim/mlic-confidence-followthrough-20261027-run01`.
Frozen source `5474e6b4d99ad432830e2d89fca900a6eec5f97c`; CONFIG.json SHA256
`60cb9ed63f18ef6e8f0af6f27adbac3fb1e8f78f63c77a5c6ec101279eb557a3`.
Verified worker PID 2703508/start ticks 230955857 waits for confidence run02's
complete omission evidence, without taking the compute lock. It then runs the
registered fold fits/panels, full confidence fits and full AIC fits/panels in
sequence. `just mlic-priority-progress` includes its live status. Read its
STATE.json/CURRENT.json and actual process tree before any intervention; never
launch duplicate stages while this supervisor is live. Its source snapshots,
config, stage plans and logs remain preserved on failure.

Five supervisor tests pass, including real process status and stale PID reuse:
run-heavy rc=0, 0s, peak RSS 0.07 GiB, minimum available 24,298 MiB, peak load
0.02 on i265. No confidence fit has started; the supervisor waits behind the
original AIC full pipeline and confidence omission queue. As of this observation,
all six original full trainers were advancing at logged epochs 72–88/120,
with no reported failure. This timestamped observation does not replace a live
check. No candidate is adopted and the production/HDR objective remains active.

### Active goal and decisions

Keep working toward a stable, production-qualified candidate while preserving
speed, scientific integrity and frozen studies. A completed experiment,
negative result, report or PR is not completion. Execute the next registered
TRAIN-only step; do not tune on EVAL. Test effective-sample masses, native
vote likelihood, matched ordering and contrast-confidence weighting separately.
After stable adoption, implement and qualify seamless native-precision HDR
that improves on PU21 while maintaining speed. Five hours is not a stopping
condition. Continue independent work when one path is blocked.

The owner's latest decision is to compare which recipes fine-tune best on
AIC2026 public_train, including in-sample panels, using **complete panels and
other statistics, without strict SROCC rejection or an automatic scalar winner**.
Earlier study verdicts remain immutable and do not exclude Neff/Vote/Order.
Source roles, numerical/pixel correctness, provenance and existing production
requirements remain binding. Confidence-map validity is an input-admission
requirement, not a candidate-selection cutoff. See the prospective
[AIC protocol](benchmarks/byconstruction_2026-10-22.md#aic-adaptation-across-training-recipes-prospective-panel-study-2026-10-27).

### Verify live work before doing anything else

**06:41 UTC failure and diagnosis:** full run02 finished its 120-epoch control,
but the newly written raw-file hash check rejected different embedded run
provenance. Both final-f32 and production-f16 models are byte-identical after
the existing native owner removes only `zentrain.repro` from diagnostic copies;
all other result fields match except output paths/model hashes. Original files
are preserved. Full panels and confidence run01 refused their failed dependency
before fitting. No worker from those failed queues should be called running.
The repaired comparison follows the established E33 control method: exact
non-provenance model bytes plus matching full recipe/admission/result fields.
It retains all numerical/serving metadata, including calibration. Run02 remains
failed historical evidence; use a fresh pinned recovery plan, binding the
completed control artifacts, instead of rerunning or overwriting its outputs.

**Earlier verified observation — 2026-10-27 06:38:03 UTC:** AIC fold run03
has completed **96/96** cells (24 verified reuses plus 72 new fits), with no
reported failures. Fold panels, including both mixture checks, are complete for **144/144** models.
Full-data training is running its unweighted control; compute-lock holder PID
2521269 belongs to that worker ancestry. Trainer PID 2521663, start ticks
230573015, was live at logged epoch 110/120 (zero-based final epoch 119).
Full panels wait behind full fits. This is progress, not control completion.
Re-resolve these observations with the command below before acting.

Confidence admission, all eight linking scopes and all 28 retained native
rating maps are complete. Exact-scope quality responses and teacher values
have been read with recorded exposure. Native normalizer and loaded-row
integration tests pass, but confidence candidate fits remain **0/21**.
The current machine-readable owner is `CONFIDENCE_PROGRESS.json` in the shared
admission run02 root; the screen coordinator alone reports an earlier stage.

**Do not train from `contrast-bindings-run02`.** Its 21 manifests bind rows
but incorrectly declare logical width 410. The original recipe has 1,853 raw
inputs plus 410 registered derived products: logical width **2,263**, with
410 direct and 410 derived selected inputs. This was caught before any
confidence training. Preserve run02. Corrected run03 is complete for all 21 manifests, index SHA256
`9dd58d78711449b9c699165ae8651ae632010acfd8875bed4cc4affdec73ad16`,
source `482485e8`. The release trainer is built and the queue below owns the remaining native
receipts, full-budget unweighted byte-parity control, and all 21 fixed-epoch
TRAIN cells. Maintain the live AIC queue's priority. Do not change frozen
inputs, sampler/tie behavior, loss definitions or historical verdicts.


Observed on dev at **2026-10-27 04:55:32 UTC**: Order had completed **21/21**
cells and both mixture checks. Its full-budget unweighted control was
byte-identical on all 7,869 predictions. The replacement confidence screen
was running after successful native decoding; four replacement AIC stages
were queued with no reported failure. This is a dated
observation, not a future claim that the processes are still alive.

Run `TMPDIR="$HOME/tmp" JUST_TEMPDIR="$HOME/tmp" just mlic-priority-progress`.
This refreshes the existing per-stage status artifacts, including PID/start-time
checks, actual completed outputs, failures and compute-lock ownership. Read the
plan and queue receipt before restarting anything. A stale `RUN_STATE.json`, a
log, or an agent's “working” label is insufficient. Never duplicate a live queue
or bypass its lock. Preserve failed outputs and make repairs in a fresh run
directory when frozen orchestration changes.

| Order | Run directory, checked on dev | Required work / state at the observation |
|---|---|---|
| 1 | `/mnt/v/output/zensim/mlic-order-20261027-run02` | 21/21 omission fits complete; historical screen rejected Order, still eligible for AIC. Source `27ed3745`; trainer `441bda58…`. |
| 2 | `/mnt/tower/output/zensim/mlic-confidence-admission-20261027-run02` | Source screen/admission, eight linking scopes and 28 native maps complete; native contrast integration validated; corrected run03 manifests and live queue; 0/21 candidate fits. |
| 3 | `/mnt/v/output/zensim/mlic-aic-adaptation-20261027-run03` | 96/96 fold continuations complete: 72 new + 24 identity-checked Mlic reuses. |
| 4 | Same fold root, `panels/` | 144/144 model panels and both mixture checks complete; separate fitted/held-out source panels. |
| 5 | `/mnt/v/output/zensim/mlic-aic-full-20261027-run02` | Full-budget control running; 9 new base fits + 3 Mlic reuses, then 24 continuations (21 new + 3 reuses). |
| 6 | Same full root, `panels/` | 36 models × 2 representations × 5 populations = 360 prediction outputs, plus both mixture tests. |

The compute lock holder was PID 1832007, an ancestor of the replacement
confidence source-screen worker. Re-resolve it with the status command; do not
act on these old PID numbers. Queues use frozen source snapshots and serialized
resource scopes. Do not rerun from moving main over their frozen paths.

Plan SHA-256 bindings:

- Confidence source screen run02: `5514892579153da37ae638aa30e4194917fdf3bb6282aeb6ca793f9430268fc1`.
- AIC fold run03: `98f72bda706481f885ddd1c295e0ce3ab9c1f38f29d8afd0c59115022a8afa54`.
- AIC full run02: `220bc378937354fc5cc7ca394eda580a343765ba83a39a3c94f471a840c1147a`.

Source owners: fold adaptation `86748f41`, fold panels `3ab3f980`, full-data
Order admission `042c89e0`, full fits `7ec5f55f`, full panels `aa300e1d`.
Their frozen snapshots and complete binary hashes are in each run's plan and
`source/`. The common warm-start source is `87864f94`; its public API remains
under [PR 73](https://github.com/imazen/zensim/pull/73), not merged by this work.
The full Order binary is separate from the running omission binary.

**Do not restart AIC fold run01.** It was cancelled before fitting because the
new caller assumed `last.bin` at zero steps; the frozen trainer exports the
receipt-selected `best.bin` at zero epochs. Run03 retains run02's correction and checks that receipt and exact
prediction parity. Positive-step continuation still requires the final epoch.
The old run01's frozen warm-start source remains a pinned input to run03.

**Do not restart the failed dependency chain.** At 04:52:35 UTC admission
run01 failed on an absent generic packed-reader path; fold run02, its panel,
and full run01 plus panel then refused their failed dependencies before fitting.
All are preserved. Admission run02 uses the unchanged original reader under
its verified hash-suffixed name and coordinator `fc8836f1`, which now requires
that dependency before queueing. Fold run03/full run02 retain identical source,
scientific inputs, recipes and budgets; only dependency/output bindings change.
The new source-screen root also contains the remapped metadata-only
`FREEZE_REQUEST.json`. It is not queued and grants no response permission.

### Recovery checks after an interrupted session

The status reader now reports `active_training` from the verified live process
tree, binding each trainer's PID/start ticks to its output directory, epoch
budget and last logged epoch. It never infers completion from epoch 119 or an
old log. Missing logs and unbound arguments remain explicit. The 24 AIC
orchestration/panel controls passed on i265 on October 27; frozen running
coordinators and scientific inputs were not changed.

First re-run the status command above. For the full control require its
`CONTROL_VERIFIED.json` and exact original final-f32/packed model hashes before
accepting the transition to matched schedules and nine new base fits. Inspect
`CURRENT_FULL.json` and live descendants, then the full-panel owner. Confidence
must wait for that pinned panel dependency; inspect its `QUEUED.json` before
considering a launch. A dead or failed worker needs diagnosis, not a duplicate
queue. The authoritative private recovery archive includes exact active goal,
document hashes, source identities and dated supervision snapshots; the archive
receipt is `DOCUMENTATION_VERIFIED.json` under the shared priority output root.

### What the AIC comparison means

Four initial recipes, each with both 8×4 and 16×8 continuation strengths, use
the same original Mlic inputs/losses, fresh Adam, LR 1e-5 and 50,000 draws per
epoch. Only initial weights differ. Do not carry Neff/Vote/Order factors into
the common continuation. Fold cells preserve the registered 24 fitted/four
held-out source split and seed rotation. Full fits use all 28 AIC TRAIN
sources (3,920 rows); those continuation panels are explicitly in sample.
Fixed seed zero is the assessment model; the three-seed mean is a stability
diagnostic, never best-seed selection.

Reports retain native statistics, source/codec/ladder results, calibration,
tails, rank bias, inversions, seed variation and both mixture tests. Source
bootstrap uses 2,000 whole-source draws, seed 1 and linear quantiles; its
contrast is the role-specific mean prediction and is conditional on fitted
models. Undefined components keep their denominators and never become zero.
AIC `quality` is the released distortion-level index, lower being better,
not a codec-native quality knob ([dataset](https://doi.org/10.18419/DARUS-6156)).

Full reports distinguish `final-f32` and `production-f16-calibrated`.
Their difference includes packing **and** the existing TRAIN output calibration;
it is not isolated f16 precision loss. Full E21 rows are in sample, including
development rows; omission panels remain separate. No EVAL/public_test read is
authorized by these new reports. AIC gains alone do not establish serving,
steering, speed or HDR qualification.

### Confidence execution queue — 2026-10-27 06:13 UTC

`/mnt/v/output/zensim/mlic-confidence-20261027-run01` is queued, with a verified
live attached worker. `QUEUED.json` binds source `d77c968934d08d34363ada295e04a6920ef33289`,
trainer SHA256 `e2ffd0c0279ed31f69c2b6ba4f32656372b9de4359051765a56819a9d3ba3c82`,
and corrected run03 manifest index. It explicitly waits for AIC full run02's
full-panel completion before taking the compute lock. It then runs the original
full-budget unweighted control, all 21 confidence cells and the registered
TRAIN panels/screen through existing owners. No confidence cell has run yet.
`just mlic-priority-progress` now includes this queue. Preserve all old outputs;
this queue is the next action, not a request to launch a duplicate control.

### Confidence weighting: implementation validated, candidate experiment pending

**Latest map update:** all eight scopes / 28 retained native rating maps are
complete. `RATING_MAPS_COMPLETE.json` in the shared admission run02 root has
SHA256 `aefbc62b2aceaed2b648e1d8bd346d2bd5c10dc0fff974fa4287896b6f0b14fb`.
Read ledger SHA `d24dabc1af9d9a575ea1b0430662d302a93f71b1d493556c5b3597eac54b37cd`;
private source owner `5a073e2f`. Native uncertainty reuse was permitted only
for identical complete populations. Mos7 excludes LIVE-MD and Colourlab;
KADID/TID scopes exclude their respective collections. All maps pass the
unchanged input gate. Two KADID and two TID native SEs are zero; preserve them
and apply the registered 0.05-JOD floor at contrast construction. Legacy
unused information diagnostics warned on those zeros, not the map outputs.
Exact legal-pair normalizers and the private native rank-loss multiplier now
pass targeted tests. Corrected run03 manifests are complete. Full-budget original-Mlic unweighted
parity and all 21 TRAIN cells remain required. The sampler owner is
`zensim-validate/src/mlp_train/sampling.rs`; tie semantics are owned by
`zensim-validate/src/mlp_train/subjective_rank.rs`. No confidence-weighted candidate fit or adoption is complete yet.


**2026-10-27 linking update:** all eight omission/full linking scopes are
complete. The shared admission run02 `LINKING_FITS_COMPLETE.json` has SHA256
`486aeae1fcc68dcce5ceb56440ca714179700ac55e32c0086d631bbb34924a2d`.
Seven scopes have four positive units across three studies, sigma_L
0.4442630190242332; mos7 has three units across two studies, sigma_L
0.5263977834858703. JPEG/BPG share observers. No JND threshold was dropped.
Exact-scope responses and teacher values **have now been read**; earlier
label-free observations below remain dated history. Source owners are private
`07cdc78d` then `707ee101` (empty registered donor caller repair). Preserve
run01's four completed scopes and mos7 failure; run02 completes the remaining
four scopes. Both plans and source snapshots are immutable. No native fitter
check was weakened. The subsequent native-map step is complete as recorded above; corrected
contrast manifests, controls and all 21 confidence cells remain required.


**2026-10-27 05:13 UTC update:** the replacement source screen is complete;
AIC fold run03 passed zero-step controls and started new continuation fits.
The 24 original Mlic cells were identity-verified; status was 30/96 after six
new cells completed. Refresh the status command for actual current counts.
Native retained-match review, reviewed membership, independent validation and
the real omission-family graph replay also completed on i265. Use the shared
run02 `FREEZE_REVIEWED_REQUEST.json`, `membership-reviewed-run01`,
`membership-reviewed-validation-run01.json` and
`omission-families-reviewed-run01.json`, not the earlier unexecuted requests.
The exact omission-bound response ledger and reader are the next work; none of
these metadata results grants permission to read quality responses by itself.


Neff changes collection mass; Vote changes native pair likelihood; Order
changes matched-codec sampling. None implements item-confidence weighting.
The latter remains outstanding: source screen → retained-match review → frozen
membership → independent validation → exact omission-bound response ledger →
linking calibration and teacher maps → native contrast weights → controls/fits.
The source screen and mapping stages are complete; continue native integration
and controls while the AIC panel/full-fit queue runs.

Native population/audit records are
`/mnt/v/output/zensim/mlic-confidence-populations-20261027-run03/POPULATIONS.json`
(SHA `e8b80de3006829f87ad8593b819b06d844612419f5eab7b49f2d0c8e379ff078`)
and `/mnt/v/output/zensim/mlic-confidence-native-audit-20261027-run03/NATIVE_UNCERTAINTY.json`
(SHA `a9e65891ab7bfa7e5520dc0791179b14f577bbce8f91576785607919fb209e58`).
They cover 6,659 KADID, 1,680 TID, 210 LIVE-MD and 320 Colourlab observations;
they are not fitted linking maps or permission to read new responses. Rejected
run02 collapsed four distinct KADID observations sharing pixel identity;
run03 preserves native `source_row_id` as well as `pair_key`.

The original omission identities now pass on real metadata in
`/mnt/tower/output/zensim/mlic-confidence-omissions-20261027-run03/identities/OMISSION_IDENTITIES.json`,
SHA `b9760b0be03fc33997d5bc054285269b77b170c36e16681c520519a47ec71194`.
Source `8369c385` binds all 65 KADID, 25 TID, 14 KonFiG and 25 CID22-A
assessment references, including references absent from fitting, and original
dataset family IDs (200 PIPAL, 111,363 pairs, 38 mos7). Full has no omissions.
Run01 refused collection-local source-row confusion; run02 refused canonical
string family IDs. Both remain preserved. Run03 uses original bank snapshot
keys and manifests, never mutable promoted metadata. No response, feature or
uncertainty values were interpreted. All 129 assessment references match the original native source inventory
by RGB hash and dimensions, with none missing; see the run03
`NATIVE_PIXEL_COVERAGE.json`. The private native-family resolver is implemented
and its six controls pass. The next metadata step is the real graph join
to the independently validated native family registry: reference pixel identity
and dimensions plus all inherited prior-family aliases. Missing or ambiguous
joins must fail before any donor, observer, map or normalizer fit.

Private research owners and exact execution paths are in the separately
committed `zenpapers/docs/mlic-recipe-review-2026-10-26.md` and
`zenpapers/docs/training-confidence.md`. The shared source-screen directory's
`FREEZE_REQUEST.json` is a concrete metadata-only request, **not queued** and
not response-read permission. The prospective JND declaration permits only
TV first-JND linking: 70 TV rows, with 140 GR/IR rows reserved; 35 scene
families across 105 views. No global JND source allowlist is permitted.

Registered contrast precision is
`Vij = si² + sj² + sigmaL² (xi-xj)²`, `si = max(0.05, SE_teacher_i)`,
`wij = (1/Vij) / Z_group,reference`. Compute Z under the exact original legal
conditional pair law, including modulo probabilities and exclusions. Preserve
collection/reference draw mass, RNG, skip behavior and absolute loss. Equal
legal precisions must give exactly one; zero legal pairs retain no update.
Remove omitted source families before any donor/map/normalizer fit. Missing
finite uncertainty, map validity or at least three admitted positive linking
units means INCOMPLETE, never an invented zero floor. The actual native loss
adapter and purpose-bound calibration-response adapter are not implemented yet.

### Completed evidence and next production work

Neff, Vote and Order each completed 21/21 original TRAIN cells and failed
their historical SROCC screens. All remain in the new AIC comparison. Their immutable
roots are `/mnt/v/output/zensim/mlic-neff-20261026` and
`/mnt/v/output/zensim/mlic-vote-20261027-run02`; the study record carries paired
statistics and resource logs. No new default or qualified winner follows.

After panels, assess full tradeoffs and run outstanding production qualification
on the actual serving bytes/API. No benchmarks or unrelated builds run on dev;
it is reserved for zensim training/evaluation. Do not use r7900x. Current builds
and regression checks use i265 after verifying its live work/lock and memory.
All heavy jobs use the shared compute lock and `run-heavy`; scratch is `~/tmp`.
Recheck private global/workspace rules for machine policy, never a historical
benchmark command. Keep completed outputs; mirror/verify before any cleanup.

Stable adoption releases the HDR frontend dependency. Reuse the existing
[HDR/SDR plan](docs/PLAN_HDR_SDR_ALIGNMENT.md): preserve native u16 PQ/HLG and
f32 nits, the SDR fast path, explicit color/display interpretation and f64
numeric controls. Compare PU21 with independently registered alternatives;
previous BHdr success is not an isolated feature-utility proof. Runtime,
near-black precision, highlight clamps, mixed SDR/HDR behavior and full steering
cost remain unmeasured for a new frontend. Do not promise a free speed win.

### Documentation and recovery discipline

Update owning docs and run artifacts as work changes; do not rely on an agent's
memory or a one-off handoff. Read global/workspace/project instructions, verify
checkout ownership and dirty state, inspect the active goal, then run the
status command above. Preserve other work and frozen snapshots. Continue the
next registered task after each result. A lost agent or model-limit error
requires inspecting its real job, taking over supervision, and recording any
verified blocker. Documentation alone does not complete the active goal.

## Historical chronology — September 15 and earlier

## September 15, latest directive: production priorities and paid-study readiness

[The controlling execution plan](docs/PRODUCTION_PRIORITIES_2026-09-15.md)
supersedes conflicting older task orders, including paused goal text. It gives
the next agent measured blockers, existing owners, bounded experiments,
dependencies and exit criteria. Start with evidence admission and fixed-model
targeting diagnosis; continue TRAIN local ranking, JXL spatial value, native
HDR training, integrity specificity and full runtime/qualification.

[The Squintly agent handoff](https://github.com/imazen/squintly/blob/main/docs/STUDY_READINESS_2026-09-15.md)
owns phone/color/HDR presentation, pixel-preserving zoom, worker instructions
and the readiness decision before full paid collection. The September 1 study
is historical, not a ready current-model multi-observer study. Relevant
`zenpapers` summaries require primary-source checks before implementation.
Documentation/handoff work does not change models, gates or release status.

## September 15: implementation and frozen recovery assessment complete

[Results, runtime and remaining product gates](benchmarks/recovery_completion_2026-09-15.md).
Basic/peak prepared maps reuse the scoring extraction; native HDR prepared
sessions preserve declared primaries and precision. JXL candidate tiles now
consume complete rectangle refinement, including maxima. Both recovered
five-seed TRAIN-only models are fully assessed: composite 0.8460 / 0.8760
versus matched B 0.8382 / D 0.8308. Fast RGB8 1 MP scalar/prepared p95
9.77 / 26.95 ms; richer 18.79 / 91.73 ms. No default changed.

Both FAIL product qualification: CID22-band, addressability, targeting-tail,
independent spatial RD and HDR human-ranking regressions. Rich map cost also
fails the 3× ratio. No qualified corruption companion. Do not restart from
September 14's unfinished list: use the exact frozen models, failure evidence
and owner map. Corrected HDR TRAIN cache and native judges are complete;
shared SDR/HDR calibration was tested and did not meet BHdr on UPIQ.
Artifact authority `/var/tmp/zensim-validation-2026-09-15/`, served packet
`recovery-completion-2026-09-15/` beside the gauntlet. Full chronology below
is historical. The old autonomous goal remains paused; this was a direct task.


## September 14, latest implementation: Rev3 maps and native color/HDR

Implemented fused basic/L8 combination with preserved rounding, shared native
SDR float peer input, declared-primary HDR conversion without SDR clipping,
and exact PQ16 EOTF lookup. Wide extraction now refuses mixed process/requested
formula revisions. Native HDR datagen requires its explicit corrected input
contract. See [implementation, tests and timing](benchmarks/rev3_native_optimization_2026-09-14.md).

Both full legacy-RGB8 Rev3 TRAIN caches are verified: CID22 17,611 pairs and
SafeSyn 196,086 pairs. Native 214-pair peer replay preserves previous feature,
score and spatial results. No new fit, EVAL/TEST read or product promotion.
Next: complete source-family fit/development/calibration admission, recover
one fast and one peak-capable recipe, remove repeated prepared-map extraction,
and implement/qualify native HDR maps and matched HDR supervision. Shared
weights across SDR/HDR remain a hypothesis, not an established calibration.

## September 14, controlling priority: baseline recovery on improved extraction

The user clarified that recovery means incumbent-level quality **using the
faster/better feature extraction**, not a return to the old product extractor.
Restore the established SafeSyn/CID22 TRAIN supervision, adequate training and
train-only calibration on current Rev3 features before more broad science.
Keep B/C/D as matched controls; initially compare one fast and one richer
extraction regime. EVAL/public TEST cannot select recipes or features.

All17,611 CID22 TRAIN pairs/201 references are re-extracted and verified at
Rev3/full944 plus same-buffer SSIM2 (134.7s, zero failures). All 196,086 SafeSyn
pairs/3,218 source paths are fully extracted and verified with zero failures.
Do not restart the completed extraction; verify its receipts before reuse. [Recovery protocol](benchmarks/baseline_recovery_2026-09-14.md).
Work/protocol: `/var/tmp/zensim-validation-2026-09-14/baseline-recovery/`.
These are canonical training caches for cheaper served read sets, not a
requirement to compute full944 at runtime. Initial recovery uses the explicit
legacy RGB8 SDR input contract; native color/HBD/HDR qualification stays separate.
No fit has launched. Do not resume corruption/infrastructure work ahead of this
priority. Prior completed evidence and qualifications remain unchanged.

## September 14, newest: color-correct corruption fixtures

[Color resolution](benchmarks/integrity_color_resolution_2026-09-14.md) adds an
explicit native SDR input route to the existing RGB8 corruption generator.
Both sRGB routes preserve all 718 old files exactly; the originally refused
mobile source now generates with full CMS provenance. Canonical TRAIN audit:
705 unique pairs, 30/31 severe proxies detected, 0/3 known-valid activations;
progressive-AC truncation remains a head miss despite negative perceptual score.
Both independent decoders reproduce the old AVIF q85 cast; exact producer cause
is unresolved. All 181 old native nonidentities stay quarantined. No fit, EVAL,
HDR qualification or promotion. Next: freeze a representative color-correct
TRAIN integrity packet before head refit; preserve source-family separation.

## September 14, newest: complete ensembles in native JXL steering

[Native comparison](benchmarks/jxl_ensemble_native_2026-09-14.md) measures seven
frozen compositions, 4,914 targeting cases and 8,442 pairs under two independent
judges. Four compositions refuse unsupported spatial terms. Correcting native
seed quantizer reparameterization restores all 2,646 neutral ladder pairs to
byte/pixel/score identity. Independent sample pixel/score parity is 21/21, with
21 libjxl v0.12 compatibility decodes. All prior human/dial qualification failures
remain. New models show mixed independent-judge RD; no model is promoted.
The full/fair gauntlets add a native discussion and measured/unsupported annotations.
Next: TRAIN-side product failures, direct native RD, corruption specificity,
color/HBD/HDR and production latency; no adaptive changes from these EVAL results.


## September 14, newest: complete frozen scalar targeting

[The bounded targeting comparison](benchmarks/rev3_targeting_2026-09-14.md)
adds 9,648 actual 1/2/3-shot cases for all eleven frozen compositions on
JXL/JPEG/WebP. Seed calibration uses nine TRAIN families after correcting the
old twelve-family admission for later reservations; the initial stopped run
is explicitly inadmissible. All eight validation families stay unchanged.
33 independent output samples have exact decoded-pixel and complete-score parity.
The existing bounds instrument now accepts hash-bound ensembles; no new public
API or controller. The gauntlet carries measured scalar-targeting scope beside
unchanged failed product qualifications. Native spatial RD, AVIF targeting,
corruption specificity, HDR and production latency remain incomplete. Do not
adapt the frozen candidates to these assessment results.

## September 14, newest: public TEST assessment and complete ensembles

The latest user permits published TEST evaluation when no separate EVAL exists,
with overfitting guards and secret holdouts untouched. CID22 oracle TRAIN and
human TEST are distinct. This supersedes the blanket ban in older entries below;
read the current top of DATA_SPLITS. Frozen public-test exposure is recorded in
the `rev3-public-human-eval/EXPOSURE.json` artifact. No training or selection
uses these new measurements.

All nine frozen MT913 candidates and matched B/D have fresh CID22/AIC3/AIC4
panels; all previous six EVAL predictions are unchanged, and composite coverage
is now 6/6. All 297 matched JPEG coherence cells (11 compositions ×27) complete
through the Rust ensemble surface. This does not establish native codec RD or
HDR qualification. All 9,593 ladder cells complete in both revisions; the
[full comparison](benchmarks/rev3_public_human_eval_2026-09-14.md) records actual
qualification failures and remaining gaps. The published assessment uses
`assessed-current/`; the first assembly incorrectly applied historical M3 to
current attribution and is superseded without changing any measurement.

## September 14, newest: native TRAIN input admission and exact CSV binding

[Native integrity admission](benchmarks/native_integrity_admission_2026-09-14.md)
adds strict trainer-v3 / diagnostic-admission-v2 for native Rev3 D228. Exact
scoring interpretation, complete feature audit, root settings and per-row f32
hashes are required before fitting. Native final scoring uses pairs-tsv/sqrt;
round-trip CSV parsing fixes47 f32 mismatches on29 of214 real audit rows.
All 214 corrected rows match canonical hashes; prior features/scores/maps remain
unchanged. Equal bytes with different primaries no longer deduplicate.
The181 unresolved pairs remain unlabelled;33 identities are valid. A real
attempt to admit that diagnostic packet for training refuses before payload
reads. No model fit/EVAL/promotion. Next resolve P3/q85 and restore mobile
coverage before a reviewed native TRAIN refit. HDR remains separate.

## September 14, newest: complete canonical-feature audit and min-max fix

[Evidence](benchmarks/complete_feature_audit_2026-09-14.md) closes the prior
ensemble gap: 214 native944 pairs match all 80 consumed canonical features
exactly. Native372 and legacy372 base/tree audits match all 228 IDs on 214
pairs each. All prior audit fields and CSVs remain unchanged. New records add
an explicit complete-read-set scope and IDs; old receipts are not upgraded.
The same work fixes replacement min-max planning that previously inspected
only placeholder network weights. A sparse masked-feature regression and a
score-cancellation negative control exercise the actual failure mechanisms.
No fit/EVAL/model promotion. Next admit native audit-v2 downstream and resolve
P3/q85/mobile TRAIN coverage before matching refits; HDR contract remains open.

## September 14, current: color/HBD/HDR status after native scoring

[Current assessment](benchmarks/color_hbd_hdr_current_2026-09-14.md) supersedes
the decoder-only status below. Native SDR extraction now exists and has bounded
TRAIN serving evidence. Nine revised models retain their six SDR EVAL panels
and NO-SHIP verdicts; native HBD/HDR product qualification remains missing.
The newest September 13–14 discussion links the exact twelve-model comparison
and all recent TRAIN studies. Complete ensemble canonical-feature auditing,
color provenance and matching TRAIN-era admission/refits remain dependencies.

## September 14, newest: native SDR scoring and complete serving audit

[Contract and results](benchmarks/native_sdr_contract_2026-09-14.md) add
`--input-contract sdr-native-clip-v1` to the existing extractor/audit. Known
SDR samples retain u16/linear-f32 precision with declared primaries; arbitrary
ICC goes through the existing full CMS. JPEG XYB output metadata is corrected
at the codec owner. Legacy RGB8 remains byte-identical on all214 admitted pairs.

All 214 TRAIN-fit pairs pass native372 base/head audits and native944 ensemble
scalar/spatial audits.30,136 accepted-head queries and626,249 ensemble refinement
queries pass; q85 remains the sole integrity rejection. No new fit, EVAL,
clean-negative admission, native RD or HDR qualification. New audit-v2 and
native input-era metadata must be admitted explicitly downstream. Next trace
P3 encoder conversion/q85 and restore mobile corruption generation with correct
color before matching TRAIN extraction/refit. Failed generic conversion routes
and the initial incompatible944/head request are retained in the replay record.
The existing audit's per-feature maximum measures companion-declared IDs;
no-head ensembles currently prove cached-score and scalar/spatial feature-path
parity, not an independent canonical comparison of every member's read set.
Add that complete-read-set audit before using this as qualification evidence.

## September 14, latest status: integrated two-day gauntlet discussion

[Updated review](benchmarks/color_hbd_hdr_update_2026-09-14.md) incorporates
native decoder admission into the September 13–14 discussion. The current
extractor still uses RGB8: retained native precision/metadata has not yet become
color-correct model evaluation. Nine candidates have six SDR EVAL panels;
all remain NO-SHIP, with no current Rev3 native HBD/HDR qualification. The
full and fair gauntlets link the updated discussion and existing TRAIN reports.
This is a documentation/publication update; no model or evaluation changes.

## September 14, newest: native decoder retains color metadata and precision

[Native admission](benchmarks/native_color_admission_2026-09-14.md) adds a
hash-bound inspection mode to the existing decode owner. All215 admitted
TRAIN-fit files reproduce their old RGB8 hashes, while native u16 buffers and
source/decoded metadata are retained separately.17 codec tests and five CLI
refusals pass. Synced AVIF/JXL decoder updates are pinned in the receipt.

None of180 P3-derived reconstructions retains the source520-byte ICC; that
does not establish whether original encoding converted to sRGB correctly.
The native hash identifier does not recognize this profile. Use the existing
full CMS where required; do not blindly retag outputs or reapply XYB ICC.
The feature extractor still requests legacy RGB8. Corrected color/precision
scoring and restored TRAIN coverage remain the next work; no model advances.

## September 14, newest: color/HBD/HDR audit and two-day gauntlet discussion

[Status and chronology](benchmarks/color_hbd_hdr_status_2026-09-14.md) distinguish
65 fresh passing engineering checks from missing product qualification. The
RGB8 adapter loses PNG ICC metadata and native HBD precision. HDR transfer and
feature routes exist, but take primaries as-is; sampling-v1 refuses HDR. No new
Rev3 HBD/HDR quality or spatial qualification follows from the six SDR panels.

The September 13–14 discussion selects the nine measured MT913 candidates and
matched baselines, with a visible review link and an index of recent records.
TRAIN comparisons stay separate; all model verdicts remain unchanged. Next
resolve the explicit color/precision contract and restore missing TRAIN coverage.

## September 14, newest: Rev3 integrity pilot trained; color coverage blocks EVAL

[Registered fit and comparison](benchmarks/integrity_rev3_train_2026-09-14.md)
produces a real198,624-byte Rev3 D228/HGB head through the strict trainer.
Calibration154/154 and development267/267 severe proxies are detected; honest
nonidentity activations are0/792 and2/1,666. All272 native attempts preserve
steering maps. Exact Rust/Python and pixel/cache/f32 parity passes. The two
false alarms are new cases, including a JPEG heatmap score drop55.67→0.

No advancement: source1556/8012/8014 carry Display P3 profiles (33 references,
213 TRAIN-fit pairs), while the PNG extractor currently flattens samples
without applying ICC. Native generation correctly refused both mobile sources.
The sRGB pilot excludes these rows and retains the failed36-source request;
35 admitted sources were used. P3/mobile coverage and the earlier AVIF q85
cast remain explicit blockers. Do not strip profiles, silently drop the class,
retag source roles, or declare the pilot qualified.

All35,582 binary/unlabelled candidate audits are retained. The head activates
on7,094/22,627 distinct ambiguous/recoverable pairs; correctness is unmeasured
there. No EVAL was scored. Next resolve native color admission and broader
error-label scope before any new fit or EVAL. Other product gates and broader
chronology recovery remain open.

## September 14, newest: broad honest integrity controls scored and inspected

[TRAIN control audit](benchmarks/corruption_honest_train_2026-09-14.md) measures
26/8,880 product reconstructions active (0.293%) and0/264 distinct native JXL
pairs (272 attempts). All26 have negative bases: scalar composition hides the
activation, but all26 reject prepared steering. Native maps remain exact over
14,144 queries. All10,771 pixel/cache/f32 results and original pixel hashes
reproduce; the gallery contains every active control.

Low-q rates are higher (AVIF q5 3.38%, JPEG q5 4.73%). A q85 AVIF pink cast also
appears with an independent decoder; retain its conservative valid label and
investigate before clean-negative training admission. Stored contexts are
singletons, not matched quality ladders. This is a frozen Rev1 TRAIN diagnostic,
not a Rev3 fit or qualification. The original strict trainer still requires an
explicit registered extension and a unified family-role admission before new
Rev3 training. No EVAL or TEST rows were extracted or scored. Broader recovery remains open.

## September 14, newest: corruption heads now bind arithmetic revisions

[Serving correction](benchmarks/corruption_revision_contract_2026-09-14.md)
reproduces and fixes legacy Rev1 tree attachment to Rev3 bases. Slot coverage
alone was insufficient. ZCTH v3 stores a hash-bound formula revision with f32
input semantics; valid v1/v2 bytes and inference remain unchanged. Matching
scalar/prepared compositions preserve maps when inactive and reject corruption
when active, even if a negative base masks activation in the scalar minimum.

This is engineering verification on synthetic data, not a new head fit or
qualification. The strict trainer still owns the original Rev1-only recipe.
Next admit representative TRAIN corruption and honest low-quality/native
controls for an explicitly registered Rev3 fit; do not retag old weights.
Broader recovery and all unmet product gates remain open.

## September 14, newest: exact-project memories complete; input layout rechecked

[Chronology recovery](benchmarks/claude_memory_chronology_2026-09-14.md) now
records 99/99 exact-project memories read in full, with all source hashes
unchanged. Broader related roots, transcripts and repository documents remain
incomplete. Later corrections reverse early linear-ceiling claims (clipping),
HDR significance claims (adaptive selection), and missing codec-loop claims.
Historical successes do not qualify current Rev3 models.

The same nine quarter-chroma bakes pass an explicit input-major inspection
check, including asymmetric and disconnected-output negative controls. All 630
inputs have nonzero structural paths; this does not establish indispensable
features or nonzero sensitivity everywhere. No models, predictions or gates
change. The earlier failed feature recipes remain failed; no immediate sweep.

## September 14, newest: trained quarter-chroma variants fail tail/quality gates

[Paired y60/y70/y80 study](benchmarks/chroma_scale_train_2026-09-14.md) completes
nine H32 fits and one exact repeat on the admitted human/codec TRAIN packet.
Adding quarter-resolution B improves fresh-control human rank .83150→.83582
and codec rank .92442→.93274, with zero native consensus conflicts. It adds one
codec outlier, worsens human clumping/max residual and lowers Butteraugli map
association. Adding X as well also fails frozen-control human noninferiority.
Neither arm passes the registered quality screen; no immediate recipe sweep.

All 17,537 panels, 3,536 native pixel/cache score checks and 244,800 exact
consumed-feature comparisons are retained. Packed IDs/revisions/live inputs
are verified. The original control reproduces its later full-precision exports;
the earliest six-decimal exports are not the exact-comparison authority.
Four timing suites retain 1,260 calls and 12/18/6/4 flagged rounds: descriptive
only, no performance gate passes. Full model artifacts, raw scatter and native
A/Bs are linked from the gauntlet. All product qualification remains open.

## September 14, newest: direct filters reduce L4 phase span, not the conflict

[Four-contract comparison](benchmarks/native_filter_2026-09-14.md) measures the
same60 y60 features through neutral Rust instruments on the same130 TRAIN
alignments. Triangle/Mitchell/RobidouxSharp reduce focal B ART L4 span68–71%,
but increase mean span15–24%; triangle/Mitchell make L4's adverse response
positive at all64 phases. This is not a refitted-model quality result.

All273,720 live box features and4,562 pixel/peer rows reproduce exactly;
full-resolution features match every contract. Each contract retains the same
six failing whole-cell period controls at f14; focal-pair controls pass.
The independent resizer reference passes and explicitly includes binary16
input quantization. No uniform feature improvement or model advancement.
Do not silently retag trained bakes or start a kernel-parameter sweep. Further
contract comparisons require admitted TRAIN re-extraction/refitting and complete
quality/native/runtime assessment. All product qualification remains open.

## September 14, newest: coarse-B native preference is phase-sensitive

[Phase diagnostic](benchmarks/native_phase_2026-09-14.md) translates the exact
TRAIN reference/decoded pixels together in a fixed canvas, using public Rust
scores and prepared bin8 maps. Two frozen y60/H32 ensembles, eight cells,
130 alignments, 8,580 candidate scores and 260 maps; no fitting or encoding.
The ordinary six-model replay is byte-identical, and the two-model original
slice reproduces 544 scores and feature vectors, plus 16 maps.

Focal screen r1-up constrained score delta spans −.08581..+.22750 across 64
phases. Coarse-B contributions vary while full-resolution contributions remain
nearly fixed; all three B artifact sensitivities remain negative. SSIM2 also
changes preference across phases; Butteraugli retains its improvement direction.
Neither is human truth. Whole-cell period controls fail 12/48 narrowly at f14
(full-resolution Y SSIM-L4); all focal baseline/r1-up controls pass. Keep both
findings, the original failed bars and the padding/binning limitations visible.

This justifies a bounded existing-sampling-contract feature comparison before
refitting, not a global chroma deletion or a claim that a filter fixes quality.
No model advances. Memory audit remains 97/99 exact-project files; broader
memory/transcript scope and all product qualification remain open.

## September 14, newest: chronological memory recovery and TRAIN report discovery

[Recovery record](benchmarks/claude_memory_chronology_2026-09-14.md) reconciles
the exact-project memory audit at97/99 full reads with source hashes. Two large
memories, related roots/transcripts and repository-document scope remain open.
Historical V2 “kill” conclusions were recipe/configuration dependent; better
HDR ramps hid a large human-quality loss. Current native failures likewise do
not prove a feature ceiling. Existing numerical and split requirements stand.

The frozen MT913 candidates already have six measured gauntlet panels. Five
later TRAIN-development comparisons are now registered and linked separately
on the board, with explicit verdicts and their full panels/spatial A/Bs.
No TRAIN evidence is promoted to EVAL, and no new models are fitted. The
coarse-B phase/filter diagnosis remains the next concrete feature question;
no immediate TV weight sweep. No model qualifies; full product work stays open.

## September 14, newest: native TV recipe fails; checkpoint trap caught

[Pair-margin study](benchmarks/native_tv_train_2026-09-14.md) completes six
local120/H128 scientific fits and two integrity controls on admitted TRAIN.
The existing TV hinge is direction-tested through Rust serving. With explicit
human/codec fitting-group checkpoint weights, zero-native removal and TV repeat
controls reproduce all predictions. Raw/packed and native pixel/cache parity pass.

The TV ensemble improves human rank .86965→.87354 but increases robust native
conflicts9→10/79. Map/M2 improve; raw clumping and maximum residual worsen.
Only95/214 fitting margins are satisfied; mean hinge is worse than control.
This fails the recipe, not proof of a feature-set ceiling. No model advances.
Full10,792 panels, all seeds and exact A/Bs are served at
`/zensim/reports/native-tv-train-2026-09-14/index.html`.

The first attempt caught a checkpoint-selection trap: all-zero validation
weights select by ALL group reports, including zero-weight native. Early-stop0
still returns best_bake. The removal control changed968/1,000 predictions; those
three fits are diagnostic only. Use explicit positive TRAIN selection weights
and the intended policy. Historical arithmetic is preserved; the playbook now
documents this. Do not start an immediate TV weight/margin sweep. Coarse-B phase/
filter diagnosis, broader science and all product qualification remain open.
Memory full-read audit30/99; no EVAL/TEST, encodes or extraction in this study.

## September 14, newest: native conflicts traced through signed features

[Contribution diagnosis](benchmarks/native_contributions_2026-09-14.md) reuses
the exact eight development cells, six frozen ensembles and frozen D. The
native replay's existing diagnostic flag now retains computed sensitivities.
Ordinary output is byte-identical; diagnostic output differs only by that field.
All 1,792 responses reconstruct within 8.89e-16; 97,886 common active feature
comparisons between heads are exact. No fit, calibration, EVAL or TEST.

All 22 peer conflicts are also wrong in feature-level linearization. Positive
sensitivities to dissimilarities account for 90–93% of opposing component mass
in local120 conflicts, chiefly X SSIM/edge terms. The y60/H32 conflict instead
follows coarse B feature changes; it does not prove sampling aliasing. No
arithmetic defect or benefit from feature removal is established.

Before fitting, verify the existing TV pair-margin path against clean admitted
native training pairs (indices, group order, polarity, actual loss reachability
and sampling effects). Diagnose the separate coarse-B phase/filter response
on exact saved pixels before changing kernels. Do not globally remove chroma
or repeat capacity/map-algebra sweeps. Signed plots and exact A/B gallery:
`/zensim/reports/native-contributions-2026-09-14/index.html`.
The explicit cumulative memory audit is 26/99 files; the full audit and all
remaining product qualification work stay open. No model qualifies.

## September 14, newest: existing constrained heads fail native advancement

[Native constraint comparison](benchmarks/native_constraints_2026-09-14.md)
measures the six already-trained product ensembles on the retained native JXL
packets. No new fits, encodes, extraction, EVAL or TEST. All three constrained
profiles fail the registered screen: y60 introduces conflicts; local120 reduces
them 10→6/79 but loses .03618 human rank. All native M2 diagnostics pass.

Internal map consistency is not quality agreement: constrained local120 has
median own-score map rank .743, versus .385 SSIM2 and .472 Butteraugli quality.
The existing native gallery now labels own-score columns explicitly and shows
both peer medians, requiring complete cell coverage. Two prior packets remain
renderable; the missing-cell negative control is refused.

All 1,220 decoded pixel/peer rows and 1,628 overlapping plain-model scores
match prior evidence. Report/gallery:
`/zensim/reports/native-constraints-2026-09-14/index.html`.
Do not repeat the already measured nonnegative-distance/capacity sweep. Next
diagnose retained native feature deltas through existing contribution owners
before registering another fit. The cumulative exact-project memory audit is
24/99 full reads; related roots/transcripts remain incomplete. No model
qualifies; the full scalar/targeting/spatial/corruption/runtime goal stays active.

## September 14, newest: robust native pairs assessed; sampler replay corrected

[Robust-pair experiment](benchmarks/native_robust_train_2026-09-14.md) reuses
the eight-fit/four-development TRAIN packet, without encodes, extraction or
EVAL/TEST. Twelve primary Rust fits compare explicit all-pair versus two-mentor
robust native supervision. Basic156 conflicts improve 2→1/79 but human rank
fails the frozen untreated-control tolerance. Y60 retains zero conflicts and
passes the numeric screen, yet human tails worsen and five of eight native
map cells remain below the .70 diagnostic line. Neither profile is promoted.

The first twelve fits are also retained: neighboring legacy sampling seeds
are raw-stream offsets, not independent sampling replicas. A real-draw test
proves 262,143 shifted attempts match; corrected fits use recorded disjoint
windows. `subset_sim` adds an opt-in overlap preflight and now honors recorded
sample seeds in fulleval metadata. Explicit historical replay is byte-exact;
the corrected metadata replay reproduces the actual training digest.

All primary members/ensembles have full canonical TRAIN panels/scatter and
public pixel/cache parity. Report/gallery:
`/zensim/reports/native-robust-train-2026-09-14/index.html`.
Before more fitting, reconcile existing constrained best-of-all/two-reference
training with these tail and native-response failures; do not start a broad
pair-weight sweep. Native RD, targeting, corruption, frozen qualification,
memory/transcript audit and cleanup remain open. The full goal stays active.

## September 14, newest: native local training completed; neither recipe advances

[Registered experiment and results](benchmarks/native_local_train_2026-09-14.md)
adds812 native JXL outputs across8fit/4development families, preserving product
partitions and excluding the prior replay families. Twelve real Rust fits compare
control versus10% native-local RankNet sampling in y60/H32 and basic156/H128;
all seeds pack and reproduce cached scores. Controls exactly reproduce1,624
frozen native pixel scores;1,056 new development pixel/cache comparisons agree.
No EVAL/TEST, calibration, new kernels, or release qualification.

Basic156 development peer conflicts improve9→5/79 and median native mass rank
.343→.440, but human rank falls.00583 (beyond registered.005 tolerance), human
geometric out4 grows.4%→.9%, and one native M2 check fails. Y60 gains rank but
introduces one screen conflict; its native map association remains weak.
Frozen D has0/79 conflicts and median association.638 on the same development
pixels. Both treatments fail advancement; do not promote them or tune on EVAL.

Full canonical panels/scatter, all12seeds/4ensembles, explicit undefined sparse
statistics, commands/models and exact development A/B gallery are retained in
`~/work/zensim-validation-2026-09-14/native-local-train/`. Served report:
`/zensim/reports/native-local-train-2026-09-14/index.html`.
Next register explicit robust local preference pairs versus indiscriminate
within-cell draws using the existing trainer, preserving feature profiles and
human/codec supervision. Native RD, targeting, corruption, full qualification,
Claude-memory audit and cleanup remain open. The full goal remains active.




## September 14, newest: native JXL replay exposes model and map weaknesses

[Actual native interventions](benchmarks/native_map_replay_2026-09-14.md) freshly
decodes408 retained JXL outputs for three admissible TRAIN families, two sizes,
two distances. Reserved document6068 is excluded. Six frozen Rev3 ensembles
and frozen Rev1 D use the public Rust API; final-source primary output reproduces
byte-for-byte. No training, EVAL/TEST, new encodes or allocator optimization.

Y60's23/23 reference-repair result does not establish native guidance: native
mass/response rank falls below the .70 diagnostic line in7/12 cells; D in4/12.
Richer complete-density models miss it in11/12. Hard-max ensembles have partial
additive density and cannot qualify that map. D's median native association is
.815 versus y60 .651 and richer complete maps .369–.401.

Among160 robust same-buffer SSIM2/Butteraugli consensus comparisons, D and y60
have zero opposite scalar preferences; richer models have11–27. This exposes
scalar preference weaknesses as well as map weaknesses. No model qualifies.
Next register broader TRAIN native-local preference supervision with source-family
separation, then compare unchanged human/codec/tail/scatter panels and native
allocation. Do not spend another cycle proving local gradient algebra.


## September 14, newest: maximum projection parity passes; timings remain unqualified

[Exact max projection](benchmarks/max_projection_2026-09-14.md) reduces redundant
source-footprint scatters using row/column maxima with unchanged f32 signals.
The old per-pixel implementation is test-only. All3,584 sampling/mask projection
cases and14,268 independent reflected-footprint queries pass. Both frozen
basic228 ensembles pass15,540 public rectangle comparisons each, including
scores/features/sensitivities, worker reuse, finite moments and parallel toggles.

The fixed four-block synthetic1MP/4MP ABBA study records240 one-call observations,
with3/8/4/3 flagged checks. Zero runs meet the registered quiet criterion; no
qualified speedup or runtime gate is claimed. Old/new1MP p95 ranges77.97–84.74 /
72.60–72.69ms;4MP338.22–338.50 /290.32–292.16ms, descriptive only.
No training, corpus, EVAL/TEST or model promotion. Existing finite-repair and
native steering failures remain; closing that product accuracy gap is still
required. Keep the full goal active. Evidence and binary sources are retained.


## September 14, newest: native latency measured; quiet admission rejected

[Native TRAIN latency screen](benchmarks/native_latency_2026-09-14.md) measures
seven frozen scalar ensembles and six supported prepared score/map ensembles,
five original-size JXL TRAIN pairs across five content classes, plus synthetic
1MP/4MP textures. All four runs completed 30 one-call rounds per group, 3,150
observations. Basic228 native map/scalar p95 ratios are3.16–4.69; y60 is cheapest
but retains its known quality failures. Full944 synthetic scalar p95 is54.66 /
243.22ms, above absolute budgets. These are descriptive, not qualified timings.

The old model instrument disabled its resource gate. Enabling it exposed a
second owner defect: Engine never propagated the strict verdict, leaving saved
unreliable=false. Runs recorded15/15/5/1 noisy checks; their original driver
acceptance flags are superseded by final quiet-qualification refusal. No noisy
rounds were dropped or runs repeated to obtain a favorable gate.

Zenbench53941021 fixes strict-verdict propagation and exports per-round
checked-clean/flagged/disabled status; a forced insufficient-RAM regression
fails before and passes after. Multi-run aggregation preserves unreliability.
Both Zensim benchmark dependencies now use the corrected owner. The root model
benchmark accepts SHA-bound native TRAIN PNGs with byte/pixel/dimension checks,
strict gating, one-call timing, and explicitly named D_current_revision context.

Next use these costs with the existing TRAIN quality/spatial panels to reduce
actual map work. Any new latency qualification needs sufficient preregistered
clean rounds, complete in-run contention/memory evidence and representative4MP
coverage. No fitting, EVAL/TEST or model promotion occurred; the full goal remains.


## September 14, newest: paired latency observations now retained

[Latency evidence infrastructure](benchmarks/latency_evidence_2026-09-14.md)
closes the summary-only timing exporter gap. The canonical zenbench owner
(revision16fb8fdd, pushed) records each completed round's actual batch count,
raw/compensated total durations and execution order. Aggregated multi-run
results clear samples; historical summaries cannot become p95 evidence.
The standalone Zensim benchmark pins that revision. Its existing saved-result
path verifies 12 real observations on two tiny synthetic fixtures and two
public scalar arms, three rounds each. This is an exporter smoke, not latency
qualification. Owner tests, Clippy and API snapshot checks pass.

Next performance campaigns can use the existing benchmark with
`ZEN_S2_SINGLE_CALL=1` and `ZENBENCH_RESULT_PATH`, verifying one actual call
in every retained round. Still require representative admitted inputs, frozen
complete candidates, required geometry/hardware, at least30 accepted paired
rounds, contention records and memory measurements. Do not infer p95 from old
min/mean/max files. No fit, EVAL/TEST, production score change or model promotion.


## September 14, newest: derivative duality passes; finite repairs fail

[Independent duality diagnostic](benchmarks/gradient_duality_2026-09-14.md)
passes 12 combined and 120 feature-basis controls. The frozen local120/H128
ensemble passes the independent forward/reverse comparison too, but only 5/24
public secants meet the earlier diagnostic tolerance, and 6/8 partial repairs
have wrong-sign gradient predictions (maximum error 52.648 score points).
All eight partial repairs lower this synthetic model score; no independent
quality judgment was made. Four full identity repairs are separate.

The earlier 1/24 public gate remains FAILED. No fitting, corpus, EVAL or TEST.
Prototype and binary are archived under `~/work/zensim-validation-2026-09-14/gradient-duality/`;
production source is restored. A fixed local gradient is not a qualified finite
repair estimator. Preserve exact/shared-moment references and these witnesses;
stop repeating local algebra as advancement toward native steering. No model
qualifies, and runtime/targeting/corruption/full evaluation gates remain open.


## September 14, newest: complete local120 derivative control fails; diagnosis retained

[Complete-gradient study](benchmarks/complete_gradient_2026-09-14.md) combines
the tested SSIM/edge/MSE derivatives, all scales and RGB pullback. The known
signed linear model passes only1/24 public pixel-direction comparisons; base
feature reconstruction max8.67e-19. The planned frozen local120 ensemble was
NOT RUN, nor any corpus, finite-repair, runtime or native-RD stage. No fitting,
EVAL or TEST. New prototype is archived and removed; parent tests remain.

First17x9 whole-field case has predicted+.00159854 versus public-.00650901
at epsilon.001 and-.000284985 at.0005. At.0005, an existing centered-f64 SSIM
reference cuts summed36-coordinate derivative discrepancy .034939→.000998,
while167 edge activations and8 residual signs differ across perturbation endpoints.
These show numerical sensitivity and nonsmooth finite steps, not complete
correctness or proof of a scalar SSIM bug. No tolerance or gate was changed.

Evidence/binary/source: `~/work/zensim-validation-2026-09-14/complete-gradient/`.
Next compare independent forward/reverse mathematical directional propagation,
measure branch margins and f32 observability, and keep actual public finite
repairs as a separate criterion. Do not fit/ablate around this failed diagnostic.
All jobs finished. No candidate qualifies; the complete goal stays active.


## September 14, newest: SSIM and public color/scale prerequisites pass

[SSIM-adjoint study](benchmarks/ssim_adjoint_2026-09-14.md) adds test-only Rev3
moment partials in the existing SSIM owner, checked against its scalar function
and actual fused H/V signal/pooling kernels. All24 partial and216 plane-direction
checks pass; max plane derivative error5.778e-6. Saturated-luma and identity
controls pass. Initial failures were a missing revision-wrapper completion
marker, now fixed with original failed logs preserved; no tolerance change.

Independent synthetic Rust MSE bake through public BakeScorer verifies the
in-gamut linear RGB color derivative, reflect/stride padding and four-scale
2x chain, including odd tails. All48 consumed features reconstruct (max6.855e-12),
and24 full pixel-score directions pass (max6.050e-7). This is separate from a
complete SSIM-model gradient, not qualification of either finite repair or RD.

No corpus, fitting, calibration, EVAL or TEST. New implementation is test-only;
production arithmetic and APIs remain unchanged. Evidence and binary are under
`~/work/zensim-validation-2026-09-14/ssim-adjoint/`. Next assemble complete-model
response from these verified components; handle remaining HF/peak behavior and
head sensitivity, then measure local directions, finite repairs and full cost
before native JXL interventions. All jobs finished; full goal remains active.


## September 14, newest: pixel-adjoint prerequisite passes; historical premise corrected

[Pixel-adjoint study](benchmarks/pixel_adjoint_2026-09-14.md) adds three synthetic
Rust tests using the existing SIMD blur and production edge pooling owner.
Reflect-101 needs K^T = D K D^-1, with edge/corner weights 1/2 and 1/4; plain K
is not its transpose at boundaries. Direct gather/scatter error max2.084e-7;
144 smooth artifact/detail p1/2/4/8 directional checks pass, maxerror1.367e-5.

The historical C2b zero-mass argument does not exclude actual repair-direction
pixel derivatives: gradient sum zero means constant-shift invariance, whereas
its contraction with nonuniform repair can be nonzero. Exact full erasure and
its local derivative differ by factor2 for residual energy. Historical eight-cell
spread failures remain valid; no finite-response/native-RD gate is relaxed.

All changes are tests/documentation. No model, production math, public API,
corpus, fitting, EVAL or TEST access. Evidence and binary are retained under
`~/work/zensim-validation-2026-09-14/adjoint-response/`. Next connect needed
feature derivatives and the color/scale chain to complete model inference;
measure local directions, finite repairs and runtime separately before native
steering. The full goal remains active and no model qualifies. All jobs finished.


## September 14, newest: shared window moments pass accuracy, fail dense cost

[Shared-moment response](benchmarks/shared_moment_response_2026-09-14.md) retains
canonical blurred moments and updates local changes through reflected f64 SATs,
then pools via the existing fused kernel. All 138 checks / 9,756 predictions on
six frozen models and 23 admitted TRAIN pairs pass: max error .003697 points,
minimum rank .999555, no material wrong signs, exact bases. No fit or EVAL/test.

Initial synthetic failure was a prototype-only endpoint defect: MaxRemoval
expects inclusive bounds, the helper passed exclusive ends. Direct enumeration
failed before correction and passes afterward; all original synthetic gates pass.
Production extrema arithmetic was correct. A comment clarifies its contract.

Cost still fails: dense8 row4589/1024² takes 1.557s for basic228/H32, 28.09 scalar
calls; moment/feature work alone takes 1.216s. H128 is 30.03 calls. Dense8 accuracy,
native RD and p95/memory qualification are unmeasured. Large-image accuracy
witnesses remain block64. This is not a controlled speedup or narrow-model test.

Prototype, failures, measured binaries and sources are archived under
`~/work/zensim-validation-2026-09-14/moment-response/`; experimental code is removed.
Next investigate shared pixel-response computation against the preserved finite
witnesses and native interventions. Avoid more per-query tile/capacity sweeps.
The complete product goal remains active; no candidate qualifies. All jobs finished.


## September 14, newest: bounded replay passes accuracy but fails the fine-grid cost

[Tile-response study](benchmarks/tile_response_2026-09-14.md) uses existing
kernels on fixed 32-column cells with real halos and complete model inference.
All 138 checks on the same six models / 23 TRAIN pairs pass the preregistered
approximation gates: minimum rank .999555, maximum gain error .008803 points.
Row-local pyramid updates and reusable scratch preserve all 9,756 predictions
exactly. Synthetic full-pyramid/restoration/public-feature and admission checks
pass. No fitting, calibration or EVAL/test access.

Cost still blocks it. Final basic228 sweeps cost 3.93–17.30 scalar calls in
feasibility timing. Read-only JXL inspection confirms bin8 / 8-aligned queries;
a separate finest-grid cost probe on admitted 1024² row4589 takes 3.144 seconds
for basic228/H32, 59.28 scalar calls. Tile replay dominates (2.858 seconds).
Dense8 accuracy and native RD are unmeasured; large-image accuracy witnesses
remain block64. Timings are not p95 qualification or an optimized narrow model.

Both prototypes are archived under `~/work/zensim-validation-2026-09-14/tile-response/`
and removed from the implementation. Production arithmetic is unchanged; only
the stale V1BasicSums peak-deprecation comment is corrected. Next share moment
work across queries / bound interior-and-boundary response and whole-head cost,
using the real caller's geometry. Do not repeat per-query tile-size sweeps or
promote the coarse screen to a fine-grid claim. All jobs finished. The complete
production goal stays active; no model qualifies.

## September 14, newest: exact finite response works; full-width bands cost too much

[Band-response feasibility](benchmarks/band_response_2026-09-14.md) reuses the
canonical band/kernel/finalizer owners and complete public ensemble inference.
On the same 23 TRAIN pairs and six frozen models, all 9,756 repair gains equal
prior public-pixel rescoring exactly; all 138 signed ranks are 1.0. Synthetic
public-feature/identity and negative admission controls pass. No fit or EVAL/test.

Reject this prototype for dense steering: basic228 sweeps cost 7–57 scalar calls
in initial feasibility timing, with band work alone 5–36 calls. These are not
p95 qualification measurements. Full RGB/pyramid work is included, and narrow
models still extract 228 features. No sparse speedup or public API is claimed.
Prototype source/patch/binary and evidence are archived under
`~/work/zensim-validation-2026-09-14/band-response/`; production source is restored
exactly to `fcc86e39`. Existing maps and failed gates remain unchanged.

Next bound the cost and math of interior signal removal plus changed-boundary
recomputation through existing owners, including new extrema and whole-model
inference. Preserve geometry/normalization and measure approximation error against
these exact responses. Do not repeat the same capacity/ablation/spread studies.
All scalar/corruption/targeting/native-RD/HDR/p95/frozen qualification requirements
remain; no model qualifies and the full goal stays active. All jobs finished.

## September 14, newest: precision is too small to explain the large map errors

[Rev3 precision probes](benchmarks/precision_rev3_2026-09-14.md) extend the
existing test-only instrument to explicit revision/form admission and direct
f64 basic-edge reference using already computed centered means. Five selected
blocks on three admitted TRAIN pairs reconstruct all served SSIM/basic-edge
features and weighted contributions against the public-API reports. Maximum
reference differences are .000234 SSIM and .000090 edge score points, far below
the large map errors. No new kernel, inference arithmetic, model or API.

All base signals meet the prior synthetic 1e-3 accuracy bound, but all five
real probes exceed its 2e-5 locality tolerance (peak 1.16e-4). Existing synthetic
controls still pass; they do not establish a universal bound. The exact direct
reference remains local. Preserve this limitation without confusing it with
the much larger finite-repair approximation failure.

Next inspect existing local recomputation/cache work and prototype finite
neighborhood response using existing extraction owners; evaluate the complete
model on predicted features rather than relying solely on frozen gradients.
Register accuracy/runtime limits and keep the same six-model/23-pair controls.
No more capacity or family-removal sweep is justified by these results.
Artifacts: `~/work/zensim-validation-2026-09-14/precision-rev3/`.
All diagnostic jobs finished. No EVAL, TEST, calibration or fitting. The full
production goal remains active; no model qualifies.

## September 14, newest: diagnose failures and compare existing family subsets

[Spatial diagnosis](benchmarks/spatial_diagnosis_2026-09-14.md) uses 69 saved
Rust analyses and six matched family-diagnostic calls. On the tiny report,
basic228/H32 M3f rises .584→.984 only when observed max response is substituted
(oracle, unavailable to runtime). On the screenshot, edge/L8/max contributions
cancel strongly and correcting max alone does not fix rank. No kernel bug is
proven; frozen signal removal differs from actual finite neighborhood edits.

A separately registered wider screen adds three existing frozen controls on
the same 23 TRAIN JXL pairs: local120/H128 passes 18, basic156/H128 passes 19,
basic192l8/H128 passes 20. All still fail the tiny screenshot. The combined
six-model gallery has 138 checks and 24 failures. Y60/H32 remains the only
23/23 spatial pass, with weaker human rank; it retains coarse chroma, not pure
luma. Every new native pixel/cache score witness passes. No fit or inference
change, EVAL, TEST or calibration access; no candidate qualifies.

Next isolate chroma/luma and scale-specific finite neighborhood response with
the existing retained feature/kernel diagnostics. Removing hard maxima alone,
finer storage, or another capacity sweep is not supported by these results.
The full scalar, corruption, native targeting/RD, HDR, p95 and qualification
requirements remain. Artifacts: `~/work/zensim-validation-2026-09-14/spatial-diagnosis/`.
All 75 new candidate cases / 5,088 repairs and saved analyses finished.

## September 14, newest: wider spatial screen rejects the five-case shortcut

[Registered coverage/results](benchmarks/spatial_coverage_2026-09-14.md) test
three frozen ensembles on 23 native JXL TRAIN development pairs across five
origins, two size bands and exact q5/50/90 nominations. Seven of thirty cells
are unavailable. All 69 calls / 4,878 repairs have finite supported maps and
native-pixel/cache-score parity. Basic228/H128 and H32 each pass 17/23, while
y60/H32 passes 23/23 but has weaker human TRAIN rank. No model is qualified.

Failures split into model-linearization (M2) and map-prediction (M3f) limits.
Tiny report/screenshot cases expose M3f failures and large magnitude errors.
Six separately registered bin1 follow-ups / 210 repairs leave all ranks
unchanged; scalar, actual-gain and linearization records remain exact. Finer
storage does not fix these aligned-rectangle failures. Finite moments also
worsen rank on some cases, so the earlier five-case improvement is not universal.

The gauntlet A/B gallery retains all 69 checks, initially showing 12 failures.
Its date/scope, model count and block-size controls now follow the packet.
Next use the existing per-feature diagnosis on these TRAIN failures before
changing feature/head formulas; investigate M2 separately from map projection.
Do not promote the cheap spatial control using this narrow screen. Native RD,
corruption, full scalar assessment, HDR, p95 and frozen qualification remain.

Artifacts: `~/work/zensim-validation-2026-09-14/spatial-coverage/`.
All 75 public-API calls and rank analysis finished. No new fit, calibration,
EVAL or TEST access. The full production goal remains active.

## September 14, newest: aligned bins improve complete finite-map cost

[Aligned bin folding](benchmarks/aligned_bins_2026-09-14.md) adds a private
fast path for coarse footprints contained in aligned bins. It preserves the
original per-bin addition chains, multiply order, partial edges and fallback
paths. An independent footprint-intersection reference passes 448 folds; all
forty previous TRAIN JSON reports remain exact (1,176 repairs). No feature,
model, score, gradient, density or rectangle-prediction changes occur.

Three bracketed complete-call timing processes pass the registered acceptance
rule. Basic228/H32 enabled medians improve from 132.09/544.36 ms to
117.76/485.68 ms at 1024²/2048²; the second old control confirms the gain.
Across all four profiles, enabled medians improve 7.7–13.1%; disabled controls
range from 1.9% faster to 0.6% slower. All arms have thirty accepted one-call
rounds with clear owner reliability flags; a 1MP drift flag is retained and
the gain holds against both controls. The option remains off by default.
RSS bounds fit the worker cap on these synthetic bin8 inputs. p95 and isolated
query latency remain unmeasured, and the complete-map cost gap remains open.

Broad attribution checks (38 library passes, two existing ignored, ten SIMD
permutations and the revision driver), finite-moment/session, HDR retention,
minimal feature build and Clippy pass. The cookbook now marks its historical
M2=1 and basic-156 map limits as dated findings. No model is qualified.

Metadata census of the existing admitted TRAIN packet finds renditions through
1024 pixels; four of the five spatial-screen sources have that size, while
the screenshot source stops at 384. Next register broader TRAIN spatial checks
across content, size and codec quality, including basic228/H128 (stronger human
rank) against H32. Use existing native decodes and public scoring; no new fit
is yet justified. Max retention and repeated map/extraction work remain measured
cost targets; export samples/quantiles at the benchmark owner for p95. All
larger human/corruption/targeting/native-RD/HDR/frozen-EVAL requirements remain.
The full production goal stays active.

Artifacts: `~/work/zensim-validation-2026-09-14/aligned-bins/`. All build, test,
replay, timing and RSS processes have finished.

## September 14, newest: reject row fusion; profile the larger map costs

[Row experiment](benchmarks/moment_rows_2026-09-14.md) reproduces all forty
finite-moment TRAIN reports exactly (1,176 repairs). Three bracketed timing
processes measure modest enabled-map gains (1.2–2.1% at 1MP, 3.7–5.7% at 4MP)
but consistent disabled-map slowdowns (0.3–1.8%). Reject the added implementation
complexity: production attribution is restored exactly to `7d964c3f`. Candidate
source/patch/binaries and all results remain reproducible in the artifact packet.
The existing SIMD integration owner gains finite-moment scalar/density parity,
finiteness and signed-curvature checks; all ten dispatch permutations pass on
both versions. Shipping-source Clippy passes. No public API or scalar change.

A 3,230-sample hardware profile of the shipping basic228/H32 enabled map path
at 4MP puts 20.65% self cycles in bin folding, 17.63% in max retention and
24.48% in horizontal/vertical SSIM kernels. Use flat instruction-pointer
attribution only: some reconstructed caller frames are unreliable. The profile
includes reference setup and thirty map constructions, no rectangle queries.
The initial query label is transparently corrected in retained metadata.

Next register an aligned dyadic fast path in the existing bin owner, preserving
summation order, partial edges, arbitrary-bin/fractional fallbacks and disabled
controls. This is a larger measured target than repeating the edge-power work.
No new fit is justified by this profile. Complete p95 measurement still needs
the canonical benchmark owner to export samples/quantiles; registry zenbench
0.1.9 currently exposes summaries only. All larger scientific/product gates
remain open; no model qualifies and the full production goal remains active.
Artifacts: `~/work/zensim-validation-2026-09-14/moment-rows/`.
All timing, RSS, profiling and test processes finished.

## September 14, newest: base-image finite moments work, but cost more

[Finite moments](benchmarks/finite_moments_2026-09-14.md) implements an opt-in
`BakeScorer::with_finite_moment_refinement(bool)`, default false. The shared
retained map owner builds binned L2/L4/L8 mass integrals and applies finite
removal corrections in `ScoredAttribution::refinement_gain`. Only base-image
signals enter prediction. Scalar scores, features, sensitivities, old density
and disabled refinement values remain exact on the registered TRAIN checks.

Forty complete cases / 1,176 repairs cover all four frozen basic/peak ensembles
at bins 1 and 8. Basic228/H32 now passes all five M2/M3f cases at both bins;
the screenshot M3f rises .6739→.9200. The H128 models retain their separate M2
failures. This is actual public-API prediction, unlike the preceding oracle,
but still a small TRAIN mechanism screen, not native encoder RD qualification.

Cost blocks recommending it: basic228/H32 prepared median is 131.97/544.58 ms
at 1024²/2048² versus 84.22/338.19 disabled, with 30 accepted paired single-call
rounds. Process RSS is 127,144/453,848 KiB, below the declared worker cap on
these synthetic inputs. p95 and isolated query latency remain unmeasured.
Keep it opt-in. Next optimize shared retained accumulation and signal-power
reuse against these exact outputs before promoting or widening this storage.

Finite-power/reuse checks, 37 attribution tests (+2 existing ignored), the
revision driver, two HDR retained-feature tests, minimal custom-profile build,
Clippy, API snapshots/check and 196 semver checks pass. An unsorted synthetic
test-bake declaration was correctly refused, then fixed; no guard was relaxed.
No training, calibration or EVAL/test access. Artifacts:
`~/work/zensim-validation-2026-09-14/finite-moments/` (`SPATIAL.json`, `timing/`,
`rss/`, original registration and binaries). All experiment processes finished.
Human rank, above-identity scores, corruption, broader spatial coverage,
native bounded targeting/RD, HDR and full frozen qualification remain required.
The complete production goal remains active.

## September 14, newest: finite-root curvature is worth correcting

[Moment diagnosis](benchmarks/moment_curvature_2026-09-14.md) reuses all twenty
saved TRAIN intervention reports (588 repairs), with zero new pixel comparisons
or fits. The Rust saved-data analysis now separates finite L2/L4/L8 root curvature
from remaining map error. All prior fields are exactly unchanged. Mathematical
tests, inconsistent/singular/legacy input controls, Clippy and hygiene pass.

For basic228/H32, a curvature-only oracle raises the failed screenshot's M3f
from .6739 to .9148; all five cases pass M3f, and aggregate squared error against
observed feature linearization falls 92.9%. This meets the preregistered bar for
implementing a base-image finite-moment approximation. **It is not a runtime map
or qualification**: the oracle uses repaired-image features. Production scores,
maps, M2 and above-identity failures are unchanged. Other H128 heads' M2 failures
remain distinct even when their root-corrected M3f passes.

Next derive finite pool removals from base-image moments through the existing
retained attribution owner; test frozen-signal versus actual-neighborhood effects
before choosing a per-feature storage layout. Preserve the current density and
scalar scores as controls; measure prepared/query cost. No new optimizer or fit
is justified by this result. Identity ordering, human quality, corruption,
broader TRAIN spatial coverage, native bounded targeting/RD, HDR, frozen EVAL
qualification and p95 still require work. The full production goal stays active.

Artifacts: `~/work/zensim-validation-2026-09-14/moment-curvature/`, especially
`RESULT-final.json`, `analysis-final/`, `CONTROLS.json` and the input registration.
The initial analysis and guarded replay are preserved and numerically identical.
All experiment processes finished; no EVAL/test or calibration was used.

## September 14, newest: basic feature decomposition and a spatial mechanism

[Basic/peak comparison](benchmarks/product_peaks_2026-09-14.md) completes twelve
paired TRAIN fits, four full ensembles, 21,344 canonical panels, source-cluster
uncertainty, twenty native audits and 588 pixel repairs. Fit/pack/export takes
214 seconds, scalar panels five more. Raw/packed predictions are bit-exact on
31,548 rows. No EVAL/test data or calibration were used.

Hard-max pools buy most of basic228's codec rank gain: basic156 .8998,
L8-only .9156, max-only .9489. Their removal saves about 27% prepared-map median
cost but essentially no scalar time. Basic228/H32 retains codec rank .9549 and
passes all five M2 / four M3f cases, but loses .0197 human SROCC versus H128 and
has four distorted scores above 100. None qualifies. Full raw tails, scatter,
Mohammadi and per-codec/source/class panels are retained; MAE is supplementary.

For H32's remaining map failure (TRAIN screenshot row 5316), M2=1 and M3f=.6739.
Oracle max substitution worsens it; edge-only or L8-only observed contributions
bring M3f above .70. This isolates a non-max approximation issue, not a proven
new formula bug. Existing L8 tests already establish finite-removal curvature
can be up to eight times the derivative estimate. Next separate finite pool
curvature from changed blur neighborhoods using existing diagnostic owners;
preserve scalar semantics and measure prepared/query cost. Human-rank loss,
identity ordering, broader TRAIN coverage, corruption, native targeting/RD,
frozen qualification, HDR and p95 remain outstanding. Full goal remains active.

Artifacts: `~/work/zensim-validation-2026-09-14/product-peaks/`. The descriptive
bootstrap initially refused an old rounded control export; corrected to the
prior verified round-trip export without weakening assertions. All experiment
processes finished. Current source is unchanged; this packet adds scientific
evidence, not a new production inference implementation.

## September 14, newest: coarse-feature frontier and a real attribution fix

[Scale frontier](benchmarks/product_scales_2026-09-14.md) completes 18 paired
TRAIN fits and six uniform ensembles on the preceding admitted packet. Basic228
has distorted-codec SROCC .9532 / human .9048, versus full944 .9605 / .9193.
It costs 17.01 / 79.87 ms scalar at 1024² / 2048², versus full944 53.30 /
240.65 ms in the same interleaved run. Basic228 prepared maps cost 79.55 /
339.27 ms and pass only 2/5 registered JXL repair cases. All five richer new
profiles pass only 1/5. None is qualified. Rich-v2 prepared sessions remain
explicitly unsupported by the existing basic/peak-only session contract.

Reproduced a coarse v2 weighted-MSE NaN through the public API on synthetic
data, then fixed the retained attribution loop: cells with zero sensitivity
are skipped before normalization by their possibly zero sample counts. No
feature formula, scalar score, training label or gate changed. Both y346
ensembles now have finite blocks on all five TRAIN JXL cases, but still fail
four. Other four ensembles' 20 complete block records are exactly unchanged.
All six pass native consumed-feature and pixel/cached score audits (30 pairs).
Historical selected619 EVAL map failures predate this fix; their immutable
verdicts have not been requalified on the corrected build.

36 attribution tests pass (two existing ignored diagnostics), two HDR parity
tests pass, and the feature-only build, Clippy, formatting and hygiene checks
pass. Fourteen isolated process RSS upper bounds fit the declared memory cap
on synthetic benchmark inputs. Timings have 30 accepted one-call rounds but
the benchmark exports summaries, not p95: latency qualification stays missing.
No EVAL, TEST or calibration rows were read in this campaign.

Artifacts: `~/work/zensim-validation-2026-09-14/product-scales/`, especially
`fits/`, `assessment-panels/`, `spatial-before/`, `spatial-after/`,
`native-audit/`, `timing/`, and `rss/`. Next separate basic228's HF/peak signal
benefit from expensive max-map work and head nonlinearity, retaining y60/H32
as the spatial control. Broader corruption/high-quality data, frozen evaluation,
native bounded targeting/RD and p95 still require work. Full goal remains active.

## September 14, newest: representative TRAIN mix and complete served development

[Product training results](benchmarks/product_train_2026-09-14.md) recover
10,499 native four-codec/identity TRAIN pairs across 148 sources and 141 families,
with historical reserved families and eval8 excluded. Fit/development/calibration
are family-separated inside TRAIN; 923 calibration rows remain unused. Human
supervision uses 7,000 TRAIN fit and 1,000 source-separated internal-development
rows. No new EVAL or TEST access; the frozen six-panel comparison below is unchanged.

Stopped eight initial fits when Rust sampler replay showed the declared 1:1 mix
was actually 8.88% human under global stratified sampling. Corrected uniform
sampling reaches all rows and gives 49.881–50.198% usable human pairs. All 24
replacement fits and eight uniform ensembles finish, with actual sample digests
matched to replay. The trainer now explains this legacy weight behavior without
changing its sequences. Round-trip score export replaces six-place rounding;
63,096 real raw/packed predictions are bit-identical. Canonical Rust panels and
scatter cover every seed/ensemble; undefined small-group statistics remain null.

On distorted TRAIN codec development, full944 plain ensemble SROCC is .9605,
versus matched native B .8488 / D .8494. Human TRAIN development is .9193.
This supports further development, not independent superiority or qualification.
Only 57 high-quality distorted pairs are present. Both y60/H32 ensembles pass
all five small JXL spatial cases; some larger heads fail M2, full944 remains
unsupported. Complete spatial scalars match cached predictions within 7.11e-15.
No native encoder RD, targeting or controlled runtime evidence is added.

Artifacts: `~/work/zensim-validation-2026-09-14/product-train/`: `fits-uniform/`,
`assessment-exact/` (verified score TSVs; its first JSON harvest failed),
`assessment-panels/RESULT.json` (completed panels), `spatial-train-v2/`, and
`incumbents-v2/`. Failed initial sampler/spatial/serialization/tool invocations
are preserved. All new models are TRAIN development only. Next use this packet
to test inexpensive per-scale features that recover scalar accuracy with full
spatial support, then frozen qualification and native product gates. The full
assigned production goal remains active.

## September 14, latest: six measured panels for the frozen Rev3 comparison

[Codec-family assessment](benchmarks/rev3_codec_eval_2026-09-14.md) adds all
20,655 distinct admitted imazen26 validation pairs: 6,953 broad, 6,142 nonphoto,
7,717 near-lossless proxy memberships. These overlap and form one historical
SSIMULACRA2 instrument. All nine candidates and matched B/D now have six
populated panels; previous human results are unchanged. No fitting or test reads.

B/D outperform every new candidate on the three aggregate proxy point scores.
Full944/H128 has signed SROCC .6776 / .6202 / -.0607, versus B
.8603 / .8498 / .3500. Wide models emit distorted scores above 100; small
models compress the near-lossless range. Source-cluster uncertainty and
same-decoded-pixel teacher drift are retained. Complete B/D native image scores
match every new verdict prediction exactly. The new comparison is served on the
all-rows gauntlet; all nine remain NO-SHIP with composite coverage only 3/6.

Artifacts: `~/work/zensim-validation-2026-09-14/rev3-codec-eval/`.
The original extraction binary and protocol are preserved. New family candidate
scores use public BakeScorer with native canonical features; complete per-model
image/map audits on this added population remain unmeasured. Selected619's
prior spatial failure and full944's unsupported refinement remain blockers.
The full goal remains active: representative train-side development, complete
composition/spatial/corruption qualification, native bounded targeting/RD and
controlled p95/memory still require work. Do not tune the frozen models on eval.

## September 14, latest: frozen Rev3 models now have broader matched eval

[Qualification expansion](benchmarks/rev3_qualification_2026-09-14.md) adds
404 KonJND JPEG SELECT and 436 KonFiG origin-validation pairs to the existing
3,125 KADID SELECT pairs. All nine frozen candidates, native matched B/D, and
CPU SSIMULACRA2 have measured panels. No fitting, calibration or test access.
New models improve KADID but trail B/D on KonJND: best new y60/H32 .4566 versus
D .5883; source-bootstrap paired delta [-.184, -.080]. Full944/H128 is .1776.
All nine fail measured G-OUT clauses. Wide models exceed 100 on distorted
pairs. Selected619 fails spatial/scalar parity on 836/840 added pairs; the
other eight complete native audits, but full944 still lacks full refinement.

The canonical report now marks incomplete composite coverage explicitly,
retaining partial arithmetic only as a diagnostic. The new matched-eval
discussion is on the all-rows gauntlet. CID22/T0 terminal populations remain
excluded; KonFiG is a four-source instrument and was used to tune SSIMULACRA2.
B/D's absent historical era metadata remains unqualified, with every reported
prediction checked against their current native Rust surface.

Artifacts: `~/work/zensim-validation-2026-09-14/rev3-qualification/`, especially
`verdicts-coverage/`, `pixel-audits-v2/RESULT.json`, `UNCERTAINTY.json`, and the
retained failure logs. The assigned full production/science goal remains active.
Next broaden admissible imazen-26/dial/corruption coverage, reproduce serving
defects on train, and complete native targeting/RD and controlled p95/memory.

## September 14, latest: canonical assessment and an integrity-head prototype

Gauntlet display correction: the main table previously omitted KADID and the
full panel defaulted to global CID22, hiding the new models’ only measured rank
corpus. The table now exposes measured corpus/count and KADID SROCC, and the
Rev3 discussion selects KADID. Sparse comparisons default to a measured corpus;
a new gate checks actual displayed values against the stored results. This
does not fill missing cross-corpus or product evaluation. An empty/partial
knob-end instrument now stays unmeasured instead of acquiring a vacuous pass.

[Protocol and results](benchmarks/steering_integrity_2026-09-14.md) now own the
latest work. `zenstats::scatter` computes full-population geometry, robust
outlier envelopes, raw tails, density and saturation; `bake_verdict` stores it,
`panel --scatter` exposes the same owner, and gauntlet renders it. Missing peer
bars/range/axes produce INCOMPLETE, never PASS. Read raw density alongside rank
normalization: the latter can erase smooth output compression. The frozen
49-model KADID study is reassessed without fitting or changing raw predictions;
it still lacks cross-corpus/product qualification.

One train-only D228/Rev1 HGB corruption head, with a fixed train-calibrated
probability threshold 0.9, detects 175/180 admitted severe eval cases and 109/113
bug reproductions, with zero activations on 320 honest controls. Five misses
remain. Another 1,086/5,173 unlabelled weak/ambiguous pixel pairs activate; that
is unresolved operational specificity, not successful detection. Full catalog:
6,024 attempts / 5,679 distinct pairs. No test segment was opened.

Prepared steering returns `CorruptionDetected` on activation and preserves the
perceptual map on acceptance. Real-model audit: 175 rejected, 325 accepted,
16,224 exact finite spatial queries. This proves pipeline parity, not native
allocation quality. D+head is about 26.91 ms scalar / 59.18 ms prepared at
1024² ST; no detectable head overhead in 30 rounds, no p95/memory qualification.
Artifacts are in `~/work/zensim-validation-2026-09-14/steering-integrity/`.
One deterministic replay has identical predictive bytes; original head and eval
remain frozen. No default changed and no model is product-qualified.

Next work: freeze better severity/operational labels on train, broaden allowed
honest low-quality controls (including JPEG/WebP and local allocations), then
qualify the perceptual base and complete composition against human panels,
raw/shape tails, HFNL/dial, per-image bounded 1/2/3-shot targeting and native
spatial RD. Keep existing gates and avoid selecting on this eval's five misses.

## September 14, latest: restore the composite/panel/scatter interpretation

The user corrected the MAE-led summary. Read the
[metric correction and frozen-model panel](benchmarks/minimal_top_2026-09-13.md#september-14-correction-use-the-full-quality-evidence)
first. MAE is auxiliary calibration evidence. Historical selection combines
named composites with floors, Mohammadi/per-band/per-reference panels,
raw G-OUT residual tails and shape-normalized scatter diagnostics. Geometric
scatter shape is distinct from `geomean3`. Existing owner statistics are
retained with hashes in `benchmarks/minimal_top_2026-09-13.metric_review.json`.
The KADID-only study has no cross-corpus composite or matched peer tail bars;
no winner or product qualification follows. No retraining or test reads.

## September 14, latest: strict minimal/wide comparison and identity correction

[The completed comparison](benchmarks/minimal_top_2026-09-13.md) fits 40 MLPs
(five paired seeds across eight layouts/head sizes) plus a train-only linear
control. Train is 8,000 canonical KADID/TID pairs; eval is 3,125 source-disjoint
canonical KADID SELECT pairs. No test/terminal segment is opened or retagged.
Final Rust evaluation carries native-decoded identity evidence for 100 pairs;
feature-only panels before that correction are superseded for served-score claims.

Coarse60/H128 ensemble: raw MAE 8.142, 8.18 ms scalar / 26.87 ms prepared bin-8
map at 1024² ST, four of four limited photo/JPEG repair cases pass. Full944/H128:
MAE 7.029 at 53.97 ms scalar, unsupported complete refinement. H256 does not
improve it. Selected619 has invalid/nonfinite maps and fails spatial/scalar
parity. The screen now refuses to treat finite summary correlations over invalid
block predictions as a pass. No model is qualified or installed as a default.

49 actual final Rust verdicts are on the all-rows gauntlet, behind its minimal /
wide review button; the fair view correctly excludes them for missing CID22.
Both board gate suites pass. Source/control/identity tests, Clippy, script lint
and API-doc checks pass. All seeds, final bakes and timings are retained under
`~/work/zensim-validation-2026-09-13/minimal-top/`; use `final-eval/` and
`FINAL_RESULTS.json`. Next: improve/validate maps and corruption, broaden legal
eval content, then native targeting/RD and p95/worker cost. Four photo sources
are diagnostic coverage, not a native codec or screen-content qualification.

## September 13, latest: explicit train/eval isolation

The user's later [split ruling](docs/DATA_SPLITS.md#september-13-user-ruling-train--eval-only-never-touch-test)
forbids every test/terminal read, even for final qualification. Train alone
owns fitting/calibration/checkpoints; eval owns post-fit gates. The existing
feature-screen owner now requires v2 recipes with separate admitted train/eval
segments and rejects legacy recipes/caches. Predictions and pixel/spatial
audits run only on admitted eval rows after fitting. Historical results remain
unchanged; do not rename their test segments or reuse their mixed caches.
Boundary regression checks and a synthetic Rust prepare/train/eval/report
smoke pass. No real corpus/test segment or new competitive model was used in
this policy implementation. Next admit fresh train/eval manifests against
canonical source/family rules before continuing the feature comparison.

## September 13, latest: rigor review before extending the feature set

The [runtime study's Claude-era gate review](benchmarks/runtime_profiles_2026-09-13.md#claude-era-gate-review-after-the-runtime-study)
records the continuation requirements and audits the actual manifests. Its 96
repair cells already cover four content classes, but only two training origins
per class; coarse60's sole failure is screen content. The selected mixture's
96/96 is development evidence. The new candidates have no full gauntlet rows;
the existing fair board's render/data gates pass, which does not qualify them.
Compare matched per-seed models and final ensembles separately, use one spatial
instrument/rule, retain floor/tail/native-loop gates, and measure p95 before
claiming a wider feature set improves the product. No new training in this audit.

## September 13, latest: measured runtime-profile candidates

[Runtime profile study](benchmarks/runtime_profiles_2026-09-13.md) continues the
local-subset study: 56 new MLP fits, two deterministic linear controls, and
13 complete compositions. Coarse X/B moments now skip actual Rust work when
their local-feature consumers are absent; canonical SDR/HDR and cached values
remain intact. No feature arithmetic or named model changed.

Y at all four native scales plus X/B only at 1/8 resolution gives 60 features.
Its five-MLP ensemble achieves human development-test MAE 7.487 and 95/96 broad
spatial passes. A mixture with 2/7 weight on its linear+spline control passes
96/96 at MAE 7.898, versus 10.188 for the prior consistent linear120 control.
These are development-selected results, not independent qualification.
The smaller-head 80/100/120 variants and luma-only control retain their failures.

All candidates still over-score the retained salt/pepper and aliasing examples;
the new severity-aware corruption head is not fitted or attached. Prepared
steering still refuses companions. Next: reviewed corruption labels/operating
points and inactive-companion steering support; improve the remaining map/scalar
cost, then independently validate frozen candidates in native codec loops.
Artifacts: `~/work/zensim-validation-2026-09-13/runtime-profiles/`.

## September 13, latest: thresholded corruption minimum and severity policy

The [activation contract](docs/CORRUPTION_ACTIVATION_2026-09-13.md) records the
user's clarification after the subset study: the catcher signals integrity
failure, not ordinary spatial allocation damage. The Rust composition is now
`C < T ? min(P, C) : P`, with equality inactive and negative tails preserved.
The fit screen checks false activation independently of score lowering, which
can hide false positives on already poor valid encodes. Review catastrophic,
recoverable, valid low-quality and ambiguous strata before the next fit;
historical synthetic labels and q20 ordering do not establish that distinction.
No labels, trained artifacts or qualification results were changed. Prepared
steering still refuses companions; inactive pass-through and an explicit failure
return remain the next API step, without a corruption-gradient requirement.

## September 13, newest: fast local subsets and prepared steering API

[Steerable subset study](benchmarks/steerable_subset_2026-09-13.md) follows the
scale-selective study below: 195 unique MLP fits, nine deterministic linear
controls, five-seed finalists and broader real block repairs. Local120 omits
peaks/HF ratios/weighted/v2 families; fine-Y local100 also omits finest X/B.
The existing SIMD extraction and map owners skip actual unused work.

`BakeScorer::prepare_steering` binds source/cache/scratch, checks complete
refinement coverage and serves declared basic/peak Rev1/2/3 arithmetic in one
process. `with_parallel(false)` gives outer workers scheduling control.
Cached scalar extraction now reuses the reference pyramid. A real HF-gain
map derivative bug was fixed using its existing canonical derivative owner.
Gradients now skip structurally unread inputs through that same read-proof
owner. The existing densifier reduces MLP artifacts from 499KB to 58–70KB,
preserving all saved ensemble predictions and 288 spatial case results.
The actual fine-Y ensemble takes 10.09ms scalar/32.86ms score+map at 1MP ST,
6.62x faster than fast-ssim2 for scalar. The 1MP map/scalar mean remains
above 3x; its p95 budget is not established. Full timings are in the study.

The linear local120+spline control passes 96/96 broad development cells,
25,248 real repairs, but fails corruption quality (saltpepper 76.28, RB 55.44,
alias 37.73). Five-MLP ensembles improve human test MAE to 7.497/7.560 but pass
only 84/96 and 87/96 spatial cells. These are **unqualified controls**, not new
shipping profiles. The public session refuses scalar-only companions rather
than silently omitting their contribution from steering.

Next: improve local corruption/aliasing sensitivity and head finite-edit
behavior together, retaining these controls. Only after development gates pass,
freeze a candidate for terminal human, witnessed per-image target bounds,
train-calibrated 1/2/3-shot and native JXL/AVIF/JPEG/WebP spatial RD validation.
Those release gates have not been run for these failed candidates. No named
model/default changed. Artifacts and executed commands:
`~/work/zensim-validation-2026-09-13/steerable/`.

## September 13, latest: real Rev3 scale dispatch and prime-scale training

[189-fit study](benchmarks/scale_selective_944_2026-09-13.md) supersedes the
"v2 read sets save no compute" limitation below. Declared IDs now narrow actual
v2-era scale execution, including adjacent gradient dependencies. Native619
costs 20.725ms versus 51.691ms ST and 7.521ms versus 17.843ms MT8 on the fixed
1MP benchmark, with consumed features unchanged. No shipped model changed.

The same 19,958-pair corpus supports per-scale legacy/v2 masked+IW tests and
fresh direct Triangle 1/2/4/8, 1/3/5/7 and 1/2/3/5 training. Prime schedules
have mixed accuracy and cost; weighted pools alone do not recover full944's
corruption benefit. Native619 remains a scalar candidate, but fails spatial
repair checks. Full944's legacy weighted IDs remain unsupported for refinement.
Fix those spatial failures and broaden codec/corruption source coverage next;
no universal feature ceiling, native target-loop gain or model qualification.

Independent mathematical checks found zenresize's float path rounds source
rows to binary16. Sampling metadata explicitly includes this precision;
the old v1 cascade already had it. Do not describe either as unquantized f32
convolution or assume near-identity fidelity. All9 new sampling contracts have
independent quantization-aware convolution and serial/threaded parity tests.
Artifacts: `~/work/zensim-validation-2026-09-13/scales944/`.

## September 13, later: representative feature/scale capability study

[162-fit study](benchmarks/feature_ceiling_2026-09-13.md) supersedes the small
proxy screens for feature capability claims. Fresh Rev3 full944 extraction
covers 19,958 human/codec/corruption pairs, three seeds, scale read sets,
H32/H128/H256 capacity and half-data controls. Data and checkpoint selection
still limit the results; no feature ceiling or model qualification is established.
Full944 substantially improves corruption detection; dropping finest chroma is
not universally safe. Reduced v2 read sets currently save almost no compute.

All 162 bakes passed native pixel audits. Spatial checks expose a separate
blocker: supported v2 rectangle predictions can have very low repair coherence
despite a good local linear score response. Next: fix those retained failures,
then implement actual per-scale v2 dispatch with the 619/800/full944 controls.
Increase codec/corruption source coverage before interpreting a training plateau.
Cached H128 fits take roughly two minutes; H256 controls exceed five minutes.
The report records explicit raw errors; legacy Rust `panel` MAE is remapped.
Artifacts: `~/work/zensim-validation-2026-09-13/ceiling/final/`.

## September 13: actual coarse-pool dispatch and a one-minute screen

[Coarse-pool dispatch](benchmarks/coarse_pool_dispatch_2026-09-13.md) now derives
the actual v1 masked/IW scale mask from declared feature IDs. It preserves
full-resolution Y, omits unread finest X/B, and skips the fine weighted-pool
kernels. Default extraction retains its full feature set. Canonical consumed
values and scores are unchanged. The existing feature-screen stage now also
runs hash-bound training corruption/JXL block-repair diagnostics within its
five-minute deadline; unsupported maps are explicitly reported.

The next fundamental feature work should target **coarse v2 families with
spatial integrands**, plus fine-scale impulse/aliasing sensitivity. Legacy
v1 masked/IW are now inexpensive scalar controls, but lack spatial support
and did not fix salt-and-pepper in this small proxy fit. Do not promote them
as steering candidates or read a spatial pass as perceptual approval of JXL.
Whole-image R/B swaps already receive negative scores; the user accepts that
rejection and lowers their spatial-correlation priority. The proposed extra
swap features below are accordingly low priority.

## September 13: spatial failure gallery

The [sampling report](benchmarks/sampling_serving_2026-09-13.md#served-ab-inspection-gallery--september-13-follow-up)
documents `gauntlet.py --spatial-gallery`: 420 recorded checks, 235 failures
shown by default, original A/B images, signed gain/error maps and selectable
reference-block repair previews. Static output is served by the existing
local gallery service under `/zensim/reports/spatial-failures-2026-09-13/index.html`.
Use this machine's configured LAN hostname. No scores or bakes changed.
The [coarse-feature report](benchmarks/coarse_legacy_features_2026-09-13.md#channel-swap-features-to-screen-next)
also records a proposed channel-correspondence feature screen, distinguishing
swap detection from correct spatial repair predictions; it is not implemented.

## September 13 follow-up: filters served, fitted and spatially tested

The [integrated sampling screen](benchmarks/sampling_serving_2026-09-13.md)
supersedes the filter-only status below. All 18 Triangle/Mitchell/RobidouxSharp
contracts (1.5×/2×/3×, full-Y or downsampled-XYB), plus two box controls,
are now executable through BakeScorer with explicit sampling metadata.
Sixty fresh Rust fits and public-API audits fit inside five minutes on the
fixed T2 proxy packet. Actual block replacements test spatial predictions;
none of these models is qualified, and impulse/color-swap failures remain.

Keep fine luma. The user's proposed coarse-only expensive features have
[legacy evidence](benchmarks/coarse_legacy_features_2026-09-13.md): B and
944 models read useful quarter/eighth-scale extras. The next bounded task is
private per-scale family dispatch, starting with coarse masked/IW on the
box/Y190 control, then coarse gradient/blockiness/append families. Merely
choosing coarse IDs still wakes global family computation today. Preserve
IDs and values, retrain, and measure actual skipped work and spatial edits.

Artifacts: `~/work/zensim-validation-2026-09-13/sampling/` under the existing
work share; full report and compact data are committed. No named bake changed.

## September 13: optional finest chroma and an eight-second experiment loop

The [channel/filter experiment](benchmarks/fullres_y_subset_2026-09-12.md)
adds automatic private const-generic omission of full-resolution X/B work for
eligible basic/peak declared-ID models, retaining full Y and coarse XYB exactly.
Full228 versus y190 costs fall 40–46% ST and 26–28% MT8 on the registered
synthetic 1/4MP benchmark. Current named models keep their inputs. Actual
zenresize main kernels now have coefficient/phase and speed measurements at
1.5×/2×/3×, including Triangle, RobidouxSharp and Mitchell; these alternate
pyramids are not yet integrated into scoring or spatial mapping.

`scripts/run_full_eval.sh --stage feature-screen` completes fresh Rev3
extraction, two Rust H32 fits and final BakeScorer pixel/cache audits in
8.75 seconds (8.47 with verified cache). The fixed 264-pair JXL packet uses
only admitted training origins and metric proxy labels. No protected holdout
scores, corruption/spatial/HDR qualification or competitive-model claim.
The earlier September 9 statement that no Rev3 model has been trained is now
superseded by these two **development** bakes; no qualified Rev3 model exists.
[Commands and limitations](docs/FULL_EVAL.md#five-minute-feature-development-screen-september-13-2026).
Next expand the fast packet to discriminating corruption/spatial examples,
and give any filter/scale variant explicit Rust serving/feature identities
before fitting it. See the report for negative controls and chronology.

> **Framing (user, 2026-09-09, verbatim):** *"zensim is about being a speedy
> and consistent dial and steering metric with useful spatial steering across
> both sdr and hdr."* And, on the SSIM precision correction: *"bounded error is
> fine, speed above minor flaws please! we must make this fast or the project
> has no purpose."* Read every accuracy/precision decision below through that:
> exactness is not the product, speed + consistency + useful steering are.
> A bounded, registered error that buys speed is the right trade; an exact
> kernel that costs +27-87% is not.


## September 9: SSIM correction — handed off, then taken up

The correction was written up as https://github.com/imazen/zensim/issues/61 for
someone else to tackle, and then tackled. The handoff draft that was sitting
uncommitted against `f7b9f39a` is preserved as its own labelled commit and the
integration was finished on top of it, so the earlier instruction to pause this
lane NO LONGER APPLIES — read the issue plus the commits above `f7b9f39a`
rather than the handoff bundle.

WHAT IS TRUE NOW: `FormulaRevision::Rev3` forms the v1 SSIM signal from a
DIRECT error moment inside the existing fused H/V pass (`blur::fused_blur_h_ssim`
carries `Σ(a−b)²` in its dead `Σab` plane; `fused::fused_vblur_features_ssim`
and the v2 dense kernels form the direct dissimilarity from it) — no second
traversal, bounded error (locality ≤ 2e-5, measured 4.277e-6 vs 4.886e-4
shipped). Basic, peak, masked and IW pools all consume that ONE retained
value; unsupported routes (`blur_passes != 1`) return an explicit
`ZensimError`; and bake/process revision disagreement is refused even between
two revisions that select the same luminance form. The exact f64 kernel
`ssim_form::stable_ssim_plane` is reference-only.

WHAT IS STILL NOT TRUE, and matters more: **no model is trained on corrected
features.** Revision 3 changes what the extractor emits, so every stored table
and every shipped bake is a revision 1 artifact. The default stays revision 1.
An old bake relabelled `3` is refused by the loader, deliberately — do not work
around that; re-extract and refit. Nothing here establishes codec RD gains or
product qualification.


Reviewed September 8, 2026. [CLAUDE.md](CLAUDE.md) contains current working rules;
[WAVE_PLAYBOOK](docs/WAVE_PLAYBOOK.md) contains the tool map and experiment cycle.
Older operational notes are in [docs/history](docs/history/).

Latest continuation [implements the stable SSIM kernel](benchmarks/stable_ssim_kernel_2026-09-08.md).
The Rust error-moment kernel passes direct f64 checks on 96 real-image planes,
with exact zero out-of-support movement in the four registered cases.
Synthetic/identity/reflection checks pass, and ten SIMD configurations produce
exact output arrays. Removing unnecessary mirror modulo and using existing
SIMD dispatch cuts three-plane means from 28.73/112.23 ms to 9.43/38.03 ms
at 1024²/2048². Scratch is 384/768 KiB; these are kernel-only timings.
**No feature era selects it yet.** Next integrate this existing kernel with
an explicit arithmetic revision and model/data refusal contract, replace
old SSIM moment work where possible, check full/cached/streaming and spatial
binding, then refresh affected features and fit/validate. Do not reimplement
or repeat the kernel mechanism campaign. The model remains unqualified;
all other product requirements remain active.

Latest continuation [isolates SSIM numerical drift](benchmarks/nonmax_diagnosis_2026-09-08.md).
The 23-cell family diagnostic saves 6,812 pixel interventions and per-feature
changes with all old fields exact. Replacing only SSIM's predicted contribution
resolves all 15 spatial failures; edge resolves two, other families none.
A direct f64 reference on the exact f32 XYB pyramid then proves production
SSIM signals change outside the edited pixels' mathematical window support.
In one screenshot the SSIM linear gain is -0.093813 in production versus
+0.000120 in the reference. Next repair canonical SSIM moment precision and
locality with an explicit feature-era contract, measure its cost, re-extract
and validate the affected candidate. Do not fit maps to floating-point drift,
retune maxima, or relabel old bakes/features. The current spatial screen stays
FAILED; no product arithmetic or default model changed. Evidence is in the
September 8 `nonmax-diagnosis` delivery; all other product gates remain open.

Latest continuation [adds finite max rectangle estimates](benchmarks/max_attribution_2026-09-08.md).
`ScoredAttribution::refinement_gain` handles max ties and source footprints;
its separate coverage report is empty on all 792 fixed blend/F/D audits.
Old density/features/scores stay exact. All 72 peak terms pass the Rust
surface on ten SIMD configurations; 14,268 footprint queries pass brute force.
Actual 25,248 pixel replacements improve mean spatial rank at every block
size, but 15/96 cells still miss M3f >=0.70 and 16/96 miss M2 >=0.99.
The saved-data oracle diagnosis is decisive: observed max deltas fix 0/15
spatial failures; observed non-max deltas fix 15/15. Next isolate basic SSIM,
residual/HF and L8 approximation errors through the existing coherence owner,
then correct the responsible terms. Do not retune max ownership or repeat
unchanged encodes. Native steering, corruption and full qualification remain
incomplete; no default model changes. Reports and source-bound binaries are
in the September 8 `max-attribution` validation delivery.

Latest continuation [implements candidate L8 attribution](benchmarks/l8_attribution_2026-09-08.md).
All 36 terms pass canonical reconstruction across ten SIMD configurations.
All 792 pixel/cache/spatial audits preserve exact features/scores; D maps stay
exact and F/blend maps gain L8. Max coverage remains incomplete (36 IDs on
241 nonidentity rows, 35 on eleven, per F/blend arm). Tests/Clippy/API/lint
pass; initial fixture, dispatch-isolation and replay-count corrections are
retained. A max-removal reference matches brute force on 7,365 rectangles:
row/column prefix/suffix maxima give exact signal-rectangle effects in O(1)
with O(width+height) storage. Next implement this non-additive rectangle
estimate in the existing Rust owner, validate source/coarse/reflected/padded
geometry and actual pixel interventions, then qualify native JXL spatial RD.
Do not treat L8 coverage as complete maps or product qualification.

Latest continuation [fits and Rust-serves a cheaper complete model](benchmarks/cheap_model_blend_2026-09-08.md):
equal F_nonneg32 seeds 4004/4005/4006 at 0.11333333333333333 each, plus D at 0.66.
All 1,647 FIT and 824 CAL training consensus pairs resolve; identities/upper bound
and 792 pixel/cache/spatial audits pass. Cost screen passes at 23.77 ms (1024²) and
84.38 ms (2048²), essentially D cost. The first 4 MP run has only 29 clean paired
samples and stays rejected; its preregistered 60-round repeat has 55 and passes.
No p95/HDR/cache/map/RSS or model qualification. All 252 nonidentity maps still
omit 72 peak terms. Next implement/check L8 and max contributions in the
existing Rust attribution owner, with ties/finite-removal semantics explicit,
then qualify native spatial RD. Keep historical recipe limitations, corruption
honest-protection failures and the full release contract intact.

The preceding continuation [reduces full-pool activity blur cost](benchmarks/padded_activity_rows_2026-09-08.md).
Separate existing producer profiles locate about 63% of full-pool incremental
sampled cycles in horizontal activity box blur. A second padded-row kernel
preserves all 1,320 pixel and 792 spatial audits exactly and passes 482 tests
(six ignored), Clippy, formatting and script lint. Complete blend observed
means improve 42.10→35.33 ms at 1024² and 141.81→133.88 ms at 2048²; both
registered gains and the 576/1152 non-trigger controls pass. The blend still
costs 1.49/1.53 times D; no quiet/p95/HDR/memory qualification. Next assess
existing cheaper training recipes and data coverage before another kernel
repair or D-regime candidate. Missing pooled attribution, corruption honest
protection and useful native spatial targeting remain required.

The preceding continuation [reduces horizontal SSIM row cost](benchmarks/padded_ssim_rows_2026-09-08.md)
without changing any feature or score bit. Native profiling locates 69% of the
mixed benchmark's sampled cycles in that kernel. Padded sixteen-row storage
reduces observed blend means 88.40→42.20 ms at 1024² and 194.94→141.44 ms at
2048²; D becomes 25.34/89.77 ms. All 1,320 pixel audits and 792 spatial audits
are byte-identical, including the blend's existing unsupported map terms.
Golden/fold/SIMD/allocation/serving tests pass. Non-trigger widths stay within
the registered 5% normalized regression tolerance; maximum observed increase
is 4.50%. Formal quiet/p95/HDR/memory qualification remains incomplete, and the
blend still exceeds 1.25 times current D. Next profile remaining full-pool
cost before choosing further compute repair or a cheaper competitive model;
missing pooled attribution, corruption protection and codec RD remain required.
Post-change native profile: fused horizontal SSIM 31%, fused vertical SSIM 15%,
single-plane horizontal blur 12% of the mixed workload. Inspect those owners
next, separating common work from A's pool increment. The initial noisy 576²
control was replaced by a passing preregistered batched control; no threshold
was relaxed and no latency percentile was inferred from batching.

The preceding continuation [measures actual A/D inference cost](benchmarks/model_blend_speed_2026-09-08.md)
and fixes zenbench waiting on its own Linux lock-heartbeat thread. Root bench
now pins pushed zenbench `1bf8a6509fce`. Corrected 40-round, single-worker means:
blend 88.05/194.45 ms at 1024²/2048²; D 76.41/136.04 ms; SSIM2 67.73/289.15 ms.
Blend and A alone cost essentially the same. Observed extrema miss the 1 MP
absolute/SSIM2 and 4 MP relative-to-D bars. Strict quiet admission still fails
(one advisory/background activity), so these are diagnostic results, not a
performance qualification. Next profile shared extraction and the full-pool
increment before choosing a compute repair or cheaper competitive regime.
Do not invest in all missing A spatial integrands before resolving cost.
No model/inference arithmetic changed; full product goal remains incomplete.

The preceding continuation fits and Rust-serves the registered A/D blend: exact
weights `[0.5900000000000001, 0.41]`. It resolves all 1,647 fit and 824 separate
training-calibration consensus pairs, preserves twelve exact identities, and
matches independent arithmetic exactly. The actual attribution check then
finds 169 active unsupported feature IDs across f156..371; all 252 nonidentity
blend maps are incomplete. D-only maps have complete coverage. Do not advance
this candidate into native steering with incomplete maps. Next price the
actual blend through `ssim2_speed_bar` against D/SSIM2 before committing to
full-pool attribution work: the shared extraction must serve A's expensive
pools too. If cost passes, implement/check those integrands in Rust; otherwise
repair cost or train a competitive cheaper model. Preserve scalar arithmetic, explicit coverage,
max/clamp/gate caveats and the existing performance/release bars.
[Blend recipe, results and spatial gap](benchmarks/model_blend_2026-09-08.md).
The existing pixel-audit owner now supports explicit ensembles and weights;
legacy single scores and both endpoint controls reproduce. No new public API.

The preceding continuation compares nine existing 372-class base models on 264
canonical training pairs, with full Rust pixel/cache scoring and independent
judges. D has zero disagreements among 2,471 resolved same-source pairs;
generation A and B each have three. The newer A_plain seeds have 170–202,
and H_anchorlad seeds 48–69, including substantial near-lossless errors.
All models pass twelve exact identities and never exceed 100. None advances
under the registered strict-improvement rule; a zero-error D baseline makes
that rule unable to resolve equal-perfect candidates. This separates strong
bulk scalar ordering from the already failed D spatial policy.
[Results and next registered A/D blend](benchmarks/model_preferences_2026-09-08.md).
That registered fit and Rust audit now completed as recorded above. Preserve
all release gates and the head's two honest false positives; the base blend
does not repair those automatically.

The preceding continuation repairs the native JXL delivered-pixel contract. Controlled
dither on/off decoding explains nearly all of the earlier drift: 8,646,810
samples change by at most one code; only 178 samples still differ between the
undithered canonical and historical decoders. The native loop now uses the
canonical zenjxl-decoder U8 path. All 756 newly encoded training ladder outputs
match independent decoded hashes and f32-reported D scores exactly; the 336-row
historical diagnostic reproduces exactly. Old decoder-era calibration is
refused before encoding. This closes the measured decoder discrepancy, not
model accuracy or spatial qualification. Continue with honest protection and
base-model preferences; do not rerun the already failed fixed D allocation
policy as though decoder parity qualified it.
[Decoder repair and evidence](../jxl-encoder/benchmarks/zensim_decode_contract_2026-09-08.md).

The preceding continuation adds 608 retained honest JXL/AVIF map-arm attempts on the
eight admitted fit origins (402 additional unique pixel pairs), then fits the
unchanged v2 cost-4 recipe once. Calibration false positives fall from three
to two, still FAIL. No validation/terminal scoring. More importantly, the
native JXL target loop's decoded pixels differ from canonical extraction on
all 336 added JXL attempts: median absolute D drift 0.185, p95 1.807, maximum
2.425 points. AVIF pixels agree exactly. Source inspection identifies a likely
cause: native JXL uses upstream jxl f32 output plus plain rounding; canonical
zenjxl-decoder U8 defaults to blue-noise dithering. The later controlled test
and repair above supersede that hypothesis. Preserve the old results as their
named decoder era. The head still fails honest calibration protection.
[Honest coverage and decoder evidence](benchmarks/corruption_honest_map_2026-09-08.md).

Latest September 8 continuation establishes the missing numerical release
contract at the top of [MODEL_SELECTION_SCORECARD](docs/MODEL_SELECTION_SCORECARD.md).
It repairs the head's precision boundary with explicit ZCTH v2 f32 inputs;
v1 behavior and all fitted numeric sections remain unchanged. The cost-4
training-only replay now has exact raw/probability/composed-score equality
on all 9,036 pixel comparisons and exact sklearn/Rust parity on 8,213 unique
rows. Accuracy still fails: three honest calibration JXL outputs are lowered.
No validation was evaluated and no model qualifies. See
[precision record](benchmarks/corruption_input_precision_2026-09-08.md).
Next corruption work is broader admitted honest coverage, especially legitimate
near-lossless JXL across independent source families; repeating cost fits or
retuning validation thresholds cannot close that gap.

Latest user direction: **keep the corruption head in D's existing feature
regime**. A new f0..227 HGB head is fully Rust-servable without expanding image
extraction. All three seeds detect every tested non-inert RGB swap and 99.57%
of validation corruptions, but lower 20/304 honest native codec outputs. It
fails the fixed development bars; no model qualifies or replaces a default.
Three bounded honest-cost follow-ups also fail on training calibration or
serving precision and never advance to validation. [Exact results, artifacts
and remaining precision issue](benchmarks/canonical_corruption_d228_2026-09-08.md).
The earlier full content-admission audit is complete; its all-372 head and
serving refusal remain preserved. [Admission record](benchmarks/canonical_corruption_refit_2026-09-08.md).

Remaining work, in order:

1. Improve the D-regime corruption head's honest-output protection with broader
   admitted honest codec/chroma/tone examples and source coverage. Resolve the
   tree's training/stored-f32 versus pixel-feature precision boundary (now
   resolved for newly exported v2 heads; preserve v1 artifacts). Preserve
   fixed safety bars and full Rust evaluation. The completed cost sweep chose
   no arm; do not tune its failed arms on validation or expand D's extraction.
2. Repair base-model identity, near-lossless and codec-floor preferences using
   the strongest existing candidates and native-codec evidence. A corruption
   gate or monotone calibration cannot repair the base model's rank inversions.
3. Prove JXL spatial benefit against a strong ordinary scalar controller at
   matched quality/bytes, with the complete frozen model and independent judges.
   Separate model-preference failures from allocation-rule failures; retain the
   already failed fixed-D policy rather than retuning it on observed outcomes.
4. Finish train-family calibrated, separate-validation 1/2/3-shot scalar/spatial
   targeting for JXL, then AVIF, JPEG and WebP. Measure attained ranges before
   judging steering, keep bounds out of runtime, and count every encode/map.
5. Freeze and qualify the full Rust artifact: perceptual/dial/tail behavior,
   supported SDR/HDR/color/alpha, latency/memory, spatial RD and terminal tests;
   deliver the reproducible recipe, examples, manifests and gauntlet report.

The immediate HDR task was a protected-reference overlap check, not a new HDR
training pipeline. HDR development PNGs already exist. That admission dependency
is now closed; the next work belongs to model serving, accuracy and JXL steering.

Latest user direction: EXR belongs in **`zenextras/zenexr`, wrapping the Rust
`exr` crate**. This explicitly authorizes that dependency and supersedes the
earlier pending exception question. The custom zenbitmaps port is retired from
the active checkout; its verified source checkpoint and results remain archived.
The replacement reproduces all 98 saved fixtures and all 30 UPIQ HDR references
bit-for-bit (124,609,944 f32 samples). [Wrapper contract and validation](../zenextras/benchmarks/zenexr_validation_2026-09-08.md).
The later fingerprinting/contextual review is complete as recorded above;
decoder parity alone does not qualify a model or authorize holdout training.

The preceding user correction still applies: HDR development uses the existing imazen-26
`variant/png-v3` inputs: **76 HDR PNGs**, all locally present and byte-verified
against branch LFS payload OIDs, with **38 train / 20 validate / 18 terminal**
origins under the family manifest. The active 1,140-image HDR scale set is also
local. [Source binding and chronology](docs/TARGET_STEERING_PROTOCOL_2026-09-08.md#existing-hdr-png-inputs--user-correction-september-8).
The separate UPIQ EXRs are holdout-overlap references, not a missing HDR
training corpus. The earlier port checkpoint is preserved at
`/mnt/v/output/zensim/native-exr-port-2026-09-08/`; active EXR implementation
now uses zenexr. One new corruption head has been fit; no model is qualified.

Previous September 8 continuation: [actual coarse JXL allocation policy](benchmarks/zensim_coarse_allocation_2026-09-08.md)
**fails its preregistered independent-judge screen**. All eight 256 cells improve
D versus every locally reachable globally rescaled raw field, but photo 2010/d3
loses 0.021502 Butteraugli and screen 8206/d1 loses 0.143189 SSIM2. The separately
registered 512 software check reveals further losses. This fixed D+policy does
not advance; no validation or terminal families were spent, and no model is
qualified. The comparator exhausts a declared local integer-field domain, not
ordinary global-scale/distance control or the full codec optimum. Policy inputs
exclude measured bounds/judges. All 249 primary encodes and 498 judge values
reproduce exactly; 36 analyzer / 11 CLI refusals and local CI-closure checks pass.
Artifacts: `/mnt/v/output/zensim/jxl-coarse-allocation-2026-09-08/`, mirrored to
`~/work/zensim-validation-2026-09-08/jxl-coarse-allocation/` with an HTML report
and original/active/failed-judge scalar comparator gallery. Next separate model
preferences from allocation-rule effects using supported existing candidate maps
under a registered fixed rule, and strengthen the ordinary scalar comparator
before any product RD claim. Do not simply repeat this failed D policy or sweep
its gain against these outcomes. The EXR reader question was subsequently
resolved by the user direction above; fingerprinting/contextual admission
remain unfinished.

Previous September 8 continuation: [coarse JXL intervention screen](benchmarks/zensim_coarse_interventions_2026-09-08.md)
finishes native PNG IO and whole-transform 4×4 grouping in the existing codec
instrument. Coarse / ±20% gives expected D direction in 244/256 probes at 256
and 245/256 at 512; independent judges mostly agree. Map mass predicts response
much better, but density versus gain per byte remains weak/negative in some
cells. Area and amplitude both changed, so their separate effects are not
identified. **No matched-RD win or model qualification.** All 272 historical
transform outputs are byte/pixel/score/map-exact under native IO, and 816 final
probes plus 1,632 judge values reproduce. Thirty-one analyzer and eleven CLI
refusals pass; final/local CI-closure checks pass. All evidence is retained at
`/mnt/v/output/zensim/jxl-coarse-interventions-2026-09-08/`.
Next spatial work: preregister an actual coarse allocation policy in the same
owner and test matched quality/bytes before separate-family 1/2/3-shot targeting.
Do not repeat the completed intervention screens or treat oracle-probe cost as
runtime cost. The earlier corruption-refit EXR question is resolved above.

Previous September 8 continuation: [canonical refit preparation](benchmarks/canonical_corruption_refit_2026-09-08.md)
adds native content admission and a trainer mode that exports and evaluates the
same single fit. All 12 training origins have zero strict flags against 182 SDR
holdout reference entries; four looser matches were reviewed as distinct content.
The source/pixel-deduplicated fit / calibration / evaluation views contain
5,504 / 2,709 / 5,679 rows and pass explicit full-key Parquet checks. An invented
800-row numeric fixture has exact weighted-HGB export parity in Rust. **No
canonical image-data fit yet:** 30 UPIQ HDR EXR references remain unaudited. The
reader question at that stage was pending. The user later explicitly selected
`zenextras/zenexr` over the Rust `exr` crate. Use that owner to finish the
remaining reference admission; do not repeat the completed SDR audit.

Previous September 8 continuation: [canonical corruption serving screen](benchmarks/canonical_corruption_serving_2026-09-08.md)
is complete on all 15,060 canonical/native rows, with exact pixel, cached and
stored-f32 feature/score parity. New `BakeScorer::score_features_with_identity`
carries proven pixel identity; raw zero features remain insufficient evidence.
Frozen D+HGB detects 93.82% of unique validation corruptions and orders 98.64%
below q20, but incorrectly sends eight near-lossless JXL outputs to zero and
detects only 64.35% of the newer real-bug family. **No new fit or qualification.**
Before fitting, finish T0 content admission, construct source/pixel-deduplicated
views (raw catalogs still fail C10), and register exact family-level fit /
calibration / evaluation identities. Extend the existing corruption trainer:
its legacy CV-ensemble report describes a different model than its exported
single fit. Evaluate the exact exported artifact through Rust. Corruption-gate
discontinuity and native adapters' rejection of companions remain product work.
Prior native intervention and RD findings remain negative/mixed; corruption
detection or restoration coherence does not prove useful spatial allocation.

The end user controls one target score. The codec chooses parameters and must
reach useful quality across codecs/content, near-lossless settings and codec
floors with few passes, small output, low latency and low memory. Negative
scores are valid. A rank correlation or research composite cannot establish
that behavior. B remains `codec_target()`; D is the explicit fast profile.
B/C/D may be replaced: there are no consumers calibrated to their current
scores. [The integration guide](docs/CODEC_TARGET_METRIC.md) owns the mapping.

Every new model must execute and serve entirely in Rust through a zensim API,
and evaluation must call that API, including heads, corruption and splines.
`BakeScorer` now provides that dynamic surface. Python remains useful for
invention and independent numerical references.

| Question | Current answer / evidence |
|---|---|
| Can we train after cleanup? | Three A_plain seeds exactly reproduce all measured verdict and ladder fields. Three H_anchorlad seeds also reproduce every earlier verdict and ladder field; three paired rav1e additions complete the floor-data control. [Reproduction](benchmarks/cleanup_training_reproduction_2026-09-07.md); [paired controls](benchmarks/cleanup_scientific_controls_2026-09-07.md). These historical recipes have documented split limits. |
| How does target search behave? | [2,730 witnessed-target cells](benchmarks/target_steering_bounds_2026-09-08.md) compare train-calibrated 1/2/3-shot policies, with exact final-code reproduction. The [360 older loops](benchmarks/cleanup_target_loop_2026-09-07.md) mixed feasible/unwitnessed targets; their median cannot establish model failure. [Native JXL targeting](https://github.com/imazen/jxl-encoder/blob/main/benchmarks/zensim_native_targeting_2026-09-08.md) now measures three arms with shared Rust calibration/search: 810 cases, 45/80 jointly witnessed targets, all arms hit ±1 in three shots on that subset. Coverage and independent RD still prohibit qualification. |
| Does native attribution already exist? | Yes: JXL fused/binned/stale H3, a recovered JPEG `jj` experiment, WebP segments and the later AVIF CQ/H3 experiment. [Fleet/source/chronology audit](benchmarks/diffmap_reuse_audit_2026-09-08.md): reuse them. The binding now exists: `BakeScorer::compute_with_ref_and_attribution` uses complete scoring and declared extraction, with explicit spatial coverage. JXL now consumes it in [commit 9f038d4d](https://github.com/imazen/jxl-encoder/commit/9f038d4dd7f4); 840 ladders and decoded map engagement pass, but D/H3 has no broad RD win. [Native record](https://github.com/imazen/jxl-encoder/blob/main/benchmarks/zensim_candidate_binding_2026-09-08.md). [Serving/coherence evidence](benchmarks/candidate_attribution_serving_2026-09-08.md): exact pixel parity, 15 serving tests and 54 coherence cells. D is the first native baseline; native engagement and product qualification remain incomplete. |
| Is good rank enough? | No. A_plain retains identity and all-codec floor failures. Qualification reports failure/incomplete evidence separately from research selection. No new common-dial winner is established. |
| Can candidates serve their whole model? | Yes, through `BakeScorer`: declared IDs, validated metadata, heads, pin, spline, codec affine, ensemble and corruption composition. [Execution plan](docs/PLAN_CRUFT_PURGE_2026-09-06.md). Pixel luminance revision must still match the process. |
| What happened to C/CHdr? | Both use explicit IDs and canonical activity semantics. The old training/serving mismatch is fixed; two alternate planners are retired. Pixel scores intentionally change. [984-case feature census and HDR/matrix evidence](benchmarks/feature_plan_cleanup_2026-09-07.md). |
| Where do I train and evaluate? | `zensim_mlp_train` → `bake_dial_refit` → `run_full_eval.sh` → `freeze_check`. Admission now checks actual headers, IDs and declarations before work. [Trainer cleanup](benchmarks/trainer_admission_cleanup_2026-09-07.md). |
| What owns calibration? | Rust `bake_dial_refit`, with pre-calibration coordinates and final full-surface evaluation. Four historical writers are retired with recipe/boundary evidence. [Spline record](benchmarks/spline_owner_cleanup_2026-09-07.md). |
| Which data are valid? | [DATA_SPLITS](docs/DATA_SPLITS.md), later [DATASET_HISTORY](docs/DATASET_HISTORY.md) entries, actual manifests and the [shared index](../DATA_PROVENANCE.md). Width is not an era. TID is train-only; historical KonJND/KADID overlaps are guards, not holdouts. |
| Where is the board? | `scripts/v_next/gauntlet.py` renders stored owner verdicts to `/mnt/v/output/zensim/reports/summer_gauntlet{_fair}.html`. Codec-q score charts include negative scores; product qualification appears before composite. |
| Why keep buffered/full features? | Remaining callers and numerical differences prohibit blanket retirement. Free/shared features are not removed on coefficient counts. The contended extraction benchmark was excluded, not presented as a speedup. |

The authorized cleanup is complete: [execution and validation record](benchmarks/cleanup_completion_2026-09-07.md); [original checklist](docs/PLAN_CRUFT_PURGE_2026-09-06.md).
The [transcript/memory audit](benchmarks/science_workflow_audit_2026-09-07.md)
records chronology and bounded confidence in Rust/Python equivalence. Scientific
controls and actual-loop measurements have their own result records; a passing
code migration cannot inherit an unmeasured product gate.

Use the existing capped `run-heavy` owner for heavy work and
`scripts/safe_push.sh` for every push. Data/bake evidence is preserved under
`/mnt/v/output/zensim/cleanup-*-2026-09-07/`; private audit and retired sources
are under `~/tmp/zensim-science-audit-2026-09-07/`. Choose fresh output paths
for reproductions. Do not overwrite historical results or infer completion
from a filename without its content identity.

September 8 repository sync + AVIF binding continuation: all 75 immediate zen
repos plus five external dependency/corpus repos fetched; 52 checkouts advanced
or rebased, local changes rescued and five conflict sets resolved. Exact private
recovery/status: `~/work/zensim-recovery-2026-09-08/sync-all/README.md`. Keep
syncing before new work and use tight local checks; do not wait on CI.

AVIF complete-candidate research binding pushed as `4777a30aa54aaab6534719f6026e4a18e91d60e3`. Exact-bake
`BakeScorer` owns scalar and current spatial scores; first map feeds encode 2,
unsupported terms fail, terminal bytes are decoded/scored independently. On the
one training-family control, legacy gain 10 is inert (all blocks saturate then
normalize to 1); existing zerosum engages and repeats byte/pixel/map-exactly.
Neutral 79.368/18,355 B; active 80.287/19,573 B; three full encodes, three maps,
four scalar comparisons including terminal. Six CLI rejection controls pass.
See sibling `benchmarks/zensim_avif_loop_2026-08-07.md` September 8 addendum and
`zensim_avif_candidate_binding_2026-09-08.json`. Artifacts at
`/mnt/v/output/zensim/avif-candidate-binding-2026-09-08/`; Windows-readable copy
under `~/work/zensim-validation-2026-09-08/avif-candidate-binding/`.
This is engagement only, not RD or competitive-model qualification. Next: AVIF
attained bounds and train-only calibrated actual 1/2/3-encode loops through the
shared Rust search, then matched-quality judging. No new model training or
terminal holdout was performed in this continuation.

Shared native probe owner added after the AVIF binding: unpublished tool feature
`zensim-target/native-probe`, with codec-owned adapter for AVIF. See
`benchmarks/native_probe_instrument_2026-09-08.md`: 51-bound/119-encode train and
validation pilots, 54 target cases, source/model/driver/family rejection checks
and independent decoded-pixel integrity checks pass. AVIF's stateful map uses
the previous complete decode; its first shot has no consumed map. Full 12/8
family matrix is next, then model qualification and native JPEG/WebP work.

Later September 8: the full AVIF 12-train/8-validation matrix and independent
judges are COMPLETE. [Instrument record](benchmarks/native_probe_instrument_2026-09-08.md)
links the codec report: 504 cases, 28/80 joint witnesses; three-shot calibrated
scalar/neutral 28/28 hits ±1 versus active 24/28. Fixed-CQ active maps modestly
help SSIMULACRA2 but worsen Butteraugli in every content class; no spatial
benefit or model qualification. Artifact `RESULT_COMPLETE.json` supersedes
the launch-time pending text in `SOURCE_PINNED.json`. Judge-pair identity,
duplicate/missing controls pass, with JXL/AVIF numerical results preserved.
The September 7 H+rav1e floor-data control already completed and failed all five
floor gates; do not repeat the older September 6 proposal as new work. Continue
with useful native allocation/model qualification and JPEG/WebP integration.

Later September 8 JPEG continuation: complete-candidate binding now runs in the
existing Zq loop (`__zensim-research`, recovered `zq_rd_probe`). The first trace
exposed old AQ defects: unit scales clamped real strengths to 0.20 and the last
strip skipped the controller. Both are repaired, along with a fractional-q
clamp panic near 100. [JPEG record](https://github.com/imazen/zenjpeg/blob/main/benchmarks/zensim_candidate_binding_2026-09-08.md):
one train family, exact D, 444/q80; scalar and neutral 77.438789 / 16,864 B,
active repeat 77.314751 / 16,795 B. All 832 blocks controlled; current maps,
JPEGs and pixels repeat exactly. SSIMULACRA2 and Butteraugli both show the small
quality loss, so no RD benefit is claimed. Eighteen Zq tests pass before/after,
22 AQ tests, two Clippy routes, API snapshot check and ten rejection controls
pass. Artifacts: `/mnt/v/output/zensim/jpeg-candidate-binding-2026-09-08/final/`;
Windows bundle under `~/work/zensim-validation-2026-09-08/jpeg-candidate-binding/`.
This is fixed-seed engagement, not calibrated targeting: two corrections cost
three full encodes, and its existing `targets_met` flag checks only the score
floor. Reuse the shared Rust native owner for actual JPEG 1/2/3-shot bounds and
train calibration; WebP binding and competitive model/spatial qualification
remain incomplete. No training or terminal holdout was performed in this step.


Later September 8 WebP continuation: [complete-candidate binding and accounting
repairs](https://github.com/imazen/zenwebp/blob/main/benchmarks/zensim_candidate_binding_2026-09-08.md)
are pushed in `8aa8a7858b97`. The existing segment loop now accepts the exact Rust
bake through `__zensim-research`; registry 0.2 and the separate recompress A
calibration remain intact. One-pass targeting is measured and strict, encode
counts are actual, ship-band flags are truthful, and finite negatives are valid.
All five old failures were observed in regression tests before repair; hard-floor
selection has its own observed failing control. Local 354 library, 15 targeting,
33 validation tests, two Clippy routes, release/API/format checks pass.

One canonical training family, D/q80/m4: scalar and neutral select 73.673485 /
16,040 B; active selects 72.955315 / 15,728 B. The fixed-q neutral correction is
byte/pixel/map exact; active quantizer 11→14 is consumed and repeats exactly.
All 208 macroblocks, including the partial edge, have explicit map/segment traces.
Both independent judges see quality loss with the smaller active file. Thirteen
full encodes, nine maps, two non-neutral map uses across active+repeat and five
terminal comparisons are counted; ten CLI rejection controls pass. Full result:
`/mnt/v/output/zensim/webp-candidate-binding-2026-09-08/`; Windows copy under
`~/work/zensim-validation-2026-09-08/webp-candidate-binding/`.

All four codec owners now have an exact complete-candidate binding and native
engagement evidence. Useful spatial RD, a competitive qualified model, JPEG/WebP
train-calibrated witnessed bounds and actual 1/2/3-shot validation remain required.
Do not repeat binding screens as model qualification. No training or terminal
holdout was performed in this continuation.

Later September 8 native JXL finite-block continuation:
[intervention record](https://github.com/imazen/jxl-encoder/blob/main/benchmarks/zensim_native_interventions_2026-09-08.md).
Codec instrument/report pushed as `feab1d7fd734c89212b449a1945cf1e6e7bf81e4`.
Four train families, distances 1/3, 16 native transform regions with ±10%
quantizer changes: 272 complete probes reproduce bytes/pixels/scores/maps and
both independent judges exactly. Separate 272-probe 512-long-edge multi-group
coverage passes; 48 total libjxl 0.12 compatibility checks complete. The existing
RD analyzer's new `--interventions` mode passes 19 rejection controls, preserves
older analysis functions, and verifies all source/byte/pixel/quantizer/judge
identities. Six CLI controls and local release/Clippy/format/script checks pass.

Restoration coherence does not establish useful native marginal allocation:
primary map-mass/D-response rank associations range from −0.259 to 0.676.
222/256 interventions affect pixels outside their selected region even though
captured quantizer changes stay local. Seven interventions preserve the final
quantizer field but change pixels; native thresholds also depend on the incoming
field. CfL is frozen here. Do not treat final quantizer identity as a no-op or
these response secants as isolated quantizer gradients. No model/policy change
or qualification. Artifacts: `/mnt/v/output/zensim/jxl-native-interventions-2026-09-08/`;
Windows copy under `~/work/zensim-validation-2026-09-08/jxl-native-interventions/`.
Next spatial hypothesis: preregister a coarse-region/rate-cost intervention
before tuning allocation. Model qualification and JPEG/WebP actual targeting
remain open; no new training or terminal holdout in this packet.

### October 27, 07:21 UTC: isolated parameter-packing result

The registered three-seed TRAIN audit completed. All three stripped f32 controls
are byte-identical to their original dense models; every one of the 12,163
predictions per seed is exactly identical. All artifact hashes and paired TSV
coverage were independently reread after completion. Before output calibration:

| Seed | Mean absolute delta | RMSE | p99 absolute (nearest rank) | Maximum absolute |
|---|---:|---:|---:|---:|
| 0 | 0.000924370962 | 0.001041982565 | 0.002151489258 | 0.003734588623 |
| 1 | 0.000397940701 | 0.000531791949 | 0.001613616943 | 0.003211975098 |
| 2 | 0.000917169125 | 0.001031623618 | 0.002243041992 | 0.003612518311 |

These are native f32-inference score-unit changes caused by the original f16
parameter pack with the spline removed, on the frozen TRAIN calibration
population. They are not native HDR precision, feature extraction error,
post-calibration bounds or speed qualification. No candidate is adopted.
Evidence: shared `zensim/mlic-precision-20261027-run01/results/COMPLETE.json`,
with sibling `VERIFIED.json` at the run root.
Completion SHA256: `47e3ab97651a81b5a9caa4825eeee291b459f6c1d150954510f211f6ae09dd7b`.

Run-heavy on i265: rc=0, 7s, peak RSS 0.47 GiB, minimum available 23,044 MiB, peak load 6.13. The duration is an execution receipt, not a runtime benchmark.

### October 27: native input baseline at Rev7

`just mlic-native-input-tests` preserves the five legacy native-HDR tests and
adds a separate Rev7 child-process check of the supported 720-feature HDR
entry. Running the legacy append-family tests wholesale with
`ZENSIM_FORMULA_REV=7` fails before color arithmetic because that feature family
is deliberately unsupported. The rejection and old test expectations remain
unchanged; this failed invocation is retained in `native-hdr-rev7.log`.

The new check uses a 97×65 image with native u16 codes and a one-code red-channel
distortion. For PQ and HLG, each in sRGB, Display-P3 and BT.2020, the native-u16
features match the equivalent normalized-f32 code features exactly, remain
finite, and differ from identity. All six cases pass. This proves the tested
input-container and supported-entry behavior; it does not prove every possible
code step survives display clamps, establish a new f64 error bound, qualify
Mlic's full model/input composition, or show HDR perceptual quality or speed.
The original five tests also pass, retaining their independent f64 color/display
reference and prepared-map checks under their original arithmetic.

Evidence: shared output `zensim/mlic-precision-20261027-run01/`, including
`native-hdr-build-initial.log`, `native-hdr-rev7.log` and
`native-hdr-supported-rev7-run02.log`. The first build/default-five-test run:
run-heavy rc=0, 147s, peak RSS 1.92 GiB, minimum available 21,921 MiB, peak load
3.80. The final six-test run including explicit Rev7 child: rc=0, 9s, peak RSS
0.59 GiB, minimum available 23,443 MiB, peak load 1.41. These are validation
resource receipts, not performance qualification. No frontend arithmetic,
feature identity, trained artifact or running experiment was changed.

### October 27: certificates for completed Neff/Vote full initial models

All six completed full-data initial models (Neff and Vote, seeds 0/1/2) pass
the existing native `bake_dial_refit certify-monotone --pin 100` owner. Inputs
were bound to their final119/120-epoch fit receipts and packed-model hashes
before checking. Every certificate reports zero free amplitude and free ramps.
Max-pair counts are Neff 2/1/2 and Vote 3/1/2. This is coordinate monotonicity
of those initial models, not pixel-mixture behavior, AIC continuation
qualification or a product-adoption decision. Order and continued models are
not covered by this receipt.

Artifacts: shared output `zensim/mlic-full-initial-certificates-20261027-run01/`
(`PLAN.json`, six frozen model copies/native logs, `COMPLETE.json`). Completion
SHA256 `3574cc7ede2010b83ae90393d291bc1b772b2613461f2a08cef5011dc64e82d6`.
The coordinator reread every copied model hash after completion. Run-heavy
on i265: rc=0, 1s, peak RSS 0.02 GiB, minimum available 23,964 MiB, peak load
1.25. This is a validation resource receipt, not a runtime benchmark.

### October 27: obsolete research entry points now fail before work

Owner request: block outdated paths that waste compute, beginning with silent
Rev6 restamping. `historical_guard.py` now denies new default invocations of
`serve_parity.py`, `candidate_registry.py`, `perf_grid.sh`, `perf_bits.sh`,
`map_study.sh`, `jxl_steer_rd.sh`, `jxl_reach.sh` and `dataset_loop.sh`. The guard
runs before argument expansion requiring data paths, scientific dependencies,
output creation or subprocess work. Exit 64 explains the obsolete assumption
and points at the current session/priority owner. The dated registry includes
a deleted checkout; the old parity owner always restamps to Rev6; the shell
studies carry fixed historical model, placement, judge or output defaults.

Intentional reproduction requires `--historical-replay` as the first argument
to the script. A trailing flag or environment setting does not authorize it.
That opt-in does not waive science, input freezing or resource rules. Historical
benchmark/map/RD launchers still refuse dev, and all guarded paths refuse
r7900x. Historical regression tests and valid old-model serving remain enabled.
Frozen as-run source snapshots, current AIC/confidence queues, artifacts and
existing numerical gates are unchanged. The parent Mdata loop propagates its
explicit replay authorization to its historical JXL child.

`just byc-historical-guard-tests` checks all eight entry points without model
files, optional Python dependencies or positional arguments; each must fail
with the guard's diagnostic instead of loading data or launching work. It also
checks explicit first-argument handling, host restrictions and unknown-entry
refusal. This is a targeted guard inventory, not a claim that every historical
script in the repository has been audited. New research launchers should use
the current frozen-plan owners instead of copying these dated defaults.

### October 27: all full-data initial fits complete

Run03 now has all twelve initial models: three original Mlic reuses and nine
new Neff/Vote/Order fits. Every seed inventory, final119/120-epoch receipt and
packed model hash was reread from `INITIAL_FULL.json`. The coordinator reached
`common-full-continuation`; six new continuation trainers were verified live.
Full panels and the separate confidence queue still follow. This is not
candidate adoption. Shared progress evidence: `mlic-priority-20261027/FULL_INITIAL_VERIFIED.json`.

### October 27: full AIC fits and remaining coordinate certificates complete

Full run03 finished all 12 initial models and 24 continuations. The coordinator
reread both model hashes and every recorded input hash for all 36 records in
`FULL_FITS_COMPLETE.json`. Full-data panels subsequently started and reached
350/360 population predictions at the recorded check; this is not panel or
mixture completion. No failures were recorded. The independent confidence
omission queue still waits for full-panel completion.

The existing native `certify-monotone --pin 100` owner also passed for all
27 remaining packed models: three Order initial models and all 24 continued
models. Every certificate reports monotone/certified true and zero free
amplitude/ramps. Earlier receipts cover the nine Mlic/Neff/Vote initial models.
This completes coordinate-certificate coverage of the 36 models, not pixel
monotonicity, mixture behavior, runtime qualification or candidate adoption.

Evidence: shared output `zensim/mlic-full-certificates-20261027-run01/`, with
frozen PLAN, model copies, native logs and COMPLETE. Completion SHA256:
`d54070c71ee3628121803b771e7011132b51fb119b4891fbe9ab1a66d96bbc05`.
All copied model and log hashes were reread after completion. Certificate
run-heavy receipt on i265: rc=0, 3s, peak RSS 0.03 GiB, minimum available
23,793 MiB, peak load 0.16. The dev full-fit receipt is rc=0, 4,822s,
peak RSS 2.55 GiB, minimum available 26,792 MiB, peak load 9.44.
These are execution resource receipts, not runtime benchmarks.

### October 27: full-data cell statistics expose an ordering tradeoff

The 360 native cell panels are complete; source/codec/interval and mixture
work remains in progress. These are partial study results, not a selection.
For the registered serving seed0, production-f16-calibrated 16×8 arms have
AIC MAE / p99 distance: Mlic 0.428860 / 0.383393; Neff 0.434599 / 0.391138;
Vote 0.429270 / 0.374389; Order 0.421418 / 0.363907. Order's AIC p99 distances
are lower than Mlic's for all three seeds. AIC is in sample here.

Order also has worse seed0 MAE than Mlic on every other TRAIN panel:
KADID 9.039121 vs 8.763874; TID2013 6.322748 vs 6.132106;
KonFiG 13.380590 vs 12.363111; CID22-A25 8.075303 vs 8.020150.
This prevents interpreting its AIC improvement as a universal improvement.
Full native statistics, geometry and per-seed records are retained in shared
`mlic-priority-20261027/FULL_CELL_EARLY_REVIEW.json`, bound to the current
`panels/CELL_PANELS.json` hash. No candidate is adopted or excluded by a new
SROCC cutoff; wait for the remaining registered panels and confidence arm.

The existing `featcanon_rev7_parity` tests also passed on i265 at source
`32ea661e768a`: two tests, zero failures/ignores. They check the full Rev7
feature vector across available dispatch permutations at five geometries,
including remainder columns and width4133, plus unchanged revision-rounding
bounds. This does not establish candidate pixel/cache score parity, another
architecture's behavior, or performance. Receipt:
`mlic-precision-20261027-run01/REV7_FEATURE_PARITY_VERIFIED.json` and its log.
Run-heavy rc=0, 47s, peak RSS 2.16 GiB, minimum available 21,557 MiB,
peak load 1.02; execution resource receipt, not a runtime benchmark.

### October 27: full-data source/codec and seed review

All 69,552 aggregate native panels and registered source intervals are now
written; pixel-mixture checks remain live. The descriptive review compares
16×8 packed/calibrated arms with Mlic16×8 using the mean predictions of three
fixed seeds. This stability diagnostic is distinct from serving seed0 and
from averaging three seed-level statistics. All AIC sources are in sample.

| Arm | AIC sources with lower MAE | AIC codecs with lower MAE | Aggregate AIC MAE delta |
|---|---:|---:|---:|
| Neff16×8 | 10/28 | 3/14 | +0.007089 |
| Vote16×8 | 16/28 | 3/14 | +0.002324 |
| Order16×8 | 13/28 | 6/14 | −0.002510 |

Every count has complete defined coverage. Order's aggregate MAE deltas on
KADID/TID2013/KonFiG/CID22-A25 are +0.204870/+0.115273/+0.435511/−0.053202.
Thus its seed0 regression on all four other populations is not identical to
the three-seed aggregate: CID22-A25 improves in the latter. Vote's corresponding
deltas are −0.043590/+0.016107/−0.082979/−0.051431. These mixed changes do not
establish a stable replacement, and no arm is rejected by a new SROCC gate.

Paired individual-seed AIC MAE deltas for Order are
−0.007442/+0.002467/−0.002893; Vote +0.000410/+0.016214/−0.007568;
Neff +0.005739/+0.016832/−0.000588. The registered 2,000 whole-source bootstrap
intervals are descriptive, conditional on these fitted models, and in-sample;
they do not measure independent test uncertainty. Preserve them alongside the
full panels rather than treating interval crossing as a selection threshold.

Evidence: shared `zensim/mlic-full-panel-review-20261027-run01/`, including
five source-artifact copies and hashes, full per-source/codec deltas in
`REVIEW.json` (SHA256 `823461f1d650307d8df519e5bde82227f8ca0d6a6909a89fde804802a7022b63`), and the exact review command/log. The i265
run-heavy receipt is rc=0, 2s, peak RSS 0.17 GiB, minimum available
23,819 MiB, peak load 0.15. No new training or calibration was performed.
Next remains the registered mixture completion and independent confidence
comparison, followed by outstanding production qualification; no adoption.

### October 27: AIC-2026 public_train full panels complete; confidence control live

Full run03 completed all 360 population predictions, 69,552 aggregate panels,
paired-seed contrasts, source intervals and both pixel-mixture sets. Each
mixture set covers 36 models × two representations = 72 cells. Every global
mixture cell has zero inversions. Local mixtures still have reversals: for
packed/calibrated seed0 at 16×8, inversion counts are Mlic 3, Neff 3, Vote 3,
Order 4 (1,152 comparisons per cell). Their maximum score rises are
5.644995, 5.004366, 6.430682 and 3.900804 respectively. Coordinate-monotone
certificates do not eliminate these pixel-mixture failures. No adoption.

The coordinator reread all 1,476 inventory-file hashes, source hashes and native
tool hashes, plus completion/plan/fold dependencies. Shared verification:
`mlic-priority-20261027/FULL_PANEL_VERIFIED.json`; full completion SHA256
`fd787030c17f916c06dc8207b30dd353873e983befdea305754cbbb6ce0f805a`.
Panel run-heavy: rc=0, 745s, peak RSS 0.65 GiB, minimum available 43,435 MiB,
peak load 5.37. This is an evaluation resource receipt, not runtime qualification.

The confidence run02 dependency completed and its full-budget unweighted
control started on dev; at 08:19 UTC its live worker was 3388770/start 231371214,
lock holder 3388749. Re-resolve with `just mlic-priority-progress`. The control
must match 7,869 original predictions before any of the 21 confidence fits.

Separately, the existing near-identity owner is running on i265 for all 12
seed0 initial/continued models, using 24 frozen TRAIN images, unchanged
N1/N2 ≥99 and N3 ≥122/144 contracts, revision 7/v3. Shared root:
`mlic-nearid-20261027-run03`; PLAN SHA256
`b43474f3c414c92a12efc279eda8b781e73bbc5cb14967c08b94ae611fc3e0e7`.
Its STATE records real PID/start ticks and per-model results. First two models
Mlic/Neff pass all three gates; remaining models were still running at entry.
The frozen source packet is unchanged apart from explicit image-path remapping;
all model and image hashes were verified. No quality labels are read.

Run01 refused uppercase CLI labels before measurements. Run02 failed parsing
its command before execution. Both are preserved, with no scientific results.
Run03 uses lowercase labels and the identical native binary/model/image bytes.
The reusable `just dataset-nearid-candidate` now validates lowercase labels
before invoking native work, then uses the same native and reporting owners.
Invalid-label refusal and command expansion were checked. Native build receipt:
rc=0, 15s, peak RSS 0.84 GiB, minimum available 22,950 MiB, peak load 0.41.

### October 27: all twelve near-identity checks complete

All twelve registered seed0 initial/continued models pass the unchanged N1,
N2 and N3 near-identity gates on the 24-reference TRAIN packet. Each model
has 648 measurement rows. N1 requires both one-pixel rungs ≥99 on every
reference; N2 requires a nonidentical score ≥99 on every reference; N3 requires
at least 122 of 144 nonincreasing ladders. Observed N3 counts are:

| Recipe | Initial | 8×4 | 16×8 |
|---|---:|---:|---:|
| Mlic | 143 | 143 | 143 |
| Neff | 140 | 139 | 139 |
| Vote | 143 | 143 | 143 |
| Order | 138 | 138 | 138 |

The coordinator checked all twelve metadata/model/packet/revision bindings,
648-row counts and gate results, then hashed every result/log file. Shared
receipt: `mlic-nearid-20261027-run03/VERIFIED.json`; completion SHA256
`c70c483ff5e30ac2e1a2b0eb59e94e8d915906ff4ce5b2b6fa7ab4ee687b6b1a`.
Run-heavy on i265: rc=0, 136s, peak RSS 0.25 GiB, minimum available
23,730 MiB, peak load 0.96. This is validation resource evidence, not timing
qualification. N1–N3 passing does not clear the local-mixture findings,
other production gates, runtime qualification or stable adoption.

### October 27: explicit Rev7 synthetic serving parity

The existing `serve_custom_bake` probe now has a separate `--prodqual-rev7`
entry, exposed by `just dataset-serving-parity-rev7`. The original `--prodqual`
still requires revision 5. Both entries require their process and bake revisions
to match exactly; no model is restamped. Three actual refusal controls cover
Rev7 process through the old entry, Rev5 process through the new entry, and
Rev5 model through the new entry. All refuse before synthetic measurements.
The arithmetic, geometries, identity bands and exact-bit comparisons are unchanged.

All twelve frozen initial/continued seed0 models completed the new probe on
i265: four geometries × seven bit-truncation levels × eight available dispatch
permutations per model, 2,688 model/pair/permutation rows. Direct pixel scores,
prepared-cache scores and identity-aware feature scores agree exactly; consumed
and cached features agree exactly; cross-permutation row mismatches are zero.
All scores are finite, pixel identity is exactly 100, feature identity remains
in the unchanged 97.5–100 band, and no tested distortion exceeds 100. The original
synthetic ladder observations remain report-only. The raw artifact's `seed`
field is the argument index, not the fitting seed: all inputs here are seed0.
No WASM, ARM, AVX-512, real-corpus or speed claim follows from this i265 run.

Shared evidence: `zensim/mlic-serving-parity-20261027-run01/` with preregistered
PLAN, frozen source/binary binding, NATIVE, SUMMARY, refusal logs and VERIFIED.
The coordinator checked source/model/binary hashes and every summary comparison.
Build run-heavy: rc=0, 14s, peak RSS 0.84 GiB, minimum available 23,132 MiB, peak load 0.48.
Measurement run-heavy: rc=0, 17s, peak RSS 0.24 GiB, minimum available 23,693 MiB,
peak load 0.96. Resource receipts are not runtime benchmarks. Rust formatting
checks pass. Confidence control training continues; no candidate is adopted.

### October 27: matched original-Mlic baseline, not a cross-cohort comparison

The original full Mlic models give AIC-2026 public_train pooled SROCC
0.907609 when the three packed/calibrated seed predictions are averaged on
all 3,920 rows. The same-population full 16×8 continuations give Mlic 0.914556,
Neff 0.911525, Vote 0.913928 and Order 0.915738. Thus the Mlic continuation
improves over the original Mlic baseline; none is yet a stable production
replacement. The older approximately 0.93 claim concerned different Tga/M64a
recipes, not original Mlic, and must not be used as Mlic's starting number.

Full-model numbers and fold aggregates are different estimands: fold models
also omit an E21 leg, have different starting fits, see 24 rather than 28
AIC-2026 TRAIN sources, and average different numbers of predictions per row.
Vote's full in-sample SROCC 0.913928 is below its fold-held-out aggregate 0.914591;
that observation is real, but it is not the train/test comparison of a single
fixed model. The precise causal reason has not been isolated. Likewise,
within-ladder SROCC near 0.9998 is not pooled SROCC near 0.908.

The completed continuation schedule is only four/eight epochs at constant
learning rate 1e-5 with mixed losses, not direct SROCC optimization or a
convergence demonstration. Do not claim a feature-capacity limit, convergence,
or an optimizer defect from these numbers. A matched TRAIN-only convergence
and objective diagnostic is the next question alongside the already registered
confidence study. Keep frozen outcomes and all independent evaluation roles.

### AIC-only comparison recovered — October 27, 08:46 UTC

The old T model's saved AIC-2026-only in-sample SROCC is 0.9742999206247418;
its seven-source-fold SROCC is 0.9526610561260485. The fit log confirms one
training group, 3,920 rows, 28 sources and 60×20,000 draws. Its feature
provenance is explicitly unqualified, and its fx1g model is not Mlic. Do not
conflate this result with AIC-3 or current qualified Rev7 measurements.
The byconstruction record now preregisters a matched seed-0 mixed-versus-
AIC-only continuation diagnostic at fixed epochs 7/59, starting from the same
Mlic checkpoint. AIC-only continuation still inherits mixed-data pretraining.
The diagnostic has not launched: its separate native/Python admission must
first be implemented without relaxing the frozen 0/4/8-epoch authority.
The historical evidence and input hashes are linked from the session results
page. Confidence control was verified live at epoch 76/120 at 08:41 UTC;
resolve current processes and artifacts before reusing that observation.

### Owner-directed return to original Claude session — October 27, 2026

The owner requested transfer of the entire production-candidate objective,
all live jobs, new directives and subsequent seamless native-precision HDR
qualification back to the original Claude session. This is a transfer of
supervision, not completion or adoption. The current report, completed study
receipts and latest matched-continuation registration remain authoritative.
Confidence control and its queued follow-through must continue under the
receiving session; neither should be restarted from the old compacted recap.
At 08:47:45 UTC the control was live at logged epoch98/120, and the existing
follow-through worker was alive waiting for omissions. Re-resolve both.

All implementation changes are committed. The matched population/budget
continuation diagnostic is registered but not implemented or launched.
Main lacks the frozen warm-start integration: reuse its verified historical
owner selectively, preserving newer vote/confidence/ordering code and all
admission guards. This source integration is the next preparation step;
it does not replace supervision of the active confidence sequence.
A dated non-release checkpoint tag records the outgoing state. The receiving
session must acknowledge actual live job identities before Codex relinquishes
supervision. No model is production-qualified by this transfer.
