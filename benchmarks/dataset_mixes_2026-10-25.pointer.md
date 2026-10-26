# Dataset training mixes — 2026-10-25

## Complete — 2026-10-26

All 123 cells, six-arm TRAIN diagnostics, both all-cell mixture tests and the
frozen finalists' EVAL assessments are complete. The registered rule selects
Mlic for balanced performance and Mw025 for the older datasets. Only Mlic
meets the new-leg licence policy. Its balanced lead is 0.0000016562213902;
EVAL differences are mixed. Serving profiles are unchanged.

The closed snapshot checked on dev is
`/var/tmp/byc-mixes-20261025/archive-ready/`. Its verified mounted NAS mirror is
`/mnt/tower/output/zensim-datamixes-archive-2026-10-25/`.
All 9,682 artifact files (13,961,632,597 bytes) passed SHA-256 readback, followed
by three random readbacks. The final inventory has 9,683 entries including the
receipt. It materializes study symlinks and preserves cell/full-data bakes,
partial attempts, the coordinator's Mlic full fits, both subset preparation
roots and the failed subset-root build. No R2 mirror is claimed.

| Closed artifact | SHA-256 |
|---|---|
| `NUMERIC_SUMMARY.json` | `842fdde7cbe67bf2a864667368f6dc22131d682e3a53ff6abdbabad9ab5bf1ea` |
| `MIRROR_VERIFIED.json` | `041c5ff14bc7612038a64ed3d35ce6849a4258b12e2ab6cbe5e47072ddd4d675` |
| `SHA256SUMS` | `f93e5e7c07f375d3ce4bbebcf7f9be6a0be95a2a1dbf45aa5e7b563ba1dc34ca` |

Snapshot source is c2727d2cb135a06f8bbffceceddad9d557748674. Hashing and
fsync/POSIX_FADV_DONTNEED readback ran on dev through the mounted export;
physical NAS cache temperature is unmeasured. No remote process was started.
The command exited 0 in 344.11 s with `/usr/bin/time -v` maximum RSS 119,448 KiB.
One 4 GiB/no-swap child ran inside the unchanged 42 GiB fleet scope; six science
slots were preserved and both child and parent recorded zero OOM.

The closed admission receipt has SHA-256
`7254fc625190e216b3766ebd0a8b19d5e16132fadec447c813591178963a457a`.
It and the complete execution log are verified beside the NAS archive as
`.postprocess.json` and `.mirror.log`. Log SHA-256:
`717cf9444814208956d94a9c06bdc8b76eb2bd563beca019090d7405742e0cae`.

## Execution history

The observations below preserve the assembly, fitting and assessment sequence.

Checked on dev: `/mnt/v/output/zensim/byconstruction-2026-10-22/data-mixes`
points to `/var/tmp/byc-mixes-20261025/`. Assembly and implementation checks
are complete; the training queue is in progress. Finalist selection and
EVAL assessment have not run. No serving profile changes.

Method and preregistered selection:
[experiment record](byconstruction_2026-10-22.md#new-dataset-training-mixes--preregistered-2026-10-25).
Source implementation: 348407450b0d; assembly record: 899e685ba01e.
Actual native build cut: 0572d08b8d0e; the subsequent dashboard-only rebase
preserves all captured training-source hashes (`SOURCE_EQUIVALENCE.json`).
The build's source patch and file pins are retained alongside the binaries.

| Local artifact | SHA-256 |
|---|---|
| `ROOTS.json` | `90eef110ab7f093507ef302db4ed58981a71bf8f839cc3505391dd3b014b3cfe` |
| `BUILD_SOURCE_FILES_SHA256SUMS` | `3d32d33f66ec1e17126c3fdbb02a601dac45a75e97b0ca40a4b41ceed023749b` |
| `BUILD_SOURCE_CHANGES.patch` | `dbb3b8badf80d1cec0a77d05bbc9cb64af0409df47cb254f78e2aad126a354cd` |
| `BIN_SHA256SUMS` | `ef2d70d6e62b4c4bbeb1a553ffabf1e17ceefa1d62d99ac50bcdc21859965202` |
| `SOURCE_EQUIVALENCE.json` | `f604b6ae0d5cb44ca887c94a060fd2b3609933c2ced0037b844fbfecc64c981b` |

The study freezes columns C and the fx1 declaration in `frozen_configuration/`.
Their bytes match every root's declared feature IDs; fits bind this local copy
through `BYC_COLS` (9ffb8c188a73). Hashes:

- `frozen_configuration/cols_c.txt`: `e662c7225c7f6aca27bf208161b0d1b99f6d2ba7be427dfa37c4ac4ec1c49c75`.
- `frozen_configuration/e33_fx1_declaration.json`: `a97a5fa63937ed8fade3c56fd7470765a031d1d309afc36e631f371a1796e648`.
- `ROOT_ROW_EQUIVALENCE.json`: `a6f0d35951932324ac69710f032985ccd50ff87facfc5a3f2cf49209ec580dda`.

Both default-rebuild controls have frozen-file SHA-256
`743d20148bbd265c829490621ed45a74bcde0842fdc47cc2b3f1f67b2e0d0115`;
all root files compare byte-identically. The original Mdata root remains
`e127eef91efce57c6bb980c3c6e047e0a49079d57f337fc122be74e70cbef1ad`.
The root inventory records every variant's full hash, collection count,
nominal mass, derived-row count and label-read ledger size. Each new root's
ledger precedes its source TRAIN reads; each new cell has a separate derived
TRAIN-read ledger. Diagnostic owners log E21 TRAIN-panel reads separately.

No completed archive of this new study exists yet, and there is no R2 mirror.
The original Mdata artifacts retain their separately verified
[archive record](mdata_2026-10-25.pointer.md). Study completion requires all
123 requested cell records (12 reused), diagnostics, both mixture sets,
frozen finalists, one assessment per finalist and a verified private archive.

The owner licence addendum adds Mlic (eight new legs at nominal 0.5).
VCL@FER, PieAPP and FG-IQA are omitted from Mlic fitting and remain
validation populations. Only Mlic is eligible under this new-leg policy;
the other five arms are research comparisons. Its completed assembly binds
`LICENCE_POLICY.json` to the copied training-confidence dataset manifest.
The five-arm ROOTS and row-equivalence inventories are preserved as
`.before-Mlic.bak.json` before the sixth root is added.

Mlic root: `a7fd4af0c2bb5a8981be636e5c1cb1a51db57496b7c60d7bf6da01d71dd16952`; 248,176 rows, eight source-label ledger entries, 40,904 reservations and zero inherited leaks.

`MLIC_ROOT.json`: `de8b7b0171c61e86d146c1fa12b954d4f9ce84d354ca95d8eaaac1d73726d8e3`.

`LICENCE_POLICY.json`: `a87165b72ef0c1991a33d677aa3ad48245aa4102b9c6bb746e4e183d32746f46`.

The initial `SOURCE_EQUIVALENCE.json` describes the 348407450b0d landing. Later Python changes add diagnostics, progress and Mlic policy handling; the native trainer and its recorded binary hash are unchanged.

`FLEET_OWNERSHIP.json`: `f68d7e1a9f960eb69dd6a1add9857e223540a6ddcb6951345538ef475d8f2ef3`. Fleet exclusively owns 60 Mw100/Mmos/Mpair cells; the full submitted Mlic batch stays local. Source scripts are pinned to b7cb1a224dc7.

`BINARY_CELL_CONTROLS.partial.json` verifies the recorded trainer hashes for
54 completed cells against the frozen binaries. The complete scorecard
requires `BINARY_CELL_CONTROLS.json` covering all 123 records; the partial
receipt cannot satisfy it. Changed-trainer and missing-cell negative
controls are recorded in `BINARY_CELL_CONTROLS_NEGATIVE.json`.

Mlic has completed all 21 cells at fixed epoch 119. Together with Mdata and
Mw025, 63 of 123 cells are complete; the remaining 60 are fleet-owned.
`MLIC_COMPLETION.json`: `bf7478ccc2d3351312888a21f85e9485c1090abf1b605b78490862fbd4e51c72`.

`LOCAL_TRAIN_GROUP_RESULTS.json` retains the held-out-new-TRAIN statistics
for the three completed local arms, before six-arm selection.

`local-e21/e21.json`: `3eaede9e77b4779ee0755bfc06686b0c4d533b0655e669c513a1298a39369f6f`;
its `label_reads/` records the original TRAIN-panel target read. The frozen
Mdata E21 values are unchanged. `LOCAL_TRAIN_GROUP_RESULTS.json`:
`f416ceeb9851933e29f41bb7c883e2ab398b7747c453bf90e509637f03fe13ad`.
`fleet-coordination/R2_PREPARATION_READY.json`:
`de9cef2b8db3554bd9d705c30dbeadcfacbddb411eaf26f0553dfec3c979830c`;
this covers image/control inputs, without claiming completed arm archives
or numeric control parity.

The initial mounted-NAS mirror control copied and verified three result artifacts,
with all hashing/readback on dev and no Docker invocation. Local evidence:
`mounted-mirror-control/`, `mounted-mirror-control.exec.log` and
`mounted-mirror-control-negative/`. The control's receipt SHA-256 is
`fe4e1094fb6608cd607992a00c7b3bafd15a0207d47a559aaa7b42ff4342ff6a`;
inventory SHA-256 `dc52fe5fbea419065923c09adf228ffc3737eef402d7650d0b17441555d85cb7`.
NAS control location:
`/mnt/tower/output/zensim-datamixes-mounted-mirror-control-2026-10-25/`.
Its initial source patch is retained in `MOUNTED_MIRROR_INITIAL_CONTROL_SOURCE.patch`.
The final archive will use the consolidated shared mounted transport with
optional cold readback. Its actual-NAS control started no remote processes:
`mounted-mirror-canonical-control/`, with receipt SHA-256
`4fdd7ba8eb8ad9dc11212841612b00979f1bbb42961114d261c72ac8d43b126b`
and inventory `04c2624b198bcaa0a656ab4bb3403164a6029732fc410ee8e5009592aedaa42e`.
NAS location:
`/mnt/tower/output/zensim-datamixes-mounted-mirror-canonical-control-2026-10-25/`.

Local diagnostics and both mixture sets are complete for the 63 local
models. The E21 mixture summaries cover twelve models per arm; all-cell
raw reports preserve every local model. This does not complete the six-arm
scorecard or select finalists. Receipts checked on dev:

- `LOCAL_DIAGNOSTIC_RESULTS.json`: `917c153bae782f3c2c2493c5253812dbe0bad0212e886d2d24fd5ed1b0790f10`.
- `LOCAL_MIXTURE_CELL_CONTROLS.json`: `12e5f3457fd06a2180ade5289808bdfd77e85739fcd5737f1e609cdccbb709ba`.
- `local-mixtures/mixture.json`: `a70e17686b3f6c8358294fdf9681047d93f4e41aaef5a604fcb869f716c74e6b`.
- `local-mixtures/mixture_global.json`: `a604f890bc53fef218d24fc598db15299cbb1d679ad224c0a366e7abc37a8b3b`.

The AMD replay passed its exact control; the study independently repeated
the prediction-byte and native checkpoint-strip comparisons. No scientific
arm harvest is claimed by these receipts:

- `fleet-coordination/AMD_CONTROL_PARITY.json`: `2b3409d09d768773ef8067da7c9d92cabe06ea39113492272f6177c3f1091c67`.
- `fleet-coordination/AMD_CONTROL_HARVEST.json`: `0e5cc3d85570ebd8991c2eeefc221970f4806c1b50f02515c859037e68f89f97`.
- `fleet-coordination/amd-parity-independent/VERIFIED.json`: `28eec6c3893cd252caa490257cf3e408663f32bfdabc520fa10242dd00fa692b`.

## Completed TRAIN selection — 2026-10-26

All 123 cells pass native inventory and binary controls. Both mixture sets cover every model; full diagnostics are complete. The registered rule selects Mlic (balanced) and Mw025 (old data). Mlic’s balanced lead is only 0.0000016562213902. Only Mlic is licence-eligible. Its seed 0 reuses the certified coordinator bake; Mw025’s required full fit is active. EVAL and final archive remain pending.

- `TRAIN_SELECTION.json`: `7bbda05e870a3cc5efa54f360c09791978d0c96ee9c5c1c00df2a95c6ead3ed1`.
- `NATIVE_CELL_CONTROLS.json`: `1337c9e49e030e4f3b35db4d0f3c37a89ccdfaf48cb030de497dccf3a5a979ee`.
- `BINARY_CELL_CONTROLS.json`: `7fa385c55d7a72b006247a9dacf946cec1ea9b6fd1eaa0fb9642aa75a3f237c7`.
- `FORCED_CHOICE_CONTROLS.json`: `12e47ee46a31b35aabc34a573325c33f2c9b6388c25cb87e8a30717df6fa5597`.
- `MIXTURE_CELL_CONTROLS.json`: `7f8449ced72ea26c1fd656315a8133536d04e6d369217718b8586e80a32be6d3`.
- `MLIC_COORDINATOR_FULL_FITS.json`: `12cf83eabda7c0fee16e620c98df2c0296f1a9e0b1fa73ded6b47c705f7371ad`.
- `PRESERVED_CANDIDATE_BINDINGS.json`: `198c85780e01864fd2ac38e74b1ff4aa4497f3152af03665b030ec2207a420f1`.
- `fleet-coordination/EXACT60_VERIFIED_HARVEST.json`: `3a8c90727d94d7ff7d15cf5cc9f30819a742ff5c4bf6ec3efacbf10b6d3e7d9d`.
- `fleet-coordination/ALL_PRESERVED_CANDIDATES_VERIFIED.json`: `dae3ed3702a1400e7ec64677f633f55749c2cda638513bf6b664335d90f1df1c`.

Final original fleet coverage is 60 DONE, zero missing/FAILED/STRANDED/outside-grid/duplicate DONE. Historical failures remain: five lost-lease partials, nine intentional rebalance partials and three failed-output/scratch retries. Every candidate bake is retained. Seventeen preserved candidate directories are bound for materialized archival; scratch remains protected under its fleet inventory. No scientific file or source receipt was rewritten.

Immutable inputs remain in LAN-store bucket `zentrain`, keys `jobs/byc-<ARM>-r2-20261025/inputs/<sha>`; `fleet-coordination/STAGED_INPUTS.json` records exact bytes. Archive digests differ from the frozen-root manifest digests. The workspace DATA_PROVENANCE index records all six object pointers.

## Frozen finalists and EVAL point report — 2026-10-26

Mw025 full seed0 has completed the registered120×50000/fixed119 fit. The finalist manifest `FINALISTS.json` SHA256 `8191f8ced4488629c9c8d37b7439df203b0f1a9b58804bd6e9e312c8692c24c8` binds Mlic `7612c2e46154fe9c0c359bfd02e521013f5717e3bf8e11967f5434150e220bcc` and Mw025 `a6d43ef2ec717cb8f9b07ecb85f7f4c76aae7995cbce9b42e4d39ef7123ea4ac`. Both certify monotone. No EVAL result selected either model. The11-collection point report is written from stored features; paired reference-bootstrap computation and BAPPS assessment are still in progress. Mlic seed0 predictions are frozen for coordinator reuse, with no second scoring.

Mw025-s0 is registered in the durable candidate registry, now19 entries. All19 bakes passed SHA256SUMS on the mounted tower mirror with dev doing checksum work. `CANDIDATE_REGISTRY_VERIFIED.json` SHA256 `20255999ae094ae034fbebed23a5ae59c70c23e42b0820a0c2cddc26ca36a42f`; prior local and mirror metadata and the ownership-error/retry logs are retained. Registry source a9a7d34997cc; mounted-copy recipe0153adc179e6. No tower compute or AIC2026 label read.

The later [linked subset root and seven-fold authority](mlic_subsets_2026-10-26.pointer.md#linked-family-fold-authority) are outside the original123 cells. Their build/control evidence is linked into this study for materialized archival. Coordinator-owned Mlic64 jobs remain separate in `SCOPE_ACCOUNTING.json`. No subset fit was launched by this lane.

## EVAL complete — 2026-10-26

Both finalists have one frozen assessment on each feature bank;54 paired MOS comparisons per model are measured. BAPPS36,344 comparisons: Mlic0.6732583104, Mw0250.6685037429, Mdata0.67392417, M7 0.6617956205, B7 0.6392169278. No image unpack or peer recompute; peers await IMI. `EVAL_COMPLETE.json` SHA256 `c5db7b5929d172732110176f7f75f66b3a3db9862f9aed5a7faeab237ba1947e`. Full point/interval tables and limitations are in the experiment record. Numeric summary and final mirror follow.
