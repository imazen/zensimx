# Dataset training mixes — 2026-10-25

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
