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
| `ROOTS.json` | `1e4a98348f9cf7f573f8790165039409a832859c65ab1057e223435fadb96438` |
| `BUILD_SOURCE_FILES_SHA256SUMS` | `3d32d33f66ec1e17126c3fdbb02a601dac45a75e97b0ca40a4b41ceed023749b` |
| `BUILD_SOURCE_CHANGES.patch` | `dbb3b8badf80d1cec0a77d05bbc9cb64af0409df47cb254f78e2aad126a354cd` |
| `BIN_SHA256SUMS` | `ef2d70d6e62b4c4bbeb1a553ffabf1e17ceefa1d62d99ac50bcdc21859965202` |
| `SOURCE_EQUIVALENCE.json` | `f604b6ae0d5cb44ca887c94a060fd2b3609933c2ced0037b844fbfecc64c981b` |

The study freezes columns C and the fx1 declaration in `frozen_configuration/`.
Their bytes match every root's declared feature IDs; fits bind this local copy
through `BYC_COLS` (9ffb8c188a73). Hashes:

- `frozen_configuration/cols_c.txt`: `e662c7225c7f6aca27bf208161b0d1b99f6d2ba7be427dfa37c4ac4ec1c49c75`.
- `frozen_configuration/e33_fx1_declaration.json`: `a97a5fa63937ed8fade3c56fd7470765a031d1d309afc36e631f371a1796e648`.
- `ROOT_ROW_EQUIVALENCE.json`: `61c8cfa792acbf866b8794155ec6f9305c47ccebe7f7a09e508ebc853e4b0097`.

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
the other five arms are research comparisons. Its queued assembly binds
`LICENCE_POLICY.json` to the copied training-confidence dataset manifest.
The five-arm ROOTS and row-equivalence inventories are preserved as
`.before-Mlic.bak.json` before the sixth root is added.
