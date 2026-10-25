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
102 requested cell records (12 reused), diagnostics, both mixture sets,
frozen finalists, one assessment per finalist and a verified private archive.
