# Mlic+sub prepared root — 2026-10-26

Built on dev without fitting or EVAL/TEST target access. The original six-arm study selected Mlic and Mw025 before any subset fit; Mlic+sub stays outside its 123-cell inventory. [Weights were registered](byconstruction_2026-10-22.md#mlicsub-root-preparation--preread-registration-2026-10-26) in 34a323396d81 before source target reads: the five nonempty collections share nominal 0.5, at 0.1 each. DIBR splits 0.1 equally across three protocol legs; no confidence or vote-count loss weighting.

Checked on dev: `/var/tmp/byc-subsets-20261026/Mlic+sub/root`; canonical `/mnt/v/output/zensim/byconstruction-2026-10-22/licence-subsets` points to its study directory. No mirror is claimed yet. All source banks/memberships retain their [verified mirrors](dataset_subsets_rev7_2026-10-25.pointer.md).

| Collection | TRAIN memberships | Derived rows | TRAIN families |
|---|---:|---:|---:|
| Cognitive Science Experiment 2 |109|109|5|
| Broken Arrows |60|60|5|
| DIBR |288|456|2|
| DWT/DT-CWT |60|60|6|
| Enrico |60|60|3|
| Toyama |0|0|0|

The 577 memberships yield 745 rows across seven protocol legs, with 18 distinct TRAIN families across collections. DIBR has 64 ACR rows, 56 scaled-score rows and 168 pair comparisons expanded to 336 hard-direction rows. Its eight native viewpoints per scalar protocol remain separate references; source families join all four viewpoints of each sequence. The three protocols share observations and are not independent votes. Native `p_a` is blank: checked row-preferred counts derive `(n_a+0.5*n_tie)/n`, then only its relation to 0.5 enters hard labels. No comparisons cross protocols or native references.

Cognitive Experiment 2 replays its hash-bound exact stimulus-to-parent-reference projection and checks the parent Colourlab membership. It supplies no independent source content. All six policy rows have A/B tier and `train` licence permission. Toyama's 112 prospective TRAIN rows remain EXCLUDED for unresolved source matches; no Toyama target was opened. Cognitive Experiment 1 and demosaicing lack pristine references, HDR-JPEG/PairComp TMO require a native HDR/viewing route, and Fourier is tier C; none enters this arm. VCL@FER, PieAPP and FG-IQA stay excluded by the Mlic licence policy.

Mlic's eight inherited dataset legs and their nominal 0.5 weights are unchanged. Frozen payload/manifest/key hashes match for all eight, and inherited real-leg receipt records match. Added nominal total0.5; total dataset rows 248,921 across 15 legs. The guard reserves 40,938 families, including prior/additive excluded held-out memberships, with zero inherited leaks. Exact native pixels and confirmed D4 aliases bound the guard; the source screen does not prove perfect crop recall.

Successful assembly has seven source TRAIN ledger entries, persisted before their reads. Its root SHA256 is `89b5f88d054200ba35d217094cfc79c3a1ab0d7e3de89dfe9d92cdb04d6ece29`; parent Mlic root `a7fd4af0c2bb5a8981be636e5c1cb1a51db57496b7c60d7bf6da01d71dd16952`. Control `MLIC_PLUS_SUB_ROOT.json`: `d058fbfc4acc31c6fb54addc4c95785398adb45d2d1259c473eb09533bbb8251`. Source builder 4a9aa18ba35f. Wrapper: `run-heavy: done rc=0 8s | peak-RSS 1.53GiB | min-avail 43475MiB | peak-load 15.15` (16 GiB scope/jobs 12).

The first attempt refused blank DIBR probabilities. Its whole partial root, original ledger, and failure mapping are retained at `Mlic+sub.failed-native-p-a-r1/`. A subsequent diagnostic repeat of one already-authorized TRAIN row was disclosed retrospectively in `DIAGNOSTIC_READ_DISCLOSURE.json`; it was not individually prelogged. The successful build's seven reads were prelogged. Library MANIFEST metadata also exposed published aggregate examples, recorded in SUBSET_POLICY; they did not choose weights or selection. No model or subset fit exists. Cells await the coordinator's separate finalist-shape rule.

## Linked-family fold authority

Before any subset cell, the coordinator required whole-collection omission when an added subset shares a source family with a held-out collection (79dcd47a and 9a3a8083). The label-free map uses registered family IDs, exact reference pixels and confirmed D4 aliases. Only `mos7` has an added exclusion: `colourlab-cognitive-science-2022`, linked through the ten registered parent families 1187718–1187727 (five fit families). The other six folds (`kadid`, `tid2013`, `konfig`, `cid22_a25`, `pipal`, `pairs`) have no added exclusions. Their existing exclusions remain in force.

The original root above is preserved. The derived root checked on dev is `/var/tmp/byc-subsets-20261026/Mlic+sub-linked/root`, frozen SHA256 `6abfac53afd6db332e39e9e556eba6e97b73717c2f37828385eac98eb8c825d6`. Its bound `datasets/linked-subset-omissions.json` has SHA256 `e7772dbc3b85c6f630a65145534f55d0ae845be6003a74ad84a5b20fa777ad03`. Tables and weights are unchanged; parent paths still refer to immutable source roots, so fleet transport must pass its relocation-closure audit. The coordinator recorded this root and map in abafc0e6 before any cell.

Driver 1076a19e38a0 enforces the omissions in both E21 and new-data folds and refuses missing, changed or unapplied authorities before target reads. Nineteen focused checks cover those paths, including the actual nested E21 key receipt. The live metadata-only admission check passed all seven folds: mos7 retains eight dataset groups after removing Cognitive, pipal and pairs retain fourteen each, and each E21 fold retains fifteen. No target payload was opened and no optimizer ran. Receipt `/var/tmp/byc-subsets-20261026/linked-admission-control/VERIFIED.json`, SHA256 `bcaf941d3f7a3dbdc485d60342292c037065b54d1f6eda598771b3f91452780a`; reproducible recipe `just dataset-subsets-admission` landed in 8501c622e668.

Both metadata jobs joined the existing 42 GiB/no-swap scope. Linked-root construction exited 0 in 4.70 s with `/usr/bin/time -v` maximum RSS 634,680 KiB; live admission exited 0 in 6.36 s with maximum RSS 382,812 KiB. These are process RSS observations, separate from the concurrently training scope's memory. The fleet owns all 21 subset cells, gated on fresh control parity and closure; this lane launched no subset fit.

## Worker relocation control — 2026-10-26

The fleet’s network-disabled in-image relocation audit passed210 archive files,82 runtime paths and all7 folds. Receipt `fleet-coordination/MLIC_SUB_RELOCATION_VERIFIED.json` in the mix study, SHA256 `e7c4df4460cd0e6b064fa248822fdd6c2a517b0c51adbd218ff66eeed3903952`, binds unchanged source1076a19e, root6abfac53 and authoritye7772dbc. The R6 transport archive checked and independently SHA256-verified on dev is `/mnt/v/output/zensim/fleet-cells-20261025/r6/data-Mlic-sub.tar.gz`,2,249,990,216 bytes, SHA256 `f70cb25352dc84152c105334bf6143ca8966508511323c1ae4b6979d606d8fde`. This is an archive digest, separate from the frozen-root digest. No uploaded object or new mirror is claimed by this observation. Fresh R6 exact parity remains the fleet gate before subset science.
