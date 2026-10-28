# Splitting zensim into a research repository and a slim library (plan, 2026-10-28)

Owner request (2026-10-28): rename this repository to **zensimx**, then start a new **zensim** repository that
branches off before the history grew, keeping churn low against the last crates.io release (0.2.7). This is a plan:
nothing on GitHub has changed. The outward steps (rename, new repository, pushes to other repositories) wait for the
owner's go-ahead.

## Measured facts

Measured with `git_growth.py` (first-parent history of `main`, each blob counted once, at the commit that first
introduced it; output in `/mnt/v/output/zensim/repo-split-20261028/` on dev).

| Point | Date | Packed history | Note |
|---|---|---:|---|
| `zensim-v0.2.7` = `2894042d` | 2026-04-26 | 0.70 MiB | the last crates.io release; 419 commits |
| first commit past 2 MiB | 2026-05-10 | 2.0 MiB | bake binaries and sweep artifacts during the May training sweeps |
| end of May | 2026-05-31 | 29.4 MiB | |
| end of August | 2026-08-31 | 50.3 MiB | includes a 41 MiB release binary under `target-cg/release` (`714da506`, 2026-08-30) |
| today | 2026-10-28 | 92.2 MiB | GitHub reports 83,605 KB |

Uncompressed, the largest contributors over the whole history are `zensim/src` (295 MiB of blob versions),
`CHANGELOG.md` (181 MiB), `zensim-validate/src` (138 MiB) and the long experiment records under `benchmarks/`
(50–71 MiB each). `zensim/src` is 0.55 MB at 0.2.7 and 5.6 MB today.

**Consumers by git URL.** Fifteen primary checkouts depend on `https://github.com/imazen/zensim` through git, most
pinned to revisions that exist only after 0.2.7: aom-rs and zenav1-aom (`aom-target`), heic, zenavif, zenjpeg,
zenmetrics (`margarine`), zenpipe (with its demo, fuzz, metadata-services and wasm-size-shim crates), zensquoosh,
butteraugli's `margarine` experiment and imazen26-gainmap-scan. Many more use crates.io `zensim` 0.2.x or a path
dependency on `../zensim/zensim`.

**Public API today against 0.2.7** (`cargo public-api -p zensim -sss diff 0.2.7`, default features, main `0fa5afec`;
full list in `/mnt/v/output/zensim/repo-split-20261028/public-api-diff-0.2.7-to-main-0fa5afec.txt` on dev): one
removal (`ZensimResult::nan`), two breaking attribute changes (`#[non_exhaustive]` on `ProfileParams` and
`ZensimError`) and about 400 added items. The additions are mostly research surface: `feature_v2` (237 lines of the
listing), `feature_set_id` (100) and `source` (95), then `ZensimError` variants (50), `profile` (34), `Zensim` methods
(29), `BakeScorer` (12) and the JND types (18). A low-churn slim crate restores `nan()`, drops the two attribute changes
(or ships as 0.3.0), and keeps only what serving the new profiles needs.

## The redirect trap

Renaming `imazen/zensim` to `imazen/zensimx` makes GitHub redirect the old URL, so nothing breaks at first. Creating
a new `imazen/zensim` ends that redirect: every git dependency on `github.com/imazen/zensim` then resolves to the new
repository, where the pinned post-0.2.7 revisions do not exist, and those builds fail. The git consumers therefore move
to `imazen/zensimx` before the new repository is created.

## Proposed sequence

1. **Freeze.** Tag today's `main` in this repository (`research-before-split`).
2. **Repoint git consumers.** Change each git dependency from `github.com/imazen/zensim` to `github.com/imazen/zensimx`
   with the same revision, one commit per repository, and check each still resolves.
3. **Rename.** `imazen/zensim` → `imazen/zensimx` on GitHub; update local remotes. In zensimx, mark the `zensim`
   package `publish = false` (or rename the package; see the decisions below) so only the new repository publishes
   `zensim`.
4. **Create the slim repository.** A new public `imazen/zensim` holding history up to `zensim-v0.2.7` with tags
   v0.1.0–v0.2.7 (packed about 0.7 MiB).
5. **Port by pruning, in small commits** on that base. Re-implementing revision 7 inside 0.2.7's code is not
   realistic: the serving path (XYB, the pyramid, the basic and v2 families at revision 7, the extraction plan,
   `BakeScorer` over zenpredict) is interwoven with the research code across about 70 modules (`zensim/src` is 5.6 MB
   against 0.55 MB at 0.2.7). Instead, copy today's crate, delete revisions 1–6 except what legacy A needs, the research
   families, sidecars and training hooks, keep the two curated bakes as profiles plus legacy A (October 25 direction),
   and commit the result on top of the 0.2.7 history. Low churn means the public API: additive against 0.2.7 (new
   profile variants, no removals), checked with `cargo semver-checks` against the 0.2.7 release, with CHANGELOG and
   README.
6. **Local checkouts.** Move this checkout to `~/work/zen/zensimx` and clone the new repository at
   `~/work/zen/zensim`. Path consumers then build against the slim crate; check each, since some use API that only
   exists in 0.3.0.
7. **Release** after CI on every required platform and the owner's README review.

## Decisions for the owner

- **Branch point:** the 0.2.7 tag (recommended: it is both before the growth and the last release), or a later commit
  before 2026-05-10.
- **Crate name inside zensimx:** keep the package `zensim` with `publish = false`, or rename it `zensimx`.
- **Version:** 0.2.8 if strictly additive, or 0.3.0 because the default profile's scores change.
- **Scope of the slim crate:** revision 7 and legacy A only, or also today's B, C and D profiles.
