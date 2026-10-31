# Reproducibility spine — how a zensim number stays true

## Current report ownership — September 7, 2026

`bake_verdict` owns scientific scores/statistics and its single-bake HTML report.
The existing `scripts/v_next/gauntlet.py` owns the multi-bake board; it renders
stored owner results and is the authorized dashboard. `run_full_eval.sh` owns
content-bound evaluation stages and reuse. `freeze_check --qualify` is separate
from research selection. This later workflow supersedes the July "only HTML
path" wording below while retaining its ban on duplicate report/statistic
implementations. See [WAVE_PLAYBOOK](WAVE_PLAYBOOK.md).

## Historical July report rule

**Ad-hoc HTML/dashboard report scripts are BANNED.** The canonical, comprehensive
report is `bake_verdict --html [--compare <ref-bake>]` — it emits the full
self-contained page: provenance, the rank summary, the CODEC-goals scorecard, the
dial panel, per-zone dial agreement, the corruption gate, and every per-corpus
section, with `--compare` folding a reference bake in. Do NOT write a small
Python "report" that renders a subset — it will be tiny, drift from the panel,
and duplicate the owner. Deleted this directive's day: `shipped_bakes_report.py`,
`bake_compare_dashboard.py`. If a multi-bake matrix is needed, run `bake_verdict
--html` per bake (each `--compare` the baseline) or extend the Rust owner — never
a new render script.


**Status 2026-07-15.** Written to answer one question: *if we published a paper
about zensim tomorrow, could a reader reproduce every number in it?*

This doc is the **traceability contract**, not a wishlist. Every claim below
either names the gate that enforces it or is listed in §4 as a known gap. A
rule with no enforcement is not a rule — that lesson is the whole reason this
file exists (see §5).

## Training releases and replays (October 31, 2026)

The spine below binds a number to a bake. A training release binds a bake to the bytes it was trained from, so a
reader can rebuild it. Revision 7's models (E, ECodec) and the 7.1 candidates (F3, F3codec) have one: every file
their trainers, extractors and packers read, content-addressed in a private bucket with a manifest, in the private
repository `imazen/zensim-training` (licence-restricted images and members-only AIC-2026 labels keep it private).
It is checked in rungs, each from the previous rung's released output:

| Rung | From → to | Checked by |
|---|---|---|
| R0 | distributed dataset files → released pixels | sha256 against the IQA library's lists; decoded pixels for format conversions |
| R1 | pixels → feature banks | re-running the frozen extractor (ed495741) on the recorded pair lists, every slot bit for bit |
| R2 | banks and label files → training tables | float32 of the bank value and the documented label transform, exact |
| R3, R4 | tables → base fits → continuations | replaying the recorded fit in a bubblewrap sandbox that hides `/var/tmp`, `/mnt/v` and `~/tmp` and mounts only released bytes at the recorded paths; checkpoints and bakes compared with `zentrain.repro` removed |
| R5 | packed bake → shipped weights | `bake_dial_refit strip` of `zentrain.repro` and `zentrain.sample_coverage` |

Measured on October 31: the four continuations (seed 0) replay from the release with every checkpoint identical,
and E's and ECodec's replayed bakes strip to the bytes zensim ships (e460a214, f7ecd602). Every feature bank the
models read re-extracts bit-identically from released pixels (BAPPS and the chroma leg by sample). The sandbox found
four inputs that no run record listed, now released. The lineage of any run's records can be closed with
`scripts/repro/lineage.py`. The release repository's STATUS.md lists what is not yet reproduced.

---

## 1. The spine

Every number we publish must chain back to bytes. The chain is:

```
  a claim in a paper
        │
        ▼
  a bake_verdict report          ← names its inputs by sha256 (§2.1)
        │  bake sha256
        ▼
  a manifest in zensim/weights/manifests/   ← found by `grep -rl <sha>` (§2.2)
        │  [inputs.*].sha256 + trainer_commit + [reproduce].command
        ▼
  data on R2/Tower + code at a commit       ← verified before training (§2.3)
```

**Each arrow is a content hash, and each is checked by a test.** Given a
number, you can walk the chain in about 30 seconds:

```sh
# 1. the verdict's Provenance table names the bake's sha
grep -A8 '## Provenance' verdict.md

# 2. which recipe produces that bake?
grep -rl b6fe5233ee9c752d zensim/weights/manifests/
#   -> b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07.toml

# 3. what does it take to rebuild it?
grep -E '^(command|asserts|lineage)' zensim/weights/manifests/b_sdr_*.toml
#   -> bash scripts/reproduce_b.sh   (asserts sha b6fe5233, cmp byte-identical)
```

The chain closed on 2026-07-15. Before that it was broken at the first arrow:
a verdict named its bake by **path** (`/mnt/v/output/zensim/r7_rust/seed7_hf0.bin`)
— one machine's scratch volume, no hash — so there was nothing to grep for.

---

## 2. What is enforced, and by what

### 2.1 A verdict names its inputs by content

`bake_verdict` emits a **Provenance** table before any result: the bake's
sha256 + size, every corpus's resolved path + sha256 + size, and the git commit
of the code. If the working tree is dirty it says so **loudly** — a number from
uncommitted code is not reproducible by anyone, including its author later.

*Enforced by:* `bake_verdict::tests::provenance_names_every_input_by_sha256`.
Hashing costs ~10 ms per corpus against a ~3.5 s run.

This also answers "*which* corpus?", which used to be genuinely ambiguous: two
plausibly-named CID22 val parquets exist —
`2026-05-15-full-features/cid22_features_372col_2026-05-15.parquet` (`a1050ace…`,
the binary's default) and `canonical-2026-05-21/val/cid22.parquet` (`6eea0825…`,
canonical per CLAUDE.md). **Measured 2026-07-15: identical.** Same 4,292 rows in
the same order; `human_score` and all 372 features byte-identical. The canonical
one merely adds 21 target columns. So no past number is wrong — but the report
could not *tell* you that, and "they happen to agree" was a fact someone had to
go measure. Now the sha is in the artifact. (Collapsing the two is §4.)

### 2.2 A shipped bake has a manifest that identifies it

`zensim/weights/manifests/README.md` has always required one per shipped bake.
Nothing checked it until 2026-07-15, and the result was:

- **`[bake]` was never parsed.** `RawBake` carried only `file`; serde dropped
  `sha256` and `file_bytes`. `verify_inputs()` checked every *input* hash;
  nothing read the *output* hash. The field was inert prose in a TOML costume.
- **128 of 142 manifests recorded the same sha256** (`d0ef7a30…` — the shipped
  Profile A bake) and the same `[eval]` (`cid22_srocc = 0.8657` — Profile A's
  number). Exactly one described that bake.
- **Neither shipped bake (B, BHdr) had a manifest at all** — and B has been the
  default profile since 2026-07-12.

*Enforced by:* `zensim-validate/tests/shipped_bake_provenance.rs`
- `every_shipped_bake_has_a_manifest_that_identifies_it` — the shipped set is
  parsed out of `profile.rs`'s `include_bytes!`, not hardcoded, so a new profile
  without a manifest fails rather than silently widening the gap.
- `no_manifest_misdescribes_a_bake_that_exists` — was 127-of-138 red; now 0.
- `no_seed_fork_manifest_families` — see §3.

### 2.3 A recipe names its data and its compiler

A manifest records every input's `sha256` + `rows` + R2/Tower mirror, and the
`trainer_commit` that produced the bake. Both are checked **before** training
starts: drift fails loud rather than producing a bake that silently won't match.

`trainer_commit` is load-bearing, not ceremony: the 2026-07-01 v47 reproduction
proved training is deterministic (same code + data + seed → byte-identical bake)
**and** that unrelated-looking trainer commits break it. This is why `v52`/`v53`
keep their per-seed manifests — their seeds ran on different builds, and one
manifest cannot honestly record two.

*Enforced by:* `train_manifest::verify_inputs` / `verify_bake`, plus the
trainer_commit gate in `zensim_mlp_train`.

---

## 3. One obvious way to do each thing

The canonical owner per task is the table in **`CLAUDE.md` §"NO DUPLICATE
IMPLEMENTATIONS"** — that is the registry; this section adds only the rules that
are specifically about *scientific* reproducibility.

### A seed is not a recipe

`--manifest foo.toml --seed 17 --out X.bin` **is** `foo_s17.toml`: the manifest
supplies defaults and explicit CLI flags win (`zensim_mlp_train.rs:1013`).

144 manifests carried 58 recipes until 2026-07-15. `w7_guard` alone was 16 files
for one recipe across 16 seeds. 76 were collapsed away. They were generated by
`scripts/v_next/make_manifest.py` — **a tool written to prevent copy-errors**,
which recomputed every `[inputs.*]` hash while carrying the base's
`[bake].sha256` / `[eval]` through untouched. It now drops outcome fields on
fork, because a forked manifest describes a bake that does not exist yet and the
only honest hash is no hash.

*Enforced by:* `no_seed_fork_manifest_families` (a **naming** rule — recipe
hashing was tried and false-positives on four legitimate cases, the worst being
`v47_mainfix_repro` ≡ `v47_strict_qat`: the artifact that *proves* v47
reproduces byte-identically would read as duplication).

### Prove equivalence before deleting a recipe

A driver/manifest/script is redundant when **measured** so, never when it looks
so. The template: `run_v47_masked_strict_2026-05-27.sh` was deleted only after
all 20 `[training]` hyperparameters and all 5 group `train_w`/`val_w` were shown
to match `v47_strict.toml` exactly (0 mismatches).

Its sibling `run_v47_masked_partial_2026-05-27.sh` was **kept**: no manifest
covers that variant, so it is still the only record of that recipe. Deleting a
doc's cited recipe to lower a file count is the V32 recipe-archaeology incident
on purpose — a recipe reconstructed from prose alone scored CID22 **0.295**
against a documented **0.8879**.

### Report the panel, not SROCC

Every ship/no-ship verdict cites the full Mohammadi 2025 panel (SROCC + PLCC +
KROCC + OR + PWRC + Z-RMSE) at aggregate **and** 10-band level, plus the dial
panel. When the panel disagrees with SROCC, the panel wins. Details:
`CLAUDE.md` §"SROCC-only verdicts BANNED" and `docs/EVAL_PANEL_REQUIREMENT.md`.

### One seed cannot measure a small effect

Demonstrated against ourselves on 2026-07-15: the r7 HF doc reported "Rust's
CID22 −0.0047 reproduces Python's −0.0041". A seed sweep showed the delta
**flips sign** across seeds (−0.0047 at seed 1, +0.0012 at seed 7) — two draws
from one noise band agreeing by luck. Retracted; see
`benchmarks/r7_hf_rust_reproduction_2026-07-15.md`. Any effect near ~0.005 needs
a sweep before it is a finding. This is the "post-selection small-n" family in
`docs/DATASET_HISTORY.md` §0.

---

## 4. Known gaps (honest list)

These are real and unfixed. Nothing below is claimed to work.

1. **The manifest schema models one of two bake-production paths.** It describes
   a `zensim_mlp_train` run. **B and BHdr — the bakes we actually ship — are
   linear bakes**, so their manifests carry `[bake]` + `[reproduce]` and no
   `[training]`. They reproduce byte-exactly (`scripts/reproduce_b.sh`,
   `scripts/reproduce_bhdr.sh`), and since 2026-07-29 both chains are
   **pure Rust** (BHdr's Python lasso stage was ported as `bake_dial_refit
   fit-lasso`, task #68). What remains of this gap is schema-shaped, not
   language-shaped: linear-fit runs still aren't described by a `[training]`
   manifest section, and the λ-pick lineage lives in
   `scripts/v_next/linear_projections_2026-07-03.py` history rather than a
   manifest.
2. **69 of 74 shell drivers bypass the manifest.** The v47 check (§3) proves the
   migration is lossless; it has not been done. The 20 `run_*_seed.sh` files are
   each cited by 1–4 benchmark docs, so they must be *migrated*, not deleted.
3. **Two CID22 val parquets** (§2.1). Measured identical, but they must be kept
   in sync by hand and `bake_verdict` defaults to the non-canonical one.
4. **Eval-root split: RESOLVED at the slot layer 2026-08-05.** The
   `hf_nearlossless` canonical bytes still live in `canonical-2026-07-15/`,
   but its slot is now root-relative via a symlink in
   `2026-05-15-full-features/` (sha-verified identical; same pattern as the
   720 root's nonphoto/imazen26 slots), so **every** slot obeys
   `--features-root` and `PINNED_OUTSIDE_FEATURES_ROOT` is empty — the goal
   state. The audit test also judges absoluteness platform-independently now
   (Windows called the old `/mnt/v/...` pin "relative" and fired the
   staleness half — the sole windows-x64 CI red for weeks). A root without
   the file hard-errors with the R2 restore command. (`nonphoto` was the
   same hazard for no benefit — its file was in the default root all along;
   fixed 2026-07-15.)
5. **~40 tracked scripts are referenced by nothing.** "Unreferenced" is not
   "dead" — one of them is six hours old. Each needs the §3 treatment.

### Closed since this file was written

- **Reproduction provenance is now EMBEDDED in the bake bytes (mandatory).**
  2026-07-27: `zensim_mlp_train` assembles a `zentrain.repro` metadata entry —
  content-addressed inputs (canonical path + sha256 + rows per parquet), seed,
  epochs, structured hyperparams, full argv, trainer HEAD at train time, host,
  timestamp — and embeds it into the ZNPR at the single write choke-point via
  `zenpredict_bake::append_metadata_utf8` (section-level splice, score/byte
  identity gated by composer-equivalence tests in zenpredict-bake). Embedding
  failure is FATAL (exit 4); there is no opt-out flag. `.spec.json` carries the
  same structured `inputs` + `seed` for sidecar-only tooling. `bake_verdict
  --full-json` emits `repro` with a source ladder (embedded > sidecar > null +
  loud warning) and the gauntlet dashboard shows a REPRO badge per model.
  Legacy bakes cannot be retro-embedded (bytes frozen) — they render as
  SIDECAR or NO-REPRO honestly.


- **BHdr's reproduction chain is Python-free.** 2026-07-29 (task #68):
  `bake_dial_refit fit-lasso` ports the `linear_projections` gram-lasso +
  f16-pack + anchor-spline chain to Rust — lasso w/bias/mu/sd f64 BIT-EXACT vs
  the Python fit (`--parity-fit` gate), whole file sha `7d7f2123…` byte-identical.
  `scripts/reproduce_bhdr.sh` now runs zero Python between fit and bake.
  Details: `benchmarks/key_bake_repro_verification_2026-07-29.md`.

- ~~`verify_bake` is not wired post-train.~~ **Fixed 2026-07-15** (`f55551e1`).
  `zensim_mlp_train` now hashes the bake it just wrote and compares it to the
  manifest's `[bake].sha256`: match → `REPRODUCED`, differ → loud mismatch +
  exit 3, no claim → silent (a recipe makes none). This is the check that would
  have caught all 128 forks at creation, and it makes `--manifest` self-checking
  — `reproduce_v47.sh` now verifies its reproduction instead of asserting it in
  prose.

---

## 5. Why this file exists

Three rules in this repo were **declared, mechanized, and never checked**, all
found on the same day:

| rule | mechanism built | what was missing | cost |
|---|---|---|---|
| "both impls must be kept in lock-step" (`zensim-train-core/src/stats.rs`) | two stat impls | the lock-step test; the header cited a parity script **that never shipped** | unknown drift for months |
| manifests verify their files | `verify_inputs()` | `verify_bake()` — inputs verified, outputs assumed | 128 manifests naming the wrong bake |
| `--mse-weight` panics on other heads | the guard | it sat **inside** the branch it tested `!` for — unreachable | the flag was silently discarded |

The pattern is identical every time: **the careful half gets built, the adjacent
check does not, and the gap is invisible precisely because the careful half
looks like diligence.** Documentation cannot fix this; documentation *is* what
failed. Only a test that fails on the bad state fixes it.

Corollary, learned the same day: **an always-red gate carries zero bits** — it
cannot distinguish "broken" from "still broken", so the next breakage is free.
And a gate that cannot fire carries none either. Every gate added here was
verified to bite (inject the bad state → red; remove it → green) before landing.
