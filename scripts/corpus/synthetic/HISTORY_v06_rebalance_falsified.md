# V_06 rebalance — FALSIFIED experiment (historical doc)

> **Status:** FALSIFIED 2026-05-18 per PR #31 re-eval. The V_06 + FiLM-gated
> MLP trained on this rebalanced corpus achieved CID22 +0.043 but
> regressed catastrophically on anchor corpora (KADID -0.115, TID -0.128,
> KonJND -0.396 vs Balanced ship). See master inventory at
> `~/work/zen/_ml-inventory-2026-05-20/00-MASTER-SYNTHESIS.md`.
>
> The synthetic content generator (`synth_nonphoto.py`) survives as a
> general-purpose CC0 synthetic corpus tool. Per the 2026-05-26 per-crop
> audit, the axis-inverted (769x513↔513x769, 1022x818↔818x1022) and
> mod-8-boundary variants this experiment used were proven redundant
> — wasted compute. Only 512sq + 1024sq carry independent signal.

---

(original 2026-05-05 handoff content below)

# V0_6 Rebalance Handoff (2026-05-05)

Branch: `v06-rebalanced-corpus`
Worktree: `/home/lilith/work/zen/zensim--v06-rebalance/`
Commit: `af95701` (push: https://github.com/imazen/zensimx/tree/v06-rebalanced-corpus)

## What landed

1. **`benchmarks/v06_cclass/synth_nonphoto.py`** — procedural generator
   for 5 non-photo content classes. CC0 (PIL + matplotlib synthesis,
   no copyrighted assets).

2. **`benchmarks/v06_cclass/expand_size_variants.py`** — derives the 4
   non-square size variants (513x769, 769x513, 1022x818, 818x1022) from
   each `gen-*_1024sq.png` via center-crop + Lanczos. Matches the
   safe-synthetic 6-bucket layout.

3. **`benchmarks/v06_cclass/encode_synth_via_zenjpeg.sh`** — symlinks
   `gen-*` PNGs into a curated dir then invokes the prebuilt
   `generate_zensim_training` binary against ONLY those sources, codec
   = `zenjpeg-420-e1` (matching the e1 fill).

4. **`benchmarks/v06_cclass/build_rebalanced_csv.sh`** — concordance-
   filters the gen-* encoder output, then concatenates safe-base + e1-
   filtered + gen-filtered into the final
   `training_safe_synthetic_rebalanced.csv`.

5. **`benchmarks/v06_cclass/build_cclass_tsv.py`** — extended to
   recognize `gen-screen__`, `gen-doc__`, `gen-chart__`, `gen-line__`,
   `gen-mixed__` prefixes (gen-mixed labelled as `photo` since base
   content is photo with overlay).

6. Train/eval scripts (`train_v06_baseline.sh`, `train_v06_cclass.sh`,
   `eval_v06_cclass.sh`, `compare_v06.py`) copied verbatim from main
   worktree. Override `EXTENDED_CSV` to point at the rebalanced CSV.

## What ran this session

- **Synthesis**: 3282 `gen-*` source PNGs across 5 categories, written
  to `/mnt/v/input/zensim/sources/`:
  - 1000 screenshots (`gen-screen__*`, 500 unique × 2 sizes)
  - 1000 documents (`gen-doc__*`, 500 unique × 2 sizes)
  - 600 line-art (`gen-line__*`, 300 unique × 2 sizes)
  - 406 charts (`gen-chart__*`, 264 unique — chart generator had a
    matplotlib `np.cumsum(generator)` bug that errored ~190 cells; bug
    is fixed in the script but the failed cells were not regenerated
    this session)
  - 276 mixed (`gen-mixed__*`, 138 unique — synthesis was killed near
    the end of the mixed phase to free time budget; mixed only adds
    `cclass_photo` rows so the cclass impact is minimal)

- **Sidecar TSV**: `/mnt/v/output/zensim/v06-rebalance/synth_sources.tsv`
  (3282 rows, columns: `source_path content_class subset seed license`).

- **Size-variant expansion** running in background to derive the 4
  non-square variants from each `gen-*_1024sq.png`. At handoff: ~1200
  of ~6500 derivative PNGs written. The script idempotently skips
  existing files; can be re-run safely.

- **Inventory of pre-existing non-photo content** (CC0/PD, NOT yet
  copied into `/mnt/v/input/zensim/sources/`):
  `/mnt/v/output/zensim/v06-rebalance/nonphoto_inventory.tsv` (169
  items: 99 documents from brown-v-board + gutenberg, 60 procedural
  patterns from `~/work/zentrain-corpus/mlp-tune/synthetic/`, 10 real
  screenshots from `~/work/zentrain-corpus/mlp-tune-fast/gb82-screen/`).
  These could be added to the corpus by copying into `/mnt/v/input/
  zensim/sources/` with a `pre-doc__`, `pre-synth__`, `pre-screen__`
  prefix and re-running the encoder.

## What did NOT run this session

The full pipeline below is wired up but un-executed. Estimated runtime
for a fresh agent picking this up:

| Step | Command | Wall time |
|---|---|---|
| 1. Finish expand_size_variants.py | already running, may be done | ~10 min |
| 2. Re-run charts (matplotlib bug fixed) | `python3 synth_nonphoto.py --screen 0 --document 0 --chart 300 --lineart 0 --mixed 0 --sizes 512,1024` | ~5 min |
| 3. Run zenanalyze on new gen-* sources to extend `zenanalyze_union_v1.tsv` | (needs zenanalyze binary; not in this scope) | ~20 min |
| 4. Encode gen-* via generate_zensim_training | `bash encode_synth_via_zenjpeg.sh` | 30-60 min |
| 5. Build rebalanced CSV | `bash build_rebalanced_csv.sh` | 1 min |
| 6. Build cclass TSV with gen- prefixes | `python3 build_cclass_tsv.py` | <1 min |
| 7. Train V0_6 dct_hf baseline | `EXTENDED_CSV=...rebalanced.csv bash train_v06_baseline.sh` | 30-40 min |
| 8. Train V0_6+cclass | `EXTENDED_CSV=...rebalanced.csv bash train_v06_cclass.sh` | 30-40 min |
| 9. Eval both vs reigning V0_6 | `bash eval_v06_cclass.sh` | 20-30 min |
| 10. Per-content-class SROCC analysis | `python3 compare_v06.py` (extend) | 10 min |

Total remaining: ~3-4 hours.

## Critical step missing from this session: zenanalyze TSV update

The `train_v06_cclass.sh` reads `zenanalyze_union_v1_cclass.tsv`, which
indexes feature rows by `stem` (the source PNG filename without `.png`
extension). The new `gen-*` source PNGs are NOT yet in
`zenanalyze_union_v1.tsv` — features (variance, edge_density,
chroma_complexity, dct_compressibility, etc.) need to be computed
for them via the `zenanalyze` binary before training can use them.

Without that update, training will fall back to default features for
the `gen-*` rows and the cclass signal will be the only differentiator
— which would still test the hypothesis ("does cclass help when content
is balanced?") but loses the per-image complexity context that
V0_6 dct_hf relies on.

The cleanest fix is to extend the build_cclass_tsv.py pipeline to
**also compute zenanalyze features for any source paths missing from
the union TSV**, then write them as new rows to a v2 TSV before adding
the cclass columns. The zenanalyze binary lives at
`./target/release/zenanalyze` and accepts a `--sources` flag.

## Why this stopped before training

Encoding 1700+ unique sources × 2 sizes (or 6 with expansion) × ~40 q
levels × 1 codec = 130k–400k pairs. At realistic local-GPU encoding
rates (the e1 fill was ~5h for 122k pairs with the v08 sweep also
running on R2), this is not safely fittable into one conversation
turn. The scripts are now designed so a follow-up agent can
unattendedly run encode_synth_via_zenjpeg.sh + build_rebalanced_csv.sh
+ training + eval back-to-back.

## Per-content-class SROCC eval — note for the follow-up

Step 10 (per-content-class SROCC) requires the eval CSV to carry a
`cclass` column for each pair. KADID and TID are mostly photo, but
CID22 has at least 5-10 non-photo sources (text, screenshots, line-art).
Splitting CID22 by stem prefix and re-computing per-class SROCC on
gen-*-trained variants is the cleanest way to test the hypothesis
"does the rebalanced cclass training improve non-photo accuracy
specifically?" — even if global SROCC is unchanged, a positive shift on
non-photo CID22 stems is the success signal.

If global SROCC on KADID/TID/CID22 is unchanged after rebalance and
cclass per-content-class shifts are also flat, that's the **negative
finding** the user asked us to be honest about. Report it directly:
"V0_6+cclass does not benefit from content rebalance; the model
architecture cannot leverage cclass signal even with diverse content."

---

## 2026-05-26 update: encode-pipeline shells retained

Per user direction, the encode-orchestration shells were added to this commit
alongside the generators so the full (ref, distorted) pair set is reproducible
end-to-end, not just the source-PNG side:

- `encode_synth_via_zenjpeg.sh` -- bridges synth_nonphoto.py output to
  coefficient's `generate_zensim_training` encoder. **Working on main with
  caveats** (see header in the file).
- `build_rebalanced_csv.sh` -- concordance-filter + concatenate per-codec
  encoded CSVs. **Historical reference only**; does not run on main
  (depends on `build_extended_safe_csv.py` which was not salvaged).
  Documented as a recipe; current-main equivalent uses the
  zensim-validate trainer's `--group` flags directly + the
  contamination_guard.rs trainer-startup check.

The analysis utilities from the original v06_cclass/ dir
(`build_cclass_tsv.py`, `compare_v06.py`, `per_class_srocc.py`) were
intentionally skipped -- their stats snapshots are 3 weeks out of date
and they referenced `dataset_metric_baseline.rs` which was deleted from
main in commit `34f796f` (2026-05-20 cleanup).
