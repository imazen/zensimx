# White-paper and dashboard data for the two curated revision-7 models (2026-10-28)

Evaluation-only helpers behind the white paper (a private Claude Doc) and the project dashboard. None of them fits or
selects anything.

- `post_runtime.sh` — after the runtime grid: SSIMULACRA2 and butteraugli (3-norm) from the pinned judge build on the
  13,817 E21 pairs (`/mnt/v/output/zensim/e21-peers-20261028/`), paired EVAL comparisons of both finalists against Mlic,
  the EVAL scatter export, and the curated TRAIN-population views with those peers (`aic_dashboard_data.py --curate`).
- `paper_data.py CURATE_DIR EVAL_DIR OUT_DIR` — six-statistic panel rows (`panels.json`) and 2-D histograms over every
  point (`scatter_bins.json`) for the paper's figures.
- `panels_md.py DATA_DIR` — the Full panels tab as Markdown tables.
- `git_growth.py REPO OUT_DIR` — where the repository's packed history grew (read-only git); behind
  `docs/REPO_SPLIT_PLAN_2026-10-28.md`.

Outputs live under `/mnt/v/output/zensim/paper-20261028/` and `/mnt/v/output/zensim/repo-split-20261028/` on dev.
