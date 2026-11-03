# E36 experiments 5 and 6 tooling

Registrations and results: `benchmarks/e36_subset_test_2026-11-01.md` (experiments 5 and 6).

The runs use F3's `base-fit` step (`zensim-training` `recipes/F3.json`) through a copy of its frozen scripts
(`.../fx2-full-general-e59/source/scripts`, the path the recipe names) with `frozen-scripts.patch` applied and
`human_subset.py` added. The patch applies to these frozen files:

| file | sha256 |
|---|---|
| `rev4_featpot/v2_confirm_fit.py` | `23da94482c58265f0483b0fc0eca61418caa43f3c0873edbedbf46dbcede6212` |
| `rev4_featpot/subjective_training_leg.py` | `3505d8d31354c9216af0d945cfc3bfc3e0bf02100b62b8697e85369a7173d46f` |

The patch adds `--human-sources` (subset the design-released human table by source), `--human-rows` (keep listed
rows), `--drop-groups` and `--mos-share` (the kept MOS legs share one total train weight; dataset legs record an
`experiment5_scale` that both the Python and the trainer's admission check apply). The original authority and
admission checks still run on every arm.

- `exp5_make_cmds.py`, `exp6_make_cmds.py`: one `{argv, env}` command per arm and seed. Experiment 6's
  pair-sampling arms run the trainer through a wrapper that rewrites only the `--pair-sampling` value.
- `run_cmd.py`, `wave.sh`: run a wave of commands, staggered, under one heavy lock.
- `prune_scores.py`, `make_prunes.py`: experiment 6's within-reference discordance and the prune row lists.
- `score_runs.sh`, `readout.py`: EVAL scoring of the packed base fits and the error-against-human readout.
- `exp5c_make_cmds.py`: experiment 5 stage 3, the TID2013 + MDID mix through F3's continuation with p30L's terms. The
  patch lets a continuation exclude dataset legs only when an experiment 5 option is given.
- Experiment 3: `step2_prep.py` / `heldout_prep.py` (the fit and held-out view samples), `map_agreement.py`
  (held-out Spearman against butteraugli), `exp3m_make_cmds.py` (map term in the continuation), `exp3b_make_cmds.py`
  (map term in the base fit), `exp3m_post.sh`, `postF.sh`, `postG.sh` (post-run measurements), `primary.py`
  (the registered primary, `SEEDS=0` for seed-0 screens).

Paths inside the scripts are the dev host's scratch layout (`~/tmp/e36test`) where the runs live.
