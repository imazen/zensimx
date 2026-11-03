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
- Revised plan (`docs/E36_METHOD_REVIEW_2026-11-03.md`), as run: `exp7_make_cmds.py` (R2), `exp8_make_cmds.py` (R3)
  and `exp9_make_cmds.py` (R7) write the commands; `chainK.sh`, `chainR2.sh`, `chainR3.sh` and `chainR7.sh` run the
  waves; `postK.sh` (experiment 5 amendment 7), `postR2.sh`, `postR3.sh` and `postR7.sh` take the measurements
  (`postR7.sh` holds confirmation read 2); `confirm1.sh` is confirmation read 1; `pack_nolock.sh` is F3's pack step
  for one run; `m1_run_existing.sh` ran the M1 panels for arms measured before the review. `confirm1.sh` writes the
  home directory as `$HOME`. The AIC2026 fold scoring scripts are in the private training repository.
- Dial comparison (record: "Dial comparison"): `ladder_smoothness.py` (label-free reversal and wrong-way shares on
  E36's held-out ladders) and `inversion_by_gap.py` (inversion share by quality-setting gap).
- R10 (the mix on F3's recipe): `r10_make_cmds.py` (in the runs' layout `r10/make_cmds.py`), `chainR10.sh`, `postR10.sh`;
  its JPEG XL loop (amendment 1): `../e36_loop_r10.sh`, `r10_chainLoop.sh` and `r10_loop_i270_run.sh`.

Paths inside the scripts are the dev host's scratch layout (`~/tmp/e36test`) where the runs live.
