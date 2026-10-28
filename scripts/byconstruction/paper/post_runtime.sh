#!/usr/bin/env bash
# After the r7 runtime grid: peer scores on the E21 sets' pixels, EVAL paired-vs-Mlic and scatter for both finalists,
# and the curated TRAIN-population views with peers. Evaluation only (no fits); all under the shared compute lock.
set -euo pipefail
cd /home/lilith/work/zen/zensimx
P=/mnt/v/output/zensim/e21-peers-20261028
S=/mnt/v/output/zensim/mlic-aic-finalists-eval-20261028
ZM=/var/tmp/byconstruction-bin/zenmetrics-judge-7a0f8394/zenmetrics
(cd /var/tmp/byconstruction-bin/zenmetrics-judge-7a0f8394 && sha256sum --quiet -c SHA256SUMS)
H="flock $HOME/tmp/heavy.lock $HOME/work/claudehints/scripts/run-heavy --mem 16G --jobs 16 --"
for m in ssim2 butteraugli; do
  [ -s $P/judge_$m.tsv ] || $H env RAYON_NUM_THREADS=16 $ZM batch --metric $m --pairs $P/judge_pairs.tsv --output $P/judge_$m.tsv > $P/judge_$m.log 2>&1
done
sha256sum $P/judge_pairs.tsv $P/judge_ssim2.tsv $P/judge_butteraugli.tsv $ZM > $P/SHA256SUMS
for name in MlicXref256e59 MlicAICOnlye29; do
  [ -d $S/eval/paired-vsMlic-$name ] || $H env RAYON_NUM_THREADS=8 TMPDIR=$HOME/tmp ZEN_PANEL_BIN=$S/bin/panel \
    python3 -B scripts/byconstruction/dataset_eval.py paired --out $S/eval --dest $S/eval/paired-vsMlic-$name --a $name --vs Mlic \
    > $S/paired-vsMlic-$name.log 2>&1
done
[ -e $S/eval/eval_scatter.copied.json ] || { mv $S/eval/eval_scatter.json $S/eval/eval_scatter.copied.json; mv $S/eval/eval_scatter_stats.json $S/eval/eval_scatter_stats.copied.json; }
[ -e $S/eval/eval_scatter.json ] || $H env TMPDIR=$HOME/tmp ZEN_PANEL_BIN=$S/bin/panel python3 -B scripts/byconstruction/dataset_eval.py scatter --out $S/eval > $S/scatter.log 2>&1
C=/mnt/v/output/zensim/dashboard-curate-20261028
[ -e $C/curate_stats.json ] || $H env TMPDIR=$HOME/tmp python3 -B scripts/byconstruction/aic_dashboard_data.py --out $C --curate > /mnt/v/output/zensim/paper-20261028/curate.log 2>&1
echo POST_RUNTIME_COMPLETE
