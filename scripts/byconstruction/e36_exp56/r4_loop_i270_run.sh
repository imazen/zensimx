#!/bin/bash
# R4 on i270 (benchmarks may not run on dev): sync the study's inputs at the same absolute paths, encode every arm fresh
# with judging, BD-rate against butteraugli (the judge), p30L, the map-300 twin and F3, then compare the shared arms'
# bitstreams with experiment 3 step 4's i270 run (the reproduction control).
set -euo pipefail
H=i270; S=$HOME/tmp/e36test/r4loop; BIN=$(cat $S/loop-bin)
ssh $H 'sudo -n mkdir -p /var/tmp/byconstruction-bin && sudo -n chown -R lilith:lilith /var/tmp/byconstruction-bin'
for p in $BIN $S; do ssh $H "mkdir -p $p"; tar -C $p --exclude ./loop-target -cf - . | ssh $H "tar -C $p -xf -"; echo "synced $p"; done
for f in $HOME/tmp/fchroma-bakes/F3.bin $HOME/tmp/e36test/arms/p30L-s0/pack_e059/refit/production-f16.bin \
         $HOME/tmp/e36test/exp3m/runs/p30L-map300-s0/pack_e059/refit/production-f16.bin $HOME/tmp/e36test/r4/runs/p30L-map300dv-s0/pack_e059/refit/production-f16.bin $HOME/tmp/byc/full/mono-thr-mp32-r7_s0/refit/production-f16.bin; do
  ssh $H "mkdir -p $(dirname $f)"; cat $f | ssh $H "cat > $f.part && mv $f.part $f"; test "$(ssh $H sha256sum $f | cut -c1-64)" = "$(sha256sum $f | cut -c1-64)"; done
ssh $H "mkdir -p ~/work/zen/zensimx-e36loop-r4"; tar -C $HOME/work/zen/zensimx --exclude ./target --exclude '*/target' --exclude ./.jj --exclude ./.git --exclude ./zensim-bench/target -cf - . | ssh $H "tar -C ~/work/zen/zensimx-e36loop-r4 -xf -"
ssh $H "bash -s" <<'REMOTE'
set -euo pipefail
S=$HOME/tmp/e36test/r4loop; RD=/var/tmp/e36-jxl-steer-r4-i270; OLD=/var/tmp/e36-jxl-steer-map-i270
test ! -e $RD; mkdir -p $RD/corpus; cp /var/tmp/byconstruction-jxl-steer/corpus/corpus.tsv $RD/corpus/
cd ~/work/zen/zensimx-e36loop-r4
P() { python3 - $S/loop-calibration.tail.tsv $1 <<'PY'
import csv, sys
rows=[r for r in csv.DictReader(open(sys.argv[1]),delimiter="\t") if r["metric"]==sys.argv[2] and r["map"]=="diffmap" and r["tail"]=="max"]
assert len(rows)==1; print(rows[0]["scale"]+","+rows[0]["exponent"])
PY
}
LIN="M7lintail=zensim-Mfull@rev7@$(P zensim-Mfull@rev7) F3lintail=zensim-F3@rev7@$(P zensim-F3@rev7) p30Llintail=zensim-p30L@rev7@$(P zensim-p30L@rev7) map300lintail=zensim-map300@rev7@$(P zensim-map300@rev7) map300dvlintail=zensim-map300dv@rev7@$(P zensim-map300dv@rev7)"
echo "$LIN" > $S/LINEAR_ARMS.txt
(nohup nice -n19 ionice -c3 bash -c "STEER_RD=$(cat $S/loop-bin)/steer_rd ZM_BIN=/var/tmp/byconstruction-bin/zenmetrics-judge-7a0f8394/zenmetrics B_TAIL=1.815210e-2,0.7 DVIFM_TAIL=2.960654e-1,0.35 THREADS=20 RD_COMPARE_MODE=bdrate ARMS=\"none butteraugli B M7lintail F3lintail p30Llintail map300lintail map300dvlintail\" LINEAR_ARMS=\"$LIN\" bash scripts/byconstruction/jxl_steer_rd.sh --historical-replay $RD > $S/run-i270.log 2>&1; echo rc=\$? >> $S/run-i270.log; for base in butteraugli p30Llintail map300lintail F3lintail; do python3 scripts/v_next/rd_probe_analyze_2026-07-18.py $RD --bdrate-vs \$base > $S/bdrate-vs-\$base.md 2>&1; done; python3 - $RD/jxl $OLD/jxl > $S/REPRODUCTION.txt 2>&1 <<'PY'
import csv, sys
from pathlib import Path
new, old = Path(sys.argv[1]), Path(sys.argv[2])
def rows(root, label):
    return {(r['image'], r['distance']): r['bitstream_sha256'] for r in csv.DictReader(open(root / f'loop_{label}.tsv'), delimiter='\t')}
for label in ('none', 'butteraugli', 'B', 'M7lintail', 'F3lintail', 'p30Llintail', 'map300lintail'):
    a, b = rows(new, label), rows(old, label)
    bad = [k for k in a.keys() | b.keys() if a.get(k) != b.get(k)]
    print(label, len(a), 'encodes;', len(bad), 'differ from experiment 3 step 4')
PY
touch $S/I270_DONE" > /dev/null 2>&1 &)
sleep 5; tail -2 $S/run-i270.log
REMOTE
