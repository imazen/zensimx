#!/usr/bin/env bash
# Regenerate the slim crate's src/ from this research crate: copy, fix features, drop orphans.
# Usage: regen_slim_src.sh <research-repo> <slim-repo>
set -euo pipefail
R="$1"; S="$2"
here="$(cd "$(dirname "$0")" && pwd)"
rm -rf "$S/zensim/src"
cp -a "$R/zensim/src" "$S/zensim/src"
python3 "$here/cfg_prune.py" \
  --set feature-regime-v2=1 --set deprecated-profiles=1 \
  --set custom-profiles=0 --set candidate-profiles=0 --set corruption-head=0 --set oracle=0 \
  --set cross-revision-diagnostic=0 --set iw-diagnostics=0 --set streaming_strips_oom=0 \
  --set cfg:gmsbank_calibration_instrument=0 --set cfg:restore_cuts_instrument=0 --set cfg:steerfix_instrument=0 \
  $(find "$S/zensim/src" -name '*.rs') | grep -v "^changed" || true
cd "$S/zensim/src"
rm -f attribution.rs corruption_head.rs gmsbank_calibration_instrument.rs gmsbank_matrix_instrument.rs \
  local_refine.rs map_paint.rs perf_clock.rs metric/mapperf_audit.rs metric/steer_exact.rs \
  metric/steerfix_p3.rs metric/steerfix_packet.rs metric/steerfix_precision.rs
