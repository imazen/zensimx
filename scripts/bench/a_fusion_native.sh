#!/usr/bin/env bash
# Native build/test/measurement owner; the shared lock is reacquired per job.
set -euo pipefail
host=$(hostname)
case "$host" in i270) memory=16G ;; dev) memory=24G ;; *) echo 'fusion native work requires an authorized host' >&2; exit 2 ;; esac
export PATH="$HOME/.cargo/bin:$HOME/.local/share/mise/shims:$PATH"
stage=${1:?build|crate|stable|test|tiers|instrument|split|identity|identity-full|identity-era|nearid|steer|measure|eval}
study=${2:?absolute study root}
label=${3:?immutable output label}
if [[ $stage = measure-cell && ${A_FUSION_HOST_LOCK_HELD:-0} != 1 ]]; then
    echo 'measure-cell requires a_fusion_host_cell.sh and its quiet gate' >&2
    exit 2
fi
if [[ $host = dev && $stage != build && $stage != measure-cell ]]; then
    echo 'dev supports short build and measure-cell batches; qualification remains on i270' >&2
    exit 2
fi
export TMPDIR="$study/scratch" CARGO_TARGET_DIR="$study/target"
export CARGO_PROFILE_RELEASE_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0 CARGO_PROFILE_DEV_DEBUG=0 CARGO_INCREMENTAL=0
export RUSTFLAGS='-C force-frame-pointers=yes'
export ZENSIM_BUILD_COMMIT
ZENSIM_BUILD_COMMIT=$(cat "$study/SOURCE_COMMIT.txt")
features=training,custom-profiles,classification
out="$study/$label"
mkdir -p "$TMPDIR" "$out/logs"
cd "$study/source"
jobs=8
[[ $stage != measure && $stage != measure-cell ]] || jobs=16
if [[ $stage != measure && $stage != measure-cell && ${A_FUSION_LOCK_HELD:-0} != 1 ]]; then
    exec flock "$HOME/tmp/heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem "$memory" --jobs "$jobs" -- \
        env A_FUSION_LOCK_HELD=1 bash "$0" "$@"
fi
trap 'status=$?; printf "%s %s %s pid=%s rc=%s\n" "$(date -u +%FT%TZ)" "$label" "$stage" "$$" "$status" >> "$out/stages.tsv"' EXIT
heavy() {
    printf '%s %s %s launcher=%s\n' "$(date -u +%FT%TZ)" "$label" "$stage" "$$" >> "$study/PROGRESS.txt"
    if [[ ${A_FUSION_LOCK_HELD:-0} = 1 ]]; then
        "$@"
    else
        flock "$HOME/tmp/heavy.lock" "$HOME/work/claudehints/scripts/run-heavy" --mem "$memory" --jobs "$jobs" -- "$@"
    fi
}
bake=${A_FUSION_BAKE:-$HOME/tmp/zensim-cost/production-f16.bin}
sha=185f43f7e5e1327c34ad18f0b73b96b872663726a0fac6ff524430d0d8430088
case "$stage" in
    build)
        test ! -e "$out/a_family_cost"
        heavy cargo fmt -p zensim
        shopt -s nullglob
        for source in zensim/src/blur*.rs zensim/src/fused_basic_fusion.rs; do
            [[ ! -f $source ]] || heavy rustfmt --edition 2024 "$source"
        done
        heavy cargo clippy --release --locked -p zensim --lib --example a_family_cost --features "$features" -- -D warnings 2>&1 | tee "$out/logs/clippy.log"
        heavy cargo build --release --locked -p zensim --example a_family_cost --example split_parity_golden --example serve_custom_bake --features "$features" 2>&1 | tee "$out/logs/build.log"
        for name in a_family_cost split_parity_golden serve_custom_bake; do
            cp "$CARGO_TARGET_DIR/release/examples/$name" "$out/$name"
        done
        sha256sum "$out/a_family_cost" "$out/split_parity_golden" "$out/serve_custom_bake" > "$out/BINARY_SHA256SUMS"
        rg --files zensim/src | sort | xargs sha256sum > "$out/SOURCE_SHA256SUMS"
        sha256sum zensim/examples/support/a_fusion.rs scripts/bench/a_fusion_native.sh >> "$out/SOURCE_SHA256SUMS"
        tar -czf "$out/source.tar.gz" zensim/src zensim/examples/a_family_cost.rs \
            zensim/examples/support/a_fusion.rs zensim/Cargo.toml Cargo.lock
        sha256sum "$out/source.tar.gz" > "$out/SOURCE_ARCHIVE_SHA256SUMS"
        ;;
    crate|test)
        # Complete normal crate contract. The separate explicit ignored-test
        # diagnosis remains a recorded failure; this stage cannot waive it.
        heavy cargo test --release --locked -p zensim --features "$features" --lib --tests -- --test-threads=1 2>&1 | tee "$out/logs/test.log"
        heavy cargo test --release --locked -p zensim --features "$features" --doc -- --test-threads=1 2>&1 | tee "$out/logs/doc.log"
        heavy python3 -B scripts/bench/a_extraction_fusion.py test-coverage "$out" "$out/TEST_COVERAGE.json"
        ;;
    stable)
        heavy python3 -B scripts/bench/a_fusion_stable.py \
            "${A_FUSION_BEFORE_TEST:-$study/baseline-stable/zensim-test}" "$out/steer_instrument" \
            "$study/tools/sde-external-10.13.1-2026-07-28-lin/sde64" "$out/stable-diff" \
            2>&1 | tee "$out/logs/stable-diff.log"
        ;;
    tiers)
        heavy cargo test --release --locked -p zensim --features "$features" --lib --test featcanon_rev5_parity --test featcanon_rev7_parity --test featcanon_tier_parity --no-run --message-format=json > "$out/tier-build.jsonl" 2> "$out/logs/tier-build.log"
        python3 - "$out" <<'PY'
import json,pathlib,shutil,sys
out=pathlib.Path(sys.argv[1])
records=[json.loads(line) for line in (out/'tier-build.jsonl').read_text().splitlines() if line.startswith('{')]
assert any(r.get('reason')=='build-finished' and r.get('success') for r in records)
for r in records:
    if r.get('reason')=='compiler-artifact' and r.get('executable'):
        shutil.copy2(r['executable'],out/('tier-'+r['target']['name']))
PY
        sde="$study/tools/sde-external-10.13.1-2026-07-28-lin/sde64"
        for revision in 4 5 7; do
            if [[ $revision = 4 ]]; then name=featcanon_tier_parity; else name=featcanon_rev${revision}_parity; fi
            heavy env ZENSIM_FORMULA_REV="$revision" "$sde" -spr -- "$out/tier-$name" \
                --exact "rev${revision}_vector_bit_identical_under_token_permutations" --nocapture --test-threads=1 \
                2>&1 | tee "$out/logs/sde-rev${revision}.log"
        done
        heavy env ZENSIM_FORMULA_REV=7 FUSED_HV_CHILD=1 FUSED_ACT_CHILD=1 "$sde" -spr -- "$out/tier-zensim" \
            --exact blur::tests::fused_moment_pass_matches_the_two_passes --nocapture --test-threads=1 \
            2>&1 | tee "$out/logs/sde-moments.log"
        # The owner-approved baseline finding is exercised by `stable` on
        # both binaries at Rev1/Rev7, native/SDE, comparing every differing bit.
        ;;
    instrument)
        test ! -e "$out/steer_instrument"
        heavy env RUSTFLAGS="$RUSTFLAGS --cfg steerfix_instrument" cargo test --release --locked -p zensim --features "$features" --lib --no-run --message-format=json > "$out/instrument-build.jsonl" 2> "$out/logs/instrument-build.log"
        python3 - "$out" <<'PY'
import json,pathlib,shutil,sys
out=pathlib.Path(sys.argv[1])
records=[json.loads(line) for line in (out/'instrument-build.jsonl').read_text().splitlines() if line.startswith('{')]
assert any(r.get('reason')=='build-finished' and r.get('success') for r in records)
paths={r['executable'] for r in records if r.get('reason')=='compiler-artifact' and r.get('executable') and r['target']['name']=='zensim'}
assert len(paths)==1
shutil.copy2(paths.pop(),out/'steer_instrument')
PY
        sha256sum "$out/steer_instrument" > "$out/STEER_SHA256SUMS"
        ;;
    split)
        heavy env ZENSIM_FORMULA_REV=7 "$out/split_parity_golden" \
            /mnt/tower/output/zensim/mlic-nearid-xrefaic256e59-20261028-run01/models/XrefAIC256e59_s0.bin \
            /mnt/tower/output/zensim/mlic-nearid-aiconlye29-20261028-run01/models/AICOnlye29_s0.bin \
            > "$out/golden.tsv" 2> "$out/logs/golden.log"
        cmp "$study/before2/golden.tsv" "$out/golden.tsv"
        ;;
    identity-full)
        heavy env ZENSIM_FORMULA_REV=7 "$study/tools/sde-external-10.13.1-2026-07-28-lin/sde64" -icl -- \
            "$out/serve_custom_bake" --e33-identity "$study/fixtures/identity.json" "$study/fixtures/fx1.json" \
            "$study/fixtures/production-r5.bin" "$out/identity.jsonl" "$bake" 2>&1 | tee "$out/logs/identity.log"
        ;;
    identity-features)
        heavy env ZENSIM_FORMULA_REV=7 A_FUSION_IDENTITY_FIRST_ONLY=1 \
            "$out/a_family_cost" fusion identity "$study/fixtures/identity.json" "$bake" "$out/identity-features.jsonl" \
            2>&1 | tee "$out/logs/identity-features.log"
        ;;
    identity|identity-era)
        heavy python3 -B scripts/bench/a_fusion_identity.py "$out/a_family_cost" \
            "$study/tools/sde-external-10.13.1-2026-07-28-lin/sde64" \
            "$study/fixtures/identity.json" "$bake" "$out/identity.jsonl" \
            2>&1 | tee "$out/logs/identity.log"
        ;;
    nearid)
        heavy env ZENSIM_FORMULA_REV=7 RAYON_NUM_THREADS=1 "$out/serve_custom_bake" --nearid \
            "$study/fixtures/nearid.json" "$out/nearid" candidate-fusion "$bake" "$sha" \
            2>&1 | tee "$out/logs/nearid.log"
        heavy python3 -B scripts/prodqual_label_free.py --nearid-candidate "$out/nearid" fusion \
            2>&1 | tee "$out/logs/nearid-report.log"
        ;;
    steer)
        heavy python3 -B scripts/byconstruction/gates_steer_sharded.py --name fusion --model "$bake" \
            --out "$out/pixel" --bin "$out/steer_instrument" --formula-rev 7 --shards 8 \
            --packet "$study/fixtures/steer.json" 2>&1 | tee "$out/logs/steer.log"
        ;;
    measure|measure-cell)
        thread_values=(1 16)
        size_values=(256 1024 large)
        if [[ $stage = measure-cell ]]; then
            size_values=("${4:?size}")
            thread_values=("${5:?threads}")
            [[ ${size_values[0]} =~ ^(256|1024|large)$ && ${thread_values[0]} =~ ^(1|16)$ ]]
        fi
        export A_FUSION_WORKER_MEM="$memory"
        for threads in "${thread_values[@]}"; do
            if [[ $host = dev ]]; then
                if [[ $threads = 1 ]]; then cpus=0; else cpus=0-7,16-23; fi
            else
                if [[ $threads = 1 ]]; then cpus=2; else cpus=0-15; fi
            fi
            for size in "${size_values[@]}"; do
                cell="$out/timing/${size}-${threads}t"
                test ! -e "$cell"
                heavy env ZENSIM_FORMULA_REV=7 ZENBENCH_NO_CALIBRATE=1 RAYON_NUM_THREADS="$threads" \
                    python3 -B scripts/bench/a_family_cost_admit.py "$cell/admission.json" -- \
                    taskset -c "$cpus" "$study/${A_FUSION_CONTROLLER_LABEL:-before3}/a_family_cost" fusion compare \
                    "$study/${A_FUSION_BEFORE_LABEL:-before3}/a_family_cost" "$out/a_family_cost" "${A_FUSION_PAIRS:-$HOME/tmp/zensim-cost/pairs}" \
                    "$bake" "$size" "$threads" "$cell" 2>&1 | tee "$out/logs/measure-${size}-${threads}t.log"
                printf '%s %s %s %s capture complete\n' "$(date -u +%FT%TZ)" "$label" "$size" "$threads" > "$cell/CAPTURE_COMPLETE.txt"
                # a_fusion_parse.sh timing replays analysis from inner/workers JSON.
                # A parser failure cannot restart this measurement.
            done
        done
        ;;
    eval)
        heavy env ZENSIM_FORMULA_REV=7 RAYON_NUM_THREADS=1 "$out/a_family_cost" fusion pairs \
            "${A_FUSION_EVAL_TABLE:-$study/fixtures/eval-pairs.tsv}" "$bake" "$out/eval.jsonl" 2>&1 | tee "$out/logs/eval.log"
        ;;
    *) exit 2 ;;
esac
