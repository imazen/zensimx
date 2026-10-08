set working-directory := '..'
evidence := '/mnt/v/output/zensim/adjudicate-2026-10-07'
scratch := '/home/lilith/tmp/adjudicate'
heavy := '/home/lilith/work/zen/scripts/run-heavy'

# Source-only quiet-window inspection. Coordinator subsequently lifted the wait.
quiet-marker:
    stat -c '%y %s' /home/lilith/tmp/zensim-paper/rev4/SPEEDQ_DONE.md
    sha256sum /home/lilith/tmp/zensim-paper/rev4/SPEEDQ_DONE.md

# Called manually by this lane, never from an uncontrolled timer.
refresh-lock:
    date -u '+%Y-%m-%dT%H:%M:%SZ codex-adjudicate frozen-model diagnosis; run-heavy only' > .workongoing

build-serve label:
    {{heavy}} --mem 16G --jobs 8 -- env TMPDIR={{scratch}} CARGO_TARGET_DIR={{scratch}}/target CARGO_INCREMENTAL=0 CARGO_PROFILE_RELEASE_DEBUG=0 cargo build -p zensim --release --features custom-profiles,candidate-profiles --example serve_custom_bake > {{evidence}}/build-{{label}}.log 2>&1

build-owner:
    {{heavy}} --mem 16G --jobs 8 -- env TMPDIR={{scratch}} CARGO_TARGET_DIR={{scratch}}/target CARGO_INCREMENTAL=0 CARGO_PROFILE_RELEASE_DEBUG=0 python3 scripts/prodqual_label_free.py --adjudicate-build-owner {{evidence}} > {{evidence}}/owner-build.log 2>&1

run mode:
    {{heavy}} --mem 8G --jobs 1 -- env TMPDIR={{scratch}} python3 scripts/prodqual_label_free.py --adjudicate-{{mode}} {{evidence}} > {{evidence}}/{{mode}}.log 2>&1

[positional-arguments]
native packet revision label *models:
    #!/usr/bin/env bash
    set -euo pipefail
    packet="$1"; revision="$2"; label="$3"; shift 3
    {{heavy}} --mem 8G --jobs 1 -- env TMPDIR={{scratch}} ZENSIM_FORMULA_REV="$revision" RAYON_NUM_THREADS=1 {{evidence}}/bin/serve_custom_bake --adjudicate "$packet" "$@" > {{evidence}}/"$label".json 2> {{evidence}}/"$label".log

clippy-serve:
    {{heavy}} --mem 16G --jobs 8 -- env TMPDIR={{scratch}} CARGO_TARGET_DIR={{scratch}}/target CARGO_INCREMENTAL=0 CARGO_PROFILE_RELEASE_DEBUG=0 cargo clippy -p zensim --release --features custom-profiles,candidate-profiles --example serve_custom_bake -- -D warnings > {{evidence}}/clippy-serve.log 2>&1
