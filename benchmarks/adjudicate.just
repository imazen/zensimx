# Source-only quiet-window inspection. This does not release the gate itself.
quiet-marker:
    stat -c '%y %s' /home/lilith/tmp/zensim-paper/rev4/SPEEDQ_DONE.md
    sha256sum /home/lilith/tmp/zensim-paper/rev4/SPEEDQ_DONE.md

# Called manually by this lane, never from an uncontrolled timer.
refresh-lock:
    date -u '+%Y-%m-%dT%H:%M:%SZ codex-adjudicate frozen-model diagnosis; explicit quiet gate' > ../.workongoing
