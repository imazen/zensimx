# Stable dependency audit

This isolated candidate starts from `ad123a8ebe9f028c73bf2cf13e8280c9a6a2e8fa`
and retains its numerical source, fixtures and model inputs. Manifest and
lock updates select the released archmage/magetypes 0.9.31 cohort.

`cargo check --locked --workspace` passes. Strict library Clippy fails with
341 deprecation diagnostics from the old Magetypes `mul_add` spelling,
including `ssim_form` and `feature_v2`. This candidate deliberately preserves
those sources for review; it is not a model, numerical or strict-Clippy
qualification. No training, evaluation, model selection or benchmark ran.

The earlier frozen review cohort remains separate from this main-based
audit. A main-based lock is not a substitute for its exact source pins.
