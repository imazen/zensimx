"""Cross-repo wrapper module — re-exports `join_safety.py`'s API under the
package name `zen_corpus_join`.

This file exists so that other repos (zenmetrics, zenanalyze) can:

    pip install -e /home/lilith/work/zen/zensimx/scripts/canonical_corpus
    from zen_corpus_join import safe_metric_join, ...

Without the wrapper, cross-repo consumers must `sys.path.insert` to point at
this directory and `from join_safety import ...` directly (which is fine, and
is the pattern used by in-tree builders).

KEEP THIS A THIN RE-EXPORT. Do NOT add helpers here — add them to
`join_safety.py` so all consumers (in-tree + cross-repo) see them.
"""
from join_safety import (  # noqa: F401
    JoinSafetyError,
    PER_PAIR_KEY_CANDIDATES,
    REF_ONLY_KEYS,
    HUMAN_SCORE_COL,
    assert_metric_not_constant_per_ref,
    assert_no_leaked_metric_columns,
    attach_metric_positional,
    attach_per_source_features,
    guard_metric_table,
    safe_metric_join,
)

__all__ = [
    "JoinSafetyError",
    "PER_PAIR_KEY_CANDIDATES",
    "REF_ONLY_KEYS",
    "HUMAN_SCORE_COL",
    "assert_metric_not_constant_per_ref",
    "assert_no_leaked_metric_columns",
    "attach_metric_positional",
    "attach_per_source_features",
    "guard_metric_table",
    "safe_metric_join",
]
