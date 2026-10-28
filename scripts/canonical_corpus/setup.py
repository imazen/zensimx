"""Minimal installable shim so other repos can `import zen_corpus_join`.

Install (from any repo's CI or dev shell):

    pip install -e /home/lilith/work/zen/zensimx/scripts/canonical_corpus

Then:

    from zen_corpus_join import (
        safe_metric_join,
        attach_metric_positional,
        attach_per_source_features,
        assert_no_leaked_metric_columns,
        assert_metric_not_constant_per_ref,
        guard_metric_table,
        JoinSafetyError,
    )

The package re-exports everything from `join_safety.py` (the single source
of truth). Builders that live in this directory can `from join_safety import`
directly — they do not need the package.
"""
from setuptools import setup

setup(
    name="zen-corpus-join",
    version="0.1.0",
    description=(
        "Shared join-safety guards (anti-corruption-mode wrappers around "
        "pandas/DuckDB joins for zensim/zenmetrics/zenanalyze corpus + "
        "training-data builders)."
    ),
    py_modules=["zen_corpus_join"],
    python_requires=">=3.10",
    install_requires=["numpy", "pyarrow"],
)
