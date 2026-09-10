"""Co-visibility-masked novel-view-synthesis benchmark (the DyCheck protocol).

See README.md for the full description. This package is the single source of
truth for the ``reference_benchmark_v1`` record contract and its comparison
logic; both ``scripts/run_reference_benchmark.py`` (Shape-of-Motion emitter) and
reifyscene's own renderer eval import from here.
"""

from reifyscene.eval.benchmarks.covisibility_masked_novel_view_synthesis.compare import compare
from reifyscene.eval.benchmarks.covisibility_masked_novel_view_synthesis.schema import (
    HIGHER_IS_BETTER,
    METRIC_KEYS,
    SCHEMA,
    build_record,
    load_record,
    write_record,
)

__all__ = [
    "SCHEMA",
    "METRIC_KEYS",
    "HIGHER_IS_BETTER",
    "build_record",
    "load_record",
    "write_record",
    "compare",
]
