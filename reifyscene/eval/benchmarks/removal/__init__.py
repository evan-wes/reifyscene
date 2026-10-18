"""Removal against the owner's four criteria: no trace, empty space, seen background shown,
unseen background reported (T1-T4 of ``docs/design/2026-10-17-removal-by-ownership.md``).

See :mod:`reifyscene.eval.benchmarks.removal.score`.
"""

from reifyscene.eval.benchmarks.removal.score import (
    METHODS,
    PASS,
    ScoreConfig,
    ranking,
    score_run,
    verdict,
)

__all__ = ["METHODS", "PASS", "ScoreConfig", "ranking", "score_run", "verdict"]
