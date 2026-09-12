"""Shared, stage-agnostic metric primitives (image quality, etc.).

These are the reusable measurement math used by per-stage evaluators (e.g.
``reifyscene/render/eval``) and by cross-stage benchmarks (``reifyscene/eval/benchmarks``),
kept here so neither has to depend on the other.
"""
