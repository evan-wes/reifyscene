"""Cross-stage / cross-method benchmarks.

Each benchmark is a self-contained subpackage (its own schema, comparison, and
README). Benchmarks depend on per-stage evaluators and the shared
``reifyscene/eval/metrics`` primitives — never the other way around.
"""
