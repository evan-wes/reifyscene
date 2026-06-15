"""Side-by-side scoring of two reference_benchmark_v1 records."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

from room4d.eval.benchmarks.covisibility_masked_novel_view_synthesis.schema import (
    HIGHER_IS_BETTER,
    METRIC_KEYS,
    SCHEMA,
)


def _usable(v: Any) -> bool:
    # json.loads accepts NaN/Infinity, and bool is an int subclass — any of
    # those would silently win/lose a direction check, so only finite reals count.
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def compare(path_a: str | Path, path_b: str | Path) -> None:
    """Print a metric-by-metric comparison table for two record files."""
    a = json.loads(Path(path_a).read_text())
    b = json.loads(Path(path_b).read_text())
    for blob, path in ((a, path_a), (b, path_b)):
        if blob.get("schema") != SCHEMA:
            sys.exit(f"{path}: schema {blob.get('schema')!r} != {SCHEMA!r}")

    name_a = f"{a.get('method', '?')}"
    name_b = f"{b.get('method', '?')}"
    seq = a.get("sequence", "?")
    if b.get("sequence") != seq:
        print(f"[warn] comparing different sequences: {seq} vs {b.get('sequence')}", flush=True)
    if a.get("eval") != b.get("eval"):
        print(
            f"[warn] comparing different eval protocols: {a.get('eval')} vs {b.get('eval')} "
            "— metric values are not directly comparable",
            flush=True,
        )

    metrics_a, metrics_b = a.get("metrics", {}), b.get("metrics", {})
    known = [k for k in METRIC_KEYS if k in metrics_a or k in metrics_b]
    extras = sorted((set(metrics_a) | set(metrics_b)) - set(known))
    keys = known + extras
    width = max(len(k) for k in keys) if keys else 10
    print(f"\nSequence: {seq}")
    print(f"  {'metric':<{width}}  {name_a:>14}  {name_b:>14}  {'better':>8}")

    for key in keys:
        va, vb = metrics_a.get(key), metrics_b.get(key)
        sa = f"{va:.4f}" if _usable(va) else "—"
        sb = f"{vb:.4f}" if _usable(vb) else "—"
        better = ""
        if _usable(va) and _usable(vb) and va != vb:
            if key in HIGHER_IS_BETTER:
                better = name_a if (va > vb) == HIGHER_IS_BETTER[key] else name_b
            else:
                better = "?"
        print(f"  {key:<{width}}  {sa:>14}  {sb:>14}  {better:>8}")
