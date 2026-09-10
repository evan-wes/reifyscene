"""The ``reference_benchmark_v1`` record contract.

The wire schema id stays ``reference_benchmark_v1`` for back-compat with the
on-disk SoM baseline JSONs and any tooling that reads them; this module is its
single source of truth (imported by both ``scripts/run_reference_benchmark.py``
and reifyscene's own emitter).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SCHEMA = "reference_benchmark_v1"

# Canonical metric order (DyCheck iPhone protocol): masked photometric first,
# then 3D/2D tracking. NVIDIA fg_/bg_ photometric splits are format extras and
# sort after these in a compare.
METRIC_KEYS: list[str] = [
    "masked_mpsnr",
    "masked_mssim",
    "masked_mlpips",
    "track3d_epe",
    "track3d_pck_10cm",
    "track3d_pck_5cm",
    "track2d_aj",
    "track2d_avg_pck",
    "track2d_occ_acc",
]

# Direction of "better" per metric (used by compare()).
HIGHER_IS_BETTER: dict[str, bool] = {
    "masked_mpsnr": True,
    "masked_mssim": True,
    "masked_mlpips": False,
    "track3d_epe": False,
    "track3d_pck_10cm": True,
    "track3d_pck_5cm": True,
    "track2d_aj": True,
    "track2d_avg_pck": True,
    "track2d_occ_acc": True,
    "fg_psnr": True,
    "fg_ssim": True,
    "fg_lpips": False,
    "bg_psnr": True,
    "bg_ssim": True,
    "bg_lpips": False,
}


def build_record(
    *,
    method: str,
    sequence: str,
    dataset_format: str,
    metrics: dict[str, Any],
    eval: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a schema-compliant record.

    ``eval`` records the measurement protocol (e.g. source + resolution_factor)
    so a cross-emitter compare can flag incomparable protocols. Non-finite/None
    metric values are dropped so they read as "—" in a compare rather than
    silently winning a direction check.
    """
    record: dict[str, Any] = {
        "schema": SCHEMA,
        "method": method,
        "sequence": sequence,
        "dataset_format": dataset_format,
        "eval": eval,
        "metrics": {k: v for k, v in metrics.items() if v is not None},
    }
    if extra:
        record.update(extra)
    return record


def write_record(record: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n")
    return path


def load_record(path: str | Path) -> dict[str, Any]:
    record = json.loads(Path(path).read_text())
    if record.get("schema") != SCHEMA:
        raise ValueError(f"{path}: schema {record.get('schema')!r} != {SCHEMA!r}")
    return record
