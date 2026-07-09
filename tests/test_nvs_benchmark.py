from __future__ import annotations

import pytest
import torch

from room4d.eval.benchmarks.covisibility_masked_novel_view_synthesis import (
    SCHEMA,
    build_record,
    compare,
    load_record,
    write_record,
)
from room4d.eval.metrics.image import MaskedSSIM, masked_psnr


# -- masked metrics -------------------------------------------------------

def test_masked_psnr_identical_is_high() -> None:
    img = torch.rand(1, 8, 8, 3)
    assert masked_psnr(img, (img + 0.01).clamp(0, 1)) > 30.0


def test_masked_psnr_ignores_unmasked_region() -> None:
    torch.manual_seed(0)
    pred = torch.rand(1, 8, 8, 3)
    gt = (pred + 0.02).clamp(0, 1)
    gt[:, :4] = 0.0  # heavy corruption in the top half
    mask = torch.ones(1, 8, 8)
    mask[:, :4] = 0.0  # ...which the mask excludes
    masked = masked_psnr(pred, gt, mask)
    full = masked_psnr(pred, gt)
    assert masked > full + 5.0  # masking out the corruption raises PSNR


def test_masked_ssim_identical_near_one() -> None:
    img = torch.rand(1, 16, 16, 3)
    metric = MaskedSSIM()
    metric.update(img, img)
    assert float(metric.compute()) > 0.99


# -- schema / record ------------------------------------------------------

def test_build_record_sets_schema_and_drops_none() -> None:
    record = build_record(
        method="room4d",
        sequence="paper-windmill",
        dataset_format="iphone",
        metrics={"masked_mpsnr": 20.0, "masked_mssim": None},
        eval={"source": "room4d_render_proof", "resolution_factor": 2},
    )
    assert record["schema"] == SCHEMA
    assert record["method"] == "room4d"
    assert record["metrics"] == {"masked_mpsnr": 20.0}  # None dropped


def test_write_and_load_record_roundtrip(tmp_path) -> None:
    record = build_record(
        method="room4d", sequence="x", dataset_format="iphone", metrics={"masked_mpsnr": 19.0}
    )
    path = write_record(record, tmp_path / "r.json")
    loaded = load_record(path)
    assert loaded["metrics"]["masked_mpsnr"] == 19.0


def test_load_record_rejects_wrong_schema(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"schema": "nope", "metrics": {}}')
    with pytest.raises(ValueError):
        load_record(path)


def test_compare_exits_on_schema_mismatch(tmp_path) -> None:
    good = write_record(
        build_record(method="a", sequence="s", dataset_format="iphone", metrics={"masked_mpsnr": 1.0}),
        tmp_path / "a.json",
    )
    bad = tmp_path / "b.json"
    bad.write_text('{"schema": "nope"}')
    with pytest.raises(SystemExit):
        compare(good, bad)
