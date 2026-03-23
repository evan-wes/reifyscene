"""Basic tests for optimizer scaffolding."""

from room4d.optimize.factors.reprojection import ReprojectionFactor


def test_reprojection_factor_placeholder() -> None:
    residual, diag = ReprojectionFactor().residual({})
    assert residual == 0.0
    assert isinstance(diag, dict)
