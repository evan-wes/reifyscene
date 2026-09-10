"""SMPL fitting wrapper (swappable seam).

Parametric body estimator. No real SMPL backend is wired yet — the world-grounded
GVHMR / smplx fitter arrives at FP5b. This wrapper honestly **skips** (returns
``None``) rather than fabricating body parameters (the prior heuristic returned
zero betas/pose with a pixel-space translation while reporting a valid fit). A real
fitter swaps in behind this seam without changing the contract.
"""

from __future__ import annotations

from typing import Any


class SMPLFitter:
    """Fit SMPL body parameters (real backend or honest skip)."""

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        self.cfg = cfg or {}
        self.backend_mode = str(self.cfg.get('smpl_backend_mode', self.cfg.get('backend_mode', 'auto')))
        self._backend_used = 'skip'

    @property
    def backend_used(self) -> str:
        return self._backend_used

    def fit_frame(self, frame, keypoints_2d, mask=None, intrinsics=None):
        # No real SMPL backend wired yet (GVHMR / smplx at FP5b) -> honest skip
        # rather than fabricating betas/pose from 2D keypoints. Swap a real fitter
        # in here behind this seam.
        self._backend_used = 'skip'
        return None

    def fit_sequence(self, human_track):
        return [None for _ in human_track]
