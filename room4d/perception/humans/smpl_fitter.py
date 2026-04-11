"""SMPL fitting utilities."""

from __future__ import annotations

from typing import Any

import numpy as np


COCO_LEFT_SHOULDER = 5
COCO_RIGHT_SHOULDER = 6
COCO_LEFT_HIP = 11
COCO_RIGHT_HIP = 12
COCO_LEFT_ANKLE = 15
COCO_RIGHT_ANKLE = 16


class SMPLFitter:
    """Fit a lightweight SMPL-compatible parameter dict to image evidence.

    This implementation provides a practical single-video integration path that
    estimates weak-perspective body parameters from 2D keypoints and masks. It
    does not require licensed SMPL assets, but it returns fields that can be
    replaced later by a true ``smplx``-based optimization stage.
    """

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        self.cfg = cfg or {}
        self.backend_mode = str(self.cfg.get('smpl_backend_mode', self.cfg.get('backend_mode', 'auto')))
        self._backend_used = 'heuristic'

    @property
    def backend_used(self) -> str:
        return self._backend_used

    @staticmethod
    def _safe_point(kpts: np.ndarray, idx: int) -> np.ndarray | None:
        if idx >= kpts.shape[0] or float(kpts[idx, 2]) <= 0.05:
            return None
        return np.asarray(kpts[idx, :2], dtype=np.float32)

    def fit_frame(self, frame, keypoints_2d, mask=None, intrinsics=None):
        if keypoints_2d is None:
            self._backend_used = 'heuristic'
            return {
                'backend': 'heuristic',
                'betas': np.zeros((10,), dtype=np.float32),
                'body_pose': np.zeros((23, 3), dtype=np.float32),
                'global_orient': np.zeros((3,), dtype=np.float32),
                'transl': np.zeros((3,), dtype=np.float32),
                'scale': 1.0,
                'valid': False,
            }
        kpts = np.asarray(keypoints_2d, dtype=np.float32)
        visible = kpts[:, 2] > 0.05
        if not np.any(visible):
            return {
                'backend': 'heuristic',
                'betas': np.zeros((10,), dtype=np.float32),
                'body_pose': np.zeros((23, 3), dtype=np.float32),
                'global_orient': np.zeros((3,), dtype=np.float32),
                'transl': np.zeros((3,), dtype=np.float32),
                'scale': 1.0,
                'valid': False,
            }
        vis_pts = kpts[visible, :2]
        center = vis_pts.mean(axis=0)
        min_xy = vis_pts.min(axis=0)
        max_xy = vis_pts.max(axis=0)
        span = np.maximum(max_xy - min_xy, 1.0)
        scale = float(max(span[0], span[1]))
        ls = self._safe_point(kpts, COCO_LEFT_SHOULDER)
        rs = self._safe_point(kpts, COCO_RIGHT_SHOULDER)
        lh = self._safe_point(kpts, COCO_LEFT_HIP)
        rh = self._safe_point(kpts, COCO_RIGHT_HIP)
        la = self._safe_point(kpts, COCO_LEFT_ANKLE)
        ra = self._safe_point(kpts, COCO_RIGHT_ANKLE)
        shoulder_vec = np.asarray([1.0, 0.0], dtype=np.float32) if ls is None or rs is None else (rs - ls)
        yaw_hint = float(np.arctan2(shoulder_vec[1], shoulder_vec[0]))
        torso_mid = center if lh is None or rh is None else (lh + rh) * 0.5
        ankle_mid = torso_mid if la is None or ra is None else (la + ra) * 0.5
        body_height = float(np.linalg.norm(ankle_mid - torso_mid)) if np.any(ankle_mid != torso_mid) else float(scale)
        transl = np.asarray([center[0], center[1], max(body_height, 1.0)], dtype=np.float32)
        global_orient = np.asarray([0.0, 0.0, yaw_hint], dtype=np.float32)
        body_pose = np.zeros((23, 3), dtype=np.float32)
        if ls is not None and rs is not None and lh is not None and rh is not None:
            torso_vec = ((lh + rh) * 0.5) - ((ls + rs) * 0.5)
            pitch_hint = float(np.arctan2(torso_vec[1], max(np.linalg.norm(shoulder_vec), 1e-6)))
            body_pose[0] = np.asarray([pitch_hint, 0.0, 0.0], dtype=np.float32)
        self._backend_used = 'heuristic'
        return {
            'backend': 'heuristic',
            'betas': np.zeros((10,), dtype=np.float32),
            'body_pose': body_pose,
            'global_orient': global_orient,
            'transl': transl,
            'scale': scale,
            'visible_keypoints': int(np.sum(visible)),
            'valid': True,
        }

    def fit_sequence(self, human_track):
        return [self.fit_frame(None, hs.keypoints_2d, None, None) for hs in human_track]
