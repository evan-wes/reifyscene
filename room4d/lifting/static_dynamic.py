"""Static/dynamic partitioning utilities."""

from __future__ import annotations

import numpy as np


class StaticDynamicPartitioner:
    """Split lifted tracks into static, dynamic, and uncertain groups."""

    def __init__(self, static_threshold_world: float = 0.05, min_lifted_frames: int = 2) -> None:
        self.static_threshold_world = float(static_threshold_world)
        self.min_lifted_frames = int(min_lifted_frames)

    def partition(self, track_artifact, geometry_artifact=None):
        """Assign per-track static/dynamic labels based on world-space spread.

        Labels are encoded as:
        - ``1``: static
        - ``0``: dynamic
        - ``-1``: uncertain / insufficient evidence
        """
        results: dict[str, dict[str, int | float]] = {}
        for stream_id, payload in track_artifact.track_payloads_by_stream.items():
            lifted_xyz = np.asarray(payload.get('lifted_xyz', np.zeros((0, 0, 3), dtype=np.float32)), dtype=np.float32)
            lifted_valid = np.asarray(payload.get('lifted_valid', np.zeros((0, 0), dtype=bool)))
            num_tracks = lifted_xyz.shape[1] if lifted_xyz.ndim == 3 else 0
            labels = np.full((num_tracks,), -1, dtype=np.int8)
            scores = np.full((num_tracks,), np.nan, dtype=np.float32)
            for track_idx in range(num_tracks):
                valid = lifted_valid[:, track_idx].astype(bool) if lifted_valid.size else np.isfinite(lifted_xyz[:, track_idx]).all(axis=1)
                xyz = lifted_xyz[valid, track_idx]
                if xyz.shape[0] < self.min_lifted_frames:
                    continue
                center = np.nanmedian(xyz, axis=0)
                spread = np.linalg.norm(xyz - center[None, :], axis=1)
                score = float(np.nanmedian(spread))
                scores[track_idx] = score
                labels[track_idx] = 1 if score <= self.static_threshold_world else 0
            payload['static_labels'] = labels
            payload['static_scores'] = scores
            static_count = int(np.sum(labels == 1))
            dynamic_count = int(np.sum(labels == 0))
            uncertain_count = int(np.sum(labels < 0))
            track_artifact.static_dynamic_hints[stream_id] = {
                'static_count': static_count,
                'dynamic_count': dynamic_count,
                'uncertain_count': uncertain_count,
                'static_threshold_world': self.static_threshold_world,
                'min_lifted_frames': self.min_lifted_frames,
            }
            results[stream_id] = {
                'static_count': static_count,
                'dynamic_count': dynamic_count,
                'uncertain_count': uncertain_count,
                'static_threshold_world': self.static_threshold_world,
            }
        track_artifact.notes.append('Computed static/dynamic labels from lifted 3D track consistency.')
        return results
