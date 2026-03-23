"""Wrapper module for a CoTracker-like point tracker."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from room4d.common.types import FrameRef, TrackObservation, VideoStreamRef
from room4d.tracking.base import PointTracker
from room4d.tracking.track_artifact import TrackArtifact


class CoTrackerWrapper(PointTracker):
    """Foundation-model wrapper for point tracking.

    TODO implementation order:
    1. Load a real tracker backend and expose checkpoint/config management.
    2. Support dense-grid seeding plus externally provided seed points.
    3. Add visibility filtering and confidence calibration.
    4. Add cross-view track merging once geometry priors are stable.
    """

    def __init__(self, cfg: dict | None = None) -> None:
        """Store backend configuration."""
        self.cfg = cfg or {}

    def track(
        self,
        stream: VideoStreamRef,
        *,
        frame_refs: list[FrameRef] | None = None,
        seed_points: np.ndarray | None = None,
    ) -> TrackArtifact:
        """Track placeholder seed points through the requested frames.

        The scaffold emits deterministic synthetic observations so downstream
        graph-building and optimization code can exercise a real artifact shape.
        """
        frame_refs = frame_refs or [FrameRef(stream_id=stream.stream_id, frame_idx=0, timestamp_sec=0.0)]
        if seed_points is None:
            seed_points = np.array([[0.25, 0.25], [0.50, 0.50], [0.75, 0.75]], dtype=float)

        observations: list[TrackObservation] = []
        for track_id, xy in enumerate(seed_points):
            for frame in frame_refs:
                observations.append(
                    TrackObservation(
                        stream_id=stream.stream_id,
                        frame_idx=frame.frame_idx,
                        track_id=int(track_id),
                        xy=np.asarray(xy, dtype=float),
                        visible=True,
                        confidence=1.0,
                    )
                )

        return TrackArtifact(
            run_id="dev",
            stage_name="tracking",
            save_dir=Path("."),
            per_stream_observations={stream.stream_id: observations},
            seed_points_by_stream={stream.stream_id: np.asarray(seed_points, dtype=float).tolist()},
            merged_track_groups={},
            visibility_masks={stream.stream_id: {"status": "placeholder"}},
            static_dynamic_hints={stream.stream_id: {"all_tracks_static_prior": True}},
            notes=[
                "TODO: replace synthetic observations with real tracker outputs.",
                "TODO: add long-range visibility and occlusion reasoning.",
            ],
        )

    def track_many(
        self,
        streams: list[VideoStreamRef],
        *,
        frame_refs_by_stream: dict[str, list[FrameRef]] | None = None,
    ) -> TrackArtifact:
        """Track multiple streams and merge them into one artifact shell."""
        per_stream_observations: dict[str, list[TrackObservation]] = {}
        seed_points_by_stream: dict[str, list[list[float]]] = {}
        visibility_masks: dict[str, dict[str, str]] = {}
        static_dynamic_hints: dict[str, dict[str, bool]] = {}
        notes: list[str] = []

        for stream in streams:
            artifact = self.track(
                stream,
                frame_refs=(frame_refs_by_stream or {}).get(stream.stream_id),
            )
            per_stream_observations.update(artifact.per_stream_observations)
            seed_points_by_stream.update(artifact.seed_points_by_stream)
            visibility_masks.update(artifact.visibility_masks)
            static_dynamic_hints.update(artifact.static_dynamic_hints)
            notes.extend(artifact.notes)

        return TrackArtifact(
            run_id="dev",
            stage_name="tracking",
            save_dir=Path("."),
            per_stream_observations=per_stream_observations,
            seed_points_by_stream=seed_points_by_stream,
            merged_track_groups={},
            visibility_masks=visibility_masks,
            static_dynamic_hints=static_dynamic_hints,
            notes=notes,
        )
