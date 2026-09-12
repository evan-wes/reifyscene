"""Dynamic scene-graph builders.

The graph holds the things that are really there -- the people and objects the
perception stages found, the room they are in, and the cameras that saw them. Point
tracks are not things; they are the evidence for them. So every entity node carries
the tracks that support it (:mod:`reifyscene.graph.track_support`), how many frames
each was inside it, whether those tracks were moving, and where the entity was over
time -- and the tracks themselves stay in the tracking artifact.

TODO implementation order:
1. Associate entities across streams: one person seen by two cameras is two nodes.
2. Add relation edges (holds, sits-on) once the entities carry 3D extent.
3. Add temporal node histories and relation scoring with uncertainty.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from reifyscene.common.types import GraphEdge, GraphNode
from reifyscene.graph.edges import EdgeFactory
from reifyscene.graph.graph_artifact import GraphArtifact
from reifyscene.graph.nodes import NodeFactory
from reifyscene.graph.track_support import assign_track_support, positions_by_frame

DEFAULT_MIN_RATIO = 0.5
DEFAULT_MIN_FRAMES = 2


class DynamicSceneGraphBuilder:
    """Fuse geometry, tracking and the perception stages into a dynamic scene graph."""

    def __init__(self, cfg: Mapping[str, Any] | None = None) -> None:
        """Read the track-support thresholds and initialize the record factories."""
        cfg = dict(cfg or {})
        self.min_ratio = float(cfg.get("track_membership_min_ratio", DEFAULT_MIN_RATIO))
        self.min_frames = int(cfg.get("track_membership_min_frames", DEFAULT_MIN_FRAMES))
        self.node_factory = NodeFactory()
        self.edge_factory = EdgeFactory()

    def build(
        self, geometry_artifact, track_artifact, human_artifact=None, object_artifact=None
    ) -> GraphArtifact:
        """Build the graph: a node per camera, person, object and room, with their evidence."""
        nodes: list[GraphNode] = []
        edges: list[GraphEdge] = []
        notes: list[str] = []

        for stream_id in geometry_artifact.stream_ids():
            nodes.append(
                self.node_factory.camera_node(
                    stream_id,
                    {
                        "num_camera_states": len(
                            geometry_artifact.camera_states_by_stream.get(stream_id, [])
                        ),
                        "has_intrinsics_prior": stream_id in geometry_artifact.intrinsics_by_stream,
                    },
                )
            )

        entities = _entities_by_stream(human_artifact, object_artifact)
        for stream_id in sorted(entities):
            payload = track_artifact.track_payloads_by_stream.get(stream_id, {})
            masks_by_frame, mask_frames = _masks_by_frame(entities[stream_id])
            support, evidence = _support_for_stream(
                payload, masks_by_frame, self.min_ratio, self.min_frames
            )
            notes.append(_coverage_note(stream_id, payload, mask_frames, support))

            for entity in entities[stream_id]:
                attrs = dict(entity.attrs)
                attrs.update(evidence.for_entity(entity.node_id))
                node = (
                    self.node_factory.human_node(entity.node_id, attrs)
                    if entity.kind == "human"
                    else self.node_factory.object_node(
                        entity.node_id, label=entity.label, attrs=attrs
                    )
                )
                nodes.append(node)
                edges.append(
                    self.edge_factory.observed_by(entity.node_id, stream_id, conf=entity.confidence)
                )

            if payload:
                background_id = f"background:{stream_id}"
                nodes.append(
                    self.node_factory.region_node(
                        background_id,
                        label="background",
                        attrs={
                            "stream_id": stream_id,
                            "num_entities": len(entities[stream_id]),
                            **evidence.for_background(),
                        },
                    )
                )
                edges.append(self.edge_factory.observed_by(background_id, stream_id, conf=1.0))

        notes.append(
            f"track support: inside its mask on at least {self.min_frames} frames and "
            f"at least {self.min_ratio:.0%} of the frames the two share"
        )
        notes.append(
            "TODO: associate entities across streams; a person seen by two cameras is two nodes."
        )

        return GraphArtifact(
            run_id=geometry_artifact.run_id,
            stage_name="graph",
            save_dir=Path("."),
            nodes=nodes,
            edges=edges,
            node_histories={},
            uncertainty={},
            source_summary={
                "num_geometry_streams": len(geometry_artifact.stream_ids()),
                "num_track_streams": len(track_artifact.stream_ids()),
                "num_entities": sum(len(v) for v in entities.values()),
            },
            notes=notes,
        )


class _Entity:
    """One real thing the perception stages found, before its evidence is attached."""

    def __init__(
        self, node_id: str, kind: str, label: str, confidence: float, attrs: dict[str, Any]
    ) -> None:
        self.node_id = node_id
        self.kind = kind
        self.label = label
        self.confidence = confidence
        self.attrs = attrs
        self.masks: dict[int, np.ndarray] = {}


def _entities_by_stream(human_artifact, object_artifact) -> dict[str, list[_Entity]]:
    """Group the human and object instances into one entity per identity, per stream."""
    by_stream: dict[str, list[_Entity]] = {}
    if human_artifact is not None:
        grouped: dict[tuple[str, str], list[Any]] = {}
        for human in human_artifact.humans:
            grouped.setdefault((human.frame_ref.stream_id, human.human_id), []).append(human)
        for (stream_id, human_id), instances in grouped.items():
            avg_conf = float(np.mean([h.confidence for h in instances]))
            keypoints = [h for h in instances if h.keypoints_2d is not None]
            entity = _Entity(
                node_id=f"human:{stream_id}:{human_id}",
                kind="human",
                label="human",
                confidence=min(1.0, avg_conf),
                attrs={
                    "stream_id": stream_id,
                    "num_observations": len(instances),
                    "frame_indices": sorted({int(h.frame_ref.frame_idx) for h in instances}),
                    "avg_confidence": avg_conf,
                    "mean_visible_keypoints": (
                        float(
                            np.mean(
                                [np.sum(np.asarray(h.keypoints_2d)[:, 2] > 0.05) for h in keypoints]
                            )
                        )
                        if keypoints
                        else 0.0
                    ),
                    "has_densepose": any(h.densepose is not None for h in instances),
                    "has_smpl": any(
                        bool((h.smpl_params or {}).get("valid", False)) for h in instances
                    ),
                },
            )
            entity.masks = _instance_masks(instances, human_artifact, "human_id")
            by_stream.setdefault(stream_id, []).append(entity)

    if object_artifact is not None:
        grouped_o: dict[tuple[str, str], list[Any]] = {}
        for obj in object_artifact.objects:
            grouped_o.setdefault((obj.frame_ref.stream_id, obj.object_id), []).append(obj)
        for (stream_id, object_id), instances in grouped_o.items():
            avg_conf = float(np.mean([o.confidence for o in instances]))
            entity = _Entity(
                node_id=f"object:{stream_id}:{object_id}",
                kind="object",
                label=instances[0].category,
                confidence=min(1.0, avg_conf),
                attrs={
                    "stream_id": stream_id,
                    "category": instances[0].category,
                    "categories": sorted({o.category for o in instances}),
                    "num_observations": len(instances),
                    "frame_indices": sorted({int(o.frame_ref.frame_idx) for o in instances}),
                    "avg_confidence": avg_conf,
                },
            )
            entity.masks = _instance_masks(instances, object_artifact, "object_id")
            by_stream.setdefault(stream_id, []).append(entity)

    for stream_id in by_stream:
        by_stream[stream_id].sort(key=lambda e: e.node_id)
    return by_stream


def _instance_masks(instances, artifact, id_field: str) -> dict[int, np.ndarray]:
    """An entity's mask per frame, from the stage's own copy of the segment masks.

    The masks are keyed by the segment stage's ``detection_id``; where an identity has
    more than one mask on a frame, they are unioned -- it is one thing either way.
    """
    masks: dict[int, np.ndarray] = {}
    for record in instances:
        det_id = record.attributes.get("detection_id", getattr(record, id_field))
        mask = artifact.mask_arrays.get(det_id)
        if mask is None:
            continue
        frame_idx = int(record.frame_ref.frame_idx)
        mask = np.asarray(mask, dtype=bool)
        masks[frame_idx] = mask if frame_idx not in masks else (masks[frame_idx] | mask)
    return masks


def _masks_by_frame(entities: list[_Entity]) -> tuple[dict[int, dict[str, np.ndarray]], set[int]]:
    """Invert the per-entity masks into per-frame masks, and the frames they cover."""
    by_frame: dict[int, dict[str, np.ndarray]] = {}
    for entity in entities:
        for frame_idx, mask in entity.masks.items():
            by_frame.setdefault(frame_idx, {})[entity.node_id] = mask
    return by_frame, set(by_frame)


def _support_for_stream(payload, masks_by_frame, min_ratio: float, min_frames: int):
    """Assign the stream's tracks to its entities and summarize the evidence."""
    if not payload or not masks_by_frame:
        return None, _Evidence(None, payload or {})
    support = assign_track_support(
        masks_by_frame,
        frame_indices=np.asarray(payload.get("frame_indices", []), dtype=int),
        points_xy=np.asarray(payload.get("points_xy", np.zeros((0, 0, 2), dtype=np.float32))),
        visibility=payload.get("visibility"),
        track_ids=np.asarray(payload.get("track_ids", []), dtype=int),
        min_ratio=min_ratio,
        min_frames=min_frames,
    )
    return support, _Evidence(support, payload)


class _Evidence:
    """Turns a stream's track support into the attributes an entity node carries."""

    def __init__(self, support, payload: Mapping[str, Any]) -> None:
        self.support = support
        self.track_ids = np.asarray(payload.get("track_ids", []), dtype=int).reshape(-1)
        self.column_of_track = {int(t): i for i, t in enumerate(self.track_ids)}
        self.static_labels = np.asarray(payload.get("static_labels", []), dtype=np.int8).reshape(-1)
        self.lifted_xyz = np.asarray(
            payload.get("lifted_xyz", np.zeros((0, 0, 3), dtype=np.float32))
        )
        self.lifted_valid = np.asarray(payload.get("lifted_valid", np.zeros((0, 0), dtype=bool)))
        self.lifted_frames = np.asarray(
            payload.get("lifted_frame_indices", []), dtype=int
        ).reshape(-1)

    def for_entity(self, node_id: str) -> dict[str, Any]:
        """The evidence block for one entity node."""
        if self.support is None:
            return _empty_evidence()
        inside = self.support.inside_frames.get(node_id, {})
        shared = self.support.shared_frames.get(node_id, {})
        track_ids = sorted(inside)
        ratios = [inside[t] / shared[t] for t in track_ids if shared.get(t)]
        return {
            "support_track_ids": track_ids,
            "support_frames": {str(t): int(inside[t]) for t in track_ids},
            "num_support_tracks": len(track_ids),
            "mean_support_ratio": float(np.mean(ratios)) if ratios else None,
            "support_motion": self._motion(track_ids),
            "positions": self._positions(track_ids),
        }

    def for_background(self) -> dict[str, Any]:
        """The evidence block for the background: the tracks no entity accounts for."""
        if self.support is None:
            return {**_empty_evidence(), "ambiguous_track_ids": [], "num_ambiguous_tracks": 0}
        free = list(self.support.free_track_ids)
        return {
            "support_track_ids": free,
            "num_support_tracks": len(free),
            "support_motion": self._motion(free),
            # Inside some entity sometimes, but never enough to be its evidence. Named
            # rather than quietly swept into the room.
            "ambiguous_track_ids": list(self.support.ambiguous_track_ids),
            "num_ambiguous_tracks": len(self.support.ambiguous_track_ids),
        }

    def _columns(self, track_ids: list[int]) -> np.ndarray:
        return np.asarray(
            [self.column_of_track[t] for t in track_ids if t in self.column_of_track], dtype=int
        )

    def _motion(self, track_ids: list[int]) -> dict[str, Any]:
        """How the tracking stage labelled the supporting tracks: 1 static, 0 moving, -1 unsure."""
        columns = self._columns(track_ids)
        if columns.size == 0 or self.static_labels.size == 0:
            return {"static": 0, "moving": 0, "uncertain": len(track_ids), "static_fraction": None}
        labels = self.static_labels[columns[columns < self.static_labels.shape[0]]]
        static = int(np.sum(labels > 0))
        moving = int(np.sum(labels == 0))
        decided = static + moving
        return {
            "static": static,
            "moving": moving,
            "uncertain": int(len(track_ids) - decided),
            "static_fraction": (static / decided) if decided else None,
        }

    def _positions(self, track_ids: list[int]) -> list[dict[str, Any]]:
        columns = self._columns(track_ids)
        if columns.size == 0 or self.lifted_xyz.size == 0:
            return []
        return positions_by_frame(
            columns,
            lifted_frame_indices=self.lifted_frames,
            lifted_xyz=self.lifted_xyz,
            lifted_valid=self.lifted_valid,
        )


def _empty_evidence() -> dict[str, Any]:
    """What an entity carries when there is nothing to support it with."""
    return {
        "support_track_ids": [],
        "support_frames": {},
        "num_support_tracks": 0,
        "mean_support_ratio": None,
        "support_motion": {"static": 0, "moving": 0, "uncertain": 0, "static_fraction": None},
        "positions": [],
    }


def _coverage_note(stream_id: str, payload, mask_frames: set[int], support) -> str:
    """Say what the support could be computed on: silence here would hide an empty join."""
    frame_indices = (
        np.asarray(payload.get("frame_indices", []), dtype=int).reshape(-1) if payload else []
    )
    tracked = {int(f) for f in frame_indices}
    shared = sorted(tracked & mask_frames)
    if not payload:
        return f"{stream_id}: no tracks; entities carry no point-track evidence"
    if not mask_frames:
        return f"{stream_id}: no entity masks; entities carry no point-track evidence"
    if not shared:
        return (
            f"{stream_id}: the tracking frames and the masked frames do not overlap "
            f"({len(tracked)} tracked, {len(mask_frames)} masked); entities carry no "
            "point-track evidence"
        )
    supported = len(support.entity_of_track) if support is not None else 0
    num_tracks = len(np.asarray(payload.get("track_ids", []), dtype=int).reshape(-1))
    return (
        f"{stream_id}: {supported} of {num_tracks} tracks support an entity "
        f"on {len(shared)} shared frames"
    )
