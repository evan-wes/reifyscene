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
from reifyscene.geometry.image_transform import MissingImageTransform, image_transform_for
from reifyscene.graph.edges import EdgeFactory
from reifyscene.graph.graph_artifact import GraphArtifact
from reifyscene.graph.nodes import NodeFactory
from reifyscene.graph.track_support import assign_track_support, positions_by_frame
from reifyscene.lifting.entity_geometry import entity_positions_from_pointmaps

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
        self, geometry_artifact, track_artifact, human_artifact=None, object_artifact=None,
        semantics_artifact=None,
    ) -> GraphArtifact:
        """Build the graph: a node per camera, person, object and room, with their evidence."""
        nodes: list[GraphNode] = []
        edges: list[GraphEdge] = []
        embeddings: dict[str, Any] = {}
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

            geometry_positions = _positions_from_geometry(
                geometry_artifact, stream_id, masks_by_frame
            )
            notes.append(_position_note(stream_id, entities[stream_id], geometry_positions))

            for entity in entities[stream_id]:
                attrs = dict(entity.attrs)
                attrs.update(evidence.for_entity(entity.node_id))
                _apply_positions(attrs, geometry_positions.get(entity.node_id))
                label, embedding = _apply_identity(attrs, entity, semantics_artifact)
                if embedding is not None:
                    embeddings[entity.node_id] = embedding
                node = (
                    self.node_factory.human_node(entity.node_id, attrs, label=label)
                    if entity.kind == "human"
                    else self.node_factory.object_node(
                        entity.node_id, label=label, attrs=attrs
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
            f"track support: inside its mask on at least {self.min_frames} frames, and on "
            f"at least {self.min_ratio:.0%} of the frames it was inside any entity"
        )
        notes.append(
            "TODO: associate entities across streams; a person seen by two cameras is two nodes."
        )

        named = sum(
            1 for n in nodes
            if n.node_type in ("human", "object") and not (n.attrs or {}).get("abstained", False)
            and (n.attrs or {}).get("semantics_source") == "semantics"
        )
        entity_count = sum(len(v) for v in entities.values())
        if semantics_artifact is not None:
            notes.append(f"semantics named {named} of {entity_count} entities")

        return GraphArtifact(
            run_id=geometry_artifact.run_id,
            stage_name="graph",
            save_dir=Path("."),
            nodes=nodes,
            edges=edges,
            embeddings=embeddings,
            node_histories={},
            uncertainty={},
            source_summary={
                "num_geometry_streams": len(geometry_artifact.stream_ids()),
                "num_track_streams": len(track_artifact.stream_ids()),
                "num_entities": entity_count,
            },
            metadata={"world_frame": _world_frame(geometry_artifact)},
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
                    # The detector's own id and the ranked alternatives `objects` carried
                    # down: dropped here for the life of the stage, because this dict was
                    # built fresh instead of extending what the instance already held.
                    "detector_backend": (instances[0].attributes or {}).get("detector_backend"),
                    "label_id": (instances[0].attributes or {}).get("label_id"),
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


def _positions_from_geometry(geometry_artifact, stream_id: str, masks_by_frame) -> dict:
    """Per-entity position and extent from the geometry's dense pointmaps.

    The geometry stage already wrote a world point for every pixel; ``segment`` already
    said which pixels are which entity. That is a far better answer than the handful of
    point tracks that happened to land on an entity -- on a clip where four of eight
    entities had no supporting track at all, this gives every one of them a position on
    every frame it is masked. See :mod:`reifyscene.lifting.entity_geometry`.

    Anything missing (no pointmaps, no recorded image transform) returns ``{}`` and the
    track-derived positions stand: empty in, empty out, never a crash.
    """
    pointmaps = getattr(geometry_artifact, "pointmaps_by_frame", None) or {}
    if not pointmaps or not masks_by_frame:
        return {}

    def load(frame_idx: int):
        payload = pointmaps.get(f"{stream_id}:{int(frame_idx)}")
        if not payload:
            return None
        points = payload.get("points")
        return None if points is None else (np.asarray(points), payload.get("confidence"))

    # The masks are in ingest pixels; their own shape is the only honest source for the
    # size the transform maps from.
    first = next(iter(masks_by_frame.values()), None)
    sample = next(iter((first or {}).values()), None)
    if sample is None:
        return {}
    height, width = np.asarray(sample).shape[:2]
    try:
        transform = image_transform_for(
            geometry_artifact, stream_id, source_size=(int(width), int(height))
        )
    except MissingImageTransform:
        return {}

    return entity_positions_from_pointmaps(
        masks_by_frame, pointmap_for_frame=load, transform=transform
    )


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
        anywhere = self.support.inside_any_frames
        track_ids = sorted(inside)
        # How decisively each track belongs to *this* entity rather than another.
        ratios = [inside[t] / anywhere[t] for t in track_ids if anywhere.get(t)]
        # And, separately, how much of the entity's own co-visible life it covers --
        # worth knowing, but not what decides whose evidence it is.
        coverage = [inside[t] / shared[t] for t in track_ids if shared.get(t)]
        return {
            "support_track_ids": track_ids,
            "support_frames": {str(t): int(inside[t]) for t in track_ids},
            "num_support_tracks": len(track_ids),
            "mean_support_ratio": float(np.mean(ratios)) if ratios else None,
            "mean_covisible_ratio": float(np.mean(coverage)) if coverage else None,
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
        "mean_covisible_ratio": None,
        "support_motion": {"static": 0, "moving": 0, "uncertain": 0, "static_fraction": None},
        "positions": [],
    }


def _world_frame(geometry_artifact) -> dict[str, Any]:
    """Whether the positions in this graph are gravity-aligned, and on which axis.

    ``above`` / ``below`` / ``on`` are defined against gravity, and the answer lives two
    stages upstream in the geometry metadata. Nothing carried it here, so a consumer
    reading only ``graph/artifact.json`` could not tell whether a missing vertical
    relation meant *false* or *not computable* -- and those are different claims.
    """
    md = getattr(geometry_artifact, "metadata", {}) or {}
    frame = dict(md.get("world_frame") or {})
    applied = bool(md.get("ground_plane_applied"))
    return {
        "gravity_aligned": applied and frame.get("type") == "ground_plane_z_up",
        "type": frame.get("type"),
        "up_axis": "+z" if frame.get("type") == "ground_plane_z_up" else None,
        "ground_plane_applied": applied,
        "inlier_ratio": frame.get("inlier_ratio"),
    }


def _apply_identity(attrs: dict[str, Any], entity, semantics_artifact) -> tuple[str, Any]:
    """Put what the `semantics` stage decided onto the node, refusal included.

    The graph never saw the semantics artifact: an object's category reached it
    second-hand through `objects`, its ranked alternatives and embedding were dropped at
    this boundary, and a human node was hardcoded ``label="human"`` with no category at
    all. Everything here already existed, correct, one stage upstream.

    An abstention travels as a value, not a gap. ``category`` is ``"unknown"`` with
    ``abstained`` and the margins beside it, so a consumer can tell a refusal from a
    question never asked -- the distinction the semantics stage exists to make.

    Returns ``(label, embedding)``: the label the node should carry, and the vector to
    store beside the graph if there is one.
    """
    record = None
    if semantics_artifact is not None:
        record = semantics_artifact.records.get(entity.node_id.split(":", 2)[-1])
    if record is None:
        # `objects` already copied a category down; keep it, and say nothing was asked.
        attrs.setdefault("category", None)
        attrs["semantics_source"] = "objects" if attrs.get("category") else "none"
        return entity.label, None

    # The separation statistics and the abstention flag arrive with the semantics
    # abstention change (PR #5), which is not on this branch yet. Read them for what they
    # are -- optional evidence a record may carry -- so the graph picks them up when the
    # stage starts producing them, rather than requiring the two changes to land together.
    abstained = bool(getattr(record, "abstained", False))
    attrs.update(
        {
            "category": record.category,
            "category_topk": [[str(lbl), float(sc)] for lbl, sc in record.category_topk],
            "category_score": float(record.score),
            "category_margin": getattr(record, "margin", None),
            "category_margin_z": getattr(record, "margin_z", None),
            "abstained": abstained,
            "abstain_reason": getattr(record, "abstain_reason", None),
            "caption": record.caption,
            "semantics_source": "semantics",
        }
    )
    embedding = None
    if semantics_artifact is not None:
        embedding = semantics_artifact.embeddings.get(entity.node_id.split(":", 2)[-1])
    label = record.category if record.category and not abstained else entity.label
    return label, embedding


def _apply_positions(attrs: dict[str, Any], geometry_positions) -> None:
    """Put the best available positions on an entity, and say which they are.

    Dense geometry wins when it is there: it covers every frame the entity is masked,
    not the handful a point track happened to reach, and it carries an extent, which a
    centroid of sparse tracks cannot. The track-derived positions stay as the fallback,
    and ``position_source`` records which one answered so a consumer weighs it rather
    than trusting it. ``extent`` is the per-frame median, the scale-free summary the
    relation vocabulary needs for above / below / in / on.
    """
    if geometry_positions:
        attrs["positions"] = geometry_positions
        attrs["position_source"] = "pointmap"
        extents = np.asarray([r["extent"] for r in geometry_positions], dtype=float)
        attrs["extent"] = [float(v) for v in np.median(extents, axis=0)]
        return
    attrs["position_source"] = "track_support" if attrs.get("positions") else "none"
    attrs["extent"] = None


def _position_note(stream_id: str, entities: list, geometry_positions) -> str:
    """How many entities dense geometry could place, against how many there are."""
    placed = sum(1 for e in entities if geometry_positions.get(e.node_id))
    return (
        f"{stream_id}: dense geometry placed {placed} of {len(entities)} entities "
        f"({'pointmaps' if geometry_positions else 'unavailable -- track support only'})"
    )


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
