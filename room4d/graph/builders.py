"""Dynamic scene-graph builders."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from room4d.common.types import GraphEdge, GraphNode
from room4d.graph.edges import EdgeFactory
from room4d.graph.graph_artifact import GraphArtifact
from room4d.graph.nodes import NodeFactory


class DynamicSceneGraphBuilder:
    """Fuse geometry and tracking into a first-pass dynamic scene graph.

    TODO implementation order:
    1. Start with camera nodes and sparse track nodes from the vertical slice.
    2. Add human and object nodes once those branches produce stable identities.
    3. Add temporal node histories and relation scoring with uncertainty.
    4. Promote track clusters into object proposals when lifting is available.
    """

    def __init__(self) -> None:
        """Initialize factories for graph records."""
        self.node_factory = NodeFactory()
        self.edge_factory = EdgeFactory()

    def build(self, geometry_artifact, track_artifact, human_artifact=None, object_artifact=None) -> GraphArtifact:
        """Create a minimal graph containing camera and sparse-track nodes."""
        nodes: list[GraphNode] = []
        edges: list[GraphEdge] = []

        for stream_id in geometry_artifact.stream_ids():
            nodes.append(
                self.node_factory.camera_node(
                    stream_id,
                    {
                        "num_camera_states": len(geometry_artifact.camera_states_by_stream.get(stream_id, [])),
                        "has_intrinsics_prior": stream_id in geometry_artifact.intrinsics_by_stream,
                    },
                )
            )

        for stream_id, observations in track_artifact.per_stream_observations.items():
            grouped_track_ids = sorted({obs.track_id for obs in observations})
            payload = track_artifact.track_payloads_by_stream.get(stream_id, {})
            track_ids = list(map(int, np.asarray(payload.get('track_ids', []), dtype=int).tolist())) if payload else []
            label_map = {}
            if payload:
                labels = np.asarray(payload.get('static_labels', np.full((len(track_ids),), -1, dtype=np.int8)), dtype=np.int8)
                scores = np.asarray(payload.get('static_scores', np.full((len(track_ids),), np.nan, dtype=np.float32)), dtype=np.float32)
                label_map = {
                    int(track_id): {
                        'static_label': int(labels[idx]) if idx < labels.shape[0] else -1,
                        'static_score': float(scores[idx]) if idx < scores.shape[0] and np.isfinite(scores[idx]) else None,
                    }
                    for idx, track_id in enumerate(track_ids)
                }
            for track_id in grouped_track_ids:
                node_id = f"track:{stream_id}:{track_id}"
                track_obs = [obs for obs in observations if obs.track_id == track_id]
                attrs = {
                    "stream_id": stream_id,
                    "track_id": track_id,
                    "num_observations": len(track_obs),
                }
                attrs.update(label_map.get(int(track_id), {}))
                nodes.append(
                    self.node_factory.object_node(
                        node_id,
                        label="track",
                        attrs=attrs,
                    )
                )
                edges.append(self.edge_factory.observed_by(node_id, stream_id, conf=1.0))

        if human_artifact is not None:
            grouped_h: dict[str, list[Any]] = {}
            for human in human_artifact.humans:
                grouped_h.setdefault(human.human_id, []).append(human)
            for human_id, instances in grouped_h.items():
                first = instances[0]
                avg_conf = float(np.mean([h.confidence for h in instances]))
                attrs = {
                    'stream_id': first.frame_ref.stream_id,
                    'num_observations': len(instances),
                    'frame_indices': sorted({int(h.frame_ref.frame_idx) for h in instances}),
                    'avg_confidence': avg_conf,
                    'support_track_ids': sorted({int(t) for h in instances for t in h.attributes.get('track_ids', [])}),
                    'mean_visible_keypoints': float(np.mean([np.sum(np.asarray(h.keypoints_2d)[:, 2] > 0.05) for h in instances if h.keypoints_2d is not None])) if any(h.keypoints_2d is not None for h in instances) else 0.0,
                    'has_densepose': any(h.densepose is not None for h in instances),
                    'has_smpl': any(bool((h.smpl_params or {}).get('valid', False)) for h in instances),
                }
                node_id = f'human:{first.frame_ref.stream_id}:{human_id}'
                nodes.append(self.node_factory.human_node(node_id, attrs))
                edges.append(self.edge_factory.observed_by(node_id, first.frame_ref.stream_id, conf=min(1.0, avg_conf)))

        if object_artifact is not None:
            grouped: dict[str, list[Any]] = {}
            for obj in object_artifact.objects:
                grouped.setdefault(obj.object_id, []).append(obj)
            for object_id, instances in grouped.items():
                first = instances[0]
                avg_conf = float(np.mean([obj.confidence for obj in instances]))
                categories = sorted({obj.category for obj in instances})
                attrs = {
                    'stream_id': first.frame_ref.stream_id,
                    'category': first.category,
                    'categories': categories,
                    'num_observations': len(instances),
                    'frame_indices': sorted({int(obj.frame_ref.frame_idx) for obj in instances}),
                    'avg_confidence': avg_conf,
                    'track_ids': sorted({int(t) for obj in instances for t in obj.attributes.get('track_ids', [])}),
                }
                node_id = f'object:{first.frame_ref.stream_id}:{object_id}'
                nodes.append(self.node_factory.object_node(node_id, label=first.category, attrs=attrs))
                edges.append(self.edge_factory.observed_by(node_id, first.frame_ref.stream_id, conf=min(1.0, avg_conf)))

        notes = [
            "TODO: replace per-stream track nodes with cross-view associated entities.",
        ]

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
            },
            notes=notes,
        )
