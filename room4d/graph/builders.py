"""Dynamic scene-graph builders."""

from __future__ import annotations

from pathlib import Path

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
            for track_id in grouped_track_ids:
                node_id = f"track:{stream_id}:{track_id}"
                track_obs = [obs for obs in observations if obs.track_id == track_id]
                nodes.append(
                    self.node_factory.object_node(
                        node_id,
                        label="track",
                        attrs={
                            "stream_id": stream_id,
                            "track_id": track_id,
                            "num_observations": len(track_obs),
                        },
                    )
                )
                edges.append(self.edge_factory.observed_by(node_id, stream_id, conf=1.0))

        notes = [
            "TODO: add human and generic-object nodes when perception branches are wired in.",
            "TODO: replace per-stream track nodes with cross-view associated entities.",
        ]
        if human_artifact is not None:
            notes.append("Human artifact input was provided but is not consumed in the first vertical slice.")
        if object_artifact is not None:
            notes.append("Object artifact input was provided but is not consumed in the first vertical slice.")

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
