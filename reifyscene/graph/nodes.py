"""Scene-graph node factories."""

from reifyscene.common.types import GraphNode


class NodeFactory:
    """Construct typed graph nodes for cameras, humans, objects, and regions."""

    def camera_node(self, stream_id: str, attrs: dict) -> GraphNode:
        """Create a camera node."""
        return GraphNode(node_id=stream_id, node_type="camera", label="camera", confidence=1.0, attrs=attrs)

    def human_node(self, human_id: str, attrs: dict, label: str = "human") -> GraphNode:
        """Create a human node.

        ``label`` is the open-vocabulary category when the `semantics` stage named this
        tracklet, and ``"human"`` otherwise. It used to be hardcoded, which meant a
        person node could not say what it had actually been recognized as -- or that it
        had not been recognized at all.
        """
        return GraphNode(node_id=human_id, node_type="human", label=label, confidence=1.0, attrs=attrs)

    def object_node(self, object_id: str, label: str, attrs: dict) -> GraphNode:
        """Create an object node."""
        return GraphNode(node_id=object_id, node_type="object", label=label, confidence=1.0, attrs=attrs)

    def region_node(self, region_id: str, label: str, attrs: dict) -> GraphNode:
        """Create a region node."""
        return GraphNode(node_id=region_id, node_type="region", label=label, confidence=1.0, attrs=attrs)
