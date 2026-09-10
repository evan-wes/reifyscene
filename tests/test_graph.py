"""Basic tests for graph scaffolding."""

from reifyscene.graph.nodes import NodeFactory


def test_node_factory_camera() -> None:
    node = NodeFactory().camera_node("cam0", {})
    assert node.node_id == "cam0"
