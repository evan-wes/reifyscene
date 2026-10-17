"""Scene-graph edge factories."""

from reifyscene.common.types import GraphEdge


class EdgeFactory:
    """Construct typed graph edges."""

    def same_instance(self, src: str, dst: str, conf: float, attrs: dict | None = None) -> GraphEdge:
        """Create a same-instance edge."""
        return GraphEdge(src=src, dst=dst, edge_type="same_instance", confidence=conf, attrs=attrs or {})

    def observed_by(self, src: str, dst: str, conf: float) -> GraphEdge:
        """Create an observed-by edge."""
        return GraphEdge(src=src, dst=dst, edge_type="observed_by", confidence=conf)

    def interacting_with(self, src: str, dst: str, conf: float) -> GraphEdge:
        """Create an interacting-with edge."""
        return GraphEdge(src=src, dst=dst, edge_type="interacting_with", confidence=conf)

    def on_top_of(self, src: str, dst: str, conf: float) -> GraphEdge:
        """Create an on-top-of edge."""
        return GraphEdge(src=src, dst=dst, edge_type="on_top_of", confidence=conf)
