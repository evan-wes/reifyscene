"""Base renderer interface."""

class DynamicRenderer:
    """Base class for a dynamic scene renderer."""

    def fit(self, optimization_artifact):
        """Fit a renderable scene representation."""
        raise NotImplementedError

    def render(self, scene, camera_path):
        """Render frames from a camera path."""
        raise NotImplementedError
