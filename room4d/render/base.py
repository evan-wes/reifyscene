"""Base renderer interface."""

class DynamicRenderer:
    """Base class for a dynamic scene renderer."""

    def fit(self, scene_artifact, optimization_artifact=None):
        """Fit or derive a renderable scene representation from a scene artifact."""
        raise NotImplementedError

    def render(self, scene, camera_path):
        """Render frames from a camera path."""
        raise NotImplementedError
