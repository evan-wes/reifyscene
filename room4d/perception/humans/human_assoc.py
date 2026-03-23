"""Cross-view and temporal human association."""

class HumanAssociator:
    """Associate human detections across views and time."""

    def associate_across_views(self, human_artifacts, geometry_artifact, sync_artifact):
        """Return a placeholder associated human artifact."""
        return human_artifacts[0] if human_artifacts else None
