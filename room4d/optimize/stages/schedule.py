"""Optimization schedule helper."""

from __future__ import annotations


class OptimizationSchedule:
    """Expose staged optimization schedules for the first vertical slice."""

    def stage1(self) -> dict:
        """Return stage-1 configuration.

        Stage 1 should only trust sync priors, sparse tracks, and coarse camera
        initialization. Intrinsics stay mostly fixed here.
        """
        return {
            "name": "stage1_sync_pose",
            "active_parameter_groups": ["time_offsets", "camera_pose", "sparse_structure"],
            "factor_names": ["time_alignment", "track_consistency", "reprojection", "priors"],
            "todo": [
                "Connect stage variables to a real optimizer backend.",
                "Add robust loss functions and outlier filtering.",
            ],
        }

    def stage2(self) -> dict:
        """Return stage-2 configuration.

        Stage 2 is where intrinsics, distortion, and static-vs-dynamic labels get
        relaxed after stage-1 geometry is reasonably stable.
        """
        return {
            "name": "stage2_intrinsics_dynamic",
            "active_parameter_groups": ["camera_intrinsics", "distortion", "static_dynamic_labels"],
            "factor_names": ["track_consistency", "reprojection", "priors"],
            "todo": [
                "Add intrinsics/distortion parameterization.",
                "Add depth and epipolar consistency terms.",
            ],
        }

    def stage3(self) -> dict:
        """Return stage-3 configuration.

        Stage 3 is reserved for humans, objects, and richer graph constraints.
        It is included in the schedule now so the pipeline contract is stable.
        """
        return {
            "name": "stage3_humans_objects",
            "active_parameter_groups": ["human_state", "object_state", "graph_confidence"],
            "factor_names": ["reprojection", "priors"],
            "todo": [
                "Add human/body factors once that branch is wired in.",
                "Add object and scene-graph consistency factors.",
            ],
        }

    def all_stages(self) -> list[dict]:
        """Return the full ordered stage list."""
        return [self.stage1(), self.stage2(), self.stage3()]
