"""Joint spatiotemporal solver scaffold."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from room4d.optimize.factors.priors import PriorFactor
from room4d.optimize.factors.reprojection import ReprojectionFactor
from room4d.optimize.factors.time_alignment import TimeAlignmentFactor
from room4d.optimize.factors.track_consistency import TrackConsistencyFactor
from room4d.optimize.optimization_artifact import OptimizationArtifact
from room4d.optimize.stages.schedule import OptimizationSchedule


class JointSpatiotemporalSolver:
    """Coordinate staged optimization across cameras, time, tracks, and actors.

    This first vertical slice does not run a numeric optimizer yet. Instead, it
    builds a stable optimization state, stage schedule, and factor diagnostics so
    the rest of the codebase can already depend on a realistic solver contract.
    """

    def __init__(self, cfg: dict | None = None) -> None:
        """Store solver configuration and stage schedule."""
        self.cfg = cfg or {}
        self.schedule = OptimizationSchedule()

    def build_state(
        self,
        sync_artifact,
        geometry_artifact,
        track_artifact,
        human_artifact=None,
        object_artifact=None,
        graph_artifact=None,
    ) -> dict[str, Any]:
        """Build an initial solver state dictionary.

        The state is intentionally plain-Python for now so it can be inspected,
        serialized, and later migrated to torch or scipy parameter containers.
        """
        return {
            "run_id": geometry_artifact.run_id,
            "stream_ids": geometry_artifact.stream_ids() or sync_artifact.stream_ids(),
            "time_offsets": dict(sync_artifact.offsets_by_stream),
            "camera_intrinsics": dict(geometry_artifact.intrinsics_by_stream),
            "camera_states": dict(geometry_artifact.camera_states_by_stream),
            "track_observations": dict(track_artifact.per_stream_observations),
            "graph_nodes": list(graph_artifact.nodes) if graph_artifact is not None else [],
            "graph_edges": list(graph_artifact.edges) if graph_artifact is not None else [],
            "human_artifact_present": human_artifact is not None,
            "object_artifact_present": object_artifact is not None,
            "stage_history": [],
            "notes": [
                "TODO: replace plain dict state with structured parameter containers.",
                "TODO: add lifted 3D track variables and robust loss bookkeeping.",
            ],
        }

    def build_default_factors(self, state: dict[str, Any], stage_cfg: dict[str, Any]) -> list[Any]:
        """Construct a minimal factor list for the requested stage."""
        factors: list[Any] = []
        factor_names = stage_cfg.get("factor_names", [])
        if "time_alignment" in factor_names:
            factors.append(TimeAlignmentFactor(offsets_by_stream=state.get("time_offsets", {}), weight=1.0))
        if "track_consistency" in factor_names:
            per_stream_counts = {
                stream_id: len(observations)
                for stream_id, observations in state.get("track_observations", {}).items()
            }
            factors.append(TrackConsistencyFactor(per_stream_counts=per_stream_counts, weight=1.0))
        if "reprojection" in factor_names:
            observations = [
                observation
                for obs_list in state.get("track_observations", {}).values()
                for observation in obs_list
            ]
            factors.append(ReprojectionFactor(observations=observations, weight=1.0))
        if "priors" in factor_names:
            num_priors = len(state.get("camera_intrinsics", {})) + len(state.get("time_offsets", {}))
            factors.append(PriorFactor(num_priors=num_priors, weight=1.0))
        return factors

    def solve_stage(self, state: dict[str, Any], factors: list[Any], stage_cfg: dict[str, Any]) -> dict[str, Any]:
        """Run one optimization stage and return updated state.

        The placeholder implementation only evaluates factor diagnostics and logs
        which parameter groups would be active in a real solve.
        """
        factor_diagnostics: list[dict[str, Any]] = []
        for factor in factors:
            residual, diagnostics = factor.residual(state)
            factor_diagnostics.append(
                {
                    "name": factor.name,
                    "residual": residual,
                    "diagnostics": diagnostics,
                }
            )

        state["stage_history"].append(
            {
                "stage_name": stage_cfg["name"],
                "active_parameter_groups": list(stage_cfg.get("active_parameter_groups", [])),
                "factor_diagnostics": factor_diagnostics,
                "todo": list(stage_cfg.get("todo", [])),
            }
        )
        return state

    def solve_all(
        self,
        sync_artifact,
        geometry_artifact,
        track_artifact,
        human_artifact=None,
        object_artifact=None,
        graph_artifact=None,
    ) -> OptimizationArtifact:
        """Run all configured stages and return an optimization artifact."""
        state = self.build_state(
            sync_artifact,
            geometry_artifact,
            track_artifact,
            human_artifact=human_artifact,
            object_artifact=object_artifact,
            graph_artifact=graph_artifact,
        )
        pending_todos: list[str] = []
        schedule_used: list[str] = []

        for stage_cfg in self.schedule.all_stages():
            factors = self.build_default_factors(state, stage_cfg)
            state = self.solve_stage(state, factors, stage_cfg)
            schedule_used.append(stage_cfg["name"])
            pending_todos.extend(stage_cfg.get("todo", []))

        return OptimizationArtifact(
            run_id=geometry_artifact.run_id,
            stage_name="optimize",
            save_dir=Path("."),
            refined_cameras=state.get("camera_states", {}),
            refined_offsets=state.get("time_offsets", {}),
            tracks_3d={},
            object_trajectories={},
            human_trajectories={},
            stage_diagnostics=list(state.get("stage_history", [])),
            schedule_used=schedule_used,
            pending_todos=pending_todos + state.get("notes", []),
        )
