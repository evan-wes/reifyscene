"""What every removal test of one run reads, gathered once: per training frame, the photo, the
geometry's depth, the camera, and the id map.

The depth is the geometry's own (the pose-refined geometry the scene stage was built from,
divided by the scene's normalisation scale exactly as the scene stage's depth targets are), not
a render. A render's depth changes with training, so tests built on it would move with the
removal method being judged; the geometry's does not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from reifyscene.api.run_outputs import RunOutputs
    from reifyscene.edit.renderer import CheckpointRenderer


@dataclass
class RunViews:
    """One run's training views, keyed by source frame index, in the render raster."""

    frames: list[int]                       # source frame indices with a training camera
    photos: dict[int, np.ndarray]           # uint8 (H, W, 3)
    depth: dict[int, np.ndarray]            # float32 (H, W), scene units; 0 where invalid
    K: dict[int, np.ndarray]                # (3, 3)
    w2c: dict[int, np.ndarray]              # (4, 4)
    id_maps: dict[int, np.ndarray]          # uint16 (H, W), reifyscene.graph.id_maps encoding
    class_of: dict[str, int]                # node id -> id-map class
    labels: dict[str, str]                  # node id -> label

    @property
    def shape(self) -> tuple[int, int]:
        return next(iter(self.id_maps.values())).shape

    def mask(self, node_id: str, frame: int) -> np.ndarray:
        """The entity's own pixels on ``frame``: its class in the id map, so never a pixel
        another entity or an unmapped tracklet also claims."""
        return self.id_maps[frame] == self.class_of[node_id]

    def present(self, node_id: str, *, min_px: int = 1) -> list[int]:
        """Frames with at least ``min_px`` of the entity's own pixels."""
        k = self.class_of[node_id]
        return [f for f in self.frames if int((self.id_maps[f] == k).sum()) >= min_px]

    @classmethod
    def build(cls, run: RunOutputs, renderer: CheckpointRenderer) -> RunViews:
        """The views of ``renderer``'s checkpoint of ``run``: its cameras, the run's id maps
        (exported first when the run has none) and the geometry the scene was built from."""
        from reifyscene.common.frame_keys import parse_key
        from reifyscene.graph.id_maps import export_id_maps, has_id_maps, load_id_maps
        from reifyscene.lifting.geometry_provider import ArtifactGeometryProvider

        if not has_id_maps(run):
            export_id_maps(run)
        manifest, maps = load_id_maps(run)
        class_of = {c["node_id"]: int(c["index"]) for c in manifest["classes"] if c["node_id"]}
        labels = {c["node_id"]: str(c["label"]) for c in manifest["classes"] if c["node_id"]}

        stream = run.streams.resolve(renderer.stream_id)
        stack = ArtifactGeometryProvider(run.artifact("pose_refine"),
                                         stream=stream).frame_stack("train")
        scale = float(np.asarray(renderer.bundle["scene_norm_scale"]))
        depth = stack.depths.numpy() / scale
        valid = (stack.valid.numpy() > 0.5) & np.isfinite(depth) & (depth > 0)
        index = {parse_key(str(n))[1]: i for i, n in enumerate(stack.cameras.frame_names)}

        frames = renderer.frames
        missing = [f for f in frames if f not in index or f not in maps]
        if missing:
            raise ValueError(f"{run.name}: training frames {missing[:5]} have no geometry depth "
                             "or no id map")
        size = renderer.size
        out = cls(frames=frames, photos={}, depth={}, K={}, w2c={}, id_maps={},
                  class_of=class_of, labels=labels)
        for f in frames:
            i, cam = index[f], renderer.cams[renderer.time_of[f]]
            if depth[i].shape != (size[1], size[0]):
                raise ValueError(f"{run.name}: geometry depth {depth[i].shape} is not the "
                                 f"checkpoint's raster {size}")
            out.photos[f] = stack.rgbs[i].numpy().astype(np.uint8)
            out.depth[f] = np.where(valid[i], depth[i], 0.0).astype(np.float32)
            out.K[f] = np.asarray(cam.K, np.float64)
            out.w2c[f] = np.asarray(cam.w2c, np.float64)
            out.id_maps[f] = maps[f]
        return out
