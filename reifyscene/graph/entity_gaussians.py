"""Which Gaussians belong to which entity: the graph's handle on what renders.

Editing a scene through its graph -- "remove that person", "move that chair" -- needs to
know which Gaussians are that person. Nothing in the pipeline computed that, and the
architecture design flags it as a first-class risk (the grounding problem).

It turns out to need no new idea. The ``graph`` stage already decides which point tracks
are evidence for an entity: project each track into every frame, test it against the
entity's mask, and keep the entity it is inside most often
(:mod:`reifyscene.graph.track_support`). Gaussians are just a different population to run
that same rule over -- project them into each training view and ask the same question.

Two things have to line up, and both are why this lives here rather than in a caller:

- **The trained Gaussians, not the seed.** Training densifies and prunes, so the cloud
  that renders is not the one the scene stage seeded; an index into the seed means
  nothing at render time. Callers pass the positions of the cloud they intend to edit.
- **The render's pixels, not the ingest frame's.** A trained scene carries its own
  cameras, and they project into the geometry's pixel space (518x518 for VGGT). Entity
  masks are in ingest-frame pixels, so a caller maps them through the geometry's recorded
  :class:`~reifyscene.geometry.image_transform.ImageTransform` first -- the same call the
  scene stage uses for its own masks.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from reifyscene.graph.track_support import TrackSupport, assign_track_support


def project_to_views(
    positions: np.ndarray, w2c: np.ndarray, intrinsics: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Project per-frame 3D positions into their views.

    ``positions`` is (T, G, 3) in the cameras' own frame -- for a dynamic scene the
    foreground moves, so each frame carries its own positions. ``w2c`` is (T, 4, 4) and
    ``intrinsics`` (T, 3, 3). Returns the pixel positions (T, G, 2) and which of them are
    in front of the camera (T, G); a point behind the camera is not an observation.
    """
    pos = np.asarray(positions, dtype=np.float64)
    w2c = np.asarray(w2c, dtype=np.float64)
    K = np.asarray(intrinsics, dtype=np.float64)
    if pos.ndim != 3 or pos.shape[-1] != 3:
        raise ValueError(f"positions must be (T, G, 3); got {pos.shape}")
    if w2c.shape[0] != pos.shape[0] or K.shape[0] != pos.shape[0]:
        raise ValueError(
            f"{pos.shape[0]} frames of positions but {w2c.shape[0]} poses "
            f"and {K.shape[0]} intrinsics"
        )

    cam = np.einsum("tij,tgj->tgi", w2c[:, :3, :3], pos) + w2c[:, None, :3, 3]
    z = cam[..., 2]
    in_front = z > 1e-6
    uv = cam[..., :2] / np.where(in_front, z, 1.0)[..., None]
    xy = np.einsum("tij,tgj->tgi", K[:, :2, :2], uv) + K[:, None, :2, 2]
    return xy, in_front


def assign_gaussians_to_entities(
    masks_by_frame: Mapping[int, Mapping[str, np.ndarray]],
    *,
    frame_indices: Sequence[int],
    positions: np.ndarray,
    w2c: np.ndarray,
    intrinsics: np.ndarray,
    min_ratio: float = 0.5,
    min_frames: int = 2,
) -> TrackSupport:
    """Assign each Gaussian to the entity whose mask it is inside most often.

    ``masks_by_frame`` maps a frame index to that frame's entity masks **in the render's
    pixel space**, keyed by graph node id. The result is the same
    :class:`~reifyscene.graph.track_support.TrackSupport` the point-track assignment
    produces: ``track_ids_for(node_id)`` gives that entity's Gaussian indices, and the
    free / ambiguous lists say which Gaussians no entity earned.
    """
    xy, in_front = project_to_views(positions, w2c, intrinsics)
    return assign_track_support(
        masks_by_frame,
        frame_indices=np.asarray(frame_indices, dtype=int),
        points_xy=xy,
        visibility=in_front,
        track_ids=np.arange(xy.shape[1], dtype=int),
        min_ratio=min_ratio,
        min_frames=min_frames,
    )
