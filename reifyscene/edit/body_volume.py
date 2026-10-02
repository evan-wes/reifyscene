"""A person's body volume per frame, so a removal takes the person and not what is behind them.

A monocular camera that only ever saw a person from the front never carved the space behind
them: FlashSplat's evidence hands that unconstrained region to the person, and removing them
takes a "shadow" two to three heights deep (measured on lab-coat, 2026-10-01). A body is not
that deep. This module bounds a person by their body.

**What the body is built from** (until a real SMPL fitter lands behind :class:`BodyVolume`):
Sapiens' Goliath keypoints, lifted into the trained scene with the scene's own rendered depth
and camera -- so the joints are in exactly the frame and units the Gaussians are, with no
alignment between the geometry and the render. A keypoint's depth is the rendered depth over
the person's own mask pixels next to it (the front surface), pushed back by that body part's
radius to its centre. The body is then capsules round the bones, radii in units of the person's
own torso length (scale-free: the scenes are not metric), plus a clothing margin.

What it cannot see it does not invent: a joint with low keypoint confidence, no mask pixels
near it, or a depth far from the torso's (an occluded joint reads the depth of whatever is in
front of it) is left out, and so is every bone through it. A frame without both shoulders and
both hips has no torso, and no body.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Goliath (Sapiens) keypoint indices of the body joints used here, from the Sapiens pose
# config (pose/configs/_base_/datasets/goliath.py).
JOINTS = {
    "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6, "left_elbow": 7, "right_elbow": 8,
    "left_hip": 9, "right_hip": 10, "left_knee": 11, "right_knee": 12,
    "left_ankle": 13, "right_ankle": 14, "left_big_toe": 15, "left_heel": 17,
    "right_big_toe": 18, "right_heel": 20, "right_wrist": 41, "left_wrist": 62, "neck": 69,
}
HEAD = ("nose", "left_eye", "right_eye", "left_ear", "right_ear")

# (from, to, radius in torso lengths). "mid_shoulder" / "mid_hip" are derived joints.
BONES = (
    ("mid_shoulder", "mid_hip", 0.35),                       # torso
    ("left_shoulder", "right_shoulder", 0.15), ("left_hip", "right_hip", 0.20),
    ("left_shoulder", "left_elbow", 0.12), ("right_shoulder", "right_elbow", 0.12),
    ("left_elbow", "left_wrist", 0.10), ("right_elbow", "right_wrist", 0.10),
    ("left_hip", "left_knee", 0.17), ("right_hip", "right_knee", 0.17),
    ("left_knee", "left_ankle", 0.12), ("right_knee", "right_ankle", 0.12),
    ("left_heel", "left_big_toe", 0.08), ("right_heel", "right_big_toe", 0.08),
    ("neck", "head", 0.20),
)
HEAD_RADIUS = 0.25                                           # a sphere at the head's centre


@dataclass
class BodyConfig:
    min_confidence: float = 0.3       # Sapiens keypoint score below which a joint is unseen
    window_px: int = 4                # mask pixels within this of a keypoint give its depth
    max_depth_from_torso: float = 0.8   # torso lengths: further and it is not this person
    margin: float = 0.15              # torso lengths added to every radius (clothing, error)


@dataclass
class BodyVolume:
    """One person's capsules per frame, in the trained scene's world frame."""

    frames: dict[int, list[tuple[np.ndarray, np.ndarray, float]]] = field(default_factory=dict)
    torso_length: dict[int, float] = field(default_factory=dict)

    def distance(self, points: np.ndarray, frame: int) -> np.ndarray:
        """Per point, how far outside the body it is on ``frame`` (<= 0: inside)."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        best = np.full(len(pts), np.inf)
        for a, b, r in self.frames[frame]:
            ab = b - a
            denom = float(ab @ ab)
            t = np.zeros(len(pts)) if denom < 1e-18 else np.clip((pts - a) @ ab / denom, 0, 1)
            d = np.linalg.norm(pts - (a + t[:, None] * ab), axis=1) - r
            best = np.minimum(best, d)
        return best


def _derived(joints: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = dict(joints)
    for name, pair in (("mid_shoulder", ("left_shoulder", "right_shoulder")),
                       ("mid_hip", ("left_hip", "right_hip"))):
        if all(p in joints for p in pair):
            out[name] = (joints[pair[0]] + joints[pair[1]]) / 2.0
    heads = [joints[h] for h in HEAD if h in joints]
    if heads:
        out["head"] = np.mean(heads, axis=0)
    return out


def _part_radius(name: str) -> float:
    """The radius of the body part a joint sits in (torso lengths): how far the joint is
    behind the surface the camera saw."""
    rs = [r for a, b, r in BONES if name in (a, b)]
    return HEAD_RADIUS if name in HEAD else (min(rs) if rs else 0.1)


def lift_frame(keypoints: np.ndarray, mask: np.ndarray, depth: np.ndarray, camera,
               transform, source_size: tuple[int, int], cfg: BodyConfig
               ) -> tuple[list[tuple[np.ndarray, np.ndarray, float]], float] | None:
    """One frame's capsules and torso length, or None when the frame shows no torso.

    ``keypoints`` (308, 3) Sapiens ``x, y, score`` in source pixels; ``mask`` (H, W) the
    person's pixels and ``depth`` (H, W) the rendered depth, both in the render raster;
    ``camera`` the frame's :class:`~reifyscene.render.gaussian4d.rasterizer.RenderCamera`;
    ``transform`` the stream's image transform (source -> render raster).
    """
    h, w = mask.shape
    kp = np.asarray(keypoints, dtype=np.float64)
    names = [n for n, i in JOINTS.items() if kp[i, 2] >= cfg.min_confidence]
    if not names:
        return None
    xy, inside = transform.map_points(kp[[JOINTS[n] for n in names], :2], from_size=source_size)
    K = np.asarray(camera.K, dtype=np.float64)
    c2w = np.linalg.inv(np.asarray(camera.viewmat, dtype=np.float64))
    surface: dict[str, tuple[np.ndarray, float]] = {}       # name -> (camera ray, depth)
    for name, (u, v), ok in zip(names, xy, inside, strict=True):
        if not ok:
            continue
        x0, x1 = max(int(u) - cfg.window_px, 0), min(int(u) + cfg.window_px + 1, w)
        y0, y1 = max(int(v) - cfg.window_px, 0), min(int(v) + cfg.window_px + 1, h)
        sel = mask[y0:y1, x0:x1]
        if not sel.any():
            continue
        d = float(np.median(depth[y0:y1, x0:x1][sel]))
        if not np.isfinite(d) or d <= 0:
            continue
        ray = np.linalg.solve(K, np.array([u, v, 1.0]))
        surface[name] = (ray, d)
    torso = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
    if not all(t in surface for t in torso):
        return None

    def world(ray, d):
        return (c2w @ np.r_[ray * d, 1.0])[:3]

    front = {n: world(*surface[n]) for n in torso}
    length = float(np.linalg.norm((front["left_shoulder"] + front["right_shoulder"]) / 2
                                  - (front["left_hip"] + front["right_hip"]) / 2))
    if length <= 0:
        return None
    torso_depth = float(np.median([surface[n][1] for n in torso]))
    joints = {}
    for name, (ray, d) in surface.items():
        if abs(d - torso_depth) > cfg.max_depth_from_torso * length:
            continue                                   # reads something in front, or behind
        joints[name] = world(ray, d + _part_radius(name) * length)   # surface -> joint centre
    joints = _derived(joints)
    capsules = [(joints[a], joints[b], (r + cfg.margin) * length)
                for a, b, r in BONES if a in joints and b in joints]
    if "head" in joints:
        capsules.append((joints["head"], joints["head"], (HEAD_RADIUS + cfg.margin) * length))
    return capsules, length


def sapiens_body(renderer, humans_artifact, track_id: str, masks_by_frame: dict,
                 node_id: str, stream, *, cfg: BodyConfig | None = None) -> BodyVolume:
    """The body of the human tracked as ``track_id``, on every frame it can be built.

    ``renderer`` a :class:`~reifyscene.edit.renderer.CheckpointRenderer` (its scene's depth
    and cameras); ``masks_by_frame`` ``{frame: {node_id: mask}}`` in the render raster
    (:func:`~reifyscene.graph.renderable_handle.render_pixel_masks`); ``stream`` the run's
    :class:`~reifyscene.ingest.streams.FrameStream`.
    """
    import torch

    cfg = cfg or BodyConfig()
    body = BodyVolume()
    states = [h for h in humans_artifact.humans
              if str(h.human_id) == str(track_id) and h.keypoints_2d is not None
              and str(h.frame_ref.stream_id) == stream.id]
    for state in states:
        frame = int(state.frame_ref.frame_idx)
        mask = masks_by_frame.get(frame, {}).get(node_id)
        if mask is None or frame not in renderer.time_of:
            continue
        t = renderer.time_of[frame]
        camera = renderer.cams[t].render_camera(*renderer.size)
        with torch.no_grad():
            c = renderer.scene.cloud_at(int(t))
            out = renderer.rasterizer.rasterize(c.means, c.quats, c.scales, c.opacities,
                                                c.colors, camera)
        lifted = lift_frame(np.asarray(state.keypoints_2d), np.asarray(mask, bool),
                            out.depth.detach().cpu().numpy(), camera, stream.transform,
                            stream.size, cfg)
        if lifted is not None:
            body.frames[frame], body.torso_length[frame] = lifted
    return body
