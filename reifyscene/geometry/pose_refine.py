"""Pose refinement via COLMAP bundle adjustment seeded with backend geometry.

Backend-agnostic: consumes a :class:`GeometryArtifact` (poses + intrinsics +
per-pixel world points) plus the source frames, runs VGGSfM track prediction,
seeds a pycolmap reconstruction with the artifact's poses / K / world points and
the static-structure tracks, then runs iterative robust bundle adjustment
(refine the shared FOCAL length, principal point fixed). It returns refined
per-keyframe world->camera poses and a refined shared intrinsic matrix per
stream; it does not re-run any geometry backend.

Why this closes the native-geometry pose gap when photometric pose-opt cannot:
BA minimizes *reprojection* error of feature correspondences — a signal tied to
the true scene independent of the current geometry — so it breaks the
pose<->geometry gauge ambiguity that leaves photometric pose-opt stuck at the
seed poses.

Validated in ``spike_step12_vggt`` (Path B): paper-windmill pairwise rotation
1.73deg -> 0.26deg, masked mPSNR 17.01 -> 19.58 (~69% of the gap to the GT-pose
oracle). Robust BA (Cauchy loss + reprojection-outlier filtering) rejects
dynamic-foreground tracks on its own at typical dynamic fractions, so no
foreground mask is required here — the spike ablation found masking is not
load-bearing (none 1.078deg ~= image-mask 1.062deg ~= motion-mask 1.098deg).

Heavy dependencies (``torch``, ``pycolmap``, the vendored VGGSfM tracker) are
imported lazily inside :meth:`PoseRefiner.refine` so this module stays importable
on CPU-only / heuristic environments; callers should treat a
:class:`PoseRefineUnavailable` error as "fall back to the un-refined geometry".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from reifyscene.common.types import CameraIntrinsics, FrameRef
from reifyscene.geometry.geometry_artifact import GeometryArtifact


class PoseRefineUnavailable(RuntimeError):
    """Raised when the refinement dependencies/runtime are not available.

    The caller is expected to catch this and emit the un-refined geometry
    unchanged (identity pass-through), the same way the geometry backend falls
    back to a heuristic when VGGT is unavailable.
    """


@dataclass(frozen=True)
class PoseRefineConfig:
    """Bundle-adjustment + tracker settings for :class:`PoseRefiner`.

    Defaults match the validated Path-B spike configuration (refine shared focal,
    principal point fixed, robust iterative BA).
    """

    device: str = "cuda"
    # VGGSfM tracker
    vis_thresh: float = 0.2
    max_query_pts: int = 2048
    query_frame_num: int = 6
    max_points_num: int = 20480
    fine_tracking: bool = False
    # Seed reprojection-outlier filter (against the seed poses), in px
    max_reproj_error: float = 8.0
    # Iterative robust BA
    ba_iters: int = 3
    ba_max_err: float = 4.0
    ba_loss_scale: float = 1.0
    refine_focal: bool = True
    refine_pp: bool = False
    min_reg_frames: int = 2
    min_points: int = 2

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "PoseRefineConfig":
        raw = dict(raw or {})
        fields = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in raw.items() if k in fields})


@dataclass
class StreamRefinement:
    """Per-stream refinement result applied by the pipeline."""

    stream_id: str
    refined_T_world_cam_by_frame_key: dict[str, np.ndarray]
    refined_intrinsics: CameraIntrinsics
    stats: dict[str, Any]


# --------------------------------------------------------------------------- #
# pycolmap seeding + iterative robust BA (ported from the validated spike).
# Generalized from the spike's hard-coded 518^2 crop to the artifact's actual
# per-stream (H, W); static-mask post-filtering dropped (not load-bearing).
# --------------------------------------------------------------------------- #
def _seed_reconstruction_via_text(
    names, w2c, K, tracks, obs_mask, pts3d, rgb, *, width, height, max_reproj_error, out_dir, shared_camera
):
    """Write a classic COLMAP text model seeded with the backend poses / K /
    world points plus the visible tracks, then read it back (pycolmap's text
    reader builds the internal rig/frame structure for us).

    Observations are reprojection-filtered against the SEED poses (drop outliers
    and residual dynamics); a track is kept only with >=2 surviving observations.
    Returns ``(reconstruction, num_points, num_obs)``.
    """
    import pycolmap
    from scipy.spatial.transform import Rotation

    N = len(names)
    R, t = w2c[:, :3, :3], w2c[:, :3, 3]
    Pc = np.einsum("nij,pj->npi", R, pts3d) + t[:, None, :]  # (N,P,3) camera coords
    z = Pc[..., 2]
    proj = np.stack(
        [K[0, 0] * Pc[..., 0] / z + K[0, 2], K[1, 1] * Pc[..., 1] / z + K[1, 2]], axis=-1
    )
    reproj_err = np.linalg.norm(proj - tracks, axis=-1)
    valid = obs_mask & (z > 1e-3) & (reproj_err < max_reproj_error)
    pidx = np.nonzero(valid.sum(0) >= 2)[0]

    img_obs: dict[int, list] = {f: [] for f in range(N)}
    point_track: dict[int, tuple] = {}
    n_obs = 0
    for new_pid, p in enumerate(pidx, start=1):
        trk = []
        for f in range(N):
            if valid[f, p]:
                idx = len(img_obs[f])
                img_obs[f].append((float(tracks[f, p, 0]), float(tracks[f, p, 1]), new_pid))
                trk.append((f + 1, idx))
                n_obs += 1
        point_track[new_pid] = (int(p), trk)

    d = out_dir / "colmap_seed"
    d.mkdir(parents=True, exist_ok=True)
    # shared_camera: ONE camera id=1 for all frames (correct for a single physical
    # camera when refining intrinsics); else one camera per image (fixed-K case).
    with open(d / "cameras.txt", "w") as f:
        cam_lines = 1 if shared_camera else N
        for ci in range(cam_lines):
            f.write(f"{ci+1} PINHOLE {width} {height} {K[0,0]} {K[1,1]} {K[0,2]} {K[1,2]}\n")
    with open(d / "images.txt", "w") as f:
        for fi in range(N):
            q = Rotation.from_matrix(w2c[fi][:3, :3]).as_quat()[[3, 0, 1, 2]]
            tw = w2c[fi][:3, 3]
            cam_id = 1 if shared_camera else fi + 1
            f.write(
                f"{fi+1} {q[0]} {q[1]} {q[2]} {q[3]} {tw[0]} {tw[1]} {tw[2]} {cam_id} {names[fi]}\n"
            )
            f.write(" ".join(f"{ox} {oy} {pid}" for ox, oy, pid in img_obs[fi]) + "\n")
    with open(d / "points3D.txt", "w") as f:
        for new_pid, (p, trk) in point_track.items():
            X = pts3d[p]
            c = rgb[p] if rgb is not None else (128, 128, 128)
            trks = " ".join(f"{im} {ix}" for im, ix in trk)
            f.write(
                f"{new_pid} {X[0]} {X[1]} {X[2]} {int(c[0])} {int(c[1])} {int(c[2])} 1.0 {trks}\n"
            )

    rec = pycolmap.Reconstruction()
    rec.read_text(str(d))
    return rec, len(pidx), n_obs


def _ba_options(loss_scale, *, refine_focal, refine_pp):
    import pycolmap

    opts = pycolmap.BundleAdjustmentOptions()
    # Refining FOCAL helps poses (removes focal->pose aliasing). The PRINCIPAL
    # POINT is weakly constrained in a forward monocular sweep and drifts (~9px),
    # wrecking render registration -> keep pp fixed by default.
    if hasattr(opts, "refine_focal_length"):
        opts.refine_focal_length = bool(refine_focal)
    if hasattr(opts, "refine_principal_point"):
        opts.refine_principal_point = bool(refine_pp)
    if hasattr(opts, "refine_extra_params"):
        opts.refine_extra_params = False  # PINHOLE has no extra params
    # Robust loss: the default is TRIVIAL (L2), under which outliers drag points
    # to infinity; Cauchy keeps the solve stable on dynamic / mismatched tracks.
    opts.loss_function_type = pycolmap.LossFunctionType.CAUCHY
    opts.loss_function_scale = float(loss_scale)
    return opts


def _run_iterative_ba(rec, *, n_iters, max_err, loss_scale, refine_focal, refine_pp, log):
    """COLMAP's standard recipe: robust BA -> drop non-finite / high-reproj
    points -> re-BA, then a final polish on the cleaned set. Fixes the single-
    pass divergence (points running off to infinity under the L2 loss)."""
    import pycolmap

    opts = _ba_options(loss_scale, refine_focal=refine_focal, refine_pp=refine_pp)
    for it in range(n_iters):
        pycolmap.bundle_adjustment(rec, opts)
        rec.update_point_3d_errors()
        before = rec.num_points3D()
        bad = [
            pid
            for pid in rec.point3D_ids()
            if not np.isfinite(rec.point3D(pid).error) or rec.point3D(pid).error > max_err
        ]
        for pid in bad:
            rec.delete_point3D(pid)
        try:
            mre = rec.compute_mean_reprojection_error()
        except Exception:
            mre = float("nan")
        log(f"BA iter {it+1}: reproj={mre:.3f}px, points {before}->{rec.num_points3D()} (dropped {len(bad)})")
    pycolmap.bundle_adjustment(rec, opts)  # final polish on the clean set
    log(
        f"BA final: reproj={rec.compute_mean_reprojection_error():.3f}px, "
        f"{rec.num_points3D()} pts, {rec.num_reg_frames()} reg frames"
    )
    return rec


def _cam_from_world_matrix(image) -> np.ndarray:
    cfw = image.cam_from_world
    cfw = cfw() if callable(cfw) else cfw  # pycolmap 3.13: method; older: attribute
    T = np.eye(4)
    T[:3, :4] = cfw.matrix()
    return T


def _center_from_w2c(w2c: np.ndarray) -> np.ndarray:
    return -w2c[:3, :3].T @ w2c[:3, 3]


def _sim3_src_to_dst(src_w2c: dict[str, np.ndarray], dst_w2c: dict[str, np.ndarray], keys: list[str]):
    """Orientation-based Sim(3) mapping the ``src`` world frame onto ``dst``.

    Rotation is estimated from camera ORIENTATIONS (well conditioned for a
    looking-around rig), not the camera-center cloud, which is near-degenerate
    for planar/linear sweeps and yields a spurious orientation error. Scale +
    translation then fit the centers under that rotation. Returns ``(s, R, t)``
    with ``x_dst = s * R @ x_src + t``.
    """
    G = [src_w2c[k][:3, :3].T @ dst_w2c[k][:3, :3] for k in keys]
    U, _, Vt = np.linalg.svd(np.sum(G, axis=0))
    Gavg = U @ Vt
    if np.linalg.det(Gavg) < 0:
        U[:, -1] *= -1
        Gavg = U @ Vt
    R = Gavg.T
    sc = np.stack([_center_from_w2c(src_w2c[k]) for k in keys])
    dc = np.stack([_center_from_w2c(dst_w2c[k]) for k in keys])
    smu, dmu = sc.mean(0), dc.mean(0)
    sch, dch = sc - smu, dc - dmu
    s = float(np.sum(dch * (R @ sch.T).T) / np.sum(sch ** 2))
    t = dmu - s * (R @ smu)
    return s, R, t


def _apply_sim3_to_c2w(c2w: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Map a camera-to-world pose through the world similarity ``x' = s*R@x + t``.

    The scale is dropped from the rotation block so it stays orthonormal; only
    the camera center (a world point) carries the scale + translation.
    """
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R @ c2w[:3, :3]
    out[:3, 3] = s * (R @ c2w[:3, 3]) + t
    return out


# --------------------------------------------------------------------------- #
# Refiner.
# --------------------------------------------------------------------------- #
class PoseRefiner:
    """BA-refine the camera poses of a :class:`GeometryArtifact`, per stream."""

    def __init__(self, cfg: dict[str, Any] | PoseRefineConfig | None = None) -> None:
        self.cfg = cfg if isinstance(cfg, PoseRefineConfig) else PoseRefineConfig.from_dict(cfg)
        self.notes: list[str] = []

    # -- public API -------------------------------------------------------- #
    def refine(self, geometry: GeometryArtifact, *, decoder, ingest_artifact) -> dict[str, StreamRefinement]:
        """Return a per-stream refinement for every stream that refines cleanly.

        Streams that are too sparse to bundle-adjust are omitted (the caller
        keeps their original poses). Raises :class:`PoseRefineUnavailable` if the
        heavy dependencies / runtime are missing.
        """
        self._import_or_unavailable()
        results: dict[str, StreamRefinement] = {}
        for stream_id in geometry.stream_ids():
            refinement = self._refine_stream(geometry, stream_id, decoder=decoder, ingest_artifact=ingest_artifact)
            if refinement is not None:
                results[stream_id] = refinement
        return results

    # -- internals --------------------------------------------------------- #
    @staticmethod
    def _import_or_unavailable():
        """Probe the heavy deps once, converting ImportError into our sentinel."""
        try:
            import importlib

            importlib.import_module("torch")
            importlib.import_module("pycolmap")
            importlib.import_module("vggt.dependency.track_predict")
            importlib.import_module("vggt.utils.load_fn")
        except Exception as exc:  # ImportError, or vendored-tree import failures
            raise PoseRefineUnavailable(
                f"pose refinement dependencies unavailable ({type(exc).__name__}: {exc})"
            ) from exc

    def _log(self, msg: str) -> None:
        self.notes.append(msg)

    @staticmethod
    def _sanitize_name(frame_key: str) -> str:
        return frame_key.replace(":", "_").replace("/", "_").replace(" ", "_")

    def _stream_frames(self, geometry: GeometryArtifact, stream_id: str) -> list[FrameRef]:
        frames = [f for f in geometry.selected_frames if str(f.stream_id) == str(stream_id)]
        # only frames that carry a per-pixel world pointmap can seed BA
        frames = [f for f in frames if f"{stream_id}:{int(f.frame_idx)}" in geometry.pointmaps_by_frame]
        return sorted(frames, key=lambda fr: int(fr.frame_idx))

    def _resolve_frame_paths(self, geometry, stream_id, frames, *, decoder, ingest_artifact, work_dir) -> list[str]:
        """Return ordered on-disk image paths for ``frames`` in the same pixel
        space the geometry backend consumed.

        Prefers the backend's materialized inputs (``metadata.debug_image_paths``,
        the exact files the backend ran on); re-materializes via the decoder when
        those have been cleaned up (e.g. resuming from ``--start-stage``).
        """
        debug_paths = dict((geometry.metadata or {}).get("debug_image_paths", {}) or {})
        resolved: list[str] = []
        missing: list[FrameRef] = []
        for fr in frames:
            key = f"{stream_id}:{int(fr.frame_idx)}"
            src = debug_paths.get(key)
            if src and Path(src).exists():
                resolved.append(str(src))
            else:
                resolved.append("")
                missing.append(fr)
        if not missing:
            return resolved
        # re-materialize the missing frames, mirroring the backend's writer:
        # decode -> optional resize_max_dim (BILINEAR) -> PNG.
        from PIL import Image

        resize_max_dim = (geometry.metadata or {}).get("vggt_long_resize_max_dim")
        resize_max_dim = None if resize_max_dim in (None, 0) else int(resize_max_dim)
        stream_lookup = {s.stream_id: s for s in ingest_artifact.streams}
        frame_dir = work_dir / "frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        for i, fr in enumerate(frames):
            if resolved[i]:
                continue
            stream = stream_lookup[fr.stream_id]
            image = decoder.get_frame(stream, fr.frame_idx)
            pil = Image.fromarray(image)
            if resize_max_dim is not None:
                w, h = pil.size
                max_dim = max(w, h)
                if max_dim > resize_max_dim:
                    scale = float(resize_max_dim) / float(max_dim)
                    pil = pil.resize((max(1, round(w * scale)), max(1, round(h * scale))), resample=Image.BILINEAR)
            out_path = frame_dir / f"{self._sanitize_name(stream_id)}_{int(fr.frame_idx):06d}.png"
            pil.save(out_path)
            resolved[i] = str(out_path)
        return resolved

    def _refine_stream(self, geometry, stream_id, *, decoder, ingest_artifact) -> StreamRefinement | None:
        import torch

        from vggt.dependency.track_predict import predict_tracks
        from vggt.utils.load_fn import load_and_preprocess_images

        frames = self._stream_frames(geometry, stream_id)
        if len(frames) < self.cfg.min_reg_frames:
            self._log(f"stream {stream_id}: only {len(frames)} keyframes with pointmaps; skipped pose refinement.")
            return None

        intr = geometry.intrinsics_by_stream.get(stream_id)
        if intr is None:
            self._log(f"stream {stream_id}: no intrinsics; skipped pose refinement.")
            return None
        K = np.array([[intr.fx, 0, intr.cx], [0, intr.fy, intr.cy], [0, 0, 1]], dtype=np.float64)

        # seed poses (w2c) + per-pixel world points + confidence, in keyframe order
        names, w2c, points, conf = [], [], [], []
        shape: tuple[int, int] | None = None
        for fr in frames:
            key = f"{stream_id}:{int(fr.frame_idx)}"
            state = geometry._find_state_for_frame(fr)
            if state is None:
                self._log(f"stream {stream_id}: frame {key} has no camera state; skipped pose refinement.")
                return None
            c2w = np.asarray(state.pose.T_world_cam, dtype=np.float64)
            w2c.append(np.linalg.inv(c2w))
            names.append(self._sanitize_name(key))
            pm = geometry.pointmaps_by_frame[key]
            pts = np.asarray(pm["points"], dtype=np.float32)
            pts = GeometryArtifact._ensure_hwc_points(pts)
            h, w = pts.shape[:2]
            if shape is None:
                shape = (h, w)
            elif shape != (h, w):
                self._log(
                    f"stream {stream_id}: pointmap resolution {(h, w)} != {shape}; non-uniform, skipped pose refinement."
                )
                return None
            points.append(pts)
            c = pm.get("confidence")
            conf.append(
                np.asarray(c, dtype=np.float32).reshape(h, w)
                if c is not None
                else np.full((h, w), 2.0, np.float32)
            )

        H, W = shape
        w2c = np.stack(w2c)
        points = np.stack(points)
        conf = np.stack(conf)

        work_dir = Path(geometry.save_dir)
        image_paths = self._resolve_frame_paths(
            geometry, stream_id, frames, decoder=decoder, ingest_artifact=ingest_artifact, work_dir=work_dir
        )

        images = load_and_preprocess_images(image_paths, mode="crop").to(self.cfg.device)
        if tuple(images.shape[-2:]) != (H, W):
            # The tracks must live in the SAME pixel space as the artifact's K /
            # world points, or the seed reprojection is meaningless. A mismatch
            # means the crop diverged from the backend's preprocessing.
            raise PoseRefineUnavailable(
                f"stream {stream_id}: tracker image space {tuple(images.shape[-2:])} != geometry pointmap space {(H, W)}; "
                "crop divergence — refusing to seed BA with mismatched coordinates."
            )

        self._log(f"stream {stream_id}: {len(frames)} keyframes at {W}x{H}; predicting VGGSfM tracks.")
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            pred_tracks, pred_vis, _, pred_pts3d, pred_rgb = predict_tracks(
                images,
                conf=conf,
                points_3d=points,
                masks=None,
                max_query_pts=self.cfg.max_query_pts,
                query_frame_num=self.cfg.query_frame_num,
                max_points_num=self.cfg.max_points_num,
                fine_tracking=self.cfg.fine_tracking,
            )
        torch.cuda.empty_cache()

        N = len(frames)
        obs_mask = pred_vis > self.cfg.vis_thresh
        refine_K_any = self.cfg.refine_focal or self.cfg.refine_pp
        rec, npts, nobs = _seed_reconstruction_via_text(
            names, w2c, K, pred_tracks, obs_mask, pred_pts3d, pred_rgb,
            width=W, height=H, max_reproj_error=self.cfg.max_reproj_error,
            out_dir=work_dir, shared_camera=refine_K_any,
        )
        if npts < self.cfg.min_points or rec.num_reg_frames() < self.cfg.min_reg_frames:
            self._log(
                f"stream {stream_id}: seed too sparse ({npts} pts, {rec.num_reg_frames()} reg frames); kept seed poses."
            )
            return None
        self._log(
            f"stream {stream_id}: seeded {npts} pts3D, {nobs} obs, {rec.num_reg_frames()}/{N} reg frames; "
            f"pre-BA reproj={rec.compute_mean_reprojection_error():.3f}px."
        )

        _run_iterative_ba(
            rec, n_iters=self.cfg.ba_iters, max_err=self.cfg.ba_max_err, loss_scale=self.cfg.ba_loss_scale,
            refine_focal=self.cfg.refine_focal, refine_pp=self.cfg.refine_pp, log=self._log,
        )

        refined_intr = intr
        if refine_K_any:
            p = rec.cameras[1].params  # PINHOLE: fx, fy, cx, cy
            refined_intr = CameraIntrinsics(
                fx=float(p[0]), fy=float(p[1]), cx=float(p[2]), cy=float(p[3]),
                distortion_model=intr.distortion_model, distortion_params=intr.distortion_params,
            )
            self._log(
                f"stream {stream_id}: refined shared K fx={p[0]:.2f} fy={p[1]:.2f} cx={p[2]:.2f} cy={p[3]:.2f} "
                f"(seed fx={K[0,0]:.2f} cx={K[0,2]:.2f})."
            )

        refined: dict[str, np.ndarray] = {}
        refined_w2c_by_key: dict[str, np.ndarray] = {}
        seed_w2c_by_key: dict[str, np.ndarray] = {}
        deltas = []
        for i, fr in enumerate(frames):
            key = f"{stream_id}:{int(fr.frame_idx)}"
            refined_w2c = _cam_from_world_matrix(rec.images[i + 1])
            refined[key] = np.linalg.inv(refined_w2c).astype(np.float64)  # back to T_world_cam
            refined_w2c_by_key[key] = refined_w2c
            seed_w2c_by_key[key] = w2c[i]
            r_delta = (np.trace(refined_w2c[:3, :3] @ w2c[i][:3, :3].T) - 1.0) / 2.0
            deltas.append(float(np.degrees(np.arccos(np.clip(r_delta, -1.0, 1.0)))))

        # Re-anchor BA's gauge back onto the input-geometry frame. pycolmap BA
        # optimizes reprojection, which is invariant to a global similarity of
        # {poses, points}; with an unpinned datum the overall scale walks freely
        # along that null-space (empirically 1e3-1e4x the input depth scale).
        # That inflation is harmless to reprojection but degenerate downstream:
        # the scene assembler places gaussians from the UNTOUCHED VGGT depth at
        # the input scale, so poses at a wildly different scale wreck it. A single
        # orientation-based Sim(3) maps the refined poses back onto the seed
        # frame; because it is a global similarity it preserves BA's *relative*
        # pose gains, and because reprojection is invariant to it no re-BA is
        # needed. Depth/point maps are left as-is (already at input scale).
        keys = list(refined)
        s_re, R_re, t_re = _sim3_src_to_dst(refined_w2c_by_key, seed_w2c_by_key, keys)
        for key in keys:
            refined[key] = _apply_sim3_to_c2w(refined[key], s_re, R_re, t_re).astype(np.float64)
        self._log(
            f"stream {stream_id}: re-anchored BA gauge to input frame "
            f"(Sim3 s={s_re:.4g} |t|={np.linalg.norm(t_re):.4g})."
        )

        stats = {
            "reanchor_scale": float(s_re),
            "num_keyframes": int(N),
            "num_points3d": int(rec.num_points3D()),
            "num_obs": int(nobs),
            "num_reg_frames": int(rec.num_reg_frames()),
            "mean_reproj_px": float(rec.compute_mean_reprojection_error()),
            "seed_to_refined_rot_delta_deg_median": float(np.median(deltas)) if deltas else 0.0,
            "seed_to_refined_rot_delta_deg_max": float(np.max(deltas)) if deltas else 0.0,
            "refined_focal": bool(self.cfg.refine_focal),
            "refined_principal_point": bool(self.cfg.refine_pp),
        }
        self._log(
            f"stream {stream_id}: seed->refined per-cam rot delta median={stats['seed_to_refined_rot_delta_deg_median']:.3f}deg "
            f"max={stats['seed_to_refined_rot_delta_deg_max']:.3f}deg."
        )
        return StreamRefinement(
            stream_id=str(stream_id),
            refined_T_world_cam_by_frame_key=refined,
            refined_intrinsics=refined_intr,
            stats=stats,
        )
