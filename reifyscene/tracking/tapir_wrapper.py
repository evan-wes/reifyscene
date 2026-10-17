# Portions adapted from Shape of Motion (https://github.com/vye16/shape-of-motion),
# Copyright (c) 2024 Vickie Ye, under the MIT License (LICENSES/MIT-shape-of-motion.txt).
"""Native TAPIR / BootsTAPIR point-tracker backend.

A real :class:`~reifyscene.tracking.base.PointTracker`, sibling to
:class:`~reifyscene.tracking.cotracker_wrapper.CoTrackerWrapper`, that runs the
vendored tapnet model (:mod:`reifyscene.tracking.tapnet`) to produce **all-pairs**
2D tracks in Shape-of-Motion's ``2d_tracks`` format — replacing the last borrowed
input in the native motion-init path. This is a faithful port of SoM's
``preproc/compute_tracks_torch.py`` onto reifyscene's tracker interface and
:class:`~reifyscene.tracking.all_pairs.AllPairsTracks`.

The all-pairs computation (:meth:`TAPIRWrapper.compute_all_pairs`) is the reusable
core; :meth:`TAPIRWrapper.track` adapts it to the scaffold ``VideoStreamRef``
pipeline (returning the all-pairs result on the extended ``TrackArtifact``
contract), while the native motion-init path calls ``compute_all_pairs`` directly
with the same frames + foreground masks the lift re-checks.

The one intentional deviation from the reference is the frame resize: SoM uses
``mediapy.resize_video`` (Lanczos); we use OpenCV ``INTER_AREA`` to avoid the extra
dependency. The model is robust to this; parity against the shipped tracks is
checked in the validation step.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from reifyscene.common import data_dir
from reifyscene.common.types import FrameRef, VideoStreamRef
from reifyscene.tracking.all_pairs import AllPairsTracks
from reifyscene.tracking.base import PointTracker
from reifyscene.tracking.track_artifact import TrackArtifact

_CKPT_FILENAMES = {
    "bootstapir": "bootstapir_checkpoint_v2.pt",
    "tapir": "tapir_checkpoint_panning.pt",
}


# Where ``scripts/fetch_tapir_checkpoint.sh`` puts checkpoints: a name in the data
# directory's checkpoints.
CHECKPOINT_DIR = "tapnet"


class TAPIRWrapper(PointTracker):
    """All-pairs point tracker backed by the vendored tapnet (bootstapir) model.

    Config keys (all optional):
      ``model_type``       'bootstapir' (default) | 'tapir'
      ``checkpoint_path``  the .pt's name in the data directory's checkpoints
                           (overrides dir+model_type)
      ``checkpoint_dir``   the name of the folder holding the .pt there (default:
                           ``tapnet``)
      ``device``           'cuda' (default, falls back to 'cpu') | 'cpu'
      ``grid_size``        query-grid stride in full-res pixels (default 4, the
                           value SoM used for the shipped DyCheck tracks)
      ``resize_height`` / ``resize_width``  model input size (default 256)
      ``chunk_size``       query points per model forward (default 128)
      ``pyramid_level``    TAPIR pyramid level (default 1, as SoM uses)
    """

    def __init__(self, cfg: dict | None = None) -> None:
        self.cfg = cfg or {}
        self.model_type = str(self.cfg.get("model_type", "bootstapir"))
        if self.model_type not in _CKPT_FILENAMES:
            raise ValueError(
                f"unknown model_type {self.model_type!r} (expected bootstapir or tapir)"
            )
        self.grid_size = int(self.cfg.get("grid_size", 4))
        self.resize_height = int(self.cfg.get("resize_height", 256))
        self.resize_width = int(self.cfg.get("resize_width", 256))
        self.chunk_size = int(self.cfg.get("chunk_size", 128))
        self.pyramid_level = int(self.cfg.get("pyramid_level", 1))
        # Reproducibility: the model's per-forward query permutation
        # (estimate_trajectories' torch.randperm) makes tracks vary run-to-run
        # (~8px). Seeding the global torch RNG before the all-pairs loop pins
        # them. ``None`` restores the legacy nondeterministic behavior.
        _seed = self.cfg.get("seed", 0)
        self.seed = None if _seed is None else int(_seed)
        # Query budget: cap the total number of tracked query points to
        # ``max_total_queries`` (spread evenly per frame). All-pairs seeds a dense
        # grid in every frame, but the native lift keeps only ~num_fg/T of them per
        # frame -- tracking the whole grid and discarding ~95% is the residual
        # O(T^2) cost. ``None`` (default) tracks the full grid (SoM behavior); the
        # dynamic scene path sets a budget (see SingleVideoSceneAssembler).
        _mtq = self.cfg.get("max_total_queries", None)
        self.max_total_queries = None if _mtq is None else int(_mtq)
        self._device_cfg = str(self.cfg.get("device", "cuda"))
        self._model = None
        self._torch = None
        self._device = None

    # -- checkpoint + model -------------------------------------------------

    def checkpoint_path(self) -> Path:
        explicit = self.cfg.get("checkpoint_path")
        if explicit:
            return data_dir.path(data_dir.CHECKPOINTS, explicit)
        ckpt_dir = data_dir.path(data_dir.CHECKPOINTS,
                                 self.cfg.get("checkpoint_dir") or CHECKPOINT_DIR)
        return ckpt_dir / _CKPT_FILENAMES[self.model_type]

    def _resolve_device(self, torch) -> str:
        if self._device_cfg.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
        return self._device_cfg

    def _load_model(self):
        if self._model is not None:
            return self._model
        import torch  # local import: torch is the vision extra

        from reifyscene.tracking.tapnet import TAPIR

        ckpt = self.checkpoint_path()
        if not ckpt.exists():
            raise FileNotFoundError(
                f"tapnet checkpoint not found: {ckpt}\n"
                "Fetch it with: scripts/fetch_tapir_checkpoint.sh "
                f"--model {self.model_type}"
            )
        device = self._resolve_device(torch)
        model = TAPIR(pyramid_level=self.pyramid_level).to(device).eval()
        model.load_state_dict(torch.load(ckpt, map_location=device))
        self._torch = torch
        self._device = device
        self._model = model
        return model

    def release_resources(self) -> None:
        self._model = None
        self._torch = None
        self._device = None

    # -- core: all-pairs computation (port of compute_tracks_torch.main) ----

    def _resize_frames(self, frames: np.ndarray) -> np.ndarray:
        """Resize (T,H,W,3) uint8 frames to model resolution (T,rh,rw,3)."""
        out = np.empty(
            (frames.shape[0], self.resize_height, self.resize_width, 3), dtype=frames.dtype
        )
        for t in range(frames.shape[0]):
            out[t] = cv2.resize(
                frames[t], (self.resize_width, self.resize_height), interpolation=cv2.INTER_AREA
            )
        return out

    def compute_all_pairs(
        self,
        frames: np.ndarray,
        masks: np.ndarray,
        frame_names: list[str],
        *,
        factor: int = 1,
    ) -> AllPairsTracks:
        """Seed a foreground grid in every frame and track each to every frame.

        Parameters
        ----------
        frames:
            ``(T, H, W, 3)`` uint8 RGB at full (``factor``) resolution.
        masks:
            ``(T, H, W)`` foreground masks (bool or float); query points are
            seeded only where ``masks[t] > 0.5``.
        frame_names:
            Length-``T`` ordered frame stems (used for the output file names).
        factor:
            Resolution factor the coords are in (recorded on the result).

        Returns
        -------
        AllPairsTracks with ``tracks[i]`` of shape ``(N_i, T, 4)`` =
        ``[x, y, occ_logit, dist_logit]`` in full-res pixels.
        """
        frames = np.asarray(frames)
        masks = np.asarray(masks)
        num_frames, height, width = frames.shape[:3]
        if len(frame_names) != num_frames:
            raise ValueError(
                f"frame_names ({len(frame_names)}) != number of frames ({num_frames})"
            )
        if masks.shape[:3] != (num_frames, height, width):
            raise ValueError(
                f"masks {masks.shape} must match frames {(num_frames, height, width)}"
            )

        model = self._load_model()
        torch = self._torch
        device = self._device
        rh, rw = self.resize_height, self.resize_width

        # Resize + preprocess the clip once: (1, T, rh, rw, 3) in [-1, 1].
        resized = self._resize_frames(frames)
        frames_t = torch.from_numpy(resized).to(device).float()
        frames_t = (frames_t / 255.0 * 2 - 1)[None]

        # Reproducibility: pin the model's per-forward query permutation.
        if self.seed is not None:
            torch.manual_seed(self.seed)

        # The feature pyramid depends only on the clip (not the query points), so
        # compute it ONCE and reuse it for every query frame/chunk below. The
        # shipped SoM loop recomputed the full-clip ResNet pass on every 128-pt
        # forward -> O(T^2) feature extraction, the scene-assembly hotspot at high
        # frame counts. Reuse makes it O(T); tracks are bit-identical.
        with torch.inference_mode():
            feature_grids = model.get_feature_grids(frames_t, False)

        # Full-res query grid, and its model-resolution counterpart.
        y, x = np.mgrid[0 : height : self.grid_size, 0 : width : self.grid_size]
        y_resize = y / (height - 1) * (rh - 1)
        x_resize = x / (width - 1) * (rw - 1)

        from reifyscene.tracking.tapnet import transforms

        # D: per-frame query cap derived from the total budget. Subsampling here
        # (before tracking) mirrors the lift's own per-frame random keep, so we
        # only track what downstream consumes. Seeded rng -> deterministic.
        per_frame_cap = (
            max(1, self.max_total_queries // max(1, num_frames))
            if self.max_total_queries is not None else None
        )
        rng = np.random.default_rng(self.seed)

        tracks_per_query: list[np.ndarray] = []
        for t in range(num_frames):
            in_mask = masks[t][y, x] > 0.5
            xm, ym = x[in_mask], y[in_mask]                            # full-res query px
            xm_model, ym_model = x_resize[in_mask], y_resize[in_mask]  # model-res query px
            if per_frame_cap is not None and xm.shape[0] > per_frame_cap:
                sel = np.sort(rng.choice(xm.shape[0], size=per_frame_cap, replace=False))
                xm, ym, xm_model, ym_model = xm[sel], ym[sel], xm_model[sel], ym_model[sel]
            all_points_t = np.stack(
                [np.full(xm_model.shape, t, dtype=xm_model.dtype), ym_model, xm_model],
                axis=-1,
            )  # (M, 3) = (t, y_model, x_model)

            if len(all_points_t) > 0:
                outputs: list[np.ndarray] = []
                num_chunks = max(1, len(all_points_t) // self.chunk_size)
                for points in np.array_split(all_points_t, num_chunks, axis=0):
                    pts = torch.from_numpy(points.astype(np.float32))[None].to(device)
                    with torch.inference_mode():
                        preds = model(frames_t, pts, feature_grids=feature_grids)
                    trk = preds["tracks"][0].detach().cpu().numpy()  # (n, T, 2) xy
                    occ = preds["occlusion"][0].detach().cpu().numpy()  # (n, T)
                    dist = preds["expected_dist"][0].detach().cpu().numpy()  # (n, T)
                    trk = transforms.convert_grid_coordinates(
                        trk, (rw - 1, rh - 1), (width - 1, height - 1)
                    )
                    outputs.append(
                        np.concatenate([trk, occ[..., None], dist[..., None]], axis=-1)
                    )
                out = np.concatenate(outputs, axis=0).astype(np.float32)  # (N_t, T, 4)
            else:
                out = np.zeros((0, num_frames, 4), dtype=np.float32)

            # Diagonal: overwrite the query frame's own slot with the exact
            # integer query pixel coords (as SoM does).
            if out.shape[0] > 0:
                out[:, t, :2] = np.stack([xm, ym], axis=-1)
            tracks_per_query.append(out)

        return AllPairsTracks(
            frame_names=list(frame_names),
            tracks=tracks_per_query,
            factor=int(factor),
            image_hw=(int(height), int(width)),
        )

    # -- scaffold-pipeline adapter (PointTracker ABC) -----------------------

    def track(
        self,
        stream: VideoStreamRef,
        *,
        frame_refs: list[FrameRef] | None = None,
        seed_points: np.ndarray | None = None,
    ) -> TrackArtifact:
        """Track a stream and return the all-pairs result on a ``TrackArtifact``.

        Decodes every frame of ``stream`` at native resolution and seeds the
        full grid (foreground = whole frame) unless a ``masks`` array (``(T,H,W)``)
        is supplied via ``cfg['masks']``. ``frame_refs`` / ``seed_points`` are
        accepted for interface compatibility; this backend seeds its own grid.
        """
        del seed_points
        from reifyscene.ingest.decoder import VideoDecoder

        decoder = VideoDecoder()
        numbers = stream.frame_numbers
        num_frames = len(numbers)
        frames = np.stack([decoder.get_frame(stream, i) for i in numbers], axis=0)
        masks = self.cfg.get("masks")
        masks = (
            np.ones(frames.shape[:3], dtype=bool) if masks is None else np.asarray(masks)
        )
        frame_names = [f"{i:05d}" for i in numbers]
        ap = self.compute_all_pairs(frames, masks, frame_names, factor=1)

        selected = frame_refs or stream.frame_refs()
        return TrackArtifact(
            run_id="dev",
            stage_name="tracking",
            save_dir=Path("."),
            selected_frames=selected,
            all_pairs_by_stream={stream.stream_id: ap},
            notes=[
                f"Native {self.model_type} all-pairs tracker: {num_frames} frames, "
                f"{ap.query_counts()} query points per frame.",
            ],
        )
