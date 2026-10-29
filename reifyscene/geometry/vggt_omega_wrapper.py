"""Wrapper module for the VGGT-Omega geometry initializer.

VGGT-Omega (facebookresearch/vggt-omega, CVPR 2026) predicts every selected frame's
camera and depth in one forward pass, so there are no chunks to join, as VGGT-Long has
to. It has no point-map head: the point maps are its depth lifted with each frame's own
predicted camera (:func:`reifyscene.common.depth.unproject_depth_to_world`).

- Code: importable as ``vggt_omega``, either installed or from ``third_party/vggt-omega``
  (where ``scripts/setup_reifyscene.sh`` clones it).
- Weights: ``facebook/VGGT-Omega`` on Hugging Face, which is gated. Request access on the
  model page, then ``hf auth login``. ``checkpoint_path`` loads a local file instead.
- Input: its own loader resizes the whole frame, with no crop, to 16-pixel patches.
  ``omega_mode`` "balanced" keeps about ``omega_resolution``^2 pixels; "max_size" takes
  the long side to ``omega_resolution`` and needs less memory. The transform is recorded
  as VGGT's is.
- Intrinsics: each camera state keeps its own frame's, as the model predicts them; the
  stream's intrinsics are their median.
- Memory: one pass needs about :data:`PEAK_GIB_FIXED` GiB plus :data:`PEAK_GIB_PER_KPATCH`
  GiB per 1000 input patches (measured on an RTX 5090, 2026-10-28: 277 frames at 448x592
  peaked at 25.3 GiB, 426 frames at 384x512 at 28.4 GiB). A clip that does not fit fails
  with that estimate and the ways out; it is never split into chunks.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from reifyscene.common.depth import unproject_depth_to_world
from reifyscene.common.types import FrameRef
from reifyscene.geometry.geometry_artifact import GeometryArtifact
from reifyscene.geometry.image_transform import (
    VGGT_OMEGA_MODES,
    ImageTransform,
    pad_to_batch,
    vggt_omega_input_transform,
)
from reifyscene.geometry.vggt_wrapper import VGGTInitializer, input_cache_root

DEFAULT_HF_REPO = "facebook/VGGT-Omega"
# The checkpoint the README recommends for real-world video.
DEFAULT_CHECKPOINT = "vggt_omega_1b_512.pt"
PATCH_SIZE = 16
# Peak GPU memory of one pass, in GiB: a fixed part (mostly the weights) plus a part per
# 1000 input patches. Fitted to the 2026-10-28 runs above (DAVIS clips of 34-83 frames at
# 688x384, and the two DyCheck clips), within 0.2 GiB of each.
PEAK_GIB_FIXED = 4.3
PEAK_GIB_PER_KPATCH = 0.073


def estimate_peak_gib(num_frames: int, input_size: tuple[int, int]) -> float:
    """Estimated peak GPU memory (GiB) of one pass over ``num_frames`` inputs of ``input_size``
    (w, h)."""
    patches = (
        int(num_frames) * (int(input_size[0]) // PATCH_SIZE) * (int(input_size[1]) // PATCH_SIZE)
    )
    return PEAK_GIB_FIXED + PEAK_GIB_PER_KPATCH * patches / 1000.0


def _import_vggt_omega():
    """``vggt_omega``, installed or from the ``third_party/vggt-omega`` checkout."""
    try:
        return importlib.import_module("vggt_omega")
    except ImportError as installed_error:
        local_checkout = Path(__file__).resolve().parents[2] / "third_party" / "vggt-omega"
        if not local_checkout.exists():
            raise ImportError(
                "Could not import 'vggt_omega': it is not installed and there is no checkout at "
                f"{local_checkout} (scripts/setup_reifyscene.sh clones it there)"
            ) from installed_error
        sys.path.insert(0, str(local_checkout))
        return importlib.import_module("vggt_omega")


class VGGTOmegaInitializer(VGGTInitializer):
    """VGGT-Omega in one forward pass over every selected frame.

    Config keys, besides VGGT's (``backend_mode``, ``checkpoint_path``, ``device``,
    ``frame_cache_dir``, ``resize_max_dim``, ``debug_shapes``):

    omega_mode:
        ``balanced`` (default) or ``max_size``, the loader's resize.
    omega_resolution:
        The loader's target resolution in pixels (default 512).
    omega_checkpoint:
        The checkpoint file in ``omega_hf_repo`` (default :data:`DEFAULT_CHECKPOINT`).
    omega_hf_repo:
        The Hugging Face repository holding it (default :data:`DEFAULT_HF_REPO`).
    """

    backend_name = "vggt-omega"
    model_label = "VGGT-Omega"

    def __init__(self, cfg: dict | None = None) -> None:
        super().__init__(cfg)
        self.omega_mode = str(self.cfg.get("omega_mode", "balanced"))
        if self.omega_mode not in VGGT_OMEGA_MODES:
            raise ValueError(
                f"omega_mode must be one of {VGGT_OMEGA_MODES}, got {self.omega_mode!r}"
            )
        self.omega_resolution = int(self.cfg.get("omega_resolution", 512))
        self.omega_checkpoint = str(self.cfg.get("omega_checkpoint", DEFAULT_CHECKPOINT))
        self.omega_hf_repo = str(self.cfg.get("omega_hf_repo", DEFAULT_HF_REPO))
        self._VGGTOmega = None
        self._encoding_to_camera = None

    def _load_backend_components(self) -> None:
        _import_vggt_omega()
        from vggt_omega.models import VGGTOmega  # type: ignore[import-not-found]
        from vggt_omega.utils.load_fn import (
            load_and_preprocess_images,  # type: ignore[import-not-found]
        )
        from vggt_omega.utils.pose_enc import encoding_to_camera  # type: ignore[import-not-found]

        self._VGGTOmega = VGGTOmega
        self._load_and_preprocess_images = load_and_preprocess_images
        self._encoding_to_camera = encoding_to_camera

    def _checkpoint_file(self) -> str:
        """A local checkpoint path: ``checkpoint_path``, or the file fetched from Hugging Face."""
        if self.checkpoint_path:
            return str(self.checkpoint_path)
        from huggingface_hub import hf_hub_download

        try:
            return hf_hub_download(self.omega_hf_repo, self.omega_checkpoint)
        except Exception as exc:
            if type(exc).__name__ == "GatedRepoError":
                raise RuntimeError(
                    f"The VGGT-Omega weights ({self.omega_hf_repo}) are gated on Hugging Face. "
                    f"Request access at https://huggingface.co/{self.omega_hf_repo} while logged "
                    "in, then run `hf auth login` on this machine."
                ) from exc
            raise

    def load_checkpoint(self) -> None:
        if self.backend_mode == "heuristic":
            self._is_loaded = False
            return
        try:
            self._set_cpu_threads()
            self._load_backend_components()
            # Its own forward autocasts to bf16 on CUDA; on CPU it runs in fp32.
            model = self._VGGTOmega(autocast=self.device.startswith("cuda"))
            model.load_state_dict(torch.load(self._checkpoint_file(), map_location="cpu"))
            self._model = model.to(self.device).eval()
            self._is_loaded = True
        except Exception as exc:  # pragma: no cover - exercised in integration usage
            self._backend_error = exc
            self._is_loaded = False
            if self.backend_mode == "real":
                raise RuntimeError(
                    f"Failed to initialize the real VGGT-Omega backend: {exc}"
                ) from exc

    def release_resources(self) -> None:
        super().release_resources()
        self._VGGTOmega = None
        self._encoding_to_camera = None

    def _input_transforms_for(
        self, frames: list[FrameRef], ingest_artifact, mode: str
    ) -> dict[str, ImageTransform]:
        sizes = self._stream_source_sizes(frames, ingest_artifact)
        return pad_to_batch(
            {
                stream_id: vggt_omega_input_transform(
                    size,
                    mode=mode,
                    resolution=self.omega_resolution,
                    resize_max_dim=self.resize_max_dim,
                )
                for stream_id, size in sizes.items()
            }
        )

    def _input_transforms(
        self, frames: list[FrameRef], ingest_artifact
    ) -> dict[str, ImageTransform]:
        return self._input_transforms_for(frames, ingest_artifact, self.omega_mode)

    def _out_of_memory_message(
        self, frames: list[FrameRef], ingest_artifact, input_size: tuple[int, int]
    ) -> str:
        n = len(frames)
        message = (
            f"VGGT-Omega ran out of GPU memory on {n} frames at {input_size[0]}x{input_size[1]} "
            f"(omega_mode {self.omega_mode}): one pass needs about "
            f"{estimate_peak_gib(n, input_size):.1f} GiB"
        )
        if self.device.startswith("cuda") and torch.cuda.is_available():
            total = torch.cuda.get_device_properties(torch.device(self.device)).total_memory / 2**30
            message += f", and this GPU has {total:.1f} GiB in all"
        ways = []
        if self.omega_mode != "max_size":
            smaller = next(
                iter(self._input_transforms_for(frames, ingest_artifact, "max_size").values())
            ).output_size
            ways.append(
                f"omega_mode max_size (--vggt-omega-mode max_size; about "
                f"{estimate_peak_gib(n, smaller):.1f} GiB)"
            )
        ways.append("fewer frames (--geometry-frame-stride)")
        ways.append(
            "VGGT-Long, which splits the clip into chunks (--geometry-backend-name vggt-long)"
        )
        return message + ". Ways out: " + "; ".join(ways) + "."

    def initialize(
        self,
        frames: list[FrameRef],
        *,
        ingest_artifact=None,
        sync_artifact=None,
    ) -> GeometryArtifact:
        """Cameras and depth for every selected frame, from one forward pass."""
        selected_frames = self._select_frames(frames, ingest_artifact)
        if not selected_frames:
            raise ValueError("VGGTOmegaInitializer.initialize() received no frames to process.")

        if not self._is_loaded and self._model is None and self.backend_mode != "heuristic":
            self.load_checkpoint()

        if self._model is None:
            return self._initialize_heuristic(selected_frames, ingest_artifact, sync_artifact)

        cache_root = input_cache_root(self.frame_cache_dir)
        image_paths, notes = self._materialize_images(selected_frames, ingest_artifact, cache_root)
        transforms = self._input_transforms(selected_frames, ingest_artifact)
        images = self._load_and_preprocess_images(
            [str(path) for path in image_paths],
            mode=self.omega_mode,
            image_resolution=self.omega_resolution,
            patch_size=PATCH_SIZE,
        ).to(self.device)
        height, width = int(images.shape[-2]), int(images.shape[-1])
        recorded = {t.output_size for t in transforms.values()}
        if recorded != {(width, height)}:
            raise RuntimeError(
                f"the recorded VGGT-Omega input transform gives {sorted(recorded)} but its loader "
                f"made {width}x{height}; the depth could not be related to the frames"
            )

        on_cuda = self.device.startswith("cuda") and torch.cuda.is_available()
        if on_cuda:
            torch.cuda.reset_peak_memory_stats(torch.device(self.device))
        try:
            with torch.inference_mode():
                predictions = self._model(images)
        except torch.OutOfMemoryError as exc:
            del images
            torch.cuda.empty_cache()
            raise RuntimeError(
                self._out_of_memory_message(selected_frames, ingest_artifact, (width, height))
            ) from exc
        peak_gib = (
            torch.cuda.max_memory_allocated(torch.device(self.device)) / 2**30 if on_cuda else None
        )

        if self.debug_shapes:
            self._debug_dump_predictions(predictions)

        extrinsic, intrinsic = self._encoding_to_camera(
            predictions["pose_enc"].float(), (height, width)
        )
        extrinsic = extrinsic[0].double().cpu().numpy()
        intrinsic = intrinsic[0].double().cpu().numpy()
        depth = predictions["depth"][0].float().cpu().numpy()
        depth = depth if depth.ndim == 4 else depth[..., None]
        depth_conf = predictions["depth_conf"][0].float().cpu().numpy()
        del predictions, images

        point_map = np.stack(
            [
                unproject_depth_to_world(
                    depth[idx],
                    self._intrinsics_from_matrix(intrinsic[idx]),
                    self._world_from_cam(extrinsic[idx]),
                )[0]
                for idx in range(len(selected_frames))
            ]
        )

        notes = list(notes)
        memory = "" if peak_gib is None else f"; peak GPU memory {peak_gib:.1f} GiB"
        notes.append(
            f"VGGT-Omega ran one pass over {len(selected_frames)} frames at {width}x{height} "
            f"(omega_mode {self.omega_mode}, {self.omega_checkpoint}){memory}."
        )
        notes.append("Point maps are the VGGT-Omega depth lifted with each frame's own camera.")
        notes.append(
            "Each camera state keeps its own frame's intrinsics; the stream's are their median."
        )
        artifact = self._initialize_from_predictions(
            selected_frames,
            {
                "extrinsic": extrinsic,
                "intrinsic": intrinsic,
                "depth": depth,
                "depth_conf": depth_conf,
                "point_map": point_map,
            },
            ingest_artifact,
            notes=notes,
            image_paths=image_paths,
            per_frame_intrinsics=True,
        )
        self._record_transforms(artifact, transforms)
        artifact.metadata["vggt_omega"] = {
            "mode": self.omega_mode,
            "resolution": self.omega_resolution,
            "checkpoint": Path(self.checkpoint_path).name
            if self.checkpoint_path
            else self.omega_checkpoint,
            "input_size": [width, height],
            "estimated_peak_gib": round(
                estimate_peak_gib(len(selected_frames), (width, height)), 2
            ),
            "peak_gib": None if peak_gib is None else round(peak_gib, 2),
        }
        return artifact

    def _debug_dump_predictions(self, predictions: Any) -> None:
        print("[VGGT-Omega DEBUG] ---- raw prediction summary ----")
        self._debug_describe_value("predictions", predictions, prefix="[VGGT-Omega DEBUG]")
        print("[VGGT-Omega DEBUG] --------------------------------")
