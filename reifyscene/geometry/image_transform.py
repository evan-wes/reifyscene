"""How an ingest frame maps into a geometry backend's pixel space.

A geometry backend does not always see the frame the rest of the pipeline sees.
VGGT and VGGT-Long resize each frame to 518 px wide and centre-crop its height to
518, so their cameras, intrinsics and depth maps describe that square, while the
photos, segment masks and track coordinates everything else works with are in the
ingest frame's pixels. An :class:`ImageTransform` records the exact resizes, crops
and pads from one to the other, so every consumer puts a photo, a mask or a pixel
position into the geometry's pixels the same way -- and none has to guess it from
image sizes (the loaders round differently, so a guess picks the wrong one).

The geometry stage records one transform per stream in
``metadata["image_transforms"]``; :func:`image_transform_for` reads it back.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

# Resample names recorded in a transform. ``unknown``: the images were resized before
# the pipeline saw them (a dataset shipping pre-scaled frames), so no exact filter is
# known -- only its sizes; masks use nearest either way.
_RESAMPLE = {
    "bicubic": Image.Resampling.BICUBIC,
    "bilinear": Image.Resampling.BILINEAR,
    "nearest": Image.Resampling.NEAREST,
    "unknown": Image.Resampling.BILINEAR,
}


class MissingImageTransform(ValueError):
    """A geometry artifact has no recorded transform and its sizes do not prove one."""


@dataclass(frozen=True)
class Resize:
    """Resize to ``size`` (w, h) with the named resample filter."""

    size: tuple[int, int]
    resample: str = "bicubic"

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": "resize",
            "size": [int(self.size[0]), int(self.size[1])],
            "resample": self.resample,
        }


@dataclass(frozen=True)
class Crop:
    """Keep ``box`` = (x0, y0, x1, y1), in the pixels of this op's input."""

    box: tuple[int, int, int, int]

    def to_dict(self) -> dict[str, Any]:
        return {"op": "crop", "box": [int(v) for v in self.box]}


@dataclass(frozen=True)
class Pad:
    """Place the input at ``offset`` (x, y) on a ``size`` (w, h) canvas filled with ``fill``.

    VGGT's loader pads a batch of differently sized images up to the largest, with
    white (rgb 255); masks pad with 0.
    """

    size: tuple[int, int]
    offset: tuple[int, int]
    fill: int = 255

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": "pad",
            "size": [int(self.size[0]), int(self.size[1])],
            "offset": [int(self.offset[0]), int(self.offset[1])],
            "fill": int(self.fill),
        }


Op = Resize | Crop | Pad


def _op_from_dict(raw: dict[str, Any]) -> Op:
    kind = raw.get("op")
    if kind == "resize":
        return Resize(tuple(int(v) for v in raw["size"]), str(raw.get("resample", "bicubic")))
    if kind == "crop":
        return Crop(tuple(int(v) for v in raw["box"]))
    if kind == "pad":
        return Pad(
            tuple(int(v) for v in raw["size"]),
            tuple(int(v) for v in raw["offset"]),
            int(raw.get("fill", 255)),
        )
    raise ValueError(f"unknown image-transform op {raw!r}")


def _op_output_size(op: Op, size: tuple[int, int]) -> tuple[int, int]:
    if isinstance(op, Resize) or isinstance(op, Pad):
        return (int(op.size[0]), int(op.size[1]))
    x0, y0, x1, y1 = op.box
    if not (0 <= x0 < x1 <= size[0] and 0 <= y0 < y1 <= size[1]):
        raise ValueError(f"crop box {op.box} does not fit a {size[0]}x{size[1]} image")
    return (int(x1 - x0), int(y1 - y0))


@dataclass(frozen=True)
class ImageTransform:
    """The ops taking an ingest frame of ``source_size`` (w, h) to the geometry's pixels."""

    source_size: tuple[int, int]
    ops: tuple[Op, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source_size", (int(self.source_size[0]), int(self.source_size[1]))
        )
        object.__setattr__(self, "ops", tuple(self.ops))
        _ = self.sizes  # validates every crop against the size it applies to

    # -- construction / serialization ------------------------------------
    @classmethod
    def identity(cls, size: tuple[int, int]) -> ImageTransform:
        return cls(tuple(size), ())

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_size": [int(self.source_size[0]), int(self.source_size[1])],
            "ops": [op.to_dict() for op in self.ops],
            "output_size": [int(self.output_size[0]), int(self.output_size[1])],
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ImageTransform:
        transform = cls(
            tuple(int(v) for v in raw["source_size"]),
            tuple(_op_from_dict(op) for op in raw.get("ops", [])),
        )
        recorded_out = raw.get("output_size")
        if (
            recorded_out is not None
            and tuple(int(v) for v in recorded_out) != transform.output_size
        ):
            raise ValueError(
                f"image transform records output_size {tuple(recorded_out)} "
                f"but its ops give {transform.output_size}"
            )
        return transform

    # -- sizes -----------------------------------------------------------
    @property
    def sizes(self) -> list[tuple[int, int]]:
        """``sizes[i]`` is op ``i``'s input size; ``sizes[-1]`` the output size."""
        sizes = [self.source_size]
        for op in self.ops:
            sizes.append(_op_output_size(op, sizes[-1]))
        return sizes

    @property
    def output_size(self) -> tuple[int, int]:
        return self.sizes[-1]

    @property
    def is_identity(self) -> bool:
        return self.output_size == self.source_size and all(
            isinstance(op, Resize) and op.size == self.source_size for op in self.ops
        )

    @property
    def is_pure_resize(self) -> bool:
        """Only resizes: every source pixel lands in the output (no crop, no pad)."""
        return all(isinstance(op, Resize) for op in self.ops)

    def _start_op(self, size: tuple[int, int], what: str) -> int:
        """Index of the first op to apply to something already at ``size``."""
        size = (int(size[0]), int(size[1]))
        sizes = self.sizes
        if size == sizes[-1]:
            return len(self.ops)
        for i, s in enumerate(sizes[:-1]):
            if s == size:
                return i
        src, out = self.source_size, sizes[-1]
        raise ValueError(
            f"{what} is {size[0]}x{size[1]}, which is neither the frame this transform starts "
            f"from ({src[0]}x{src[1]}), one of its intermediate sizes, nor the geometry's "
            f"{out[0]}x{out[1]}; it cannot be put into the geometry's pixels"
        )

    # -- images ----------------------------------------------------------
    def apply_to_image(self, img: Any, kind: str = "rgb") -> np.ndarray:
        """``img`` (PIL image or HxW[xC] array) in the geometry's pixel space.

        An image at the source size gets every op; one already at an intermediate or
        the output size gets the remaining ops (or none). RGB resizes use the
        recorded filter, so a VGGT input is reproduced pixel for pixel; masks use
        nearest and keep their dtype. Any other size raises ``ValueError`` -- nothing
        is ever squashed to fit.
        """
        if kind not in ("rgb", "mask"):
            raise ValueError(f"kind must be 'rgb' or 'mask', got {kind!r}")
        arr = np.asarray(img)
        if kind == "rgb":
            pil = (
                img.convert("RGB")
                if isinstance(img, Image.Image)
                else Image.fromarray(np.asarray(arr, dtype=np.uint8))
            )
            dtype = np.uint8
        else:
            arr = np.squeeze(arr) if arr.ndim == 3 and arr.shape[-1] == 1 else arr
            dtype = arr.dtype
            pil = Image.fromarray(np.asarray(arr, dtype=np.float32), mode="F")
        start = self._start_op(pil.size, "the image")
        for op in self.ops[start:]:
            if isinstance(op, Resize):
                resample = _RESAMPLE[op.resample] if kind == "rgb" else Image.Resampling.NEAREST
                pil = pil.resize(tuple(op.size), resample)
            elif isinstance(op, Crop):
                pil = pil.crop(tuple(op.box))
            else:
                fill = op.fill if kind == "rgb" else 0
                canvas = Image.new(
                    pil.mode, tuple(op.size), (fill, fill, fill) if kind == "rgb" else float(fill)
                )
                canvas.paste(pil, tuple(op.offset))
                pil = canvas
        out = np.asarray(pil)
        return out.astype(dtype) if out.dtype != dtype else np.array(out)

    # -- points ----------------------------------------------------------
    def map_points(
        self, xy: Any, *, from_size: tuple[int, int] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pixel positions (..., 2) in the geometry's pixels, and whether each lands in it.

        Resizes scale by (w_out / w_in, h_out / h_in) -- the ``u * w / W`` convention the
        pipeline has always used, so a crop-free transform maps exactly as before;
        crops subtract their offset, pads add theirs. ``inside`` is True where the
        mapped point lies in the output image (``0 <= x < w``, ``0 <= y < h``): a point
        the crop removed has no geometry pixel and must not borrow a border one.
        ``from_size`` is the size the positions are measured in (default: the source).
        """
        pts = np.asarray(xy, dtype=np.float64).copy()
        start = (
            0 if from_size is None else self._start_op(from_size, "the point coordinates' frame")
        )
        sizes = self.sizes
        for i in range(start, len(self.ops)):
            op, (w_in, h_in) = self.ops[i], sizes[i]
            if isinstance(op, Resize):
                pts[..., 0] *= float(op.size[0]) / float(w_in)
                pts[..., 1] *= float(op.size[1]) / float(h_in)
            elif isinstance(op, Crop):
                pts[..., 0] -= float(op.box[0])
                pts[..., 1] -= float(op.box[1])
            else:
                pts[..., 0] += float(op.offset[0])
                pts[..., 1] += float(op.offset[1])
        w, h = sizes[-1]
        inside = (
            np.isfinite(pts).all(axis=-1)
            & (pts[..., 0] >= 0)
            & (pts[..., 0] < w)
            & (pts[..., 1] >= 0)
            & (pts[..., 1] < h)
        )
        return pts, inside


# --------------------------------------------------------------------------- #
# VGGT's image loaders.
# --------------------------------------------------------------------------- #
VGGT_TARGET_SIZE = 518


def materialize_resize(source_size: tuple[int, int], resize_max_dim: int | None) -> Resize | None:
    """The optional downscale the VGGT frame cache applies before the model's loader.

    Mirrors ``VGGTInitializer._materialize_images``: shrink to ``resize_max_dim`` on
    the long side (bilinear) only when the frame is larger; ``None`` when untouched.
    """
    if resize_max_dim in (None, 0):
        return None
    width, height = int(source_size[0]), int(source_size[1])
    max_dim = max(width, height)
    if max_dim <= int(resize_max_dim):
        return None
    scale = float(resize_max_dim) / float(max_dim)
    return Resize(
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))), "bilinear"
    )


def vggt_loader_ops(
    size: tuple[int, int],
    *,
    loader: str = "upstream",
    target_size: int = VGGT_TARGET_SIZE,
    mode: str = "crop",
) -> list[Op]:
    """The resize + crop one of VGGT's loaders applies to an image of ``size`` (w, h).

    ``upstream`` mirrors ``vggt.utils.load_fn.load_and_preprocess_images``: width to
    518, height to ``round(h * 518 / w / 14) * 14``, bicubic, then the rows centred
    on 518 when taller. ``bounded`` mirrors VGGT-Long's no-upscale loader
    (``VGGTLongInitializer._load_and_preprocess_images_without_upscale``): width to
    ``min(target, w)`` in whole 14-pixel patches, height in proportion (floored to 14),
    then the rows centred on the target when taller -- like upstream, never upscaling.
    ``pad`` mode letterboxes, which no backend uses: it is not implemented rather than
    guessed.
    """
    if mode != "crop":
        raise NotImplementedError(
            f"VGGT loader mode {mode!r} is not recorded; only 'crop' is implemented"
        )
    width, height = int(size[0]), int(size[1])
    ops: list[Op] = []
    if loader == "upstream":
        new_width = VGGT_TARGET_SIZE
        new_height = round(height * (new_width / width) / 14) * 14
        ops.append(Resize((new_width, new_height), "bicubic"))
        if new_height > VGGT_TARGET_SIZE:
            start_y = (new_height - VGGT_TARGET_SIZE) // 2
            ops.append(Crop((0, start_y, new_width, start_y + VGGT_TARGET_SIZE)))
    elif loader == "bounded":
        target = max(14, int(target_size))
        new_width = max(14, (min(target, width) // 14) * 14)
        new_height = max(14, int(np.floor(height * new_width / width / 14.0)) * 14)
        ops.append(Resize((new_width, new_height), "bicubic"))
        if new_height > target:
            start_y = (new_height - target) // 2
            ops.append(Crop((0, start_y, new_width, start_y + target)))
    else:
        raise ValueError(
            f"unknown VGGT loader {loader!r}: expected 'upstream' or 'bounded'"
        )
    return ops


def vggt_input_transform(
    source_size: tuple[int, int],
    *,
    target_size: int = VGGT_TARGET_SIZE,
    loader: str = "upstream",
    mode: str = "crop",
    resize_max_dim: int | None = None,
) -> ImageTransform:
    """Ingest frame -> VGGT's input pixels: the cache downscale (if any), then the loader."""
    ops: list[Op] = []
    size = (int(source_size[0]), int(source_size[1]))
    pre = materialize_resize(size, resize_max_dim)
    if pre is not None:
        ops.append(pre)
        size = pre.size
    ops.extend(vggt_loader_ops(size, loader=loader, target_size=target_size, mode=mode))
    return ImageTransform(tuple(source_size), tuple(ops))


def pad_to_batch(
    transforms: dict[str, ImageTransform], fill: int = 255
) -> dict[str, ImageTransform]:
    """Add the pad VGGT's loader applies when one batch mixes image sizes.

    Every image is centred on a canvas of the batch's largest height and width
    (``pad_top = dh // 2``, ``pad_left = dw // 2``, white), as the loader does.
    A batch of one size is returned unchanged.
    """
    sizes = {t.output_size for t in transforms.values()}
    if len(sizes) <= 1:
        return dict(transforms)
    max_w = max(s[0] for s in sizes)
    max_h = max(s[1] for s in sizes)
    out = {}
    for key, t in transforms.items():
        w, h = t.output_size
        if (w, h) == (max_w, max_h):
            out[key] = t
            continue
        offset = ((max_w - w) // 2, (max_h - h) // 2)
        out[key] = ImageTransform(t.source_size, t.ops + (Pad((max_w, max_h), offset, fill),))
    return out


# --------------------------------------------------------------------------- #
# Records on geometry artifacts.
# --------------------------------------------------------------------------- #
def transform_record(transform: ImageTransform, *, recorded_by: str) -> dict[str, Any]:
    """The ``metadata["image_transforms"][stream]`` entry a geometry backend writes."""
    return {**transform.to_dict(), "source": "recorded", "recorded_by": str(recorded_by)}


def image_transform_for(geometry_artifact, stream_id: str) -> ImageTransform:
    """The recorded ingest-frame -> geometry-pixels transform for one stream.

    Every geometry backend records one; a stream without a record raises
    :class:`MissingImageTransform` rather than being given a guessed mapping.
    """
    records = (getattr(geometry_artifact, "metadata", {}) or {}).get("image_transforms") or {}
    raw = records.get(str(stream_id))
    if raw is None:
        raise MissingImageTransform(
            f"geometry stream {stream_id!r} has no recorded image transform; its pixels cannot "
            "be related to the frames"
        )
    return ImageTransform.from_dict(raw)