"""Crops reach the SigLIP 2 and DINOv2 processors as height x width x RGB, whatever their size.

From the shape alone transformers' image processors read a crop 1 or 3 pixels tall as
channels-first: a 3-pixel-tall crop was silently mis-read, and a batch holding a 1-pixel-tall one
next to an ordinary crop failed to stack, which took the semantics stage down to its COCO floor on
a SAM 3 run of DAVIS crossing. The processors themselves are only loaded when cached.
"""

from __future__ import annotations

import numpy as np
import pytest

from reifyscene.perception.reid.dino_reid import DinoReid
from reifyscene.perception.semantics.siglip_semantics import SiglipSemantics

THIN = [np.zeros((40, 30, 3), np.uint8), np.zeros((1, 25, 3), np.uint8), np.zeros((3, 25, 3), np.uint8)]


class _RecordingProcessor:
    def __init__(self):
        self.kwargs = None

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        raise StopIteration  # the call is all this checks


@pytest.mark.parametrize("cls, method", [(SiglipSemantics, "_embed_image_crops"), (DinoReid, "_embed_crops")])
def test_crops_are_declared_channels_last(cls, method) -> None:
    stage = cls.__new__(cls)
    stage._processor = _RecordingProcessor()
    stage._device_resolved = "cpu"
    with pytest.raises(StopIteration):
        getattr(stage, method)(THIN)
    assert stage._processor.kwargs["input_data_format"] == "channels_last"


@pytest.mark.parametrize("model_id, factory", [
    ("google/siglip2-so400m-patch14-384", "AutoProcessor"),
    ("facebook/dinov2-large", "AutoImageProcessor"),
])
def test_the_real_processors_take_thin_crops(model_id: str, factory: str) -> None:
    transformers = pytest.importorskip("transformers")
    try:
        processor = getattr(transformers, factory).from_pretrained(model_id, local_files_only=True)
    except OSError:
        pytest.skip(f"{model_id} is not cached")
    pixels = processor(images=THIN, return_tensors="pt", input_data_format="channels_last")["pixel_values"]
    assert pixels.shape[:2] == (3, 3)
