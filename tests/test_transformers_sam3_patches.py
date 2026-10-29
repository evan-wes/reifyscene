"""The patched SAM 3 video functions (third_party_patches/transformers/) against what they replace.

test_third_party_patches.py checks the patches apply; this checks what they compute, on the
installed transformers. Skips when it is not the patched checkout (setup_reifyscene.sh --with-sam3).
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
m3v = pytest.importorskip("transformers.models.sam3_video.modeling_sam3_video")
if not hasattr(m3v, "greedy_nms_from_ious"):
    pytest.skip("transformers is not the patched checkout", allow_module_level=True)


def _reference_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """mask_iou as transformers v5.12.1 wrote it: through the (N, M, H*W) intermediate."""
    n, h, w = a.shape
    pa, pb = a.view(n, 1, h * w), b.view(1, b.shape[0], h * w)
    return (pa & pb).sum(2).float() / (pa | pb).sum(2).float().clamp(min=1)


def _reference_nms(ious: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    """SAM 3's own CPU fallback (facebookresearch/sam3, sam3/perflib/nms.py generic_nms_cpu), in numpy."""
    order = scores.argsort()[::-1]
    kept = []
    while order.size > 0:
        i = order.item(0)
        kept.append(i)
        order = order[np.where(ious[i, order[1:]] <= threshold)[0] + 1]
    return kept


def _random_masks(n: int, seed: int, size: int = 24) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    masks = np.zeros((n, size, size), bool)
    for k in range(n):
        x0, y0 = rng.integers(0, size - 4, 2)
        w, h = rng.integers(3, size // 2, 2)
        masks[k, y0:y0 + h, x0:x0 + w] = True
    return torch.from_numpy(masks)


@pytest.mark.parametrize("seed", range(5))
def test_mask_iou_matches_the_original(seed: int) -> None:
    a, b = _random_masks(9, seed), _random_masks(6, seed + 100)
    a[0] = False                                   # an empty mask: union clamped, IoU 0
    assert torch.equal(m3v.mask_iou(a, b), _reference_iou(a, b))


def test_mask_iou_is_exact_under_bf16_autocast() -> None:
    a = _random_masks(7, 0, size=300)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        ious = m3v.mask_iou(a, a)
    assert ious.dtype == torch.float32 and torch.equal(ious, _reference_iou(a, a))


@pytest.mark.parametrize("seed", range(5))
def test_greedy_nms_matches_sam3s_cpu_fallback(seed: int) -> None:
    masks = _random_masks(30, seed)
    ious = _reference_iou(masks, masks)
    scores = torch.from_numpy(np.random.default_rng(seed).random(30).astype(np.float32))
    for threshold in (0.1, 0.5):
        kept = m3v.greedy_nms_from_ious(ious, scores, threshold)
        assert kept.dtype == torch.int64
        assert kept.tolist() == _reference_nms(ious.numpy(), scores.numpy(), threshold)


def test_nms_masks_suppresses_duplicates_without_the_kernel(monkeypatch) -> None:
    monkeypatch.setattr(m3v, "cv_utils_kernel", False)       # as when `kernels` is not installed
    base = _random_masks(1, 0)[0]
    shifted = torch.roll(base, 1, dims=1)
    masks = torch.stack([base, shifted, ~base]).float() * 2 - 1   # logits: > 0 inside
    probs = torch.tensor([0.9, 0.8, 0.3])
    keep = m3v.nms_masks(probs, masks, prob_threshold=0.5, iou_threshold=0.1)
    assert keep.tolist() == [True, False, False]     # duplicate suppressed, low score filtered


def test_cross_prompt_nms_is_off_by_default_and_keeps_the_best_detection() -> None:
    from transformers import Sam3VideoConfig

    assert Sam3VideoConfig().cross_prompt_nms_thresh == 0.0
    person = _random_masks(1, 3)[0]
    other = torch.zeros_like(person)
    other[:2, :2] = True
    det = {
        "bbox": torch.arange(12, dtype=torch.float32).view(3, 4),
        "mask": torch.stack([person, person, other]).float() * 2 - 1,
        "scores": torch.tensor([0.7, 0.9, 0.6]),
    }
    model = type("M", (), {"cross_prompt_nms_thresh": 0.5})()
    kept, idx_to_prompt = m3v.Sam3VideoModel._suppress_cross_prompt_duplicates(model, det, {0: 4, 1: 7, 2: 9})
    assert kept["scores"].tolist() == pytest.approx([0.9, 0.6])   # "person" by prompt 7 beat prompt 4
    assert idx_to_prompt == {0: 7, 1: 9}
    assert torch.equal(kept["bbox"], det["bbox"][[1, 2]])
