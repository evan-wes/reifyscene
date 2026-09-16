"""What the keyframe proposer could deliver, under any mask-filter setting.

    python scripts/sweep_mask_filter.py            # build the cache (GPU, ~10 min)
    python scripts/sweep_mask_filter.py --cache benchmarks/amg_keyframe_cache.json --sweep

The DAVIS inventory benchmark says 63 of 205 annotated objects are never recovered. Only
7 of those fall outside the accepted size band, so the filter is not the main story -- but
it *is* the part that can be answered without re-running anything, because the expensive
step (SAM 2's automatic mask generator) depends on none of the thresholds.

So run the generator once per keyframe and cache, for every proposal, the four quantities
the filter tests plus its IoU against each ground-truth object. The masks themselves are
not kept: once those numbers exist the sweep is arithmetic, and any threshold can be
scored in milliseconds.

**It does not bound the pipeline, and an earlier version of this docstring wrongly said
it did.** The numbers here are raw proposal quality, but `add_new_mask` hands a seed to
SAM 2's *video predictor*, which re-segments: the propagated frame-0 mask is a refinement
of the proposal, not the proposal. Measured on DAVIS, kept proposals match 85 of 205
ground-truth objects at IoU 0.5 while the pipeline's own propagated masks match **141** on
that same frame -- and proposals reach only 116 even counting a match at IoU 0.01.
So more than a quarter of what the inventory recovers does not come from a kept proposal
at all: it comes from the other seeding source (person detections, which bypass the mask
generator entirely) and from SAM 2 growing a seed onto a neighbouring object.

What this *does* answer, cheaply and conclusively, is whether the filter thresholds are
leaving recall on the table. They are not: across 36 settings of min-area, max-area and
the instance cap, kept proposals move between 85 and 89 of 205. Raising `max_area_frac`
far enough to admit a koala at 58% of frame gains exactly one object. That question cost
ten minutes of GPU instead of a 1.75-hour pipeline resume, which is the point of caching
the scored table rather than re-running anything.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DAVIS = REPO.parent / "data/Davis2017/DAVIS-2017-trainval-480p/DAVIS"


def _touches_all_borders(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def build_cache(davis: Path, out: Path, frame_idx: int = 0) -> None:
    """Run the mask generator once per sequence keyframe; keep only the scored table."""
    from PIL import Image

    from reifyscene.ingest.decoder import VideoDecoder
    from reifyscene.perception.instance.sam2_segment import SAM2InstanceSegmenter
    from reifyscene.perception.objects.detector import ObjectDetector

    seg = SAM2InstanceSegmenter(
        VideoDecoder(default_fps=30.0),
        ObjectDetector({"backend_mode": "real", "device": "cuda"}),
        {"device": "cuda"},
    )
    seg._load_models()

    payload: dict[str, Any] = {"frame_idx": frame_idx, "sequences": {}}
    images = davis / "JPEGImages/480p"
    annotations = davis / "Annotations/480p"
    for seq_dir in sorted(images.iterdir()):
        img = seq_dir / f"{frame_idx:05d}.jpg"
        ann = annotations / seq_dir.name / f"{frame_idx:05d}.png"
        if not img.exists() or not ann.exists():
            continue
        frame = np.asarray(Image.open(img).convert("RGB"))
        gt_arr = np.asarray(Image.open(ann))
        gt = {int(i): (gt_arr == i) for i in np.unique(gt_arr) if int(i) != 0}
        h, w = frame.shape[:2]

        proposals = []
        for rec in seg._amg.generate(frame):
            m = np.asarray(rec["segmentation"], dtype=bool)
            proposals.append({
                "area_frac": float(m.sum()) / float(h * w),
                "stability": float(rec.get("stability_score", 0.0)),
                "predicted_iou": float(rec.get("predicted_iou", 0.0)),
                "borders": _touches_all_borders(m),
                "iou": {
                    str(gid): round(float((m & g).sum() / (m | g).sum()), 4)
                    for gid, g in gt.items() if (m | g).any()
                },
            })
        payload["sequences"][seq_dir.name] = {
            "gt": {str(gid): round(float(g.sum()) / (h * w), 6) for gid, g in gt.items()},
            "proposals": proposals,
        }
        print(f"  {seq_dir.name:22} {len(proposals):3} proposals, {len(gt)} gt")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload) + "\n")
    print(f"cached {len(payload['sequences'])} sequences -> {out}")


def evaluate(cache: dict[str, Any], *, min_area: float, max_area: float,
             stability: float, predicted_iou: float, max_instances: int,
             match_iou: float = 0.5) -> tuple[int, int, int]:
    """Recovered / total / kept-masks under one filter setting.

    Mirrors `_discover_keyframe_masks`: filter, sort by area descending, keep the first
    ``max_instances``. The sort matters -- admitting large masks can push smaller ones
    past the cap, so a looser bound is not automatically better.
    """
    recovered = total = kept_total = 0
    for entry in cache["sequences"].values():
        gt_ids = list(entry["gt"])
        total += len(gt_ids)
        kept = [
            p for p in entry["proposals"]
            if min_area <= p["area_frac"] <= max_area
            and p["stability"] >= stability
            and p["predicted_iou"] >= predicted_iou
            and not p["borders"]
        ]
        kept.sort(key=lambda p: -p["area_frac"])
        kept = kept[:max_instances]
        kept_total += len(kept)

        pairs = sorted(
            ((p["iou"].get(g, 0.0), g, i) for i, p in enumerate(kept) for g in gt_ids),
            key=lambda t: -t[0],
        )
        taken_g: set[str] = set()
        taken_p: set[int] = set()
        for iou, g, i in pairs:
            if iou >= match_iou and g not in taken_g and i not in taken_p:
                taken_g.add(g)
                taken_p.add(i)
        recovered += len(taken_g)
    return recovered, total, kept_total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, default=REPO / "benchmarks/amg_keyframe_cache.json")
    ap.add_argument("--davis", type=Path, default=DAVIS)
    ap.add_argument("--sweep", action="store_true", help="score settings against an existing cache")
    args = ap.parse_args()

    if not args.sweep:
        build_cache(args.davis, args.cache)
        return

    cache = json.loads(args.cache.read_text())
    shipped = dict(min_area=0.001, max_area=0.5, stability=0.0, predicted_iou=0.0,
                   max_instances=20)
    base = evaluate(cache, **shipped)
    print(
        f"shipped  min={shipped['min_area']} max={shipped['max_area']} "
        f"cap={shipped['max_instances']}  ->  {base[0]}/{base[1]} keyframe objects "
        f"proposed-and-kept, {base[2]} masks\n"
    )

    print(f"{'min_area':>9} {'max_area':>9} {'cap':>4} {'recovered':>10} {'masks':>7}")
    best = (base[0], shipped)
    for min_a, max_a, cap in itertools.product(
        [0.0002, 0.0005, 0.001], [0.5, 0.7, 0.9, 1.0], [20, 40, 80]
    ):
        got, total, masks = evaluate(cache, min_area=min_a, max_area=max_a, stability=0.0,
                                     predicted_iou=0.0, max_instances=cap)
        mark = ""
        if got > best[0]:
            best = (got, dict(min_area=min_a, max_area=max_a, max_instances=cap))
            mark = "  <-"
        print(f"{min_a:9.4f} {max_a:9.2f} {cap:4} {got:7}/{total} {masks:7}{mark}")
    print(f"\nbest: {best[0]}/{base[1]} with {best[1]}  (shipped: {base[0]})")


if __name__ == "__main__":
    main()
