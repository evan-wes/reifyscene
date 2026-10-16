"""Would an open-vocabulary detector propose the objects our keyframe proposer misses?

    python scripts/probe_detector_coverage.py            # build the cache (GPU)
    python scripts/probe_detector_coverage.py --report

71% of the objects the DAVIS inventory benchmark misses have *no* SAM 2 proposal
overlapping them at all -- not a weak one, none. Fusion cannot reconcile evidence that was
never gathered, so the question is whether a different *source* sees them. The person
detector already showed this works for one class: it recovered a person the automatic
proposer never put forward, and person detection needs no vocabulary.

This asks the same question for everything else, and cheaply. The detector depends on
nothing downstream, so it runs once per keyframe and the cache keeps each detection's IoU
against each ground-truth object -- the same trick that settled the mask-filter question
in ten minutes instead of a 1.75-hour pipeline resume.

**Vocabularies are fixed and blind.** Writing a word list after looking at the clips is
how the earlier semantics experiment fooled itself: it proved a hand-picked list names a
scene, which is not a claim a pipeline can use. These lists are chosen once and applied to
all ninety sequences unseen. COCO-80 is a published standard that predates DAVIS; the
generic list contains no noun more specific than "animal".

Detections are boxes and ground truth is masks, so coverage is scored box-to-box against
each object's bounding box. That is the right proxy: in the pipeline a box seeds SAM 2,
which produces the mask, exactly as the person seeds already do.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DAVIS = REPO.parent / "data/Davis2017/DAVIS-2017-trainval-480p/DAVIS"
MODEL = "IDEA-Research/grounding-dino-base"

GENERIC = ["object", "animal", "person", "vehicle", "thing"]


def _coco80() -> list[str]:
    from reifyscene.perception.objects.detector import COCO_INSTANCE_CATEGORY_NAMES

    return [c for c in COCO_INSTANCE_CATEGORY_NAMES if c not in ("__background__", "N/A")]


def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area = ((a[2] - a[0]) * (a[3] - a[1])) + ((b[2] - b[0]) * (b[3] - b[1])) - inter
    return float(inter / area) if area > 0 else 0.0


def _gt_boxes(arr: np.ndarray) -> dict[str, np.ndarray]:
    out = {}
    for i in np.unique(arr):
        if int(i) == 0:
            continue
        ys, xs = np.nonzero(arr == i)
        out[str(int(i))] = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], float)
    return out


def build(davis: Path, out: Path, threshold: float = 0.25) -> None:
    import torch
    from PIL import Image
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    proc = AutoProcessor.from_pretrained(MODEL)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL).to("cuda").eval()
    vocabs = {"coco80": _coco80(), "generic": GENERIC}

    payload: dict[str, Any] = {"model": MODEL, "threshold": threshold, "sequences": {}}
    from reifyscene.ingest.datasets import DavisSequence

    for seq in DavisSequence.all(davis):
        files = seq.frame_files()
        ann = seq.annotation_for(files[0]) if files else None
        if ann is None:
            continue
        image = Image.open(files[0]).convert("RGB")
        gt = _gt_boxes(ann)
        entry: dict[str, Any] = {"gt": {k: v.tolist() for k, v in gt.items()}, "vocabs": {}}
        for name, words in vocabs.items():
            prompt = ". ".join(words) + "."
            inputs = proc(images=image, text=prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                res = proc.post_process_grounded_object_detection(
                    model(**inputs), inputs.input_ids, threshold=threshold,
                    target_sizes=[image.size[::-1]],
                )[0]
            dets = []
            labels = res.get("text_labels", res.get("labels"))
            for box, label in zip(res["boxes"].cpu().numpy(), labels, strict=False):
                dets.append({
                    "label": str(label),
                    "iou": {g: round(_box_iou(box, gb), 4) for g, gb in gt.items()},
                })
            entry["vocabs"][name] = dets
        payload["sequences"][seq.name] = entry
        print(f"  {seq.name:22} gt={len(gt)} "
              + " ".join(f"{k}={len(v)}" for k, v in entry["vocabs"].items()))

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload) + "\n")
    print(f"cached {len(payload['sequences'])} sequences -> {out}")


def report(cache: dict[str, Any], bench: dict[str, Any], amg: dict[str, Any],
           match_iou: float = 0.5) -> None:
    missed = {r["sequence"]: {str(g) for g in r["missed_ids"]} for r in bench["sequences"]}
    unproposed: set[tuple[str, str]] = set()
    for seq, entry in amg["sequences"].items():
        for gid in missed.get(seq, set()):
            if max((p["iou"].get(gid, 0.0) for p in entry["proposals"]), default=0.0) < 0.01:
                unproposed.add((seq, gid))

    print(f"objects the benchmark misses:                 {sum(len(v) for v in missed.values())}")
    print(f"   of those, with no SAM 2 proposal at all:   {len(unproposed)}\n")
    for name in ("coco80", "generic"):
        covered_all = covered_dark = total_dets = 0
        for seq, entry in cache["sequences"].items():
            dets = entry["vocabs"].get(name, [])
            total_dets += len(dets)
            for gid in entry["gt"]:
                best = max((d["iou"].get(gid, 0.0) for d in dets), default=0.0)
                if best >= match_iou:
                    covered_all += 1
                    if (seq, gid) in unproposed:
                        covered_dark += 1
        gt_total = sum(len(e["gt"]) for e in cache["sequences"].values())
        print(f"{name:8}  detections={total_dets:5}  covers {covered_all:3}/{gt_total} gt objects"
              f"   and {covered_dark:3}/{len(unproposed)} of the ones nothing proposed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, default=REPO / "benchmarks/detector_keyframe_cache.json")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if not args.report:
        build(DAVIS, args.cache)
        return
    report(
        json.loads(args.cache.read_text()),
        json.loads((REPO / "benchmarks/davis_inventory.json").read_text()),
        json.loads((REPO / "benchmarks/amg_keyframe_cache.json").read_text()),
    )


if __name__ == "__main__":
    main()
