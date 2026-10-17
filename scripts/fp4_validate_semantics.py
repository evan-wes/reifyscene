"""FP4a validation: SigLIP2 open-vocab semantics — two complementary gates.

`semantics` does not feed render (design §4A back-half contract), so there is no
mPSNR gate; the bar is intrinsic label quality, measured two ways:

1. **coco-agreement** (quantitative) — on a COCO-object clip (``spin``, a person
   scene), run the Mask R-CNN floor segmenter to get instances with their COCO
   labels, then SigLIP2 zero-shot over the COCO-80 vocab on the same masked
   tracks. Report top-1 / top-5 agreement of SigLIP2 vs the detector. A faithful
   labeler agrees with the independent detector on the closed vocab.

2. **windmill-qualitative** (open-vocab value-add) — on ``paper-windmill`` (whose
   Mask R-CNN guesses are nonsense: potted plant / sports ball / vase), run SAM 2
   class-agnostic tracks + SigLIP2 over an *extended* vocabulary and print the
   top-3 categories per track. Shows the open-vocab naming the detector floor can't.

Run (GPU; Mask R-CNN / SAM 2 / SigLIP2 are inference-only):
    python scripts/fp4_validate_semantics.py --mode both \
        --coco-root dycheck_iphone/spin \
        --windmill-root dycheck_iphone/paper-windmill --max-frames 16
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from reifyscene.common import data_dir
from reifyscene.common.types import FrameRef, VideoStreamRef
from reifyscene.ingest.ingest_artifact import IngestArtifact
from reifyscene.lifting.geometry_provider import DyCheckGeometryProvider
from reifyscene.perception.instance.instance_artifact import InstanceArtifact
from reifyscene.perception.semantics.siglip_semantics import DEFAULT_VOCAB, FloorSemantics, SiglipSemantics

WINDMILL_EXTRA_VOCAB = [
    'pinwheel', 'windmill', 'paper windmill', 'toy', 'fan', 'flower', 'star', 'origami', 'umbrella', 'kite',
]


def _load_frames(root: str, max_frames: int) -> np.ndarray:
    prov = DyCheckGeometryProvider(root, factor=1, depth_type='depth_anything_colmap', camera_type='refined')
    rgbs = prov.frame_stack('train', device='cpu').rgbs.cpu().numpy()[:max_frames]
    if rgbs.dtype != np.uint8:
        rgbs = np.clip(rgbs * 255.0, 0, 255).astype(np.uint8)
    return rgbs


def _ingest_and_decoder(rgbs: np.ndarray):
    images = [rgbs[i] for i in range(rgbs.shape[0])]
    n, h, w = rgbs.shape[:3]

    class _Decoder:
        def get_frame(self, stream, idx):
            return images[int(idx)]

    ingest = IngestArtifact(
        run_id='fp4', stage_name='ingest', save_dir=data_dir.path(data_dir.CACHE, 'fp4_validate'),
        streams=[VideoStreamRef(stream_id='cam_00', video_path=Path('clip'), fps=20.0, num_frames=n, width=w, height=h, has_audio=False)],
    )
    refs = [FrameRef(stream_id='cam_00', frame_idx=i, timestamp_sec=float(i) * 0.05) for i in range(n)]
    return ingest, _Decoder(), refs


def coco_agreement(args) -> None:
    from reifyscene.perception.instance.segment import FloorInstanceSegmenter
    from reifyscene.perception.objects.detector import ObjectDetector
    from reifyscene.perception.objects.segmenter import ObjectSegmenter

    print(f"[coco] loading {args.max_frames} frames from {args.coco_root}")
    rgbs = _load_frames(args.coco_root, args.max_frames)
    ingest, decoder, refs = _ingest_and_decoder(rgbs)

    print("[coco] Mask R-CNN floor segment (detect + IoU tracks)...")
    floor_seg = FloorInstanceSegmenter(decoder, ObjectDetector({}), ObjectSegmenter({}), {})
    obs, masks = floor_seg.run(ingest, refs)
    art = InstanceArtifact(run_id='fp4', stage_name='segment', save_dir=data_dir.path(data_dir.CACHE, 'fp4_validate/seg'), observations=obs, mask_arrays=masks)
    art.rebuild_indices()
    # detector "ground truth" per track = majority COCO concept
    gt_records, _ = FloorSemantics().run(ingest, art)
    print(f"    -> {len(obs)} detections, {len(gt_records)} tracks")

    print("[coco] SigLIP2 zero-shot over COCO-80...")
    records, _ = SiglipSemantics(decoder, None, {'vocab': DEFAULT_VOCAB, 'device': args.device, 'topk': 5}).run(ingest, art)

    top1 = top5 = total = 0
    rows = []
    for tid, gt in gt_records.items():
        if gt.category == 'unknown' or tid not in records:
            continue
        pred = records[tid]
        topk_labels = [lbl for lbl, _ in pred.category_topk]
        total += 1
        top1 += int(pred.category == gt.category)
        top5 += int(gt.category in topk_labels)
        rows.append((gt.category, pred.category, pred.score))

    print("\n============== FP4 coco-agreement (SigLIP2 vs Mask R-CNN) ==============")
    print(f"tracks compared: {total}")
    if total:
        print(f"top-1 agreement: {top1}/{total} = {100*top1/total:.0f}%")
        print(f"top-5 agreement: {top5}/{total} = {100*top5/total:.0f}%")
    for gt, pred, score in sorted(rows):
        mark = '==' if gt == pred else '!='
        print(f"  detector={gt:16s} {mark} siglip={pred:16s} ({score:+.3f})")
    print("=======================================================================")


def windmill_qualitative(args) -> None:
    from reifyscene.perception.instance.sam2_segment import SAM2InstanceSegmenter

    print(f"\n[windmill] loading {args.max_frames} frames from {args.windmill_root}")
    rgbs = _load_frames(args.windmill_root, args.max_frames)
    ingest, decoder, refs = _ingest_and_decoder(rgbs)

    class _Det:
        backend_used = 'none'
        def detect(self, frame):
            return []

    print("[windmill] SAM 2 class-agnostic segment...")
    obs, masks = SAM2InstanceSegmenter(decoder, _Det(), {}).run(ingest, refs)
    art = InstanceArtifact(run_id='fp4', stage_name='segment', save_dir=data_dir.path(data_dir.CACHE, 'fp4_validate/seg2'), observations=obs, mask_arrays=masks)
    art.rebuild_indices()
    n_tracks = len({o.track_id for o in obs})
    print(f"    -> {len(obs)} obs across {n_tracks} class-agnostic tracks")

    vocab = DEFAULT_VOCAB + WINDMILL_EXTRA_VOCAB
    print(f"[windmill] SigLIP2 zero-shot over COCO-80 + {len(WINDMILL_EXTRA_VOCAB)} extras...")
    records, _ = SiglipSemantics(decoder, None, {'vocab': vocab, 'device': args.device, 'topk': 3}).run(ingest, art)

    print("\n============== FP4 windmill open-vocab (SigLIP2 names SAM 2 tracks) =====")
    for tid in sorted(records):
        topk = ', '.join(f"{lbl} ({s:+.2f})" for lbl, s in records[tid].category_topk)
        print(f"  {tid:18s} -> {topk}")
    print("=======================================================================")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['both', 'coco', 'windmill'], default='both')
    ap.add_argument('--coco-root', default='dycheck_iphone/spin',
                    help='a sequence name in datasets/')
    ap.add_argument('--windmill-root', default='dycheck_iphone/paper-windmill',
                    help='a sequence name in datasets/')
    ap.add_argument('--max-frames', type=int, default=16)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    args.coco_root = str(data_dir.path(data_dir.DATASETS, args.coco_root))
    args.windmill_root = str(data_dir.path(data_dir.DATASETS, args.windmill_root))
    if args.mode in ('both', 'coco'):
        coco_agreement(args)
    if args.mode in ('both', 'windmill'):
        windmill_qualitative(args)


if __name__ == '__main__':
    main()
