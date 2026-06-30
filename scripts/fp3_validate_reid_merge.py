"""FP3 validation: DINOv2 cross-occlusion re-acquisition on real windmill tracks.

The `reid` DINO rung emits a ``merge_map`` that folds appearance-similar,
time-disjoint tracklets onto one identity (design §4A). Because `reid` does not
feed the render path, there is no mPSNR gate; the acceptance bar is **intrinsic
merge precision/recall** on real imagery, which this driver measures.

Method (synthetic occlusion on real tracks):

1. Run the SAM 2 segmenter over N paper-windmill frames -> real per-instance
   tracks (the spinning pinwheel splits into per-petal tracks; FP2b).
2. Each sufficiently long track is split into a time-disjoint **early** half and
   **late** half with a dropped middle gap (the simulated occlusion). The two
   halves are relabelled ``T__a`` / ``T__b``.
3. Run the real :class:`DinoReid` once over the relabelled artifact and read:
   * **recall** — does ``merge_map`` re-merge ``T__a`` with ``T__b``? (the same
     entity, seen before/after a gap during which the petal rotated)
   * **precision** — do *different* tracks' disjoint halves (``Ti__a`` vs
     ``Tj__b``) stay separate?
4. Report the same-entity vs cross-entity cosine separation and the
   precision/recall at the operating threshold.

Run (GPU; SAM 2 + DINOv2 are inference-only, light):
    python scripts/fp3_validate_reid_merge.py \
        --dataset-root ../data/shape_of_motion/iphone/paper-windmill \
        --max-frames 24 --top-k 5 --threshold 0.6
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from itertools import combinations
from pathlib import Path

import numpy as np

from room4d.common.types import FrameRef, VideoStreamRef
from room4d.ingest.ingest_artifact import IngestArtifact
from room4d.lifting.geometry_provider import DyCheckGeometryProvider
from room4d.perception.instance.instance_artifact import InstanceArtifact
from room4d.perception.reid.dino_reid import DinoReid


def _segment_windmill(rgbs_np: np.ndarray):
    """SAM 2 over the frames -> (observations, mask_arrays). Mirrors f0_validate."""
    from room4d.perception.instance.sam2_segment import SAM2InstanceSegmenter

    images = [rgbs_np[i] for i in range(rgbs_np.shape[0])]

    class _D:
        def get_frame(self, stream, idx):
            return images[int(idx)]

    class _S:
        stream_id = "cam_00"

    class _Det:
        backend_used = "none"
        def detect(self, frame):
            return []

    class _I:
        streams = [_S()]

    seg = SAM2InstanceSegmenter(_D(), _Det(), {})
    refs = [FrameRef(stream_id="cam_00", frame_idx=i, timestamp_sec=float(i) * 0.05) for i in range(len(images))]
    obs, masks = seg.run(_I(), refs)
    return obs, masks, images


def _split_long_tracks(observations, *, top_k: int, min_len: int):
    """Relabel the top-k longest tracks into time-disjoint __a/__b halves (gap dropped).

    Returns (new_observations, split_track_ids). Each split keeps the first 40%
    of frames as ``T__a`` and the last 40% as ``T__b``; the middle 20% is dropped
    to leave a genuine temporal gap. Tracks shorter than ``min_len`` are left
    whole.
    """
    by_track: dict[str, list] = {}
    for o in observations:
        by_track.setdefault(o.track_id, []).append(o)
    ranked = sorted(by_track.items(), key=lambda kv: len(kv[1]), reverse=True)

    chosen = [tid for tid, obs in ranked if len(obs) >= min_len][:top_k]
    out, split_ids = [], []
    for tid, obs in by_track.items():
        if tid not in chosen:
            out.extend(obs)
            continue
        ordered = sorted(obs, key=lambda o: int(o.frame_ref.frame_idx))
        n = len(ordered)
        lo, hi = int(n * 0.4), int(n * 0.6)  # early [:lo], gap [lo:hi], late [hi:]
        for o in ordered[:lo]:
            out.append(replace(o, track_id=f"{tid}__a"))
        for o in ordered[hi:]:
            out.append(replace(o, track_id=f"{tid}__b"))
        split_ids.append(tid)
    return out, split_ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--max-frames", type=int, default=24)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--min-len", type=int, default=8)
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--ratio-margin", type=float, default=0.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model-id", default="facebook/dinov2-large")
    args = ap.parse_args()

    # 1. Real windmill frames.
    provider = DyCheckGeometryProvider(args.dataset_root, factor=1, depth_type="depth_anything_colmap", camera_type="refined")
    stack = provider.frame_stack("train", device="cpu")
    rgbs_np = stack.rgbs.cpu().numpy()[: args.max_frames]
    if rgbs_np.dtype != np.uint8:  # frame_stack rgbs are float [0,1]; the decoder/DINO want uint8
        rgbs_np = np.clip(rgbs_np * 255.0, 0, 255).astype(np.uint8)
    n_frames, h, w = rgbs_np.shape[:3]
    print(f"[1] loaded {n_frames} windmill frames ({w}x{h})")

    # 2. Real SAM 2 tracks.
    print("[2] SAM 2 segmenting (discover + propagate)...")
    observations, masks, images = _segment_windmill(rgbs_np)
    n_tracks = len({o.track_id for o in observations})
    print(f"    -> {len(observations)} observations across {n_tracks} tracks")

    # 3. Split the longest tracks into disjoint halves (simulated occlusion gap).
    split_obs, split_ids = _split_long_tracks(observations, top_k=args.top_k, min_len=args.min_len)
    print(f"[3] split {len(split_ids)} long tracks into __a/__b halves: {split_ids}")

    # 4. Real DinoReid over the relabelled artifact.
    class _Decoder:
        def get_frame(self, stream, idx):
            return images[int(idx)]

    ingest = IngestArtifact(
        run_id="fp3", stage_name="ingest", save_dir=Path("/tmp/fp3_validate"),
        streams=[VideoStreamRef(stream_id="cam_00", video_path=Path("windmill"), fps=20.0, num_frames=n_frames, width=w, height=h, has_audio=False)],
    )
    art = InstanceArtifact(run_id="fp3", stage_name="segment", save_dir=Path("/tmp/fp3_validate/seg"), observations=split_obs, mask_arrays=masks)
    art.rebuild_indices()

    print(f"[4] DINOv2 embed + merge (model={args.model_id}, tau={args.threshold})...")
    reid = DinoReid(_Decoder(), {"model_id": args.model_id, "device": args.device, "merge_threshold": args.threshold, "ratio_margin": args.ratio_margin})
    embeddings, merge_map = reid.run(ingest, art)

    def cos(a: str, b: str) -> float:
        return float(np.dot(embeddings[a], embeddings[b]))

    # --- recall: same-entity halves should re-merge ---
    same_cos, recall_hits = [], 0
    for tid in split_ids:
        a, b = f"{tid}__a", f"{tid}__b"
        if a in embeddings and b in embeddings:
            same_cos.append(cos(a, b))
            merged = art_merged(merge_map, a, b)
            recall_hits += int(merged)
    # --- precision: different-entity disjoint halves should stay separate ---
    cross_cos, false_merges = [], 0
    halves_a = [f"{t}__a" for t in split_ids if f"{t}__a" in embeddings]
    halves_b = [f"{t}__b" for t in split_ids if f"{t}__b" in embeddings]
    for a in halves_a:
        for b in halves_b:
            if a[:-3] == b[:-3]:
                continue  # same track (handled in recall)
            cross_cos.append(cos(a, b))
            false_merges += int(art_merged(merge_map, a, b))

    print("\n==================== FP3 reid merge validation ====================")
    print(f"split tracks (positives):     {len(same_cos)}")
    print(f"cross-track disjoint pairs:    {len(cross_cos)}  (negatives)")
    if same_cos:
        print(f"same-entity cosine:  min={min(same_cos):.3f}  mean={np.mean(same_cos):.3f}  max={max(same_cos):.3f}")
    if cross_cos:
        print(f"cross-entity cosine: min={min(cross_cos):.3f}  mean={np.mean(cross_cos):.3f}  max={max(cross_cos):.3f}")
    if same_cos and cross_cos:
        print(f"separation (min_same - max_cross): {min(same_cos) - max(cross_cos):+.3f}")
    print(f"recall  (same-entity halves merged):  {recall_hits}/{len(same_cos)}")
    print(f"precision (false cross merges):       {false_merges} false merges over {len(cross_cos)} pairs")
    print(f"merge_map: {merge_map}")
    print("===================================================================")


def art_merged(merge_map: dict, a: str, b: str) -> bool:
    """True if a and b resolve to the same canonical id under merge_map."""
    def canon(x):
        seen = set()
        while x in merge_map and merge_map[x] != x and x not in seen:
            seen.add(x)
            x = merge_map[x]
        return x
    return canon(a) == canon(b)


if __name__ == "__main__":
    main()
