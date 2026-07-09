"""FP5a validation: Sapiens dense on real segmented humans (spin).

`humans` output is dead data downstream (nothing consumes pose/seg/depth/normals
until the P3 graph re-cut), so there is no render/mPSNR gate. This checks the
Sapiens dense rung produces *sane real* outputs on real segmented people:
keypoints land inside the person box, body-part seg has multiple parts, and
depth/normals are well-formed. Runs the actual dense-ladder rung over Mask R-CNN
human instances.

Run (GPU; weights via setup_room4d.sh --with-sapiens):
    python scripts/fp5_validate_humans.py --dataset-root ../data/shape_of_motion/iphone/spin --max-frames 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from room4d.common.types import FrameRef, VideoStreamRef
from room4d.ingest.ingest_artifact import IngestArtifact
from room4d.lifting.geometry_provider import DyCheckGeometryProvider
from room4d.perception.humans.dense_ladder import SapiensDenseRung
from room4d.perception.instance.instance_artifact import InstanceArtifact
from room4d.perception.instance.segment import FloorInstanceSegmenter
from room4d.perception.objects.detector import ObjectDetector
from room4d.perception.objects.segmenter import ObjectSegmenter


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset-root', default='../data/shape_of_motion/iphone/spin')
    ap.add_argument('--max-frames', type=int, default=8)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    prov = DyCheckGeometryProvider(args.dataset_root, factor=1, depth_type='depth_anything_colmap', camera_type='refined')
    rgbs = prov.frame_stack('train', device='cpu').rgbs.cpu().numpy()[: args.max_frames]
    if rgbs.dtype != np.uint8:
        rgbs = np.clip(rgbs * 255.0, 0, 255).astype(np.uint8)
    images = [rgbs[i] for i in range(len(rgbs))]
    n, h, w = rgbs.shape[:3]

    class _Decoder:
        def get_frame(self, stream, idx):
            return images[int(idx)]

    ingest = IngestArtifact(
        run_id='fp5', stage_name='ingest', save_dir=Path('/tmp/fp5'),
        streams=[VideoStreamRef(stream_id='cam_00', video_path=Path('spin'), fps=20.0, num_frames=n, width=w, height=h, has_audio=False)],
    )
    refs = [FrameRef(stream_id='cam_00', frame_idx=i, timestamp_sec=float(i) * 0.05) for i in range(n)]

    print(f"[1] segment {n} spin frames (Mask R-CNN floor)...")
    floor = FloorInstanceSegmenter(_Decoder(), ObjectDetector({}), ObjectSegmenter({}), {})
    obs, masks = floor.run(ingest, refs)
    human_obs = [o for o in obs if o.kind == 'human']
    art = InstanceArtifact(run_id='fp5', stage_name='segment', save_dir=Path('/tmp/fp5/seg'), observations=obs, mask_arrays=masks)
    art.rebuild_indices()
    print(f"    -> {len(human_obs)} human instances across {len({o.track_id for o in human_obs})} tracks")

    print("[2] Sapiens dense (pose + seg + depth + normals)...")
    rung = SapiensDenseRung(_Decoder(), {'device': args.device})
    records = rung.run(ingest, art)

    box_by_inst = {o.instance_id: np.asarray(o.bbox_xyxy, float) for o in human_obs}
    print(f"\n=========== FP5a Sapiens dense validation (backend={rung.backend_used}) ===========")
    for iid in sorted(r for r in records if r in box_by_inst):
        rec = records[iid]
        x1, y1, x2, y2 = box_by_inst[iid]
        kp, seg, depth, nrm = rec.get('keypoints_2d'), rec.get('seg'), rec.get('depth'), rec.get('normals')
        vis = kp[kp[:, 2] > 0.3] if kp is not None else np.zeros((0, 3))
        inside = float(np.mean((vis[:, 0] >= x1 - 5) & (vis[:, 0] <= x2 + 5) & (vis[:, 1] >= y1 - 5) & (vis[:, 1] <= y2 + 5))) if len(vis) else 0.0
        depth_s = '-' if depth is None else f"[{depth.min():.2f},{depth.max():.2f}]"
        nrm_s = '-' if nrm is None else f"{np.linalg.norm(nrm, axis=2).mean():.2f}"
        print(f"  {iid}: kpts={0 if kp is None else kp.shape[0]} vis={len(vis)} in_box={inside:.2f} | "
              f"seg_parts={0 if seg is None else len(np.unique(seg))} | depth={depth_s} | normals_unit={nrm_s}")
    print("=" * 70)


if __name__ == '__main__':
    main()
