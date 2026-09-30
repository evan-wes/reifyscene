"""Score a ConceptGraphs run against DAVIS with the same scorer the inventory uses.

    python scripts/benchmark_conceptgraphs.py \
        --export ../runs/cg_export --out benchmarks/davis_conceptgraphs.json

Reads each sequence's fused objects (``pcd_saves/full_pcd_*.pkl.gz``), turns them back
into per-frame instance masks, and puts them through ``score_masks_by_frame`` -- the same
function ``benchmark_inventory.py`` calls. That shared path is the point: two systems
scored by two functions cannot be compared, however similar the functions look.

Read the result with its caveat attached. ConceptGraphs fuses per-frame masks into one
world point cloud per object and associates by 3D overlap, which assumes a static scene;
DAVIS annotates exactly the objects that move. ``--split-by-motion`` reports static and
moving ground-truth objects separately so that assumption is visible in the numbers
rather than buried in the total.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from reifyscene.eval.benchmarks.inventory.score import score_masks_by_frame
from reifyscene.eval.reference.conceptgraphs import masks_by_frame

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO.parent / "runs"


def _objects_file(sequence_dir: Path) -> Path | None:
    """ConceptGraphs names its output by the detector variant and a run suffix."""
    candidates = sorted((sequence_dir / "pcd_saves").glob("full_pcd_*.pkl.gz"))
    # Prefer the post-processed cloud when both exist; it is the pipeline's final answer.
    post = [p for p in candidates if p.name.endswith("_post.pkl.gz")]
    return (post or candidates or [None])[0]


def _gt_motion(run_dir: Path, frames: list[int], stream_id: str | None = None
               ) -> dict[int, float]:
    """Per ground-truth id, how far it travels **in the world**, in its own 3D extents.

    Image-space centroid travel cannot answer this: a camera that tracks a walking subject
    keeps it centred, so the object looks static exactly when it is not. (An earlier version
    of this function made that mistake and called DAVIS's walking bear static.) The world
    position comes from the same pointmaps and the same recorded image transform the entity
    geometry uses, so camera motion is already accounted for.

    Returned as a covariate for interpreting the score, never to exclude anything. Objects
    whose world position cannot be recovered are absent from the result rather than assumed
    static.
    """
    from reifyscene.eval.benchmarks.inventory.score import stream_ground_truth
    from reifyscene.ingest.streams import RunStreams, parse_key
    from reifyscene.lifting.entity_geometry import entity_positions_from_pointmaps

    geometry = run_dir / "geometry"
    try:
        stream = RunStreams.from_run(run_dir).resolve(stream_id)
        transform = stream.transform
        art = json.loads((geometry / "artifact.json").read_text())
    except Exception:
        return {}
    # the pointmap files are the ones geometry's own index names
    pointmap_of = {parse_key(k)[1]: v for k, v in (art.get("point_index") or {}).items()
                   if parse_key(k)[0] == stream.id}

    def pointmap_for_frame(frame_idx: int):
        name = pointmap_of.get(int(frame_idx))
        path = geometry / "pointmaps" / name if name else None
        if path is None or not path.exists():
            return None
        payload = np.load(path)
        return np.asarray(payload["points"]), payload.get("confidence")

    ground_truth = stream_ground_truth(stream)
    masks_by_frame: dict[int, dict[str, np.ndarray]] = {}
    for frame in frames:
        for gid, mask in ground_truth(frame).items():
            masks_by_frame.setdefault(frame, {})[str(gid)] = mask
    if not masks_by_frame:
        return {}

    placed = entity_positions_from_pointmaps(
        masks_by_frame, pointmap_for_frame=pointmap_for_frame, transform=transform
    )
    out: dict[int, float] = {}
    for gid, rows in placed.items():
        if len(rows) < 2:
            continue
        xyz = np.asarray([r["xyz"] for r in rows], dtype=float)
        extent = float(np.median([np.linalg.norm(r["extent"]) for r in rows]))
        if not np.isfinite(extent) or extent <= 1e-9:
            continue
        span = float(np.linalg.norm(xyz.max(axis=0) - xyz.min(axis=0)))
        out[int(gid)] = span / extent
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--export", type=Path, required=True, help="the export_for_conceptgraphs dir")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--match-iou", type=float, default=0.5)
    ap.add_argument("--min-detections", type=int, default=3,
                    help="ConceptGraphs' own obj_min_detections default")
    ap.add_argument("--split-by-motion", action="store_true", default=True)
    ap.add_argument("--runs", type=Path, default=RUNS,
                    help="where the exported runs live: each export's sequence is matched to "
                         "the run whose input is that DAVIS sequence, for its ground truth and "
                         "world geometry")
    ap.add_argument("--motion-threshold", type=float, default=1.0,
                    help="world travel, in the object's own 3D extents, above which it is "
                         "'moving'. One extent means it moved its own length.")
    args = ap.parse_args()

    from reifyscene.eval.benchmarks.inventory.score import stream_ground_truth
    from reifyscene.ingest.streams import RunStreams, StreamError

    candidates: dict[str, list[Path]] = {}   # DAVIS sequence -> runs built from it
    for r in sorted(args.runs.iterdir()):
        try:
            src = RunStreams.from_run(r).only().source
        except (FileNotFoundError, NotADirectoryError, StreamError, KeyError):
            continue
        if src.kind == "davis" and (r / "geometry").is_dir():
            candidates.setdefault(src.sequence, []).append(r)

    def run_for(manifest: dict[str, Any], sequence: str) -> Path | None:
        """The run an export came from: as its manifest records, else the one run built from
        that sequence -- never a pick among several."""
        if manifest.get("run_dir"):
            return Path(manifest["run_dir"])
        found = candidates.get(sequence, [])
        return found[0] if len(found) == 1 else None

    rows: list[dict[str, Any]] = []
    for manifest_path in sorted(args.export.glob("*/manifest.json")):
        sequence_dir = manifest_path.parent
        sequence = sequence_dir.name
        objects_path = _objects_file(sequence_dir)
        if objects_path is None:
            rows.append({"sequence": sequence, "notes": ["no ConceptGraphs output"]})
            print(f"  {sequence:24s} -- no output")
            continue
        by_frame, info = masks_by_frame(
            objects_path, manifest_path, min_detections=args.min_detections
        )
        run = run_for(json.loads(manifest_path.read_text()), sequence)
        if run is None:
            n = len(candidates.get(sequence, []))
            why = ("no run whose input is this sequence" if not n else
                   f"{n} runs were built from this sequence and the export does not record "
                   "which it came from (re-export it)")
            rows.append({"sequence": sequence, "notes": [why]})
            print(f"  {sequence:24s} -- {why}")
            continue
        score = score_masks_by_frame(
            by_frame, stream_ground_truth(RunStreams.from_run(run).only()), sequence=sequence,
            match_iou=args.match_iou,
        )
        row = score.to_dict()
        row["conceptgraphs"] = info
        if args.split_by_motion:
            motion = _gt_motion(run, sorted(by_frame))
            matched = {m["gt_id"] for m in score.matches}
            moving = {g for g, t in motion.items() if t >= args.motion_threshold}
            row["motion"] = {
                "travel_by_gt_id": {str(k): round(v, 3) for k, v in sorted(motion.items())},
                "static_total": len(motion) - len(moving),
                "static_recovered": len((set(motion) - moving) & matched),
                "moving_total": len(moving),
                "moving_recovered": len(moving & matched),
            }
        rows.append(row)
        print(f"  {sequence:24s} gt {score.num_gt:2d}  recovered {score.recovered:2d}  "
              f"objects {info['objects_kept']:3d}  recall {score.recall:.3f}")

    scored = [r for r in rows if r.get("num_gt")]
    total_gt = sum(r["num_gt"] for r in scored)
    total_rec = sum(r["recovered"] for r in scored)
    summary = {
        "generated": datetime.now(UTC).isoformat(),
        "reference": "ConceptGraphs",
        "match_iou": args.match_iou,
        "min_detections": args.min_detections,
        "num_sequences": len(rows),
        "num_scored": len(scored),
        "total_gt": total_gt,
        "total_recovered": total_rec,
        "recall": round(total_rec / total_gt, 4) if total_gt else 0.0,
        "caveat": (
            "ConceptGraphs assumes a static scene: it fuses per-frame masks into one world "
            "point cloud per object and associates by 3D overlap. DAVIS annotates the objects "
            "that move. Read the motion split before reading the total."
        ),
        "sequences": rows,
    }
    if args.split_by_motion:
        summary["static_recall"] = _ratio(rows, "static")
        summary["moving_recall"] = _ratio(rows, "moving")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"\nrecall {summary['recall']:.3f} over {total_gt} objects -> {args.out}")
    if args.split_by_motion:
        print(f"  static {summary['static_recall']}   moving {summary['moving_recall']}")


def _ratio(rows: list[dict[str, Any]], kind: str) -> float:
    total = sum((r.get("motion") or {}).get(f"{kind}_total", 0) for r in rows)
    got = sum((r.get("motion") or {}).get(f"{kind}_recovered", 0) for r in rows)
    return round(got / total, 4) if total else 0.0


if __name__ == "__main__":
    main()
