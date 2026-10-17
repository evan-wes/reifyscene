"""Export DAVIS runs as posed RGB-D for the ConceptGraphs reference comparison.

    python scripts/export_for_conceptgraphs.py
    python scripts/export_for_conceptgraphs.py --sequences bear lab-coat

ConceptGraphs takes posed RGB-D; DAVIS ships neither depth nor poses, so it gets VGGT's.
Each sequence is written in the on-disk layout ConceptGraphs' Replica loader already
reads, which is what lets the reference tool run completely unmodified -- a reference
whose code we edited is not a reference.

The export is scored later by the same scorer our own inventory goes through, so the
comparison is like-for-like. ``summary.json`` carries the per-sequence scale factor and
pinhole residual; a sequence whose residual is high is one whose geometry the reference
tool cannot be blamed for.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

from reifyscene.eval.reference.conceptgraphs import export_run
from reifyscene.eval.reference.conceptgraphs.export import REFERENCE_DIAGONAL_M


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="conceptgraphs",
                    help="where to write sequences: a name in the data directory's reference/")
    ap.add_argument("--sequences", nargs="*",
                    help="DAVIS sequence names; default every run in runs/ whose input is "
                         "a DAVIS sequence (recognised from what ingest recorded)")
    ap.add_argument("--stage", default="geometry",
                    help="geometry (default) or pose_refine. pose_refine copies geometry's "
                         "pointmaps but replaces its poses, so the two disagree -- its "
                         "pinhole residual is ~14%% against geometry's 0.4-3.5%%")
    ap.add_argument("--reference-diagonal-m", type=float, default=REFERENCE_DIAGONAL_M)
    ap.add_argument("--limit", type=int, default=0, help="stop after N sequences (a smoke test)")
    args = ap.parse_args()

    from reifyscene.api.run_outputs import RunOutputs
    from reifyscene.common import data_dir
    from reifyscene.ingest.streams import StreamError

    out = data_dir.path(data_dir.REFERENCE, args.out)

    def davis_sequence(run: RunOutputs) -> str | None:
        try:
            src = run.streams.only().source
        except StreamError:             # several streams: not one DAVIS sequence
            return None
        return src.sequence if src.kind == "davis" else None

    found = {r: davis_sequence(r) for r in RunOutputs.discover(data_dir.area(data_dir.RUNS))
             if r.has("ingest") and r.has(args.stage)}
    runs = [p for p, seq in found.items()
            if seq is not None and (not args.sequences or seq in args.sequences)]
    if args.limit:
        runs = runs[: args.limit]

    out.mkdir(parents=True, exist_ok=True)
    rows, failed = [], []
    for run in runs:
        sequence = found[run]
        try:
            result = export_run(
                run, out / sequence,
                stage=args.stage, reference_diagonal_m=args.reference_diagonal_m,
            )
        except Exception as exc:  # one bad run must not lose the other eighty-nine
            failed.append({"sequence": sequence, "error": f"{type(exc).__name__}: {exc}"})
            print(f"  {sequence:24s} FAILED {type(exc).__name__}: {exc}")
            continue
        rows.append(result.to_dict())
        print(f"  {sequence:24s} {result.num_frames:4d} frames  "
              f"scale {result.scale_factor:7.3f}  "
              f"pinhole {result.pinhole_residual_pct:5.2f}%  "
              f"clipped {result.depth_clipped_pct:.2f}%")

    summary = {
        "generated": datetime.now(UTC).isoformat(),
        "stage": args.stage,
        "reference_diagonal_m": args.reference_diagonal_m,
        "num_exported": len(rows),
        "num_failed": len(failed),
        "sequences": rows,
        "failed": failed,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n{len(rows)} exported, {len(failed)} failed -> reference/{args.out}/summary.json")


if __name__ == "__main__":
    main()
