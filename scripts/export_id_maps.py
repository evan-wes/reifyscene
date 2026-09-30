#!/usr/bin/env python
"""Export per-frame entity id maps for finished runs, and verify each round trip.

    python scripts/export_id_maps.py ../runs/loop_*

Writes ``<run>/identity/frame_NNNNN.png`` (uint16) and ``manifest.json`` -- the targets the
per-Gaussian identity loss trains against (I0 of the identity scope). See
:mod:`reifyscene.graph.id_maps` for the encoding: 0 background, 1..K entities, 65535 ignore.

After writing, every run is read back and compared against the masks the renderable handle
uses; any disagreement off ignore pixels fails the run. The table reports per scene the class
count (which sets the loss's normalisation) and how much of the raster is ignored, and why.

No GPU: this reads segment, reid, graph, geometry and ingest artifacts only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from reifyscene.graph.id_maps import (  # noqa: E402
    export_id_maps,
    load_id_maps,
    round_trip_mismatches,
)
from reifyscene.graph.renderable_handle import render_pixel_masks  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", type=Path, nargs="+")
    ap.add_argument("--stream", default=None, help="stream id (default: the run's only one)")
    args = ap.parse_args()

    header = (f"{'run':<22} {'frames':>6} {'classes':>7} {'entity%':>8} {'overlap%':>9} "
              f"{'unmapped%':>10} {'pad%':>6} {'round trip':>10}")
    print(header)
    print("-" * len(header))
    failed = 0
    for run in args.run_dirs:
        manifest = export_id_maps(run, stream_id=args.stream)
        loaded, maps = load_id_maps(run / "identity")
        source = render_pixel_masks(run, stream_id=args.stream).by_frame
        bad = round_trip_mismatches(loaded, maps, source)
        f = manifest["fractions"]
        print(f"{run.name:<22} {len(manifest['frames']):>6} {manifest['num_classes']:>7} "
              f"{100 * f['entity']:>8.2f} {100 * f['overlap']:>9.3f} "
              f"{100 * f['unmapped']:>10.3f} {100 * f['padding']:>6.2f} "
              f"{'ok' if not bad else 'MISMATCH':>10}")
        for nid, n in sorted(bad.items()):
            print(f"    {nid}: {n} px disagree")
        for note in manifest["notes"]:
            print(f"    note: {note}")
        failed += bool(bad)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
