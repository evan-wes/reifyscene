"""Run ConceptGraphs on our DAVIS runs and score it against the same ground truth.

ConceptGraphs takes *posed RGB-D*; DAVIS ships neither depth nor poses. This package
exports VGGT's geometry into the on-disk layout ConceptGraphs already reads, and reads
its fused objects back as per-frame masks the inventory scorer can take.
"""

from reifyscene.eval.reference.conceptgraphs.export import (
    ExportedSequence,
    export_run,
    scale_to_reference,
)
from reifyscene.eval.reference.conceptgraphs.read_objects import masks_by_frame

__all__ = ["ExportedSequence", "export_run", "masks_by_frame", "scale_to_reference"]
