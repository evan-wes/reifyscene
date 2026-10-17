"""Adapters that run published reference implementations on this pipeline's data.

A benchmark number with nothing to compare against says only that it is reproducible.
These adapters export our artifacts into a reference tool's own input convention, so a
published system can be run on the same footage and scored by the same scorer -- the
pattern ``setup_reference_tool.sh`` established for geometry, applied to the graph.
"""
