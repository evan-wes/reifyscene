"""Enum definitions for shared pipeline concepts."""

from enum import Enum


class NodeType(str, Enum):
    """Supported scene-graph node categories."""

    CAMERA = "camera"
    HUMAN = "human"
    OBJECT = "object"
    REGION = "region"
    UNKNOWN = "unknown"


class EdgeType(str, Enum):
    """Supported scene-graph edge categories."""

    SAME_INSTANCE = "same_instance"
    OBSERVED_BY = "observed_by"
    INTERACTING_WITH = "interacting_with"
    ON_TOP_OF = "on_top_of"
    INSIDE = "inside"
    ATTACHED_TO = "attached_to"
    OCCLUDES = "occludes"
    # VLA-3D's view-independent predicates (reifyscene/graph/relations.py)
    ABOVE = "above"
    BELOW = "below"
    ON = "on"
    IN = "in"
    NEAR = "near"
    BETWEEN = "between"
    CLOSEST = "closest"
    FARTHEST = "farthest"
