"""A graph entity's node id: how it is built and read, in one place.

``<kind>:<stream id>:<track id>`` -- ``human:cam_00:sam2:cam_00:2`` is the human tracked as
``sam2:cam_00:2`` in stream ``cam_00``. A track id carries colons of its own, so nothing else
splits a node id by hand: the short name people refer to an entity by (``2``) is
:func:`tracklet_number`.
"""

from __future__ import annotations


def entity_node_id(kind: str, stream_id: str, track_id: str) -> str:
    """The node id of the ``kind`` entity tracked as ``track_id`` in ``stream_id``."""
    return f"{kind}:{stream_id}:{track_id}"


def tracklet_number(node_id: str) -> str:
    """The trailing tracklet number of a node id (``"2"`` for ``...:sam2:cam_00:2``): the
    short name an entity is referred to by."""
    return str(node_id).rsplit(":", 1)[-1]
