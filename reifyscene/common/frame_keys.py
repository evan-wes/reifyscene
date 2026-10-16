"""The frame-key format: ``"<stream>:<frame>"``, the one way views, cameras and per-frame
artifacts name a frame. Built and parsed only here."""

from __future__ import annotations


def frame_key(stream_id: str, frame: int) -> str:
    return f"{stream_id}:{int(frame)}"


def parse_key(key: str) -> tuple[str, int]:
    """``"cam_00:12"`` -> ``("cam_00", 12)``. The stream id may itself contain colons."""
    stream, sep, frame = str(key).rpartition(":")
    if not sep or not stream:
        raise ValueError(f"not a frame key: {key!r} (expected '<stream>:<frame>')")
    return stream, int(frame)
