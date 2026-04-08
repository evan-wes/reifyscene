"""Schemas for track metadata."""

from dataclasses import dataclass


@dataclass
class TrackSchema:
    """Serializable track description."""
    track_id: str
