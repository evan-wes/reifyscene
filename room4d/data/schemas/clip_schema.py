"""Schemas for clip metadata."""

from dataclasses import dataclass


@dataclass
class ClipSchema:
    """Serializable clip description."""
    clip_id: str
