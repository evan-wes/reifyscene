"""Schemas for 4D scene exports."""

from dataclasses import dataclass


@dataclass
class SceneSchema:
    """Serializable scene description."""
    scene_id: str
