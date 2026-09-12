"""Artifact for parsed prompts and generated camera paths."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from reifyscene.common.types import BaseArtifact


@dataclass
class PromptArtifact(BaseArtifact):
    """Store parsed query state and planned camera outputs."""

    parsed_prompt: dict[str, Any] = field(default_factory=dict)
    camera_path: dict[str, Any] = field(default_factory=dict)
