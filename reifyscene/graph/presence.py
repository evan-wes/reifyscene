"""When an entity is in the scene: its span, the frames it was seen, and the gaps inside.

The graph contract (§3.3 of ``docs/design/2026-09-12-back-half-ladders-and-graph-contract.md``):
presence spans first to last observation, and **interior gaps are occlusion candidates, not the
entity ceasing to exist**. A gap carries no claim about where the entity was.

Short gaps are usually a segmenter missing a frame or two rather than anything passing in front,
so each gap is labelled by length: a *dropout* (under ``occlusion_min`` frames) or an *occlusion
candidate*. Both are reported; neither ends the entity.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any

OCCLUSION_MIN_FRAMES = 3


@dataclass
class Gap:
    start: int          # first missing frame
    end: int            # last missing frame
    kind: str           # "dropout" | "occlusion_candidate"

    @property
    def length(self) -> int:
        return self.end - self.start + 1


@dataclass
class Presence:
    first: int | None
    last: int | None
    observed: list[int] = field(default_factory=list)
    gaps: list[Gap] = field(default_factory=list)

    @property
    def span(self) -> tuple[int, int] | None:
        return None if self.first is None else (self.first, self.last)

    def present_at(self, frame: int) -> bool:
        """Observed on that frame (not merely inside the span)."""
        return frame in set(self.observed)

    def in_span(self, frame: int) -> bool:
        return self.first is not None and self.first <= frame <= self.last

    def to_dict(self) -> dict[str, Any]:
        return {
            "first": self.first,
            "last": self.last,
            "num_observed": len(self.observed),
            "observed": list(self.observed),
            "gaps": [asdict(g) | {"length": g.length} for g in self.gaps],
        }


def presence_from_frames(
    frames: Iterable[int], *, occlusion_min: int = OCCLUSION_MIN_FRAMES
) -> Presence:
    """Presence from the frames an entity was observed on."""
    obs = sorted({int(f) for f in frames})
    if not obs:
        return Presence(first=None, last=None)
    gaps = []
    for a, b in zip(obs, obs[1:], strict=False):
        if b - a > 1:
            n = b - a - 1
            kind = "occlusion_candidate" if n >= occlusion_min else "dropout"
            gaps.append(Gap(a + 1, b - 1, kind))
    return Presence(first=obs[0], last=obs[-1], observed=obs, gaps=gaps)
