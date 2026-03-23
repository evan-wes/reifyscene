"""Global time alignment solver."""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path

from room4d.sync.sync_artifact import SyncArtifact


class GlobalTimeAlignmentSolver:
    """Convert pairwise offsets into globally consistent stream offsets.

    TODO implementation order:
    1. Replace BFS propagation with a weighted least-squares graph solve.
    2. Add support for dropped frames and piecewise-linear time warps.
    3. Add residual diagnostics against motion and epipolar consistency.
    """

    def __init__(self, min_confidence: float = 0.0) -> None:
        """Configure the threshold used when ingesting pairwise edges."""
        self.min_confidence = min_confidence

    def solve_pairwise_offsets(
        self,
        pairwise_offsets: dict[tuple[str, str], tuple[float, float]],
        *,
        run_id: str = "dev",
        save_dir: str | Path = ".",
        anchor_stream_id: str | None = None,
    ) -> SyncArtifact:
        """Produce globally consistent per-stream offsets from pairwise estimates.

        Parameters
        ----------
        pairwise_offsets:
            Mapping ``(stream_a, stream_b) -> (offset_sec, confidence)`` with the
            convention ``t_b ~= t_a + offset_sec``.
        anchor_stream_id:
            Optional stream fixed to offset 0.0. If omitted, the first stream in
            sorted order becomes the anchor.
        """
        all_streams: set[str] = set()
        graph: dict[str, list[tuple[str, float, float]]] = defaultdict(list)
        serial_pairwise: dict[str, dict[str, float]] = {}
        confidence_by_pair: dict[str, float] = {}

        for (stream_a, stream_b), (offset_sec, confidence) in pairwise_offsets.items():
            all_streams.update((stream_a, stream_b))
            if confidence < self.min_confidence:
                continue
            graph[stream_a].append((stream_b, offset_sec, confidence))
            graph[stream_b].append((stream_a, -offset_sec, confidence))
            pair_key = f"{stream_a}|{stream_b}"
            serial_pairwise[pair_key] = {"offset_sec": float(offset_sec), "confidence": float(confidence)}
            confidence_by_pair[pair_key] = float(confidence)

        if not all_streams:
            return SyncArtifact(
                run_id=run_id,
                stage_name="sync",
                save_dir=Path(save_dir),
                offsets_by_stream={},
                confidence_by_stream={},
                pairwise_offsets={},
                confidence_by_pair={},
                anchor_stream_id=anchor_stream_id,
                notes=["No pairwise offsets were provided."],
            )

        anchor = anchor_stream_id or sorted(all_streams)[0]
        offsets_by_stream: dict[str, float] = {anchor: 0.0}
        confidence_by_stream: dict[str, float] = {anchor: 1.0}
        queue: deque[str] = deque([anchor])
        visited = {anchor}
        notes: list[str] = []

        while queue:
            current = queue.popleft()
            current_offset = offsets_by_stream[current]
            for neighbor, delta, confidence in graph[current]:
                if neighbor in visited:
                    continue
                offsets_by_stream[neighbor] = current_offset + delta
                confidence_by_stream[neighbor] = confidence
                visited.add(neighbor)
                queue.append(neighbor)

        disconnected = sorted(all_streams - visited)
        for stream_id in disconnected:
            offsets_by_stream[stream_id] = 0.0
            confidence_by_stream[stream_id] = 0.0
        if disconnected:
            notes.append(
                "Disconnected streams were anchored to 0.0 because no confident path was available: "
                + ", ".join(disconnected)
            )

        return SyncArtifact(
            run_id=run_id,
            stage_name="sync",
            save_dir=Path(save_dir),
            offsets_by_stream=offsets_by_stream,
            confidence_by_stream=confidence_by_stream,
            pairwise_offsets=serial_pairwise,
            confidence_by_pair=confidence_by_pair,
            anchor_stream_id=anchor,
            notes=notes,
        )
