"""A run's outputs: what a run folder holds, and the only code that knows how it is laid out.

The output counterpart to :class:`~reifyscene.ingest.streams.RunStreams`. Nothing else joins a
run folder with a stage name, names an artifact or checkpoint file, or decides what a run is:

- **A run is recognised by its manifest** (``run_manifest.json``, written when the run starts),
  never by its folder's name. :meth:`RunOutputs.discover` finds every run under a folder.
- **A stage's artifact** is loaded with :meth:`RunOutputs.artifact`, through the stage
  registry, which owns each stage's folder and artifact class; :meth:`RunOutputs.record` is its
  saved record alone (metadata, indexes) for a caller that needs no arrays.
  :meth:`RunOutputs.stage_dir` is where a stage writes.
- **A trained checkpoint** is a :class:`Checkpoint`: the render stage's own (``render``), or a
  retrain beside it, each a folder holding its weights. :meth:`RunOutputs.checkpoint` names one;
  nothing picks between them by folder name.
- **Everything else a run accumulates** -- identity maps, edits, evaluations -- has its place
  here (:attr:`RunOutputs.identity_dir`, :meth:`RunOutputs.output_dir`).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any


class NotARun(ValueError):
    """The folder is not a run (it has no run manifest), or lacks what was asked of it."""


@dataclass(frozen=True)
class Checkpoint:
    """One trained Gaussian scene of a run: a folder holding its weights and the render record
    that says what they are (``scene_state``: the weights file, the scene's kind, its fit)."""

    run: RunOutputs
    name: str                     # the folder: "render" for the render stage's own

    @property
    def dir(self) -> Path:
        return self.run.root / self.name

    @property
    def _record_path(self) -> Path:
        from reifyscene.common.types import BaseArtifact

        return self.dir / BaseArtifact.FILE

    @property
    def state(self) -> dict[str, Any]:
        """The render record's ``scene_state``."""
        return self.record()["scene_state"]

    @property
    def kind(self) -> str:
        """``dynamic`` or ``static``, as the render recorded it."""
        return str(self.state["kind"])

    @property
    def weights(self) -> Path:
        """The weights file the render recorded writing."""
        return self.dir / str(self.state["checkpoint"])

    def exists(self) -> bool:
        """A render record (it has a ``scene_state``) that names its weights, and the weights
        it names. Any other stage's folder is not a checkpoint."""
        if not self._record_path.is_file():
            return False
        state = self.record().get("scene_state") or {}
        return bool(state.get("checkpoint")) and (self.dir / str(state["checkpoint"])).is_file()

    def load(self) -> dict[str, Any]:
        """The weights (``np.load`` of the checkpoint, materialised)."""
        import numpy as np

        if not self.exists():
            raise NotARun(f"{self.run.name}: no checkpoint {self.name!r} ({self.weights})")
        with np.load(self.weights, allow_pickle=True) as z:
            return {k: z[k] for k in z.files}

    def record(self) -> dict[str, Any]:
        """The render record saved with the weights (its ``scene_state``: what was trained,
        on what, and how well it fits)."""
        from reifyscene.common.io import read_json

        if not self._record_path.is_file():
            raise NotARun(f"{self.run.name}: checkpoint {self.name!r} has no record "
                          f"({self._record_path})")
        return read_json(self._record_path)

    def digest(self) -> str:
        """Content digest of the weights (tens of MB: a fraction of a second)."""
        h = hashlib.sha256()
        with open(self.weights, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()[:16]


class RunOutputs:
    """One run folder's outputs."""

    MANIFEST = "run_manifest.json"
    RENDER = "render"             # the render stage's own checkpoint

    def __init__(self, root: str | Path) -> None:
        """Use :meth:`open` (an existing run) or :meth:`create` (a run being written)."""
        self.root = Path(root)
        self._artifacts: dict[str, Any] = {}
        self._records: dict[str, dict[str, Any]] = {}

    # -- finding runs ----------------------------------------------------------------------------

    @classmethod
    def is_run(cls, path: str | Path) -> bool:
        return (Path(path) / cls.MANIFEST).is_file()

    @classmethod
    def open(cls, path: str | Path) -> RunOutputs:
        if not cls.is_run(path):
            raise NotARun(f"{path}: not a run (no {cls.MANIFEST})")
        return cls(path)

    @classmethod
    def create(cls, path: str | Path) -> RunOutputs:
        """The outputs of a run being written: its folder exists, its manifest may not yet."""
        Path(path).mkdir(parents=True, exist_ok=True)
        return cls(path)

    @classmethod
    def discover(cls, root: str | Path) -> list[RunOutputs]:
        """Every run directly under ``root``, by name."""
        root = Path(root)
        return [cls(p) for p in sorted(root.iterdir()) if p.is_dir() and cls.is_run(p)]

    @property
    def name(self) -> str:
        return self.root.name

    # -- the manifest ------------------------------------------------------------------------------

    @cached_property
    def manifest(self) -> dict[str, Any]:
        return json.loads((self.root / self.MANIFEST).read_text())

    @property
    def config(self) -> dict[str, Any]:
        """The configuration the run was built with."""
        return self.manifest["config"]

    def write_manifest(self, manifest: dict[str, Any]) -> Path:
        from reifyscene.common.io import write_json

        out = self.root / self.MANIFEST
        write_json(out, manifest)
        self.__dict__.pop("manifest", None)
        return out

    # -- stages ------------------------------------------------------------------------------------

    @staticmethod
    def _spec(stage: str):
        from reifyscene.api.stages import build_stage_registry

        registry = build_stage_registry()
        if stage not in registry:
            raise KeyError(f"no stage {stage!r}; stages: {', '.join(registry)}")
        return registry[stage]

    def stage_dir(self, stage: str) -> Path:
        """Where ``stage`` keeps its artifact (not created)."""
        return self.root / self._spec(stage).dir_name

    def has(self, stage: str) -> bool:
        """Whether the run holds ``stage``'s artifact."""
        from reifyscene.common.types import BaseArtifact

        return (self.stage_dir(stage) / BaseArtifact.FILE).is_file()

    def record(self, stage: str) -> dict[str, Any]:
        """``stage``'s saved record (its ``artifact.json``) -- metadata and indexes, none of the
        arrays a full :meth:`artifact` load reads."""
        from reifyscene.common.io import read_json
        from reifyscene.common.types import BaseArtifact

        if stage not in self._records:
            if not self.has(stage):
                raise NotARun(f"{self.name}: no {stage} artifact")
            self._records[stage] = read_json(self.stage_dir(stage) / BaseArtifact.FILE)
        return self._records[stage]

    def artifact(self, stage: str) -> Any:
        """``stage``'s artifact, loaded by its own class (once per :class:`RunOutputs`)."""
        if stage not in self._artifacts:
            if not self.has(stage):
                raise NotARun(f"{self.name}: no {stage} artifact")
            self._artifacts[stage] = self._spec(stage).load_fn(self.stage_dir(stage))
        return self._artifacts[stage]

    # -- checkpoints -------------------------------------------------------------------------------

    def checkpoint(self, name: str = RENDER) -> Checkpoint:
        return Checkpoint(self, name)

    def checkpoints(self) -> list[Checkpoint]:
        """Every trained checkpoint the run holds: the render stage's first, then retrains."""
        found = [c for c in (Checkpoint(self, p.name) for p in sorted(self.root.iterdir())
                             if p.is_dir()) if c.exists()]
        return sorted(found, key=lambda c: (c.name != self.RENDER, c.name))

    # -- what accumulates beside the stages --------------------------------------------------------

    @property
    def calibration_dir(self) -> Path:
        """Where a static-frame calibration of the run's input is kept (not a pipeline stage)."""
        return self.root / "calibration"

    @property
    def identity_dir(self) -> Path:
        """The per-frame entity-id maps (``reifyscene.graph.id_maps``)."""
        return self.root / "identity"

    def output_dir(self, kind: str, *parts: str) -> Path:
        """Where an edit or an evaluation of this run writes (``edits/<kind>/...``; not
        created)."""
        return self.root.joinpath("edits", kind, *parts)

    @cached_property
    def streams(self):
        """The run's input streams."""
        from reifyscene.ingest.streams import RunStreams

        return RunStreams.from_run(self.root)

    def __repr__(self) -> str:
        return f"RunOutputs({str(self.root)!r})"
