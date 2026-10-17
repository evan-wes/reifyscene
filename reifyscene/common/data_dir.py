"""The data directory: where everything the pipeline reads and writes lives.

One setting names it, the ``REIFYSCENE_DATA_DIR`` environment variable, and this module is the
only place that reads it. Nothing in the repository names a folder on any machine: every input,
run, checkpoint and cache is found by its name inside this directory. A record stores those
names, never a full path, so a whole data directory can be moved or renamed and every run in it
still opens.

The layout is fixed::

    datasets/     inputs, each dataset laid out as it ships (davis2017/, dycheck_iphone/,
                  nvidia_dynamic/, panoptic/), and any video or frame folder of your own
    checkpoints/  model weights the pipeline fetches (tapnet/)
    runs/         pipeline runs
    reference/    what reference tools produce (Shape-of-Motion runs and motion inits)
    cache/        inputs a stage derives and rebuilds when missing (VGGT inputs, COLMAP work)
    archive/      kept, read by nothing

Two kinds of name, both relative paths with ``/`` separators:

- an **area name** says where a thing is inside one area, as a command line takes it
  (``--run-dir lab-coat`` is ``runs/lab-coat``);
- a **record name** says where it is inside the data directory, as a record stores it
  (``datasets/davis2017/.../lab-coat``).

An absolute path, or a name that climbs out with ``..``, is refused: nothing points outside the
data directory.
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath

ENV = "REIFYSCENE_DATA_DIR"

DATASETS = "datasets"
CHECKPOINTS = "checkpoints"
RUNS = "runs"
REFERENCE = "reference"
CACHE = "cache"
ARCHIVE = "archive"
AREAS = (DATASETS, CHECKPOINTS, RUNS, REFERENCE, CACHE, ARCHIVE)


class DataDirError(RuntimeError):
    """The data directory is not set, or a name does not stay inside it."""


def root() -> Path:
    """The data directory, as ``REIFYSCENE_DATA_DIR`` names it (absolute, existing)."""
    raw = os.environ.get(ENV, "").strip()
    if not raw:
        raise DataDirError(
            f"{ENV} is not set: point it at your data directory (the README describes its layout)")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise DataDirError(f"{ENV} must be an absolute path, not {raw!r}")
    if not path.is_dir():
        raise DataDirError(f"{ENV} is {raw!r}, which is not a directory")
    return path


def area(name: str) -> Path:
    """One of the data directory's areas (:data:`AREAS`)."""
    if name not in AREAS:
        raise ValueError(f"{name!r} is not an area of the data directory ({', '.join(AREAS)})")
    return root() / name


def _relative(name: str | os.PathLike[str], what: str) -> PurePosixPath:
    text = os.fspath(name).strip()
    rel = PurePosixPath(text)
    if not text or text == ".":
        raise DataDirError(f"an empty {what}")
    if rel.is_absolute() or text.startswith("~"):
        raise DataDirError(
            f"{text!r} is a path on this machine: give its {what} inside the data directory "
            f"instead ({ENV})")
    if ".." in rel.parts:
        raise DataDirError(f"{text!r} climbs out with '..': a {what} stays inside its folder")
    return rel


def path(area_name: str, name: str | os.PathLike[str]) -> Path:
    """The file or folder an area name names (``path(RUNS, "lab-coat")``)."""
    return area(area_name) / _relative(name, "name")


def from_record(name: str | os.PathLike[str]) -> Path:
    """The file or folder a record name stands for."""
    return root() / _relative(name, "record name")


def record_name(target: str | os.PathLike[str]) -> str:
    """The record name of a file or folder inside the data directory.

    Symbolic links are not followed: a name is where something sits in the data directory, not
    where a link inside it happens to point.
    """
    base = Path(os.path.abspath(root()))
    full = Path(os.path.abspath(target))
    try:
        return full.relative_to(base).as_posix()
    except ValueError:
        raise DataDirError(
            f"{full} is outside the data directory {base}: a record names only what is inside "
            "it") from None
