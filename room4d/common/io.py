"""Simple JSON and YAML IO helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def read_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file into a Python dictionary."""
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def write_yaml(path: str | Path, data: dict[str, Any]) -> None:
    """Write a dictionary to a YAML file."""
    Path(path).write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON file into a Python dictionary."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    """Write a dictionary to a JSON file."""
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
