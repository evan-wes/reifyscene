"""reifyscene package."""

from pathlib import Path

__all__ = ["__version__"]


def _detect_version() -> str:
    """Resolve the package version without keeping a second copy of it.

    The ``VERSION`` file is the single source of truth: ``pyproject.toml`` reads
    it at build time (``[tool.setuptools.dynamic]``) and the release workflow
    reads the same file to name the ``v<version>`` tag. A hardcoded
    ``__version__`` here is a fourth copy that nothing keeps in sync — it sat at
    "0.1.0" for 49 releases.

    Order matters. In a dev/editable checkout the ``VERSION`` file wins, because
    installed metadata is baked at install time and goes stale the moment
    ``VERSION`` is bumped without reinstalling — which is every version-bump PR.
    A built wheel does not ship ``VERSION``, so there we fall back to the dist
    metadata setuptools populated from that same file.
    """
    version_file = Path(__file__).resolve().parent.parent / "VERSION"
    try:
        text = version_file.read_text(encoding="utf-8").strip()
        if text:
            return text
    except OSError:
        pass

    try:
        from importlib.metadata import version

        return version("reifyscene")
    except Exception:
        return "0.0.0+unknown"


__version__ = _detect_version()
