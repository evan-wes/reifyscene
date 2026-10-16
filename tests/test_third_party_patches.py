"""Every third-party patch applies cleanly to its pinned checkout, and applies only once.

These patches are the only edits reifyscene makes to other people's code. The failure this guards
is silent: before the patches lived here they were `sed -i` expressions, and `sed -i` exits 0 when
it matches nothing -- so an upstream that drifted shipped unpatched code that failed much later
and somewhere else. A patch that no longer applies should fail here, at the point where it can be
read and fixed.

Skips when a checkout is absent: `third_party/` is gitignored and populated by the installers, so
a clone without it is normal rather than broken.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PATCHES = REPO / "third_party_patches"

# Patch directory -> the checkout it applies to. A tool here with no checkout is skipped.
CHECKOUTS = {
    "shape-of-motion": "shape-of-motion",
    "hloc": "Hierarchical-Localization",
    "vggt-long": "VGGT-Long",
    "concept-graphs": "concept-graphs",
    "vla-3d": "VLA-3D",
}


def _patch_dirs() -> list[str]:
    return sorted(p.name for p in PATCHES.iterdir() if p.is_dir()) if PATCHES.is_dir() else []


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], capture_output=True, text=True)


def test_every_patch_directory_has_a_known_checkout() -> None:
    """A new tool must be registered here, so it cannot be added without a test covering it."""
    assert set(_patch_dirs()) <= set(CHECKOUTS), (
        f"unregistered patch directories: {set(_patch_dirs()) - set(CHECKOUTS)}"
    )


def test_every_patch_carries_an_explanation() -> None:
    """The header before the diff is where the reason lives; a bare diff loses it."""
    for name in _patch_dirs():
        for patch in sorted((PATCHES / name).glob("*.patch")):
            head = patch.read_text().split("diff --git", 1)[0].strip()
            assert len(head.splitlines()) >= 3, f"{patch.relative_to(REPO)} has no header"
            assert "Upstream pin:" in head, f"{patch.relative_to(REPO)} names no pinned commit"


@pytest.mark.parametrize("tool", sorted(CHECKOUTS))
def test_patches_apply_to_a_clean_pinned_tree(tool: str, tmp_path: Path) -> None:
    """Check against a pristine worktree, not the live one, which is already patched."""
    checkout = REPO / "third_party" / CHECKOUTS[tool]
    patch_dir = PATCHES / tool
    if not (checkout / ".git").exists():
        pytest.skip(f"{CHECKOUTS[tool]} is not checked out")
    if not patch_dir.is_dir():
        pytest.skip(f"no patches for {tool}")

    worktree = tmp_path / tool
    add = _git("-C", str(checkout), "worktree", "add", "--quiet", "--detach", str(worktree), "HEAD")
    assert add.returncode == 0, add.stderr
    try:
        for patch in sorted(patch_dir.glob("*.patch")):
            forward = _git("-C", str(worktree), "apply", "--check", str(patch))
            assert forward.returncode == 0, (
                f"{patch.name} does not apply to a clean {tool} at its pin:\n{forward.stderr}"
            )
            applied = _git("-C", str(worktree), "apply", str(patch))
            assert applied.returncode == 0, applied.stderr
            # Applying twice must be refused, which is what makes re-running an installer safe.
            again = _git("-C", str(worktree), "apply", "--check", str(patch))
            assert again.returncode != 0, f"{patch.name} applies twice; it is not idempotent"
            reverse = _git("-C", str(worktree), "apply", "--reverse", "--check", str(patch))
            assert reverse.returncode == 0, f"{patch.name} is not detectable as already applied"
    finally:
        _git("-C", str(checkout), "worktree", "remove", "--force", str(worktree))
        _git("-C", str(checkout), "worktree", "prune")
