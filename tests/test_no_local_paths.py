"""Guard: nothing in the repository names a place on one machine.

Everything the pipeline reads or writes is named inside the data directory
(``reifyscene.common.data_dir``), so no tracked file needs a home folder, a temporary folder,
a folder beside the repository, or anything from one person's own tools. This scans the text of
every file git would commit.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
THIS = Path(__file__).resolve().relative_to(REPO).as_posix()

FORBIDDEN = {
    "a home folder": r"/home/|/Users/|~/",
    "a temporary folder": r"/tmp/",
    "the retired data folder": r"Z_DATA|ZDATA",
    "the retired runs folder": r"e2e_runs|preserved_runs",
    "a folder beside the repository": r"reifyscene_project|room4d_project",
    "a personal tool": r"\.gstack|claude-scripts",
    "a personal email address": r"@gmail\.com",
    "a note kept outside the repository": r"\[\[[a-z0-9]+(?:-[a-z0-9]+)+\]\]",
}
PATTERN = re.compile("|".join(f"(?P<g{i}>{p})" for i, p in enumerate(FORBIDDEN.values())))
WHAT = list(FORBIDDEN)


def _files() -> list[str]:
    out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                         cwd=REPO, check=True, capture_output=True, text=True).stdout
    return [f for f in out.splitlines() if f != THIS and (REPO / f).is_file()]


def _mentions(text: str) -> list[tuple[int, str, str]]:
    found = []
    for lineno, line in enumerate(text.splitlines(), 1):
        for m in PATTERN.finditer(line):
            found.append((lineno, WHAT[int(m.lastgroup[1:])], m.group()))
    return found


def test_no_tracked_file_names_a_place_on_one_machine() -> None:
    problems = []
    for rel in _files():
        data = (REPO / rel).read_bytes()
        if b"\0" in data:
            continue                        # binary
        text = data.decode("utf-8", errors="replace")
        problems += [f"{rel}:{n}: {what} ({hit!r})" for n, what, hit in _mentions(text)]
    assert not problems, "\n".join(problems)


def test_the_guard_sees_each_kind_of_mention() -> None:
    text = ("a = '/home/me/x'\nb = '~/data'\nc = '/tmp/run'\nd = '../Z_DATA/x'\n"
            "e = '../e2e_runs'\nf = 'reifyscene_project'\ng = '~/.gstack/x'\n"
            "h = 'me@gmail.com'\ni = 'see [[some-note]]'\nj = [[0]]\nk = 'tzdata==1'\n")
    lines = {n for n, _, _ in _mentions(text)}
    assert lines == set(range(1, 10))       # [[0]] and tzdata are not mentions
