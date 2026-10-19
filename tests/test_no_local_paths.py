"""Guard: nothing in the repository names a place on one machine, or its owner.

Everything the pipeline reads or writes is named inside the data directory
(``reifyscene.common.data_dir``), so no tracked file needs a home folder, a temporary folder,
a folder beside the repository, a link to a Claude Code session, or anything from one person's
own tools; and the owner is named by their public handle alone. This scans the text of every file
git would commit (this one excepted, since it spells out the patterns).

The owner's full name is not written here: it is held as the SHA-256 of each of its words,
lowercased, and every run of words in a file is compared with it.
"""

from __future__ import annotations

import functools
import hashlib
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
THIS = Path(__file__).resolve().relative_to(REPO).as_posix()

FORBIDDEN = {
    "a home folder": r"/home/|/Users/|[A-Za-z]:\\Users\\|~/",
    "a temporary folder": r"/tmp/",
    "a link to a Claude Code session": r"claude\.ai/code/session",
    "the retired data folder": r"Z_DATA|ZDATA",
    "the retired runs folder": r"e2e_runs|preserved_runs",
    "a folder beside the repository": r"reifyscene_project|room4d_project",
    "a personal tool": r"\.gstack|claude-scripts",
    "a personal email address": r"[\w.+-]*(?i:@gmail\.com)",
    "a note kept outside the repository": r"\[\[[a-z0-9]+(?:-[a-z0-9]+)+\]\]",
}
PATTERN = re.compile("|".join(f"(?P<g{i}>{p})" for i, p in enumerate(FORBIDDEN.values())))
WHAT = list(FORBIDDEN)

# The owner's full name: the SHA-256 of each word, lowercased, in order.
NAME = (
    "ae74f72d212fb9871302a2459aeaf7b20bc2f792e4852be648a7d4e63967d9b1",
    "2be4e33aa3e21f6ff7ad144f8fd58228a1cce3b50340b4805ff0c63480eec488",
)
_WORD = re.compile(r"[^\W\d_]+")


@functools.cache
def _digest(word: str) -> str:
    return hashlib.sha256(word.lower().encode()).hexdigest()


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


def _names(text: str, name: tuple[str, ...] = NAME) -> list[int]:
    """The line of each place the name's words appear in order, split by anything that is not a
    letter (a space, a line break, a dot), so a name wrapped across lines or written into an
    address is found too."""
    words = list(_WORD.finditer(text))
    lines = []
    for i in range(len(words) - len(name) + 1):
        if all(_digest(words[i + k].group()) == name[k] for k in range(len(name))):
            lines.append(text.count("\n", 0, words[i].start()) + 1)
    return lines


def _problems(rel: str, data: bytes) -> list[str]:
    if b"\0" in data:
        return []                           # binary
    text = data.decode("utf-8", errors="replace")
    problems = [f"{rel}:{n}: {what} ({hit!r})" for n, what, hit in _mentions(text)]
    return problems + [f"{rel}:{n}: the owner's full name" for n in _names(text)]


def test_no_tracked_file_names_a_place_on_one_machine_or_its_owner() -> None:
    problems = [p for rel in _files() for p in _problems(rel, (REPO / rel).read_bytes())]
    assert not problems, "\n".join(problems)


def test_the_guard_sees_each_kind_of_mention() -> None:
    cases = [
        ("a = '/home/me/x'", "a home folder"),
        ("b = '/Users/me/x'", "a home folder"),
        ("dir C:\\Users\\me\\x", "a home folder"),
        ("c = '~/data'", "a home folder"),
        ("d = '/tmp/run'", "a temporary folder"),
        ("e = 'https://claude.ai/code/session_0123'", "a link to a Claude Code session"),
        ("f = '../Z_DATA/x'", "the retired data folder"),
        ("g = '../e2e_runs'", "the retired runs folder"),
        ("h = 'reifyscene_project'", "a folder beside the repository"),
        ("i = '.gstack/x'", "a personal tool"),
        ("j = 'me@Gmail.com'", "a personal email address"),
        ("k = 'see [[some-note]]'", "a note kept outside the repository"),
    ]
    alike = ["l = [[0]]", "m = 'tzdata==1'", "n = 'https://claude.ai/code'", "o = 'me@example.com'"]
    found = _mentions("\n".join([line for line, _ in cases] + alike) + "\n")
    # one mention per case, of its kind; the alike lines give none
    expected = [(n, what) for n, (_, what) in enumerate(cases, 1)]
    assert [(n, what) for n, what, _ in found] == expected
    assert "me@Gmail.com" in {hit for _, _, hit in found}      # the whole address, any case


def test_a_binary_file_is_skipped() -> None:
    text = b"x = '/home/me/x'\n"
    assert _problems("a.bin", b"\0" + text) == []
    assert _problems("a.txt", text) == ["a.txt:1: a home folder ('/home/')"]


def test_the_guard_sees_a_name_by_its_digests() -> None:
    name = (_digest("ada"), _digest("lovelace"))
    text = "Ada Lovelace\nwrote to ada.\nLovelace\nLovelace Ada\nada lovelaces\n"
    assert _names(text, name) == [1, 2]     # in order, across a line break or a dot; whole words
