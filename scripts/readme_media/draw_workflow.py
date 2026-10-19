#!/usr/bin/env python
"""The README's workflow diagram, ``docs/media/workflow.svg``: how reifyscene is built.

    python scripts/readme_media/draw_workflow.py [OUT.svg]

The layout is drawn by hand, not by a graph engine (Mermaid, Graphviz), so the picture keeps what
an engine would not: one gap between every two steps, the four stages centred on the line that
joins them, and the gate loop's arrows on chosen sides (out of the gate runs' left side and into
their right; into the owner's hexagon at its left point and out at its right). To change the
diagram, edit ``stages()`` and run this again.

Text is measured with Liberation Sans (``fonts-liberation`` on Debian and Ubuntu), which has
Arial's metrics; the SVG names Arial first. Each shape's text is placed by its glyphs' outlines,
so its ink is centred in the shape whatever its descenders (``centred_text``). The colours are
the app's dark theme, with the owner's steps in amber.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import ImageFont

FAMILY = "Arial, 'Liberation Sans', Helvetica, sans-serif"
BG, PANEL, NODE, NODE_EDGE, TEXT = "#101416", "#202729", "#181d20", "#3a464b", "#e6ecec"
LINE, AMBER, INK = "#8d9a9f", "#f2c14e", "#08181b"

SIZE, LH, PADX, PADY = 15, 19, 18, 11
NODE_H = 2 * LH + 2 * PADY      # every step is two lines, so every step is one height
GAP = 46                        # the arrow between two steps, everywhere
TITLE, TITLE_TOP, TITLE_GAP = 21, 16, 18
STAGE_PAD, STAGE_GAP, MARGIN = 26, 64, 24
LABEL = 13
TAG_H = LABEL + 11              # a coloured label's block
SLANT = 17                      # how far the hexagon's points stand out
RADIUS = 12                     # rounded corners on bent arrows
HEAD = 10                       # arrowhead length
STROKE = 2.2
FINE = 8                        # ink is measured at this many times the size, for sub-pixel extents
# The gate loop's two ways out of the owner's judgement: back to the runs, or on.
AGAIN, AGAIN_FILL = "change and run again", "#f0884a"
ACCEPTED, ACCEPTED_FILL = "accepted", "#5fd38d"

_fonts: dict[tuple[bool, int], ImageFont.FreeTypeFont] = {}

Point = tuple[float, float]


def font(bold: bool, size: int) -> ImageFont.FreeTypeFont:
    if (bold, size) not in _fonts:
        name = "LiberationSans-Bold.ttf" if bold else "LiberationSans-Regular.ttf"
        _fonts[(bold, size)] = ImageFont.truetype(name, size)
    return _fonts[(bold, size)]


def text_width(text: str, bold: bool = False, size: int = SIZE) -> float:
    return font(bold, size).getlength(text)


@dataclass(frozen=True)
class TextStyle:
    """How a piece of text is set. ``caps`` centres it by the capitals' height (top of an H to the
    baseline) rather than by its ink, so titles side by side share one rule whatever their
    letters."""

    size: int
    fill: str
    bold: bool = False
    caps: bool = False


def centred_text(centre: Point, lines: list[str], style: TextStyle) -> str:
    """``lines`` as SVG text whose ink is centred on ``centre``.

    SVG's own centring (``dominant-baseline``) centres a fixed line box, which leaves a line with
    descenders sitting low. Here each line's glyph outlines are measured instead: the block's ink,
    from its highest glyph's top to its lowest glyph's bottom, is centred vertically, and each
    line's ink horizontally. The baselines are written out, so no renderer has to place them.
    """
    fine = font(style.bold, style.size * FINE)
    # ink boxes (left, top, right, bottom) relative to each line's left end on its baseline
    boxes = [tuple(v / FINE for v in fine.getbbox(t, anchor="ls")) for t in lines]
    if style.caps:
        top = fine.getbbox("H", anchor="ls")[1] / FINE
        bottom = float((len(lines) - 1) * LH)
    else:
        top = min(b[1] + i * LH for i, b in enumerate(boxes))
        bottom = max(b[3] + i * LH for i, b in enumerate(boxes))
    first = centre[1] - (top + bottom) / 2
    weight = ' font-weight="700"' if style.bold else ""
    out = []
    for i, (line, b) in enumerate(zip(lines, boxes, strict=True)):
        # text-anchor centres the advance width; move it so the ink is centred instead
        x = centre[0] + fine.getlength(line) / FINE / 2 - (b[0] + b[2]) / 2
        out.append(f'<text x="{x:.2f}" y="{first + i * LH:.2f}" font-size="{style.size}" '
                   f'fill="{style.fill}"{weight} text-anchor="middle">{escape(line)}</text>')
    return "".join(out)


@dataclass
class Step:
    """One step: two lines of text in a box, a hexagon (the owner decides) or a pill (the owner
    starts). ``x``, ``y`` are its centre once placed; ``w`` comes from its text."""

    lines: list[str]
    shape: str = "box"
    x: float = 0.0
    y: float = 0.0
    w: float = field(init=False)

    def __post_init__(self) -> None:
        self.w = max(text_width(t, self.owner) for t in self.lines) + 2 * PADX
        if self.shape == "hex":
            self.w += 2 * SLANT - 8

    @property
    def owner(self) -> bool:
        """Whether the owner takes this step (drawn in amber)."""
        return self.shape in ("hex", "pill")

    def left(self) -> Point:
        return self.x - self.w / 2, self.y

    def right(self) -> Point:
        return self.x + self.w / 2, self.y

    def top(self) -> Point:
        return self.x, self.y - NODE_H / 2

    def bottom(self) -> Point:
        return self.x, self.y + NODE_H / 2


@dataclass
class Tag:
    """A coloured label on the arrow between two steps, given a row of its own in the column."""

    text: str
    fill: str
    x: float = 0.0
    y: float = 0.0


Row = Step | Tag | None


@dataclass
class Stage:
    """One stage: a titled column of steps, top to bottom, one gap between every two rows.

    ``None`` marks the gate loop's row: ``side`` on the left of the column, the way back on the
    right. A ``Tag`` labels the arrow that passes through its row.
    """

    title: str
    colour: str
    steps: list[Row]
    side: Step | None = None


def step(text: str, shape: str = "box") -> Step:
    """A step whose two lines are ``text`` split at its newline."""
    return Step(text.split("\n"), shape)


def stages() -> tuple[Step, list[Stage]]:
    """The diagram's content: the owner's start, then the four stages left to right."""
    goal = step("The owner sets\nthe next goal", "pill")
    return goal, [
        Stage("1 · Design", "#a596ff", [
            step("Research agents read the\npapers and reference code"),
            step("Design note drafted:\noptions, tests, pass marks"),
            step("Owner decides\neach fork", "hex"),
            step("Design note committed\nwith its branch"),
        ]),
        Stage("2 · Build", "#f0884a", [
            step("A feature branch\noff develop"),
            step("Pipeline stages in Python;\nversioned, resumable records"),
            step("Unit tests\nwith each change"),
            step("Touched checks per commit;\nmessages checked claim by claim"),
        ]),
        Stage("3 · Verify", "#5fd38d", [
            step("Runs on public data:\nDAVIS 2017, DyCheck"),
            step("Gate runs on chosen clips,\nagainst a baseline"),
            None,
            step("Owner judges\nnumbers and viewer", "hex"),
            Tag(ACCEPTED, ACCEPTED_FILL),
            step("Results and limits written\ninto benchmarks/ and the design"),
            step("Tests and lint\nbefore the pull request"),
        ], side=step("Scored: removal tests,\nheld-out PSNR, graph vs DAVIS")),
        Stage("4 · Ship", "#3fb6c6", [
            step("Pull request into develop;\nmeasurements and decisions listed"),
            step("Owner's\nmerge Yes", "hex"),
            step("Commits verified;\nversion tag made"),
            step("Owner's release call:\ndevelop to main", "hex"),
        ]),
    ]


def loop_offset(stage: Stage) -> float:
    """How far the gate loop's two sides sit from the stage's centre line: clear of the step above
    the loop, which the loop leaves and re-enters."""
    above = stage.steps[stage.steps.index(None) - 1]
    assert isinstance(above, Step)
    return above.w / 2 + 42


def row_height(row: Row) -> float:
    return TAG_H if isinstance(row, Tag) else NODE_H


def stage_size(stage: Stage) -> tuple[float, float]:
    """A stage's width and height, symmetric about its centre line."""
    half = max(s.w / 2 for s in stage.steps if isinstance(s, Step))
    if stage.side:
        dx = loop_offset(stage)
        half = max(half, dx + stage.side.w / 2, dx + text_width(AGAIN, True, LABEL) / 2 + 8)
    half = max(half, text_width(stage.title, True, TITLE) / 2)
    rows = sum(row_height(r) for r in stage.steps) + (len(stage.steps) - 1) * GAP
    return 2 * (half + STAGE_PAD), TITLE_TOP + TITLE + TITLE_GAP + rows + STAGE_PAD


def rounded(points: list[Point]) -> str:
    """SVG path data for an orthogonal polyline with rounded corners, its end pulled back by the
    arrowhead's length so the head's tip lands on the last point."""
    pts = list(points)
    (x0, y0), (x1, y1) = pts[-2], pts[-1]
    d = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
    pts[-1] = (x1 - (x1 - x0) / d * HEAD, y1 - (y1 - y0) / d * HEAD)
    out = [f"M{pts[0][0]:.1f},{pts[0][1]:.1f}"]
    for (ax, ay), (bx, by), (cx, cy) in zip(pts, pts[1:], pts[2:], strict=False):
        nin = ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5
        nout = ((cx - bx) ** 2 + (cy - by) ** 2) ** 0.5
        r = min(RADIUS, nin / 2, nout / 2)
        p = (bx - (bx - ax) / nin * r, by - (by - ay) / nin * r)
        q = (bx + (cx - bx) / nout * r, by + (cy - by) / nout * r)
        out.append(f"L{p[0]:.1f},{p[1]:.1f} Q{bx:.1f},{by:.1f} {q[0]:.1f},{q[1]:.1f}")
    out.append(f"L{pts[-1][0]:.1f},{pts[-1][1]:.1f}")
    return " ".join(out)


def arrow(points: list[Point]) -> str:
    return (f'<path d="{rounded(points)}" fill="none" stroke="{LINE}" stroke-width="{STROKE}" '
            f'stroke-linejoin="round" marker-end="url(#head)"/>')


def label(x: float, y: float, text: str, fill: str) -> str:
    """A label in a coloured block centred on a point of an arrow, its text dark and bold."""
    w = text_width(text, True, LABEL) + 16
    return (f'<rect x="{x - w / 2:.1f}" y="{y - TAG_H / 2:.1f}" width="{w:.1f}" height="{TAG_H}" '
            f'rx="4" fill="{fill}"/>'
            + centred_text((x, y), [text], TextStyle(LABEL, INK, bold=True)))


def draw_step(s: Step) -> str:
    x0, y0, x1, y1 = s.x - s.w / 2, s.y - NODE_H / 2, s.x + s.w / 2, s.y + NODE_H / 2
    fill, stroke, ink = (AMBER, AMBER, INK) if s.owner else (NODE, NODE_EDGE, TEXT)
    if s.shape == "hex":
        corners = [(x0, s.y), (x0 + SLANT, y0), (x1 - SLANT, y0),
                   (x1, s.y), (x1 - SLANT, y1), (x0 + SLANT, y1)]
        body = f'<polygon points="{" ".join(f"{a:.1f},{b:.1f}" for a, b in corners)}"'
    else:
        rx = NODE_H / 2 if s.shape == "pill" else 5
        body = f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{s.w:.1f}" height="{NODE_H}" rx="{rx}"'
    shape = f'{body} fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>'
    return shape + centred_text((s.x, s.y), s.lines, TextStyle(SIZE, ink, bold=s.owner))


@dataclass
class Layers:
    """The drawing's elements by layer, back to front: stage boxes and titles, arrows, steps,
    labels on arrows."""

    boxes: list[str] = field(default_factory=list)
    edges: list[str] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)


def draw_stage(stage: Stage, left: float, mid: float, layers: Layers) -> float:
    """Place a stage with its left side at ``left``, centred on the line ``mid`` that joins the
    stages, and draw it into ``layers``. Returns its right side."""
    w, h = stage_size(stage)
    top, cx = mid - h / 2, left + w / 2
    title = (cx, top + TITLE_TOP + TITLE / 2)
    layers.boxes.append(
        f'<rect x="{left:.1f}" y="{top:.1f}" width="{w:.1f}" height="{h:.1f}" rx="10" '
        f'fill="{PANEL}" stroke="{stage.colour}" stroke-width="2"/>'
        + centred_text(title, [stage.title], TextStyle(TITLE, stage.colour, bold=True, caps=True)))
    y = top + TITLE_TOP + TITLE + TITLE_GAP
    for row in stage.steps:
        if row is not None:
            row.x, row.y = cx, y + row_height(row) / 2
        y += row_height(row) + GAP
    # one arrow between two steps, passing behind any tag between them; none across the loop row
    column = [r for r in stage.steps if not isinstance(r, Tag)]
    for a, b in pairwise(column):
        if a and b:
            layers.edges.append(arrow([a.bottom(), b.top()]))
    layers.labels.extend(label(t.x, t.y, t.text, t.fill) for t in stage.steps if isinstance(t, Tag))
    if stage.side:
        draw_loop(stage, cx, layers)
    layers.steps.extend(draw_step(s) for s in stage.steps if isinstance(s, Step))
    return left + w


def draw_loop(stage: Stage, cx: float, layers: Layers) -> None:
    """The gate loop: the runs, their scores at the side, the owner's judgement, and back."""
    assert stage.side is not None
    row = stage.steps.index(None)
    above, below = stage.steps[row - 1], stage.steps[row + 1]
    assert isinstance(above, Step)
    assert isinstance(below, Step)
    dx = loop_offset(stage)
    side = stage.side
    side.x, side.y = cx - dx, (above.y + below.y) / 2
    layers.steps.append(draw_step(side))
    # out of the runs' left side into the scores; the scores into the owner's left point
    layers.edges.append(arrow([above.left(), (cx - dx, above.y), side.top()]))
    layers.edges.append(arrow([side.bottom(), (cx - dx, below.y), below.left()]))
    # out of the owner's right point, up through "change and run again", into the runs' right side
    layers.edges.append(
        arrow([below.right(), (cx + dx, below.y), (cx + dx, above.y), above.right()]))
    layers.labels.append(label(cx + dx, side.y, AGAIN, AGAIN_FILL))


def render() -> str:
    goal, columns = stages()
    height = max(stage_size(s)[1] for s in columns) + 2 * MARGIN
    mid = height / 2
    goal.x, goal.y = MARGIN + goal.w / 2, mid
    layers = Layers(steps=[draw_step(goal)])
    x = MARGIN + goal.w + STAGE_GAP
    layers.edges.append(arrow([goal.right(), (x, mid)]))
    for i, stage in enumerate(columns):
        x = draw_stage(stage, x, mid, layers)
        if i < len(columns) - 1:
            layers.edges.append(arrow([(x + 1, mid), (x + STAGE_GAP - 1, mid)]))
            x += STAGE_GAP
    width = x + MARGIN
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" '
        f'viewBox="0 0 {width:.0f} {height:.0f}" font-family="{FAMILY}">'
        f"<title>How reifyscene is built: design, build, verify, ship</title>"
        f'<defs><marker id="head" viewBox="0 0 10 10" refX="0" refY="5" markerWidth="{HEAD}" '
        f'markerHeight="{HEAD}" markerUnits="userSpaceOnUse" orient="auto">'
        f'<path d="M0,0.5 L10,5 L0,9.5 z" fill="{LINE}"/></marker></defs>'
        f'<rect width="100%" height="100%" fill="{BG}"/>'
        + "".join(layers.boxes + layers.edges + layers.steps + layers.labels)
        + "</svg>\n"
    )


def main(argv: list[str]) -> int:
    out = (Path(argv[0]) if argv
           else Path(__file__).resolve().parents[2] / "docs" / "media" / "workflow.svg")
    out.write_text(render(), encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
