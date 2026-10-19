# Not every candidate is a thing — background classes and a noise floor for the scene graph

**Status:** ADOPTED (2026-10-17), built on `feat/graph-noise`
**Date:** 2026-10-17
**Follows:** `docs/design/2026-10-17-removal-by-ownership.md` (§7, decision 7: the clip ranking
waits on a cleaner graph)

---

## 0. Why

The owner, looking at the DAVIS runs in the viewer: the scene graph holds a lot of noise. Measured
on the ten `slice_*` runs: **139 entities, of which 31 match a DAVIS-annotated object.** Some of the
other 108 are real things nobody annotated (cars, signs); many are not things at all:

- **Nothing filtered a candidate.** Every SAM 2 tracklet became a node, one seen on 2 frames with
  no point track inside it included.
- **The backdrop was proposed as things.** Candidates are SAM 2's automatic masks on the first
  frame, the 20 largest kept, so floors, lawns, sky and crowds are proposed as readily as objects.
- **Labels could not say "floor".** SigLIP chose among COCO-80's object classes only, with no
  background class and no abstention: a hillside became "bear", the judo mat "bed" (three times),
  the sky over the sea "airplane".

## 1. Reference

ConceptGraphs (`third_party/concept-graphs`, its `slam` pipeline) drops detections of its
background classes (`BG_CLASSES` wall, floor, ceiling; `skip_bg`, on by default), masks under
25 px (`mask_area_threshold`) or under its confidence threshold (`mask_conf_threshold`, 0.2 by
default), and objects associated on fewer than 3 frames (`obj_min_detections`). The configuration
this repository ran it with also dropped non-background boxes over half the frame
(`max_bbox_area_ratio` 0.5, 1.0 by default) and masks under 0.95 confidence.

## 2. The design

**Background classes, in the semantics stage** (`reifyscene/perception/semantics/vocab.py`,
`SiglipSemantics._backdrop`). The labels on offer are COCO-80's things, LVIS v1's frequent tier
(346 more), and 27 background classes (floor, wall, ceiling, ground, grass, ..., sky, water,
building, fence, tree, crowd). A track the person detector calls a person chooses among COCO-80 and
the background classes only, and is never backdrop. Any other track is **backdrop** when a
background class beats every COCO-80 thing *and* no LVIS thing beats it by 0.045 or more; its record
says so (`SemanticRecord.background`) and its category becomes that background class.

The asymmetry is measured, not chosen (§3): LVIS's long tail scores close to any backdrop, so an
LVIS thing must win clearly, while a COCO thing wins on equal terms.

**The graph gives backdrop to the background** (`reifyscene/graph/builders.py`). An object track
the semantics stage called backdrop becomes part of the background node (`absorbed_tracks`); the id
maps (`pixel_masks`) give its pixels to the background, so its Gaussians are background-owned.

**A noise floor in the graph.** A candidate is an entity only when its mask covers at least 0.1% of
the frame (median over the frames it is seen; `graph.entity_min_area`) and at least 5 point tracks
support it (`graph.entity_min_support_tracks`). Below either, it is no node, and its pixels are
unaddressed (never removable, ignored by the owner loss). On the ten runs the smallest annotated
object covered 0.18% of the frame and had 12 support tracks. Absorbed and dropped candidates, with
the reason, are listed in the graph record (`metadata["candidates"]`).

**The viewer ranks entities by their Gaussians.** Node size and the order of the graph's circle
and the timeline follow each entity's Gaussian count (the `gaussians` layer); entities with under
0.1% of the scene's Gaussians, or none, are folded behind "show N small".

Contract versions: semantics 1 → 2, graph 6 → 7.

## 3. Measurements (the ten `slice_*` runs, resumed at semantics, 2026-10-17)

Each change was run on the same upstream (segment, tracking and reid loaded from the slice runs),
so node ids stay the same and the DAVIS matches of `benchmarks/davis_edit_damage.json` still apply.
The 90-sequence DAVIS inventory (`benchmarks/davis_inventory.json`) was not re-scored: the owner
chose the ten-clip measurement (2026-10-17), so the margin and the floor are checked only on the
clips they were chosen on.

| vocabulary and rule | entities (of 139) | unmatched (of 108) | DAVIS objects kept (of 31) |
|---|---|---|---|
| COCO-80 + background, backdrop = a background class wins | 67 | 40 | 27: gold-fish's four fish became "water" and "sea" (COCO has no fish) |
| all of LVIS + background | 112 | 81 | 31, but people became "CD player", backdrop "plume" |
| COCO + LVIS frequent + background, a background class wins | 101 | 70 | 31; backdrop pieces became "tape", "paper towel" |
| COCO + LVIS frequent + background, any thing must lead by 0.045 | 50 | 20 | 30: blackswan's swan became "water" ("bird" led by 0.044) |
| **COCO + LVIS frequent + background, the asymmetric rule (adopted)** | **72** | **41** | **31** |

The margin comes from the tracks a background class won against COCO: every backdrop piece's best
LVIS thing led by +0.040 or less, the four fish ("fish") by +0.049 to +0.061, a black bin bag
("plastic bag") by +0.050. Two label-free signals were tried first and separate nothing: whether a
mask touches the frame border (a fish reaches the edge, sky through branches does not) and how far
an entity's 3D centre moves in its own size (backdrop centres drift as much as swimming fish).

Of the 41 unmatched entities left, judged by their labels: 11 are people DAVIS does not annotate
(judo's onlookers, bystanders in india and lindy-hop); about 10 read as real objects nobody
annotated (two cars, a motor scooter, bags, a signboard, a mattress, a monitor); about 20 are
doubtful pieces with odd labels ("skirt", "tape", "cork", "lettuce"). The viewer's Gaussian ranking
is for those, and the follow-on's scores (§4) are the next filter for them.

## 4. Not done here (the follow-on)

- **Carry the real scores** (SAM 2's predicted IoU and stability, the person detector's score)
  into node confidence, which is 1.0 everywhere today: a segment-stage change.
- **Replace the largest-first cap** of 20 masks with the score, so small real things are not
  crowded out (recall: 61 of DAVIS's 205 annotated objects are never proposed,
  `benchmarks/davis_inventory.json`). Needs the scores.
- **Late arrivals:** a thing first seen after frame 0 is never proposed.
