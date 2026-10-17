# Removal that leaves no trace — fixed ownership at birth

**Status:** ADOPTED (2026-10-17), being built on `feat/removal-ownership`
**Date:** 2026-10-16 (draft), 2026-10-17 (the owner's decisions, §7)
**Follows:** `docs/design/2026-09-29-per-gaussian-identity-scope.md` (§4.7–4.9: every post-hoc
assignment tops out near 0.4 precision rendered alone)

---

## 0. Why

**What the app shows today.** On `exp_a_lab-coat` (single-pass VGGT, pose_refine off, depth
supervision), removing a person leaves the person standing in "after", although "took" looks
complete. Measured on 2026-10-16 with a one-off script, now the residual-share diagnostic of §1
(all 47 frames, inside the person's own mask, FlashSplat + floater filter as the app runs it):

| removal | took, alone (alpha) | put back by the filter | put back, alone (alpha) | share of the render in the mask: took / put back / everything else | what lies behind (alpha, with both gone) |
|---|---|---|---|---|---|
| person 8, slackness −0.8 | 0.975 | 6,919 Gaussians (5,316 oversized) | 0.969 | 0.631 / 0.358 / 0.012 | 0.974 |
| person 8, slackness 0 | 0.960 | 6,503 | — | 0.610 / 0.358 / 0.031 | — |
| person 2, slackness +0.4 | 0.954 | 4,618 (4,402 oversized) | 0.875 | 0.592 / 0.354 / 0.054 | 0.999 |

Each person is drawn **twice**: a fine layer the removal takes, and a coarse layer of large
Gaussians that the assignment also gave to the person and the floater filter
(`filter_members`, `reifyscene/graph/assignment.py`: largest axis > 3× the selection's median)
put back. Alone, the fine layer covers 95–98% of the silhouette and the coarse one 87–97%.
Removing the fine layer exposes the coarse one. Background exists behind (97–100%), so this is
not a missing-background problem, and slackness barely matters (person 8: took's share 0.61 at
0, 0.63 at −0.8). Unticking the filter in the app looked "marginally better" to the owner.

**What the 2026-09-28 gate already found.** Removing one of lab-coat's three women tore the
heads off the two beside her. Every post-hoc method since — the projection vote, learned
identity (I2 + I3), FlashSplat with and without the filter — tops out near **0.4 precision
rendered alone** on the ten DAVIS loop scenes (identity scope doc §4.7–4.9). In that doc's words,
Gaussians that render as a neighbour's face contribute mostly to the target's pixels, "which no
assignment rule undoes".

**Root cause.** The render stage trains on photos (and now depth) only. Entities exist at the
start — every seed point comes from a pixel that lay in one mask — and the scene stage throws
that away: `_fg_mask_by_frame` (`reifyscene/scene/assembler.py`) unions every instance into one
foreground. Training then moves, splits, clones and prunes Gaussians with nothing tying any of
them to an entity. With one camera and little parallax, many arrangements reproduce the photos
equally well, and the one training finds layers and interleaves entities. Ownership is then
applied afterwards, by a vote, to a scene that was never asked to be separable.

## 1. Acceptance criteria — what "removed" means

The owner's four expectations, and how each is measured from the clip itself (no ground truth
beyond the photos and our own masks, except where noted).

**1. No visual trace of the entity in any rendered image.**
- *Test T1 (independent of the removal method):* run the entity's category detector on every
  after-render — training frames and the orbit trajectory. People: `PersonSegmenter`
  (`reifyscene/eval/benchmarks/edit_damage/silhouettes.py`: the pipeline's `HumanDetector` +
  SAM 2 box prompt). Other categories: the open-vocabulary detector already used for the
  inventory (`grounding-dino-base`, `benchmarks/detector_keyframe_cache.json`). A *trace* is a
  detection of the entity's category whose mask overlaps the entity's before-mask (IoU ≥ 0.3,
  proposed). Matched against the removed entity's own region only, so a neighbour of the same
  category (lab-coat's adjacent women) does not count. Reported: share of frames with a trace.
- *Diagnostic (baseline only):* the residual share of §0 — how much of the before-render inside
  the mask came from Gaussians that stayed. Under ownership it is zero by construction, so it
  is not a test of the new design; T1 is.

**2. In 3D, the space it occupied is empty, at every moment.**
- *Test T2:* per training frame *t*, the entity's volume *V_t*; after removal, the opacity mass of
  remaining Gaussians whose positions at *t* (`DynamicScene.cloud_at(t)`) fall inside *V_t*,
  as a share of the same mass before removal.
- *V_t* for people: the Sapiens body volume (`reifyscene/edit/body_volume.py`, `sapiens_body` /
  `lift_frame`: capsules round the bones, from the rendered depth over the person's mask). Frames
  without a torso use the slab below.
- *V_t* otherwise: a slab — the mask's pixels back-projected from the front surface (the bundle's
  `target_depths`, scene stage `_train_targets`) to front + the entity's own width at *t*. A slab,
  not a hull (identity scope doc §3.2: a hull reaches into neighbours).

**3. Background the video saw appears, unoccluded, where the entity was.**
- *Test T3:* for frame *t* and each pixel *p* of the entity's mask, forward-project every other
  frame's background pixels (id map = background, valid target depth) into *t* through the
  training cameras, z-buffered. *p* is **seen** if a background point lands on it from some frame
  *t′* and is not in front of the entity's surface at *p*; its expected colour is the photo at the
  nearest such *t′*. Score: PSNR of the after-render against the expected colour over seen pixels.
- *Control (the geometry floor):* the same projection scored on pixels outside every mask, where
  nothing occludes. If T3 is no worse than the floor, the removal revealed what the video saw. If
  both are poor, the cameras and depth disagree across frames — a geometry problem (SLAM It!),
  not a removal one.

**4. Background the video never saw: left as is, or blurred.**
- *Test T4:* the complement of "seen" inside the mask. Reported: unseen share per entity. The
  render is left as is (default); a blur of unseen pixels is a cosmetic option for the site.

T1 and T2 run on training views and the orbit; T3 and T4 need photos, so training views only.
**A person looking stays the final gate** — the 09-28 gate read "92% intact" for a removal that
tore heads off.

## 2. Baseline — today's removal against the four tests

Run first, before any change, so the design is judged against numbers rather than impressions.

- **Runs:** `exp_a_lab-coat` (the option-A geometry: single-pass VGGT, pose_refine off, depth
  supervision) — all eight entities, FlashSplat at slackness 0, with and without the filter. And
  the ten DAVIS `slice_*` runs as they stand (VGGT-Long + pose_refine, trained before the scene
  graph PR), which also give the clip ranking (§7). The clips chosen from that ranking are then
  rebuilt on the option-A geometry and measured again, before and after ownership.
- **Already measured:** §0's table (residual share; lab-coat persons 8 and 2 only).
- **Not yet measured:** T1–T4 on any clip.

Building T1–T4 is the first part of the PR; the baseline is its first output.

## 3. Reference systems

| system | how an object's Gaussians are kept | what is behind a removed object |
|---|---|---|
| **Street Gaussians** (ECCV 2024) | Separate per-vehicle models from the start: "collect points inside the bounding boxes and transform them into the local coordinate system"; background from the remaining LiDAR. Object Gaussians live in the object's frame, moved by optimisable tracked poses; ones that leave the box are pruned. | Not addressed. |
| **OmniRe** (ICLR 2025) | A Gaussian scene graph: background, rigid vehicle nodes, SMPL nodes for people ("initialize the Gaussians on the template mesh in their canonical space", each bound to a vertex), deformable nodes for other non-rigid actors. Nodes initialised from LiDAR inside each actor's box; background from LiDAR with dynamic objects filtered out. No per-node mask loss. | Not addressed. |
| **DreamScene4D** (NeurIPS 2024) | "Decompose-recompose" from monocular video: mask trackers segment objects; each object track is lifted to its own Gaussians and its motion optimised independently; recomposed with monocular depth guidance. | Occluded regions completed with an inpainting diffusion model (SD-Inpaint, adapted for video amodal completion). |
| **Gaussian Grouping** (ECCV 2024) | A learned 16-dim identity per Gaussian, supervised by SAM masks through rendering, plus a 3D neighbour-consistency term. Membership is learned, not fixed. (Built here as I2/I3.) | Removal leaves an "unseen region mask"; LaMa inpaints it in 2D, and the 3D Gaussians are fine-tuned on those pseudo-labels (SPIn-NeRF's pipeline; `third_party/gaussian-grouping/docs/edit_removal_inpaint.md`). |
| **FlashSplat** (ECCV 2024) | Post-hoc: each Gaussian takes the label with the larger accumulated α·T, one-vs-rest, with a background bias γ (γ = −0.4 for removal). Our current path. They state "3D Gaussians do not exclusively belong to a single object" — about 20% of one scene's Gaussians are shared by more than two objects. | "A noisier background or even black holes"; inpainting with a pretrained 2D inpainter (their ref. [43], LaMa) and 200K new Gaussians near the object. |

What this says: the systems that remove cleanly (Street Gaussians, OmniRe, DreamScene4D) never
assign after the fact — each object is its own set from initialisation. They use 3D boxes or
per-object optimisation where we have 2D masks plus depth; the principle carries, the mechanism
has to be ours. The post-hoc methods (Gaussian Grouping, FlashSplat) report the entanglement we
measured. Everyone who addresses unseen background inpaints; that is criterion 4's territory,
outside this design.

## 4. The design — fixed ownership at birth

### 4.1 Who owns a seed

The source of truth for "which entity is this pixel" is the id map
(`reifyscene/graph/id_maps.py`): `0` background, `1..K` graph nodes, `65535` ignore (overlap,
unmapped tracklet, padding), in the geometry's raster, built from the same mapping the removal uses
(`render_pixel_masks`). The scene stage reads only `segment` today; it gains `reid` and `graph`
upstream (both run before it).

- **Foreground seeds** are 2D tracks lifted to 3D (`lift_tracks_3d`,
  `reifyscene/lifting/native_track_lift.py`), each sampled at a query pixel in a query frame. Its
  owner is the id map at that pixel. Seeds whose neighbourhood (radius 2 px, proposed) is not one
  class are not used: mask edges are where SAM is least reliable.
- **Background seeds** come from pixels outside every mask (`lift_bkgd_points`): owner 0.
- **Unmapped tracklets** (a segment the graph dropped) sit inside the foreground union and seed
  Gaussians too. Their owner is a reserved *unaddressed* class: never removable, and their pixels
  are ignored by the loss, as the id map already ignores them.
- **Owner is not motion.** fg/bg is the motion split; ownership is semantic and lives on both
  fields (identity scope doc §3.3). A static entity under the motion gate is a background-field
  Gaussian with owner *k*, still removable.

The owner is an integer buffer on each field (`GaussianField`), not a parameter and never
optimised, plus a class table (index → node id) in the bundle.

### 4.2 Inheritance

Densification (`reifyscene/render/gaussian4d/densify.py`) makes new Gaussians only by cloning and
splitting. `_clone_and_split_rows` copies the parent's parameters (a split moves the position and
shrinks the scale; everything else is inherited). A child's owner is its parent's; pruning removes
it. The owner buffer rides through `_append` / `_prune` beside the optimizer groups. I found no
other path that creates Gaussians.

### 4.3 The per-entity silhouette loss

Each step, for each view, the one-hot owner is rendered as feature channels (background,
unaddressed, *K* entities) through the same rasterizer. This is the mechanism the existing mask
term already uses (the `fg_ind` channel in `DynamicGaussianTrainer.fit`), and §8.1 of the identity
doc measured wide channels as cheap (19 channels, 1.31 ms for 80k Gaussians at 518×294).

- **Loss:** per pixel, the rendered owner shares against the id map's one-hot, ignoring `65535`,
  with total alpha driven to 1 (Shape-of-Motion's mask term, per entity). Equivalently: each
  entity's Gaussians, rendered alone, must cover its mask and nothing else.
- **Gradients reach geometry** — means, scales, opacities, motion coefficients. That is the point.
- **Densification sees photometric gradients only.** The owner channels go in a second
  rasterization, so the `means2d` gradient densification thresholds on is the photo pass's. The
  coupled-identity run lost 23% of its foreground Gaussians when identity gradients reached it
  (identity doc §4.3). Cost: one more pass per view (not yet measured).
- **Background** Gaussians are supervised to cover background pixels. Behind an entity nothing
  supervises them — that is criterion 4's unseen region, by definition.
- **Why not the −1.9 dB mask term again:** that term drove the foreground *field* to match every
  segmented instance, forcing a motion split to carry identity (render stage, 2026-10-01). Here
  ownership is independent of the field split. Its PSNR cost is still unknown (§5).

**Why fixed owners force separation, and relabelling would not.** If a Gaussian drifts so that it
draws part of person B while owned by A, the loss sees A's Gaussians covering B's pixels. If the
label could change, the cheapest fix is to relabel it B: the Gaussian stays half in both people
and the loss calls it fine. The coupled identity experiment showed exactly that — labels and
geometry trained together separated worse (identity doc §4.3). With the owner fixed, the only ways
to reduce the error are to move the Gaussian back into A, fade it, or let densification replace
it. The scene is pushed to separate.

### 4.4 Removal

Removal gets a method, chosen per edit:

- **`owner`** (the default): every Gaussian in either field whose owner is the node. No evidence,
  no slackness, no filter.
- **`flashsplat`**: today's path — the evidence matrix, `assign` / slackness, `filter_members`
  and the optional body bound. The app's slackness and filter controls stay and apply to this
  method only.
- **`identity`**: the learned identity (I2/I3), where the run has it.

The old two stay for comparison (§7, decision 4); ownership is the default because it is the
only one that is right by construction. The handles stage's `gaussians` layer is written from
the owner table on runs trained with owners.

### 4.5 The drift check (the owner's re-labelling idea)

Every *N* steps (500, proposed), compute FlashSplat's evidence (`evidence_matrix` over
`id_map_views`) and compare it with each Gaussian's owner. A Gaussian whose evidence goes clearly
to another class is **flagged, never relabelled** — per identity doc §3.1, a Gaussian that is
genuinely A at one time and B at another makes `remove` ill-posed, and the remedy is to split or
prune it. First version: count flags per entity pair over training and log them; act (prune, or
split — unproven) only if the counts are non-trivial.

### 4.6 Format

The bundle gains `fg_owner` / `bg_owner` and the class table. Contract versions: scene 5 → 6 (seeds
carry owners; upstream adds reid and graph), render 4 → 5 (silhouette loss; the checkpoint carries
owners), handles 2 → 3 (the selection is ownership). Clean break: no reader for old checkpoints.
The runs the site needs are rebuilt from the scene stage on.

## 5. Measurement and gate

On the same clips as the baseline, today's removal vs ownership:

| test | proposed pass |
|---|---|
| T1 trace | no trace on ≥ 95% of frames, per entity |
| T2 space | remaining opacity mass in *V_t* ≤ 5% of before |
| T3 seen background | within 1 dB of the geometry floor |
| T4 unseen | reported, no pass mark |
| training-view PSNR | ≤ 0.5 dB below the same run without the loss (the identity work's noise band) |
| by eye | the owner, on stills of each removal |

The thresholds are proposals for the owner. Also reported: the drift check's flag counts, and the
render stage's time and memory.

## 6. Risks

- **Mask errors become permanent.** About 30% of DAVIS's annotated objects are never proposed and
  will be owned by background. Masks that bleed across a neighbour give the neighbour's pixels to
  the wrong owner at birth. The edge erosion (§4.1) limits the second, not the first.
- **Contact and occlusion.** lab-coat's women stand shoulder to shoulder. A pixel claimed by two
  masks is ignored, so the loss says nothing exactly where separation is hardest.
- **PSNR cost.** A constraint on where colour may come from can cost fidelity; the SoM mask term
  cost 1.9 dB for a different reason. Unknown until measured.
- **Criterion 3 needs consistent geometry.** If cameras or depth disagree across frames, seen
  background lands in the wrong place, and T3 fails whatever the removal does. The control in T3
  separates the two; the fix is SLAM It!, not this PR.
- **Motion errors.** Canonical neighbours fly apart on the worst 10% of frames (identity doc §4.4).
  Under fixed owners this shows as drift flags, not wrong labels.

## 7. Decisions (the owner, 2026-10-17)

1. **Clips: chosen by measurement.** Once T1–T4 exist, rank the DAVIS clips by how much of the
   background behind each entity the video saw (T4's complement), propose the top few, and the
   owner picks. Those clips are both the gate set and the site's. The ranking uses the existing
   runs' geometry; the picks are re-checked after their rebuild.
2. **Pass marks as in §5.** The owner's look at stills is the final gate.
3. **Unmapped tracklets: the reserved *unaddressed* owner.** Never removable; ignored by the loss.
4. **Keep the old paths.** FlashSplat selection and the learned identity (I2/I3) stay, for
   comparison, after ownership passes; ownership becomes the default removal (§4.4).
5. **Drift check: flag-only first.**

Settled before: unseen background is left as is (criterion 4); a blur stays a cosmetic option
for the site.

## 8. Build order

One branch, one pull request, a commit per step:

1. This document.
2. T1–T4 as a measurement tool, and the baseline: today's removal on `exp_a_lab-coat` and the ten
   DAVIS runs.
3. The clip ranking, for the owner to choose from.
4. Ownership: owners at seeding, inheritance through densification, the silhouette loss in a
   second pass, removal by owner, the drift check; scene 5 → 6, render 4 → 5, handles 2 → 3.
5. The chosen clips rebuilt on the option-A geometry, each trained with and without the loss.
6. Today's removal against ownership on all tests, and stills for the owner.

## References

- Street Gaussians — Yan et al., ECCV 2024. [arXiv 2401.01339](https://arxiv.org/html/2401.01339v1)
- OmniRe — Chen et al., ICLR 2025. [arXiv 2408.16760](https://arxiv.org/html/2408.16760v1)
- DreamScene4D — Chu, Ke, Fragkiadaki, NeurIPS 2024. [arXiv 2405.02280](https://arxiv.org/html/2405.02280)
- Gaussian Grouping — Ye et al., ECCV 2024. [arXiv 2312.00732](https://arxiv.org/abs/2312.00732);
  removal and inpainting steps from `third_party/gaussian-grouping/docs/edit_removal_inpaint.md`
- FlashSplat — Shen, Yang, Wang, ECCV 2024. [arXiv 2409.08270](https://arxiv.org/html/2409.08270v1)
- `docs/design/2026-09-29-per-gaussian-identity-scope.md` (§3.1–3.3, §4.3–4.9, §8.1)
