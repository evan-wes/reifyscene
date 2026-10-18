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
beyond the photos and our own masks, except where noted). As built in
`reifyscene/eval/benchmarks/removal/` (`scripts/benchmark_removal.py`). Every test reads the
geometry's depth (the pose-refined geometry the scene was built from, normalised as the scene
stage normalises its targets), never a render's, so a test does not move with the method it
judges.

**1. No visual trace of the entity in any rendered image.**
- *Test T1 (independent of the removal method):* the entity's category, found in the
  after-render. One detector serves every category: the pipeline's Mask R-CNN over the COCO
  categories the graph's labels come from, with SAM 2 box prompts (`InstanceSegmenter`,
  `reifyscene/eval/benchmarks/edit_damage/silhouettes.py`). On each view the detector first finds
  the entity in the **before**-render (score ≥ 0.5, the silhouette with IoU ≥ 0.3 against where
  the entity is: its id-map mask on a training view, where the Gaussians inside its volume show on
  an orbited one); a view where it cannot is not judged. A *trace* is a detection of the category
  in the after-render (score ≥ 0.3: a faint ghost counts) with IoU ≥ 0.3 against that
  silhouette, so a neighbour of the same category (lab-coat's adjacent women) does not count.
  Views: every training frame with ≥ 200 of the entity's pixels, and eight of them orbited
  ±20°. Reported: the share of judged views without a trace, for each set.
- *Diagnostic:* the residual share of §0 — how much of the before-render inside the entity's
  pixels came from Gaussians the removal kept.

**2. In 3D, the space it occupied is empty, at every moment.**
- *Test T2:* per training frame *t*, the entity's volume *V_t*; after removal, the opacity mass of
  remaining Gaussians whose positions at *t* (`DynamicScene.cloud_at(t)`) fall inside *V_t*,
  as a share of the same mass before removal.
- *V_t* for people: the Sapiens body volume (`reifyscene/edit/body_volume.py`, `sapiens_body` /
  `lift_frame`: capsules round the bones, from the geometry's depth over the person's mask). Frames
  without a torso use the slab below.
- *V_t* otherwise: a slab — the entity's own pixels, from a quarter of its width in front of the
  geometry's surface there back to the surface plus the entity's own width at *t* (the 5th–95th
  percentile spread of its pixels across the image, in scene units). A slab, not a hull
  (identity scope doc §3.2: a hull reaches into neighbours). Reported: the remaining opacity of
  all frames together over that before.

**3. Background the video saw appears, unoccluded, where the entity was.**
- *Test T3:* for frame *t* and each pixel *p* of the entity's mask, forward-project every other
  frame's background pixels (id map = background, valid depth, at least 2 px from anything that
  is not background) into *t* through the training cameras, each onto the pixels less than a
  pixel from where it lands, z-buffered. A point more than 5% in front of *t*'s own surface at
  *p* is dropped (nothing can be in front of what the photo at *t* shows). *p* is **seen** if a
  point lands on it; its expected colour is the photo of the frame nearest in time among those
  whose point lies within 5% of the nearest surface. Score: PSNR of the after-render against the
  expected colour over seen pixels, all frames pooled.
- *Control (the geometry floor):* the unedited render against the same projection on background
  pixels 4–20 px from the entity and at least 4 px from any mask, where nothing occludes. Right
  at a mask's edge the photo still shows the entity (hair, a sleeve): on lab-coat's person 8 the
  floor read 17.0–17.6 dB with no gap and 20.2–21.5 dB with gaps of 2–8 px. The floor also
  depends on how far in time the frame that saw a pixel is: on lab-coat's person 2 it is 23.1 dB
  from the next frame and 16.0 dB from frames 16 or more away, and 47% of what lies behind her
  was seen only from that far, while the band is seen from the next frame. So the floor is
  measured per band of source distance (1, 2–3, 4–7, 8–15, 16+ frames: the band projected from
  frames at least that far) and weighted like the entity's seen pixels. If T3 is no worse
  than the floor, the removal revealed what the video saw. If both are poor, the cameras and
  depth disagree across frames — a geometry problem (SLAM It!), not a removal one. The photo
  against the projection on the same pixels is also reported (a geometry-only floor).
- *Limit:* only background is projected. Where another entity stands behind the removed one,
  the expected colour is the background behind both.

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
(`reifyscene/graph/id_maps.py`): `0` background, `1..K` graph nodes, `65534` **unaddressed**
(only a tracklet the graph dropped claims it; new in id-map format 2), `65535` ignore (overlap,
padding), in the geometry's raster, built from the same mapping the removal uses
(`render_pixel_masks`). The scene stage gains `reid` and `graph` upstream (both run before it)
and exports the id maps itself.

- **Foreground seeds** are 2D tracks lifted to 3D (`lift_tracks_3d`,
  `reifyscene/lifting/native_track_lift.py`), each sampled at a query pixel in a query frame. Its
  owner is the id map at that pixel. Seeds within 2 px (a square) of another code, or on an
  ignore pixel, are not used: mask edges are where SAM is least reliable (`owner_purity`).
- **Background seeds** come from background pixels only (`lift_bkgd_points`): owner 0.
- **Unmapped tracklets** (a segment the graph dropped) sit inside the foreground union and seed
  Gaussians too. Their owner is the id-map code `65534`, the reserved *unaddressed* owner: never
  removable, and its pixels carry no loss.
- **Static tracks** the motion gate moves into the background field keep their owners.
- **Owner is not motion.** fg/bg is the motion split; ownership is semantic and lives on both
  fields (identity scope doc §3.3). A static entity under the motion gate is a background-field
  Gaussian with owner *k*, still removable.

The owner is an integer buffer on each field (`GaussianField.owner`: the id-map code), not a
parameter and never optimised, plus a class table (code → node id) on the scene and in the bundle.

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

- **Loss:** per labelled pixel (background or an entity), the summed absolute difference between
  the rendered owner shares and the id map's one-hot, halved to [0, 1]: 0 when the pixel's own
  owner alone covers it fully, 0.5 when nothing covers it, 1 when another owner does. As in
  Shape-of-Motion's mask term, the worst 2% of pixels are left out (a mask is wrong somewhere in
  every frame), and the weight is theirs, 1.0 (`lambda_owner`). The unaddressed owner has a
  channel of its own and is wrong on every labelled pixel.
- **Gradients reach geometry** — means, scales, opacities, motion coefficients. That is the point.
- **Densification sees photometric gradients only.** The owner channels go in a second
  rasterization, so the `means2d` gradient densification thresholds on is the photo pass's. The
  coupled-identity run lost 23% of its foreground Gaussians when identity gradients reached it
  (identity doc §4.3). Cost: one more pass per view (not yet measured).
- **Background** Gaussians are supervised to cover background pixels. Behind an entity nothing
  supervises them — that is criterion 4's unseen region, by definition.
- **Why not the −1.9 dB mask term again:** that term drove the foreground *field* to match every
  segmented instance, forcing a motion split to carry identity (render stage, 2026-10-01). Here
  ownership is independent of the field split. Its PSNR cost is measured in §5.

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

Every 500 steps and at the end (`owner_drift_every`), compute FlashSplat's evidence
(`evidence_matrix` over every training view) and compare it with each Gaussian's owner. A
Gaussian owned by an entity or the background whose evidence goes clearly (slackness 0) to
another class is **flagged, never relabelled** — per identity doc §3.1, a Gaussian that is
genuinely A at one time and B at another makes `remove` ill-posed, and the remedy is to split or
prune it. First version: count flags per (owner, evidence) pair and keep them in the render
record (`scene_state.ownership.drift`); act (prune, or split — unproven) only if the counts are
non-trivial.

### 4.6 Containment of the background

The owner's look at lab-coat (`own_lab-coat_hold_on`, 2026-10-17) found the middle woman's
removal still leaving a ghost at her boundaries, and the owner-coloured orbit showed why: the
streaks behind the people are **background-owned** Gaussians stretched along the line of sight.
Nothing in training reaches them. Every loss works through a render, and a Gaussian hidden behind
a person in every frame gets almost no gradient; the depth prior there is the person's depth; and
Shape-of-Motion's only shape term is a variance of scales. The literature review
(`reports/Gaussian object identity and containment.md`) found no published fix aimed at this,
and two that carry over:

- **Scale-ratio cap** (`lambda_scale_ratio`, `max_scale_ratio`, `scale_ratio_every`):
  splatfacto's term after PhysGaussian, the mean of max(longest / shortest axis, *r*) − *r* over
  background-owned Gaussians, at its defaults (*r* = 10, weight 0.1, every 10 steps). It acts on
  each Gaussian alone, so it needs no second view.
- **Ghost prune** (`ghost_max_bg_share`): the drift check's evidence (§4.5) is already an
  occlusion-aware record of where each Gaussian's blending weight lands, split by id-map class,
  over every training view. At every drift check but the last, a background-owned Gaussian whose
  weight lands on entities' pixels, with under this share (0.1) of it on the background's, is
  pruned: it paints only an entity, so removing the entity would leave it behind. The last check
  only counts, so that every prune is followed by training that lets the entity's own Gaussians
  fill in. Each check's record carries `ghosts`, and `pruned` (all prunes at that check).

Nothing in training made the people occupy the space of standing people either: seeding puts
their Gaussians on the surface the camera sees, the depth term pins that surface and nothing
behind it, and the owner loss is a silhouette, which a Gaussian stretched straight back along
the line of sight still satisfies. **Body containment** (the owner's question, 2026-10-17) acts
on each Gaussian directly, not through a render, so it reaches Gaussians hidden in every view:

- **Bodies at the scene stage.** Each person's Sapiens body (`reifyscene.edit.body_volume`,
  the body T2 measures against) is built on every training frame from that frame's own depth
  target, owner pixels and camera, and written into the bundle (`target_body_*`): capsules,
  their frame and owner, each body's torso length and clothing margin.
- **The loss** (`lambda_body`, `body_sigmas`): an entity's Gaussian may reach (centre plus 2
  standard deviations) inside its body, or up to a band behind its own surface where the frame
  shows the entity at its pixel -- the clothing margin for a person, so loose clothes and hair
  outside the capsules are not squeezed in, and one entity width for anything without a body
  on that frame (T2's slab). Its excess over the nearer of the two, in its entity's size, is
  averaged over the Gaussians the view can judge. Another entity's Gaussians behind it are left
  alone: one person standing behind another is real. The nearest published form is drivestudio's
  soft out-of-bound loss on deformable nodes, against a tracked box this pipeline does not have.
- **Background in a body** (`prune_background_in_body`): at the same checks, background-owned
  Gaussians whose centre lies in a body's core (capsules less the clothing margin) on some
  training frame are pruned; that space was a person's at that moment (criterion 2). The core,
  not the whole body, so the ground under a foot is not taken. Recorded as `in_body`.

**Measured on lab-coat, and the owner's look (2026-10-17).** First on `own_lab-coat`'s scene,
the ratio cap and the ghost prune (arm A) against the owner loss alone (`own_lab-coat_hold_on`):
background-owned Gaussians drawn mostly on an entity's pixels at the last drift check 11,655 →
1,838, held-out PSNR 22.08 → 21.72 dB; the owner judged A's removals cleaner. Body containment
needs the bodies in the bundle, and the scene stage is not deterministic (its seeds differ from
one build to the next), so every arm below trains on one rebuilt scene
(`own_lab-coat_contain_b`'s), with every 6th training frame held out:

| on one scene | ownership training off | A2: owner loss, ratio cap, ghost prune | B: A2 + body containment and its prune | C: B with the owner loss at 0.1 |
|---|---|---|---|---|
| PSNR, training frames | 27.95 dB | 24.86 dB | 24.65 dB | 26.27 dB |
| PSNR, held-out frames | 21.77 dB | 21.41 dB | 21.65 dB | 21.34 dB |
| Gaussians drawn mostly in another owner's pixels (last drift check) | 94,546 | 6,721 | 5,905 | 24,436 |
| … background-owned ones, drawn mostly on entities | 29,264 | 1,789 | 1,789 | 1,863 |
| background-owned Gaussians whose longest axis is over 10× their shortest | 178,086 of 378,122 | 3,216 of 425,012 | 3,215 of 425,351 | 1,793 of 391,307 |
| middle woman removed: share of her pixels drawn by background outside her body | 0.10 | 0.32 | 0.38 | 0.26 |
| … by background inside her body | 0.55 | 0.14 | 0.16 | 0.15 |
| … by the other two women inside her body | 0.25 | 0.21 | 0.15 | 0.17 |
| … by nothing | ≈ 0 | ≈ 0.04 | ≈ 0.10 | ≈ 0.23 |

The removal rows are over the frames where she covers at least 200 px, split by whether the
drawing Gaussian's centre lies in her T2 volume; "nothing" is what the rows over 0.005 leave.

- **The owner's look:** B removes more cleanly than A2; C leaves a black void where B leaves a
  natural fill; B is a little softer than the arm without ownership training, otherwise alike.
  **All four are on by default** whenever the scene has owners, like the owner loss, and the
  owner loss stays at 1.0.
- **Without ownership training the ghost is background-owned:** over half of the middle woman's
  pixels are still drawn after her removal, by background Gaussians inside her body.
- **The background-in-body prune does not lower that share on its own** (A2 0.14, B 0.16): it
  takes 344–2,554 Gaussians per check, but only centres in the core, while the measurement counts
  the clothed body, and densification refills between checks.
- **The ghost prune peaks after the opacity reset** (step 3000), when hidden background shows
  through the people: 12,427 in B there, 426–1,279 at the other checks.

### 4.7 Format

The bundle gains `fg_owner` / `bg_owner`, the class table `owner_node_ids`, and the id maps the
loss trains on (`target_ids`, beside the depth targets), so the render stage still reads only
the scene. Contract versions: scene 5 → 6 (seeds carry owners; upstream adds reid and graph),
render 4 → 5 (silhouette loss; the checkpoint carries owners), handles 2 → 3 (the selection is
ownership). Clean break: no reader for old checkpoints. The runs the site needs are rebuilt
from the scene stage on.

## 5. Measurement and gate

On the same clips as the baseline, today's removal vs ownership:

| test | proposed pass |
|---|---|
| T1 trace | no trace on ≥ 95% of frames, per entity |
| T2 space | remaining opacity mass in *V_t* ≤ 5% of before |
| T3 seen background | within 1 dB of the geometry floor |
| T4 unseen | reported, no pass mark |
| held-out PSNR | ≤ 0.5 dB below the same run without ownership training (the identity work's noise band), on every 6th training frame held out; the training-frame gap reported |
| by eye | the owner, on stills of each removal |

Also reported: the drift check's flag counts, and the render stage's time and memory. The two
arms: `configs/experiment/davis_graph_loop_gate_ownership_on.yaml` and
`davis_graph_loop_gate_ownership_off.yaml` (the off arm resumes the on arm's run at the render
stage, so both train on one scene).

**Why held-out frames, not training frames** (the owner, 2026-10-17; the mark was first on
training frames). On lab-coat (§4.6's scene) B is 3.30 dB below the arm without ownership
training on training frames and 0.12 dB on held-out frames; on `own_lab-coat`'s scene the owner
loss alone was 2.31 dB below on training frames (25.49 against 27.80) and 0.18 dB above on
held-out ones (22.08 against 21.90). By region (PSNR pooled over pixels):

| region | training frames, off | training frames, B | held-out, off | held-out, B |
|---|---|---|---|---|
| background, beyond 3 px of a mask edge | 27.87 | 23.86 | 21.23 | 21.33 |
| mask edges, ±3 px | 24.76 | 22.40 | 18.09 | 17.93 |
| entity interiors | 27.73 | 26.43 | 19.14 | 18.61 |

In the arm without ownership training, on training frames, 31% of the weight on those
background pixels comes from the people's moving Gaussians; in B, 0.2%. B's extra error sits on
thin, high-contrast background -- fence posts and rail, branches against the sky -- not on the
grass or the shadows: the pattern of a background that does not land in the same place on every
frame (camera and depth error), patched frame by frame by the people's moving Gaussians. The
patch does not carry to frames it was not fitted to, and it is what a removal takes along with a
person. A crisper
background is the geometry's to give (pose refinement, SLAM It!), not this design's. What
ownership itself costs is the interiors' 0.5 dB on held-out frames.

**The result on lab-coat, and the decision (the owner, 2026-10-17).** The three people
(`benchmarks/removal_ownership_lab-coat.json`), the removal this design replaces (FlashSplat with
the filter, on the arm without ownership training) → removal by owner on the ownership arm:

| | T1 training views clean | T2 opacity left | T3 dB below the floor |
|---|---|---|---|
| left woman | 0.152 → 0.915 | 0.214 → 0.014 | 7.47 → 2.36 |
| right woman | 0.064 → 0.674 | 0.263 → 0.192 | 5.85 → 3.37 |
| middle woman | 0.021 → 0.170 | 0.313 → 0.169 | 4.44 → 5.51 |

No person passes every mark. The left woman passes T2 and nearly T1. The middle woman, between
the other two, keeps a trace on 83% of the judged training views. Where she stood, her stills
show a pale column smeared out of the background no frame saw behind her and her neighbours'
coats -- the likeliest thing the detector still reads as a person. Her T3 is 1.1 dB below the
replaced removal's. The
owner judged the removals clearly better by eye and chose to merge on that: **the gate this
design passed is "better than the replaced removal on T1–T3 for the people, and the owner's
look"**. The absolute marks stay the target for the work that follows: contact between people,
the unseen column, and T3.

## 6. Risks

- **Mask errors become permanent.** About 30% of DAVIS's annotated objects are never proposed and
  will be owned by background. Masks that bleed across a neighbour give the neighbour's pixels to
  the wrong owner at birth. The edge erosion (§4.1) limits the second, not the first.
- **Contact and occlusion.** lab-coat's women stand shoulder to shoulder. A pixel claimed by two
  masks is ignored, so the loss says nothing exactly where separation is hardest.
- **PSNR cost.** A constraint on where colour may come from can cost fidelity; the SoM mask term
  cost 1.9 dB for a different reason. Measured on lab-coat (§5): 3.30 dB on training frames, 0.12
  dB on held-out ones.
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
2. **Pass marks as in §5.** The owner's look at stills is the final gate. The PSNR mark is on
   held-out frames, after §5's measurement (it was on training frames).
3. **Unmapped tracklets: the reserved *unaddressed* owner.** Never removable; ignored by the loss.
4. **Keep the old paths.** FlashSplat selection and the learned identity (I2/I3) stay, for
   comparison, after ownership passes; ownership becomes the default removal (§4.4).
5. **Drift check: flag-only first.**
6. **Containment on by default, the owner loss at 1.0** (the owner's look at lab-coat, §4.6).
7. **Merge on lab-coat alone, as better than the replaced removal** (§5); the absolute marks are
   unmet and recorded there. The clip ranking (decision 1) waits on a cleaner graph -- its top is
   noise entities -- and the other clips are rebuilt with the site's.

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
5. The chosen clips rebuilt on the option-A geometry, each trained with and without ownership
   training (§5's two arms). Built: lab-coat only, and containment (§4.6) before it; steps 3
   and 5–6 for the other clips wait (§7, decision 7).
6. Today's removal against ownership on all tests, and stills for the owner.

## References

- Street Gaussians — Yan et al., ECCV 2024. [arXiv 2401.01339](https://arxiv.org/html/2401.01339v1)
- OmniRe — Chen et al., ICLR 2025. [arXiv 2408.16760](https://arxiv.org/html/2408.16760v1)
- DreamScene4D — Chu, Ke, Fragkiadaki, NeurIPS 2024. [arXiv 2405.02280](https://arxiv.org/html/2405.02280)
- Gaussian Grouping — Ye et al., ECCV 2024. [arXiv 2312.00732](https://arxiv.org/abs/2312.00732);
  removal and inpainting steps from `third_party/gaussian-grouping/docs/edit_removal_inpaint.md`
- FlashSplat — Shen, Yang, Wang, ECCV 2024. [arXiv 2409.08270](https://arxiv.org/html/2409.08270v1)
- `docs/design/2026-09-29-per-gaussian-identity-scope.md` (§3.1–3.3, §4.3–4.9, §8.1)
