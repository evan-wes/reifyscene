# Per-Gaussian identity — making the trained scene separable by entity

**Status:** PROPOSED
**Date:** 2026-09-29
**Unblocks:** step 1 of `docs/design/2026-07-09-queryable-graph-definition-spike-plan.md` (§5a: the
gate failed, 1 of 10) and, behind it, G1–G4 of
`docs/design/2026-09-20-conceptgraphs-verdict-and-graph-plan.md`
**Overturns:** §2.4 of `docs/research/2026-09-12-third-party-components-for-the-back-half.md`
("Entity → Gaussian grounding — keep ours, cite theirs")
**Adopts:** Gaussian Grouping (ECCV 2024), read at `third_party/gaussian-grouping` @ `0ab60af`

---

## 0. Why

The step-1 gate was run on ten sequences and one passed. On `lab-coat` the target's 2D mask is a
clean person silhouette, and removing her Gaussians **tears the heads off the two people standing
beside her**. The bridge code is not what failed — `blackswan` removes 99.5% of its entity through
the same code path.

The cause is in the render stage, and it is simple: **the entity masks never reach training.** The
dynamic trainer's whole objective is

```
loss = 0.8·L1(pred, gt) + 0.2·(1 − SSIM) + λ·motion_smoothness
```

No mask term, no entity term. No `instance`, `segment` or `dyn_mask` input reaches the render
stage at all, and `object_branch.py` — the module named for this — is a seven-line placeholder
whose `attach()` returns `None`. The scene stage's own docstring records the boundary as
deliberate: *"Entity semantics … are a graph-stage concern and live no longer here."*

So one undifferentiated foreground field is optimised to reproduce pixels, and entity identity is
applied afterwards by projecting the finished Gaussians into masks and taking a majority vote
(`graph/entity_gaussians.py`). **The representation was never asked to be separable, so it is
not.** `blackswan` works because a single isolated object makes the vote unambiguous.

Our post-hoc vote is also the naive form of the idea. `project_to_views` returns
`in_front = z > 1e-6` — "not behind the camera", not "visible". A Gaussian fifty metres behind a
person, projecting inside her silhouette, counts as evidence that it is her.

---

## 1. The method, as read from source

Gaussian Grouping gives each Gaussian a learnable identity and supervises it through the
renderer, so occlusion is handled by alpha compositing rather than by a hand-rolled visibility
test. From `third_party/gaussian-grouping` @ `0ab60af`:

| piece | what it is |
|---|---|
| identity | `_objects_dc`, a learnable **16-dim** vector per Gaussian, beside colour and opacity |
| rasterization | emitted through the same alpha compositing as colour → a 16-channel identity image (`render_object`) |
| classifier | `torch.nn.Conv2d(16, num_classes, kernel_size=1)`, trained jointly, `lr 5e-4` |
| 2D loss | cross-entropy of `classifier(render_object)` against the frame's integer mask ids, divided by `log(num_classes)` |
| 3D loss | `loss_cls_3d`: sample points, take top-k nearest neighbours **in 3D position**, penalise KL divergence between a point's class distribution and its neighbours' |
| total | `(1−λ)·L1 + λ·(1−SSIM) + loss_obj [+ loss_obj_3d]` |
| removal | classify each Gaussian from its identity, select `softmax(logits)[target] > thresh`, union with the convex hull of that set, delete |

**Input contract:** a folder of per-frame PNGs whose pixel values are integer object ids,
consistent across frames. Nothing more.

The 3D term is what addresses `lab-coat`: it forces spatial neighbours to agree, so a Gaussian
inside person A cannot quietly carry person B's label.

**Licensing.** The repository is Apache-2.0, but `submodules/diff-gaussian-rasterization` carries
the Inria/MPII Gaussian-Splatting research licence — they modified Inria's CUDA rasterizer to emit
the extra channel. We do not touch it (§2), so the question does not arise; this project is
non-commercial in any case.

---

## 2. What is different here, and why most of it is easier

**We supply the identity channel through gsplat, not their rasterizer.** `gsplat.rasterization`
is generic over channel count — it is feature rasterization, not RGB-specific. Our wrapper
hardcodes `render_mode="RGB+ED"` and slices `img[..., :3]`. Extending it is a change to one
function. Adopting their rasterizer instead would mean replacing our entire render backend, which
is far larger than the thing we came here for.

**We already have consistent ids across frames, so DEVA is not needed.** They bundle a video
tracker because SAM hands them unassociated per-frame masks. SAM 2 propagation plus the reid merge
map already gives us track ids that are stable through a clip, and the graph collapses them to
canonical node ids. Writing those out as id-map PNGs, through the geometry's recorded
`ImageTransform`, is the whole of our data preparation. **Their hardest supporting machinery is
the part we skip.**

**Ours is dynamic; theirs is static — and identity is the one thing that does not care.** Identity
is a property of the Gaussian, not of the frame: a person is the same person at t=0 and t=40, and
what changes is deformation, which the motion bases already carry. An identity vector sits exactly
parallel to `motion_coefs` as a per-Gaussian `(G, 16)` parameter, and our densifier inherits it
for free — *"All other params (quats, opacities, colors, motion_coefs, …) are inherited from the
parent"*. The 3D regularizer runs on **canonical** means, the deformation-invariant frame, which
is a cleaner fit than their static `_xyz`.

---

## 3. Three decisions, taken deliberately

### 3.1 Identity is time-invariant — and the conflict is the instrument

A Gaussian could drift, ending up on entity A at one time and entity B at another. We know this
happens: `motocross-bumps` swung 2.3% → 73.5% → 6.9% across three frames of one clip.

**We do not let identity vary with time.** If a Gaussian is genuinely A at t=0 and B at t=40, then
`remove(A)` has no correct answer — delete it and B is damaged, keep it and A is still visible.
The decomposition is wrong, not the labelling, and a time-varying identity would let the model
*describe* that confidently rather than fix it.

Under time-invariant identity, a drifting Gaussian receives conflicting gradients and converges to
low confidence in every class. That is worth more than it costs:

- **it fails safe** — the probability threshold excludes it from removal rather than confidently
  deleting something half-owned by someone else;
- **it is a measurement we do not currently have** — per-Gaussian identity entropy is a direct
  read on where the motion model is wrong. Building the time-varying version first would destroy
  this diagnostic, because a model that can explain drift stops reporting it.

The principled response to *genuine* drift is to **split the Gaussian** — two Gaussians, each with
its own identity and its own motion coefficients — which keeps edits well-posed and uses
densification machinery we already have. **Flagged as unproven:** the identity mechanism and the
3D regularizer come from a published method; split-on-identity-conflict does not, and it is out of
scope here. §5 measures whether it is needed.

### 3.2 No convex hull

Their removal unions the selected set with its convex hull, to catch interior Gaussians that never
earn a confident label because they are never visible. **We do not.** A person mid-stride has a
hull that contains the air between their limbs, and on `lab-coat` one person's hull reaches
straight into her neighbours — amplifying the exact failure this work exists to fix.

The hull matters more for their next step, inpainting, where leftovers could show through the
fill. For removal an occluded interior Gaussian contributes almost nothing to the image, so
leaving it still leaves a hole.

If interior leakage shows up, the replacement is **local**: include an unlabelled Gaussian when
its nearest labelled neighbours agree on the target. Same signal the 3D regularizer already uses,
and it cannot jump a gap into another entity. Off by default until measured, like every other
threshold here.

### 3.3 The background field carries identity too

**Foreground and background are split by motion; identity is semantic. Conflating them is a bug.**

Today foreground is the union of the segment masks (`assembler._fg_mask_by_frame`), so a detected
person is foreground whether or not they move, and would receive an identity. But the split is one
config flag away from being motion-based: `motion_gate` demotes never-moved tracks to the static
cloud, and its own comment says why it exists — *"static tracks (e.g. SAM-2-segmented background
the segment fg let in) become static bg"*. It is not set in any config today.

Turning it on looks like a reasonable optimisation and would **silently make a standing person
unaddressable** — swept into an undifferentiated static cloud with no identity and no way to name
them. A person leaning on a wall is semantically an entity and dynamically static; those two facts
must not be forced into one decision.

So identity goes on **both** fields. A static entity keeps its handle and stays removable; the
motion gate goes back to being a question about deformation, which is all it should ever have
been. The cost is small — `blackswan` carries 120,634 background Gaussians to 87,599 foreground, so
a 16-dim vector on both is a few megabytes.

What this does **not** fix: an entity the segmenter never proposes is background by omission, and
no identity scheme recovers it. That is the 73-unproposed-object problem, and it stays where it is.

### 3.4 The learned identity is a rung, not a replacement

`graph/entity_gaussians.py`'s projection vote stays as the floor rung: it needs no retraining and
runs on any existing checkpoint. The learned identity is the rung above it, and the
`CapabilityManifest` records which produced a given run's `renderable`. Same shape as every other
stage.

---

## 4. Work

| | build | done when |
|---|---|---|
| **I0** ✅ | id-map export: per-frame PNGs of canonical node ids, through the recorded `ImageTransform` | ids match the graph's nodes; a round-trip read reproduces the masks the handle uses today — **done, see §4.1** |
| **I1** ✅ | gsplat wrapper carries an N-channel feature alongside RGB | a known feature rasterizes to the expected image; RGB output is unchanged for an existing checkpoint — **done, see §4.2** |
| **I2** ✅ | identity parameter (**D=20**, §4.2) on the dynamic field **and the background field** + classifier + 2D loss | identity survives densification and pruning; a static segmented entity is addressable; the one-batch overfit reaches ≈ zero classification loss — **done, see §4.3; geometry detached** |
| **I3** ✅ | 3D consistency regularizer on canonical means (fg) and 3D means (bg), identity only (§4.4) — **done, see §4.5** | neighbouring Gaussians agree measurably more than without it, on all ten; PSNR within noise of each scene's control |
| **I4** | removal by probability threshold; no hull | the ten-scene gate re-run and re-reviewed against the recorded baseline (`blackswan` alone) |
| **I5** | the conflict measurement (§5) | reported for all ten, and `motocross-bumps` explained |

### 4.1 I0 — done

`reifyscene/graph/id_maps.py`, run by `scripts/export_id_maps.py`, writes
`<run>/identity/frame_NNNNN.png` (uint16) and `manifest.json` for every ingest frame. Encoding:
`0` background, `1..K` the graph's entity nodes in sorted `node_id` order, `65535` **ignore**.
Ignore covers three things: a pixel two entities claim (picking a winner would train a guess),
a pixel a tracklet the graph dropped covers (background would teach that the thing is scenery),
and transform padding. `num_classes` (= K + 1) lives in the manifest because the loss's
normalisation depends on it.

All ten runs round-trip with **zero** mismatched pixels against `entity_masks_in_render_pixels`:

| run | classes | entity % | overlap % |
|---|---|---|---|
| blackswan | 9 | 10.0 | 0.18 |
| crossing | 20 | 19.3 | 0.20 |
| dogs-scale | 17 | 77.0 | 2.56 |
| gold-fish | 19 | 23.5 | 0.27 |
| india | 24 | 26.0 | 0.97 |
| judo | 19 | 72.0 | 1.48 |
| lab-coat | 9 | 36.7 | 0.03 |
| lindy-hop | 21 | 67.3 | 1.50 |
| miami-surf | 7 | 29.6 | 0.00 |
| motocross-bumps | 4 | 34.1 | 0.00 |

Unmapped and padding are 0% on all ten (every tracklet has a node; the transform is a pure
resize). What it surfaced:

- **Classes are 4–24, not DAVIS's 1–8.** The class table is every entity node the graph has,
  clutter proposals included (`blackswan` carries a "spoon", a "toothbrush", two "cups"). That is
  the contract — ids match the graph — but it means the feature width cannot be one channel per
  class under the measured `D=19` budget on `india` (24) or `lindy-hop` (21). I2 must choose:
  a fixed-width embedding plus a linear classifier (Gaussian Grouping's shape), or a per-scene width.
- **Overlap is small** — under 3% of pixels everywhere — so ignoring it costs little supervision.
- **Two `gold-fish` nodes (both "tv") own no unambiguous pixel on any frame**: every pixel they
  have is contested. Near-certain duplicates; they get a class and no supervision, which fails safe.

### 4.2 I1 — done; the identity width is D=20

`GaussianCloud.features` (N, D) and `rasterize(..., features=)` carry D channels in the **same**
gsplat call as RGB — concatenated onto the colours, so they are sorted and alpha-composited with
exactly RGB's weights — and `RasterOutput.features` (H, W, D) returns them with grad. Without
`features` the call is unchanged. `concat` refuses mixing clouds with and without features:
zero-filling would assert "no identity" for Gaussians never given one.

Done-when, in `tests/test_raster_features.py`: a known vector renders as vector × alpha
(D = 1, 13, 16, 20); overlapping Gaussians give features identical to RGB; the gradient reaches
exactly the right channel of the right Gaussian; and **`loop_blackswan`'s trained checkpoint
(208k Gaussians) renders bit-identical RGB, depth and alpha with 20 random channels attached.**

**The width is set by gsplat's kernels, not by class count.** gsplat 1.5.3 compiles
`rasterize_to_pixels` for 1–5, 8, 9, 16, 17, 24, 32, 33, 64, … channels and zero-pads anything
else to the next power of two. Our call is `RGB+ED`, so it renders `3 + D + 1` channels.
`scripts/bench_raster_channels.py`, RTX 5090 at 400 W, 518×294, forward + backward:

| D | channels | kernel | synthetic 80k | blackswan 208k |
|---|---|---|---|---|
| 0 | 4 | 4 | 0.95 ms | 2.55 ms |
| 13 | 17 | 17 | 1.05 | 3.76 |
| 16 | 20 | 32 | 1.45 | 5.70 |
| 19 | 23 | 32 | 1.47 | 5.73 |
| 20 | 24 | 24 | 1.40 | 4.54 |
| 28 | 32 | 32 | 1.62 | 5.76 |

**D=20** is chosen: unpadded, cheaper than Gaussian Grouping's 16 on a real scene (4.54 vs
5.70 ms), and a linear classifier from 20 dims covers the 4–24 classes I0 found. It costs about
2 ms per step over plain RGB at 208k Gaussians.

### 4.3 I2 — done; identity reads geometry and never writes it

Both fields carry a learnable identity vector (D=20); a linear classifier maps rendered
features to the run's classes; the loss is `CE(logits, id_map, ignore=65535) / log(C)` at
λ=1, with Gaussian Grouping's learning rates. Identity is in each densification controller's
parameter list, so splits and prunes carry it. The render stage exports fresh id maps when
`dynamic.identity.enabled`; the checkpoint carries both fields' vectors, the classifier and
the class table. `scripts/retrain_with_identity.py` writes beside the pipeline's `render/`.

**The departure from Gaussian Grouping: the identity loss does not reach geometry.** Theirs
renders identity in the same pass and lets its gradient move means, scales and opacities. The
photometric loss matches measured pixels; our id maps are pseudo-labels — SAM 2 masks through
reid and the graph, with ~30% of annotated objects never proposed and therefore labelled
*background*. Letting those errors shape geometry is letting the weakest signal overrule the
strongest. Measured on `lab-coat` (5000 iters, one seed each):

| | control (no identity) | coupled (theirs) | **detached (ours)** |
|---|---|---|---|
| train-view PSNR | 25.63 | 24.42 | **25.79** |
| fg / bg Gaussians | 72.7k / 241.2k | 55.9k / 265.4k | 73.1k / 244.9k |
| final identity CE (normalised) | — | 0.030 | 0.043 |
| mean identity entropy fg / bg | — | 0.42 / 0.26 | **0.27 / 0.16** |

Coupling cost 1.2 dB and 23% of the foreground Gaussians (identity gradients enter
`means2d.grad`, which densification thresholds on) and bought a lower loss on labels we do not
fully trust — while separating *worse*. Detached, the features get a second, feature-only
rasterization over a geometry-detached cloud (~2 ms/step, §4.2); densification sees
photometric gradients only. `identity_detach_geometry=False` keeps the coupled mode for
comparison.

**Static entities are addressable**: each of `lab-coat`'s three people owns 12.6k–14.6k
background Gaussians at p ≥ 0.5 — the ones the motion split put in the static cloud, which the
projection vote could not reach.

**Next question the label source raises**: DAVIS ships hand-annotated masks for its main
objects. Training identity on those instead of ours would separate "does the mechanism work"
from "what do our upstream labels cost" — if GT-trained identity passes the gate and ours does
not, the remaining problem is segmentation/reid/graph, not identity.

### 4.4 I2 on all ten, and what it decided about I3

`scripts/measure_identity_coherence.py` over the ten loop runs, each retrained with detached
identity and with a no-identity control (`render_identity/`, `render_control/`):

| scene | PSNR id / ctrl | fg agreement canonical / trajectory | bg agreement | fg entropy |
|---|---|---|---|---|
| lab-coat | 25.79 / 25.63 | 0.69 / 0.73 | 0.91 | 0.27 |
| blackswan | 22.39 / 22.35 | 0.94 / 0.95 | 0.95 | 0.04 |
| dogs-scale | 20.17 / 20.15 | 0.65 / 0.72 | 0.79 | 0.27 |
| lindy-hop | 19.34 / 19.24 | 0.60 / 0.63 | 0.83 | 0.43 |
| india | 21.37 / 21.91 | 0.53 / 0.52 | 0.65 | 0.91 |
| motocross-bumps | 27.33 / 27.45 | 0.79 / 0.79 | 0.89 | 0.39 |
| judo | 32.36 / 31.82 | 0.76 / 0.78 | 0.81 | 0.21 |
| gold-fish | 24.60 / 24.54 | 0.77 / 0.76 | 0.89 | 0.28 |
| crossing | 15.65 / 15.77 | 0.63 / 0.65 | 0.70 | 0.58 |
| miami-surf | 19.04 / 18.77 | 0.86 / 0.85 | 0.93 | 0.74 |

(Agreement: fraction of each Gaussian's 5 nearest neighbours sharing its argmax class.
Trajectory: neighbours in the space of positions at 8 frames across the clip.)

- **Detached identity costs no PSNR**: mean +0.04 dB against the control, range −0.54 to
  +0.54 — run-to-run noise, in both directions.
- **`india` and `miami-surf` barely learn foreground identity** (entropy 0.91, 0.74) — the
  known non-substantive foreground, not an identity failure.
- **Foreground agreement is 0.53–0.94**: I3 has work to do.

**The neighbourhood.** Canonical-pose k-NN assumes neighbours stay neighbours. Measured for
opaque Gaussians as distance-over-time / canonical distance ("stretch"):

| | lab-coat | blackswan | dogs-scale | lindy-hop | india | motocross | judo | gold-fish | crossing | miami-surf |
|---|---|---|---|---|---|---|---|---|---|---|
| median frame | 2.8× | 2.0× | 2.5× | 1.9× | 5.1× | 3.8× | 1.3× | 4.2× | 2.3× | 6.6× |
| worst 10% of frames | 19× | 3.4× | 5.6× | 3.8× | 12× | 11× | 2.1× | 13× | 4.6× | 21× |

Neighbours mostly stay together and fly apart on a minority of frames: the motion model's
error is **episodic**, not a uniformly non-rigid field. (A first reading used the max over
frames and overstated it as ~50×.) So canonical k-NN is sound, and trajectory-space k-NN —
barely higher agreement, and built from sampled frames that include the glitches — does not
earn its cost. Which frames go wrong, and why, is §8's new open item 4.

**I3 as built:** KL between each sampled Gaussian's class distribution and its k=5 neighbours'
(Gaussian Grouping's `loss_cls_3d`), canonical means for the foreground, 3D means for the
background, positions detached — identity only, as in I2. Every 2 steps, 800 samples, λ=2.
A stretch filter on neighbour pairs is held in reserve, for if the gate shows identity leaking
across entities that were adjacent in the canonical pose.

### 4.5 I3 — done; and why their lambda did not carry over at first

`DynamicGaussianTrainer.identity_consistency_loss`: every 2 steps, 800 sampled Gaussians per
field, KL from each one's class distribution to its k=5 nearest neighbours' (chunked k-NN,
self excluded) — canonical means for the foreground, means for the background, never across
fields. Positions detached; the loss reaches identity vectors and the classifier only. λ=2.

**The normalisation.** Gaussian Grouping divides the KL by `num_classes`, which is 200 in their
setup. The first build divided by ours (4–24 per scene), which made λ=2 8–50× stronger than
theirs *and* scene-dependent — fewer classes, harder smoothing. It bought agreement by
flattening: `motocross-bumps` foreground entropy 0.39 → 0.66, `crossing` confident entity
Gaussians 50.5k → 22.8k. The divisor is now a fixed `identity_3d_norm = 200`, so their λ means
the same here and no scene is smoothed harder for having fewer entities.

All ten, I2 / I3 with per-scene divisor / **I3 as committed**:

| scene | fg agreement | bg agreement | fg entropy | confident entity Gaussians (k, p ≥ 0.5) | PSNR / control |
|---|---|---|---|---|---|
| lab-coat | 0.69 / 0.94 / **0.87** | 0.91 / 0.96 / 0.94 | 0.27 / 0.27 / 0.22 | 93.8 / 86.3 / 96.4 | 25.60 / 25.63 |
| blackswan | 0.94 / 0.99 / 0.98 | 0.95 / 0.98 / 0.98 | 0.04 / 0.04 / 0.04 | 58.7 / 56.6 / 58.0 | 22.33 / 22.35 |
| dogs-scale | 0.65 / 0.84 / 0.74 | 0.79 / 0.92 / 0.90 | 0.27 / 0.34 / 0.28 | 93.9 / 85.3 / 90.1 | 19.73 / 20.15 |
| lindy-hop | 0.60 / 0.87 / 0.76 | 0.83 / 0.91 / 0.89 | 0.43 / 0.46 / 0.43 | 73.7 / 64.5 / 71.5 | 19.89 / 19.24 |
| india | 0.53 / 0.92 / 0.92 | 0.65 / 0.82 / 0.73 | 0.91 / 0.99 / 0.96 | 19.0 / 12.5 / 17.8 | 21.72 / 21.91 |
| motocross-bumps | 0.79 / 0.98 / 0.93 | 0.89 / 0.96 / 0.92 | 0.39 / 0.66 / 0.44 | 163.9 / 153.6 / 162.8 | 27.34 / 27.45 |
| judo | 0.76 / 0.94 / 0.89 | 0.81 / 0.94 / 0.91 | 0.21 / 0.26 / 0.24 | 129.9 / 122.7 / 126.6 | 32.39 / 31.82 |
| gold-fish | 0.77 / 0.95 / 0.91 | 0.89 / 0.96 / 0.94 | 0.28 / 0.24 / 0.22 | 77.3 / 77.5 / 82.3 | 24.49 / 24.54 |
| crossing | 0.63 / 0.92 / 0.87 | 0.70 / 0.91 / 0.83 | 0.58 / 0.55 / 0.59 | 50.5 / 22.8 / 34.4 | 15.64 / 15.77 |
| miami-surf | 0.86 / 0.99 / 0.99 | 0.93 / 0.97 / 0.95 | 0.74 / 0.78 / 0.65 | 50.2 / 48.4 / 55.0 | 18.96 / 18.77 |

Done-when met: foreground and background agreement rise on all ten (fg +0.04 to +0.39, +0.09
or more on seven), PSNR within noise of each control (−0.42 to +0.65 dB, I2's spread).

Still open, and for I4 to show rather than for tuning now:

- **`crossing` loses a third of its confident entity Gaussians** (50.5k → 34.4k), mostly in the
  background field. It has the weakest geometry of the ten (PSNR 15.6); smoothing may be merging
  entities whose Gaussians interleave.
- **Small classes are absorbed** where they have few Gaussians: `india` keeps 12 of 15
  confidently-owned classes, `lindy-hop` 17 of 18.

### 4.6 Identity reset — built, measured, a small effect (off by default)

Opacity reset's counterpart: every 1000 steps until 3000, identity vectors are zeroed with
their Adam state, so a Gaussian must re-earn its label from the pixels. The idea is ours, not
from the literature. `lab-coat`, `gold-fish`, `judo`, 5k steps, reset off → on: background
Gaussians with no identity gradient 0.49 → 0.32 / 0.57 → 0.49 / 0.43 → 0.35, entropy down,
PSNR unchanged — but selection precision rendered alone barely moves (trace 0.34 → 0.35,
0.13 → 0.15, 0.47 → 0.52; gate against DAVIS within ±0.05 per object).

The reason corrects the hypothesis the trace suggested: only 16–26% of the Gaussians a
selection wrongly takes are unsupervised. Most receive gradient — large Gaussians whose
footprint grazes the entity in some views while mostly covering something else. A loss that
only pushes each Gaussian toward the pixels it touches has no rule against a Gaussian whose
evidence is mostly *elsewhere*. FlashSplat's assignment (majority of rendered evidence, with
a background bias) is that rule; it is the next step. `identity_reset_every` stays in the
trainer, off by default, so the null result is on file.

### 4.7 Evidence-majority assignment (FlashSplat) — the removal path, pending sharper scenes

FlashSplat (ECCV 2024) reads labels off a trained scene instead of learning them: each
Gaussian accumulates its blending weight per id-map class over every view, and belongs to a
class when that evidence outweighs all its other evidence (plus a *slackness* bias).
`reifyscene/graph/assignment.py` implements it on our gsplat wrapper -- the evidence is the
gradient of a C-channel feature render, one backward per frame, each at its own time -- from the
paper and a reading of their code, whose rasterizer is Inria-licensed and not used.

Gate on all ten, 31 matched DAVIS objects, medians, scored against ground truth. *Halo* is
changed pixels outside the object per pixel of object; *neighbours* is the share of the other
annotated objects' pixels changed -- added because "rest intact" divides by the whole frame and
read ~97% for removals that visibly tore neighbours:

| method | took | precision visible / alone | halo | neighbours |
|---|---|---|---|---|
| old projection vote | 0.21 | 0.59 / 0.29 | 0.77× | 3.6% |
| learned identity (I2 + I3) | 0.53 | 0.88 / 0.24 | 1.40× | 11.2% |
| FlashSplat, slackness 0 | 0.47 | 0.89 / 0.33 | 0.57× | 3.6% |
| FlashSplat, −0.8 (take more) | 0.84 | 0.61 / 0.19 | 1.96× | 16.1% |
| FlashSplat, +0.4 (be surer) | 0.34 | 0.96 / 0.40 | 0.22× | 1.3% |

FlashSplat matches or beats learned identity everywhere but "took", needs no training, and its
slackness is a per-edit dial (their own scripts use −0.8 to −0.98 for removal). Alone precision
tops out near 0.4 for every method: on `lab-coat` Gaussians that render as a neighbour's face
contribute mostly to the target's pixels in the training views -- the reconstruction entangles
them, which no assignment rule undoes.

Visual review was inconclusive for a reason that blocks the gate itself: at 5k steps the scenes
are too soft to tell a removal defect from detail that was never modelled. Training quality
comes next (position LR decay, the opacity-reset momentum bug, views per step, a Gaussian cap);
the gate is re-judged on sharper scenes.

### 4.8 Training quality: 8 views per step, and the gate on sharp scenes

The 5k-step scenes were too soft to judge a removal by eye. Measured with
`scripts/eval_train_views.py` (PSNR and SSIM over every training frame; *sharpness* =
Laplacian variance of render over photo, 1.0 = as sharp as the footage), lab-coat / gold-fish /
judo, same 5k-step schedule throughout:

| setting | PSNR | SSIM | sharpness |
|---|---|---|---|
| baseline, 1 view per step | 23.2 / 23.1 / 29.4 | 0.75 / 0.74 / 0.92 | 0.21 / 0.08 / 0.59 |
| opacity-reset momentum fix | 23.1 / 22.9 / 29.3 | 0.74 / 0.74 / 0.92 | 0.21 / 0.08 / 0.59 |
| + scales LR decay to 0.1× (SoM) | 22.9 / 22.4 / 28.8 | 0.73 / 0.72 / 0.91 | 0.15 / 0.07 / 0.56 |
| 8 views + decay | 28.3 / 27.7 / 35.9 | 0.89 / 0.85 / 0.97 | 0.50 / 0.23 / 0.83 |
| **8 views per step** | **28.8 / 28.1 / 36.8** | **0.90 / 0.86 / 0.97** | **0.56 / 0.27 / 0.85** |

8 views per step (Shape-of-Motion's batch; the densification gradient is rescaled by the batch
size, as theirs is) is now the default, at ~5× render-stage time. Scale decay stays available,
off. Densification is capped at 1M Gaussians per field; `motocross-bumps` and `miami-surf`
reach it.

On all ten at 8 views per step, 31 matched objects, medians against DAVIS ground truth:

| method | took | precision visible / alone | halo | neighbours |
|---|---|---|---|---|
| old projection vote | 0.36 | 0.58 / 0.26 | 1.25× | 5.0% |
| **FlashSplat, slackness 0** | **0.55** | **0.94** / 0.30 | **0.61×** | 4.7% |
| FlashSplat, −0.8 | 0.90 | 0.72 / 0.20 | 1.86× | 19.9% |
| FlashSplat, +0.4 | 0.43 | 0.96 / 0.36 | 0.31× | 2.1% |

Reviewed by eye, FlashSplat looks better than the old vote, with visible background
contamination of two different kinds: (1) **the hole** -- what shows where a removed entity
stood is whatever blurry Gaussians sit behind it, because that background was never observed;
no selection fixes it, it needs inpainting (FlashSplat's paper follows removal with it), and it
is recorded here as a known limitation outside the step-1 gate; (2) **selection junk** --
needle-shaped and oversized Gaussians straddling the object's edge. The gate is judged on
selection; (2) is the next thing addressed.

### 4.9 A floater filter on the selection — the best so far, not yet good enough

`assignment.filter_members` drops from an evidence-majority selection, per field in its own
space, Gaussians more than 2 standard deviations isolated from their 8 nearest selected
neighbours (statistical outlier removal) and Gaussians whose largest axis exceeds 3× the
selection's median. Median per object: ~140 isolated and ~550 oversized dropped.

Ten sharp scenes, 31 matched objects, medians against DAVIS ground truth:

| method | took | precision visible / alone | halo | neighbours |
|---|---|---|---|---|
| FlashSplat | 0.55 | 0.94 / 0.30 | 0.61× | 4.7% |
| FlashSplat, +0.4 | 0.43 | 0.96 / 0.36 | 0.31× | 2.1% |
| **FlashSplat + filter** | 0.43 | 0.95 / **0.40** | 0.43× | 2.8% |

Reviewed by eye, **FlashSplat + filter is the best selection**, and still not good enough:
background contamination remains, and precision rendered alone tops out near 0.4. It is the
removal path for now; improving it is open.

---

## 5. The measurement, as a deliverable

Not an afterthought: **per-Gaussian identity entropy**, and its correlation with the gate's
per-frame swing.

This is the first instrument the project would have for *where the motion model is wrong*. It
should answer, for `motocross-bumps`, which of three things is true — the Gaussians drift, the
motion bases are wrong, or the node labelled "motorcycle" was never the right entity. All three
are currently consistent with what we have measured, and I have been guessing between them.

It also decides whether §3.1's split is worth building.

---

## 6. Risks

- **It requires retraining**, which is a render-stage change, and the cost §2.4 declined. The ten
  loop runs rebuild in ~90 minutes, so the ensemble stays cheap to re-measure.
- **Entities that enter or leave mid-clip.** Their ids exist for part of the sequence only; the
  classifier sees them on some frames and not others. Untested.
- **`india` and `miami-surf` are not addressed by any of this.** Their foreground never became
  substantive (median Gaussian scale 4–16× below `blackswan`'s) and the background carries the
  image. Identity cannot separate a foreground that does not render. That is a separate failure
  and it stays open.

---

## 7. What this overturns

§2.4 of the survey chose our projection vote over Gaussian Grouping:

> Ours needs **no retraining** and already runs on the **dynamic** trained cloud — both of which
> the published alternatives lack. It is the cheap version of a well-studied idea, and it is the
> right rung for us today.

Both stated advantages are real and both still hold. The unstated cost was **separability**, which
is the whole capability the handle exists to provide, and the limitation column — "noisy for small
or occluded entities" — undersells it: adjacent entities share Gaussians, and removing one
mutilates its neighbour. The verdict was sound on cost and wrong on consequence.

The roadmap had already reached the right conclusion for the other renderer — the 4D NeRF is
specified "compositional **by requirement, not taste** … because `remove(entity)` must leave a
hole". The same requirement was never applied to the Gaussian path we actually use.

---

## 8. Open

1. ~~**Wide-channel cost in gsplat**~~ — **closed 2026-09-29, measured.** gsplat 1.5.3, 80k
   Gaussians at 518x294, forward + backward: `D=3` 0.67 ms, `D=16` 0.91 ms, `D=19` 1.31 ms. Sixteen
   identity channels cost **+0.24 ms** over three, and RGB *plus* identity in a single `D=19` pass
   costs 1.31 ms. Wide channels are not a constraint, and one pass is viable rather than two.
   (A `RGB+ED` row in the same run read 13.5 ms, but it ran first and absorbed CUDA warm-up — it is
   not comparable, and the D=3/16/19 progression that answers the question is.)
   **Correction (§4.2):** those were raw channel counts, not features on top of `RGB+ED` — a
   16-feature pass is 20 channels, which gsplat pads to 32. The conclusion stands (wide channels
   are not a constraint), but the numbers above are superseded by §4.2's committed benchmark.
2. **Where the identity classifier lives at inference.** Theirs is a checkpoint beside the point
   cloud; ours has to reach the graph stage, which currently reads only the bundle.
3. **`num_classes` per scene.** Theirs defaults to 200. Ours is entities + background, known per
   run — which makes the normalisation term scene-dependent and worth stating in the manifest.
4. **Episodic motion error.** Canonical neighbours sit within ~2–4× of their spacing on a
   typical frame and fly apart (up to ~20×) on the worst 10% (§4.4). Which frames, whether
   they coincide with the gate's per-frame swings (`motocross-bumps` 2.3% → 73.5% → 6.9%),
   and whether a local-rigidity term on neighbours' motion coefficients removes them — the
   motion model has smoothness over time on its bases and nothing across space.