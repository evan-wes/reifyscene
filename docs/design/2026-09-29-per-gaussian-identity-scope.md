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
| **I0** | id-map export: per-frame PNGs of canonical node ids, through the recorded `ImageTransform` | ids match the graph's nodes; a round-trip read reproduces the masks the handle uses today |
| **I1** | gsplat wrapper carries an N-channel feature alongside RGB | a known feature rasterizes to the expected image; RGB output is unchanged for an existing checkpoint |
| **I2** | identity parameter on the dynamic field **and the background field** + classifier + 2D loss | identity survives densification and pruning; a static segmented entity is addressable; the one-batch overfit reaches ≈ zero classification loss |
| **I3** | 3D consistency regularizer on canonical means | neighbouring Gaussians agree measurably more than without it |
| **I4** | removal by probability threshold; no hull | the ten-scene gate re-run and re-reviewed against the recorded baseline (`blackswan` alone) |
| **I5** | the conflict measurement (§5) | reported for all ten, and `motocross-bumps` explained |

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
- **Wide-channel backward pass in gsplat.** Whether a 16-channel feature is efficient or a
  performance problem — the one thing worth checking before I1.
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

1. **Wide-channel cost in gsplat** (§6) — check before I1.
2. **Where the identity classifier lives at inference.** Theirs is a checkpoint beside the point
   cloud; ours has to reach the graph stage, which currently reads only the bundle.
3. **`num_classes` per scene.** Theirs defaults to 200. Ours is entities + background, known per
   run — which makes the normalisation term scene-dependent and worth stating in the manifest.
