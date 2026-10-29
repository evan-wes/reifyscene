# Third-party notices

reifyscene is licensed under the PolyForm Noncommercial License 1.0.0 ([LICENSE](LICENSE)). The
files below contain code or data from other projects and stay under those projects' licences,
whose texts are in [LICENSES/](LICENSES).

## Code in this repository under another licence

### VGGT (Meta) — VGGT License

- **`reifyscene/geometry/vggt_derived.py`** is adapted from VGGT
  (<https://github.com/facebookresearch/vggt>): `load_and_preprocess_images` and
  `VGGT.forward`. Copyright (c) Meta Platforms, Inc. and affiliates. Licensed under the VGGT
  License ([LICENSES/VGGT-LICENSE.txt](LICENSES/VGGT-LICENSE.txt)); its use is subject to that
  licence's Acceptable Use Policy.
- **`third_party_patches/vggt-long/`** patches VGGT-Long (<https://github.com/DengKaiCQ/VGGT-Long>),
  which is distributed under the VGGT License. `0001-dinov2-path-cwd.patch` changes
  `LoopModels/backbones/dinov2.py`, a file VGGT-Long took from SALAD
  (<https://github.com/serizba/salad>, GPL-3.0); three of the patch's unchanged context lines
  come from it.

### TAPIR (Google DeepMind) — Apache License 2.0

- **`reifyscene/tracking/tapnet/`** is the TAPIR / BootsTAPIR PyTorch model from
  <https://github.com/google-deepmind/tapnet>, taken from Shape of Motion's copy. Copyright 2024
  DeepMind Technologies Limited. Apache License 2.0
  ([LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt)). `tapir_model.py` is modified; its header
  says how.

### Shape of Motion — MIT

Copyright (c) 2024 Vickie Ye. MIT License
([LICENSES/MIT-shape-of-motion.txt](LICENSES/MIT-shape-of-motion.txt)).
<https://github.com/vye16/shape-of-motion>. Portions are adapted in:

- `reifyscene/eval/metrics/image.py` (the co-visibility-masked mPSNR / mSSIM / mLPIPS)
- `reifyscene/render/gaussian4d/losses.py`, `transforms.py`, `motion.py`, `procrustes_init.py`
- `reifyscene/lifting/native_track_lift.py`
- `reifyscene/tracking/tapir_wrapper.py`
- `reifyscene/geometry/depth_alignment.py`
- `third_party_patches/shape-of-motion/`

### Patches to other tools

A patch contains lines of the code it changes, so each is under its upstream's licence
([third_party_patches/README.md](third_party_patches/README.md)):

- `third_party_patches/hloc/`: Hierarchical-Localization
  (<https://github.com/cvg/Hierarchical-Localization>), Apache License 2.0.
- `third_party_patches/concept-graphs/`: ConceptGraphs
  (<https://github.com/concept-graphs/concept-graphs>), Copyright (c) 2023 concept-graphs, MIT
  ([LICENSES/MIT-concept-graphs.txt](LICENSES/MIT-concept-graphs.txt)).
- `third_party_patches/vla-3d/`: VLA-3D (<https://github.com/HaochenZ11/VLA-3D>),
  Copyright (c) 2024 Haochen Z, MIT ([LICENSES/MIT-vla-3d.txt](LICENSES/MIT-vla-3d.txt)).

## Methods implemented from others' work

These are reifyscene's own code, under its own licence, and are credited here:

- **FlashSplat** (Shen, Yang & Wang, ECCV 2024; <https://github.com/florinshen/FlashSplat>):
  `reifyscene/graph/assignment.py` implements its per-Gaussian assignment rule from the paper
  and a reading of its code. No FlashSplat code is included. FlashSplat's code is under the
  Gaussian-Splatting licence of Inria and MPII, for non-commercial research use.
- **Gaussian Grouping** (<https://github.com/lkeab/gaussian-grouping>, Apache License 2.0): the
  identity loss's normalisation and its 3D neighbour-consistency term in
  `reifyscene/render/gaussian4d/dynamic_trainer.py`.
- **gsplat** (<https://github.com/nerfstudio-project/gsplat>, Apache License 2.0): the
  densification thresholds and schedule in `reifyscene/render/gaussian4d/densify.py`.
- **Sapiens** (<https://github.com/facebookresearch/sapiens>, CC BY-NC 4.0): the Goliath keypoint
  indices in `reifyscene/edit/body_volume.py`.
- **VLA-3D**: `reifyscene/graph/relations.py` imports its relation functions from the checkout
  at run time; none of its code is copied.

## Data in this repository

- **DAVIS 2017** (<https://davischallenge.org>; J. Pont-Tuset et al., "The 2017 DAVIS Challenge on
  Video Object Segmentation", arXiv:1704.00675). The files in `benchmarks/` hold scores against
  DAVIS 2017, and `benchmarks/detector_keyframe_cache.json` holds bounding boxes derived from its
  annotations. The README's animations in `docs/media/` show DAVIS's `lab-coat` sequence: the
  app's capture shows its frames, scaled down, and the removal animations are renders of a scene
  trained on them. No other DAVIS frames, and no DAVIS masks, are included. The dataset's own
  README gives two licences: CC BY 4.0 (<https://creativecommons.org/licenses/by/4.0/>) in its
  credits and CC BY-NC 4.0 (<https://creativecommons.org/licenses/by-nc/4.0/>) in its terms of
  use; this repository is non-commercial and follows the stricter. That README also refers each
  video to its online source for that video's terms; `lab-coat`'s source is a YouTube video that
  is no longer available.
- `benchmarks/dycheck_spin_heldout_edit.json` holds scores on DyCheck's `spin` sequence only.
- **LVIS v1 category names** (<https://www.lvisdataset.org>; A. Gupta, P. Dollár, R. Girshick,
  "LVIS: A Dataset for Large Vocabulary Instance Segmentation", CVPR 2019).
  `reifyscene/perception/semantics/vocab.py` lists the names of LVIS v1's frequent-tier categories
  as label candidates, taken from detectron2's `lvis_v1_categories.py` (Apache-2.0). The LVIS
  annotations are licensed under CC BY 4.0 (<https://creativecommons.org/licenses/by/4.0/>).

## What you install or download

None of the following is included in this repository. The setup scripts check out the tools, and
the pipeline downloads the models when it first needs them; each comes with its own terms, which
apply to you when you use it.

| What | Licence | Used for |
|---|---|---|
| VGGT code (<https://github.com/facebookresearch/vggt>) | VGGT License | geometry |
| VGGT-1B weights | CC BY-NC 4.0 | geometry |
| VGGT-Long (<https://github.com/DengKaiCQ/VGGT-Long>) | VGGT License | long-sequence geometry |
| VGGT-Omega code and weights (<https://github.com/facebookresearch/vggt-omega>; weights gated on Hugging Face) | FAIR Noncommercial Research License | one-pass geometry |
| What VGGT-Long's weight script downloads: SALAD checkpoint, ORB-SLAM3 vocabulary; MapAnything; Pi3 | from GPL-3.0 repositories; CC BY-NC 4.0; BSD-2-Clause | loop closure |
| CoTracker code and CoTracker3 weights (<https://github.com/facebookresearch/co-tracker>) | CC BY-NC 4.0 | point tracking (the default tracker) |
| VGGSfM tracker weights | CC BY-NC 4.0 | pose refinement |
| Sapiens weights | CC BY-NC 4.0 | human perception |
| SAM 2 code and weights (<https://github.com/facebookresearch/sam2>) | Apache 2.0 | segmentation |
| MoGe code and MoGe-2 weights (<https://github.com/microsoft/MoGe>) | MIT | depth |
| DINOv2 and SigLIP 2 weights | Apache 2.0 | re-identification, semantics |
| DINOv3 weights (optional) | DINOv3 License | re-identification |
| torchvision's pretrained Mask R-CNN, Keypoint R-CNN and ResNet-18 | see torchvision's documentation | detection, keypoints, embeddings |
| Hierarchical-Localization (<https://github.com/cvg/Hierarchical-Localization>) | Apache 2.0 | calibration |
| SuperPoint and SuperGlue (in Hierarchical-Localization) | Magic Leap, non-commercial research use | calibration (the default matcher) |
| TAPIR / BootsTAPIR checkpoints | Apache 2.0 | native point tracking |
| gsplat, detectron2 | Apache 2.0 | rasterizer, detection |
| Grounding DINO base | Apache 2.0 | `scripts/probe_detector_coverage.py` |
| Shape of Motion (<https://github.com/vye16/shape-of-motion>) | MIT | the reference comparison |
| ConceptGraphs and its environment, which installs ultralytics | MIT; ultralytics is AGPL-3.0 | the ConceptGraphs reference comparison |
| ffmpeg (conda-forge GPL build, for libx264) | GPL | turning image-sequence datasets into videos |
| DAVIS 2017 | CC BY-NC 4.0 (its README also names CC BY 4.0); each video's source keeps its own terms | the scene-graph clips, benchmarks and the README's animations |
| DyCheck iPhone, NVIDIA Dynamic Scenes (Shape of Motion's processed downloads) | their authors' terms | reconstruction benchmarks |
| CMU Panoptic | research use only; no redistribution without permission | calibration experiments |
