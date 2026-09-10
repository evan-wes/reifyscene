"""Semantics stage (design §4A): open-vocab category + attributes per tracklet.

Names the class-agnostic instances `segment` produces. The SigLIP2 rung (FP4a)
assigns an open-vocab category + a semantic embedding per canonical (reid-merged)
track; the floor rung carries the detector's COCO-80 label, degrading to
``"unknown"``. Qwen2.5-VL free-form attributes/captions arrive at FP4b.
"""
