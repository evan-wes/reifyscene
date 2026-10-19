"""Watch per-Gaussian identity develop during training, against the id maps it trains on.

Identity is judged at the end by the removal gate, but the gate cannot say *why* a selection
is wrong: whether training stopped too early, whether identity never settled, or whether the
Gaussians it mislabels are ones the 2D loss hardly reaches. This records, at intervals, on a
fixed set of training views:

- **2D fit** -- per entity, IoU between the rendered argmax and the id map (ignore pixels
  excluded). Whether identity reproduces its labels, and when that stops improving.
- **stability** -- the fraction of Gaussians whose argmax class changed since the last record
  (only while the cloud keeps its size: densification re-indexes it), the fraction of rendered
  pixels whose class changed, and mean normalised entropy per field.
- **supervision** -- each Gaussian's identity-loss gradient on these views, from a separate
  backward pass that touches nothing else. The fraction that gets essentially none, their mean
  entropy against the rest, and the share of *selected* Gaussians (p >= 0.5 for some entity)
  that were unsupervised: the hypothesis that the junk in a selection is Gaussians the 2D
  loss never reached.
- **selection** -- each entity's Gaussians at p >= 0.5, scored against its id-map mask two
  ways: rendered **alone** (alpha > 0.5; shows parts hidden in the full scene -- the view a
  person judges) and by **visible share** of the full render (a one-channel feature, so
  occlusion is exact). Recall and precision for each; medians over entities.

Everything is measured against the id maps (the training labels), never DAVIS ground truth,
so the trace can be used to tune training without tuning against the benchmark.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from reifyscene.graph.id_maps import BACKGROUND, UNADDRESSED, labelled


def _median(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    return float(np.median(xs)) if xs else None


@dataclass
class IdentityTracer:
    """Stateful: remembers the previous record's argmaxes to measure flips."""

    views: list[tuple]                      # (camera, gt, time_id, id_map), a fixed subset
    select_threshold: float = 0.5
    supervision_eps: float = 1e-9
    _prev_arg: dict[str, torch.Tensor] = field(default_factory=dict)
    _prev_pix: list[torch.Tensor] = field(default_factory=list)

    @classmethod
    def from_train_views(cls, train_views: list[tuple], n: int = 8, **kw) -> IdentityTracer:
        with_maps = [v for v in train_views if len(v) > 3 and v[3] is not None]
        if not with_maps:
            raise ValueError("no training view has an id map to trace against")
        idx = np.unique(np.linspace(0, len(with_maps) - 1, min(n, len(with_maps))).round())
        return cls(views=[with_maps[int(i)] for i in idx], **kw)

    # -- helpers --------------------------------------------------------------------------

    @staticmethod
    def _fields(scene) -> list[tuple[str, Any]]:
        return [("fg", scene.fg)] + ([("bg", scene.bg)] if scene.bg is not None else [])

    @staticmethod
    def _probs(scene) -> torch.Tensor:
        ids = [scene.fg.identity] + ([scene.bg.identity] if scene.bg is not None else [])
        return torch.softmax(scene.identity_classifier(torch.cat(ids, 0)), dim=-1)

    # -- the record -----------------------------------------------------------------------

    def record(self, trainer, step: int) -> dict[str, Any]:
        scene, rast = trainer.scene, trainer.rasterizer
        head = scene.identity_classifier
        n_fg = scene.fg.num_gaussians
        n_cls = head.out_features
        rec: dict[str, Any] = {"step": int(step), "num_fg": n_fg,
                               "num_bg": scene.num_gaussians - n_fg,
                               "last_identity_reset": getattr(trainer, "last_identity_reset", None)}

        # supervision: identity-loss gradient per Gaussian on these views, identity only
        params = [f.identity for _, f in self._fields(scene)]
        grads = [torch.zeros(p.shape[0], device=p.device) for p in params]
        ious: dict[int, list[float]] = {}
        psnrs, pix_args = [], []
        for cam, gt, t, id_map in self.views:
            cloud = scene.cloud_at(t)
            out = rast.rasterize(cloud.means.detach(), cloud.quats.detach(),
                                 cloud.scales.detach(), cloud.opacities.detach(),
                                 cloud.colors.detach(), cam, features=cloud.features)
            loss = trainer.identity_loss(out.features, id_map)
            if loss is not None:
                gs = torch.autograd.grad(loss, params, allow_unused=True)
                for i, g in enumerate(gs):
                    if g is not None:
                        grads[i] += g.detach().abs().sum(dim=1)
            with torch.no_grad():
                pred = head(out.features.detach()).argmax(-1)
                pix_args.append(pred)
                valid = labelled(id_map)
                for c in torch.unique(id_map[valid]).tolist():
                    p, g = (pred == c) & valid, (id_map == c) & valid
                    union = int((p | g).sum())
                    if union:
                        ious.setdefault(int(c), []).append(int((p & g).sum()) / union)
                mse = float(((out.rgb.detach().clamp(0, 1) - gt) ** 2).mean())
                psnrs.append(10 * math.log10(1.0 / max(mse, 1e-12)))
        rec["psnr"] = float(np.mean(psnrs))
        per_cls = {c: float(np.mean(v)) for c, v in ious.items()}
        rec["iou_background"] = per_cls.get(0)
        rec["iou_entities_mean"] = (float(np.mean([v for c, v in per_cls.items() if c]))
                                    if any(c for c in per_cls) else None)
        rec["iou_per_class"] = {str(c): round(v, 4) for c, v in sorted(per_cls.items())}

        with torch.no_grad():
            probs = self._probs(scene)
            arg = probs.argmax(-1)
            ent = -(probs * probs.clamp_min(1e-12).log()).sum(-1) / math.log(n_cls)
            sup = torch.cat(grads) > self.supervision_eps
            selected = (probs[:, 1:] >= self.select_threshold).any(-1)
            for name, sl in (("fg", slice(0, n_fg)), ("bg", slice(n_fg, None))):
                a, e, s = arg[sl], ent[sl], sup[sl]
                rec[f"entropy_{name}"] = float(e.mean()) if len(e) else None
                rec[f"unsupervised_frac_{name}"] = float((~s).float().mean()) if len(s) else None
                rec[f"entropy_unsupervised_{name}"] = float(e[~s].mean()) if (~s).any() else None
                rec[f"entropy_supervised_{name}"] = float(e[s].mean()) if s.any() else None
                prev = self._prev_arg.get(name)
                rec[f"flip_frac_{name}"] = (float((a != prev).float().mean())
                                            if prev is not None and prev.shape == a.shape else None)
                self._prev_arg[name] = a.clone()
            rec["selected_unsupervised_frac"] = (float((~sup[selected]).float().mean())
                                                 if selected.any() else None)
            if self._prev_pix and len(self._prev_pix) == len(pix_args):
                rec["pixel_flip_frac"] = float(np.mean(
                    [float((a != b).float().mean()) for a, b in zip(pix_args, self._prev_pix,
                                                                    strict=True)]))
            else:
                rec["pixel_flip_frac"] = None
            self._prev_pix = pix_args
            rec.update(self._selection(trainer, probs))
        return rec

    @torch.no_grad()
    def _selection(self, trainer, probs: torch.Tensor) -> dict[str, Any]:
        """Per entity, its p >= threshold Gaussians against its id-map mask, two ways."""
        scene, rast = trainer.scene, trainer.rasterizer
        stats: dict[str, list[float]] = {k: [] for k in
                                         ("alone_recall", "alone_precision",
                                          "visible_recall", "visible_precision")}
        present = sorted({int(c) for *_, m in self.views for c in torch.unique(m).tolist()
                          if c != BACKGROUND and c < UNADDRESSED})
        for c in present:
            sel = probs[:, c] >= self.select_threshold
            if not sel.any():
                for k in ("alone_recall", "visible_recall"):
                    stats[k].append(0.0)
                continue
            inter_a = took_a = inter_v = took_v = area = 0
            for cam, _gt, t, id_map in self.views:
                gt = id_map == c
                valid = labelled(id_map)
                cloud = scene.cloud_at(t)
                alone = rast.rasterize(cloud.means[sel], cloud.quats[sel], cloud.scales[sel],
                                       cloud.opacities[sel], cloud.colors[sel], cam).alpha > 0.5
                share = rast.rasterize(cloud.means, cloud.quats, cloud.scales, cloud.opacities,
                                       cloud.colors, cam,
                                       features=sel.float()[:, None]).features[..., 0] > 0.5
                alone, share = alone & valid, share & valid
                area += int(gt.sum())
                inter_a += int((alone & gt).sum())
                took_a += int(alone.sum())
                inter_v += int((share & gt).sum())
                took_v += int(share.sum())
            if area:
                stats["alone_recall"].append(inter_a / area)
                stats["visible_recall"].append(inter_v / area)
            if took_a:
                stats["alone_precision"].append(inter_a / took_a)
            if took_v:
                stats["visible_precision"].append(inter_v / took_v)
        out = {f"sel_{k}_median": _median(v) for k, v in stats.items()}
        out["sel_entities"] = len(present)
        return out
