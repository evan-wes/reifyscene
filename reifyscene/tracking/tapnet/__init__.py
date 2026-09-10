"""Vendored TAPIR / BootsTAPIR point-tracker model (PyTorch).

reifyscene's native point tracker (:class:`reifyscene.tracking.tapir_wrapper.TAPIRWrapper`)
runs this model to produce the same all-pairs 2D tracks Shape of Motion's preproc
emits — replacing the last borrowed input in the native motion-init path.

Provenance
----------
These four modules (``nets.py``, ``tapir_model.py``, ``transforms.py``,
``utils.py``) are vendored **byte-for-byte** from Shape of Motion's
``preproc/tapnet_torch/`` (itself DeepMind's TAPIR torch port), so they can be
re-diffed against upstream without noise. They carry their original DeepMind
Apache-2.0 headers. Do NOT edit them in place: to update, re-copy from the
pinned ``third_party/shape-of-motion`` checkout and re-run the parity check.

The single non-torch dependency is ``einshape`` (declared in the ``vision``
extra in ``pyproject.toml`` / pinned in ``scripts/locks/reifyscene.lock``), kept as
a real dependency specifically so this port stays byte-faithful.

Usage
-----
The model is constructed as ``TAPIR(pyramid_level=1)`` and loaded from the
``bootstapir_checkpoint_v2.pt`` / ``tapir_checkpoint_panning.pt`` checkpoints
fetched by ``scripts/fetch_tapir_checkpoint.sh``. Its forward signature is
``model(frames, query_points) -> {"tracks", "occlusion", "expected_dist"}``.
"""

from reifyscene.tracking.tapnet import nets, tapir_model, transforms, utils
from reifyscene.tracking.tapnet.tapir_model import TAPIR

__all__ = ["TAPIR", "tapir_model", "transforms", "utils", "nets"]
