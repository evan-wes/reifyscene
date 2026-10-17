"""Vendored TAPIR / BootsTAPIR point-tracker model (PyTorch).

reifyscene's native point tracker (:class:`reifyscene.tracking.tapir_wrapper.TAPIRWrapper`)
runs this model to produce the same all-pairs 2D tracks Shape of Motion's preproc
emits — replacing the last borrowed input in the native motion-init path.

Provenance
----------
These four modules (``nets.py``, ``tapir_model.py``, ``transforms.py``,
``utils.py``) are vendored from Shape of Motion's ``preproc/tapnet_torch/`` (itself
DeepMind's TAPIR torch port) under the Apache License 2.0 (LICENSES/Apache-2.0.txt), with
their original DeepMind headers. Three are byte-for-byte copies; ``tapir_model.py`` has one
change, marked in its header (``TAPIR.forward`` takes precomputed ``feature_grids``). To
update, re-copy from the pinned ``third_party/shape-of-motion`` checkout, re-apply that change,
and re-run the parity check.

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
