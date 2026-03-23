"""Optional real-VGGT integration smoke test.

This test is disabled by default because it may download large checkpoints and
require substantial GPU memory. Enable it explicitly with
ROOM4D_RUN_VGGT=1.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from room4d.api.pipeline import Room4DPipeline
from tests.conftest import write_test_video


pytestmark = pytest.mark.skipif(
    os.environ.get("ROOM4D_RUN_VGGT") != "1",
    reason="Set ROOM4D_RUN_VGGT=1 to run the real VGGT integration test.",
)


def test_single_video_real_vggt_optional(tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    write_test_video(clip, color_offset=0)

    pipeline = Room4DPipeline({
        'run_dir': tmp_path / 'run',
        'max_keyframes_per_stream': 2,
        'keyframe_stride_sec': 1.0,
        'geometry': {
            'backend_mode': 'real',
            'device': 'cpu',
            'use_autocast': False,
        },
    })
    outputs = pipeline.run([clip])
    assert outputs['geometry'].backend_name == 'vggt'
