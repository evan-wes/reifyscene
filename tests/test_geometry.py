"""Basic tests for geometry scaffolding."""

from room4d.geometry.vggt_wrapper import VGGTInitializer


def test_vggt_initializer_constructs() -> None:
    init = VGGTInitializer()
    assert init.cfg == {}
