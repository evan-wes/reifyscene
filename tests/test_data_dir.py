"""The data directory: one setting, names inside it, nothing outside it."""

from __future__ import annotations

from pathlib import Path

import pytest

from reifyscene.common import data_dir
from reifyscene.common.data_dir import DataDirError
from reifyscene.common.types import VideoStreamRef
from reifyscene.ingest.ingest_artifact import IngestArtifact


def test_root_is_the_setting(isolated_data_dir: Path) -> None:
    assert data_dir.root() == isolated_data_dir
    assert data_dir.area(data_dir.RUNS) == isolated_data_dir / "runs"


@pytest.mark.parametrize("value, message", [
    ("", "is not set"),
    ("relative/dir", "must be an absolute path"),
])
def test_root_refuses_a_missing_or_relative_setting(monkeypatch, value, message) -> None:
    monkeypatch.setenv(data_dir.ENV, value)
    with pytest.raises(DataDirError, match=message):
        data_dir.root()


def test_root_refuses_a_folder_that_is_not_there(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv(data_dir.ENV, str(tmp_path / "absent"))
    with pytest.raises(DataDirError, match="not a directory"):
        data_dir.root()


def test_names_resolve_inside_their_area(isolated_data_dir: Path) -> None:
    assert data_dir.path(data_dir.RUNS, "lab-coat") == isolated_data_dir / "runs" / "lab-coat"
    assert (data_dir.path(data_dir.DATASETS, "davis2017/x/lab-coat")
            == isolated_data_dir / "datasets" / "davis2017" / "x" / "lab-coat")
    with pytest.raises(ValueError, match="not an area"):
        data_dir.area("elsewhere")


@pytest.mark.parametrize("name", ["/abs/run", "~me/run", "../outside", "runs/../../outside", ""])
def test_a_name_never_points_outside(name: str) -> None:
    with pytest.raises(DataDirError):
        data_dir.path(data_dir.RUNS, name)
    with pytest.raises(DataDirError):
        data_dir.from_record(name)


def test_record_names_round_trip(isolated_data_dir: Path) -> None:
    target = isolated_data_dir / "datasets" / "seq" / "rgb"
    assert data_dir.record_name(target) == "datasets/seq/rgb"
    assert data_dir.from_record("datasets/seq/rgb") == target


def test_every_test_path_is_inside_the_tests_data_directory(tmp_path: Path) -> None:
    assert data_dir.record_name(tmp_path) == tmp_path.name


def test_a_record_cannot_name_a_place_outside(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(data_dir.ENV, str(tmp_path / "data"))
    (tmp_path / "data").mkdir()
    outside = tmp_path / "elsewhere" / "clip"
    with pytest.raises(DataDirError, match="outside the data directory"):
        data_dir.record_name(outside)


def test_record_names_do_not_follow_links(tmp_path: Path, monkeypatch) -> None:
    """A dataset linked into the data directory is named where it sits, not where it points."""
    root, real = tmp_path / "data", tmp_path / "store" / "seq"
    real.mkdir(parents=True)
    (root / "datasets").mkdir(parents=True)
    (root / "datasets" / "seq").symlink_to(real)
    monkeypatch.setenv(data_dir.ENV, str(root))
    assert data_dir.record_name(root / "datasets" / "seq") == "datasets/seq"


def test_an_artifact_records_paths_as_names_and_survives_a_move(
        tmp_path: Path, monkeypatch) -> None:
    """The whole data directory can move: a reloaded record resolves inside the new one."""
    root = tmp_path / "data"
    monkeypatch.setenv(data_dir.ENV, str(root))
    clip = root / "datasets" / "clip"
    clip.mkdir(parents=True)
    stream = VideoStreamRef(stream_id="cam_00", video_path=clip, fps=None, num_frames=1,
                            width=4, height=4, has_audio=False,
                            metadata={"source": "frames", "frame_index": {"7": "f_007.png"}})
    run = root / "runs" / "r" / "ingest"
    IngestArtifact(run_id="r", stage_name="ingest", save_dir=run, streams=[stream]).save()
    raw = (run / IngestArtifact.FILE).read_text()
    assert str(root) not in raw and '"datasets/clip"' in raw

    moved = tmp_path / "moved"
    moved.mkdir()
    monkeypatch.setenv(data_dir.ENV, str(moved))
    loaded = IngestArtifact.load(run)
    assert loaded.streams[0].video_path == moved / "datasets" / "clip"
    assert loaded.streams[0].frame_file(7) == moved / "datasets" / "clip" / "f_007.png"
