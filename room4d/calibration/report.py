"""Visualization and reporting for calibration mode."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go

from room4d.common.io import write_json
from room4d.common.paths import ensure_dir
from room4d.common.types import TimedCameraState


def _camera_centers(states: list[TimedCameraState]) -> np.ndarray:
    centers = []
    for state in states:
        T = np.asarray(state.pose.T_world_cam, dtype=np.float32)
        centers.append(T[:3, 3])
    return np.asarray(centers, dtype=np.float32) if centers else np.zeros((0, 3), dtype=np.float32)


def write_calibration_viewer(save_dir: str | Path, camera_states_by_stream: dict[str, list[TimedCameraState]], sparse_xyz: np.ndarray, sparse_rgb: np.ndarray | None = None) -> Path:
    save_dir = ensure_dir(save_dir)
    fig = go.Figure()
    pts = np.asarray(sparse_xyz, dtype=np.float32)
    if pts.size:
        cols = None
        if sparse_rgb is not None and np.asarray(sparse_rgb).size:
            rgb = np.asarray(sparse_rgb, dtype=np.uint8)
            cols = [f"rgb({int(r)},{int(g)},{int(b)})" for r, g, b in rgb[: len(pts)]]
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers",
            marker=dict(size=2, color=cols or "#66c2ff", opacity=0.7),
            name="Sparse points",
        ))
    palette = ["#ff6b6b", "#4dabf7", "#51cf66", "#fcc419", "#b197fc", "#ffa94d"]
    for idx, (stream_id, states) in enumerate(sorted(camera_states_by_stream.items())):
        centers = _camera_centers(states)
        if not centers.size:
            continue
        fig.add_trace(go.Scatter3d(
            x=centers[:, 0], y=centers[:, 1], z=centers[:, 2],
            mode="markers+lines", marker=dict(size=4, color=palette[idx % len(palette)]),
            line=dict(color=palette[idx % len(palette)], width=3), name=f"{stream_id} cameras",
        ))
    fig.update_layout(
        title="Calibration sparse reconstruction",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"),
        margin=dict(l=0, r=0, t=48, b=0),
    )
    out = Path(save_dir) / "calibration_viewer.html"
    out.write_text(fig.to_html(include_plotlyjs="cdn", full_html=True), encoding="utf-8")
    return out


def write_calibration_report(
    save_dir: str | Path,
    *,
    streams: list[str],
    sampled_counts: dict[str, int],
    reprojection_stats: dict[str, float],
    connected_components: list[list[str]],
    notes: list[str],
    viewer_path: Path | None,
) -> Path:
    save_dir = ensure_dir(save_dir)
    rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in reprojection_stats.items())
    sampled_rows = "".join(f"<tr><td>{sid}</td><td>{sampled_counts.get(sid, 0)}</td></tr>" for sid in streams)
    comp_items = "".join(f"<li>component {idx + 1}: {len(comp)} images ({', '.join(comp[:6])}{'…' if len(comp) > 6 else ''})</li>" for idx, comp in enumerate(connected_components))
    note_items = "".join(f"<li>{note}</li>" for note in notes)
    viewer_link = f'<p><a href="{viewer_path.name}">Open calibration viewer</a></p>' if viewer_path is not None else ""
    html = f"""
<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>room4d calibration report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; }}
    table {{ border-collapse: collapse; margin-bottom: 20px; }}
    td, th {{ border: 1px solid #ccc; padding: 6px 10px; }}
  </style>
</head>
<body>
  <h1>room4d calibration report</h1>
  {viewer_link}
  <h2>Sampled frames</h2>
  <table><tr><th>Stream</th><th>Frames</th></tr>{sampled_rows}</table>
  <h2>Reprojection statistics</h2>
  <table><tr><th>Metric</th><th>Value</th></tr>{rows}</table>
  <h2>Connected components</h2>
  <ul>{comp_items}</ul>
  <h2>Notes</h2>
  <ul>{note_items}</ul>
</body>
</html>
"""
    out = Path(save_dir) / "calibration_report.html"
    out.write_text(html, encoding="utf-8")
    write_json(Path(save_dir) / "calibration_summary.json", {
        "streams": streams,
        "sampled_counts": sampled_counts,
        "reprojection_stats": reprojection_stats,
        "connected_components": connected_components,
        "notes": notes,
        "viewer_html": viewer_path.name if viewer_path is not None else None,
    })
    return out
