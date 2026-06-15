"""Benchmark runner for end-to-end evaluation and failure reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from room4d.common.io import write_json
from room4d.common.paths import ensure_dir
from room4d.geometry.eval.geometry_metrics import GeometryMetrics
from room4d.render.eval.rendering_metrics import RenderingMetrics
from room4d.tracking.eval.tracking_metrics import TrackingMetrics


class BenchmarkRunner:
    """Run evaluation suites over artifacts and persist reports."""

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        self.cfg = cfg or {}
        self.geometry_metrics = GeometryMetrics(self.cfg.get("geometry", {}))
        self.tracking_metrics = TrackingMetrics(self.cfg.get("tracking", {}))
        self.rendering_metrics = RenderingMetrics(self.cfg.get("rendering", {}))
        self.thresholds = {
            "reprojection_error_p90_px_max": float(self.cfg.get("reprojection_error_p90_px_max", 24.0)),
            "mean_human_fragments_max": float(self.cfg.get("mean_human_fragments_max", 1.5)),
            "height_abs_p90_max": float(self.cfg.get("height_abs_p90_max", 0.40)),
            "heldout_psnr_mean_min": float(self.cfg.get("heldout_psnr_mean_min", 11.0)),
        }

    def _failure_report(self, metrics: dict[str, Any]) -> dict[str, Any]:
        failures: list[dict[str, Any]] = []
        if metrics.get("reprojection_error_p90_px") is not None and metrics["reprojection_error_p90_px"] > self.thresholds["reprojection_error_p90_px_max"]:
            failures.append({"metric": "reprojection_error_p90_px", "value": metrics["reprojection_error_p90_px"], "threshold": self.thresholds["reprojection_error_p90_px_max"], "severity": "warning"})
        if metrics.get("mean_human_fragments") is not None and metrics["mean_human_fragments"] > self.thresholds["mean_human_fragments_max"]:
            failures.append({"metric": "mean_human_fragments", "value": metrics["mean_human_fragments"], "threshold": self.thresholds["mean_human_fragments_max"], "severity": "warning"})
        if metrics.get("height_abs_p90") is not None and metrics["height_abs_p90"] > self.thresholds["height_abs_p90_max"]:
            failures.append({"metric": "height_abs_p90", "value": metrics["height_abs_p90"], "threshold": self.thresholds["height_abs_p90_max"], "severity": "warning"})
        if metrics.get("heldout_psnr_mean") is not None and metrics["heldout_psnr_mean"] < self.thresholds["heldout_psnr_mean_min"]:
            failures.append({"metric": "heldout_psnr_mean", "value": metrics["heldout_psnr_mean"], "threshold": self.thresholds["heldout_psnr_mean_min"], "severity": "warning"})
        return {"num_failures": int(len(failures)), "failures": failures}

    def _write_html(self, out_path: Path, metrics: dict[str, Any], failures: dict[str, Any]) -> None:
        rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in metrics.items() if not isinstance(v, (list, dict)))
        fail_rows = "".join(f"<li>{f['metric']}: value={f['value']} threshold={f['threshold']}</li>" for f in failures.get("failures", []))
        html = f"""<!doctype html><html><head><meta charset='utf-8'><title>room4d evaluation report</title>
<style>body{{font-family:sans-serif;margin:24px}} table{{border-collapse:collapse}} td,th{{border:1px solid #ccc;padding:6px 10px}}</style></head>
<body><h2>room4d evaluation report</h2><table><tr><th>Metric</th><th>Value</th></tr>{rows}</table>
<h3>Failure report</h3><ul>{fail_rows or '<li>None</li>'}</ul></body></html>"""
        out_path.write_text(html, encoding="utf-8")

    def run(self, *, geometry_artifact=None, track_artifact=None, human_artifact=None, scene_artifact=None, render_artifact=None, renderer=None, output_dir: str | Path | None = None) -> dict:
        metrics: dict[str, Any] = {}
        metrics.update(self.geometry_metrics.compute(geometry_artifact))
        metrics.update(self.tracking_metrics.compute(track_artifact=track_artifact, geometry_artifact=geometry_artifact, human_artifact=human_artifact))
        metrics.update(self.rendering_metrics.compute(scene_artifact=scene_artifact, render_artifact=render_artifact, geometry_artifact=geometry_artifact, renderer=renderer))
        failures = self._failure_report(metrics)
        report = {"metrics": metrics, "failure_report": failures}
        if output_dir is not None:
            out_dir = ensure_dir(output_dir)
            write_json(out_dir / "evaluation_summary.json", report)
            self._write_html(out_dir / "evaluation_report.html", metrics, failures)
        return report
