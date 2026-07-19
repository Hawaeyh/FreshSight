"""Generate standalone local reports with optional enhanced MATLAB evidence."""

from __future__ import annotations

import base64
import html
import mimetypes
from datetime import datetime, timezone
from pathlib import Path

from config.paths import OUTPUTS_DIR

REPORT_DIR = OUTPUTS_DIR / "analysis_reports"


def _value(value, suffix=""):
    return "Not available." if value is None else f"{value}{suffix}"


def _source_data_url(path: str) -> str | None:
    source = Path(path)
    if not source.is_file():
        return None
    mime = mimetypes.guess_type(source.name)[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(source.read_bytes()).decode('ascii')}"


def _rows(values: dict) -> str:
    rows = "".join(
        f"<tr><th>{html.escape(str(key).replace('_', ' ').title())}</th>"
        f"<td>{html.escape(_value(value))}</td></tr>"
        for key, value in values.items() if not isinstance(value, (dict, list))
    )
    return rows or "<tr><td colspan='2'>Not available.</td></tr>"


class ReportService:
    def __init__(self, report_dir: Path = REPORT_DIR):
        self.report_dir = Path(report_dir)

    def generate(self, analysis_uuid: str, original_filename: str, stored_image_path: str,
                 result: dict, recommendation: dict) -> Path:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        ai = result.get("ai_detection") or {}
        matlab = result.get("matlab_analysis") or {}
        assessment = result.get("system_assessment") or {}
        measurements = matlab.get("measurements") or {}
        colours = matlab.get("colour_analysis") or {}
        texture = matlab.get("texture_analysis") or {}
        segmentation = matlab.get("segmentation_quality") or {}
        reliability = matlab.get("measurement_reliability") or {}
        probabilities = ai.get("probabilities") or {}
        images = matlab.get("images") or {}
        image_specs = [
            ("Original", None), ("Background removed", "background_removed"),
            ("Papaya mask", "papaya_mask_clean"), ("Inner mask", "papaya_mask_inner"),
            ("Hue", "hue_channel"), ("Saturation", "saturation_channel"), ("Value", "value_channel"),
            ("Colour segmentation", "colour_segmentation_overlay"),
            ("L channel", "l_channel"), ("a channel", "a_channel"), ("b channel", "b_channel"),
            ("Texture map", "texture_map"), ("Edge map", "edge_map"),
            ("Damage mask", "damage_mask"), ("Damage highlight", "damage_highlight"),
        ]
        image_cards = []
        for title, key in image_specs:
            source = _source_data_url(stored_image_path) if key is None else (
                f"data:image/png;base64,{images[key]}" if images.get(key) else None
            )
            if source:  # Unavailable images are deliberately omitted.
                image_cards.append(f"<div><h3>{html.escape(title)}</h3><img src='{source}' alt='{html.escape(title)}'></div>")
        probability_rows = "".join(
            f"<li>{name}: {float(probabilities[name]) * 100:.2f}%</li>"
            for name in ("Fresh", "Unripe", "Rotten") if probabilities.get(name) is not None
        ) or "<li>Not available.</li>"
        reasons = "".join(f"<li>{html.escape(str(item))}</li>" for item in assessment.get("review_reasons", [])) or "<li>None</li>"
        recommendations = "".join(f"<li>{html.escape(str(item))}</li>" for item in recommendation.get("recommendations", [])) or "<li>Not available.</li>"
        document = f"""<!doctype html><html><head><meta charset='utf-8'><title>FreshSight Report</title>
<style>body{{font:16px Arial;max-width:1050px;margin:30px auto;color:#17202a}}h1{{color:#087f5b}}section{{border:1px solid #ddd;border-radius:12px;padding:18px;margin:16px 0}}table{{border-collapse:collapse;width:100%}}th,td{{padding:7px;border-bottom:1px solid #eee;text-align:left}}.images{{display:grid;grid-template-columns:repeat(2,1fr);gap:16px}}img{{max-width:100%;max-height:360px;object-fit:contain}}.warning{{background:#fff3bf}}@media print{{.images{{break-inside:auto}}}}</style></head><body>
<h1>FreshSight Analysis Report</h1><p>Analysis ID: {html.escape(analysis_uuid)}<br>Generated: {datetime.now(timezone.utc).isoformat()}<br>Original filename: {html.escape(original_filename)}</p>
<section class='warning'><h2>Food Safety Assessment</h2><p><strong>{html.escape(recommendation.get('food_safety_status', 'Manual Inspection Required'))}</strong></p><p>{html.escape(recommendation.get('food_safety_explanation', 'Not available.'))}</p><p><strong>Recommended action:</strong> {html.escape(recommendation.get('precautionary_action', recommendation.get('handling_guidance', 'Inspect manually.')))}</p></section>
<section><h2>AI Detection — Primary Classification</h2><p>Class: {_value(ai.get('predicted_class'))}<br>Confidence: {_value(f"{float(ai['confidence']) * 100:.2f}%" if ai.get('confidence') is not None else None)}<br>Model: {_value(ai.get('model_name'))} / {_value(ai.get('model_version'))}<br>Device: {_value(ai.get('device'))}</p><ul>{probability_rows}</ul></section>
<section><h2>MATLAB Supporting Analysis</h2><p>Rule class: {_value(matlab.get('rule_class'))}<br>Grade: {_value(matlab.get('grade'))}<br>Damage severity: {_value(matlab.get('damage_severity'))}<br>Damage: {_value(matlab.get('damage_percentage'), '%')}<br>Healthy: {_value(matlab.get('healthy_percentage'), '%')}</p></section>
<section class='warning'><h2>Segmentation quality and reliability</h2><h3>Segmentation</h3><table>{_rows(segmentation)}</table><h3>Measurement reliability</h3><table>{_rows(reliability)}</table></section>
<section><h2>Colour percentages</h2><table>{_rows(colours)}</table></section>
<section><h2>Texture features</h2><table>{_rows(texture)}</table></section>
<section><h2>All measurements</h2><table>{_rows(measurements)}</table></section>
<section><h2>System Assessment</h2><p>AI/MATLAB agreement: {_value(assessment.get('ai_matlab_agreement'))}<br>Manual review required: {_value(assessment.get('requires_manual_review'))}</p><ul>{reasons}</ul></section>
<section><h2>{html.escape(recommendation.get('title', 'FreshSight Recommendation Assistant'))}</h2><p>{html.escape(recommendation.get('matlab_support_statement', ''))}</p><p>{html.escape(recommendation.get('disagreement_statement', ''))}</p><ul>{recommendations}</ul></section>
<section><h2>Available images</h2><div class='images'>{''.join(image_cards) or '<p>Not available.</p>'}</div></section>
<footer><strong>Disclaimer:</strong> {html.escape(recommendation.get('disclaimer', 'Not available.'))}</footer></body></html>"""
        path = self.report_dir / f"{analysis_uuid}.html"
        path.write_text(document, encoding="utf-8")
        return path
