from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

import torch
from flask import Blueprint, current_app, jsonify, redirect, render_template, request, send_file, url_for

from config.paths import BASE_DIR, DB_PATH, EVALUATION_OUTPUTS_DIR, EXPLAINABILITY_DIR
from web.services.gradcam_service import GradCAMService
from web.services.history_service import get_history_service
from web.services.matlab_service import get_matlab_service
from web.services.prediction_service import get_prediction_service

dashboard_bp = Blueprint("dashboard", __name__)


@dashboard_bp.get("/history")
def history():
    return render_template("analysis_history.html", analyses=get_history_service().list_analyses())


@dashboard_bp.get("/analysis/<analysis_uuid>")
def analysis_detail(analysis_uuid):
    row = get_history_service().get_analysis(analysis_uuid)
    if row is None:
        return jsonify({"error": "Analysis not found."}), 404
    return render_template("analysis_detail.html", analysis=row)


@dashboard_bp.get("/admin/feedback")
def feedback_review():
    service = get_history_service()
    return render_template(
        "feedback_review.html", feedback=service.list_feedback(), candidates=service.list_candidates()
    )


@dashboard_bp.post("/admin/feedback/<int:feedback_id>/review")
def review_feedback(feedback_id):
    try:
        get_history_service().review_feedback(feedback_id, request.form.get("status", ""))
    except (ValueError, LookupError) as exc:
        return jsonify({"error": str(exc)}), 400
    return redirect(url_for("dashboard.feedback_review"))


def _load_json(path: Path):
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@dashboard_bp.get("/dashboard/performance")
def performance():
    ai = _load_json(EVALUATION_OUTPUTS_DIR / "mobilenetv2_cleaned_baseline" / "evaluation_summary.json")
    matlab_csv = EVALUATION_OUTPUTS_DIR / "matlab_rule_based" / "tables" / "metrics_summary.csv"
    matlab = {}
    if matlab_csv.is_file():
        with matlab_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("Metric") and row.get("Value"):
                    matlab[row["Metric"]] = float(row["Value"])
    return render_template("performance_dashboard.html", ai=ai, matlab=matlab)


@dashboard_bp.get("/system-health")
def system_health():
    registry = current_app.config["MODEL_REGISTRY"]["active_model"]
    checkpoint = (BASE_DIR / registry["checkpoint_path"]).resolve()
    matlab = get_matlab_service()
    db_ok = False
    try:
        with sqlite3.connect(DB_PATH) as connection:
            connection.execute("SELECT 1").fetchone()
        db_ok = True
    except sqlite3.Error:
        pass
    health = {
        "model_version": registry["model_version"],
        "checkpoint_exists": checkpoint.is_file(),
        "checkpoint_path": str(checkpoint),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "matlab_engine_importable": bool(getattr(matlab, "_engine_module", None)),
        "matlab_engine_started": bool(getattr(matlab, "engine_started", False)),
        "database_ok": db_ok,
    }
    return render_template("system_health.html", health=health)


@dashboard_bp.post("/analysis/<analysis_uuid>/gradcam")
def gradcam(analysis_uuid):
    row = get_history_service().get_analysis(analysis_uuid)
    if row is None:
        return jsonify({"error": "Analysis not found."}), 404
    prediction = get_prediction_service(
        current_app.config["MODEL_CONFIG"], current_app.config["MODEL_REGISTRY"],
        current_app.config["WEB_CONFIG"],
    )
    try:
        path = GradCAMService(prediction).generate(row["stored_image_path"], analysis_uuid)
    except Exception as exc:
        current_app.logger.exception("Grad-CAM generation failed")
        return jsonify({"error": f"Grad-CAM generation failed: {exc}"}), 500
    return jsonify({"status": "generated", "image_url": url_for("dashboard.gradcam_image", analysis_uuid=analysis_uuid), "path": str(path)})


@dashboard_bp.get("/analysis/<analysis_uuid>/gradcam.png")
def gradcam_image(analysis_uuid):
    path = EXPLAINABILITY_DIR / f"{analysis_uuid}.png"
    if not path.is_file():
        return jsonify({"error": "Grad-CAM has not been generated."}), 404
    return send_file(path, mimetype="image/png")
