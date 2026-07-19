from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from flask import Blueprint, current_app, jsonify, request, send_file, url_for
from werkzeug.utils import secure_filename

from config.paths import OUTPUTS_DIR, UPLOADS_DIR
from recommendation.recommendation_engine import RecommendationEngine
from web.services.history_service import get_history_service
from web.services.active_learning_service import score_candidate
from web.services.hybrid_analysis_service import HybridAnalysisService
from web.services.matlab_service import get_matlab_service
from web.services.prediction_service import get_prediction_service
from web.services.report_service import ReportService

analysis_bp = Blueprint("analysis", __name__)


def _allowed_file(filename, allowed_ext):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_ext


@analysis_bp.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return jsonify({"error": "No image file was uploaded."}), 400
    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Please choose an image file."}), 400
    if not _allowed_file(file.filename, current_app.config["UPLOAD_EXTENSIONS"]):
        return jsonify({"error": "Unsupported file type."}), 400

    original_name = secure_filename(file.filename)
    extension = original_name.rsplit(".", 1)[1].lower()
    analysis_uuid = uuid4().hex
    saved_path = UPLOADS_DIR / f"{analysis_uuid}.{extension}"
    try:
        file.save(saved_path)
    except Exception as exc:
        return jsonify({"error": f"Failed to save uploaded file: {exc}"}), 500

    prediction = get_prediction_service(
        current_app.config["MODEL_CONFIG"], current_app.config["MODEL_REGISTRY"],
        current_app.config["WEB_CONFIG"],
    )
    hybrid = HybridAnalysisService(
        prediction, get_matlab_service(), current_app.config["WEB_CONFIG"]
    ).analyze_image(str(saved_path))
    recommendation = asdict(RecommendationEngine().recommend(hybrid))
    hybrid["recommendation_assistant"] = recommendation
    hybrid["analysis_uuid"] = analysis_uuid

    history = get_history_service()
    analysis_id = history.record_analysis(
        analysis_uuid, str(saved_path.resolve()), original_name, hybrid
    )
    candidate_score, candidate_reasons = score_candidate(hybrid)
    history.save_candidate(analysis_id, candidate_score, candidate_reasons)
    try:
        report_path = ReportService().generate(
            analysis_uuid, original_name, str(saved_path), hybrid, recommendation
        )
        hybrid["report"] = {
            "available": True,
            "download_url": url_for("analysis.download_report", analysis_uuid=analysis_uuid),
            "local_path": str(report_path),
        }
    except Exception as exc:
        current_app.logger.exception("Local report generation failed")
        hybrid["report"] = {"available": False, "error": str(exc)}
    hybrid["verification"] = {
        "question": "Was the AI prediction correct?",
        "options": ["correct", "incorrect"],
    }
    return jsonify(hybrid)


@analysis_bp.route("/verify", methods=["POST"])
def verify():
    data = request.get_json(silent=True) or {}
    try:
        feedback = get_history_service().record_feedback(
            str(data.get("analysis_uuid", "")),
            str(data.get("feedback_status", "")),
            data.get("corrected_class"),
        )
    except ValueError as exc:
        return jsonify({"status": "invalid", "error": str(exc)}), 400
    except LookupError as exc:
        return jsonify({"status": "not_found", "error": str(exc)}), 404
    if feedback["feedback_status"] == "incorrect":
        get_history_service().save_candidate(
            feedback["analysis_id"], 1.0, ["User correction submitted"]
        )
    return jsonify({
        "status": "recorded", "feedback": feedback,
        "message": "Feedback stored for later review. No retraining was started.",
    })


@analysis_bp.route("/report/<analysis_uuid>", methods=["GET"])
def download_report(analysis_uuid):
    if not analysis_uuid.isalnum():
        return jsonify({"error": "Invalid analysis ID."}), 400
    path = OUTPUTS_DIR / "analysis_reports" / f"{analysis_uuid}.html"
    if not path.is_file():
        return jsonify({"error": "Report not found."}), 404
    return send_file(path, as_attachment=True, download_name=f"FreshSight-{analysis_uuid}.html")
