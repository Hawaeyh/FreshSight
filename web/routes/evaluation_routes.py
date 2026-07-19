from flask import Blueprint, jsonify

evaluation_bp = Blueprint("evaluation", __name__)


@evaluation_bp.route("/status")
def status():
    return jsonify({"status": "ready", "message": "Evaluation endpoints are not yet implemented."})
