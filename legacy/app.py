import json
import os
from pathlib import Path
from uuid import uuid4

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

import matlab.engine


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "tif", "tiff"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

UPLOAD_DIR.mkdir(exist_ok=True)

print("Starting MATLAB engine for FreshSight V2...")
eng = matlab.engine.start_matlab()
eng.addpath(str(BASE_DIR), nargout=0)
eng.addpath(str(BASE_DIR / "functions"), nargout=0)
print("MATLAB engine is ready.")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return jsonify({"error": "No image file was uploaded."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Please choose an image file."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Please upload a valid image."}), 400

    original_name = secure_filename(file.filename)
    extension = original_name.rsplit(".", 1)[1].lower()
    saved_name = f"{uuid4().hex}.{extension}"
    saved_path = UPLOAD_DIR / saved_name

    try:
        file.save(saved_path)
        image_path = os.path.abspath(saved_path)
        matlab_json = eng.runFreshSightAPI(image_path, nargout=1)
        parsed = json.loads(matlab_json)
        return jsonify(parsed)
    except Exception as exc:
        app.logger.exception("FreshSight analysis failed")
        return jsonify({"error": f"MATLAB processing failed: {exc}"}), 500


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, host="127.0.0.1", port=5000)
