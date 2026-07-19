import json
from pathlib import Path

from flask import Flask

from config.paths import BASE_DIR
from web.routes.main_routes import main_bp
from web.routes.analysis_routes import analysis_bp
from web.routes.evaluation_routes import evaluation_bp
from web.routes.dashboard_routes import dashboard_bp
from web.services.history_service import HistoryService


def create_app():
    app = Flask(__name__, template_folder=str(Path(__file__).resolve().parent / "templates"))
    app.config.from_mapping({
        "MAX_CONTENT_LENGTH": 16 * 1024 * 1024,
        "UPLOAD_EXTENSIONS": {"jpg", "jpeg", "png", "bmp", "tif", "tiff"},
    })

    app.register_blueprint(main_bp)
    app.register_blueprint(analysis_bp)
    app.register_blueprint(evaluation_bp)
    app.register_blueprint(dashboard_bp)

    with open(BASE_DIR / "config" / "model_config.json", "r", encoding="utf-8") as config_file:
        app.config["MODEL_CONFIG"] = json.load(config_file)
    with open(BASE_DIR / "config" / "model_registry.json", "r", encoding="utf-8") as registry_file:
        app.config["MODEL_REGISTRY"] = json.load(registry_file)
    with open(BASE_DIR / "config" / "web_config.json", "r", encoding="utf-8") as web_file:
        app.config["WEB_CONFIG"] = json.load(web_file)

    HistoryService()

    return app
