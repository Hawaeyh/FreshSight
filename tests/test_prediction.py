import json
import os
from pathlib import Path
from unittest.mock import patch

from PIL import Image
import pytest
import torch

from web.services.prediction_service import PredictionService


MODEL_CONFIG = {
    "image_size": 224, "resize_size": 256, "dropout": 0.3,
    "mixed_precision": True,
}
REGISTRY = {"active_model": {
    "model_name": "FreshSight MobileNetV2", "model_version": "test-v1",
    "checkpoint_path": "missing.pth", "class_order": ["Fresh", "Unripe", "Rotten"],
    "active": True,
}}
WEB_CONFIG = {
    "web_inference": {"device": "auto", "allow_cpu_fallback": True},
    "confidence": {"high_threshold": 0.85, "moderate_threshold": 0.65,
                   "close_probability_margin": 0.10},
}


class FixedModel:
    def __init__(self, logits):
        self.logits = torch.tensor([logits], dtype=torch.float32)

    def __call__(self, tensor):
        return self.logits


def image_path(tmp_path):
    path = tmp_path / "papaya.jpg"
    Image.new("RGB", (300, 260), (100, 180, 60)).save(path)
    return path


def service_with_logits(tmp_path, logits):
    service = PredictionService(MODEL_CONFIG, REGISTRY, WEB_CONFIG)
    service.model = FixedModel(logits)
    service.device = torch.device("cpu")
    return service, image_path(tmp_path)


def test_active_model_registry_points_to_original_checkpoint():
    registry = json.loads(Path("config/model_registry.json").read_text(encoding="utf-8"))
    active = registry["active_model"]
    assert active["active"] is True
    assert active["checkpoint_path"] == "ai/checkpoints/mobilenetv2_cleaned_baseline/best_model.pth"
    assert active["class_order"] == ["Fresh", "Unripe", "Rotten"]


def test_checkpoint_missing_returns_structured_error(tmp_path):
    service = PredictionService(MODEL_CONFIG, REGISTRY, WEB_CONFIG)
    result = service.predict(str(image_path(tmp_path)))
    assert result["available"] is False
    assert result["status"] == "model_unavailable"
    assert "predicted_class" not in result


def test_class_mapping_and_probability_sum(tmp_path):
    service, path = service_with_logits(tmp_path, [0.1, 3.0, -1.0])
    result = service.predict(str(path))
    assert result["predicted_class"] == "Unripe"
    assert sum(result["probabilities"].values()) == pytest.approx(1.0)
    assert result["model_version"] == "test-v1"


def test_low_confidence_and_close_margin_warnings(tmp_path):
    service, path = service_with_logits(tmp_path, [1.0, 0.98, 0.0])
    result = service.predict(str(path))
    assert result["confidence_level"] == "low"
    assert result["top_two_probability_margin"] < 0.10
    assert result["uncertainty_warning"] is True
    assert len(result["uncertainty_reasons"]) == 2


@patch("web.services.prediction_service.torch.cuda.device_count", return_value=1)
@patch("web.services.prediction_service.torch.cuda.is_available", return_value=True)
def test_cuda_is_selected_when_available(mock_available, mock_count):
    service = PredictionService(MODEL_CONFIG, REGISTRY, WEB_CONFIG)
    assert service._select_web_device() == torch.device("cuda:0")


@patch("web.services.prediction_service.torch.cuda.device_count", return_value=0)
@patch("web.services.prediction_service.torch.cuda.is_available", return_value=False)
def test_controlled_cpu_fallback(mock_available, mock_count):
    service = PredictionService(MODEL_CONFIG, REGISTRY, WEB_CONFIG)
    assert service._select_web_device() == torch.device("cpu")


def test_corrupt_image_has_no_fabricated_prediction(tmp_path):
    path = tmp_path / "bad.jpg"
    path.write_bytes(b"not an image")
    result = PredictionService(MODEL_CONFIG, REGISTRY, WEB_CONFIG).predict(str(path))
    assert result["status"] == "invalid_image"
    assert "confidence" not in result


@pytest.mark.skipif(
    os.getenv("FRESHSIGHT_RUN_CHECKPOINT_TEST") != "1",
    reason="Set FRESHSIGHT_RUN_CHECKPOINT_TEST=1 for controlled real-checkpoint inference.",
)
def test_real_checkpoint_single_image_integration():
    registry = json.loads(Path("config/model_registry.json").read_text(encoding="utf-8"))
    web = json.loads(Path("config/web_config.json").read_text(encoding="utf-8"))
    model = json.loads(Path("config/model_config.json").read_text(encoding="utf-8"))
    sample = next((Path("dataset") / "fresh").glob("*.jpg"))
    result = PredictionService(model, registry, web).predict(str(sample))
    assert result["status"] == "success"
    assert sum(result["probabilities"].values()) == pytest.approx(1.0, abs=1e-5)
