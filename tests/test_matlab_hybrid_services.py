import json

import pytest

import web.services.matlab_service as matlab_module
from web.services.hybrid_analysis_service import HybridAnalysisService
from web.services.matlab_service import MatlabService


class FakeFuture:
    def __init__(self, value=None, error=None):
        self.value = value
        self.error = error

    def result(self, timeout=None):
        if self.error:
            raise self.error
        return self.value


class FakeEngine:
    def __init__(self, api_path="C:/project/matlab/api/run_freshsight_api.m", result=None, error=None):
        self.api_path = api_path
        self.result = result
        self.error = error
        self.quit_called = False

    def genpath(self, path):
        return path

    def addpath(self, path, nargout=0):
        self.added_path = path

    def which(self, name, nargout=1):
        return self.api_path

    def run_freshsight_api(self, image_path, nargout=1, background=True):
        return FakeFuture(self.result, self.error)

    def quit(self):
        self.quit_called = True


class FakeEngineModule:
    def __init__(self, engine):
        self.engine = engine

    def start_matlab(self):
        return self.engine


class StubService:
    def __init__(self, result):
        self.result = result

    def predict(self, image_path):
        return self.result

    def analyze_image(self, image_path):
        return self.result


def test_engine_unavailable_returns_structured_failure(monkeypatch, tmp_path):
    image = tmp_path / "papaya.jpg"
    image.write_bytes(b"image")
    monkeypatch.setattr(matlab_module, "_matlab_engine", None)
    result = MatlabService().analyze_image(str(image))
    assert result["available"] is False
    assert result["status"] == "unavailable"
    assert "measurements" not in result


def test_missing_matlab_api_is_reported_and_engine_closed(tmp_path):
    image = tmp_path / "papaya.jpg"
    image.write_bytes(b"image")
    engine = FakeEngine(api_path="")
    result = MatlabService(FakeEngineModule(engine)).analyze_image(str(image))
    assert result["status"] == "unavailable"
    assert "not found" in result["error"]
    assert engine.quit_called


def test_missing_image_does_not_start_engine(tmp_path):
    engine = FakeEngine()
    service = MatlabService(FakeEngineModule(engine))
    result = service.analyze_image(str(tmp_path / "missing.jpg"))
    assert result["status"] == "unavailable"
    assert service.engine_started is False
    assert "measurements" not in result


def test_matlab_processing_error_has_no_fake_measurements(tmp_path):
    image = tmp_path / "papaya.jpg"
    image.write_bytes(b"image")
    engine = FakeEngine(error=RuntimeError("MATLAB exploded"))
    result = MatlabService(FakeEngineModule(engine)).analyze_image(str(image))
    assert result["available"] is True
    assert result["status"] == "error"
    assert "MATLAB exploded" in result["error"]
    assert "measurements" not in result


def test_matlab_success_preserves_actual_fields(tmp_path):
    image = tmp_path / "papaya.jpg"
    image.write_bytes(b"image")
    payload = {
        "status": "success",
        "error": "",
        "rule_class": "Fresh",
        "measurements": {"damage_percentage": 2.5},
    }
    engine = FakeEngine(result=json.dumps(payload))
    result = MatlabService(FakeEngineModule(engine)).analyze_image(str(image))
    assert result["available"] is True
    assert result["rule_class"] == "Fresh"
    assert result["measurements"] == {"damage_percentage": 2.5}


def test_unreliable_segmentation_suppresses_measurements(tmp_path):
    image = tmp_path / "papaya.jpg"
    image.write_bytes(b"image")
    payload = {
        "status": "success", "rule_class": "Rotten", "grade": "Grade D",
        "freshness_score": 80, "damage_percentage": 7, "healthy_percentage": 93,
        "measurements": {"damage_percentage": 7, "healthy_percentage": 93, "lesion_percentage": 4},
        "measurement_reliability": {"segmentation_reliable": False, "matlab_class_reliable": False},
    }
    result = MatlabService(FakeEngineModule(FakeEngine(result=json.dumps(payload)))).analyze_image(str(image))
    assert result["rule_class"] == "Unavailable"
    assert result["damage_percentage"] is None
    assert result["measurements"]["healthy_percentage"] is None


def _hybrid(ai, matlab):
    return HybridAnalysisService(StubService(ai), StubService(matlab), {
        "confidence": {"moderate_threshold": 0.65}
    }).analyze_image("unused.jpg")


def test_ai_missing_checkpoint_does_not_block_matlab():
    result = _hybrid(
        {"available": False, "status": "model_unavailable", "error": "AI model checkpoint is unavailable."},
        {"available": True, "status": "success", "error": "", "rule_class": "Fresh"},
    )
    assert result["ai_detection"]["status"] == "model_unavailable"
    assert result["matlab_analysis"]["rule_class"] == "Fresh"
    assert result["system_assessment"]["status"] == "partial"
    assert result["system_assessment"]["requires_manual_review"] is True


def test_matlab_unavailable_does_not_block_ai():
    result = _hybrid(
        {"available": True, "status": "success", "predicted_class": "Fresh", "confidence": 0.9, "uncertainty_reasons": []},
        {"available": False, "status": "unavailable", "error": "No engine"},
    )
    assert result["ai_detection"]["available"] is True
    assert result["system_assessment"]["primary_classification"] == "Fresh"
    assert result["system_assessment"]["requires_manual_review"] is True


@pytest.mark.parametrize(
    "ai_class,rule_class,confidence,agreement,review",
    [
        ("Fresh", "Fresh", 0.90, True, False),
        ("Fresh", "Rotten", 0.90, False, True),
        ("Fresh", "Fresh", 0.40, True, True),
    ],
)
def test_agreement_disagreement_and_low_confidence(
    ai_class, rule_class, confidence, agreement, review
):
    result = _hybrid(
        {"available": True, "status": "success", "predicted_class": ai_class, "confidence": confidence,
         "uncertainty_reasons": ["AI confidence is low."] if confidence < 0.65 else []},
        {"available": True, "status": "success", "error": "", "rule_class": rule_class},
    )
    assessment = result["system_assessment"]
    assert assessment["ai_matlab_agreement"] is agreement
    assert assessment["requires_manual_review"] is review
    assert assessment["primary_classification_source"] == "MobileNetV2"


def test_both_unavailable_is_clear_failure_without_fabricated_values():
    result = _hybrid(
        {"available": False, "status": "model_unavailable", "error": "AI model checkpoint is unavailable."},
        {"available": False, "status": "unavailable", "error": "No MATLAB"},
    )
    assert result["system_assessment"]["status"] == "failed"
    assert result["system_assessment"]["primary_classification"] is None
    assert "measurements" not in result["matlab_analysis"]


def test_unreliable_matlab_class_does_not_create_agreement_and_requires_review():
    result = _hybrid(
        {"available": True, "status": "success", "predicted_class": "Rotten", "confidence": 0.99, "uncertainty_reasons": []},
        {"available": True, "status": "success", "rule_class": "Rotten",
         "measurement_reliability": {"segmentation_reliable": False, "matlab_class_reliable": False}},
    )
    assessment = result["system_assessment"]
    assert assessment["ai_matlab_agreement"] is None
    assert assessment["primary_classification"] == "Rotten"
    assert assessment["requires_manual_review"] is True
