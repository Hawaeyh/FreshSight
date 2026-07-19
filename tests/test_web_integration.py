import sqlite3
from dataclasses import asdict

from PIL import Image
import pytest
import torch
from torchvision import transforms
import threading

from recommendation.recommendation_engine import RecommendationEngine
from web.services.history_service import HistoryService
from web.services.report_service import ReportService
from web.services.active_learning_service import score_candidate
from web.services.gradcam_service import GradCAMService


def successful_result():
    return {
        "ai_detection": {
            "available": True, "status": "success", "predicted_class": "Fresh",
            "confidence": 0.91,
            "probabilities": {"Fresh": 0.91, "Unripe": 0.06, "Rotten": 0.03},
            "model_version": "v1",
        },
        "matlab_analysis": {
            "available": True, "status": "success", "rule_class": "Fresh",
            "grade": "Grade A", "freshness_score": 95.0,
            "damage_percentage": 5.0, "healthy_percentage": 95.0,
            "measurements": {"damage_percentage": 5.0, "healthy_percentage": 95.0,
                             "lesion_percentage": 1.0, "white_mold_percentage": 0.0},
            "images": {},
        },
        "system_assessment": {
            "ai_matlab_agreement": True, "requires_manual_review": False,
            "review_reasons": [],
        },
    }


def test_sqlite_schema_and_analysis_persistence(tmp_path):
    service = HistoryService(tmp_path / "freshsight.db")
    analysis_id = service.record_analysis("uuid-1", "image.jpg", "original.jpg", successful_result())
    assert analysis_id == 1
    stored = service.get_analysis("uuid-1")
    assert stored["predicted_class"] == "Fresh"
    assert stored["damage_percentage"] == 5.0
    with sqlite3.connect(service.database_path) as connection:
        tables = {row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
    assert {"analysis_history", "prediction_feedback"} <= tables


def test_correct_and_incorrect_feedback_do_not_trigger_retraining(tmp_path):
    service = HistoryService(tmp_path / "freshsight.db")
    service.record_analysis("yes", "image.jpg", "original.jpg", successful_result())
    service.record_analysis("no", "image.jpg", "original.jpg", successful_result())
    correct = service.record_feedback("yes", "correct")
    incorrect = service.record_feedback("no", "incorrect", "Rotten")
    assert correct["corrected_class"] is None
    assert incorrect["predicted_class"] == "Fresh"
    assert incorrect["corrected_class"] == "Rotten"
    assert correct["included_in_retraining"] is False
    assert incorrect["included_in_retraining"] is False


def test_unsupported_corrected_label_is_rejected(tmp_path):
    service = HistoryService(tmp_path / "freshsight.db")
    service.record_analysis("uuid", "image.jpg", "original.jpg", successful_result())
    with pytest.raises(ValueError, match="corrected_class"):
        service.record_feedback("uuid", "incorrect", "Semi-Fresh")


def test_recommendation_generation_and_manual_review_guidance():
    result = successful_result()
    result["system_assessment"] = {
        "ai_matlab_agreement": False, "requires_manual_review": True,
        "review_reasons": ["AI and MATLAB classifications disagree."],
    }
    recommendation = RecommendationEngine().recommend(result)
    assert recommendation.title == "FreshSight Recommendation Assistant"
    assert any("manual verification" in item for item in recommendation.recommendations)
    assert not any("immediate consumption" in item for item in recommendation.recommendations)
    assert any("better lighting" in item for item in recommendation.recommendations)
    assert "not food-safety certification" in recommendation.disclaimer


def test_recommendation_omits_unreliable_matlab_percentages():
    result = successful_result()
    result["ai_detection"]["predicted_class"] = "Rotten"
    result["matlab_analysis"]["measurement_reliability"] = {
        "segmentation_reliable": False, "damage_reliable": False,
        "lesion_reliable": False, "mold_reliable": False,
    }
    result["matlab_analysis"]["segmentation_quality"] = {"status": "poor"}
    result["system_assessment"]["requires_manual_review"] = True
    recommendation = RecommendationEngine().recommend(result)
    combined = " ".join(recommendation.rationale + recommendation.recommendations)
    assert "93.01%" not in combined
    assert "healthy area" not in combined
    assert "unavailable or unreliable" in recommendation.matlab_support_statement
    assert recommendation.primary_assessment == "MobileNetV2 predicts Rotten."


def test_reliable_disagreement_with_fresh_ai_uses_cautious_guidance():
    result = successful_result()
    result["matlab_analysis"]["rule_class"] = "Rotten"
    result["matlab_analysis"]["measurement_reliability"] = {
        "segmentation_reliable": True, "damage_reliable": True,
        "lesion_reliable": True, "mold_reliable": True,
    }
    result["matlab_analysis"]["damage_evidence"] = {
        "dark_decay_percentage": 4.2, "abnormal_texture_percentage": 2.1,
    }
    result["system_assessment"] = {
        "ai_matlab_agreement": False, "requires_manual_review": True,
        "review_reasons": ["AI and MATLAB classifications disagree."],
    }
    recommendation = RecommendationEngine().recommend(result)
    combined = " ".join(recommendation.recommendations)
    assert "manual verification" in combined
    assert "Retake" in combined
    assert "immediate consumption" not in combined
    assert recommendation.predicted_class == "Fresh"


def test_report_generation_embeds_available_values_and_not_available(tmp_path):
    image = tmp_path / "source.jpg"
    Image.new("RGB", (80, 60), (120, 180, 60)).save(image)
    result = successful_result()
    recommendation = asdict(RecommendationEngine().recommend(result))
    path = ReportService(tmp_path / "reports").generate(
        "uuid-report", "papaya.jpg", str(image), result, recommendation
    )
    content = path.read_text(encoding="utf-8")
    assert "FreshSight Analysis Report" in content
    assert "FreshSight MobileNetV2" not in content or "Fresh" in content
    assert "data:image/jpeg;base64" in content
    assert "Not available." in content


def test_failure_values_are_not_fabricated_in_history(tmp_path):
    service = HistoryService(tmp_path / "freshsight.db")
    failed = {
        "ai_detection": {"available": False, "status": "model_unavailable", "error": "missing"},
        "matlab_analysis": {"available": False, "status": "unavailable", "error": "missing"},
    }
    service.record_analysis("failed", "image.jpg", "original.jpg", failed)
    stored = service.get_analysis("failed")
    assert stored["predicted_class"] is None
    assert stored["confidence"] is None
    assert stored["damage_percentage"] is None


def test_duplicate_feedback_image_is_rejected_and_original_prediction_is_preserved(tmp_path):
    image_a = tmp_path / "a.jpg"
    image_b = tmp_path / "b.jpg"
    Image.new("RGB", (20, 20), "green").save(image_a)
    image_b.write_bytes(image_a.read_bytes())
    service = HistoryService(tmp_path / "freshsight.db")
    service.record_analysis("first", str(image_a), "a.jpg", successful_result())
    service.record_analysis("second", str(image_b), "b.jpg", successful_result())
    service.record_feedback("first", "incorrect", "Rotten")
    with pytest.raises(ValueError, match="identical image content"):
        service.record_feedback("second", "correct")
    assert service.get_analysis("first")["predicted_class"] == "Fresh"


def test_active_learning_scoring_prioritizes_disagreement_and_correction():
    result = successful_result()
    result["ai_detection"]["confidence"] = 0.60
    result["ai_detection"]["top_two_probability_margin"] = 0.04
    result["system_assessment"]["ai_matlab_agreement"] = False
    result["system_assessment"]["requires_manual_review"] = True
    score, reasons = score_candidate(result, feedback_status="incorrect")
    assert score > 0.8
    assert "AI/MATLAB disagreement" in reasons
    assert "User correction submitted" in reasons


def test_feedback_review_changes_review_state_only(tmp_path):
    image = tmp_path / "papaya.jpg"
    Image.new("RGB", (20, 20), "orange").save(image)
    service = HistoryService(tmp_path / "freshsight.db")
    analysis_id = service.record_analysis("uuid", str(image), "papaya.jpg", successful_result())
    feedback = service.record_feedback("uuid", "correct")
    service.review_feedback(feedback["id"], "approved")
    row = service.list_feedback()[0]
    assert row["review_status"] == "approved"
    assert row["included_in_retraining"] == 0
    service.save_candidate(analysis_id, 0.7, ["Manual review required"])
    assert service.list_candidates()[0]["candidate_score"] == pytest.approx(0.7)


def test_gradcam_does_not_change_model_weights(tmp_path, monkeypatch):
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.features = torch.nn.Sequential(
                torch.nn.Conv2d(3, 4, 3, padding=1), torch.nn.ReLU()
            )
            self.classifier = torch.nn.Linear(4, 3)

        def forward(self, value):
            value = self.features(value).mean(dim=(2, 3))
            return self.classifier(value)

    model = TinyModel().eval()
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}

    class FakePrediction:
        device = torch.device("cpu")
        transform = transforms.Compose([transforms.Resize((24, 24)), transforms.ToTensor()])
        _inference_lock = threading.Lock()

        def _ensure_model(self):
            return model

    image = tmp_path / "papaya.jpg"
    Image.new("RGB", (30, 20), "orange").save(image)
    monkeypatch.setattr("web.services.gradcam_service.EXPLAINABILITY_DIR", tmp_path / "cams")
    output = GradCAMService(FakePrediction()).generate(str(image), "test-cam")
    assert output.is_file()
    assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())
