import json

import numpy as np
import pandas as pd
import pytest

import evaluation.evaluate_mobilenetv2 as evaluation_module
from ai.dataset import CLASS_NAMES
from evaluation.evaluate_mobilenetv2 import calculate_metrics, save_model_comparison


def test_calculate_metrics_includes_per_class_roc_auc():
    y_true = [0, 0, 1, 1, 2, 2]
    probabilities = np.array([
        [0.90, 0.05, 0.05], [0.80, 0.10, 0.10],
        [0.05, 0.90, 0.05], [0.10, 0.80, 0.10],
        [0.05, 0.05, 0.90], [0.10, 0.10, 0.80],
    ])
    metrics = calculate_metrics(y_true, y_true, probabilities)
    assert metrics["report"]["accuracy"] == 1.0
    assert metrics["matrix"].tolist() == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    assert all(metrics["roc"][name]["auc"] == 1.0 for name in CLASS_NAMES)


def test_model_comparison_uses_real_ai_and_matlab_values(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation_module, "OUTPUT_DIR", tmp_path)
    ai = {
        "overall_accuracy": 0.9, "macro_f1_score": 0.88,
        "weighted_f1_score": 0.89, "average_inference_time_seconds": 0.01,
        "per_class": {
            name: {"f1-score": value}
            for name, value in zip(CLASS_NAMES, (0.85, 0.95, 0.84))
        },
    }
    matlab = {
        "overall_accuracy": 0.6395, "macro_f1_score": 0.6402,
        "weighted_f1_score": 0.6483, "average_processing_time_seconds": 0.296,
        "per_class": {
            name: {"f1-score": value}
            for name, value in zip(CLASS_NAMES, (0.5385, 0.8085, 0.5738))
        },
    }
    save_model_comparison(ai, matlab)
    expected = {
        "model_comparison.csv", "model_comparison.md", "accuracy_comparison.png",
        "macro_f1_comparison.png", "per_class_f1_comparison.png",
        "processing_time_comparison.png",
    }
    assert expected == {path.name for path in tmp_path.iterdir()}
    comparison = (tmp_path / "model_comparison.csv").read_text(encoding="utf-8")
    assert "0.9" in comparison
    assert "0.6395" in comparison


def test_configured_class_order_is_exact():
    config = evaluation_module.load_config()
    assert config["classes"] == ["Fresh", "Unripe", "Rotten"]


def test_no_ai_metrics_are_hardcoded_in_evaluator_source():
    source = (
        evaluation_module.PROJECT_ROOT / "evaluation" / "evaluate_mobilenetv2.py"
    ).read_text(encoding="utf-8")
    assert '"overall_accuracy": float(report["accuracy"])' in source
    assert "0.6395" not in source


def test_input_validation_rejects_train_test_hash_overlap(tmp_path, monkeypatch):
    manifest_path = tmp_path / "manifest.csv"
    checkpoint_path = tmp_path / "best_model.pth"
    manifest_path.write_text("placeholder", encoding="utf-8")
    checkpoint_path.write_bytes(b"checkpoint")
    rows = []
    counts = {"Fresh": 42, "Unripe": 53, "Rotten": 52}
    index = 0
    for class_index, class_name in enumerate(CLASS_NAMES):
        for _ in range(counts[class_name]):
            rows.append({
                "source_path": str(tmp_path / f"image_{index}.jpg"),
                "class_name": class_name, "class_index": class_index,
                "split": "test", "sha256": f"test-{index}",
                "duplicate_group_id": f"test-{index}",
            })
            index += 1
    rows.append({
        "source_path": str(tmp_path / "train.jpg"), "class_name": "Fresh",
        "class_index": 0, "split": "train", "sha256": "test-0",
        "duplicate_group_id": "test-0",
    })
    fake_manifest = pd.DataFrame(rows)

    def resolve(path_value):
        return checkpoint_path if str(path_value).endswith("best_model.pth") else manifest_path

    monkeypatch.setattr(evaluation_module, "resolve_project_path", resolve)
    monkeypatch.setattr(evaluation_module, "load_manifest", lambda *args, **kwargs: fake_manifest)
    with pytest.raises(RuntimeError, match="Train/test SHA-256 overlap"):
        evaluation_module.validate_evaluation_inputs({
            "manifest_path": "manifest.csv", "best_checkpoint_path": "best_model.pth"
        })
