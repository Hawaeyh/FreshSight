import csv

import pytest
import numpy as np
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

from scripts.run_matlab_evaluation import (
    CLASSES,
    EXPECTED_COUNTS,
    metric_rows,
    read_and_validate_test_rows,
)
from scripts.matlab_evaluation_reporting import (
    generate_report_artifacts,
    infer_rule_activations,
)


def _write_manifest(tmp_path, *, duplicate=False, missing=False):
    rows = []
    first_path = None
    for class_name in CLASSES:
        for index in range(EXPECTED_COUNTS[class_name]):
            path = tmp_path / class_name / f"image_{index}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"test")
            if first_path is None:
                first_path = path
            rows.append({"source_path": str(path), "class_name": class_name, "split": "test"})
    rows.extend([
        {"source_path": str(tmp_path / "ignored_train.jpg"), "class_name": "Fresh", "split": "train"},
        {"source_path": str(tmp_path / "ignored_validation.jpg"), "class_name": "Fresh", "split": "validation"},
    ])
    if duplicate:
        rows[1]["source_path"] = rows[0]["source_path"]
    if missing:
        rows[0]["source_path"] = str(tmp_path / "does_not_exist.jpg")
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_path", "class_name", "split"])
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def test_manifest_gate_accepts_exact_test_split_and_ignores_other_splits(tmp_path):
    rows, validation = read_and_validate_test_rows(_write_manifest(tmp_path))
    assert len(rows) == 147
    assert validation["class_counts"] == EXPECTED_COUNTS
    assert validation["test_rows_only"] is True
    assert validation["train_or_validation_rows_processed"] is False


def test_manifest_gate_rejects_duplicate_test_source(tmp_path):
    with pytest.raises(ValueError, match="Duplicate test source path"):
        read_and_validate_test_rows(_write_manifest(tmp_path, duplicate=True))


def test_manifest_gate_rejects_missing_source(tmp_path):
    with pytest.raises(FileNotFoundError, match="source files are missing"):
        read_and_validate_test_rows(_write_manifest(tmp_path, missing=True))


def test_metric_rows_report_perfect_predictions_without_matlab():
    expected = ["Fresh", "Unripe", "Rotten"]
    rows, report = metric_rows(expected, expected)
    assert report["accuracy"] == 1.0
    assert [row["class_name"] for row in rows[:3]] == list(CLASSES)
    assert all(row["f1_score"] == 1.0 for row in rows)


def _successful_record(path, expected, predicted, offset=0.0):
    return {
        "source_path": str(path), "expected_class": expected,
        "predicted_class": predicted, "correct": expected == predicted,
        "grade": "Grade A", "freshness_score": 80.0,
        "green_percentage": 40.0 + offset, "yellow_percentage": 50.0,
        "brown_percentage": 2.0, "dark_percentage": 4.0,
        "white_mold_percentage": 0.0, "lesion_percentage": 3.0,
        "rough_percentage": 4.0, "damage_percentage": 8.0,
        "healthy_percentage": 92.0, "largest_damage_percentage": 3.0,
        "largest_lesion_percentage": 3.0, "unripe_rule_score": 10.0,
        "fresh_rule_score": 20.0, "rotten_rule_score": 5.0,
        "processing_time_seconds": 0.5, "error": "",
    }


def test_rule_activation_reconstructs_dark_override_without_changing_classifier():
    record = _successful_record("unused.jpg", "Fresh", "Rotten")
    record.update({
        "green_percentage": 88.85, "yellow_percentage": 76.08,
        "dark_percentage": 14.06, "damage_percentage": 13.71,
        "largest_damage_percentage": 4.79, "largest_lesion_percentage": 4.79,
        "unripe_rule_score": 33.92, "fresh_rule_score": 41.67,
        "rotten_rule_score": 33.38,
    })
    activations = infer_rule_activations(record)
    assert activations["Condition: dark >= 8"] is True
    assert activations["Strong Rotten gate selected"] is True
    assert activations["Score fallback reached"] is False
    assert activations["Clean yellow rescue changed Rotten to Fresh"] is False


def test_report_artifacts_are_generated_from_real_records(tmp_path):
    images = []
    for index in range(3):
        path = tmp_path / f"image_{index}.jpg"
        Image.new("RGB", (80, 60), (30 + index * 40, 130, 60)).save(path)
        images.append(path)
    successful = [
        _successful_record(images[0], "Fresh", "Rotten", 0),
        _successful_record(images[1], "Unripe", "Unripe", 1),
        _successful_record(images[2], "Rotten", "Rotten", 2),
    ]
    y_true = [record["expected_class"] for record in successful]
    y_pred = [record["predicted_class"] for record in successful]
    report = classification_report(
        y_true, y_pred, labels=list(CLASSES), output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=list(CLASSES))
    normalized = matrix / matrix.sum(axis=1, keepdims=True)
    summary = {
        "attempted_images": 3, "successful_images": 3, "failed_image_count": 0,
        "total_processing_time_seconds": 1.5,
        "average_processing_time_seconds": 0.5,
        "evaluation_wall_time_seconds": 2.0,
    }
    validation = {
        "manifest_path": "manifest.csv",
        "class_counts": {"Fresh": 1, "Unripe": 1, "Rotten": 1},
    }
    paths = generate_report_artifacts(
        output_dir=tmp_path / "outputs", successful=successful,
        misclassified=[successful[0]], summary=summary, validation=validation,
        report=report, matrix=matrix, normalized=normalized, classes=CLASSES,
    )
    expected_names = {
        "confusion_matrix.png", "normalized_confusion_matrix.png",
        "per_class_precision.png", "per_class_recall.png", "per_class_f1.png",
        "class_distribution.png", "prediction_distribution.png",
        "processing_time_histogram.png", "metrics_summary.csv",
        "misclassification_summary.csv", "feature_statistics.csv",
        "rule_activation_summary.csv", "misclassified_images_contact_sheet.png",
        "matlab_rule_based_report.md",
    }
    assert {path.name for path in paths} == expected_names
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)
