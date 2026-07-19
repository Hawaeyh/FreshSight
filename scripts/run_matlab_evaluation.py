"""Evaluate MATLAB rules on only the cleaned manifest's held-out test split."""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.matlab_evaluation_reporting import generate_report_artifacts

MATLAB_ROOT = PROJECT_ROOT / "matlab"
MANIFEST_PATH = PROJECT_ROOT / "evaluation" / "outputs" / "dataset_manifest.csv"
OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "outputs" / "matlab_rule_based"
CLASSES = ("Fresh", "Unripe", "Rotten")
EXPECTED_COUNTS = {"Fresh": 42, "Unripe": 53, "Rotten": 52}
EXPECTED_TOTAL = 147
REQUIRED_MANIFEST_COLUMNS = {"source_path", "class_name", "split"}
PREDICTION_COLUMNS = (
    "source_path", "expected_class", "predicted_class", "correct", "grade",
    "freshness_score", "green_percentage", "yellow_percentage",
    "brown_percentage", "dark_percentage", "white_mold_percentage",
    "lesion_percentage", "rough_percentage", "damage_percentage",
    "healthy_percentage", "largest_damage_percentage",
    "largest_lesion_percentage", "unripe_rule_score", "fresh_rule_score",
    "rotten_rule_score", "processing_time_seconds", "error",
)


def read_and_validate_test_rows(manifest_path: Path = MANIFEST_PATH) -> tuple[list[dict], dict]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dataset manifest does not exist: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_MANIFEST_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
        all_rows = list(reader)

    test_rows = [row for row in all_rows if row["split"].strip().lower() == "test"]
    if any(row["split"].strip().lower() != "test" for row in test_rows):
        raise AssertionError("A non-test row passed the test-row filter.")
    if len(test_rows) != EXPECTED_TOTAL:
        raise ValueError(f"Expected exactly {EXPECTED_TOTAL} test rows, found {len(test_rows)}.")

    unsupported = sorted({row["class_name"].strip() for row in test_rows} - set(CLASSES))
    if unsupported:
        raise ValueError(f"Unsupported test classes found: {unsupported}")

    counts = {name: 0 for name in CLASSES}
    normalized_rows = []
    seen_paths = set()
    missing_paths = []
    for row in test_rows:
        class_name = row["class_name"].strip()
        source = Path(row["source_path"]).expanduser().resolve()
        source_key = str(source).casefold()
        if source_key in seen_paths:
            raise ValueError(f"Duplicate test source path: {source}")
        seen_paths.add(source_key)
        if not source.is_file():
            missing_paths.append(str(source))
        counts[class_name] += 1
        normalized_rows.append({"source_path": str(source), "expected_class": class_name})

    if missing_paths:
        preview = "\n".join(missing_paths[:10])
        raise FileNotFoundError(
            f"{len(missing_paths)} manifest source files are missing. First entries:\n{preview}"
        )
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"Unexpected test class counts: {counts}; expected {EXPECTED_COUNTS}.")

    validation = {
        "manifest_path": str(manifest_path.resolve()),
        "test_rows_only": True,
        "train_or_validation_rows_processed": False,
        "supported_classes_only": True,
        "all_source_paths_exist": True,
        "unique_source_paths": True,
        "expected_total": EXPECTED_TOTAL,
        "actual_total": len(normalized_rows),
        "class_counts": counts,
    }
    return normalized_rows, validation


def empty_record(source_path: str, expected_class: str) -> dict:
    record = {column: "" for column in PREDICTION_COLUMNS}
    record["source_path"] = source_path
    record["expected_class"] = expected_class
    return record


def write_csv(path: Path, rows: Iterable[dict], columns: Iterable[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def metric_rows(y_true: list[str], y_pred: list[str]) -> tuple[list[dict], dict]:
    report = classification_report(
        y_true, y_pred, labels=list(CLASSES), output_dict=True, zero_division=0
    )
    rows = []
    for name in (*CLASSES, "macro avg", "weighted avg"):
        values = report[name]
        rows.append({
            "class_name": name,
            "precision": values["precision"],
            "recall": values["recall"],
            "f1_score": values["f1-score"],
            "support": int(values["support"]),
        })
    return rows, report


def save_matrix_plot(matrix: np.ndarray, path: Path, title: str, normalized: bool) -> None:
    figure, axis = plt.subplots(figsize=(7, 6))
    display_matrix = np.nan_to_num(matrix, nan=0.0)
    image = axis.imshow(display_matrix, interpolation="nearest", cmap="Blues", vmin=0)
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(CLASSES)), yticks=np.arange(len(CLASSES)),
        xticklabels=CLASSES, yticklabels=CLASSES,
        xlabel="Predicted class", ylabel="Expected class", title=title,
    )
    threshold = float(np.nanmax(display_matrix)) / 2 if display_matrix.size else 0
    for row in range(len(CLASSES)):
        for column in range(len(CLASSES)):
            value = matrix[row, column]
            label = "N/A" if np.isnan(value) else (f"{value:.2%}" if normalized else str(int(value)))
            axis.text(column, row, label, ha="center", va="center",
                      color="white" if not np.isnan(value) and value > threshold else "black")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def evaluate() -> int:
    print("=== FreshSight MATLAB Rule-Based Evaluation ===")
    print(f"Manifest: {MANIFEST_PATH}")
    print("Split restriction: test only")
    print("Source handling: read-only; no source images will be modified.")

    try:
        rows, validation = read_and_validate_test_rows()
    except Exception as exc:
        print(f"ERROR: Manifest validation failed: {exc}", file=sys.stderr)
        return 1

    print("\n=== Manifest validation ===")
    for key, value in validation.items():
        print(f"{key}: {value}")

    try:
        import matlab.engine
    except Exception as exc:
        print(f"ERROR: MATLAB Engine import failed: {exc}", file=sys.stderr)
        return 1

    engine = None
    records = []
    evaluation_started = time.perf_counter()
    try:
        print("\n=== MATLAB Engine ===")
        engine = matlab.engine.start_matlab()
        print("MATLAB Engine start: PASS")
        engine.addpath(engine.genpath(str(MATLAB_ROOT)), nargout=0)
        api_path = str(engine.which("evaluate_rule_based_pipeline", nargout=1)).strip()
        if not api_path:
            raise RuntimeError("evaluate_rule_based_pipeline was not found on the MATLAB path.")
        print(f"Evaluation function: {api_path}")
        print("Engine session reuse: one session for all images")

        print("\n=== Image processing ===")
        for index, row in enumerate(rows, start=1):
            print(f"Processing {index}/{EXPECTED_TOTAL}: {row['source_path']}", flush=True)
            record = empty_record(row["source_path"], row["expected_class"])
            try:
                raw = engine.evaluate_rule_based_pipeline(row["source_path"], nargout=1)
                result = json.loads(str(raw))
                if result.get("status") != "success" or result.get("error"):
                    raise RuntimeError(result.get("error") or f"MATLAB status: {result.get('status')}")
                for column in PREDICTION_COLUMNS:
                    if column in result:
                        record[column] = result[column]
                record["predicted_class"] = result["predicted_class"]
                record["correct"] = result["predicted_class"] == row["expected_class"]
                record["error"] = ""
            except Exception as exc:
                record["error"] = str(exc)
                print(f"  FAILED: {exc}", file=sys.stderr, flush=True)
            records.append(record)
    except Exception as exc:
        print(f"ERROR: MATLAB evaluation could not continue: {exc}", file=sys.stderr)
        return 1
    finally:
        if engine is not None:
            try:
                engine.quit()
                print("\nMATLAB Engine closed cleanly.")
            except Exception as exc:
                print(f"WARNING: MATLAB Engine close failed: {exc}", file=sys.stderr)

    evaluation_wall_time = time.perf_counter() - evaluation_started
    successful = [record for record in records if not record["error"]]
    failed = [record for record in records if record["error"]]
    misclassified = [record for record in successful if record["correct"] is False]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIR / "matlab_predictions.csv", records, PREDICTION_COLUMNS)
    write_csv(OUTPUT_DIR / "matlab_successful_predictions.csv", successful, PREDICTION_COLUMNS)
    write_csv(OUTPUT_DIR / "matlab_failed_images.csv", failed, PREDICTION_COLUMNS)
    write_csv(OUTPUT_DIR / "matlab_misclassified_images.csv", misclassified, PREDICTION_COLUMNS)

    print("\n=== Evaluation metrics ===")
    summary = {
        "status": "success" if not failed else "partial",
        "classes": list(CLASSES),
        "validation": validation,
        "attempted_images": len(records),
        "successful_images": len(successful),
        "failed_image_count": len(failed),
        "misclassified_image_count": len(misclassified),
        "evaluation_wall_time_seconds": evaluation_wall_time,
    }

    report_columns = ("class_name", "precision", "recall", "f1_score", "support")
    if successful:
        y_true = [record["expected_class"] for record in successful]
        y_pred = [record["predicted_class"] for record in successful]
        metrics_rows, report = metric_rows(y_true, y_pred)
        matrix = confusion_matrix(y_true, y_pred, labels=list(CLASSES))
        row_totals = matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(
            matrix.astype(float), row_totals, out=np.full(matrix.shape, np.nan), where=row_totals != 0
        )
        processing_times = [float(record["processing_time_seconds"]) for record in successful]
        summary.update({
            "metrics_scope": "successful images only",
            "overall_accuracy": float(report["accuracy"]),
            "per_class": {name: report[name] for name in CLASSES},
            "macro_precision": report["macro avg"]["precision"],
            "macro_recall": report["macro avg"]["recall"],
            "macro_f1_score": report["macro avg"]["f1-score"],
            "weighted_precision": report["weighted avg"]["precision"],
            "weighted_recall": report["weighted avg"]["recall"],
            "weighted_f1_score": report["weighted avg"]["f1-score"],
            "confusion_matrix": matrix.tolist(),
            "normalized_confusion_matrix": [
                [None if math.isnan(value) else float(value) for value in row]
                for row in normalized
            ],
            "total_processing_time_seconds": sum(processing_times),
            "average_processing_time_seconds": sum(processing_times) / len(processing_times),
        })
        write_csv(OUTPUT_DIR / "matlab_classification_report.csv", metrics_rows, report_columns)
        save_matrix_plot(matrix, OUTPUT_DIR / "matlab_confusion_matrix.png",
                         "MATLAB Rule-Based Confusion Matrix", False)
        save_matrix_plot(normalized, OUTPUT_DIR / "matlab_normalized_confusion_matrix.png",
                         "MATLAB Rule-Based Normalized Confusion Matrix", True)
        print(f"Overall accuracy: {summary['overall_accuracy']:.4f}")
        print(f"Macro F1-score: {summary['macro_f1_score']:.4f}")
        print(f"Weighted F1-score: {summary['weighted_f1_score']:.4f}")
    else:
        summary.update({
            "metrics_scope": "unavailable because every image failed",
            "overall_accuracy": None,
            "per_class": None,
            "macro_precision": None, "macro_recall": None, "macro_f1_score": None,
            "weighted_precision": None, "weighted_recall": None,
            "weighted_f1_score": None, "confusion_matrix": None,
            "normalized_confusion_matrix": None,
            "total_processing_time_seconds": None,
            "average_processing_time_seconds": None,
        })
        write_csv(OUTPUT_DIR / "matlab_classification_report.csv", [], report_columns)
        print("Metrics unavailable: every image failed; no values were fabricated.")

    summary_path = OUTPUT_DIR / "matlab_evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")

    if successful:
        print("\n=== Report-ready artifacts ===")
        try:
            report_paths = generate_report_artifacts(
                output_dir=OUTPUT_DIR,
                successful=successful,
                misclassified=misclassified,
                summary=summary,
                validation=validation,
                report=report,
                matrix=matrix,
                normalized=normalized,
                classes=CLASSES,
            )
            for path in report_paths:
                print(path)
        except Exception as exc:
            print(f"ERROR: Report artifact generation failed: {exc}", file=sys.stderr)
            return 1
    else:
        print("\nReport-ready metrics and figures were not generated because every image failed.")
    print(f"Successful images: {len(successful)}")
    print(f"Failed images: {len(failed)}")
    print(f"Misclassified images: {len(misclassified)}")
    print(f"Evaluation wall time: {evaluation_wall_time:.3f} seconds")

    print("\n=== Outputs ===")
    for path in sorted(OUTPUT_DIR.rglob("*")):
        if path.is_file():
            print(path)
    print("\n=== Result ===")
    if failed:
        print("PARTIAL: Evaluation completed with failed images; metrics cover successful images only.")
        return 2
    print(f"PASSED: Evaluated all {EXPECTED_TOTAL} held-out test images.")
    return 0


if __name__ == "__main__":
    raise SystemExit(evaluate())
