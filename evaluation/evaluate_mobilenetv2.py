"""Evaluate the trained MobileNetV2 baseline on the held-out test split only."""

from __future__ import annotations

import csv
import json
import math
import multiprocessing
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from torch.torch_version import TorchVersion
from torch.utils.data import DataLoader

from ai.dataset import CLASS_NAMES, ManifestImageDataset, load_manifest
from ai.models.model_factory import create_model
from ai.train_model import build_transforms, load_config, resolve_project_path, select_device


OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "outputs" / "mobilenetv2_cleaned_baseline"
MATLAB_SUMMARY = (
    PROJECT_ROOT / "evaluation" / "outputs" / "matlab_rule_based"
    / "matlab_evaluation_summary.json"
)
EXPECTED_TEST_COUNTS = {"Fresh": 42, "Unripe": 53, "Rotten": 52}
EXPECTED_TEST_TOTAL = 147


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def validate_evaluation_inputs(config: dict) -> tuple[object, object, dict]:
    manifest_path = resolve_project_path(config["manifest_path"])
    checkpoint_path = resolve_project_path(config["best_checkpoint_path"])
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dataset manifest is missing: {manifest_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Best checkpoint is missing: {checkpoint_path}")

    inspection = PROJECT_ROOT / "evaluation" / "outputs" / "dataset_inspection.json"
    if inspection.is_file() and inspection.stat().st_mtime > manifest_path.stat().st_mtime:
        raise RuntimeError("Dataset manifest is older than the latest inspection report.")

    manifest = load_manifest(manifest_path, validate=True)
    test_rows = manifest[manifest["split"] == "test"].reset_index(drop=True)
    train_rows = manifest[manifest["split"] == "train"]
    counts = test_rows["class_name"].value_counts().to_dict()
    counts = {name: int(counts.get(name, 0)) for name in CLASS_NAMES}
    if len(test_rows) != EXPECTED_TEST_TOTAL or counts != EXPECTED_TEST_COUNTS:
        raise ValueError(
            f"Held-out test split mismatch: total={len(test_rows)}, counts={counts}."
        )
    overlap = sorted(set(train_rows["sha256"]) & set(test_rows["sha256"]))
    if overlap:
        raise RuntimeError(f"Train/test SHA-256 overlap detected: {len(overlap)} hashes.")
    if test_rows["source_path"].map(lambda value: str(value).casefold()).duplicated().any():
        raise RuntimeError("A test source path occurs more than once.")
    missing = [path for path in test_rows["source_path"] if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing test images: {len(missing)}")

    validation = {
        "manifest_path": str(manifest_path),
        "checkpoint_path": str(checkpoint_path),
        "test_rows_only": True,
        "test_count": len(test_rows),
        "test_class_counts": counts,
        "train_test_hash_overlap_count": 0,
        "unique_test_source_paths": True,
        "all_test_sources_exist": True,
    }
    return manifest, test_rows, validation


def load_best_model(config: dict, device: torch.device) -> tuple[torch.nn.Module, dict]:
    checkpoint_path = resolve_project_path(config["best_checkpoint_path"])
    with torch.serialization.safe_globals([TorchVersion]):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint.get("classes") != CLASS_NAMES:
        raise ValueError(
            f"Checkpoint class order {checkpoint.get('classes')} does not match {CLASS_NAMES}."
        )
    model = create_model(
        num_classes=len(CLASS_NAMES),
        dropout=config["dropout"],
        pretrained=False,
        freeze_backbone=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, checkpoint


def calculate_metrics(y_true: list[int], y_pred: list[int], probabilities: np.ndarray) -> dict:
    labels = list(range(len(CLASS_NAMES)))
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=CLASS_NAMES,
        output_dict=True, zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix.astype(float), row_totals,
        out=np.full(matrix.shape, np.nan), where=row_totals != 0,
    )
    binary = label_binarize(y_true, classes=labels)
    roc = {}
    for index, class_name in enumerate(CLASS_NAMES):
        false_positive_rate, true_positive_rate, thresholds = roc_curve(
            binary[:, index], probabilities[:, index]
        )
        roc[class_name] = {
            "fpr": false_positive_rate,
            "tpr": true_positive_rate,
            "thresholds": thresholds,
            "auc": float(auc(false_positive_rate, true_positive_rate)),
        }
    return {"report": report, "matrix": matrix, "normalized": normalized, "roc": roc}


def save_matrix(matrix: np.ndarray, path: Path, normalized: bool) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 6.2))
    display = np.nan_to_num(matrix.astype(float), nan=0.0)
    image = axis.imshow(display, cmap="Blues", interpolation="nearest", vmin=0)
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(CLASS_NAMES)), yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        xlabel="Predicted class", ylabel="Expected class",
        title="Normalized Confusion Matrix" if normalized else "Confusion Matrix",
    )
    threshold = float(np.nanmax(display)) / 2 if display.size else 0
    for row in range(len(CLASS_NAMES)):
        for column in range(len(CLASS_NAMES)):
            value = matrix[row, column]
            label = "N/A" if np.isnan(value) else (
                f"{value:.1%}" if normalized else str(int(value))
            )
            axis.text(
                column, row, label, ha="center", va="center", fontweight="bold",
                color="white" if not np.isnan(value) and value > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_roc_curves(roc: dict, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 6.2))
    colors = {"Fresh": "#2E8B57", "Unripe": "#E5A823", "Rotten": "#B7472A"}
    for class_name in CLASS_NAMES:
        values = roc[class_name]
        axis.plot(
            values["fpr"], values["tpr"], linewidth=2.2,
            color=colors[class_name], label=f"{class_name} (AUC={values['auc']:.4f})",
        )
    axis.plot([0, 1], [0, 1], "--", color="#666666", label="Chance")
    axis.set(xlabel="False positive rate", ylabel="True positive rate",
             title="One-vs-Rest ROC Curves", xlim=(0, 1), ylim=(0, 1.02))
    axis.legend(loc="lower right")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_comparison_figures(ai_summary: dict, matlab_summary: dict) -> None:
    models = ("MATLAB rules", "MobileNetV2")
    comparisons = (
        ("accuracy_comparison.png", "Overall Accuracy", "Score",
         (matlab_summary["overall_accuracy"], ai_summary["overall_accuracy"])),
        ("macro_f1_comparison.png", "Macro F1-Score", "Score",
         (matlab_summary["macro_f1_score"], ai_summary["macro_f1_score"])),
        ("processing_time_comparison.png", "Average Processing Time", "Seconds per image",
         (matlab_summary["average_processing_time_seconds"],
          ai_summary["average_inference_time_seconds"])),
    )
    for filename, title, ylabel, values in comparisons:
        figure, axis = plt.subplots(figsize=(7.2, 5.2))
        bars = axis.bar(models, values, color=("#6B7280", "#2563EB"))
        axis.set_title(title, fontweight="bold")
        axis.set_ylabel(ylabel)
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, value,
                      f"{value:.4f}", ha="center", va="bottom", fontweight="bold")
        figure.tight_layout()
        figure.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
        plt.close(figure)

    matlab_f1 = [matlab_summary["per_class"][name]["f1-score"] for name in CLASS_NAMES]
    ai_f1 = [ai_summary["per_class"][name]["f1-score"] for name in CLASS_NAMES]
    positions = np.arange(len(CLASS_NAMES))
    width = 0.36
    figure, axis = plt.subplots(figsize=(8.2, 5.4))
    axis.bar(positions - width / 2, matlab_f1, width, label="MATLAB rules", color="#6B7280")
    axis.bar(positions + width / 2, ai_f1, width, label="MobileNetV2", color="#2563EB")
    axis.set(xticks=positions, xticklabels=CLASS_NAMES, ylabel="F1-score",
             title="Per-Class F1-Score Comparison", ylim=(0, 1.05))
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "per_class_f1_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_model_comparison(ai_summary: dict, matlab_summary: dict) -> None:
    rows = [
        {
            "model": "MATLAB rule-based",
            "overall_accuracy": matlab_summary["overall_accuracy"],
            "macro_f1": matlab_summary["macro_f1_score"],
            "weighted_f1": matlab_summary["weighted_f1_score"],
            "average_processing_time_seconds": matlab_summary["average_processing_time_seconds"],
        },
        {
            "model": "MobileNetV2 cleaned baseline",
            "overall_accuracy": ai_summary["overall_accuracy"],
            "macro_f1": ai_summary["macro_f1_score"],
            "weighted_f1": ai_summary["weighted_f1_score"],
            "average_processing_time_seconds": ai_summary["average_inference_time_seconds"],
        },
    ]
    columns = list(rows[0])
    write_csv(OUTPUT_DIR / "model_comparison.csv", rows, columns)
    markdown = [
        "# MATLAB and MobileNetV2 Baseline Comparison", "",
        "| Model | Accuracy | Macro F1 | Weighted F1 | Average processing time (s/image) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        markdown.append(
            f"| {row['model']} | {row['overall_accuracy']:.4f} | {row['macro_f1']:.4f} | "
            f"{row['weighted_f1']:.4f} | {row['average_processing_time_seconds']:.4f} |"
        )
    markdown.extend([
        "", "MATLAB time is its reported internal processing time. MobileNetV2 time is "
        "synchronized CUDA forward-pass time; data loading and report generation are excluded.", "",
    ])
    (OUTPUT_DIR / "model_comparison.md").write_text("\n".join(markdown), encoding="utf-8")
    save_comparison_figures(ai_summary, matlab_summary)


def evaluate() -> int:
    config = load_config()
    if config["classes"] != CLASS_NAMES:
        raise ValueError(f"Class order must be exactly {CLASS_NAMES}.")
    device = select_device(config["device"])
    if device.type != "cuda":
        raise RuntimeError("The cleaned baseline evaluation requires CUDA and will not use CPU.")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    manifest, test_rows, validation = validate_evaluation_inputs(config)
    model, checkpoint = load_best_model(config, device)
    transform = build_transforms(config)["test"]
    dataset = ManifestImageDataset(config["manifest_path"], split="test", transform=transform)
    loader = DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=False,
        num_workers=config["num_workers"], pin_memory=True,
        persistent_workers=config["num_workers"] > 0,
    )

    print("=== FreshSight MobileNetV2 Held-Out Evaluation ===")
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Best checkpoint: {validation['checkpoint_path']}")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Class order: {', '.join(f'{i}:{name}' for i, name in enumerate(CLASS_NAMES))}")
    print(f"Held-out test images: {len(test_rows)}")
    print("Train/test SHA-256 overlap: 0")
    print("Test transforms: deterministic Resize -> CenterCrop -> Normalize")

    y_true: list[int] = []
    y_pred: list[int] = []
    probability_batches = []
    per_image_times = []
    amp_enabled = bool(config["mixed_precision"])
    with torch.inference_mode():
        for batch_index, (images, labels) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
                logits = model(images)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            probabilities = torch.softmax(logits.float(), dim=1)
            predictions = probabilities.argmax(dim=1)
            batch_size = labels.size(0)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())
            probability_batches.append(probabilities.cpu().numpy())
            per_image_times.extend([elapsed / batch_size] * batch_size)
            print(f"Batch {batch_index}/{len(loader)} | processed={len(y_true)}/{len(dataset)}")

    probabilities = np.concatenate(probability_batches, axis=0)
    metrics = calculate_metrics(y_true, y_pred, probabilities)
    report = metrics["report"]
    predictions = []
    for index, row in test_rows.iterrows():
        expected = CLASS_NAMES[y_true[index]]
        predicted = CLASS_NAMES[y_pred[index]]
        predictions.append({
            "source_path": row["source_path"],
            "sha256": row["sha256"],
            "expected_class": expected,
            "predicted_class": predicted,
            "correct": expected == predicted,
            "confidence": float(probabilities[index].max()),
            "probability_Fresh": float(probabilities[index, 0]),
            "probability_Unripe": float(probabilities[index, 1]),
            "probability_Rotten": float(probabilities[index, 2]),
            "inference_time_seconds": per_image_times[index],
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prediction_columns = list(predictions[0])
    write_csv(OUTPUT_DIR / "predictions.csv", predictions, prediction_columns)
    write_csv(
        OUTPUT_DIR / "misclassified_images.csv",
        [row for row in predictions if not row["correct"]], prediction_columns,
    )
    report_rows = []
    for name in (*CLASS_NAMES, "macro avg", "weighted avg"):
        values = report[name]
        report_rows.append({
            "class_name": name, "precision": values["precision"],
            "recall": values["recall"], "f1_score": values["f1-score"],
            "support": int(values["support"]),
        })
    write_csv(
        OUTPUT_DIR / "classification_report.csv", report_rows,
        ["class_name", "precision", "recall", "f1_score", "support"],
    )
    write_csv(
        OUTPUT_DIR / "roc_auc.csv",
        [{"class_name": name, "auc": metrics["roc"][name]["auc"]} for name in CLASS_NAMES],
        ["class_name", "auc"],
    )
    save_matrix(metrics["matrix"], OUTPUT_DIR / "confusion_matrix.png", False)
    save_matrix(metrics["normalized"], OUTPUT_DIR / "normalized_confusion_matrix.png", True)
    save_roc_curves(metrics["roc"], OUTPUT_DIR / "roc_curves.png")

    summary = {
        "status": "success",
        "model": "MobileNetV2 cleaned baseline",
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "class_order": CLASS_NAMES,
        "validation": validation,
        "test_images": len(y_true),
        "correct_images": int(sum(a == b for a, b in zip(y_true, y_pred))),
        "misclassified_images": int(sum(a != b for a, b in zip(y_true, y_pred))),
        "overall_accuracy": float(report["accuracy"]),
        "per_class": {name: report[name] for name in CLASS_NAMES},
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1_score": report["macro avg"]["f1-score"],
        "weighted_precision": report["weighted avg"]["precision"],
        "weighted_recall": report["weighted avg"]["recall"],
        "weighted_f1_score": report["weighted avg"]["f1-score"],
        "confusion_matrix": metrics["matrix"].tolist(),
        "normalized_confusion_matrix": [
            [None if math.isnan(value) else float(value) for value in row]
            for row in metrics["normalized"]
        ],
        "roc_auc": {name: metrics["roc"][name]["auc"] for name in CLASS_NAMES},
        "total_inference_time_seconds": float(sum(per_image_times)),
        "average_inference_time_seconds": float(np.mean(per_image_times)),
        "timing_scope": "synchronized CUDA forward pass; excludes data loading and reporting",
    }
    (OUTPUT_DIR / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )

    if not MATLAB_SUMMARY.is_file():
        raise FileNotFoundError(f"MATLAB summary is missing: {MATLAB_SUMMARY}")
    matlab_summary = json.loads(MATLAB_SUMMARY.read_text(encoding="utf-8"))
    save_model_comparison(summary, matlab_summary)

    print("=== AI Evaluation Metrics ===")
    print(f"Accuracy: {summary['overall_accuracy']:.4f}")
    print(f"Macro F1: {summary['macro_f1_score']:.4f}")
    print(f"Weighted F1: {summary['weighted_f1_score']:.4f}")
    for name in CLASS_NAMES:
        values = summary["per_class"][name]
        print(
            f"{name}: precision={values['precision']:.4f} recall={values['recall']:.4f} "
            f"f1={values['f1-score']:.4f} support={int(values['support'])} "
            f"auc={summary['roc_auc'][name]:.4f}"
        )
    print(f"Average inference time: {summary['average_inference_time_seconds']:.6f} seconds/image")
    print("=== Outputs ===")
    for path in sorted(OUTPUT_DIR.iterdir()):
        if path.is_file():
            print(path)
    print("=== Result ===")
    print(f"PASSED: Evaluated {len(y_true)} held-out test images without deployment.")
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(evaluate())
