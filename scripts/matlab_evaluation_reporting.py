"""Publication-ready artifacts for the MATLAB rule-based evaluation."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


FEATURES = (
    "green_percentage", "yellow_percentage", "brown_percentage",
    "dark_percentage", "white_mold_percentage", "lesion_percentage",
    "damage_percentage", "healthy_percentage",
)
COLORS = ("#2E8B57", "#E5A823", "#B7472A")


def _float(record: dict, key: str) -> float:
    return float(record[key])


def _write_csv(path: Path, rows: Iterable[dict], columns: Iterable[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _prepare_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("default")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def _save_bar(values: list[float], labels: tuple[str, ...], path: Path,
              title: str, ylabel: str, percent: bool = False) -> None:
    figure, axis = plt.subplots(figsize=(8, 5.2))
    bars = axis.bar(labels, values, color=COLORS, edgecolor="white", linewidth=0.8)
    axis.set_title(title, fontweight="bold", pad=12)
    axis.set_ylabel(ylabel)
    if percent:
        axis.set_ylim(0, 1.05)
    upper = max(values, default=0)
    padding = max(upper * 0.025, 0.015 if percent else 0.5)
    for bar, value in zip(bars, values):
        label = f"{value:.3f}" if percent else f"{int(value)}"
        axis.text(bar.get_x() + bar.get_width() / 2, value + padding, label,
                  ha="center", va="bottom", fontweight="bold")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _save_confusion(matrix: np.ndarray, path: Path, classes: tuple[str, ...],
                    normalized: bool) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 6.2))
    display = np.nan_to_num(matrix.astype(float), nan=0.0)
    image = axis.imshow(display, cmap="Blues", interpolation="nearest", vmin=0)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set(
        xticks=np.arange(len(classes)), yticks=np.arange(len(classes)),
        xticklabels=classes, yticklabels=classes,
        xlabel="Predicted class", ylabel="Expected class",
        title=("Normalized Confusion Matrix" if normalized else "Confusion Matrix"),
    )
    threshold = float(np.nanmax(display)) / 2 if display.size else 0
    for row in range(len(classes)):
        for column in range(len(classes)):
            value = matrix[row, column]
            label = "N/A" if np.isnan(value) else (f"{value:.1%}" if normalized else str(int(value)))
            axis.text(column, row, label, ha="center", va="center", fontweight="bold",
                      color="white" if not np.isnan(value) and value > threshold else "black")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def _metrics_summary(report: dict, average_time: float, classes: tuple[str, ...]) -> list[dict]:
    rows = []
    for class_name in classes:
        values = report[class_name]
        rows.append({
            "Section": "Per-class metrics", "Class": class_name, "Metric": "",
            "Value": "", "Support": int(values["support"]),
            "Precision": values["precision"], "Recall": values["recall"],
            "F1-score": values["f1-score"], "Unit": "proportion",
        })
    summary_values = (
        ("Overall Accuracy", report["accuracy"], "proportion"),
        ("Macro Precision", report["macro avg"]["precision"], "proportion"),
        ("Macro Recall", report["macro avg"]["recall"], "proportion"),
        ("Macro F1", report["macro avg"]["f1-score"], "proportion"),
        ("Weighted Precision", report["weighted avg"]["precision"], "proportion"),
        ("Weighted Recall", report["weighted avg"]["recall"], "proportion"),
        ("Weighted F1", report["weighted avg"]["f1-score"], "proportion"),
        ("Average Processing Time", average_time, "seconds per image"),
    )
    for metric, value, unit in summary_values:
        rows.append({
            "Section": "Overall metrics", "Class": "", "Metric": metric,
            "Value": value, "Support": "", "Precision": "", "Recall": "",
            "F1-score": "", "Unit": unit,
        })
    return rows


def _misclassification_rows(successful: list[dict], classes: tuple[str, ...]) -> list[dict]:
    support = Counter(record["expected_class"] for record in successful)
    counts = Counter(
        (record["expected_class"], record["predicted_class"])
        for record in successful if record["expected_class"] != record["predicted_class"]
    )
    rows = []
    for expected in classes:
        for predicted in classes:
            if expected == predicted:
                continue
            count = counts[(expected, predicted)]
            denominator = support[expected]
            rows.append({
                "Expected Class": expected,
                "Predicted Class": predicted,
                "Count": count,
                "Percentage": (count / denominator * 100) if denominator else "",
            })
    return rows


def _feature_rows(successful: list[dict], classes: tuple[str, ...]) -> list[dict]:
    rows = []
    for class_name in classes:
        class_records = [r for r in successful if r["expected_class"] == class_name]
        for feature in FEATURES:
            values = np.array([_float(record, feature) for record in class_records], dtype=float)
            rows.append({
                "Class": class_name,
                "Feature": feature,
                "Count": int(values.size),
                "Average": float(np.mean(values)) if values.size else "",
                "Minimum": float(np.min(values)) if values.size else "",
                "Maximum": float(np.max(values)) if values.size else "",
                "Standard Deviation": float(np.std(values, ddof=1)) if values.size > 1 else "",
            })
    return rows


def infer_rule_activations(record: dict) -> dict[str, bool]:
    """Reconstruct classifier conditions without changing or calling its thresholds."""
    green = _float(record, "green_percentage")
    yellow = _float(record, "yellow_percentage")
    dark = _float(record, "dark_percentage")
    white = _float(record, "white_mold_percentage")
    damage = _float(record, "damage_percentage")
    largest_damage = _float(record, "largest_damage_percentage")
    largest_lesion = _float(record, "largest_lesion_percentage")

    unripe_gate = green >= 70 and yellow < 25 and damage < 10
    strong_dark = dark >= 8
    strong_white = white >= 3
    strong_damage = damage >= 25
    strong_lesion = largest_lesion >= 12
    strong_largest_damage = largest_damage >= 15
    strong_gate = (not unripe_gate) and any((
        strong_dark, strong_white, strong_damage, strong_lesion, strong_largest_damage
    ))
    yellow_branch = (not unripe_gate) and (not strong_gate) and yellow >= 70 and green < 25
    yellow_direct = yellow_branch and dark >= 5.5
    yellow_compound = (
        yellow_branch and not yellow_direct and dark >= 3.5
        and any((largest_lesion >= 6, largest_damage >= 10, white >= 1.5, damage >= 20))
    )
    score_fallback = not unripe_gate and not strong_gate and not yellow_branch

    if unripe_gate:
        pre_safety_status = "Unripe"
    elif strong_gate or yellow_direct or yellow_compound:
        pre_safety_status = "Rotten"
    elif yellow_branch:
        pre_safety_status = "Fresh"
    else:
        scores = [
            _float(record, "unripe_rule_score"),
            _float(record, "fresh_rule_score"),
            _float(record, "rotten_rule_score"),
        ]
        pre_safety_status = ("Unripe", "Fresh", "Rotten")[int(np.argmax(scores))]

    safety_white = white >= 5
    safety_dark_damage = dark >= 8 and largest_damage >= 6
    safety_matched = safety_white or safety_dark_damage
    post_safety_status = "Rotten" if safety_matched else pre_safety_status
    clean_rescue = (
        post_safety_status == "Rotten" and yellow >= 90 and dark < 4
        and white < 1.5 and largest_lesion < 6 and largest_damage < 10
    )

    return {
        "Unripe primary gate selected": unripe_gate,
        "Condition: dark >= 8": strong_dark,
        "Condition: white mold >= 3": strong_white,
        "Condition: damage >= 25": strong_damage,
        "Condition: largest lesion >= 12": strong_lesion,
        "Condition: largest damage >= 15": strong_largest_damage,
        "Strong Rotten gate selected": strong_gate,
        "Predominantly yellow branch reached": yellow_branch,
        "Yellow branch: dark >= 5.5 selected Rotten": yellow_direct,
        "Yellow branch: compound damage rule selected Rotten": yellow_compound,
        "Score fallback reached": score_fallback,
        "Score fallback selected Unripe": score_fallback and pre_safety_status == "Unripe",
        "Score fallback selected Fresh": score_fallback and pre_safety_status == "Fresh",
        "Score fallback selected Rotten": score_fallback and pre_safety_status == "Rotten",
        "Safety condition: white mold >= 5": safety_white,
        "Safety condition: dark >= 8 and largest damage >= 6": safety_dark_damage,
        "Safety rule changed a non-Rotten class to Rotten": safety_matched and pre_safety_status != "Rotten",
        "Clean yellow rescue changed Rotten to Fresh": clean_rescue,
    }


def _rule_rows(successful: list[dict]) -> list[dict]:
    rule_definitions = None
    counts = Counter()
    for record in successful:
        activations = infer_rule_activations(record)
        rule_definitions = tuple(activations)
        counts.update(name for name, active in activations.items() if active)
    rule_definitions = rule_definitions or (
        "Unripe primary gate selected", "Condition: dark >= 8",
        "Condition: white mold >= 3", "Condition: damage >= 25",
        "Condition: largest lesion >= 12", "Condition: largest damage >= 15",
        "Strong Rotten gate selected", "Predominantly yellow branch reached",
        "Yellow branch: dark >= 5.5 selected Rotten",
        "Yellow branch: compound damage rule selected Rotten", "Score fallback reached",
        "Score fallback selected Unripe", "Score fallback selected Fresh",
        "Score fallback selected Rotten", "Safety condition: white mold >= 5",
        "Safety condition: dark >= 8 and largest damage >= 6",
        "Safety rule changed a non-Rotten class to Rotten",
        "Clean yellow rescue changed Rotten to Fresh",
    )
    total = len(successful)
    return [{
        "Rule or condition": name,
        "Count": counts[name],
        "Percentage of successful images": (counts[name] / total * 100) if total else "",
        "Evaluation scope": "successful images only",
    } for name in rule_definitions]


def _font(size: int, bold: bool = False):
    candidates = (
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _contact_sheet(misclassified: list[dict], path: Path) -> None:
    selected = misclassified[:25]
    if not selected:
        sheet = Image.new("RGB", (1400, 300), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((70, 115), "No misclassified images among successful evaluations.",
                  fill="black", font=_font(34, bold=True))
        sheet.save(path, quality=95)
        return

    columns, cell_width, cell_height = 5, 320, 285
    rows = (len(selected) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    title_font, label_font = _font(18, True), _font(16)
    for index, record in enumerate(selected):
        row, column = divmod(index, columns)
        left, top = column * cell_width, row * cell_height
        try:
            with Image.open(record["source_path"]) as source:
                image = ImageOps.fit(ImageOps.exif_transpose(source).convert("RGB"),
                                     (cell_width - 12, 205), method=Image.Resampling.LANCZOS)
        except Exception:
            image = Image.new("RGB", (cell_width - 12, 205), "#DDDDDD")
            ImageDraw.Draw(image).text((20, 90), "Image unavailable", fill="black", font=label_font)
        sheet.paste(image, (left + 6, top + 5))
        draw = ImageDraw.Draw(sheet)
        score = _float(record, "freshness_score")
        draw.text((left + 8, top + 215), f"Expected: {record['expected_class']}",
                  fill="#222222", font=title_font)
        draw.text((left + 8, top + 238), f"Predicted: {record['predicted_class']}",
                  fill="#A52A2A", font=label_font)
        draw.text((left + 8, top + 259), f"Freshness score: {score:.2f}",
                  fill="#222222", font=label_font)
    sheet.save(path, quality=95)


def _report_text(summary: dict, validation: dict, report: dict,
                 misclassification_rows: list[dict], rule_rows: list[dict],
                 classes: tuple[str, ...]) -> str:
    patterns = sorted(
        (row for row in misclassification_rows if row["Count"]),
        key=lambda row: row["Count"], reverse=True,
    )[:5]
    recalls = {name: report[name]["recall"] for name in classes}
    precisions = {name: report[name]["precision"] for name in classes}
    strongest_recall = max(recalls, key=recalls.get)
    weakest_recall = min(recalls, key=recalls.get)
    most_active = sorted(rule_rows, key=lambda row: row["Count"], reverse=True)[:5]
    pattern_lines = [
        f"- {row['Expected Class']} → {row['Predicted Class']}: "
        f"{row['Count']} ({row['Percentage']:.2f}% of successful {row['Expected Class']} images)"
        for row in patterns
    ] or ["- No misclassification pattern was observed among successful images."]
    class_lines = [
        f"| {name} | {int(report[name]['support'])} | {report[name]['precision']:.4f} | "
        f"{report[name]['recall']:.4f} | {report[name]['f1-score']:.4f} |"
        for name in classes
    ]
    active_lines = [f"- {row['Rule or condition']}: {row['Count']} images" for row in most_active]
    return "\n".join([
        "# MATLAB Rule-Based Evaluation Report", "",
        "## Dataset", "",
        f"- Manifest: `{validation['manifest_path']}`",
        "- Evaluation split: held-out test set only",
        f"- Attempted images: {summary['attempted_images']}",
        f"- Successful images: {summary['successful_images']}",
        f"- Failed images: {summary['failed_image_count']}",
        f"- Class counts: Fresh {validation['class_counts']['Fresh']}, "
        f"Unripe {validation['class_counts']['Unripe']}, Rotten {validation['class_counts']['Rotten']}", "",
        "## Classification performance", "",
        f"- Overall accuracy: {report['accuracy']:.4f}",
        f"- Macro precision: {report['macro avg']['precision']:.4f}",
        f"- Macro recall: {report['macro avg']['recall']:.4f}",
        f"- Macro F1-score: {report['macro avg']['f1-score']:.4f}",
        f"- Weighted precision: {report['weighted avg']['precision']:.4f}",
        f"- Weighted recall: {report['weighted avg']['recall']:.4f}",
        f"- Weighted F1-score: {report['weighted avg']['f1-score']:.4f}", "",
        "| Class | Support | Precision | Recall | F1-score |",
        "|---|---:|---:|---:|---:|", *class_lines, "",
        "## Processing speed", "",
        f"- Total MATLAB processing time: {summary['total_processing_time_seconds']:.3f} seconds",
        f"- Average MATLAB processing time: {summary['average_processing_time_seconds']:.3f} seconds per successful image",
        f"- End-to-end evaluation wall time: {summary['evaluation_wall_time_seconds']:.3f} seconds", "",
        "## Top five misclassification patterns", "", *pattern_lines, "",
        "## Observed strengths", "",
        f"- {strongest_recall} had the highest recall ({recalls[strongest_recall]:.4f}).",
        f"- {max(precisions, key=precisions.get)} had the highest precision "
        f"({max(precisions.values()):.4f}).", "",
        "## Observed weaknesses", "",
        f"- {weakest_recall} had the lowest recall ({recalls[weakest_recall]:.4f}).",
        "- Green and yellow masks overlap in hue range 0.15–0.19, so their percentages may sum above 100%.",
        "- The freshness score is a surface-condition formula, not a class probability.", "",
        "Most frequently matched rules or conditions:", "", *active_lines, "",
        "## Potential threshold improvements", "",
        "No threshold change is recommended automatically by this report. Candidate thresholds must be reviewed manually against the confusion matrix, per-class metrics, rule-activation counts, and the corresponding misclassified images before any classifier edit.", "",
    ])


def generate_report_artifacts(*, output_dir: Path, successful: list[dict],
                              misclassified: list[dict], summary: dict,
                              validation: dict, report: dict,
                              matrix: np.ndarray, normalized: np.ndarray,
                              classes: tuple[str, ...]) -> list[Path]:
    """Generate figures, tables, preview and Markdown from real successful records."""
    figures = output_dir / "figures"
    tables = output_dir / "tables"
    reports = output_dir / "reports"
    previews = output_dir / "previews"
    for folder in (figures, tables, reports, previews):
        folder.mkdir(parents=True, exist_ok=True)
    _prepare_style()

    generated = []
    confusion_path = figures / "confusion_matrix.png"
    normalized_path = figures / "normalized_confusion_matrix.png"
    _save_confusion(matrix, confusion_path, classes, False)
    _save_confusion(normalized, normalized_path, classes, True)
    generated.extend((confusion_path, normalized_path))

    for metric_key, filename, title in (
        ("precision", "per_class_precision.png", "Per-Class Precision"),
        ("recall", "per_class_recall.png", "Per-Class Recall"),
        ("f1-score", "per_class_f1.png", "Per-Class F1-Score"),
    ):
        path = figures / filename
        _save_bar([report[name][metric_key] for name in classes], classes, path,
                  title, metric_key.replace("-", " ").title(), True)
        generated.append(path)

    class_distribution = figures / "class_distribution.png"
    _save_bar([validation["class_counts"][name] for name in classes], classes,
              class_distribution, "Held-Out Test Class Distribution", "Number of images")
    generated.append(class_distribution)
    prediction_counts = Counter(record["predicted_class"] for record in successful)
    prediction_distribution = figures / "prediction_distribution.png"
    _save_bar([prediction_counts[name] for name in classes], classes,
              prediction_distribution, "MATLAB Prediction Distribution", "Number of predictions")
    generated.append(prediction_distribution)

    processing_histogram = figures / "processing_time_histogram.png"
    figure, axis = plt.subplots(figsize=(8, 5.2))
    times = [_float(record, "processing_time_seconds") for record in successful]
    axis.hist(times, bins=min(20, max(5, int(np.sqrt(len(times))))),
              color="#376996", edgecolor="white")
    axis.axvline(np.mean(times), color="#B7472A", linestyle="--", linewidth=2,
                 label=f"Mean = {np.mean(times):.3f} s")
    axis.set(title="MATLAB Processing-Time Distribution", xlabel="Seconds per image",
             ylabel="Number of images")
    axis.legend()
    figure.tight_layout()
    figure.savefig(processing_histogram)
    plt.close(figure)
    generated.append(processing_histogram)

    table_columns = ("Section", "Class", "Metric", "Value", "Support", "Precision",
                     "Recall", "F1-score", "Unit")
    metrics_path = tables / "metrics_summary.csv"
    _write_csv(metrics_path, _metrics_summary(
        report, summary["average_processing_time_seconds"], classes), table_columns)
    generated.append(metrics_path)

    misclassification_rows = _misclassification_rows(successful, classes)
    misclassification_path = tables / "misclassification_summary.csv"
    _write_csv(misclassification_path, misclassification_rows,
               ("Expected Class", "Predicted Class", "Count", "Percentage"))
    generated.append(misclassification_path)

    feature_path = tables / "feature_statistics.csv"
    _write_csv(feature_path, _feature_rows(successful, classes),
               ("Class", "Feature", "Count", "Average", "Minimum", "Maximum",
                "Standard Deviation"))
    generated.append(feature_path)

    rule_rows = _rule_rows(successful)
    rule_path = tables / "rule_activation_summary.csv"
    _write_csv(rule_path, rule_rows,
               ("Rule or condition", "Count", "Percentage of successful images",
                "Evaluation scope"))
    generated.append(rule_path)

    contact_path = previews / "misclassified_images_contact_sheet.png"
    _contact_sheet(misclassified, contact_path)
    generated.append(contact_path)

    report_path = reports / "matlab_rule_based_report.md"
    report_path.write_text(
        _report_text(summary, validation, report, misclassification_rows,
                     rule_rows, classes), encoding="utf-8"
    )
    generated.append(report_path)
    return generated
