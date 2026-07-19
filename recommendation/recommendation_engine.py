"""Transparent local recommendations; not a food-safety certification."""

from __future__ import annotations

import json
from pathlib import Path

from config.paths import BASE_DIR
from recommendation.recommendation_models import RecommendationResult


class RecommendationEngine:
    def __init__(self, rules_path: Path | None = None):
        path = rules_path or BASE_DIR / "recommendation" / "recommendation_rules.json"
        self.rules = json.loads(Path(path).read_text(encoding="utf-8"))

    @staticmethod
    def _safety_status(predicted_class: str | None, requires_review: bool,
                       confidence_level: str | None) -> tuple[str, str, str, str]:
        if predicted_class == "Rotten":
            return (
                "Unsafe to Consume",
                "unsafe",
                "FreshSight detected external visual characteristics consistent with a rotten papaya. "
                "Visible mold or advanced decay may indicate microbial spoilage beyond the visibly affected area.",
                "As a precaution, do not consume the fruit; isolate and dispose of it responsibly.",
            )
        if predicted_class == "Unripe":
            if requires_review and confidence_level == "low":
                return (
                    "Manual Inspection Required", "manual_review",
                    "The AI result is uncertain, so ripeness cannot be assessed reliably from this image.",
                    "Retake the image under even lighting and inspect the fruit manually.",
                )
            return (
                "Ripen Further", "ripen",
                "The papaya appears unripe and has not yet reached normal eating ripeness.",
                "Keep it at room temperature and reassess after further ripening.",
            )
        if predicted_class == "Fresh":
            if requires_review:
                return (
                    "Manual Inspection Required", "manual_review",
                    "MobileNetV2 predicts Fresh, but uncertainty or conflicting supporting evidence was detected.",
                    "Inspect the fruit and retake the image before relying on the result.",
                )
            return (
                "Safe to Consume", "safe",
                "No significant external visual evidence of spoilage was identified in this image.",
                "Wash before consumption and refrigerate after cutting.",
            )
        return (
            "Manual Inspection Required", "manual_review",
            "FreshSight could not produce a reliable primary classification.",
            "Inspect the fruit manually and retake the image.",
        )

    def recommend(self, hybrid_result: dict) -> RecommendationResult:
        ai = hybrid_result.get("ai_detection") or {}
        matlab = hybrid_result.get("matlab_analysis") or {}
        assessment = hybrid_result.get("system_assessment") or {}
        predicted_class = ai.get("predicted_class") if ai.get("available") else None
        recommendations = list(self.rules.get(predicted_class, {}).get("base", []))
        rationale: list[str] = []
        measurements = matlab.get("measurements") or {}
        reliability = matlab.get("measurement_reliability") or {}
        segmentation = matlab.get("segmentation_quality") or {}
        segmentation_reliable = reliability.get("segmentation_reliable", True)
        damage_reliable = reliability.get("damage_reliable", segmentation_reliable)
        lesion_reliable = reliability.get("lesion_reliable", segmentation_reliable)
        mold_reliable = reliability.get("mold_reliable", segmentation_reliable)
        damage = matlab.get("damage_percentage", measurements.get("damage_percentage"))
        healthy = matlab.get("healthy_percentage", measurements.get("healthy_percentage"))
        lesion = measurements.get("lesion_percentage")
        mold = measurements.get("white_mold_percentage")

        primary_assessment = (
            f"MobileNetV2 predicts {predicted_class}." if predicted_class
            else "MobileNetV2 classification is unavailable."
        )
        confidence = ai.get("confidence")
        confidence_statement = (
            f"AI confidence is {float(confidence) * 100:.2f}%." if confidence is not None
            else "AI confidence is unavailable."
        )
        rationale.append(primary_assessment)

        if not predicted_class:
            recommendations.append("Request manual classification because AI is unavailable.")

        if damage_reliable and damage is not None:
            rationale.append(f"Reliable MATLAB damage estimate: {float(damage):.2f}%.")
        if damage_reliable and healthy is not None:
            rationale.append(f"Reliable MATLAB healthy-area estimate: {float(healthy):.2f}%.")
        if lesion_reliable and lesion is not None and float(lesion) > 0:
            rationale.append(f"Reliable lesion estimate: {float(lesion):.2f}%.")
        if mold_reliable and mold is not None and float(mold) > 0:
            rationale.append(f"Reliable white-mold-like estimate: {float(mold):.2f}%.")

        if not segmentation_reliable:
            if predicted_class != "Rotten":
                recommendations = list(self.rules.get("UnreliableSegmentation", []))
            else:
                recommendations.extend(
                    item for item in self.rules.get("UnreliableSegmentation", [])
                    if item not in recommendations
                )
            matlab_support = (
                "MATLAB measurements are unavailable or unreliable because papaya segmentation quality was poor."
            )
            retake = "Capture the whole fruit clearly against a simple background with even lighting."
            rationale.extend([matlab_support, retake])
        else:
            matlab_support = "Reliable MATLAB visual measurements are available as supporting evidence."
            retake = "Retake the image if the fruit is obstructed, blurred, cropped, or unevenly lit."

        disagreement = ""
        if assessment.get("ai_matlab_agreement") is False:
            disagreement = "AI and the reliable MATLAB rule result disagree; manual review is required."
            evidence = matlab.get("damage_evidence") or {}
            detected = [
                label for key, label in (
                    ("brown_decay_percentage", "brown decay"),
                    ("dark_decay_percentage", "dark decay"),
                    ("mold_percentage", "mold-like regions"),
                    ("lesion_percentage", "lesion regions"),
                    ("abnormal_texture_percentage", "abnormal texture"),
                ) if evidence.get(key) is not None and float(evidence[key]) > 0
            ]
            if detected:
                matlab_support = "MATLAB detected supporting " + ", ".join(detected) + "."
            if predicted_class in {"Fresh", "Unripe"}:
                recommendations = list(self.rules.get("ReliableDisagreement", []))

        if assessment.get("requires_manual_review"):
            for item in self.rules.get("LowConfidence", []):
                if item not in recommendations:
                    recommendations.append(item)
            rationale.extend(assessment.get("review_reasons", []))

        status, code, explanation, precaution = self._safety_status(
            predicted_class,
            bool(assessment.get("requires_manual_review")),
            ai.get("confidence_level"),
        )
        return RecommendationResult(
            title="FreshSight Recommendation Assistant",
            predicted_class=predicted_class,
            recommendations=recommendations,
            rationale=rationale,
            disclaimer=(
                "FreshSight evaluates external visual characteristics only and is not food-safety certification. "
                "It does not detect bacteria or confirm internal contamination. When spoilage, mold, "
                "leakage, unusual odour, or uncertainty is present, discard the fruit or seek qualified advice."
            ),
            primary_assessment=primary_assessment,
            confidence_statement=confidence_statement,
            matlab_support_statement=matlab_support,
            disagreement_statement=disagreement,
            retake_guidance=retake,
            handling_guidance=(recommendations[0] if recommendations else "Request manual review."),
            evidence_reliability=(
                "reliable" if segmentation_reliable else
                f"unreliable ({segmentation.get('status', 'poor')} segmentation)"
            ),
            food_safety_status=status,
            food_safety_code=code,
            food_safety_explanation=explanation,
            precautionary_action=precaution,
        )
