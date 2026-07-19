"""Independent AI and MATLAB analysis with transparent uncertainty handling."""

from __future__ import annotations

from typing import Any


class HybridAnalysisService:
    def __init__(self, prediction_service: Any, matlab_service: Any, web_config: dict):
        self.prediction_service = prediction_service
        self.matlab_service = matlab_service
        self.web_config = web_config

    @staticmethod
    def _normalise_result(result: dict, unavailable_status: str) -> dict:
        if result.get("available") is True and result.get("status") == "success":
            return result
        if "available" not in result:
            return {
                "available": False, "status": unavailable_status,
                "error": result.get("error", "Subsystem unavailable."),
            }
        return result

    def analyze_image(self, image_path: str) -> dict:
        try:
            ai = self._normalise_result(
                self.prediction_service.predict(image_path), "model_unavailable"
            )
        except Exception as exc:
            ai = {"available": False, "status": "inference_error", "error": str(exc)}
        try:
            matlab = self._normalise_result(
                self.matlab_service.analyze_image(image_path), "unavailable"
            )
        except Exception as exc:
            matlab = {"available": False, "status": "unavailable", "error": str(exc)}

        ai_ok = ai.get("available") is True and ai.get("status") == "success"
        matlab_ok = matlab.get("available") is True and matlab.get("status") == "success"
        if matlab_ok:
            measurements = matlab.get("measurements") or {}
            if "damage_percentage" in measurements:
                matlab["damage_percentage"] = measurements["damage_percentage"]
            if "healthy_percentage" in measurements:
                matlab["healthy_percentage"] = measurements["healthy_percentage"]
            matlab.setdefault(
                "processing_time_seconds",
                matlab.get("service_processing_time_seconds"),
            )

        reliability = matlab.get("measurement_reliability") or {}
        matlab_class_reliable = reliability.get("matlab_class_reliable", True)
        rule_available = matlab.get("rule_class") not in {None, "", "Unavailable"}
        agreement = (
            ai.get("predicted_class") == matlab.get("rule_class")
            if ai_ok and matlab_ok and rule_available and matlab_class_reliable else None
        )
        reasons = []
        if not ai_ok:
            reasons.append("AI classification is unavailable.")
        if not matlab_ok:
            reasons.append("MATLAB supporting analysis is unavailable.")
        elif not reliability.get("segmentation_reliable", True):
            reasons.append("MATLAB measurements are unavailable because papaya segmentation was unreliable.")
        elif not matlab_class_reliable:
            reasons.append("MATLAB classification has low reliability and requires manual review.")
        if ai_ok:
            reasons.extend(ai.get("uncertainty_reasons", []))
        if agreement is False:
            reasons.append("AI and MATLAB classifications disagree.")

        requires_review = bool(reasons)
        status = "success" if ai_ok and matlab_ok else ("partial" if ai_ok or matlab_ok else "failed")
        return {
            "ai_detection": ai,
            "matlab_analysis": matlab,
            "system_assessment": {
                "status": status,
                "ai_matlab_agreement": agreement,
                "requires_manual_review": requires_review,
                "review_reasons": reasons,
                "primary_classification_source": "MobileNetV2" if ai_ok else None,
                "primary_classification": ai.get("predicted_class") if ai_ok else None,
                "explanation": (
                    "MobileNetV2 is the primary classification model. MATLAB provides "
                    "supporting measurements and rule-based comparison."
                ),
            },
        }
