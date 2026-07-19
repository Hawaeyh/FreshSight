"""Score review candidates without training or modifying the active model."""


def score_candidate(result: dict, feedback_status: str | None = None) -> tuple[float, list[str]]:
    ai = result.get("ai_detection") or {}
    assessment = result.get("system_assessment") or {}
    reasons = []
    score = 0.0
    confidence = ai.get("confidence")
    margin = ai.get("top_two_probability_margin")
    if confidence is not None:
        score += max(0.0, 1.0 - float(confidence)) * 0.45
    if margin is not None and float(margin) < 0.10:
        score += 0.20
        reasons.append("Close top-two probabilities")
    if assessment.get("ai_matlab_agreement") is False:
        score += 0.20
        reasons.append("AI/MATLAB disagreement")
    if assessment.get("requires_manual_review"):
        score += 0.10
        reasons.append("Manual review required")
    if feedback_status == "incorrect":
        score += 0.45
        reasons.append("User correction submitted")
    if confidence is not None and float(confidence) < 0.65:
        reasons.append("Low AI confidence")
    return min(score, 1.0), reasons
