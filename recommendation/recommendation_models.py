from dataclasses import dataclass


@dataclass
class RecommendationResult:
    title: str
    predicted_class: str | None
    recommendations: list[str]
    rationale: list[str]
    disclaimer: str
    primary_assessment: str = ""
    confidence_statement: str = ""
    matlab_support_statement: str = ""
    disagreement_statement: str = ""
    retake_guidance: str = ""
    handling_guidance: str = ""
    evidence_reliability: str = ""
    food_safety_status: str = "Manual Inspection Required"
    food_safety_code: str = "manual_review"
    food_safety_explanation: str = ""
    precautionary_action: str = ""
