# Result and report mapping

The upload endpoint returns three top-level analysis objects.

| Object | Source | Important fields |
|---|---|---|
| `ai_detection` | MobileNetV2 `PredictionService` | `available`, `status`, `predicted_class`, `confidence`, `probabilities`, `error` |
| `matlab_analysis` | `run_freshsight_api` | `available`, `status`, `rule_class`, `grade`, `freshness_score`, `suggestion`, `measurements`, `rule_scores`, `images`, `processing_time_seconds`, `error` |
| `system_assessment` | `HybridAnalysisService` | `status`, `ai_matlab_agreement`, `requires_manual_review`, `review_reasons`, `primary_classification_source`, `primary_classification` |

MATLAB `measurements` contains only values computed by
`extract_papaya_features`: `green_percentage`, `yellow_percentage`,
`brown_percentage`, `dark_percentage`, `white_mold_percentage`,
`lesion_percentage`, `rough_percentage`, `damage_percentage`,
`healthy_percentage`, `largest_damage_percentage`,
`largest_lesion_percentage`, and brown/dark/white-mold/damage/lesion region counts.

MATLAB `images` contains PNG Base64 for `original`, `papaya_mask`, `green_area`,
`yellow_area`, `brown_area`, `white_mold`, `dark_area`, `lesion_area`, `rough_area`,
`damage_mask`, and `damage_highlight`. Failure objects do not contain placeholder
measurements or fabricated zero values.

AI results additionally contain `confidence_level`, `top_two_probability_margin`,
`uncertainty_warning`, model identity, device, and processing time. Unavailable
fields are omitted or null and the interface displays “Not available.”
