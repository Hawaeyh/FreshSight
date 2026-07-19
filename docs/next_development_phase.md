# FreshSight operational and review phase

This phase adds local analysis history, feedback administration, duplicate-image feedback protection, active-learning candidate scoring, on-demand Grad-CAM, a performance dashboard, and a system-health page.

## Data integrity

Each new analysis records the uploaded image SHA-256, AI device and processing time, MATLAB processing time, agreement, confidence metadata, and manual-review status. User corrections are stored in `prediction_feedback.corrected_class`; `analysis_history.predicted_class` remains unchanged. A unique SHA-256 feedback index rejects a second feedback submission for byte-identical image content.

Approving or rejecting feedback changes only `review_status`. `included_in_retraining` remains false. Candidate scores rank uncertain, close-margin, disagreeing, manually reviewed, or user-corrected examples. There is no training call in this workflow.

## Grad-CAM

Open an entry from `/history`, then select **Generate Grad-CAM**. The overlay is written to `outputs/explainability/<analysis_uuid>.png`. Generation uses the active checkpoint already registered for web inference. Hooks are removed after the request, gradients are cleared, and no checkpoint is written.

## Dashboards

- `/history`
- `/admin/feedback`
- `/dashboard/performance`
- `/system-health`

The performance dashboard reads the existing held-out evaluation files. The health page does not start MATLAB or run inference; it reports whether the reusable MATLAB session has already started.
