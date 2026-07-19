# FreshSight system architecture

FreshSight keeps its two analysis subsystems separate:

1. `PredictionService` lazily loads the registered epoch-19 MobileNetV2 checkpoint
   once and reuses it. MobileNetV2 is the primary classifier. Training remains
   CUDA-only; local web inference prefers CUDA and may use explicitly configured CPU
   fallback.
2. `MatlabService` is the active deterministic image-processing subsystem. It
   lazily starts MATLAB on the first request, adds `matlab/` through `genpath`,
   verifies `which run_freshsight_api`, serializes calls through one reusable Engine
   session, and closes the session during process shutdown.
3. `HybridAnalysisService` calls both independently and returns `ai_detection`,
   `matlab_analysis`, and `system_assessment`. MATLAB does not impersonate AI and AI
   does not overwrite MATLAB measurements.

If the AI checkpoint is absent, AI reports `model_not_trained` while MATLAB can
still succeed. If MATLAB is unavailable, AI can still succeed. If both fail, the
system status is `failed`. Manual review is required when either subsystem is
unavailable, AI confidence is below the configured threshold, or the two class
labels disagree. When AI is available its label is the primary displayed class;
the MATLAB rule class remains visible as a separate result.

CUDA configuration applies only to PyTorch. MATLAB Engine execution is controlled
by MATLAB and is not moved onto the PyTorch CUDA device by this project.

SQLite stores immutable analysis predictions and separate user corrections. A
correction never overwrites the prediction and never starts retraining. Standalone
HTML reports are generated locally under `outputs/analysis_reports/`.
