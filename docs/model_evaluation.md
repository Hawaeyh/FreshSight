# MobileNetV2 model evaluation and registry

The active registry is `config/model_registry.json`. It points directly to
`ai/checkpoints/mobilenetv2_cleaned_baseline/best_model.pth`; the checkpoint is not
copied, renamed, or modified.

The model was evaluated once on 147 held-out test images: accuracy 96.60%, macro
F1-score 96.53%, and weighted F1-score 96.62%. MATLAB rule accuracy was 63.95%.
MobileNetV2 is therefore the primary classifier, but external-data validation is
still recommended. One high-confidence held-out error occurred, so confidence is
not a guarantee of correctness.

Web inference uses the same deterministic resize, centre-crop, tensor conversion,
and ImageNet normalization as test evaluation. The service uses
`torch.inference_mode()` and never performs retraining.
