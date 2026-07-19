"""Lazy, reusable web inference for the registered FreshSight MobileNetV2 model."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image, UnidentifiedImageError
from torch.torch_version import TorchVersion
from torchvision import transforms

from ai.models.model_factory import create_model
from config.paths import BASE_DIR


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _error(status: str, message: str) -> dict:
    return {"available": False, "status": status, "error": message}


class PredictionService:
    def __init__(self, model_config: dict, registry: dict, web_config: dict):
        self.model_config = model_config
        self.registry = registry["active_model"]
        self.web_config = web_config
        self.model: Any = None
        self.device: torch.device | None = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self.transform = transforms.Compose([
            transforms.Resize(model_config["resize_size"]),
            transforms.CenterCrop(model_config["image_size"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def _select_web_device(self) -> torch.device:
        settings = self.web_config["web_inference"]
        requested = str(settings.get("device", "auto")).lower()
        fallback = bool(settings.get("allow_cpu_fallback", False))
        cuda_ready = torch.cuda.is_available() and torch.cuda.device_count() > 0
        if requested == "cpu":
            return torch.device("cpu")
        if requested in {"auto", "cuda"} and cuda_ready:
            return torch.device("cuda:0")
        if requested in {"auto", "cuda"} and fallback:
            return torch.device("cpu")
        raise RuntimeError(
            f"Web inference requested '{requested}', CUDA is unavailable, and CPU fallback is disabled."
        )

    def _ensure_model(self):
        if self.model is not None:
            return self.model
        with self._load_lock:
            if self.model is not None:
                return self.model
            if not self.registry.get("active"):
                raise RuntimeError("The registered model is not active.")
            checkpoint_path = (BASE_DIR / self.registry["checkpoint_path"]).resolve()
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"Active model checkpoint is missing: {checkpoint_path}")
            classes = self.registry["class_order"]
            if classes != ["Fresh", "Unripe", "Rotten"]:
                raise ValueError(f"Unsupported active-model class order: {classes}")
            device = self._select_web_device()
            with torch.serialization.safe_globals([TorchVersion]):
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
            if checkpoint.get("classes") != classes:
                raise ValueError("Checkpoint class order does not match the active registry.")
            model = create_model(
                num_classes=len(classes), dropout=self.model_config.get("dropout", 0.3),
                pretrained=False, freeze_backbone=False,
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device).eval()
            self.device = device
            self.model = model
            return model

    def _confidence_details(self, probabilities: list[float]) -> dict:
        thresholds = self.web_config["confidence"]
        confidence = max(probabilities)
        ordered = sorted(probabilities, reverse=True)
        margin = ordered[0] - ordered[1]
        if confidence >= float(thresholds["high_threshold"]):
            level = "high"
        elif confidence >= float(thresholds["moderate_threshold"]):
            level = "moderate"
        else:
            level = "low"
        low = level == "low"
        close = margin < float(thresholds["close_probability_margin"])
        return {
            "confidence_level": level,
            "top_two_probability_margin": float(margin),
            "uncertainty_warning": low or close,
            "uncertainty_reasons": [
                reason for condition, reason in (
                    (low, "AI confidence is low."),
                    (close, "The two highest class probabilities are close."),
                ) if condition
            ],
        }

    def predict(self, image_path: str) -> dict:
        source = Path(image_path).expanduser().resolve()
        if not source.is_file():
            return _error("invalid_image", f"Image file does not exist: {source}")
        if source.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            return _error("invalid_image", f"Unsupported image type: {source.suffix}")
        try:
            with Image.open(source) as candidate:
                candidate.verify()
            with Image.open(source) as candidate:
                image = candidate.convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            return _error("invalid_image", f"Image is corrupt or unreadable: {exc}")

        try:
            model = self._ensure_model()
        except Exception as exc:
            return _error("model_unavailable", str(exc))

        started = time.perf_counter()
        try:
            tensor = self.transform(image).unsqueeze(0).to(
                self.device, non_blocking=self.device.type == "cuda"
            )
            with self._inference_lock, torch.inference_mode():
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                with torch.amp.autocast(
                    device_type=self.device.type,
                    enabled=self.device.type == "cuda" and self.model_config.get("mixed_precision", True),
                ):
                    logits = model(tensor)
                probabilities = torch.softmax(logits.float(), dim=1)[0].cpu().tolist()
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
            prediction_index = int(max(range(len(probabilities)), key=probabilities.__getitem__))
            classes = self.registry["class_order"]
            confidence = float(probabilities[prediction_index])
            result = {
                "available": True,
                "status": "success",
                "error": "",
                "predicted_class": classes[prediction_index],
                "confidence": confidence,
                "probabilities": {name: float(probabilities[i]) for i, name in enumerate(classes)},
                "model_name": self.registry["model_name"],
                "model_version": self.registry["model_version"],
                "device": str(self.device),
                "processing_time_seconds": time.perf_counter() - started,
            }
            result.update(self._confidence_details(probabilities))
            return result
        except Exception as exc:
            return _error("inference_error", f"AI inference failed: {exc}")


_SERVICE = None
_SERVICE_LOCK = threading.Lock()


def get_prediction_service(model_config: dict, registry: dict, web_config: dict) -> PredictionService:
    global _SERVICE
    if _SERVICE is None:
        with _SERVICE_LOCK:
            if _SERVICE is None:
                _SERVICE = PredictionService(model_config, registry, web_config)
    return _SERVICE
