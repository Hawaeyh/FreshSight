"""On-demand Grad-CAM generation; never changes model weights or checkpoints."""

from __future__ import annotations

from pathlib import Path
import threading

import numpy as np
import torch
from PIL import Image

from config.paths import EXPLAINABILITY_DIR


class GradCAMService:
    def __init__(self, prediction_service):
        self.prediction_service = prediction_service
        self._lock = threading.Lock()

    def generate(self, image_path: str, analysis_uuid: str) -> Path:
        output = EXPLAINABILITY_DIR / f"{analysis_uuid}.png"
        if output.is_file():
            return output
        source = Path(image_path)
        with Image.open(source) as opened:
            image = opened.convert("RGB")
        service = self.prediction_service
        model = service._ensure_model()
        layer = model.features[-1]
        activations, gradients = {}, {}

        def save_activation(_module, _inputs, value):
            activations["value"] = value

        def save_gradient(_module, _grad_input, grad_output):
            gradients["value"] = grad_output[0]

        with self._lock, service._inference_lock:
            forward = layer.register_forward_hook(save_activation)
            backward = layer.register_full_backward_hook(save_gradient)
            try:
                tensor = service.transform(image).unsqueeze(0).to(service.device)
                model.zero_grad(set_to_none=True)
                with torch.enable_grad():
                    logits = model(tensor)
                    target = int(logits.argmax(dim=1).item())
                    logits[0, target].backward()
                weights = gradients["value"].mean(dim=(2, 3), keepdim=True)
                cam = torch.relu((weights * activations["value"]).sum(dim=1))[0]
                maximum = cam.max()
                if maximum > 0:
                    cam = cam / maximum
                cam_image = Image.fromarray((cam.detach().cpu().numpy() * 255).astype("uint8"))
                cam_image = cam_image.resize(image.size, Image.Resampling.BILINEAR)
                heat = np.asarray(cam_image, dtype=np.float32) / 255.0
                original = np.asarray(image, dtype=np.float32)
                color = np.zeros_like(original)
                color[..., 0] = 255.0 * heat
                color[..., 1] = 80.0 * (1.0 - heat)
                overlay = Image.fromarray(np.uint8(np.clip(original * 0.62 + color * 0.38, 0, 255)))
                EXPLAINABILITY_DIR.mkdir(parents=True, exist_ok=True)
                overlay.save(output, format="PNG")
            finally:
                forward.remove()
                backward.remove()
                model.zero_grad(set_to_none=True)
        return output
