"""Research-only DenseNet121 inference + Grad-CAM. No checkpoint means no prediction."""
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
    "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema",
    "Fibrosis", "Pleural_Thickening", "Hernia",
]


class ModelUnavailable(RuntimeError):
    """Raised rather than returning fabricated output when trained weights are absent."""


class ChestXrayService:
    def __init__(self, checkpoint_path: str | Path | None = None):
        self.checkpoint_path = Path(checkpoint_path or os.getenv("MODEL_PATH", "models/densenet121_nih.pth"))
        self.model = None
        self.torch = None
        self.transforms = None
        self.version = None
        self.labels = LABELS
        self.input_size = 224
        self.last_error = None
        if not self.checkpoint_path.is_file():
            self.last_error = "Trained checkpoint not found"
            return
        try:
            import torch
            from torchvision import models, transforms

            state = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
            labels = state.get("labels", LABELS) if isinstance(state, dict) else LABELS
            input_size = int(state.get("input_size", 224)) if isinstance(state, dict) else 224
            model = models.densenet121(weights=None)
            model.classifier = torch.nn.Linear(model.classifier.in_features, len(labels))
            state_dict = state.get("state_dict", state) if isinstance(state, dict) else state
            model.load_state_dict(state_dict)
            model.eval()
            self.model = model
            self.torch = torch
            self.labels = labels
            self.input_size = input_size
            self.transforms = transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self.version = state.get("model_version", self.checkpoint_path.name) if isinstance(state, dict) else self.checkpoint_path.name
        except Exception as exc:  # fail closed: never fall back to random weights
            self.last_error = f"Checkpoint could not be loaded: {type(exc).__name__}: {exc}"
            self.model = None

    @property
    def available(self) -> bool:
        return self.model is not None and self.torch is not None

    def predict(self, image_bytes: bytes) -> dict[str, Any]:
        if not self.available:
            raise ModelUnavailable(self.last_error or "Model checkpoint is not configured")
        torch = self.torch
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            raise ValueError("Uploaded file is not a supported image") from exc
        tensor = self.transforms(image).unsqueeze(0)
        self.model.zero_grad(set_to_none=True)
        logits = self.model(tensor)
        probabilities = torch.sigmoid(logits)[0]
        max_index = int(torch.argmax(probabilities).item())
        cam = self._gradcam(tensor, max_index)
        overlay = self._overlay(image, cam)
        findings = [
            {"label": self.labels[i], "score": round(float(probabilities[i].item()), 6)}
            for i in range(len(self.labels))
        ]
        findings.sort(key=lambda item: item["score"], reverse=True)
        return {
            "findings": findings,
            "top_finding": findings[0]["label"],
            "top_score": findings[0]["score"],
            "heatmap_png": overlay,
            "model_version": str(self.version or "unknown"),
            "model_kind": "DenseNet121 multi-label chest X-ray classifier",
            "clinical_use": False,
        }

    def _gradcam(self, tensor, class_index: int) -> np.ndarray:
        """Compute Grad-CAM at DenseNet's final dense block."""
        torch = self.torch
        activation = {}
        gradients = {}
        layer = self.model.features.denseblock4
        def capture(_module, _inputs, output):
            activation["value"] = output
            output.register_hook(lambda grad: gradients.__setitem__("value", grad))
        handle = layer.register_forward_hook(capture)
        try:
            logits = self.model(tensor)
            logits[0, class_index].backward()
            feature_map = activation["value"].detach()
            grads = gradients.get("value")
            if grads is None:
                raise RuntimeError("Grad-CAM gradients unavailable for target layer")
            weights = grads.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * feature_map).sum(dim=1, keepdim=True))
            cam = torch.nn.functional.interpolate(cam, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
            cam = cam[0, 0]
            cam = cam / (cam.max() + 1e-8)
            return cam.detach().cpu().numpy().astype(np.float32)
        finally:
            handle.remove()

    @staticmethod
    def _overlay(image: Image.Image, cam: np.ndarray) -> bytes:
        # A lightweight red/yellow heatmap composite. It explains activation, not pathology.
        import matplotlib.cm as cm
        import numpy as np
        heat = (cm.jet(cam)[..., :3] * 255).astype("uint8")
        heat_img = Image.fromarray(heat).resize(image.size, Image.Resampling.BILINEAR)
        base = image.convert("RGB")
        overlay = Image.blend(base, heat_img, alpha=0.38)
        output = io.BytesIO()
        overlay.save(output, format="PNG")
        return output.getvalue()
