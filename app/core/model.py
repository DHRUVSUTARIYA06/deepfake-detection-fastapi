"""
Model loader and inference engine for deepfake detection.
Loads the trained EfficientNet-B0 model once at application startup
and provides thread-safe inference for both image and video endpoints.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np

from app.config import settings

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


class DeepfakeModel:
    def __init__(self):
        self.session: Optional[Any] = None
        self.framework: str = "unknown"
        self.weights_path: Optional[Path] = None
        self.input_name: Optional[str] = None
        self.output_name: Optional[str] = None
        self.device: str = "CPU"
        self.is_loaded: bool = False
        self.validation_accuracy: Optional[float] = 0.9599  # From checkpoint metadata

    def load(self) -> None:
        """
        Load the model weights.
        Prioritizes ONNX model (lightweight, optimized for CPU and deployment).
        Falls back to PyTorch checkpoint if ONNX is unavailable.
        """
        # 1. Try ONNX model
        onnx_path = settings.MODEL_ONNX_PATH
        # Also check alternative folder naming if path doesn't exist
        if not onnx_path.exists():
            alt_onnx = settings.BASE_DIR / "model" / "deepfake_efficientnet_b0.onnx"
            if alt_onnx.exists():
                onnx_path = alt_onnx

        if ONNX_AVAILABLE and onnx_path.exists():
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.intra_op_num_threads = 4

            # Detect available providers
            available = ort.get_available_providers()
            providers = ["CPUExecutionProvider"]
            if "CUDAExecutionProvider" in available:
                providers.insert(0, "CUDAExecutionProvider")
                self.device = "CUDA / GPU"
            else:
                self.device = "CPU"

            self.session = ort.InferenceSession(
                str(onnx_path),
                sess_options=opts,
                providers=providers,
            )
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self.weights_path = onnx_path
            self.framework = "ONNX Runtime"
            self.is_loaded = True
            return

        # 2. Fallback: PyTorch Checkpoint
        pt_path = settings.MODEL_PT_PATH
        if not pt_path.exists():
            alt_pt = settings.BASE_DIR / "model" / "deepfake_efficientnet_b0.pt"
            if alt_pt.exists():
                pt_path = alt_pt

        if pt_path.exists():
            try:
                import torch
                from torchvision import models

                checkpoint = torch.load(str(pt_path), map_location="cpu", weights_only=False)
                if isinstance(checkpoint, dict):
                    state_dict = checkpoint.get("model_state_dict", checkpoint)
                    self.validation_accuracy = checkpoint.get("validation_accuracy", 0.9599)
                else:
                    state_dict = checkpoint

                model = models.efficientnet_b0(weights=None)
                num_classes = len(settings.CLASS_MAPPING)
                model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
                model.load_state_dict(state_dict)
                model.eval()

                self.session = model
                self.weights_path = pt_path
                self.framework = "PyTorch"
                self.device = "CPU"
                self.is_loaded = True
                return
            except Exception as e:
                raise RuntimeError(f"Failed to load PyTorch checkpoint: {e}")

        raise FileNotFoundError(
            f"No trained model found. Checked:\n"
            f" - ONNX: {settings.MODEL_ONNX_PATH}\n"
            f" - PyTorch: {settings.MODEL_PT_PATH}"
        )

    def predict_batch(self, batch_tensor: np.ndarray) -> list[dict[str, Any]]:
        """
        Run inference on a batch of preprocessed face tensors of shape (N, 3, 224, 224).
        Returns list of dicts with fake_probability, real_probability, prediction.
        """
        if not self.is_loaded or self.session is None:
            raise RuntimeError("Model is not loaded. Ensure startup sequence completed.")

        if self.framework == "ONNX Runtime":
            logits = self.session.run([self.output_name], {self.input_name: batch_tensor})[0]
        elif self.framework == "PyTorch":
            import torch
            with torch.no_grad():
                tensor_torch = torch.from_numpy(batch_tensor)
                logits = self.session(tensor_torch).numpy()
        else:
            raise RuntimeError(f"Unsupported framework: {self.framework}")

        results: list[dict[str, Any]] = []
        for logit in logits:
            # Numerically stable Softmax
            exp_logits = np.exp(logit - np.max(logit))
            probs = exp_logits / exp_logits.sum()

            fake_prob = float(probs[0])
            real_prob = float(probs[1])

            # Apply classification threshold
            prediction = "FAKE" if fake_prob >= settings.CLASSIFICATION_THRESHOLD else "REAL"

            results.append({
                "fake_probability": round(fake_prob, 4),
                "real_probability": round(real_prob, 4),
                "prediction": prediction,
            })

        return results


# Global model singleton loaded on application startup
model_engine = DeepfakeModel()
