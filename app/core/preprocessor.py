"""
Image preprocessing pipeline matching the model training pipeline.
"""

from typing import List, Union
import cv2
import numpy as np
from app.config import settings


class Preprocessor:
    def __init__(
        self,
        image_size: int = settings.IMAGE_SIZE,
        mean: list[float] | None = None,
        std: list[float] | None = None,
    ):
        self.image_size = image_size
        self.mean = np.array(mean or settings.MEAN, dtype=np.float32)
        self.std = np.array(std or settings.STD, dtype=np.float32)

    def preprocess_face(self, face_rgb: np.ndarray) -> np.ndarray:
        """
        Preprocess a single RGB face crop:
        1. Resize to (224, 224)
        2. Scale pixel values to [0.0, 1.0]
        3. Standardize using ImageNet mean & std
        4. Transpose from HWC (224, 224, 3) to CHW (3, 224, 224)
        5. Add batch dimension -> (1, 3, 224, 224)
        """
        if face_rgb.size == 0:
            raise ValueError("Empty face crop passed to preprocessor.")

        resized = cv2.resize(face_rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        normalized = resized.astype(np.float32) / 255.0
        standardized = (normalized - self.mean) / self.std
        chw = standardized.transpose(2, 0, 1)
        return chw[np.newaxis, ...].astype(np.float32)

    def preprocess_batch(self, faces_rgb: list[np.ndarray]) -> np.ndarray:
        """
        Preprocess a list of RGB face crops into a single batched tensor (N, 3, 224, 224).
        """
        if not faces_rgb:
            raise ValueError("No faces provided for batch preprocessing.")

        tensors = [self.preprocess_face(f)[0] for f in faces_rgb]
        return np.stack(tensors, axis=0).astype(np.float32)


preprocessor = Preprocessor()
