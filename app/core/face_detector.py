"""
Face detection and cropping module.
Uses OpenCV Haar Cascade detector with configurable margin/padding.
"""

from typing import List, Tuple
import cv2
import numpy as np
from app.config import settings


class FaceDetector:
    def __init__(self, padding_ratio: float = settings.FACE_PADDING_RATIO):
        self.padding_ratio = padding_ratio
        self.cascade: cv2.CascadeClassifier | None = None
        self._load_cascade()

    def _load_cascade(self) -> None:
        """Load OpenCV frontal face Haar Cascade."""
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            raise RuntimeError(f"Failed to load Haar Cascade from {cascade_path}")

    @property
    def is_ready(self) -> bool:
        return self.cascade is not None and not self.cascade.empty()

    def detect_and_crop(
        self, bgr_image: np.ndarray
    ) -> list[tuple[np.ndarray, list[int]]]:
        """
        Detect faces in a BGR image and return:
        List of tuples: (face_crop_rgb, [x1, y1, x2, y2])

        If no faces are found, returns an empty list.
        """
        if self.cascade is None or self.cascade.empty():
            raise RuntimeError("Face detector is not loaded.")

        if bgr_image is None or bgr_image.size == 0:
            return []

        h, w = bgr_image.shape[:2]
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

        # Equalize histogram slightly for better face detection under varied lighting
        detections = self.cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(settings.MIN_FACE_SIZE, settings.MIN_FACE_SIZE),
        )

        results: list[tuple[np.ndarray, list[int]]] = []
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        for (x, y, fw, fh) in detections:
            # Apply padding around face bbox to include forehead, jawline, and hairline
            pad_x = max(int(fw * self.padding_ratio), 8)
            pad_y = max(int(fh * self.padding_ratio), 8)

            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w, x + fw + pad_x)
            y2 = min(h, y + fh + pad_y)

            crop = rgb_image[y1:y2, x1:x2]
            if crop.size > 0:
                results.append((crop, [int(x1), int(y1), int(x2), int(y2)]))

        return results


face_detector = FaceDetector()
