"""
Detector service orchestrating image deepfake detection.
"""

import time
from pathlib import Path
from typing import List

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile

from app.config import settings
from app.core.face_detector import face_detector
from app.core.model import model_engine
from app.core.preprocessor import preprocessor
from app.schemas import FaceDetail, ImageDetectionResponse

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class ImageDetectorService:
    def __init__(self):
        pass

    async def validate_and_decode(self, file: UploadFile) -> np.ndarray:
        """
        Validate image extension, size, and decode into BGR numpy array.
        """
        suffix = Path(file.filename or "upload.jpg").suffix.lower()
        if suffix not in ALLOWED_IMAGE_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported image format '{suffix}'. Allowed formats: {', '.join(sorted(ALLOWED_IMAGE_EXTENSIONS))}",
            )

        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="Uploaded image file is empty.")

        max_bytes = settings.MAX_IMAGE_SIZE_MB * 1024 * 1024
        if len(raw) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Image file exceeds maximum allowed size of {settings.MAX_IMAGE_SIZE_MB}MB.",
            )

        nparr = np.frombuffer(raw, np.uint8)
        bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise HTTPException(
                status_code=400,
                detail="Could not decode image. The file may be corrupt or in an invalid format.",
            )

        return bgr

    async def detect_image(self, file: UploadFile) -> ImageDetectionResponse:
        """
        Complete image deepfake detection pipeline:
        1. Validate and decode image
        2. Detect faces
        3. Preprocess face crops
        4. Pass through trained model
        5. Aggregate and return prediction
        """
        start_time = time.time()

        bgr = await self.validate_and_decode(file)

        # Detect faces
        detected_faces = face_detector.detect_and_crop(bgr)
        num_faces = len(detected_faces)

        if num_faces == 0:
            raise HTTPException(
                status_code=422,
                detail="No detectable faces found in the uploaded image. Face detection is required for deepfake analysis.",
            )

        # Preprocess all faces
        crops = [f[0] for f in detected_faces]
        batch_tensor = preprocessor.preprocess_batch(crops)

        # Model inference
        preds = model_engine.predict_batch(batch_tensor)

        face_details: list[FaceDetail] = []
        for i, (pred, (_, bbox)) in enumerate(zip(preds, detected_faces)):
            face_details.append(
                FaceDetail(
                    face_index=i + 1,
                    bounding_box=bbox,
                    fake_probability=pred["fake_probability"],
                    real_probability=pred["real_probability"],
                    prediction=pred["prediction"],
                )
            )

        # Image-level verdict:
        # If any face in the image is fake, the image is considered FAKE
        image_fake_prob = max(p["fake_probability"] for p in preds)
        image_real_prob = round(1.0 - image_fake_prob, 4)
        image_prediction = (
            "FAKE" if image_fake_prob >= settings.CLASSIFICATION_THRESHOLD else "REAL"
        )

        processing_time = round(time.time() - start_time, 2)

        return ImageDetectionResponse(
            success=True,
            media_type="image",
            prediction=image_prediction,
            fake_probability=round(image_fake_prob, 4),
            real_probability=round(image_real_prob, 4),
            faces_detected=num_faces,
            faces_analyzed=num_faces,
            processing_time_seconds=processing_time,
            model_version=settings.APP_VERSION,
            details=face_details,
        )


image_detector_service = ImageDetectorService()
