"""
Video processing service:
- Validates uploaded video files
- Extracts uniformly sampled frames
- Coordinates face detection, cropping, and inference
- Aggregates frame and face predictions to final video prediction
- Ensures safe temporary file cleanup
"""

import os
import tempfile
import time
from pathlib import Path
from typing import Any, Tuple

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile

from app.config import settings
from app.core.face_detector import face_detector
from app.core.model import model_engine
from app.core.preprocessor import preprocessor
from app.schemas import FrameDetail, VideoDetectionResponse

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".wmv", ".flv"}


class VideoProcessor:
    def __init__(self):
        pass

    async def save_temp_video(self, file: UploadFile) -> str:
        """
        Validate and save uploaded video to a secure temporary file.
        Returns the absolute path to the temporary file.
        """
        suffix = Path(file.filename or "upload.mp4").suffix.lower()
        if suffix not in ALLOWED_VIDEO_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported video format '{suffix}'. Allowed: {', '.join(sorted(ALLOWED_VIDEO_EXTENSIONS))}",
            )

        # Enforce max size limit while streaming to temp file
        max_bytes = settings.MAX_VIDEO_SIZE_MB * 1024 * 1024
        bytes_written = 0

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                bytes_written += len(chunk)
                if bytes_written > max_bytes:
                    tmp.close()
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
                    raise HTTPException(
                        status_code=413,
                        detail=f"Video file exceeds maximum allowed size of {settings.MAX_VIDEO_SIZE_MB}MB.",
                    )
                tmp.write(chunk)

        if bytes_written == 0:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise HTTPException(status_code=400, detail="Uploaded video file is empty.")

        return tmp_path

    def process_video(
        self, temp_video_path: str, sample_frames_count: int = settings.DEFAULT_SAMPLE_FRAMES
    ) -> VideoDetectionResponse:
        """
        Process the video at temp_video_path:
        1. Open with OpenCV and validate duration and frame count.
        2. Uniformly sample frames.
        3. Detect and crop faces from each sampled frame.
        4. Pass face crops through the model.
        5. Aggregate predictions into frame-level and video-level results.
        """
        start_time = time.time()
        cap = cv2.VideoCapture(temp_video_path)

        try:
            if not cap.isOpened():
                raise HTTPException(
                    status_code=400,
                    detail="Could not open video file. File may be corrupted or in an unsupported codec.",
                )

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

            if total_frames <= 0:
                raise HTTPException(
                    status_code=400, detail="Video contains no readable frames."
                )

            duration = total_frames / fps
            if duration > settings.MAX_VIDEO_DURATION_SECONDS:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Video duration ({duration:.1f}s) exceeds maximum allowed duration "
                        f"of {settings.MAX_VIDEO_DURATION_SECONDS} seconds."
                    ),
                )

            # Cap sample frames to allowed limit
            num_samples = min(sample_frames_count, settings.MAX_SAMPLE_FRAMES, total_frames)
            num_samples = max(num_samples, 1)

            # Uniform sampling indices
            sample_indices = np.linspace(0, total_frames - 1, num_samples, dtype=int).tolist()

            # Read frames
            sampled_frames: list[tuple[int, float, np.ndarray]] = []
            current_frame_idx = 0

            # Set of target indices for fast lookup
            target_set = set(sample_indices)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if current_frame_idx in target_set:
                    timestamp = current_frame_idx / fps
                    sampled_frames.append((current_frame_idx, timestamp, frame))
                current_frame_idx += 1
                if len(sampled_frames) == len(target_set):
                    break

        finally:
            cap.release()

        if not sampled_frames:
            raise HTTPException(
                status_code=400, detail="Failed to extract frames from video."
            )

        # Analyze frames
        frames_analyzed = len(sampled_frames)
        frames_with_faces = 0
        total_faces_analyzed = 0
        frame_details: list[FrameDetail] = []
        valid_frame_results: list[dict[str, Any]] = []

        for frame_idx, timestamp, frame_bgr in sampled_frames:
            detected_faces = face_detector.detect_and_crop(frame_bgr)
            num_faces = len(detected_faces)

            if num_faces == 0:
                continue

            frames_with_faces += 1
            total_faces_analyzed += num_faces

            # Batch preprocess faces in this frame
            crops = [face[0] for face in detected_faces]
            batch_tensor = preprocessor.preprocess_batch(crops)
            face_preds = model_engine.predict_batch(batch_tensor)

            # Frame fake probability is the maximum fake probability among detected faces
            frame_fake_prob = max(fp["fake_probability"] for fp in face_preds)
            frame_real_prob = round(1.0 - frame_fake_prob, 4)
            frame_pred_label = (
                "FAKE" if frame_fake_prob >= settings.CLASSIFICATION_THRESHOLD else "REAL"
            )

            frame_details.append(
                FrameDetail(
                    frame_index=frame_idx,
                    timestamp_seconds=round(timestamp, 2),
                    faces_detected=num_faces,
                    fake_probability=round(frame_fake_prob, 4),
                    real_probability=round(frame_real_prob, 4),
                    prediction=frame_pred_label,
                )
            )

            valid_frame_results.append({
                "fake_probability": frame_fake_prob,
                "real_probability": frame_real_prob,
                "prediction": frame_pred_label,
            })

        # Check if any faces were detected
        if frames_with_faces == 0 or total_faces_analyzed == 0:
            raise HTTPException(
                status_code=422,
                detail="No detectable faces found in the video frames. Face detection is required for deepfake analysis.",
            )

        # Aggregate video results
        # 1. Average fake probability across frames with faces
        video_fake_prob = float(np.mean([r["fake_probability"] for r in valid_frame_results]))
        video_real_prob = round(1.0 - video_fake_prob, 4)

        # 2. Ratio of frames classified as FAKE
        fake_frame_ratio = sum(
            1 for r in valid_frame_results if r["prediction"] == "FAKE"
        ) / len(valid_frame_results)

        # Video prediction verdict
        final_prediction = (
            "FAKE"
            if (video_fake_prob >= settings.CLASSIFICATION_THRESHOLD or fake_frame_ratio >= settings.CLASSIFICATION_THRESHOLD)
            else "REAL"
        )

        processing_time = round(time.time() - start_time, 2)

        return VideoDetectionResponse(
            success=True,
            media_type="video",
            prediction=final_prediction,
            fake_probability=round(video_fake_prob, 4),
            real_probability=round(video_real_prob, 4),
            frames_analyzed=frames_analyzed,
            frames_with_faces=frames_with_faces,
            faces_analyzed=total_faces_analyzed,
            processing_time_seconds=processing_time,
            model_version=settings.APP_VERSION,
            details=frame_details,
        )


video_processor = VideoProcessor()
