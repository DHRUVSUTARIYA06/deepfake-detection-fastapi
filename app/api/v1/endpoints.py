"""
API v1 route handlers for Deepfake Detection.
"""

import os
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status

from app.config import settings
from app.core.face_detector import face_detector
from app.core.model import model_engine
from app.schemas import (
    ErrorResponse,
    ImageDetectionResponse,
    ModelInfoResponse,
    VideoDetectionResponse,
)
from app.services.detector_service import image_detector_service
from app.services.video_processor import video_processor

router = APIRouter(prefix="/v1", tags=["Detection"])


@router.post(
    "/detect/image",
    response_model=ImageDetectionResponse,
    summary="Detect deepfakes in an image",
    description="Upload an image (JPG, PNG, WEBP, BMP). Detects faces, crops them, runs deepfake model inference, and returns real/fake probabilities.",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file format or corrupt image"},
        413: {"model": ErrorResponse, "description": "Image file too large"},
        422: {"model": ErrorResponse, "description": "No detectable faces in the image"},
        503: {"model": ErrorResponse, "description": "Model not loaded or unavailable"},
    },
)
async def detect_image(
    file: UploadFile = File(..., description="Image file to analyze (field name: 'file')")
):
    if not model_engine.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready. Please try again in a few moments.",
        )

    return await image_detector_service.detect_image(file)


@router.post(
    "/detect/video",
    response_model=VideoDetectionResponse,
    summary="Detect deepfakes in a video",
    description="Upload a video (MP4, AVI, MOV, WEBM). Uniformly extracts frames, crops faces, runs inference, and aggregates frame/face results into a video-level verdict.",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid format, excessive duration, or corrupt video"},
        413: {"model": ErrorResponse, "description": "Video file too large"},
        422: {"model": ErrorResponse, "description": "No detectable faces across sampled frames"},
        503: {"model": ErrorResponse, "description": "Model not loaded or unavailable"},
    },
)
async def detect_video(
    file: UploadFile = File(..., description="Video file to analyze (field name: 'file')"),
    max_frames: Optional[int] = Query(
        default=settings.DEFAULT_SAMPLE_FRAMES,
        ge=1,
        le=settings.MAX_SAMPLE_FRAMES,
        description="Number of uniformly sampled frames to analyze (default: 20, max: 60)",
    ),
):
    if not model_engine.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready. Please try again in a few moments.",
        )

    temp_path: Optional[str] = None
    try:
        temp_path = await video_processor.save_temp_video(file)
        return video_processor.process_video(
            temp_video_path=temp_path,
            sample_frames_count=max_frames or settings.DEFAULT_SAMPLE_FRAMES,
        )
    finally:
        # Guarantee cleanup even on exceptions
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass


@router.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Get model details and metadata",
    description="Returns metadata about the active deepfake detection model, input dimensions, preprocessing parameters, and training accuracy.",
)
async def get_model_info():
    return ModelInfoResponse(
        model_name=settings.MODEL_NAME,
        model_version=settings.APP_VERSION,
        framework=model_engine.framework,
        weights_format=Path(model_engine.weights_path).suffix if model_engine.weights_path else ".onnx",
        weights_path=str(model_engine.weights_path.name) if model_engine.weights_path else "unknown",
        input_dimensions=[1, 3, settings.IMAGE_SIZE, settings.IMAGE_SIZE],
        input_color_space="RGB",
        normalization={
            "mean": settings.MEAN,
            "std": settings.STD,
            "scale": "1/255.0",
        },
        classes={str(k): v for k, v in settings.CLASS_MAPPING.items()},
        classification_threshold=settings.CLASSIFICATION_THRESHOLD,
        validation_accuracy=model_engine.validation_accuracy,
        face_detector="OpenCV Frontal Face Haar Cascade (with margin padding)",
        device=model_engine.device,
    )
