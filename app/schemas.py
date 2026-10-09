"""
Pydantic schemas for Deepfake Detection API responses and requests.
"""

from typing import Any, List, Optional
from pydantic import BaseModel, Field


class FaceDetail(BaseModel):
    face_index: int
    bounding_box: list[int] = Field(
        ..., description="[x1, y1, x2, y2] bounding box coordinates"
    )
    fake_probability: float
    real_probability: float
    prediction: str


class ImageDetectionResponse(BaseModel):
    success: bool = True
    media_type: str = "image"
    prediction: str = Field(..., description="Classification result: 'REAL' or 'FAKE'")
    fake_probability: float = Field(..., description="Calculated fake probability between 0.0 and 1.0")
    real_probability: float = Field(..., description="Calculated real probability between 0.0 and 1.0")
    faces_detected: int = Field(..., description="Total number of faces detected in the image")
    faces_analyzed: int = Field(..., description="Number of faces passed through the model")
    processing_time_seconds: float = Field(..., description="Total processing time in seconds")
    model_version: str = Field("v1", description="Model version identifier")
    details: Optional[List[FaceDetail]] = Field(
        None, description="Detailed per-face predictions and bounding boxes"
    )


class FrameDetail(BaseModel):
    frame_index: int
    timestamp_seconds: float
    faces_detected: int
    fake_probability: float
    real_probability: float
    prediction: str


class VideoDetectionResponse(BaseModel):
    success: bool = True
    media_type: str = "video"
    prediction: str = Field(..., description="Aggregated video prediction: 'REAL' or 'FAKE'")
    fake_probability: float = Field(..., description="Aggregated fake probability between 0.0 and 1.0")
    real_probability: float = Field(..., description="Aggregated real probability between 0.0 and 1.0")
    frames_analyzed: int = Field(..., description="Total sampled frames analyzed")
    frames_with_faces: int = Field(..., description="Number of frames with at least one detected face")
    faces_analyzed: int = Field(..., description="Total number of face crops analyzed across all frames")
    processing_time_seconds: float = Field(..., description="Total processing time in seconds")
    model_version: str = Field("v1", description="Model version identifier")
    details: Optional[List[FrameDetail]] = Field(
        None, description="Detailed per-frame analysis"
    )


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    framework: str
    weights_format: str
    weights_path: str
    input_dimensions: list[int]
    input_color_space: str
    normalization: dict[str, Any]
    classes: dict[str, str]
    classification_threshold: float
    validation_accuracy: Optional[float] = None
    face_detector: str
    device: str


class HealthResponse(BaseModel):
    status: str
    ready: bool
    model_loaded: bool
    face_detector_loaded: bool
    device: str
    version: str


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    detail: Optional[str] = None
    status_code: int
