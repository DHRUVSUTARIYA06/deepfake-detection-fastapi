"""
Automated integration tests for Deepfake Detection API using actual model inference.
"""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"


def test_root_endpoint(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "endpoints" in data
    assert "/docs" in data["endpoints"]["docs"]


def test_health_endpoint(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["ready"] is True
    assert data["model_loaded"] is True
    assert data["face_detector_loaded"] is True


def test_model_info_endpoint(client: TestClient):
    response = client.get("/api/v1/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "EfficientNet-B0"
    assert data["input_dimensions"] == [1, 3, 224, 224]
    assert data["input_color_space"] == "RGB"
    assert "mean" in data["normalization"]
    assert "std" in data["normalization"]
    assert data["classes"]["0"] == "FAKE"
    assert data["classes"]["1"] == "REAL"
    assert data["classification_threshold"] == 0.5


def test_detect_image_real(client: TestClient):
    sample_path = SAMPLES_DIR / "real_face.jpg"
    assert sample_path.exists(), f"Sample file missing: {sample_path}"

    with open(sample_path, "rb") as f:
        response = client.post(
            "/api/v1/detect/image",
            files={"file": ("real_face.jpg", f, "image/jpeg")},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["media_type"] == "image"
    assert data["prediction"] == "REAL"
    assert data["real_probability"] > 0.5
    assert data["faces_detected"] >= 1
    assert data["faces_analyzed"] >= 1
    assert data["processing_time_seconds"] > 0


def test_detect_image_fake(client: TestClient):
    sample_path = SAMPLES_DIR / "fake_face.jpg"
    assert sample_path.exists(), f"Sample file missing: {sample_path}"

    with open(sample_path, "rb") as f:
        response = client.post(
            "/api/v1/detect/image",
            files={"file": ("fake_face.jpg", f, "image/jpeg")},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["media_type"] == "image"
    assert data["prediction"] == "FAKE"
    assert data["fake_probability"] > 0.5
    assert data["faces_detected"] >= 1
    assert data["faces_analyzed"] >= 1


def test_detect_image_no_face_rejected(client: TestClient):
    sample_path = SAMPLES_DIR / "no_face.jpg"
    assert sample_path.exists()

    with open(sample_path, "rb") as f:
        response = client.post(
            "/api/v1/detect/image",
            files={"file": ("no_face.jpg", f, "image/jpeg")},
        )

    assert response.status_code == 422
    data = response.json()
    assert data["success"] is False
    assert "No detectable faces found" in data["error"]


def test_detect_image_invalid_extension(client: TestClient):
    response = client.post(
        "/api/v1/detect/image",
        files={"file": ("document.txt", b"plain text content", "text/plain")},
    )
    assert response.status_code == 400
    data = response.json()
    assert data["success"] is False
    assert "Unsupported image format" in data["error"]


def test_detect_image_corrupt_file(client: TestClient):
    response = client.post(
        "/api/v1/detect/image",
        files={"file": ("corrupt.jpg", b"invalid-binary-data", "image/jpeg")},
    )
    assert response.status_code == 400
    data = response.json()
    assert data["success"] is False
    assert "Could not decode image" in data["error"]


def test_detect_video_real(client: TestClient):
    sample_path = SAMPLES_DIR / "real_video.mp4"
    assert sample_path.exists()

    with open(sample_path, "rb") as f:
        response = client.post(
            "/api/v1/detect/video?max_frames=10",
            files={"file": ("real_video.mp4", f, "video/mp4")},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["media_type"] == "video"
    assert data["prediction"] == "REAL"
    assert data["real_probability"] > 0.5
    assert data["frames_analyzed"] >= 1
    assert data["frames_with_faces"] >= 1
    assert data["faces_analyzed"] >= 1


def test_detect_video_fake(client: TestClient):
    sample_path = SAMPLES_DIR / "fake_video.mp4"
    assert sample_path.exists()

    with open(sample_path, "rb") as f:
        response = client.post(
            "/api/v1/detect/video?max_frames=10",
            files={"file": ("fake_video.mp4", f, "video/mp4")},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["media_type"] == "video"
    assert data["prediction"] == "FAKE"
    assert data["fake_probability"] > 0.5
    assert data["frames_analyzed"] >= 1
    assert data["frames_with_faces"] >= 1
    assert data["faces_analyzed"] >= 1


def test_detect_video_invalid_format(client: TestClient):
    response = client.post(
        "/api/v1/detect/video",
        files={"file": ("bad_video.pdf", b"fake pdf data", "application/pdf")},
    )
    assert response.status_code == 400
    data = response.json()
    assert data["success"] is False
    assert "Unsupported video format" in data["error"]
