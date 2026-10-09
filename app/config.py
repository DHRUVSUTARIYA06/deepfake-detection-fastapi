"""
Application configuration using Pydantic Settings.
Environment variables can override any of these settings.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # App Information
    APP_NAME: str = "Deepfake Detection API"
    APP_VERSION: str = "v1"
    DEBUG: bool = False

    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    MODEL_DIR: Path = BASE_DIR / "models"
    MODEL_ONNX_PATH: Path = MODEL_DIR / "deepfake_efficientnet_b0.onnx"
    MODEL_PT_PATH: Path = MODEL_DIR / "deepfake_efficientnet_b0.pt"

    # Model specifications (from trained checkpoint)
    MODEL_NAME: str = "EfficientNet-B0"
    IMAGE_SIZE: int = 224
    MEAN: list[float] = [0.485, 0.456, 0.406]
    STD: list[float] = [0.229, 0.224, 0.225]
    CLASS_MAPPING: dict[int, str] = {0: "FAKE", 1: "REAL"}
    CLASSIFICATION_THRESHOLD: float = 0.50

    # Face detection
    FACE_PADDING_RATIO: float = 0.20  # 20% margin around face bbox
    MIN_FACE_SIZE: int = 30           # Minimum face box size in pixels

    # Processing limits & Security
    MAX_IMAGE_SIZE_MB: int = 15
    MAX_VIDEO_SIZE_MB: int = 100
    MAX_VIDEO_DURATION_SECONDS: int = 120
    DEFAULT_SAMPLE_FRAMES: int = 20
    MAX_SAMPLE_FRAMES: int = 60

    # CORS
    CORS_ORIGINS: list[str] = ["*"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
