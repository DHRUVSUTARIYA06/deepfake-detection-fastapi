"""
FastAPI Deepfake Detection Application.
Single application providing Image and Video Deepfake Detection APIs.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.router import api_router
from app.config import settings
from app.core.face_detector import face_detector
from app.core.model import model_engine
from app.schemas import ErrorResponse, HealthResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("deepfake_api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.
    Loads the trained model once during startup and keeps it in memory.
    """
    logger.info("Initializing Deepfake Detection Application...")
    try:
        model_engine.load()
        logger.info(
            f"Successfully loaded model using {model_engine.framework} on {model_engine.device}."
        )
        logger.info(f"Model weights file: {model_engine.weights_path}")
    except Exception as exc:
        logger.error(f"Critical error loading model during startup: {exc}")

    yield

    logger.info("Shutting down Deepfake Detection Application...")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Production-ready Deepfake Detection API for Image and Video media. "
        "Powered by an EfficientNet-B0 classifier with automated face cropping."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API v1 router
app.include_router(api_router)


# Global Exception Handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Structured error handler for standard HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail if isinstance(exc.detail, str) else "Request error",
            "detail": exc.detail if not isinstance(exc.detail, str) else None,
            "status_code": exc.status_code,
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Catch-all error handler preventing sensitive stack traces from leaking."""
    logger.exception(f"Unhandled server error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": "Internal server error occurred while processing request.",
            "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR,
        },
    )


# Root & Health Endpoints
@app.get(
    "/",
    summary="Root Welcome & Index",
    tags=["General"],
)
async def root():
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "online",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "model_info": "/api/v1/model-info",
            "detect_image": "/api/v1/detect/image",
            "detect_video": "/api/v1/detect/video",
        },
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health and readiness check",
    tags=["General"],
)
async def health():
    is_ready = model_engine.is_loaded and face_detector.is_ready
    return HealthResponse(
        status="healthy" if is_ready else "degraded",
        ready=is_ready,
        model_loaded=model_engine.is_loaded,
        face_detector_loaded=face_detector.is_ready,
        device=model_engine.device,
        version=settings.APP_VERSION,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=False)
