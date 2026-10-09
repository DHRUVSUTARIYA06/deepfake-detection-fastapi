"""
Router for API v1.
"""

from fastapi import APIRouter
from app.api.v1.endpoints import router as detection_router

api_router = APIRouter(prefix="/api")
api_router.include_router(detection_router)
