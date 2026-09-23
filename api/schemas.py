"""
Pydantic Schemas for ADVANCE-FER Production REST API.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional


class FaceDetectionResult(BaseModel):
    face_id: int = Field(..., description="Zero-based index of the detected face in frame")
    bbox: List[int] = Field(..., description="Bounding box [x, y, width, height] in pixels")
    dominant_emotion: str = Field(..., description="Predicted dominant emotion category")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score of dominant emotion")
    probabilities: Dict[str, float] = Field(..., description="Full probability distribution over 7 emotions")
    valence: float = Field(..., ge=-1.0, le=1.0, description="Pleasantness valence score on Russell's Circumplex")
    arousal: float = Field(..., ge=-1.0, le=1.0, description="Activation arousal score on Russell's Circumplex")


class PredictResponse(BaseModel):
    success: bool = True
    image_width: int
    image_height: int
    faces_detected: int
    faces: List[FaceDetectionResult]
    latency_ms: float = Field(..., description="End-to-end processing latency in milliseconds")


class Base64PredictRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded JPEG/PNG image string")
    confidence_threshold: Optional[float] = Field(0.3, ge=0.0, le=1.0)


class HealthResponse(BaseModel):
    status: str = "healthy"
    model: str = "ViT-FER-Quantized-ONNX"
    version: str = "2.0.0"
    device: str
    emotions: List[str]
    uptime_seconds: float
