"""
Production FastAPI Application for Facial Expression Recognition (ADVANCE-FER).
Exposes RESTful endpoints for single/multi-face emotion classification, health monitoring, and annotated streaming.
"""

# Protobuf / MediaPipe compatibility patch for protobuf >= 3.20 on Python 3.12
import google.protobuf.message_factory as _mf
from google.protobuf import symbol_database as _sym_db
if not hasattr(_mf, 'GetMessageClass'):
    _mf.GetMessageClass = lambda descriptor: _sym_db.Default().GetPrototype(descriptor)

import time
import base64
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Response, status
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import PredictResponse, FaceDetectionResult, Base64PredictRequest, HealthResponse
from deployment.inference import InferenceEngine
from agent.affective_agent import AffectiveAgent
from models.emotion_engine import EMOTION_LABELS

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("advance_fer_api")

start_time = time.time()
engine: InferenceEngine = None
agent: AffectiveAgent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes models and resources on service boot and cleanly shuts down."""
    global engine, agent, start_time
    start_time = time.time()
    logger.info("Initializing ADVANCE-FER Production Engine and Affective Agent...")
    engine = InferenceEngine()
    agent = AffectiveAgent()
    app.state.engine = engine
    app.state.agent = agent
    logger.info("ADVANCE-FER Production Engine successfully loaded and ready for inference.")
    yield
    logger.info("Shutting down ADVANCE-FER API Service.")


app = FastAPI(
    title="ADVANCE-FER Facial Expression Recognition API",
    description="Production-grade, zero-training SOTA Facial Expression Recognition API powered by MediaPipe and ONNX Runtime.",
    version="2.0.0",
    lifespan=lifespan
)

# Enable CORS for web apps and dashboard clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _decode_image_bytes(file_bytes: bytes) -> np.ndarray:
    """Safely decodes raw image bytes into a BGR numpy array."""
    if not file_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty image data provided.")

    np_arr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Could not decode image. Supported formats: JPEG, PNG, WEBP, BMP.")
    return image


@app.get("/", tags=["General"])
async def root():
    """Service status and quick links."""
    return {
        "service": "ADVANCE-FER API",
        "version": "2.0.0",
        "documentation": "/docs",
        "health": "/healthz",
        "endpoints": [
            "POST /v1/predict/image",
            "POST /v1/predict/base64",
            "POST /v1/predict/annotated",
            "GET /healthz",
            "GET /v1/info"
        ]
    }


@app.get("/healthz", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """Kubernetes liveness and readiness probe."""
    if engine is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Inference engine not ready")

    uptime = time.time() - start_time
    providers = engine.emotion_engine.session.get_providers()
    device = "CUDA" if "CUDAExecutionProvider" in providers else "CPU"

    return HealthResponse(
        status="healthy",
        model="ViT-FER-Quantized-ONNX",
        version="2.0.0",
        device=device,
        emotions=EMOTION_LABELS,
        uptime_seconds=round(uptime, 2)
    )


@app.get("/v1/info", tags=["General"])
async def model_info():
    """Returns model architecture, input shapes, and supported emotion classes."""
    return {
        "model_architecture": "Vision Transformer (ViT) Quantized ONNX",
        "emotions": EMOTION_LABELS,
        "input_resolution": [224, 224],
        "affective_space": "Russell's Circumplex Model (Valence & Arousal)",
        "detector": "MediaPipe FaceMesh (478 3D Landmarks)",
        "backend": "ONNX Runtime"
    }


@app.post("/v1/predict/image", response_model=PredictResponse, tags=["Inference"])
async def predict_image(file: UploadFile = File(..., description="JPEG or PNG image file")):
    """
    Analyzes an uploaded image file, detects all human faces, and classifies emotions.
    """
    contents = await file.read()
    image = _decode_image_bytes(contents)
    h, w = image.shape[:2]

    faces_raw, latency_ms = engine.process_frame(image)

    faces_pydantic = [
        FaceDetectionResult(
            face_id=f["face_idx"],
            bbox=f["bbox"],
            dominant_emotion=f["dominant_emotion"],
            confidence=f["confidence"],
            probabilities=f["probabilities"],
            valence=f["valence"],
            arousal=f["arousal"]
        )
        for f in faces_raw
    ]

    return PredictResponse(
        success=True,
        image_width=w,
        image_height=h,
        faces_detected=len(faces_pydantic),
        faces=faces_pydantic,
        latency_ms=round(latency_ms, 2)
    )


@app.post("/v1/predict/base64", response_model=PredictResponse, tags=["Inference"])
async def predict_base64(request: Base64PredictRequest):
    """
    Accepts base64 encoded image string, detects faces, and classifies emotions.
    """
    try:
        raw_b64 = request.image_base64
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]
        image_bytes = base64.b64decode(raw_b64)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid base64 encoding: {e}")

    image = _decode_image_bytes(image_bytes)
    h, w = image.shape[:2]

    faces_raw, latency_ms = engine.process_frame(image)

    faces_pydantic = [
        FaceDetectionResult(
            face_id=f["face_idx"],
            bbox=f["bbox"],
            dominant_emotion=f["dominant_emotion"],
            confidence=f["confidence"],
            probabilities=f["probabilities"],
            valence=f["valence"],
            arousal=f["arousal"]
        )
        for f in faces_raw
    ]

    return PredictResponse(
        success=True,
        image_width=w,
        image_height=h,
        faces_detected=len(faces_pydantic),
        faces=faces_pydantic,
        latency_ms=round(latency_ms, 2)
    )


@app.post("/v1/predict/annotated", tags=["Inference"])
async def predict_annotated(
    file: UploadFile = File(..., description="JPEG or PNG image file"),
    draw_mesh: bool = Query(False, description="Whether to draw full facial landmark mesh")
):
    """
    Directly returns an annotated JPEG image with emotion bounding boxes and confidence badges.
    """
    contents = await file.read()
    image = _decode_image_bytes(contents)

    annotated = engine.predict(image, draw_landmarks=draw_mesh, show_metrics=True)

    success, encoded_jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not success:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to encode annotated image")

    return Response(content=encoded_jpeg.tobytes(), media_type="image/jpeg")


@app.post("/v1/agent/analyze", tags=["Affective AI Agent"])
async def agent_analyze(payload: Dict[str, Any]):
    """
    Invokes the Affective AI Agent to reason over face emotion telemetry,
    computing psychological stress, engagement scores, and actionable empathy coaching.
    """
    if agent is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Affective Agent not initialized")

    face_data = payload.get("face_data")
    context = payload.get("context", "general")

    if not face_data:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing 'face_data' payload")

    analysis = agent.analyze_emotion(face_data, context=context)
    return {"success": True, "analysis": analysis}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=True)
