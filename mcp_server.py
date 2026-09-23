"""
ADVANCE-FER: Production Model Context Protocol (MCP) Server.
Exposes real-time facial expression recognition, multi-face tracking, and affective reasoning
as standard tools and resources for LLMs and AI Agents (Claude Desktop, Antigravity, Cursor).
"""

import os
import sys
import time
import base64
import logging
from typing import Dict, List, Any, Optional

import cv2
import numpy as np
from mcp.server.fastmcp import FastMCP

# Setup logging to stderr (stdio transport requires stdout reserved for JSON-RPC)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr
)
logger = logging.getLogger("advance_fer_mcp")

# Initialize FastMCP Server
mcp = FastMCP("ADVANCE-FER Emotion AI")

# Lazy-loaded singletons
_engine = None
_agent = None
_start_time = time.time()


def get_engine():
    """Lazily loads the unified InferenceEngine."""
    global _engine
    if _engine is None:
        logger.info("Initializing InferenceEngine for MCP Server...")
        from deployment.inference import InferenceEngine
        _engine = InferenceEngine()
    return _engine


def get_agent():
    """Lazily loads the AffectiveAgent."""
    global _agent
    if _agent is None:
        logger.info("Initializing AffectiveAgent for MCP Server...")
        from agent.affective_agent import AffectiveAgent
        _agent = AffectiveAgent()
    return _agent


# ==========================================
# MCP TOOLS
# ==========================================

@mcp.tool()
def detect_emotions_from_file(image_path: str) -> Dict[str, Any]:
    """
    Detects human faces and predicts emotions from a local image file.

    Args:
        image_path: Absolute or relative file path to a JPEG, PNG, or WEBP image.

    Returns:
        Dictionary containing detected faces, bounding boxes [x, y, w, h],
        dominant emotions, confidence scores, probability distributions,
        and continuous Valence/Arousal scores.
    """
    if not os.path.exists(image_path):
        return {"error": f"Image file not found at path: {image_path}"}

    image = cv2.imread(image_path)
    if image is None:
        return {"error": f"Could not decode image at path: {image_path}"}

    engine = get_engine()
    h, w = image.shape[:2]
    faces, latency_ms = engine.process_frame(image)

    return {
        "success": True,
        "image_path": image_path,
        "image_width": w,
        "image_height": h,
        "faces_detected": len(faces),
        "faces": faces,
        "latency_ms": round(latency_ms, 2)
    }


@mcp.tool()
def detect_emotions_from_base64(image_base64: str) -> Dict[str, Any]:
    """
    Detects faces and classifies emotions from a base64-encoded image string.

    Args:
        image_base64: Raw or data-URI base64 encoded image string.

    Returns:
        Dictionary with detected faces, emotions, and valence/arousal.
    """
    try:
        raw_b64 = image_base64
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]
        image_bytes = base64.b64decode(raw_b64)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            return {"error": "Failed to decode base64 into valid image"}
    except Exception as e:
        return {"error": f"Invalid base64 payload: {e}"}

    engine = get_engine()
    h, w = image.shape[:2]
    faces, latency_ms = engine.process_frame(image)

    return {
        "success": True,
        "image_width": w,
        "image_height": h,
        "faces_detected": len(faces),
        "faces": faces,
        "latency_ms": round(latency_ms, 2)
    }


@mcp.tool()
def analyze_affective_behavior(face_data: Dict[str, Any], context: str = "general") -> Dict[str, Any]:
    """
    Analyzes facial emotion telemetry using the Affective AI Agent to generate
    psychological stress indexes, engagement scores, and actionable empathy coaching.

    Args:
        face_data: Face telemetry containing 'dominant_emotion', 'confidence', 'valence', 'arousal', 'probabilities'.
        context: Domain scenario ('customer_service', 'interview', 'mental_wellness', or 'general').

    Returns:
        Structured psychological behavioral assessment and empathy recommendation.
    """
    agent = get_agent()
    analysis = agent.analyze_emotion(face_data, context=context)
    return {
        "success": True,
        "analysis": analysis
    }


@mcp.tool()
def capture_webcam_and_detect(camera_index: int = 0) -> Dict[str, Any]:
    """
    Captures a single live frame from a connected webcam and immediately analyzes facial emotions.

    Args:
        camera_index: Device index of the camera (default is 0 for standard webcam).

    Returns:
        Detected faces, emotions, and affective analytics from the live camera frame.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return {"error": f"Could not access camera device index {camera_index}"}

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return {"error": "Failed to capture frame from webcam"}

    engine = get_engine()
    h, w = frame.shape[:2]
    faces, latency_ms = engine.process_frame(frame)

    return {
        "success": True,
        "source": f"webcam_{camera_index}",
        "image_width": w,
        "image_height": h,
        "faces_detected": len(faces),
        "faces": faces,
        "latency_ms": round(latency_ms, 2)
    }


@mcp.tool()
def get_model_status() -> Dict[str, Any]:
    """
    Returns the operational status, hardware execution provider (CPU/CUDA),
    model architecture, and supported emotion taxonomy.
    """
    engine = get_engine()
    uptime = time.time() - _start_time
    providers = engine.emotion_engine.session.get_providers()

    from models.emotion_engine import CIRCUMPLEX_MAPPING
    return {
        "status": "healthy",
        "model_architecture": "Vision Transformer (ViT) Quantized ONNX",
        "providers": providers,
        "device": "CUDA" if "CUDAExecutionProvider" in providers else "CPU",
        "emotions": engine.emotion_engine.labels,
        "circumplex_coordinates": CIRCUMPLEX_MAPPING,
        "uptime_seconds": round(uptime, 2)
    }


@mcp.tool()
def detect_motion(image_path: str) -> Dict[str, Any]:
    """
    Analyzes an image or video frame for motion changes and pixel activity.

    Args:
        image_path: File path to the image to analyze.

    Returns:
        Dict with has_motion (bool), motion_score, changed_pixels, and motion_bbox.
    """
    if not os.path.exists(image_path):
        return {"error": f"Image file not found: {image_path}"}

    image = cv2.imread(image_path)
    if image is None:
        return {"error": f"Could not read image: {image_path}"}

    global _motion_detector
    if '_motion_detector' not in globals() or _motion_detector is None:
        from preprocessing.motion_detector import MotionDetector
        _motion_detector = MotionDetector()

    res = _motion_detector.detect_frame_motion(image)
    return {"success": True, **res}


# ==========================================
# MCP RESOURCES
# ==========================================

@mcp.resource("fer://taxonomy")
def get_fer_taxonomy() -> str:
    """Returns the affective computing emotion taxonomy and Russell Circumplex model definition."""
    import json
    from models.emotion_engine import EMOTION_LABELS, CIRCUMPLEX_MAPPING
    return json.dumps({
        "taxonomy": "Paul Ekman 7 Basic Facial Expressions",
        "discrete_classes": EMOTION_LABELS,
        "circumplex_space": {
            "dimensions": ["Valence (Pleasantness: -1.0 to +1.0)", "Arousal (Activation: -1.0 to +1.0)"],
            "coordinates": CIRCUMPLEX_MAPPING
        }
    }, indent=2)


@mcp.resource("fer://health")
def get_fer_health() -> str:
    """Returns the real-time health probe and uptime of the FER engine."""
    import json
    return json.dumps({
        "status": "healthy",
        "service": "ADVANCE-FER MCP Server",
        "version": "2.0.0",
        "uptime_seconds": round(time.time() - _start_time, 2)
    }, indent=2)


if __name__ == "__main__":
    # Support --transport sse --port 8001 or default stdio
    import argparse
    parser = argparse.ArgumentParser(description="ADVANCE-FER MCP Server")
    parser.add_argument("--transport", default="stdio", choices=["stdio", "sse"], help="MCP transport mode")
    parser.add_argument("--port", type=int, default=8001, help="Port for SSE transport")
    args = parser.parse_args()

    if args.transport == "sse":
        logger.info("Starting ADVANCE-FER MCP Server with SSE transport on port %d...", args.port)
        mcp.run(transport="sse")
    else:
        logger.info("Starting ADVANCE-FER MCP Server with stdio transport...")
        mcp.run(transport="stdio")
