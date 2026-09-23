"""
Production Flask Microservice for ADVANCE-FER.
Demonstrates multi-framework backend architecture, providing equivalent endpoints
to the FastAPI gateway and integrating the AffectiveAgent and EmotionCache.
"""

import time
import logging
import cv2
import numpy as np
from flask import Flask, request, jsonify

from deployment.inference import InferenceEngine
from agent.affective_agent import AffectiveAgent
from cache.redis_client import EmotionCache

logger = logging.getLogger("advance_fer_flask")

app = Flask(__name__)

# Initialize singletons
engine = InferenceEngine()
agent = AffectiveAgent()
cache = EmotionCache()
start_time = time.time()


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "ADVANCE-FER Flask Service",
        "version": "2.0.0",
        "framework": "Flask 3.x",
        "endpoints": [
            "GET /healthz",
            "GET /v1/info",
            "POST /v1/predict/image",
            "POST /v1/agent/analyze"
        ]
    })


@app.route("/healthz", methods=["GET"])
def healthz():
    uptime = time.time() - start_time
    return jsonify({
        "status": "healthy",
        "framework": "Flask",
        "model": "ViT-FER-Quantized-ONNX",
        "uptime_seconds": round(uptime, 2)
    }), 200


@app.route("/v1/info", methods=["GET"])
def info():
    return jsonify({
        "model_architecture": "Vision Transformer (ViT) Quantized ONNX",
        "framework": "Flask",
        "supported_emotions": engine.emotion_engine.labels,
        "backend": "ONNX Runtime"
    }), 200


@app.route("/v1/predict/image", methods=["POST"])
def predict_image():
    if "file" not in request.files:
        return jsonify({"error": "No file field found in form-data"}), 400

    file = request.files["file"]
    file_bytes = file.read()
    if not file_bytes:
        return jsonify({"error": "Uploaded file is empty"}), 400

    np_arr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"error": "Could not decode image"}), 400

    h, w = image.shape[:2]
    faces_raw, latency_ms = engine.process_frame(image)

    return jsonify({
        "success": True,
        "framework": "Flask",
        "image_width": w,
        "image_height": h,
        "faces_detected": len(faces_raw),
        "faces": faces_raw,
        "latency_ms": round(latency_ms, 2)
    }), 200


@app.route("/v1/agent/analyze", methods=["POST"])
def agent_analyze():
    data = request.get_json(silent=True) or {}
    face_data = data.get("face_data")
    context = data.get("context", "general")

    if not face_data:
        return jsonify({"error": "Missing 'face_data' payload"}), 400

    analysis = agent.analyze_emotion(face_data, context=context)
    return jsonify({
        "success": True,
        "analysis": analysis
    }), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
