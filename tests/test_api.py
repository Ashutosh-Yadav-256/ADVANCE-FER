"""
Integration Tests for ADVANCE-FER FastAPI Endpoints.
"""

import io
import base64
import pytest
import cv2
import numpy as np
from fastapi.testclient import TestClient
from api.server import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_healthz_endpoint(client):
    """GET /healthz should return 200 OK with healthy status."""
    response = client.get("/healthz")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "device" in data
    assert "emotions" in data


def test_info_endpoint(client):
    """GET /v1/info should return model metadata and labels."""
    response = client.get("/v1/info")
    assert response.status_code == 200
    data = response.json()
    assert "model_architecture" in data
    assert "emotions" in data
    assert len(data["emotions"]) == 7


def test_predict_image_endpoint(client):
    """POST /v1/predict/image with a valid image file."""
    # Create test image in memory
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".jpg", img)
    img_bytes = io.BytesIO(encoded.tobytes())

    response = client.post(
        "/v1/predict/image",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["image_width"] == 300
    assert data["image_height"] == 300
    assert "faces_detected" in data
    assert "latency_ms" in data


def test_predict_base64_endpoint(client):
    """POST /v1/predict/base64 with valid base64 payload."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".png", img)
    b64_str = base64.b64encode(encoded.tobytes()).decode("utf-8")

    response = client.post(
        "/v1/predict/base64",
        json={"image_base64": b64_str, "confidence_threshold": 0.4}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["image_width"] == 200


def test_predict_annotated_endpoint(client):
    """POST /v1/predict/annotated should return JPEG image bytes."""
    img = np.zeros((250, 250, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".jpg", img)
    img_bytes = io.BytesIO(encoded.tobytes())

    response = client.post(
        "/v1/predict/annotated",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert len(response.content) > 0
