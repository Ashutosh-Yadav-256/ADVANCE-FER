"""
Tests for Flask Microservice Endpoints.
"""

import io
import pytest
import cv2
import numpy as np
from api.flask_app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def test_flask_healthz(client):
    """GET /healthz on Flask app returns 200 OK."""
    response = client.get("/healthz")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert data["framework"] == "Flask"


def test_flask_agent_endpoint(client):
    """POST /v1/agent/analyze on Flask app executes behavioral reasoning."""
    payload = {
        "face_data": {
            "dominant_emotion": "happy",
            "confidence": 0.90,
            "valence": 0.80,
            "arousal": 0.50
        },
        "context": "customer_service"
    }
    response = client.post("/v1/agent/analyze", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert "analysis" in data
    assert data["analysis"]["dominant_emotion"] == "happy"


def test_flask_predict_image(client):
    """POST /v1/predict/image on Flask app handles multipart form-data."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".jpg", img)

    data = {
        "file": (io.BytesIO(encoded.tobytes()), "test.jpg")
    }
    response = client.post("/v1/predict/image", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    res_json = response.get_json()
    assert res_json["success"] is True
    assert res_json["image_width"] == 200
