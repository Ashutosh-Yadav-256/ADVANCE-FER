"""
Automated Unit and Integration Tests for ADVANCE-FER MCP Server.
"""

import os
import cv2
import base64
import pytest
import numpy as np

from mcp_server import (
    mcp,
    get_model_status,
    analyze_affective_behavior,
    detect_emotions_from_file,
    detect_emotions_from_base64
)


def test_mcp_tools_registered():
    """Verifies that all 5 required tools are registered on FastMCP."""
    tool_names = [t.name for t in mcp._tool_manager.list_tools()]
    expected_tools = [
        "detect_emotions_from_file",
        "detect_emotions_from_base64",
        "analyze_affective_behavior",
        "capture_webcam_and_detect",
        "get_model_status"
    ]
    for expected in expected_tools:
        assert expected in tool_names, f"Tool '{expected}' not found in MCP tool registry"


def test_get_model_status_tool():
    """Tool get_model_status should return healthy device and model details."""
    status = get_model_status()
    assert status["status"] == "healthy"
    assert "model_architecture" in status
    assert "emotions" in status
    assert len(status["emotions"]) == 7


def test_analyze_affective_behavior_tool():
    """Tool analyze_affective_behavior should execute behavioral reasoning."""
    telemetry = {
        "dominant_emotion": "happy",
        "confidence": 0.94,
        "valence": 0.81,
        "arousal": 0.51,
        "probabilities": {"happy": 0.94, "neutral": 0.04, "surprise": 0.02}
    }
    result = analyze_affective_behavior(telemetry, context="interview")
    assert result["success"] is True
    assert "analysis" in result
    assert result["analysis"]["dominant_emotion"] == "happy"


def test_detect_emotions_from_file_not_found():
    """Tool detect_emotions_from_file handles missing files gracefully."""
    result = detect_emotions_from_file("non_existent_image_12345.jpg")
    assert "error" in result


def test_detect_emotions_from_file_valid(tmp_path):
    """Tool detect_emotions_from_file successfully processes a real image."""
    img_path = str(tmp_path / "test_frame.jpg")
    blank = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.imwrite(img_path, blank)

    result = detect_emotions_from_file(img_path)
    assert result["success"] is True
    assert result["image_width"] == 300
    assert result["image_height"] == 300
    assert "faces_detected" in result


def test_detect_emotions_from_base64():
    """Tool detect_emotions_from_base64 processes encoded image string."""
    blank = np.zeros((200, 200, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".png", blank)
    b64_str = base64.b64encode(encoded.tobytes()).decode("utf-8")

    result = detect_emotions_from_base64(b64_str)
    assert result["success"] is True
    assert result["image_width"] == 200
