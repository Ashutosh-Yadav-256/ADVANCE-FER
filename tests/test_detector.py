"""
Tests for FaceDetector module.
"""

import pytest
import numpy as np
from preprocessing.face_detector import FaceDetector


@pytest.fixture
def detector():
    return FaceDetector(max_num_faces=2)


def test_detector_empty_frame(detector):
    """Empty or None image should not raise errors and return empty list."""
    assert detector.extract_faces(None) == []
    blank = np.zeros((100, 100, 3), dtype=np.uint8)
    assert detector.extract_faces(blank) == []


def test_detector_bbox_calculation(detector):
    """Bbox calculation should correctly handle landmark boundaries with padding."""
    # Synthetic landmarks in middle of a 500x500 frame
    landmarks = np.zeros((478, 3), dtype=np.float32)
    landmarks[:, 0] = np.linspace(100, 200, 478)
    landmarks[:, 1] = np.linspace(100, 200, 478)

    x, y, w, h = detector.get_bbox(landmarks, (500, 500), padding=0.1)
    assert x <= 100
    assert y <= 100
    assert w >= 100
    assert h >= 100
