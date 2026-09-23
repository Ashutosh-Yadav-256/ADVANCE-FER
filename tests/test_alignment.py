"""
Tests for FaceAligner module.
"""

import pytest
import numpy as np
from preprocessing.alignment import FaceAligner


@pytest.fixture
def aligner():
    return FaceAligner(desired_face_width=224, desired_face_height=224)


def test_aligner_output_shape(aligner):
    """Aligner should return an image of exactly (224, 224, 3)."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_lms = np.zeros((478, 3), dtype=np.float32)
    dummy_lms[468] = [200, 200, 0]  # Left eye
    dummy_lms[473] = [300, 200, 0]  # Right eye

    aligned = aligner.align(img, dummy_lms)
    assert aligned.shape == (224, 224, 3)


def test_aligner_zero_distance_guard(aligner):
    """When eye distance is zero, aligner should not raise ZeroDivisionError."""
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    dummy_lms = np.zeros((478, 3), dtype=np.float32)
    # Eyes at exact same coordinate (dist = 0)
    dummy_lms[468] = [150, 150, 0]
    dummy_lms[473] = [150, 150, 0]

    aligned = aligner.align(img, dummy_lms)
    assert aligned.shape == (224, 224, 3)


def test_aligner_fallback_with_short_landmarks(aligner):
    """When landmarks array has fewer than 474 landmarks, aligner uses fallback indices."""
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    dummy_lms = np.zeros((468, 3), dtype=np.float32)
    dummy_lms[33] = [100, 150, 0]
    dummy_lms[263] = [200, 150, 0]

    aligned = aligner.align(img, dummy_lms)
    assert aligned.shape == (224, 224, 3)
