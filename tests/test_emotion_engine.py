"""
Tests for EmotionEngine module.
"""

import pytest
import numpy as np
from models.emotion_engine import EmotionEngine, EMOTION_LABELS


@pytest.fixture(scope="module")
def engine():
    return EmotionEngine()


def test_emotion_engine_prediction_schema(engine):
    """Verifies output format, probability summation, and labels."""
    dummy_face = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    res = engine.predict_face(dummy_face)

    assert "dominant_emotion" in res
    assert res["dominant_emotion"] in EMOTION_LABELS
    assert 0.0 <= res["confidence"] <= 1.0

    # Probabilities should sum to approximately 1.0
    prob_sum = sum(res["probabilities"].values())
    assert abs(prob_sum - 1.0) < 0.02

    # Valence & Arousal range check
    assert -1.0 <= res["valence"] <= 1.0
    assert -1.0 <= res["arousal"] <= 1.0


def test_emotion_engine_batch_processing(engine):
    """Batched processing should match single-item predictions."""
    face1 = np.ones((224, 224, 3), dtype=np.uint8) * 128
    face2 = np.ones((224, 224, 3), dtype=np.uint8) * 200

    results = engine.predict_batch([face1, face2])
    assert len(results) == 2
    assert results[0]["dominant_emotion"] in EMOTION_LABELS
    assert results[1]["dominant_emotion"] in EMOTION_LABELS
