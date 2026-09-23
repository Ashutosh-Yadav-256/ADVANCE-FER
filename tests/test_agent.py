"""
Tests for AffectiveAgent module.
"""

import pytest
from agent.affective_agent import AffectiveAgent


@pytest.fixture
def agent():
    return AffectiveAgent()


def test_affective_agent_analysis_schema(agent):
    """Verifies that AffectiveAgent produces structured psychological output."""
    sample_data = {
        "dominant_emotion": "happy",
        "confidence": 0.95,
        "valence": 0.81,
        "arousal": 0.51,
        "probabilities": {"happy": 0.95, "neutral": 0.03, "surprise": 0.02}
    }
    result = agent.analyze_emotion(sample_data, context="customer_service")

    assert "dominant_emotion" in result
    assert result["dominant_emotion"] == "happy"
    assert 0.0 <= result["stress_index"] <= 1.0
    assert 0.0 <= result["engagement_score"] <= 1.0
    assert "interpretation" in result or "agent_reasoning" in result
    assert "recommendation" in result or "agent_reasoning" in result


def test_affective_agent_stress_calculation(agent):
    """High negative valence and angry emotion should generate elevated stress index."""
    negative_sample = {
        "dominant_emotion": "angry",
        "confidence": 0.90,
        "valence": -0.65,
        "arousal": 0.70,
        "probabilities": {"angry": 0.90, "disgust": 0.05, "neutral": 0.05}
    }
    result = agent.analyze_emotion(negative_sample, context="interview")
    assert result["stress_index"] > 0.5
