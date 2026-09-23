"""
Affective AI Agent: Multimodal Emotional Intelligence & Behavioral Reasoning.
Integrates GenAI, LLMs, and Affective Psychology (Russell's Circumplex & Ekman FACS)
to interpret facial emotion trajectories and generate actionable behavioral insights.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class AffectiveAgent:
    """
    GenAI Agent that reasons over emotional, valence, and arousal telemetry
    to produce psychological context, customer sentiment analysis, or empathy coaching.
    """
    def __init__(self, model_name: str = "gpt-4o-mini", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = None

        if self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
                logger.info("AffectiveAgent initialized with OpenAI LLM backend (%s)", model_name)
            except ImportError:
                logger.warning("openai package not installed. Running in heuristic reasoning mode.")
        else:
            logger.info("AffectiveAgent initialized in deterministic Affective Reasoning mode (no API key required).")

    def analyze_emotion(
        self,
        face_data: Dict[str, Any],
        context: str = "general"
    ) -> Dict[str, Any]:
        """
        Analyzes a single face emotion telemetry dictionary and produces psychological insights.

        Args:
            face_data: Dict containing 'dominant_emotion', 'confidence', 'probabilities', 'valence', 'arousal'
            context: Domain scenario (e.g., 'customer_service', 'interview', 'mental_wellness', 'general')

        Returns:
            Dict containing engagement_score, stress_index, interpretation, and recommendation.
        """
        dominant = face_data.get("dominant_emotion", "neutral")
        conf = face_data.get("confidence", 0.5)
        valence = face_data.get("valence", 0.0)
        arousal = face_data.get("arousal", 0.0)
        probs = face_data.get("probabilities", {})

        # Compute psychological indices
        # Stress Index: high arousal + negative valence + fear/angry/disgust
        negative_mass = probs.get("angry", 0) + probs.get("fear", 0) + probs.get("disgust", 0) + probs.get("sad", 0)
        stress_index = round(max(0.0, min(1.0, (arousal * 0.5) + (negative_mass * 0.7) - (valence * 0.3))), 2)

        # Engagement Score: non-neutral activation + positive or alert surprise
        engagement_score = round(max(0.0, min(1.0, abs(arousal) * 0.6 + (1.0 - probs.get("neutral", 0.5)) * 0.4)), 2)

        # If LLM client is available, run prompt chain
        if self.client:
            try:
                system_prompt = (
                    "You are an expert Affective Computing and Behavioral Psychology AI Agent. "
                    "Analyze facial expression telemetry (discrete emotion, confidence, valence [-1, 1], arousal [-1, 1]) "
                    f"in the context of: '{context}'. Provide a concise psychological interpretation and 1 actionable recommendation."
                )
                user_content = (
                    f"Emotion: {dominant} ({conf*100:.1f}%), Valence: {valence:+.2f}, Arousal: {arousal:+.2f}, "
                    f"Stress Index: {stress_index}, Engagement: {engagement_score}"
                )
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    temperature=0.3,
                    max_tokens=200
                )
                interpretation = response.choices[0].message.content.strip()
                return {
                    "context": context,
                    "dominant_emotion": dominant,
                    "stress_index": stress_index,
                    "engagement_score": engagement_score,
                    "agent_reasoning": interpretation,
                    "llm_augmented": True
                }
            except Exception as e:
                logger.warning("LLM completion failed (%s). Falling back to heuristic reasoning.", e)

        # Fallback Deterministic Affective Reasoning Engine
        interpretation, recommendation = self._heuristic_reasoning(dominant, valence, arousal, stress_index, context)

        return {
            "context": context,
            "dominant_emotion": dominant,
            "stress_index": stress_index,
            "engagement_score": engagement_score,
            "interpretation": interpretation,
            "recommendation": recommendation,
            "llm_augmented": False
        }

    @staticmethod
    def _heuristic_reasoning(
        emotion: str,
        valence: float,
        arousal: float,
        stress: float,
        context: str
    ) -> tuple[str, str]:
        """Provides domain-specific affective interpretation without external APIs."""
        if context == "customer_service":
            if emotion in ["angry", "disgust"]:
                return (
                    "Customer exhibits high dissatisfaction and friction signals.",
                    "Proactively offer resolution assistance or escalate to a senior empathy specialist."
                )
            elif emotion == "happy":
                return (
                    "Customer shows high satisfaction and positive brand affinity.",
                    "Reinforce rapport and suggest relevant value-add opportunities."
                )
            else:
                return (
                    "Customer maintains a neutral, receptive baseline.",
                    "Ensure clear communication and concise pacing."
                )
        elif context == "interview":
            if stress > 0.6:
                return (
                    "Candidate is experiencing noticeable cognitive stress or anxiety.",
                    "Ask an open-ended warm-up question to re-establish psychological safety."
                )
            elif emotion == "happy" and valence > 0.3:
                return (
                    "Candidate demonstrates high confidence, comfort, and positive engagement.",
                    "Proceed to in-depth technical inquiry."
                )
            else:
                return (
                    "Candidate is composed and focused on the discussion.",
                    "Continue planned interview cadence."
                )
        else:
            if valence > 0.2:
                return (
                    f"Positive affective state detected ({emotion}) with constructive valence.",
                    "Maintain current interaction flow."
                )
            elif valence < -0.2:
                return (
                    f"Negative affective tension detected ({emotion}) with elevated stress.",
                    "Consider pausing or addressing underlying concerns."
                )
            else:
                return (
                    "Equilibrium state with neutral affective valence.",
                    "No immediate intervention indicated."
                )
