"""
High-Performance ONNX Emotion Recognition Engine.
Provides sub-15ms facial expression recognition with Russell's Circumplex Valence-Arousal mapping.
"""

import cv2
import numpy as np
import onnxruntime as ort
import logging
from typing import Dict, List, Tuple, Any, Optional

from models.weights_manager import ensure_model_weights

logger = logging.getLogger(__name__)

# Standard 7-class emotion mapping for the ViT FER model
EMOTION_LABELS = ['sad', 'disgust', 'angry', 'neutral', 'fear', 'surprise', 'happy']

# Russell's Circumplex Model of Affect (Valence: Pleasantness, Arousal: Activation)
CIRCUMPLEX_MAPPING = {
    'happy': (0.81, 0.51),
    'surprise': (0.40, 0.67),
    'neutral': (0.00, 0.00),
    'sad': (-0.63, -0.27),
    'fear': (-0.64, 0.60),
    'angry': (-0.43, 0.67),
    'disgust': (-0.60, 0.35)
}


class EmotionEngine:
    """
    Production-grade ONNX Runtime inference engine for facial expression recognition.
    """
    def __init__(
        self,
        model_path: Optional[str] = None,
        providers: Optional[List[str]] = None,
        target_size: Tuple[int, int] = (224, 224)
    ):
        self.model_path = ensure_model_weights(model_path)
        self.target_size = target_size

        if providers is None:
            available = ort.get_available_providers()
            # Prioritize CUDA if available, fallback to CPU
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in available else ['CPUExecutionProvider']

        logger.info("Initializing ONNX InferenceSession with providers: %s", providers)
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.intra_op_num_threads = 4

        self.session = ort.InferenceSession(self.model_path, sess_options=session_options, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.labels = EMOTION_LABELS

    def preprocess(self, face_bgr: np.ndarray) -> np.ndarray:
        """
        Preprocesses a cropped/aligned face image for ViT model input.
        Input: BGR image (H, W, 3)
        Output: Tensor (1, 3, 224, 224) normalized with mean=0.5, std=0.5
        """
        if face_bgr is None or face_bgr.size == 0:
            raise ValueError("Input face image is empty or invalid.")

        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(face_rgb, self.target_size, interpolation=cv2.INTER_LINEAR)

        # Scale to [0.0, 1.0]
        tensor = resized.astype(np.float32) / 255.0

        # Normalize with mean=0.5, std=0.5 -> (tensor - 0.5) / 0.5
        tensor = (tensor - 0.5) / 0.5

        # Transpose from (H, W, C) to (C, H, W)
        tensor = np.transpose(tensor, (2, 0, 1))

        # Add batch dimension: (1, 3, 224, 224)
        return np.expand_dims(tensor, axis=0)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        """Computes numerically stable softmax."""
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    def predict_face(self, face_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Predicts emotion distribution, dominant emotion, and Valence/Arousal for a single face.

        Args:
            face_bgr: Cropped BGR face image.

        Returns:
            Dict containing:
                - dominant_emotion: str
                - confidence: float (0.0 to 1.0)
                - probabilities: Dict[str, float]
                - valence: float (-1.0 to 1.0)
                - arousal: float (-1.0 to 1.0)
        """
        tensor = self.preprocess(face_bgr)
        outputs = self.session.run([self.output_name], {self.input_name: tensor})
        logits = outputs[0][0]
        probs = self._softmax(logits)

        # Build probabilities dictionary
        prob_dict = {label: float(probs[i]) for i, label in enumerate(self.labels)}

        # Dominant emotion
        top_idx = int(np.argmax(probs))
        dominant_emotion = self.labels[top_idx]
        confidence = float(probs[top_idx])

        # Compute continuous Valence & Arousal as expectation over Russell Circumplex
        valence = sum(prob_dict[e] * CIRCUMPLEX_MAPPING[e][0] for e in self.labels)
        arousal = sum(prob_dict[e] * CIRCUMPLEX_MAPPING[e][1] for e in self.labels)

        return {
            "dominant_emotion": dominant_emotion,
            "confidence": round(confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in prob_dict.items()},
            "valence": round(float(valence), 4),
            "arousal": round(float(arousal), 4)
        }

    def predict_batch(self, faces_bgr: List[np.ndarray]) -> List[Dict[str, Any]]:
        """
        Performs high-throughput batched inference on multiple faces simultaneously.

        Args:
            faces_bgr: List of cropped BGR face images.

        Returns:
            List of prediction dictionaries.
        """
        if not faces_bgr:
            return []

        # Batch preprocessing
        tensors = [self.preprocess(face)[0] for face in faces_bgr]
        batch_tensor = np.stack(tensors, axis=0)  # Shape: (B, 3, 224, 224)

        outputs = self.session.run([self.output_name], {self.input_name: batch_tensor})
        logits_batch = outputs[0]  # Shape: (B, 7)
        probs_batch = self._softmax(logits_batch)

        results = []
        for i, probs in enumerate(probs_batch):
            prob_dict = {label: float(probs[j]) for j, label in enumerate(self.labels)}
            top_idx = int(np.argmax(probs))
            dominant_emotion = self.labels[top_idx]
            confidence = float(probs[top_idx])

            valence = sum(prob_dict[e] * CIRCUMPLEX_MAPPING[e][0] for e in self.labels)
            arousal = sum(prob_dict[e] * CIRCUMPLEX_MAPPING[e][1] for e in self.labels)

            results.append({
                "dominant_emotion": dominant_emotion,
                "confidence": round(confidence, 4),
                "probabilities": {k: round(v, 4) for k, v in prob_dict.items()},
                "valence": round(float(valence), 4),
                "arousal": round(float(arousal), 4)
            })

        return results
