"""
Production Inference Pipeline for ADVANCE-FER.
Combines MediaPipe Face Mesh, Canonical Alignment, and SOTA ONNX Emotion Engine.
"""

import cv2
import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Any, Optional

from preprocessing.face_detector import FaceDetector
from preprocessing.alignment import FaceAligner
from models.emotion_engine import EmotionEngine

logger = logging.getLogger(__name__)

# Emotion badge color scheme (BGR format)
EMOTION_COLORS = {
    'happy': (46, 204, 113),     # Green
    'surprise': (52, 152, 219),  # Light Blue
    'neutral': (149, 165, 166),  # Gray
    'sad': (155, 89, 182),       # Purple
    'fear': (241, 196, 15),      # Yellow
    'angry': (41, 128, 185),     # Dark Red/Orange
    'disgust': (39, 174, 96)     # Emerald
}


class InferenceEngine:
    """
    Production-ready inference pipeline for real-time video streams and images.
    """
    def __init__(
        self,
        model_path: Optional[str] = None,
        max_num_faces: int = 4,
        confidence_threshold: float = 0.4
    ):
        self.detector = FaceDetector(max_num_faces=max_num_faces)
        self.aligner = FaceAligner()
        self.emotion_engine = EmotionEngine(model_path=model_path)
        self.confidence_threshold = confidence_threshold
        logger.info("InferenceEngine initialized with SOTA ONNX backend.")

    def process_frame(self, frame: np.ndarray) -> Tuple[List[Dict[str, Any]], float]:
        """
        Detects faces and predicts emotions without drawing overlays.

        Args:
            frame: Input BGR image.

        Returns:
            Tuple of (face_results, latency_ms).
        """
        t0 = time.perf_counter()
        if frame is None or frame.size == 0:
            return [], 0.0

        faces = self.detector.extract_faces(frame)
        if not faces:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return [], latency_ms

        aligned_faces = []
        for face in faces:
            aligned = self.aligner.align(frame, face["landmarks"])
            aligned_faces.append(aligned)

        # High-throughput batch inference
        predictions = self.emotion_engine.predict_batch(aligned_faces)

        results = []
        for face, pred in zip(faces, predictions):
            x, y, w, h = face["bbox"]
            results.append({
                "face_idx": face["face_idx"],
                "bbox": [x, y, w, h],
                "dominant_emotion": pred["dominant_emotion"],
                "confidence": pred["confidence"],
                "probabilities": pred["probabilities"],
                "valence": pred["valence"],
                "arousal": pred["arousal"]
            })

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return results, latency_ms

    def predict(
        self,
        frame: np.ndarray,
        draw_landmarks: bool = False,
        show_metrics: bool = True
    ) -> np.ndarray:
        """
        Processes a frame and returns an annotated image with emotion tags and bounding boxes.
        Compatible with the existing Streamlit app and OpenCV video loop.

        Args:
            frame: Input BGR image.
            draw_landmarks: Whether to draw full facial mesh contours.
            show_metrics: Whether to display FPS / latency tag.

        Returns:
            Annotated BGR frame.
        """
        if frame is None or frame.size == 0:
            return frame

        annotated_frame = frame.copy()
        face_results, latency_ms = self.process_frame(frame)

        if draw_landmarks:
            mesh_results = self.detector.process(frame)
            if mesh_results:
                annotated_frame = self.detector.draw_landmarks(annotated_frame, mesh_results)

        # Draw overlays for each face
        for face in face_results:
            x, y, w, h = face["bbox"]
            emotion = face["dominant_emotion"]
            conf = face["confidence"]
            val = face["valence"]
            aro = face["arousal"]

            color = EMOTION_COLORS.get(emotion.lower(), (0, 255, 0))

            # Bounding box
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)

            # Label banner
            label = f"{emotion.capitalize()} ({conf*100:.0f}%)"
            val_aro_label = f"V:{val:+.2f} A:{aro:+.2f}"

            # Calculate text size for background badge
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            badge_y1 = max(0, y - text_h - 10)
            badge_y2 = y
            cv2.rectangle(annotated_frame, (x, badge_y1), (x + text_w + 10, badge_y2), color, -1)
            cv2.putText(annotated_frame, label, (x + 5, y - 5), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

            # Valence & Arousal indicator below box
            va_y = min(annotated_frame.shape[0] - 5, y + h + 18)
            cv2.putText(annotated_frame, val_aro_label, (x, va_y), font, 0.5, color, 1, cv2.LINE_AA)

        if show_metrics and latency_ms > 0:
            fps = 1000.0 / latency_ms
            metrics_text = f"Latency: {latency_ms:.1f}ms | FPS: {fps:.1f} | Faces: {len(face_results)}"
            cv2.putText(annotated_frame, metrics_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        return annotated_frame
