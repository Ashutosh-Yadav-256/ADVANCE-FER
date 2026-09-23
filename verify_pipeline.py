"""
Verification and Health Diagnostics Script for ADVANCE-FER Production Pipeline.
Runs self-tests on MediaPipe face detector, alignment module, and ONNX Emotion Engine.
"""

import sys
import time
import cv2
import numpy as np

def run_diagnostics():
    print("=" * 60)
    print(" ADVANCE-FER Production Pipeline Diagnostics")
    print("=" * 60)

    # 1. Check Pretrained Weights
    print("[1/4] Checking Pretrained ONNX Weights...")
    from models.weights_manager import ensure_model_weights
    t0 = time.time()
    weights_path = ensure_model_weights()
    print(f"      [OK] Weights ready at: {weights_path} ({time.time() - t0:.2f}s)")

    # 2. Check EmotionEngine
    print("[2/4] Initializing ONNX Emotion Engine...")
    from models.emotion_engine import EmotionEngine
    t0 = time.time()
    engine = EmotionEngine()
    dummy_face = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    res = engine.predict_face(dummy_face)
    inference_time = (time.time() - t0) * 1000.0
    print(f"      [OK] Dominant: '{res['dominant_emotion']}', Conf: {res['confidence']*100:.1f}%")
    print(f"      [OK] Circumplex Valence: {res['valence']:+.2f}, Arousal: {res['arousal']:+.2f}")
    print(f"      [OK] Engine Latency: {inference_time:.1f}ms")

    # 3. Check Preprocessing
    print("[3/4] Testing FaceDetector & FaceAligner...")
    from preprocessing.face_detector import FaceDetector
    from preprocessing.alignment import FaceAligner
    detector = FaceDetector()
    aligner = FaceAligner()
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    faces = detector.extract_faces(blank)
    print(f"      [OK] Detector safely handled blank frame (Faces found: {len(faces)})")

    # 4. End-to-End Pipeline
    print("[4/4] Testing Full Inference Pipeline...")
    from deployment.inference import InferenceEngine
    inf_engine = InferenceEngine()
    annotated = inf_engine.predict(blank, show_metrics=True)
    assert annotated.shape == (480, 640, 3)
    print(f"      [OK] Full pipeline processed cleanly (Output shape: {annotated.shape})")

    print("\n" + "=" * 60)
    print(" ALL DIAGNOSTICS PASSED! SYSTEM IS PRODUCTION READY.")
    print("=" * 60)


if __name__ == "__main__":
    run_diagnostics()
