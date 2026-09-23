"""
Robust Face Detection and Landmark Extraction module using MediaPipe FaceMesh.
Provides multi-face support, bounding box calculation, and normalized landmark coordinate handling.
"""

# Protobuf / MediaPipe compatibility patch for protobuf >= 3.20 on Python 3.12
import google.protobuf.message_factory as _mf
from google.protobuf import symbol_database as _sym_db
if not hasattr(_mf, 'GetMessageClass'):
    _mf.GetMessageClass = lambda descriptor: _sym_db.Default().GetPrototype(descriptor)

import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, List, Optional, Dict, Any


class FaceDetector:
    """
    Production wrapper around MediaPipe Face Mesh with OpenCV Haar Cascade fallback.
    Supports multi-face detection, bounding box extraction, and landmark extraction.
    """
    def __init__(
        self,
        max_num_faces: int = 4,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        self.max_num_faces = max_num_faces
        self.mp_face_mesh = None
        self.face_mesh = None
        self.mp_drawing = None
        self.mp_drawing_styles = None
        self.use_mediapipe = False

        # 1. Attempt MediaPipe FaceMesh
        try:
            solutions = getattr(mp, "solutions", None)
            if solutions is None:
                import mediapipe.python.solutions as mp_solutions
                solutions = mp_solutions

            if solutions is not None and hasattr(solutions, "face_mesh"):
                self.mp_face_mesh = solutions.face_mesh
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    max_num_faces=max_num_faces,
                    refine_landmarks=True,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence
                )
                self.mp_drawing = getattr(solutions, "drawing_utils", None)
                self.mp_drawing_styles = getattr(solutions, "drawing_styles", None)
                self.use_mediapipe = True
        except Exception:
            self.use_mediapipe = False
            self.face_mesh = None

        # 2. Resilient OpenCV Haar Cascade Fallback
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.haar_cascade = cv2.CascadeClassifier(cascade_path)
        except Exception:
            self.haar_cascade = None

    def process(self, image: np.ndarray) -> Optional[Any]:
        """
        Process an image with MediaPipe Face Mesh.
        Args:
            image: Input image in BGR format.
        Returns:
            MediaPipe FaceMesh results object or None.
        """
        if image is None or image.size == 0 or not self.use_mediapipe or self.face_mesh is None:
            return None
        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return self.face_mesh.process(image_rgb)
        except Exception:
            return None

    def get_landmarks(
        self,
        results: Any,
        image_shape: Tuple[int, int],
        face_idx: int = 0
    ) -> Optional[np.ndarray]:
        """
        Extract 478 landmarks for a specific face index.
        Args:
            results: MediaPipe results object.
            image_shape: (height, width) of the image.
            face_idx: Index of the face to extract (defaults to 0).
        Returns:
            Numpy array of shape (478, 3) with (x, y, z) coordinates in pixels, or None.
        """
        if not results or not getattr(results, "multi_face_landmarks", None):
            return None

        if face_idx >= len(results.multi_face_landmarks):
            return None

        face_landmarks = results.multi_face_landmarks[face_idx]
        h, w = image_shape[:2]

        landmarks = []
        for lm in face_landmarks.landmark:
            landmarks.append([lm.x * w, lm.y * h, lm.z])

        return np.array(landmarks, dtype=np.float32)

    def get_bbox(
        self,
        landmarks: np.ndarray,
        image_shape: Tuple[int, int],
        padding: float = 0.2
    ) -> Tuple[int, int, int, int]:
        """
        Calculates face bounding box (x, y, w, h) from landmarks with padding.
        """
        h, w = image_shape[:2]
        x_min = float(np.min(landmarks[:, 0]))
        y_min = float(np.min(landmarks[:, 1]))
        x_max = float(np.max(landmarks[:, 0]))
        y_max = float(np.max(landmarks[:, 1]))

        box_w = x_max - x_min
        box_h = y_max - y_min

        pad_x = box_w * padding
        pad_y = box_h * padding

        x1 = max(0, int(x_min - pad_x))
        y1 = max(0, int(y_min - pad_y))
        x2 = min(w, int(x_max + pad_x))
        y2 = min(h, int(y_max + pad_y))

        return x1, y1, max(1, x2 - x1), max(1, y2 - y1)

    def extract_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detects all faces in the frame and returns rich structured metadata for each.

        Returns:
            List of dicts:
                - face_idx: int
                - bbox: (x, y, w, h)
                - landmarks: np.ndarray (478, 3)
                - face_crop: np.ndarray (BGR crop)
        """
        if image is None or image.size == 0:
            return []

        h, w = image.shape[:2]

        # 1. Primary: MediaPipe Face Mesh
        if self.use_mediapipe and self.face_mesh is not None:
            results = self.process(image)
            if results and getattr(results, "multi_face_landmarks", None):
                faces = []
                for idx, face_lms in enumerate(results.multi_face_landmarks):
                    landmarks = self.get_landmarks(results, (h, w), face_idx=idx)
                    if landmarks is None:
                        continue

                    x, y, bw, bh = self.get_bbox(landmarks, (h, w))
                    face_crop = image[y:y+bh, x:x+bw]

                    if face_crop.size == 0:
                        continue

                    faces.append({
                        "face_idx": idx,
                        "bbox": (x, y, bw, bh),
                        "landmarks": landmarks,
                        "face_crop": face_crop
                    })

                if faces:
                    return faces

        # 2. Resilient Fallback: OpenCV Haar Cascade
        if self.haar_cascade is not None:
            try:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                detected = self.haar_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=4,
                    minSize=(30, 30)
                )
                faces = []
                for idx, (x, y, bw, bh) in enumerate(detected[:self.max_num_faces]):
                    landmarks = np.zeros((478, 3), dtype=np.float32)
                    landmarks[:, 0] = x + bw / 2.0
                    landmarks[:, 1] = y + bh / 2.0
                    face_crop = image[y:y+bh, x:x+bw]
                    if face_crop.size == 0:
                        continue
                    faces.append({
                        "face_idx": idx,
                        "bbox": (int(x), int(y), int(bw), int(bh)),
                        "landmarks": landmarks,
                        "face_crop": face_crop
                    })
                return faces
            except Exception:
                return []

        return []

    def draw_landmarks(self, image: np.ndarray, results: Any) -> np.ndarray:
        """Draws facial mesh landmarks on image copy."""
        if not results or not getattr(results, "multi_face_landmarks", None):
            return image

        if not self.mp_drawing or not self.mp_face_mesh:
            return image

        annotated_image = image.copy()
        try:
            for face_landmarks in results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style() if self.mp_drawing_styles else None
                )
        except Exception:
            pass
        return annotated_image
