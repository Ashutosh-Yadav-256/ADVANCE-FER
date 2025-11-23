import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, List, Optional

class FaceDetector:
    """
    Wrapper around MediaPipe Face Mesh for robust face detection and landmark extraction.
    """
    def __init__(self, max_num_faces: int = 1, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def process(self, image: np.ndarray) -> Optional[object]:
        """
        Process an image to find faces and landmarks.
        Args:
            image: Input image in BGR format (OpenCV default).
        Returns:
            MediaPipe FaceMesh results object or None if no face found.
        """
        # Convert the BGR image to RGB before processing.
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        return results

    def get_landmarks(self, results, image_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        Extract 468 landmarks from the results.
        Args:
            results: MediaPipe results object.
            image_shape: (height, width) of the image.
        Returns:
            Numpy array of shape (478, 3) containing (x, y, z) coordinates.
            Note: MediaPipe returns 478 landmarks with refine_landmarks=True (includes iris).
        """
        if not results.multi_face_landmarks:
            return None
        
        # Assume single face for now
        face_landmarks = results.multi_face_landmarks[0]
        h, w = image_shape
        
        landmarks = []
        for lm in face_landmarks.landmark:
            landmarks.append([lm.x * w, lm.y * h, lm.z])
            
        return np.array(landmarks)

    def draw_landmarks(self, image: np.ndarray, results) -> np.ndarray:
        """
        Draw landmarks on the image.
        """
        if not results.multi_face_landmarks:
            return image
            
        annotated_image = image.copy()
        for face_landmarks in results.multi_face_landmarks:
            self.mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style())
            self.mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style())
                
        return annotated_image
