"""
Face Alignment module for canonical normalization based on facial landmarks.
Guards against zero-division, missing landmarks, and invalid eye geometry.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class FaceAligner:
    """
    Aligns and normalizes face pose to a canonical eye-level view using affine transforms.
    """
    def __init__(
        self,
        desired_left_eye: Tuple[float, float] = (0.35, 0.35),
        desired_face_width: int = 224,
        desired_face_height: int = 224
    ):
        self.desired_left_eye = desired_left_eye
        self.desired_face_width = desired_face_width
        self.desired_face_height = desired_face_height

    def align(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        Aligns the face image based on eye coordinates.

        Args:
            image: Input full frame or face crop (BGR).
            landmarks: (N, 3) numpy array of landmarks (N >= 468).

        Returns:
            Aligned canonical face image of shape (desired_face_height, desired_face_width, 3).
        """
        if image is None or image.size == 0 or landmarks is None or len(landmarks) == 0:
            return cv2.resize(image, (self.desired_face_width, self.desired_face_height)) if image is not None else np.zeros((self.desired_face_height, self.desired_face_width, 3), dtype=np.uint8)

        # Eye indices:
        # MediaPipe iris centers: 468 (left iris center) and 473 (right iris center)
        # Fallback eye outer corners: 33 (left eye outer) and 263 (right eye outer)
        if len(landmarks) >= 474:
            left_eye_center = landmarks[468][:2]
            right_eye_center = landmarks[473][:2]
        elif len(landmarks) >= 264:
            left_eye_center = landmarks[33][:2]
            right_eye_center = landmarks[263][:2]
        else:
            # Insufficient landmarks, return direct resize
            return cv2.resize(image, (self.desired_face_width, self.desired_face_height))

        # Compute angle and distance between eye centers
        dY = float(right_eye_center[1] - left_eye_center[1])
        dX = float(right_eye_center[0] - left_eye_center[0])
        dist = np.sqrt((dX ** 2) + (dY ** 2))

        # Guard against zero-distance / collapsed eyes
        if dist < 1.0:
            return cv2.resize(image, (self.desired_face_width, self.desired_face_height))

        angle = float(np.degrees(np.arctan2(dY, dX)))

        # Desired right eye coordinate based on desired left eye
        desired_right_eye_x = 1.0 - self.desired_left_eye[0]
        desired_dist = (desired_right_eye_x - self.desired_left_eye[0]) * self.desired_face_width
        scale = float(desired_dist / dist)

        # Clamp scale to prevent pathological zooming on edge cases
        scale = max(0.2, min(scale, 5.0))

        # Center point between eyes
        eyes_center = (
            float((left_eye_center[0] + right_eye_center[0]) * 0.5),
            float((left_eye_center[1] + right_eye_center[1]) * 0.5)
        )

        # Affine rotation & scaling matrix
        M = cv2.getRotationMatrix2D(eyes_center, angle, scale)

        # Translation update
        tX = self.desired_face_width * 0.5
        tY = self.desired_face_height * self.desired_left_eye[1]
        M[0, 2] += (tX - eyes_center[0])
        M[1, 2] += (tY - eyes_center[1])

        # Warp image
        output = cv2.warpAffine(
            image,
            M,
            (self.desired_face_width, self.desired_face_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101
        )

        return output
