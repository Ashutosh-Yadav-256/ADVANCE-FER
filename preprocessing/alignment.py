import cv2
import numpy as np
from typing import Tuple

class FaceAligner:
    """
    Aligns the face based on eye landmarks to a standard canonical view.
    """
    def __init__(self, desired_left_eye: Tuple[float, float] = (0.35, 0.35),
                 desired_face_width: int = 256, desired_face_height: int = 256):
        self.desired_left_eye = desired_left_eye
        self.desired_face_width = desired_face_width
        self.desired_face_height = desired_face_height

    def align(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        Aligns the face image.
        Args:
            image: Input image.
            landmarks: 478x3 numpy array of landmarks.
        Returns:
            Aligned face image.
        """
        # MediaPipe indices for eyes:
        # Left eye (center): 468, Right eye (center): 473
        # Or we can use average of eye contours.
        # Let's use the specific iris landmarks for precision if available, or just the eye corners.
        # Left eye: 33 (inner), 133 (outer) -> center approx
        # Right eye: 362 (inner), 263 (outer) -> center approx
        # But MediaPipe provides iris centers at 468 and 473.
        
        left_eye_center = landmarks[468][:2]
        right_eye_center = landmarks[473][:2]

        # Compute the angle between the eye centers
        dY = right_eye_center[1] - left_eye_center[1]
        dX = right_eye_center[0] - left_eye_center[0]
        angle = np.degrees(np.arctan2(dY, dX))

        # Compute the desired right eye x-coordinate based on the desired x-coordinate of the left eye
        desired_right_eye_x = 1.0 - self.desired_left_eye[0]

        # Determine the scale of the new resulting image by taking the ratio of the distance
        # between eyes in the *current* image to the ratio of distance between eyes in the
        # *desired* image
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desired_dist = (desired_right_eye_x - self.desired_left_eye[0])
        desired_dist *= self.desired_face_width
        scale = desired_dist / dist

        # Compute center (x, y) coordinates (i.e., the median point) between the two eyes in the input image
        eyes_center = ((left_eye_center[0] + right_eye_center[0]) // 2,
                       (left_eye_center[1] + right_eye_center[1]) // 2)

        # Grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyes_center, angle, scale)

        # Update the translation component of the matrix
        tX = self.desired_face_width * 0.5
        tY = self.desired_face_height * self.desired_left_eye[1]
        M[0, 2] += (tX - eyes_center[0])
        M[1, 2] += (tY - eyes_center[1])

        # Apply the affine transformation
        (w, h) = (self.desired_face_width, self.desired_face_height)
        output = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

        return output
