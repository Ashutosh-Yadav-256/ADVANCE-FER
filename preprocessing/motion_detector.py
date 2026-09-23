"""
Motion Detection and Facial Action Dynamics module for ADVANCE-FER.
Provides dual-layer motion analysis:
1. Frame-level Motion Detection: Background frame differencing & optical flow to detect scene activity and gate inference.
2. Facial Micro-Motion Tracking: Landmark velocity analysis to detect head gestures (nodding, shaking) and speech activity.
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional


class MotionDetector:
    """
    High-speed Computer Vision Motion Detector.
    Used for video stream motion gating and facial dynamic analysis.
    """
    def __init__(
        self,
        motion_threshold: int = 25,
        min_motion_area: int = 400,
        history_frames: int = 5
    ):
        self.motion_threshold = motion_threshold
        self.min_motion_area = min_motion_area
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_landmarks: Optional[np.ndarray] = None

    def detect_frame_motion(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Detects pixel-level motion by computing absolute difference against previous frame.

        Args:
            frame: Input BGR image.

        Returns:
            Dict containing:
                - has_motion: bool
                - motion_score: float (0.0 to 1.0)
                - changed_pixels: int
                - motion_bbox: Optional bounding box around highest motion region
        """
        if frame is None or frame.size == 0:
            return {"has_motion": False, "motion_score": 0.0, "changed_pixels": 0}

        # Convert to grayscale and apply Gaussian blur to reduce high-frequency camera noise
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            self.prev_gray = gray
            return {"has_motion": True, "motion_score": 1.0, "changed_pixels": 0, "is_first_frame": True}

        # Absolute difference between current and previous frame
        frame_diff = cv2.absdiff(self.prev_gray, gray)
        _, thresh = cv2.threshold(frame_diff, self.motion_threshold, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Count non-zero changed pixels
        changed_pixels = cv2.countNonZero(thresh)
        total_pixels = frame.shape[0] * frame.shape[1]
        motion_score = round(float(changed_pixels / total_pixels), 4)
        has_motion = changed_pixels >= self.min_motion_area

        # Find largest motion contour
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_bbox = None
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) >= self.min_motion_area:
                x, y, w, h = cv2.boundingRect(largest)
                motion_bbox = [x, y, w, h]

        self.prev_gray = gray

        return {
            "has_motion": has_motion,
            "motion_score": motion_score,
            "changed_pixels": changed_pixels,
            "motion_bbox": motion_bbox
        }

    def detect_facial_micro_motion(
        self,
        current_landmarks: np.ndarray,
        fps: float = 30.0
    ) -> Dict[str, Any]:
        """
        Calculates facial landmark velocity vectors and head displacement dynamics.

        Args:
            current_landmarks: (N, 3) array of current facial landmarks.
            fps: Frame rate of the video source.

        Returns:
            Dict containing:
                - average_velocity: float (pixels per second)
                - max_landmark_velocity: float
                - is_talking: bool (mouth region dynamic velocity)
                - head_motion_type: str ('stable', 'nodding', 'turning', 'tilting')
        """
        if current_landmarks is None or len(current_landmarks) < 468:
            return {"average_velocity": 0.0, "head_motion_type": "unknown", "is_talking": False}

        if self.prev_landmarks is None or len(self.prev_landmarks) != len(current_landmarks):
            self.prev_landmarks = current_landmarks.copy()
            return {"average_velocity": 0.0, "head_motion_type": "calibrating", "is_talking": False}

        dt = 1.0 / max(fps, 1.0)

        # Displacement vector (dx, dy) across all landmarks
        displacement = current_landmarks[:, :2] - self.prev_landmarks[:, :2]
        distances = np.linalg.norm(displacement, axis=1)  # Euclidean pixel distances
        velocities = distances / dt  # Pixels per second

        avg_velocity = float(np.mean(velocities))
        max_velocity = float(np.max(velocities))

        # Mouth landmarks (inner & outer lips): indices 13, 14 (upper/lower inner lip)
        mouth_dist = float(np.linalg.norm(current_landmarks[13, :2] - current_landmarks[14, :2]))
        mouth_velocity = float(np.linalg.norm(displacement[13:15]) / dt)
        is_talking = mouth_velocity > 40.0 and mouth_dist > 5.0

        # Head motion estimation based on nose tip (index 1) displacement
        nose_dx = float(displacement[1, 0])
        nose_dy = float(displacement[1, 1])

        if abs(nose_dy) > 4.0 and abs(nose_dy) > abs(nose_dx) * 1.5:
            head_motion_type = "nodding"
        elif abs(nose_dx) > 4.0 and abs(nose_dx) > abs(nose_dy) * 1.5:
            head_motion_type = "turning"
        elif avg_velocity > 50.0:
            head_motion_type = "active_movement"
        else:
            head_motion_type = "stable"

        self.prev_landmarks = current_landmarks.copy()

        return {
            "average_velocity_px_s": round(avg_velocity, 2),
            "max_landmark_velocity": round(max_velocity, 2),
            "is_talking": is_talking,
            "head_motion_type": head_motion_type
        }

    def reset(self):
        """Resets background reference frame."""
        self.prev_gray = None
        self.prev_landmarks = None
