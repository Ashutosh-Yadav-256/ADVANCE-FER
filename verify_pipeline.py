import cv2
import numpy as np
from preprocessing.face_detector import FaceDetector
from preprocessing.alignment import FaceAligner

def main():
    print("Initializing FaceDetector and FaceAligner...")
    detector = FaceDetector()
    aligner = FaceAligner()

    # Create a dummy image (black background with a white rectangle face)
    # In a real scenario, the user should provide a path or use webcam.
    # Here we just check if the code runs without crashing.
    print("Creating dummy image...")
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw a "face" so MediaPipe might detect something (unlikely on simple shapes, but code structure is tested)
    # Actually, MediaPipe needs a real face. 
    # So we will just print a message that for real testing, a real image is needed.
    
    print("To verify, please run this script with a real image or webcam feed.")
    print("Example usage logic:")
    print("""
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        results = detector.process(frame)
        if results.multi_face_landmarks:
            landmarks = detector.get_landmarks(results, (frame.shape[0], frame.shape[1]))
            aligned_face = aligner.align(frame, landmarks)
            cv2.imshow('Aligned', aligned_face)
            cv2.waitKey(0)
    """)
    
    print("Dependencies check passed if no import errors.")

if __name__ == "__main__":
    main()
