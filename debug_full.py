print("Debug script starting...", flush=True)

try:
    print("Importing torch...", flush=True)
    import torch
    print("Importing torchvision...", flush=True)
    import torchvision
    print("Importing cv2...", flush=True)
    import cv2
    print("Importing mediapipe...", flush=True)
    import mediapipe
    print("Importing local modules...", flush=True)
    from models.fer_model import FERModel
    from data.fer_dataset import FER2013Dataset
    print("All imports successful!", flush=True)
except Exception as e:
    print(f"CRASHED: {e}", flush=True)
    import traceback
    traceback.print_exc()
