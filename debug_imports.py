import sys
print("Starting debug...", flush=True)

try:
    print("Importing os...", flush=True)
    import os
    print("Success.", flush=True)
except Exception as e:
    print(f"Failed: {e}", flush=True)

try:
    print("Importing torch...", flush=True)
    import torch
    print(f"Success. Version: {torch.__version__}", flush=True)
except Exception as e:
    print(f"Failed: {e}", flush=True)

try:
    print("Importing cv2...", flush=True)
    import cv2
    print(f"Success. Version: {cv2.__version__}", flush=True)
except Exception as e:
    print(f"Failed: {e}", flush=True)

try:
    print("Importing mediapipe...", flush=True)
    import mediapipe
    print("Success.", flush=True)
except Exception as e:
    print(f"Failed: {e}", flush=True)

try:
    print("Importing tqdm...", flush=True)
    from tqdm import tqdm
    print("Success.", flush=True)
except Exception as e:
    print(f"Failed: {e}", flush=True)

print("Debug complete.", flush=True)
