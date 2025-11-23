import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from preprocessing.face_detector import FaceDetector
from preprocessing.alignment import FaceAligner
from PIL import Image

class FER2013Dataset(Dataset):
    """
    Dataset class for FER2013.
    Assumes structure:
    root_dir/
        train/
            angry/
            ...
        test/
            angry/
            ...
    """
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.classes = sorted(os.listdir(self.root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = []
        
        # Load all samples
        for cls_name in self.classes:
            cls_dir = os.path.join(self.root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(cls_dir, img_name), self.class_to_idx[cls_name]))

        # Initialize detector (lazy loading might be better for multiprocessing, but let's try this)
        # Note: MediaPipe is not fork-safe. If num_workers > 0, we need to init inside __getitem__ or worker_init_fn.
        # For simplicity, we will assume num_workers=0 or init in getitem (which is slow).
        # Let's try to init in __init__ and see if it works with num_workers=0.
        self.detector = FaceDetector(min_detection_confidence=0.3) 
        self.aligner = FaceAligner(desired_face_width=224, desired_face_height=224)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image (Grayscale)
        image = cv2.imread(img_path)
        if image is None:
            # Skip or return dummy
            return torch.zeros(3, 224, 224), torch.zeros(478, 3), label, torch.zeros(2)

        # FER2013 is 48x48. Upscale for MediaPipe
        image = cv2.resize(image, (224, 224))
        
        # Detect landmarks
        # Note: MediaPipe expects RGB. cv2.imread is BGR.
        # But FER2013 is grayscale usually saved as RGB/Grayscale.
        # If it was read as BGR, it's fine.
        
        results = self.detector.process(image)
        
        landmarks_tensor = torch.zeros(478, 3)
        
        if results.multi_face_landmarks:
            landmarks_np = self.detector.get_landmarks(results, (224, 224))
            # Align
            image = self.aligner.align(image, landmarks_np)
            landmarks_tensor = torch.tensor(landmarks_np, dtype=torch.float32)
        else:
            # If no face detected, just use the upscaled image and zero landmarks
            pass
            
        # Convert to PIL for transforms
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if self.transform:
            image_tensor = self.transform(image_pil)
        else:
            image_tensor = torch.tensor(np.array(image_pil)).permute(2, 0, 1).float() / 255.0

        # Dummy Valence/Arousal (FER2013 doesn't have it)
        va_target = torch.tensor([0.0, 0.0]) 

        return image_tensor, landmarks_tensor, label, va_target
