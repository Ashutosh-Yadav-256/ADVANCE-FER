import torch
import numpy as np
import cv2
from models.fer_model import FERModel
from preprocessing.face_detector import FaceDetector
from preprocessing.alignment import FaceAligner
from utils.grad_cam import GradCAM
import torchvision.transforms as transforms
from PIL import Image

class InferenceEngine:
    """
    Handles model loading and inference.
    """
    def __init__(self, model_path=None, device='cpu'):
        self.device = torch.device(device)
        self.model = FERModel(num_classes=7).to(self.device)
        self.model.eval()
        
        if model_path:
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Loaded model from {model_path}")
            except FileNotFoundError:
                print(f"Model file {model_path} not found. Using random weights.")
        else:
            print("No model path provided. Using random weights.")

        self.detector = FaceDetector()
        self.aligner = FaceAligner()
        
        # Grad-CAM
        # Target layer for ResNet50 is layer4[-1]
        # But we wrapped it in FeatureExtractor.backbone
        # So it is model.visual_extractor.backbone.layer4[-1]
        target_layer = self.model.visual_extractor.backbone.layer4[-1]
        self.grad_cam = GradCAM(self.model, target_layer)
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    def predict(self, frame):
        """
        Run inference on a frame.
        Returns:
            annotated_frame: Frame with bounding boxes and emotion labels.
        """
        results = self.detector.process(frame)
        if not results.multi_face_landmarks:
            return frame
            
        annotated_frame = frame.copy()
        h, w, _ = frame.shape
        
        for face_landmarks in results.multi_face_landmarks:
            # 1. Get Landmarks
            landmarks_np = self.detector.get_landmarks(results, (h, w))
            
            # 2. Align Face
            aligned_face = self.aligner.align(frame, landmarks_np)
            
            # 3. Preprocess for Model
            # Convert to PIL for transforms
            aligned_pil = Image.fromarray(cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB))
            input_tensor = self.transform(aligned_pil).unsqueeze(0).to(self.device)
            
            # Landmarks tensor
            landmarks_tensor = torch.tensor(landmarks_np, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # 4. Inference
            with torch.no_grad():
                logits, va = self.model(input_tensor, landmarks_tensor)
                probs = torch.softmax(logits, dim=1)
                pred_idx = torch.argmax(probs, dim=1).item()
                emotion = self.emotions[pred_idx]
                conf = probs[0, pred_idx].item()
                
            # 5. Explainability (Grad-CAM)
            # We need gradients, so we can't use no_grad for this part if we want backprop
            # But GradCAM handles zero_grad and backward internally.
            # However, we need to enable grad for the input/model temporarily if we were in no_grad mode globally.
            # Since we are in eval mode, we can still compute gradients if requires_grad is True.
            # But standard inference usually doesn't need it.
            # For demo, let's compute it.
            heatmap = self.grad_cam(input_tensor, landmarks_tensor, pred_idx)
            
            # Overlay heatmap on the aligned face (just for visualization)
            # In the main frame, we might just show the label.
            # Or we can draw a mini-map.
            
            # Draw bounding box (approximate from landmarks)
            x_min = int(np.min(landmarks_np[:, 0]))
            y_min = int(np.min(landmarks_np[:, 1]))
            x_max = int(np.max(landmarks_np[:, 0]))
            y_max = int(np.max(landmarks_np[:, 1]))
            
            cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"{emotion} ({conf:.2f})", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        
            # Visualize Valence/Arousal
            val, aro = va[0].tolist()
            cv2.putText(annotated_frame, f"V:{val:.2f} A:{aro:.2f}", (x_min, y_max + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        return annotated_frame
