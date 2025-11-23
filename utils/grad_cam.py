import torch
import torch.nn.functional as F
import cv2
import numpy as np

class GradCAM:
    """
    Grad-CAM implementation for visualizing model attention.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, image, landmarks, class_idx=None):
        """
        Generate Grad-CAM heatmap.
        Args:
            image: (1, 3, H, W) tensor
            landmarks: (1, 478, 3) tensor
            class_idx: Target class index. If None, uses the predicted class.
        """
        self.model.zero_grad()
        
        # Forward pass
        logits, _ = self.model(image, landmarks)
        
        if class_idx is None:
            class_idx = torch.argmax(logits, dim=1).item()
            
        # Backward pass
        score = logits[0, class_idx]
        score.backward()
        
        # Generate heatmap
        gradients = self.gradients
        activations = self.activations
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # Weighted combination of activations
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        
        # ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - torch.min(cam)
        cam = cam / (torch.max(cam) + 1e-8)
        
        return cam.detach().cpu().numpy()[0, 0]

    @staticmethod
    def overlay_heatmap(heatmap, image, alpha=0.5, colormap=cv2.COLORMAP_JET):
        """
        Overlay heatmap on image.
        Args:
            heatmap: (H, W) float array in [0, 1]
            image: (H, W, 3) uint8 array (BGR)
        """
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, colormap)
        
        overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
        return overlay
