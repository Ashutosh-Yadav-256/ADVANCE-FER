import torch
import torch.nn as nn
import torchvision.models as models

class FeatureExtractor(nn.Module):
    """
    CNN Backbone for visual feature extraction.
    Defaults to ResNet50.
    """
    def __init__(self, backbone_name='resnet50', pretrained=True):
        super(FeatureExtractor, self).__init__()
        if backbone_name == 'resnet50':
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            self.out_features = self.backbone.fc.in_features
            # Remove the classification head
            self.backbone.fc = nn.Identity()
        elif backbone_name == 'efficientnet_b0':
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            self.out_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Backbone {backbone_name} not supported yet.")

    def forward(self, x):
        return self.backbone(x)

class LandmarkEncoder(nn.Module):
    """
    MLP to encode 468 3D landmarks.
    Input: (Batch, 478 * 3) or (Batch, 478, 3) flattened.
    """
    def __init__(self, input_dim=478*3, hidden_dim=256, output_dim=128):
        super(LandmarkEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        # Flatten if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.net(x)

class FERModel(nn.Module):
    """
    Advanced FER Model combining Visual and Geometric features.
    """
    def __init__(self, num_classes=7, backbone_name='resnet50'):
        super(FERModel, self).__init__()
        self.visual_extractor = FeatureExtractor(backbone_name=backbone_name)
        self.landmark_encoder = LandmarkEncoder()
        
        # Fusion
        fusion_dim = self.visual_extractor.out_features + 128
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
        
        # Optional: Valence-Arousal Head
        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2) # Valence, Arousal
        )

    def forward(self, image, landmarks):
        """
        Args:
            image: (B, 3, H, W) tensor
            landmarks: (B, 478, 3) tensor
        """
        vis_feats = self.visual_extractor(image)
        geo_feats = self.landmark_encoder(landmarks)
        
        # Concatenate
        combined = torch.cat((vis_feats, geo_feats), dim=1)
        
        logits = self.classifier(combined)
        va = self.regressor(combined)
        
        return logits, va

    def get_features(self, image, landmarks):
        """
        Extract fusion features (before classification).
        """
        vis_feats = self.visual_extractor(image)
        geo_feats = self.landmark_encoder(landmarks)
        combined = torch.cat((vis_feats, geo_feats), dim=1)
        return combined
