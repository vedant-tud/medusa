import torch
import torch.nn as nn
import timm

class SingleStreamModel(nn.Module):
    def __init__(self, num_classes=3, dropout_rate=0.5):
        super().__init__()
        
        # Motion (Flow) processed by ResNet50
        self.stream = timm.create_model('resnet50', pretrained=True, num_classes=0)
        
        # Infer feature shape
        try:
            dummy_rgb = torch.zeros(2, 3, 256, 256)
            was_training = self.stream.training
            self.stream.eval()
            with torch.no_grad():
                feat_size = self.stream(dummy_rgb).shape[1]
            if was_training:
                self.stream.train()
        except:
            feat_size = 2048 # standard resnet50 fallback
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(feat_size),
            nn.Dropout(dropout_rate),
            nn.Linear(feat_size, num_classes)
        )
        
    def forward(self, flow_x, spatial_x):
        # We ONLY use the motion (flow) frame, ignoring the spatial apex frame entirely
        features = self.stream(flow_x)
        return self.classifier(features)

# Keeping the old name as an alias so we don't have to rewrite imports
DualStreamModel = SingleStreamModel
