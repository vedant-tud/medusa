import torch
import torch.nn as nn
import timm

class SingleStreamModel(nn.Module):
    def __init__(self, num_classes=3, dropout_rate=0.5):
        super().__init__()
        
        # Spatial (Apex Face) processed by SwinV2
        self.stream = timm.create_model('swinv2_tiny_window8_256', pretrained=True, num_classes=0)
        
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
            feat_size = 768 # standard swinv2_tiny fallback
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(feat_size),
            nn.Dropout(dropout_rate),
            nn.Linear(feat_size, num_classes)
        )
        
    def forward(self, flow_x, spatial_x):
        # We ONLY use the spatial (apex) frame, ignoring the flow frame entirely
        features = self.stream(spatial_x)
        return self.classifier(features)

# Keeping the old name as an alias so we don't have to rewrite 10 different files 
# that import "DualStreamModel".
DualStreamModel = SingleStreamModel
