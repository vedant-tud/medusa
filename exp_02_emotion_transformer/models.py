import torch
import torch.nn as nn
import timm

class DualStreamModel(nn.Module):
    def __init__(self, num_classes=3, dropout_rate=0.5):
        super().__init__()
        
        # Stream 1: Motion (Flow) processed by SwinV2
        self.stream1 = timm.create_model('swinv2_tiny_window8_256', pretrained=True, num_classes=0)

        # Stream 2: Spatial (Faces) processed by ResNet50
        self.stream2 = timm.create_model('resnet50', pretrained=True, num_classes=0)
        
        # Try to infer feature shapes
        try:
            dummy_flow = torch.zeros(2, 3, 256, 256)
            was_training1 = self.stream1.training
            self.stream1.eval()
            with torch.no_grad():
                feat1_size = self.stream1(dummy_flow).shape[1]
            if was_training1:
                self.stream1.train()
        except:
            feat1_size = 768 # standard swin fallback

        try:
            dummy_rgb = torch.zeros(2, 3, 256, 256)
            was_training2 = self.stream2.training
            self.stream2.eval()
            with torch.no_grad():
                feat2_size = self.stream2(dummy_rgb).shape[1]
            if was_training2:
                self.stream2.train()
        except:
            feat2_size = 2048 # standard resnet50 fallback
        
        out_features = feat1_size + feat2_size
            
        self.fusion = nn.Sequential(
            nn.LayerNorm(out_features),
            nn.Dropout(dropout_rate),
            nn.Linear(out_features, num_classes)
        )
        
    def forward(self, x1, x2):
        # x1: motion (flow)
        # x2: spatial (apex face)
        out1 = self.stream1(x1)
        out2 = self.stream2(x2)
        
        fused = torch.cat((out1, out2), dim=1)
        return self.fusion(fused)
