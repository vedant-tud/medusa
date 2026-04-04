import torch
import torch.nn as nn
import timm

class DualStreamModel(nn.Module):
    def __init__(self, num_classes=3, dropout_rate=0.5):
        super().__init__()
        
        # Stream 1: Timm Resnet50 for flow (motion)
        self.stream1 = timm.create_model('resnet50', pretrained=True, num_classes=0)
            
        # Note: insightface returns embedding of size 512 normally
        
        # Stream 2: swinv2_tiny_window8_256 for spatial (face)
        self.stream2 = timm.create_model('swinv2_tiny_window8_256', pretrained=True, num_classes=0)
        
        # Try to infer feature shapes
        try:
            dummy_rgb = torch.zeros(2, 3, 256, 256)
            was_training = self.stream1.training
            self.stream1.eval()
            with torch.no_grad():
                feat1_size = self.stream1(dummy_rgb).shape[1]
            if was_training:
                self.stream1.train()
        except Exception as e:
            print("Failed to infer stream1 shape", e)
            if hasattr(self.stream1, 'fc'):
                feat1_size = self.stream1.fc.in_features
            else:
                feat1_size = 2048 # standard resnet50 fallback

        dummy_flow = torch.zeros(2, 3, 256, 256)
        was_training_stream2 = self.stream2.training
        self.stream2.eval()
        with torch.no_grad():
            feat2_size = self.stream2(dummy_flow).shape[1]
        if was_training_stream2:
            self.stream2.train()
        
        out_features = feat1_size + feat2_size
            
        self.fusion = nn.Sequential(
            nn.LayerNorm(out_features),
            nn.Dropout(dropout_rate),
            nn.Linear(out_features, num_classes)
        )
        
    def forward(self, x1, x2):
        # x1: flow, x2: spatial RGB. 
        out1 = self.stream1(x1)
            
        # out1 from insightface might need flattening if it returns namedtuple or similar depending on loaded model
        if not isinstance(out1, torch.Tensor):
            out1 = out1[0] # Try getting first element if list/tuple
            
        out2 = self.stream2(x2)
        
        fused = torch.cat((out1, out2), dim=1)
        return self.fusion(fused)
