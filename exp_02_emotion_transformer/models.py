import torch
import torch.nn as nn
import timm

class DualStreamModel(nn.Module):
    def __init__(self, num_classes=3, dropout_rate=0.5):
        super().__init__()
        
        # Stream 1: Face-Trained Backbone (Facenet-PyTorch or Timm Resnet50 Fallback)
        try:
            from facenet_pytorch import InceptionResnetV1
            self.stream1 = InceptionResnetV1(classify=False, pretrained='vggface2')
        except Exception as e:
            print("Failed to load Facenet PyTorch, falling back to timm resnet50", e)
            
            self.stream1 = timm.create_model('resnet50', pretrained=True, num_classes=0)
            
        # Stream 2: Emotion-Trained Vision Transformer (AffectNet)
        self.stream2 = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        
        dummy_flow = torch.zeros(2, 3, 224, 224)
        was_training_stream2 = self.stream2.training
        self.stream2.eval()
        with torch.no_grad():
            feat2_size = self.stream2(dummy_flow).shape[1]
        if was_training_stream2:
            self.stream2.train()
        
        # Approximate feature size 1
        try:
            # First map to correct device to prevent out of bounds crashes during shape inference if on GPU
            dummy_rgb = torch.zeros(2, 3, 224, 224)
            was_training_stream1 = self.stream1.training
            self.stream1.eval()
            with torch.no_grad():
                feat1_size = self.stream1(dummy_rgb).shape[1]
            if was_training_stream1:
                self.stream1.train()
        except:
            if hasattr(self.stream1, 'fc'):
                feat1_size = self.stream1.fc.in_features
            else:
                feat1_size = 2048 # standard resnet50 fallback
            
        out_features = feat1_size + feat2_size
            
        self.fusion = nn.Sequential(
            nn.LayerNorm(out_features),
            nn.Dropout(dropout_rate),
            nn.Linear(out_features, num_classes)
        )
        
    def forward(self, x1, x2):
        # x1: spatial (apex), x2: motion (flow)
        out1 = self.stream1(x1)
        if not isinstance(out1, torch.Tensor):
            out1 = out1[0]
        out2 = self.stream2(x2)
        
        fused = torch.cat((out1, out2), dim=1)
        return self.fusion(fused)
