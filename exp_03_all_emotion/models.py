import torch
import torch.nn as nn
import timm

class DualStreamModel(nn.Module):
    def __init__(self, num_classes=3, dropout_rate=0.5, in_channels=3):
        super().__init__()
        
        # Stream 1: Emotion-Trained ConvNeXt
        self.stream1 = timm.create_model("convnext_tiny", pretrained=True, num_classes=0)
        
        if in_channels != 3:
            # We need to manually convert the first layer to match in_channels
            old_conv = self.stream1.stem[0]
            new_conv = nn.Conv2d(in_channels, old_conv.out_channels, kernel_size=old_conv.kernel_size, stride=old_conv.stride, padding=old_conv.padding, bias=(old_conv.bias is not None))
            # Sum the weights along the channel dimension
            with torch.no_grad():
                new_conv.weight.data = old_conv.weight.data.sum(dim=1, keepdim=True)
                if old_conv.bias is not None:
                    new_conv.bias.data = old_conv.bias.data
            self.stream1.stem[0] = new_conv
        
        # Stream 2: Emotion-Trained Vision Transformer (AffectNet)
        self.stream2 = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        
        dummy_flow1 = torch.zeros(2, in_channels, 224, 224)
        dummy_flow2 = torch.zeros(2, 3, 224, 224)
        
        was_training_stream1 = self.stream1.training
        was_training_stream2 = self.stream2.training
        self.stream1.eval()
        self.stream2.eval()
        
        with torch.no_grad():
            feat1_size = self.stream1(dummy_flow1).shape[1]
            feat2_size = self.stream2(dummy_flow2).shape[1]
            
        if was_training_stream1:
            self.stream1.train()
        if was_training_stream2:
            self.stream2.train()
        
        out_features = feat1_size + feat2_size
            
        self.fusion = nn.Sequential(
            nn.LayerNorm(out_features),
            nn.Dropout(0.7),
            nn.Linear(out_features, 64),
            nn.GELU(),
            nn.Dropout(0.7),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x1, x2):
        out1 = self.stream1(x1)
        out2 = self.stream2(x2)
        
        fused = torch.cat((out1, out2), dim=1)
        return self.fusion(fused)
