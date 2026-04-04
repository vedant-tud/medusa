import torch
from models import DualStreamModel

def test_model():
    # batch size 2, 3 channels, 256x256 (swinv2 expects 256, resnet handles it)
    x1 = torch.randn(2, 3, 256, 256)
    x2 = torch.randn(2, 3, 256, 256)
    
    # Instantiate model
    model = DualStreamModel(num_classes=3)
    model.eval()
    
    # Forward pass
    with torch.no_grad():
        out = model(x1, x2)
    
    assert out.shape == (2, 3), f"Expected shape (2, 3), got {out.shape}"
    print("Test passed: Forward pass successful, output shape:", out.shape)
    
if __name__ == '__main__':
    test_model()
