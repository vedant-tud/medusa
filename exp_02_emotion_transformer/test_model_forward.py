import torch
from models import DualStreamModel

def test_model():
    # batch size 2, 3 channels, 224x224
    x1 = torch.randn(2, 3, 224, 224)
    x2 = torch.randn(2, 3, 224, 224)
    
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
