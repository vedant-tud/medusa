import unittest
import torch
import numpy as np
import pandas as pd
from utils import flow_to_rgb_heatmap, WeightedFocalLoss
from dataset import CasmeDualSwinDataset

class TestDatasetAndUtils(unittest.TestCase):
    def test_flow_to_rgb_heatmap(self):
        dummy_flow = np.random.rand(2, 64, 64).astype(np.float32)
        heatmap = flow_to_rgb_heatmap(dummy_flow)
        self.assertEqual(heatmap.shape, (3, 64, 64))
        self.assertEqual(heatmap.dtype, np.float32)

    def test_weighted_focal_loss(self):
        loss_fn = WeightedFocalLoss(gamma=2.0)
        logits = torch.randn(4, 3) # 4 samples, 3 classes
        targets = torch.tensor([0, 1, 2, 0])
        loss = loss_fn(logits, targets)
        self.assertTrue(torch.is_tensor(loss))
        self.assertTrue(loss.dim() == 0) # scalar

    def test_casme_dataset_output_shape(self):
        # Create a tiny dummy dataframe
        df = pd.DataFrame({
            "emotion_id": [0],
            "clip_folder": ["dummy_casme_processed/sub01/ep01"],
            "flow_npy": ["dummy_flow.npy"],
            "flow_flip_npy": ["dummy_flow_flip.npy"]
        })
        # Note: the dataset code reading files might fail if files don\'t exist, 
        # but the original code uses a try-except for numpy load, and cv2.imread checks for None.
        
        dataset = CasmeDualSwinDataset(annotations=df, image_size=224, augment=False)
        (flow_tensor, spatial_tensor), label = dataset[0]
        
        self.assertEqual(flow_tensor.shape, (3, 224, 224))
        # Depending on gray or RGB, channel might be 1 or 3. Default is gray in original, so 1.
        self.assertEqual(spatial_tensor.shape, (1, 224, 224)) 
        self.assertEqual(label, 0)

if __name__ == "__main__":
    unittest.main()
