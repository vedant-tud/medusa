import os
import sys
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# Import local modules
from dataset import CasmeDualSwinDataset
from models import DualStreamModel
from train import load_casme_processed

TARGET_CLASSES = ['Positive', 'Negative', 'Surprise']

def generate_activations(model_path, data_root, out_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DualStreamModel(num_classes=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # The spatial stream is ResNet50
    # Target the last convolutional layer of the ResNet50 for Grad-CAM
    target_layers = [model.stream1.layer4[-1]]

    # Define custom wrappers for each stream
    class FlowWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.current_spatial = None
            
        def forward(self, flow_x):
            return self.model(flow_x, self.current_spatial)

    class SpatialWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.current_flow = None
            
        def forward(self, spatial_x):
            return self.model(self.current_flow, spatial_x)

    def reshape_transform_swin(tensor):
        # Swin outputs shape: [batch, H, W, channels]
        # GradCAM needs: [batch, channels, H, W]
        if tensor.ndim == 4:
            return tensor.permute(0, 3, 1, 2)
        # Fallback if flattened
        elif tensor.ndim == 3:
            B, L, C = tensor.shape
            H = W = int(L**0.5)
            return tensor.transpose(1, 2).reshape(B, C, H, W)
        return tensor

    flow_wrapper = FlowWrapper(model)
    target_layers_flow = [model.stream1.layer4[-1]]
    cam_flow = GradCAM(model=flow_wrapper, target_layers=target_layers_flow)

    spatial_wrapper = SpatialWrapper(model)
    target_layers_spatial = [model.stream2.layers[-1].blocks[-1].norm1]
    cam_spatial = GradCAM(model=spatial_wrapper, target_layers=target_layers_spatial, reshape_transform=reshape_transform_swin)

    df = load_casme_processed(data_root)
    # Shuffle dataframe significantly for diverse samples
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    ds = CasmeDualSwinDataset(df, image_size=256, augment=False, frame_type='apex', frame_channels='rgb')
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Tracking counts for each true/pred combination
    confusion_counts = {(t, p): 0 for t in range(3) for p in range(3)}
    
    # Process samples iteratively until we find 3 of each combination
    for idx in range(len(ds)):
        if all(v >= 3 for v in confusion_counts.values()):
            print("Successfully collected 3 samples for all confusion matrix cells!")
            break
            
        (flow_x, spatial_x), label = ds[idx]
        
        flow_b = flow_x.unsqueeze(0).to(device)
        spatial_b = spatial_x.unsqueeze(0).to(device)
        label_id = label if isinstance(label, int) else label.item()
        
        # Determine prediction first
        with torch.no_grad():
            logits = model(flow_b, spatial_b)
            pred_id = logits.argmax(dim=-1).item()
            
        # Check if we already have 3 for this true/pred pair
        if confusion_counts[(label_id, pred_id)] >= 3:
            continue
            
        confusion_counts[(label_id, pred_id)] += 1
        sample_num = confusion_counts[(label_id, pred_id)]
        
        # Set parameters for the wrappers
        flow_wrapper.current_spatial = spatial_b
        spatial_wrapper.current_flow = flow_b
        
        # Generate CAM masks
        grayscale_cam_flow = cam_flow(input_tensor=flow_b, targets=None)[0, :]
        grayscale_cam_spatial = cam_spatial(input_tensor=spatial_b, targets=None)[0, :]
        
        # Un-normalize images
        img_rgb = spatial_x.permute(1, 2, 0).cpu().numpy()
        img_flow = flow_x.permute(1, 2, 0).cpu().numpy()
        
        mean_rgb = np.array([0.485, 0.456, 0.406])
        std_rgb = np.array([0.229, 0.224, 0.225])
        img_rgb = std_rgb * img_rgb + mean_rgb
        img_rgb = np.clip(img_rgb, 0, 1)

        # Flow images might use different normalization, but assuming same for display
        img_flow = std_rgb * img_flow + mean_rgb
        img_flow = np.clip(img_flow, 0, 1)        
        
        # Create the overlaid image
        vis_flow = show_cam_on_image(img_flow, grayscale_cam_flow, use_rgb=True)
        vis_spatial = show_cam_on_image(img_rgb, grayscale_cam_spatial, use_rgb=True)
        
        fig, ax = plt.subplots(2, 3, figsize=(15, 10))
        
        # Top Row: Spatial (RGB Apex)
        ax[0, 0].imshow(img_rgb)
        ax[0, 0].set_title(f"Stream 2 (Spatial/Face)\nTrue: {TARGET_CLASSES[label_id]} | Pred: {TARGET_CLASSES[pred_id]}")
        ax[0, 0].axis('off')
        
        ax[0, 1].imshow(grayscale_cam_spatial, cmap='jet')
        ax[0, 1].set_title("SwinV2 Spatial Map")
        ax[0, 1].axis('off')
        
        ax[0, 2].imshow(vis_spatial)
        ax[0, 2].set_title("Spatial Overlay")
        ax[0, 2].axis('off')
        
        # Bottom Row: Flow (Optical Flow)
        ax[1, 0].imshow(img_flow)
        ax[1, 0].set_title("Stream 1 (Optical Flow)")
        ax[1, 0].axis('off')
        
        ax[1, 1].imshow(grayscale_cam_flow, cmap='jet')
        ax[1, 1].set_title("ResNet50 Flow Map")
        ax[1, 1].axis('off')
        
        ax[1, 2].imshow(vis_flow)
        ax[1, 2].set_title("Flow Overlay")
        ax[1, 2].axis('off')
        
        out_path = os.path.join(out_dir, f"CM_True{TARGET_CLASSES[label_id]}_Pred{TARGET_CLASSES[pred_id]}_Sample{sample_num}.png")
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()
        print(f"Saved {out_path} ({idx}/{len(ds)})")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='save_models/best_model_fold1.pth')
    parser.add_argument('--data', type=str, default='/scratch/smiyyapuram/medusa/casme_tvl1_processed_amp5')
    parser.add_argument('--out', type=str, default='heatmaps')
    args = parser.parse_args()
    
    generate_activations(args.model, args.data, args.out)