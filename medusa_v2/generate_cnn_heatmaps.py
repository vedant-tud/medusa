import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from trainKfold_scl_cnn import DualCnnMER, CasmeDualCNNDataset, load_casme_processed

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, flow_x, spatial_x, class_idx=None):
        self.model.eval()
        self.model.zero_grad()
        
        logits = self.model(flow_x, spatial_x)
        if class_idx is None:
            class_idx = logits.argmax(dim=-1).item()
            
        score = logits[0, class_idx]
        score.backward()
        
        bl, c, h, w = self.gradients.size()
        weights = torch.mean(self.gradients, dim=(2, 3))[0]
        
        cam = torch.zeros((h, w), dtype=torch.float32, device=self.activations.device)
        activations = self.activations[0]
        
        for i, w_i in enumerate(weights):
            cam += w_i * activations[i]
            
        cam = F.relu(cam)
        cam = cam - torch.min(cam)
        if torch.max(cam) > 0:
            cam = cam / torch.max(cam)
            
        return cam.cpu().numpy(), class_idx

class IntermediateFeatures:
    def __init__(self, layers_dict):
        self.features = {}
        for name, layer in layers_dict.items():
            layer.register_forward_hook(self.create_hook(name))
            
    def create_hook(self, name):
        def hook(module, input, output):
            # Average across channels to get a spatial activity map for the block
            feat_map = torch.mean(output.detach(), dim=1)[0]
            feat_map = feat_map - torch.min(feat_map)
            if torch.max(feat_map) > 0:
                feat_map = feat_map / torch.max(feat_map)
            self.features[name] = feat_map.cpu().numpy()
        return hook

def overlay_heatmap(image_np, cam, alpha=0.5, colormap=cv2.COLORMAP_JET):
    cam = cv2.resize(cam, (image_np.shape[1], image_np.shape[0]))
    cam = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(cam, colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    if image_np.dtype != np.uint8:
        image_np = np.uint8(255 * image_np)
        
    overlay = cv2.addWeighted(image_np, 1 - alpha, heatmap, alpha, 0)
    return overlay, heatmap

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='save_dual_cnn/best_scl_dual_cnn_fold6_apex_rgb.pth')
    parser.add_argument('--frame_type', type=str, default='apex', choices=['apex', 'onset'])
    parser.add_argument('--channels', type=str, default='rgb', choices=['gray', 'rgb'])
    parser.add_argument('--data_root', type=str, default='../casme_raft_processed10')
    parser.add_argument('--out_dir', type=str, default='cnn_heatmaps_blocks')
    parser.add_argument('--num_samples', type=int, default=5)
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    spatial_chans = 1 if args.channels == 'gray' else 3
    model = DualCnnMER(num_classes=3, spatial_chans=spatial_chans).to(device)
    state = torch.load(args.weights, map_location=device, weights_only=False)
    if 'model' in state: 
        state = state['model']
    elif 'model_state_dict' in state: 
        state = state['model_state_dict']
    model.load_state_dict(state)
    model.eval()
    
    # 1. Setup Grad-CAM for the last layers
    gradcam_sp = GradCAM(model, model.stream_spatial.features[3])
    gradcam_mo = GradCAM(model, model.stream_motion.features[3])
    
    # 2. Setup Intermediate Activation hooks for every block
    inter_layers = {
        'sp_b1': model.stream_spatial.features[0],
        'sp_b2': model.stream_spatial.features[1],
        'sp_b3': model.stream_spatial.features[2],
        'sp_b4': model.stream_spatial.features[3],
        'mo_b1': model.stream_motion.features[0],
        'mo_b2': model.stream_motion.features[1],
        'mo_b3': model.stream_motion.features[2],
        'mo_b4': model.stream_motion.features[3],
    }
    feat_collector = IntermediateFeatures(inter_layers)
    
    df = load_casme_processed(args.data_root)
    dataset = CasmeDualCNNDataset(df, image_size=112, augment=False, frame_type=args.frame_type, frame_channels=args.channels)
    
    classes = {0: 'positive', 1: 'negative', 2: 'surprise'}
    samples_done = {0: 0, 1: 0, 2: 0}
    
    for idx in range(len(dataset)):
        if all(v >= args.num_samples for v in samples_done.values()): break
            
        (flow_x, spatial_x), label_idx = dataset[idx]
        if samples_done[label_idx] >= args.num_samples: continue
            
        flow_x = flow_x.unsqueeze(0).to(device)
        spatial_x = spatial_x.unsqueeze(0).to(device)
        
        cam_sp, pred_idx = gradcam_sp.generate(flow_x, spatial_x, class_idx=None)
        cam_mo, _ = gradcam_mo.generate(flow_x, spatial_x, class_idx=pred_idx)
        
        sp_img = spatial_x[0].cpu().numpy()
        if args.channels == 'gray':
            sp_img = sp_img[0]
            sp_img = (sp_img * 0.5 + 0.5)
            sp_img = np.stack([sp_img]*3, axis=-1)
        else:
            sp_img = np.transpose(sp_img, (1, 2, 0))
            sp_img = (sp_img * 0.5 + 0.5)
        sp_img = np.clip(sp_img, 0, 1)
        
        fl_img = flow_x[0].cpu().numpy()
        fl_u, fl_v = fl_img[0] * 0.5 + 0.5, fl_img[1] * 0.5 + 0.5
        fl_mag = np.sqrt(fl_u**2 + fl_v**2)
        fl_mag = fl_mag / (np.max(fl_mag) + 1e-5)
        fl_vis = np.stack([fl_mag]*3, axis=-1)
        
        # Overlay Heatmaps
        gc_sp_overlay, _ = overlay_heatmap(sp_img, cam_sp)
        gc_mo_overlay, _ = overlay_heatmap(fl_vis, cam_mo)
        
        b1_sp_overlay, _ = overlay_heatmap(sp_img, feat_collector.features['sp_b1'])
        b2_sp_overlay, _ = overlay_heatmap(sp_img, feat_collector.features['sp_b2'])
        b3_sp_overlay, _ = overlay_heatmap(sp_img, feat_collector.features['sp_b3'])
        b4_sp_overlay, _ = overlay_heatmap(sp_img, feat_collector.features['sp_b4'])
        
        b1_mo_overlay, _ = overlay_heatmap(fl_vis, feat_collector.features['mo_b1'])
        b2_mo_overlay, _ = overlay_heatmap(fl_vis, feat_collector.features['mo_b2'])
        b3_mo_overlay, _ = overlay_heatmap(fl_vis, feat_collector.features['mo_b3'])
        b4_mo_overlay, _ = overlay_heatmap(fl_vis, feat_collector.features['mo_b4'])

        # Create Plot
        fig, axes = plt.subplots(2, 6, figsize=(24, 8))
        
        # Spatial Stream row
        axes[0,0].imshow(sp_img)
        axes[0,0].set_title("Input Spatial")
        axes[0,0].axis('off')
        
        axes[0,1].imshow(b1_sp_overlay)
        axes[0,1].set_title("SP Block 1 Map")
        axes[0,1].axis('off')
        
        axes[0,2].imshow(b2_sp_overlay)
        axes[0,2].set_title("SP Block 2 Map")
        axes[0,2].axis('off')
        
        axes[0,3].imshow(b3_sp_overlay)
        axes[0,3].set_title("SP Block 3 Map")
        axes[0,3].axis('off')
        
        axes[0,4].imshow(b4_sp_overlay)
        axes[0,4].set_title("SP Block 4 Map")
        axes[0,4].axis('off')
        
        axes[0,5].imshow(gc_sp_overlay)
        axes[0,5].set_title("SP Grad-CAM")
        axes[0,5].axis('off')
        
        # Motion Stream row
        axes[1,0].imshow(fl_vis)
        axes[1,0].set_title("Input Motion Flow")
        axes[1,0].axis('off')
        
        axes[1,1].imshow(b1_mo_overlay)
        axes[1,1].set_title("MO Block 1 Map")
        axes[1,1].axis('off')
        
        axes[1,2].imshow(b2_mo_overlay)
        axes[1,2].set_title("MO Block 2 Map")
        axes[1,2].axis('off')
        
        axes[1,3].imshow(b3_mo_overlay)
        axes[1,3].set_title("MO Block 3 Map")
        axes[1,3].axis('off')
        
        axes[1,4].imshow(b4_mo_overlay)
        axes[1,4].set_title("MO Block 4 Map")
        axes[1,4].axis('off')
        
        axes[1,5].imshow(gc_mo_overlay)
        axes[1,5].set_title("MO Grad-CAM")
        axes[1,5].axis('off')
        
        true_label = classes[label_idx]
        pred_label = classes[pred_idx]
        plt.suptitle(f"True: {true_label} | Pred: {pred_label} (Idx: {idx})")
        plt.tight_layout()
        
        out_name = os.path.join(args.out_dir, f"blocks_sample_{idx}_true{true_label}_pred{pred_label}.png")
        plt.savefig(out_name)
        plt.close()
        
        samples_done[label_idx] += 1
        print(f"Saved {out_name}")

if __name__ == "__main__":
    main()
