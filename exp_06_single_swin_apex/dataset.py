import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import cv2
from utils import flow_to_rgb_heatmap

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def _make_transform(augment: bool) -> transforms.Compose:
    return transforms.Compose([transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

class CasmeDualSwinDataset(Dataset):
    def __init__(self, annotations: pd.DataFrame, image_size: int = 224, augment: bool = False, frame_type='apex', frame_channels='gray'):
        self.df = annotations.reset_index(drop=True)
        self.size = image_size
        self.augment = augment
        self.transform = _make_transform(augment)
        self.frame_type = frame_type
        self.frame_channels = frame_channels
        self.eraser = transforms.RandomErasing(p=0.25, scale=(0.02, 0.10)) if augment else None

    def __len__(self): return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        label = int(row["emotion_id"])
        use_flip = self.augment and (torch.rand(1).item() > 0.5)
        
        flow_path = row["flow_flip_npy"] if use_flip else row["flow_npy"]
        folder_orig = row["clip_folder"].replace("casme_raft_processed10", "casme_processed").replace("casme_raft_processed", "casme_processed").replace("casme_tvl1_processed_amp5", "casme_processed")
        frame_path = os.path.join(folder_orig, f"{self.frame_type}.jpg")

        try: flow = np.load(flow_path).astype(np.float32) 
        except: flow = np.zeros((2, self.size, self.size), dtype=np.float32)

        if flow.shape[1] != self.size or flow.shape[2] != self.size:
            flow = cv2.resize(flow.transpose(1, 2, 0), (self.size, self.size)).transpose(2, 0, 1)

        rgb_heatmap = flow_to_rgb_heatmap(flow)
        tensor_flow = self.transform(torch.from_numpy(rgb_heatmap))

        frame_img = cv2.imread(frame_path, cv2.IMREAD_COLOR)
        if frame_img is None:
            frame_img = np.zeros((self.size, self.size, 3), dtype=np.float32)
        else:
            frame_img = cv2.resize(frame_img, (self.size, self.size))
            frame_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
            if use_flip: frame_img = cv2.flip(frame_img, 1)

        if self.frame_channels == 'gray':
            frame_img = cv2.cvtColor(frame_img, cv2.COLOR_RGB2GRAY)
            tensor_spatial = torch.from_numpy(frame_img).unsqueeze(0).float() / 255.0
            tensor_spatial = transforms.Normalize([0.5], [0.5])(tensor_spatial)
        else:
            tensor_spatial = torch.from_numpy(frame_img).permute(2, 0, 1).float() / 255.0
            tensor_spatial = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)(tensor_spatial)

        if self.augment and self.eraser is not None:
            tensor_flow = self.eraser(tensor_flow)
            tensor_spatial = self.eraser(tensor_spatial)

        return (tensor_flow, tensor_spatial), label
