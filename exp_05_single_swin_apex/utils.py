import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Optional

def flow_to_rgb_heatmap(flow_npy: np.ndarray) -> np.ndarray:
    flow = flow_npy.transpose(1, 2, 0)
    u, v = flow[..., 0], flow[..., 1]
    mag, ang = cv2.cartToPolar(u, v)
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    max_mag = np.max(mag)
    if max_mag > 1e-5: hsv[..., 2] = (mag / max_mag) * 255
    else: hsv[..., 2] = 0
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)

class WeightedFocalLoss(nn.Module):
    def __init__(self, weight: Optional[torch.Tensor] = None, gamma: float = 3.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss_unweighted = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss_unweighted)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss_unweighted
        if self.weight is not None:
            focal_loss = focal_loss * self.weight[targets]
        return focal_loss.mean()
