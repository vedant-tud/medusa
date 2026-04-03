"""
train_dual_swin_flexible.py
Dual-Stream Swin TransformerMER
Stream 1: Optical Flow (HSV->RGB, 3-channels)
Stream 2: Frame Image (Apex or Onset | Gray or RGB)
"""

import os

# --- Set Local Caching & Load .env ---
env_path = os.path.join(os.getcwd(), ".env")
if os.path.exists(env_path):
    with open(env_path, "r") as f:
        for line in f:
            if line.strip() and not line.startswith("#") and "=" in line:
                key, val = line.strip().split("=", 1)
                os.environ[key] = val

os.environ["TORCH_HOME"] = "/scratch/smiyyapuram/medusa/.cache/torch"
os.environ["HF_HOME"] = "/scratch/smiyyapuram/medusa/.cache/huggingface"

import gc
import argparse
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import recall_score, f1_score
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from tqdm import tqdm
from typing import Optional, List, Dict, Tuple

CONFIG = {
    "data_root":        "./casme_raft_processed10",
    "sample_log_path":  "./save_dual_swin/scl_log_dual_swin.csv",
    "num_classes":      3,
    "image_size":       256,
    "pretrained":       True,
    "sample_per_class": 100,      
    "epochs":           100,      
    "early_stop_patience": 30,    
    "batch_size":       32,
    "lr":               1e-4,
    "weight_decay":     1e-4,
    "grad_clip":        1.0,
    "warmup_epochs":    5,
    "seed":             42,
    "num_workers":      4,
    "save_path":        "./save_dual_swin/best_scl_dual_swin.pth",
    "frame_type":       "apex",  # 'onset' or 'apex'
    "frame_channels":   "gray",  # 'gray' or 'rgb'
}

TARGET_CLASSES = {
    "happiness": "positive", "disgust": "negative", "anger": "negative",
    "fear": "negative", "sadness": "negative", "surprise": "surprise"
}
LABEL_MAP = {"positive":  0, "negative":  1, "surprise":  2}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

IMAGENET_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STD  = [0.5, 0.5, 0.5]

def get_device() -> torch.device:
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def _parse_info_txt(path: str) -> Optional[dict]:
    fields = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if ":" in line:
                    key, _, val = line.partition(":")
                    fields[key.strip().lower()] = val.strip()
    except OSError: return None
    required = {"subject", "video", "onset", "apex", "emotion"}
    if not required.issubset(fields): return None
    return {"subject": int(fields["subject"]), "emotion": fields["emotion"].lower()}

def load_casme_processed(data_root: str) -> pd.DataFrame:
    records = []
    if not os.path.isdir(data_root): raise FileNotFoundError()
    for subj_dir in sorted(os.listdir(data_root)):
        subj_path = os.path.join(data_root, subj_dir)
        if not os.path.isdir(subj_path): continue
        for clip_dir in sorted(os.listdir(subj_path)):
            clip_path  = os.path.join(subj_path, clip_dir)
            if not os.path.isdir(clip_path): continue
            info_path  = os.path.join(clip_path, "info.txt")
            flow_npy   = os.path.join(clip_path, "flow.npy")
            flow_flip_npy   = os.path.join(clip_path, "flow_flip.npy")
            
            if not os.path.isfile(info_path): continue
            info = _parse_info_txt(info_path)
            if info is None: continue

            raw_emotion = info["emotion"]
            if raw_emotion not in TARGET_CLASSES: continue

            emotion = TARGET_CLASSES[raw_emotion]
            emotion_id = LABEL_MAP[emotion]

            records.append({
                "subject":     info["subject"],
                "clip_folder": clip_path,
                "flow_npy":    flow_npy,
                "flow_flip_npy": flow_flip_npy,
                "emotion":     emotion,
                "emotion_id":  emotion_id,
                "raw_emotion": raw_emotion
            })
    return pd.DataFrame(records)

def sample_clips(df: pd.DataFrame, n_per_class: int, seed: int = 42) -> pd.DataFrame:
    parts = []
    for _, group in df.groupby("emotion"):
        parts.append(group.sample(min(n_per_class, len(group)), random_state=seed))
    return pd.concat(parts).reset_index(drop=True)

def compute_class_weights(df: pd.DataFrame, num_classes: int, device: torch.device) -> torch.Tensor:
    counts = np.zeros(num_classes)
    for eid, cnt in df["emotion_id"].value_counts().items(): counts[int(eid)] = cnt
    counts = np.where(counts == 0, 1, counts)
    # Using inverse class frequency to handle the massive 30:1 imbalance
    total_samples = np.sum(counts)
    weights = total_samples / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)

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

def _make_transform(augment: bool) -> transforms.Compose:
    return transforms.Compose([transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

class CasmeDualSwinDataset(Dataset):
    def __init__(self, annotations: pd.DataFrame, image_size: int = 256, augment: bool = False, frame_type='apex', frame_channels='gray'):
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
        # Find spatial image in casme_processed folder structure
        folder_orig = row["clip_folder"].replace("casme_raft_processed10", "casme_processed").replace("casme_raft_processed", "casme_processed")
        frame_path = os.path.join(folder_orig, f"{self.frame_type}.jpg")

        # --- STREAM 1: OPTICAL FLOW ---
        try: flow = np.load(flow_path).astype(np.float32) 
        except: flow = np.zeros((2, self.size, self.size), dtype=np.float32)

        if flow.shape[1] != self.size or flow.shape[2] != self.size:
            flow = cv2.resize(flow.transpose(1, 2, 0), (self.size, self.size)).transpose(2, 0, 1)

        rgb_heatmap = flow_to_rgb_heatmap(flow)
        tensor_flow = self.transform(torch.from_numpy(rgb_heatmap))

        # --- STREAM 2: SPATIAL STRUCTURE ---
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


class WeightedFocalLoss(nn.Module):
    def __init__(self, weight: Optional[torch.Tensor] = None, gamma: float = 3.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 1. Get raw cross entropy loss WITHOUT weights to correctly calculate probability the model assigned to target class
        ce_loss_unweighted = F.cross_entropy(logits, targets, reduction='none')
        # 2. Probability assigned to the correct class
        pt = torch.exp(-ce_loss_unweighted)
        # 3. Compute focal loss
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss_unweighted
        # 4. Apply class weights manually if provided
        if self.weight is not None:
            focal_loss = focal_loss * self.weight[targets]
        return focal_loss.mean()

class FlowDecoder(nn.Module):
    def __init__(self, in_features, out_channels=3):
        super().__init__()
        # Project 768-dim global feature back to 256 * 8 * 8 spatial
        self.fc = nn.Linear(in_features, 256 * 8 * 8)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 64x64
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),   # 128x128
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, out_channels, kernel_size=4, stride=2, padding=1) # 256x256
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 256, 8, 8)
        return self.deconv(x)


class DualSwinMER(nn.Module):
    def __init__(self, num_classes: int = 3, pretrained: bool = True, spatial_chans: int = 1):
        super().__init__()
        
        self.stream_motion = timm.create_model("swinv2_tiny_window8_256", pretrained=pretrained, num_classes=0, in_chans=3)
        self.stream_spatial = timm.create_model("swinv2_tiny_window8_256", pretrained=pretrained, num_classes=0, in_chans=spatial_chans)
        
        # Decoder for Auxiliary Task: Reconstructing optical flow
        self.flow_decoder = FlowDecoder(self.stream_spatial.num_features, out_channels=3)
        
        total_feat_dim = self.stream_motion.num_features + self.stream_spatial.num_features
        
        # Split classifier into encoder and head for SupCon
        self.encoder = nn.Sequential(
            nn.LayerNorm(total_feat_dim),
            nn.Dropout(p=0.5),         
        )
        self.head = nn.Linear(total_feat_dim, num_classes)

    def forward(self, flow_x: torch.Tensor, spatial_x: torch.Tensor, return_flow: bool = False, return_feat: bool = False):
        f_motion = self.stream_motion(flow_x)
        f_spatial = self.stream_spatial(spatial_x)
        f_fused = torch.cat((f_motion, f_spatial), dim=1)
        
        embed = self.encoder(f_fused)
        logits = self.head(embed)
        
        if return_flow and return_feat:
            pred_flow = self.flow_decoder(f_spatial)
            return logits, embed, pred_flow
        elif return_flow:
            pred_flow = self.flow_decoder(f_spatial)
            return logits, pred_flow
        elif return_feat:
            return logits, embed
            
        return logits

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf."""
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = features.device

        if len(features.shape) == 2:
            features = features.unsqueeze(1) # shape: [bsz, 1, embed_dim]

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both labels and mask')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        
        # normalize features
        contrast_feature = F.normalize(contrast_feature, p=2, dim=1)
        
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_feature = F.normalize(anchor_feature, p=2, dim=1)
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # compute mean of log-likelihood over positive
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

def train_one_epoch(model, loader, optimizer, criterion_cls, criterion_scl, device, grad_clip) -> Tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for (x_f, x_s), y in tqdm(loader, desc="  train", leave=False):
        x_f, x_s, y = x_f.to(device), x_s.to(device), y.to(device)
        
        # SCL + FDP-style Multi-Task: Forward pass returning optical flow & embedded features
        logits, embed, pred_flow = model(x_f, x_s, return_flow=True, return_feat=True)
        
        # Primary Classification Loss
        cls_loss = criterion_cls(logits, y)

        # Contrastive Loss
        scl_loss = criterion_scl(embed, y)
        
        # Auxiliary Task: MSE measuring reconstructed flow vs actual geometric flow
        aux_loss = torch.nn.functional.mse_loss(pred_flow, x_f)
        
        # Total loss config: CLS + 0.5 * SCL + 0.5 * AUX
        loss = cls_loss + (0.5 * scl_loss) + (0.5 * aux_loss)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        
        total_loss += loss.item() * y.size(0)
        correct    += (logits.argmax(1) == y).sum().item()
        total      += y.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)

@torch.no_grad()
def evaluate(model, loader, criterion, device) -> Tuple[float, float, float, float]:
    model.eval()
    total_loss, all_preds, all_targets = 0.0, [], []
    for (x_f, x_s), y in tqdm(loader, desc="  eval ", leave=False):
        x_f, x_s, y = x_f.to(device), x_s.to(device), y.to(device)
        logits = model(x_f, x_s)
        loss   = criterion(logits, y)
        total_loss += loss.item() * y.size(0)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_targets.extend(y.cpu().numpy())
    avg_loss = total_loss / max(len(all_targets), 1)
    acc = np.mean(np.array(all_preds) == np.array(all_targets))
    uar = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    uf1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    return avg_loss, acc, uar, uf1

def train(model, train_loader, val_loader, device, config, save_path, class_weights=None) -> List[Dict]:
    epochs, patience = config["epochs"], config["early_stop_patience"]
    
    # Use Focal Loss with class weights to handle imbalance without sample duplication
    criterion_cls = WeightedFocalLoss(weight=class_weights, gamma=2.0)
    criterion_scl = SupConLoss(temperature=0.1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    warmup = config["warmup_epochs"]
    def lr_lambda(epoch):
        if epoch < warmup: return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, epochs - warmup)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    best_val_uar = 0.0
    history = []
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion_cls, criterion_scl, device, config["grad_clip"])
        val_loss, val_acc, val_uar, val_uf1 = evaluate(model, val_loader, criterion_cls, device)
        scheduler.step()

        is_best = val_uar > best_val_uar
        print(f"Epoch {epoch:03d}/{epochs} | lr={lr:.2e} | train loss={train_loss:.4f} acc={train_acc:.3f} | val loss={val_loss:.4f} acc={val_acc:.3f} UAR={val_uar:.3f} UF1={val_uf1:.3f}" + (" <- best" if is_best else ""))

        if is_best:
            best_val_uar = val_uar
            epochs_no_improve = 0
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({"epoch": epoch, "model": model.state_dict(), "val_acc": val_acc, "config": config}, save_path)
        else:
            epochs_no_improve += 1

        history.append({"epoch": epoch, "lr": lr, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc, "val_uar": val_uar, "val_uf1": val_uf1})
        
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement.")
            break

    return history

def make_loaders(train_df, val_df, config, pin_memory):
    train_ds = CasmeDualSwinDataset(train_df, config["image_size"], augment=True, frame_type=config['frame_type'], frame_channels=config['frame_channels'])
    val_ds   = CasmeDualSwinDataset(val_df,   config["image_size"], augment=False, frame_type=config['frame_type'], frame_channels=config['frame_channels'])
    
    # Use standard shuffling to preserve variety, paired with Focal and SCL loss.
    train_loader = DataLoader(
        train_ds, 
        batch_size=config["batch_size"], 
        shuffle=True, 
        drop_last=True, 
        num_workers=config["num_workers"], 
        pin_memory=pin_memory, 
        persistent_workers=(config["num_workers"] > 0)
    )
    
    val_loader = DataLoader(
        val_ds,   
        batch_size=config["batch_size"], 
        shuffle=False, 
        drop_last=False, 
        num_workers=config["num_workers"], 
        pin_memory=pin_memory, 
        persistent_workers=(config["num_workers"] > 0)
    )
    return train_loader, val_loader

def run_kfold(annotations, config, device, pin_memory, k: int = 10):
    subjects = np.array(sorted(annotations["subject"].unique()))
    rng = np.random.default_rng(config["seed"])
    folds = np.array_split(rng.permutation(subjects), k)
    results = []
    
    spatial_chans = 1 if config['frame_channels'] == 'gray' else 3

    for fold_idx, val_subjects in enumerate(folds):
        train_subjects = np.concatenate([folds[j] for j in range(k) if j != fold_idx])
        train_df = annotations[annotations["subject"].isin(train_subjects)]
        val_df   = annotations[annotations["subject"].isin(val_subjects)]
        if len(val_df) == 0: continue

        class_weights = compute_class_weights(train_df, config["num_classes"], device)
        train_loader, val_loader = make_loaders(train_df, val_df, config, pin_memory)
        model = DualSwinMER(config["num_classes"], config["pretrained"], spatial_chans=spatial_chans).to(device)
        save_path = config["save_path"].replace(".pth", f"_fold{fold_idx+1}_{config['frame_type']}_{config['frame_channels']}.pth")

        history = train(model, train_loader, val_loader, device, config, save_path, class_weights)
        best_row = max(history, key=lambda h: h["val_uar"])
        best_ep  = best_row["epoch"]
        best_acc_for_ep = best_row["val_acc"]
        best_uar_for_ep = best_row["val_uar"]
        best_uf1_for_ep = best_row["val_uf1"]

        results.append({"fold": fold_idx + 1, "best_acc": best_acc_for_ep, "best_uar": best_uar_for_ep, "best_uf1": best_uf1_for_ep, "best_epoch": best_ep, "saved_model": save_path})
        del model, train_loader, val_loader
        gc.collect()
        if device.type == "cuda": torch.cuda.empty_cache()
        
        print(f"\n--- Intermediate Summary after Fold {fold_idx + 1} ---")
        print(pd.DataFrame(results).to_string(index=False))
        print("-" * 50 + "\n")
        
    df = pd.DataFrame(results)
    csv_path = f"kfold{k}_dual_swin_{config['frame_type']}_{config['frame_channels']}_results.csv"
    df.to_csv(csv_path, index=False)
    
    print("\n" + "="*50)
    print(f"K-FOLD CROSS VALIDATION SUMMARY ({k} Folds)")
    print(f"Model: Dual Swin | Frame: {config['frame_type']} | Channels: {config['frame_channels']}")
    print("="*50)
    print(df.to_string(index=False))
    print("-" * 50)
    print(f"Mean Best Acc: {df['best_acc'].mean():.4f} ± {df['best_acc'].std():.4f}")
    print(f"Mean Best UAR: {df['best_uar'].mean():.4f} ± {df['best_uar'].std():.4f}")
    print(f"Mean Best UF1: {df['best_uf1'].mean():.4f} ± {df['best_uf1'].std():.4f}")
    print("="*50 + "\n")
    return df

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-root",        default=CONFIG["data_root"])
    p.add_argument("--epochs",           type=int,   default=CONFIG["epochs"])
    p.add_argument("--batch-size",       type=int,   default=CONFIG["batch_size"])
    p.add_argument("--sample-per-class", type=int,   default=CONFIG["sample_per_class"])
    p.add_argument("--num-workers",      type=int,   default=CONFIG["num_workers"])
    p.add_argument("--lr",               type=float, default=CONFIG["lr"])
    p.add_argument("--seed",             type=int,   default=CONFIG["seed"])
    p.add_argument("--kfold",            type=int,   default=None)
    p.add_argument("--all-data",         action="store_true")
    
    # New options
    p.add_argument("--frame-type",       type=str,   default=CONFIG["frame_type"], choices=["onset", "apex"])
    p.add_argument("--frame-channels",   type=str,   default=CONFIG["frame_channels"], choices=["gray", "rgb"])
    
    args = p.parse_args()

    config = {**CONFIG, "data_root": args.data_root, "epochs": args.epochs, "batch_size": args.batch_size,
              "sample_per_class": args.sample_per_class, "num_workers": args.num_workers,
              "lr": args.lr, "seed": args.seed, "frame_type": args.frame_type, "frame_channels": args.frame_channels}

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    device = get_device()
    pin_memory = (device.type == "cuda")

    all_clips = load_casme_processed(config["data_root"])
    annotations = all_clips.copy() if args.all_data else sample_clips(all_clips, config["sample_per_class"], config["seed"])

    if args.kfold is not None:
        run_kfold(annotations, config, device, pin_memory, k=args.kfold)
