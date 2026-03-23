"""
train.py — Baseline MER training script
6-Channel Swin Transformer on CASME3 (casme_processed folder)

Run:
    python train.py                          # full LOSO with defaults
    python train.py --no-loso               # single train/val split
    python train.py --kfold 10              # 10-fold subject-grouped CV
    python train.py --kfold 10 --all-data   # use all available clips (no sampling cap)
"""

import os

# Set cache directories before importing torch/torchvision/huggingface to read offline weights
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


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

CONFIG = {
    "data_root":        "./casme_raft_processed10",
    "sample_log_path":  "./save_focal_loss/focal_loss_log_hsv.csv",
    "num_classes":      3,
    "image_size":       256,      # <--- CHANGE THIS FROM 224 TO 256
    "pretrained":       True,
    "sample_per_class": 100,      
    "epochs":           100,      
    "early_stop_patience": 30,    
    "batch_size":      8,
    "lr":               1e-4,
    "weight_decay":     1e-4,
    "label_smoothing":  0.1,
    "grad_clip":        1.0,
    "warmup_epochs":    5,
    "seed":             42,
    "num_workers":      4,
    "save_path":        "./save_focal_loss/best_focal_loss_hsv.pth",
}

TARGET_CLASSES = {
    "happiness": "positive",
    "disgust": "negative",
    "anger": "negative",
    "fear": "negative",
    "sadness": "negative",
    "surprise": "surprise"
}

LABEL_MAP = {
    "positive":  0,
    "negative":  1,
    "surprise":  2,
}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

IMAGENET_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STD  = [0.5, 0.5, 0.5]


# ─────────────────────────────────────────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def _parse_info_txt(path: str) -> Optional[dict]:
    fields = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if ":" in line:
                    key, _, val = line.partition(":")
                    fields[key.strip().lower()] = val.strip()
    except OSError:
        return None

    required = {"subject", "video", "onset", "apex", "emotion"}
    if not required.issubset(fields):
        return None

    return {
        "subject": int(fields["subject"]),
        "video":   fields["video"],
        "onset":   int(fields["onset"]),
        "apex":    int(fields["apex"]),
        "emotion": fields["emotion"].lower(),
    }


def load_casme_processed(data_root: str) -> pd.DataFrame:
    records = []
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"data_root not found: {data_root}")

    for subj_dir in sorted(os.listdir(data_root)):
        subj_path = os.path.join(data_root, subj_dir)
        if not os.path.isdir(subj_path):
            continue
        for clip_dir in sorted(os.listdir(subj_path)):
            clip_path  = os.path.join(subj_path, clip_dir)
            if not os.path.isdir(clip_path):
                continue
            info_path  = os.path.join(clip_path, "info.txt")
            onset_gray = os.path.join(clip_path, "onset_gray.jpg")
            flow_npy   = os.path.join(clip_path, "flow.npy")
            onset_gray_flip = os.path.join(clip_path, "onset_gray_flip.jpg")
            flow_flip_npy   = os.path.join(clip_path, "flow_flip.npy")

            if not all(os.path.isfile(p) for p in [info_path, onset_gray, flow_npy, onset_gray_flip, flow_flip_npy]):
                continue

            info = _parse_info_txt(info_path)
            if info is None:
                continue

            raw_emotion = info["emotion"]
            if raw_emotion not in TARGET_CLASSES:
                continue

            emotion = TARGET_CLASSES[raw_emotion]
            emotion_id = LABEL_MAP[emotion]

            records.append({
                "subject":     info["subject"],
                "clip_folder": clip_path,
                "onset_gray":  onset_gray,
                "flow_npy":    flow_npy,
                "onset_gray_flip": onset_gray_flip,
                "flow_flip_npy": flow_flip_npy,
                "emotion":     emotion,
                "emotion_id":  emotion_id,
                "raw_emotion": raw_emotion
            })

    df = pd.DataFrame(records)
    print(f"Scanned {data_root!r}: {len(df)} clips, {df['subject'].nunique()} subjects")
    print(f"Emotion distribution:\n{df['emotion'].value_counts()}\n")
    return df


def sample_clips(df: pd.DataFrame, n_per_class: int, seed: int = 42) -> pd.DataFrame:
    if df.empty or "emotion" not in df.columns:
        raise ValueError("No clips found — check data_root.")
    parts = []
    for _, group in df.groupby("emotion"):
        parts.append(group.sample(min(n_per_class, len(group)), random_state=seed))
    sampled = pd.concat(parts).reset_index(drop=True)
    print(f"Sampled {len(sampled)} clips ({n_per_class}/class, seed={seed}):")
    print(sampled["emotion"].value_counts().to_string(), "\n")
    return sampled


def save_sample_log(df: pd.DataFrame, path: str) -> None:
    # if os.dirname(path) and not os.path.isdir(os.path.dirname(path)):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df[["clip_folder", "subject", "emotion", "emotion_id"]].to_csv(path, index=False)
    print(f"Sample log saved → {path}  ({len(df)} rows)")


def compute_class_weights(df: pd.DataFrame, num_classes: int, device: torch.device) -> torch.Tensor:
    """
    Inverse-frequency class weights to handle imbalanced data.
    Classes with fewer samples get higher weight in the loss.
    """
    counts = np.zeros(num_classes)
    for eid, cnt in df["emotion_id"].value_counts().items():
        counts[int(eid)] = cnt
    counts = np.where(counts == 0, 1, counts)   # avoid divide-by-zero
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes  # normalise
    print("Class weights:", {ID_TO_LABEL[i]: f"{w:.3f}" for i, w in enumerate(weights)})
    return torch.tensor(weights, dtype=torch.float32, device=device)


# ─────────────────────────────────────────────────────────────────────────────
# FACE DETECTION
# ─────────────────────────────────────────────────────────────────────────────

_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def align_and_crop_face(img: np.ndarray, target_size: int = 224) -> np.ndarray:
    gray  = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = _face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )
    if len(faces) > 0:
        x, y, w, h = faces[0]
        m  = int(0.15 * min(w, h))
        x1 = max(0, x - m);   y1 = max(0, y - m)
        x2 = min(img.shape[1], x + w + m);  y2 = min(img.shape[0], y + h + m)
        crop = img[y1:y2, x1:x2]
    else:
        h, w = img.shape[:2];  s = min(h, w)
        crop = img[(h-s)//2:(h-s)//2+s, (w-s)//2:(w-s)//2+s]
    return cv2.resize(crop, (target_size, target_size))


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# DATASET & PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def flow_to_rgb_heatmap(flow_npy: np.ndarray) -> np.ndarray:
    """
    Converts a (2, H, W) optical flow array into a (3, H, W) RGB heatmap.
    Uses dynamic normalization to act as an 'auto-gain' for micro-expressions.
    """
    # 1. Transpose to (H, W, 2) for OpenCV processing
    flow = flow_npy.transpose(1, 2, 0)
    u, v = flow[..., 0], flow[..., 1]
    
    # 2. Calculate Magnitude and Angle
    mag, ang = cv2.cartToPolar(u, v)
    
    # 3. Create HSV image shell
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    
    # Hue: Map angle to [0, 180] for OpenCV's 8-bit HSV format
    hsv[..., 0] = ang * 180 / np.pi / 2
    
    # Saturation: Max out saturation for vibrant colors
    hsv[..., 1] = 255
    
    # Value (Brightness): Dynamic Normalization
    # Find the maximum movement in THIS specific clip
    max_mag = np.max(mag)
    if max_mag > 1e-5: # Prevent division by zero
        # Scale the maximum motion to 255 (100% brightness)
        hsv[..., 2] = (mag / max_mag) * 255
    else:
        hsv[..., 2] = 0
        
    # 4. Convert HSV to RGB
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    # 5. Convert back to (3, H, W) float32 in [0, 1] range for PyTorch
    rgb_tensor = rgb.astype(np.float32) / 255.0
    return rgb_tensor.transpose(2, 0, 1)

def _make_transform(augment: bool) -> transforms.Compose:
    # Removed RandomErasing because it destroys sparse optical flow flares.
    # Added RandomHorizontalFlip to the augment pipeline.
    if augment:
        return transforms.Compose([
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

class CasmeBaselineDataset(Dataset):
    def __init__(self, annotations: pd.DataFrame, image_size: int = 256, augment: bool = False):
        self.df        = annotations.reset_index(drop=True)
        self.size      = image_size
        self.augment   = augment
        self.transform = _make_transform(augment)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row   = self.df.iloc[idx]
        label = int(row["emotion_id"])

        use_flip = self.augment and (torch.rand(1).item() > 0.5)

        if use_flip:
            flow_path  = row["flow_flip_npy"]
        else:
            flow_path  = row["flow_npy"]

        # --- STREAM 1: OPTICAL FLOW ---
        try:
            flow = np.load(flow_path).astype(np.float32) 
        except:
            flow = np.zeros((2, self.size, self.size), dtype=np.float32)

        if flow.shape[1] != self.size or flow.shape[2] != self.size:
            flow = flow.transpose(1, 2, 0)
            flow = cv2.resize(flow, (self.size, self.size))
            flow = flow.transpose(2, 0, 1)

        rgb_heatmap = flow_to_rgb_heatmap(flow)
        tensor_flow = self.transform(torch.from_numpy(rgb_heatmap))

        return tensor_flow, label


# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────

class WeightedLabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing cross entropy with per-class weights for imbalanced data."""
    def __init__(self, num_classes: int, smoothing: float = 0.1,
                 weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.smoothing = smoothing
        self.n_classes = num_classes
        self.weight    = weight   # shape: (num_classes,)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        confidence = 1.0 - self.smoothing
        smooth_val = self.smoothing / (self.n_classes - 1)
        soft_targets = torch.full_like(logits, smooth_val)
        soft_targets.scatter_(1, targets.unsqueeze(1), confidence)
        log_probs = F.log_softmax(logits, dim=1)
        loss = -(soft_targets * log_probs).sum(dim=1)   # (batch,)

        if self.weight is not None:
            w = self.weight[targets]
            loss = loss * w

        return loss.mean()

class WeightedFocalLoss(nn.Module):
    """
    Focal Loss with per-class weights.
    Forces the model to ignore 'easy' majority classes and focus on 'hard' minority classes.
    """
    def __init__(self, weight: Optional[torch.Tensor] = None, gamma: float = 2.0):
        super().__init__()
        self.weight = weight # shape: (num_classes,)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction='none', weight=self.weight)
        pt = torch.exp(-ce_loss) # Probability of the true class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        return focal_loss.mean()

# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────

class SingleStreamMER(nn.Module):
    def __init__(self, num_classes: int = 7, pretrained: bool = True):
        super().__init__()
        
        # --- Single Stream: Motion + Spatial (Swin-v2) ---
        self.stream = timm.create_model(
            "swinv2_tiny_window8_256", 
            pretrained=pretrained,
            num_classes=0, 
        )
        feat_dim = self.stream.num_features # Usually 768

        # --- Classification ---
        self.classifier = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(p=0.5),         
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f_feat = self.stream(x)
        return self.classifier(f_feat)

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip) -> Tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    
    for x, y in tqdm(loader, desc="  train", leave=False):
        x = x.to(device)
        y = y.to(device)
        
        logits = model(x)
        loss   = criterion(logits, y)
        
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
    total_loss = 0.0
    
    all_preds = []
    all_targets = []
    
    for x, y in tqdm(loader, desc="  eval ", leave=False):
        x = x.to(device)
        y = y.to(device)
        
        logits = model(x)
        loss   = criterion(logits, y)
        total_loss += loss.item() * y.size(0)
        
        preds = logits.argmax(1)
        
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(y.cpu().numpy())
        
    avg_loss = total_loss / max(len(all_targets), 1)
    
    acc = np.mean(np.array(all_preds) == np.array(all_targets))
    uar = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    uf1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    
    return avg_loss, acc, uar, uf1


def train(model, train_loader, val_loader, device, config, save_path,
          class_weights=None) -> List[Dict]:
    epochs    = config["epochs"]
    patience  = config["early_stop_patience"]
    criterion = WeightedFocalLoss(weight=class_weights)

    optimizer = torch.optim.AdamW([
        {"params": model.stream.parameters(),  "lr": config["lr"] * 0.1},
        {"params": model.classifier.parameters(),     "lr": config["lr"]},
    ], weight_decay=config["weight_decay"])

    warmup = config["warmup_epochs"]
    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, epochs - warmup)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler  = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    best_val_acc   = 0.0
    epochs_no_improv = 0
    history    = []

    for epoch in range(1, epochs + 1):
        lr = optimizer.param_groups[1]["lr"]
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, config["grad_clip"]
        )
        val_loss, val_acc, val_uar, val_uf1 = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        is_best = val_uar > best_val_acc
        
        print(
            f"Epoch {epoch:03d}/{epochs} | lr={lr:.2e} | "
            f"train loss={train_loss:.4f} acc={train_acc:.3f} | "
            f"val loss={val_loss:.4f} acc={val_acc:.3f} UAR={val_uar:.3f} UF1={val_uf1:.3f}"
            + (" <- best" if is_best else "")
        )

        if is_best:
            best_val_acc = val_uar
            epochs_no_improv = 0
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "val_acc": val_acc, "config": config}, save_path)
        else:
            epochs_no_improv += 1

        history.append({
            "epoch": epoch, "lr": lr, "train_loss": train_loss,
            "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc,
            "val_uar": val_uar, "val_uf1": val_uf1
        })

        if epochs_no_improv >= patience:
            print(f"  Early stopping triggered — no improvement for {patience} epochs.")
            break

    print(f"\nBest val accuracy: {best_val_acc:.4f}  ->  {save_path}")
    return history


def make_loaders(train_df, val_df, config, pin_memory):
    train_ds = CasmeBaselineDataset(train_df, config["image_size"], augment=True)
    val_ds   = CasmeBaselineDataset(val_df,   config["image_size"], augment=False)
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True,
                              num_workers=config["num_workers"], pin_memory=pin_memory,
                              persistent_workers=(config["num_workers"] > 0))
    val_loader   = DataLoader(val_ds,   batch_size=config["batch_size"], shuffle=False,
                              num_workers=config["num_workers"], pin_memory=pin_memory,
                              persistent_workers=(config["num_workers"] > 0))
    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
# LOSO
# ─────────────────────────────────────────────────────────────────────────────

def run_loso(annotations, config, device, pin_memory) -> pd.DataFrame:
    subjects = sorted(annotations["subject"].unique())
    results  = []
    print(f"Starting LOSO: {len(subjects)} subjects, up to {config['epochs']} epochs each")
    print("=" * 60)

    for i, subj in enumerate(subjects):
        train_df = annotations[annotations["subject"] != subj]
        val_df   = annotations[annotations["subject"] == subj]
        print(f"\n[{i+1}/{len(subjects)}] Held-out subject {subj} | "
              f"train={len(train_df)} val={len(val_df)}")
        if len(val_df) == 0:
            print("  Skipped.")
            continue

        class_weights  = compute_class_weights(train_df, config["num_classes"], device)
        train_loader, val_loader = make_loaders(train_df, val_df, config, pin_memory)
        model     = SingleStreamMER(config["num_classes"], config["pretrained"]).to(device)
        save_path = config["save_path"].replace(".pth", f"_subj{subj}.pth")

        history  = train(model, train_loader, val_loader, device, config, save_path, class_weights)
        best_acc = max(h["val_acc"] for h in history)
        best_ep  = max(history, key=lambda h: h["val_acc"])["epoch"]

        results.append({"subject": subj, "n_val": len(val_df),
                        "best_acc": best_acc, "best_epoch": best_ep,
                        "checkpoint": save_path})
        del model, train_loader, val_loader
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("LOSO SUMMARY")
    print(df.to_string(index=False))
    print(f"\nMean accuracy: {df['best_acc'].mean():.4f} +/- {df['best_acc'].std():.4f}")
    df.to_csv("loso_results.csv", index=False)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# K-FOLD
# ─────────────────────────────────────────────────────────────────────────────

def run_kfold(annotations, config, device, pin_memory, k: int = 10) -> pd.DataFrame:
    """Subject-grouped k-fold CV — no subject appears in both train and val."""
    subjects = np.array(sorted(annotations["subject"].unique()))
    rng      = np.random.default_rng(config["seed"])
    folds    = np.array_split(rng.permutation(subjects), k)

    results = []
    print(f"Starting {k}-Fold CV: {len(subjects)} subjects, "
          f"up to {config['epochs']} epochs per fold (early stop patience={config['early_stop_patience']})")
    print("=" * 60)

    for fold_idx, val_subjects in enumerate(folds):
        train_subjects = np.concatenate([folds[j] for j in range(k) if j != fold_idx])
        train_df = annotations[annotations["subject"].isin(train_subjects)]
        val_df   = annotations[annotations["subject"].isin(val_subjects)]

        print(f"\n[Fold {fold_idx+1}/{k}] "
              f"val subjects={sorted(val_subjects.tolist())} | "
              f"train={len(train_df)} val={len(val_df)}")
        if len(val_df) == 0:
            print("  Skipped.")
            continue

        class_weights  = compute_class_weights(train_df, config["num_classes"], device)
        train_loader, val_loader = make_loaders(train_df, val_df, config, pin_memory)
        model     = SingleStreamMER(config["num_classes"], config["pretrained"]).to(device)
        save_path = config["save_path"].replace(".pth", f"_fold{fold_idx+1}.pth")

        history  = train(model, train_loader, val_loader, device, config, save_path, class_weights)
        
        best_uar = max(h["val_uar"] for h in history)
        best_uf1 = max(h["val_uf1"] for h in history)
        best_ep  = max(history, key=lambda h: h["val_uar"])["epoch"]
        best_acc_for_ep = next(h["val_acc"] for h in history if h["epoch"] == best_ep)

        results.append({
            "fold":       fold_idx + 1,
            "n_train":    len(train_df),
            "n_val":      len(val_df),
            "best_acc":   best_acc_for_ep,
            "best_uar":   best_uar,
            "best_uf1":   best_uf1,
            "best_epoch": best_ep,
            "checkpoint": save_path,
        })
        del model, train_loader, val_loader
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    df = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print(f"{k}-FOLD CV SUMMARY")
    print(df[["fold", "n_train", "n_val", "best_acc", "best_uar", "best_uf1", "best_epoch"]].to_string(index=False))
    print(f"\nMean UF1: {df['best_uf1'].mean():.4f} +/- {df['best_uf1'].std():.4f}")
    print(f"Mean UAR: {df['best_uar'].mean():.4f} +/- {df['best_uar'].std():.4f}")
    df.to_csv(f"kfold{k}_results.csv", index=False)
    print(f"Results saved -> kfold{k}_results.csv")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Baseline MER -- 6ch Swin Transformer")
    p.add_argument("--data-root",        default=CONFIG["data_root"])
    p.add_argument("--epochs",           type=int,   default=CONFIG["epochs"])
    p.add_argument("--batch-size",       type=int,   default=CONFIG["batch_size"])
    p.add_argument("--sample-per-class", type=int,   default=CONFIG["sample_per_class"])
    p.add_argument("--num-workers",      type=int,   default=CONFIG["num_workers"])
    p.add_argument("--lr",               type=float, default=CONFIG["lr"])
    p.add_argument("--seed",             type=int,   default=CONFIG["seed"])
    p.add_argument("--no-loso",          action="store_true",
                   help="Single fold only (first subject held out)")
    p.add_argument("--kfold",            type=int,   default=None,
                   help="Run subject-grouped k-fold CV (e.g. --kfold 10)")
    p.add_argument("--all-data",         action="store_true",
                   help="Use all available clips — disables sample_per_class cap")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    config = {**CONFIG,
              "data_root":        args.data_root,
              "epochs":           args.epochs,
              "batch_size":       args.batch_size,
              "sample_per_class": args.sample_per_class,
              "num_workers":      args.num_workers,
              "lr":               args.lr,
              "seed":             args.seed}

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    device     = get_device()
    pin_memory = (device.type == "cuda")
    print(f"Device: {device}")

    all_clips = load_casme_processed(config["data_root"])

    if args.all_data:
        print("Using ALL available clips (no sampling cap)")
        annotations = all_clips.copy()
    else:
        annotations = sample_clips(all_clips, config["sample_per_class"], config["seed"])

    save_sample_log(annotations, config["sample_log_path"])

    if args.no_loso:
        subjects = sorted(annotations["subject"].unique())
        held_out = subjects[0]
        train_df = annotations[annotations["subject"] != held_out]
        val_df   = annotations[annotations["subject"] == held_out]
        print(f"\nSingle fold -- held-out: {held_out} | train={len(train_df)} val={len(val_df)}")
        class_weights  = compute_class_weights(train_df, config["num_classes"], device)
        train_loader, val_loader = make_loaders(train_df, val_df, config, pin_memory)
        model = SingleStreamMER(config["num_classes"], config["pretrained"]).to(device)
        train(model, train_loader, val_loader, device, config, config["save_path"], class_weights)

    elif args.kfold is not None:
        run_kfold(annotations, config, device, pin_memory, k=args.kfold)

    else:
        run_loso(annotations, config, device, pin_memory)