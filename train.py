"""
train.py — Baseline MER training script
6-Channel Swin Transformer on CASME3 (casme_processed folder)

Run:
    python train.py                          # full LOSO with defaults
    python train.py --epochs 10              # faster run
    python train.py --sample-per-class 50   # smaller subset
    python train.py --no-loso               # single train/val split (first subject held out)
"""

import os
import re
import gc
import copy
import time
import argparse
import warnings
warnings.filterwarnings("ignore")

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

CONFIG = {
    "data_root":        "./casme_processed",
    "sample_log_path":  "sample_log.csv",
    "num_classes":      7,
    "image_size":       224,
    "pretrained":       True,
    "sample_per_class": 100,
    "epochs":           30,
    "batch_size":       16,
    "lr":               1e-4,
    "weight_decay":     1e-4,
    "label_smoothing":  0.1,
    "grad_clip":        1.0,
    "warmup_epochs":    5,
    "seed":             42,
    "num_workers":      2,        # works correctly in a .py file (spawn-safe)
    "save_path":        "best_baseline.pth",
}

LABEL_MAP = {
    "happiness":  0,
    "disgust":    1,
    "surprise":   2,
    "anger":      3,
    "fear":       4,
    "sadness":    5,
    "others":     6,
}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


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
            onset_path = os.path.join(clip_path, "onset.jpg")
            apex_path  = os.path.join(clip_path, "apex.jpg")

            if not all(os.path.isfile(p) for p in [info_path, onset_path, apex_path]):
                continue

            info = _parse_info_txt(info_path)
            if info is None:
                continue

            emotion_id = LABEL_MAP.get(info["emotion"], LABEL_MAP.get("others", 6))
            records.append({
                "subject":     info["subject"],
                "clip_folder": clip_path,
                "onset_path":  onset_path,
                "apex_path":   apex_path,
                "emotion":     info["emotion"],
                "emotion_id":  emotion_id,
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
    df[["clip_folder", "subject", "emotion", "emotion_id"]].to_csv(path, index=False)
    print(f"Sample log saved → {path}  ({len(df)} rows)")


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

_transform_base = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

_transform_aug = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class CasmeBaselineDataset(Dataset):
    def __init__(self, annotations: pd.DataFrame, image_size: int = 224, augment: bool = False):
        self.df        = annotations.reset_index(drop=True)
        self.size      = image_size
        self.augment   = augment
        self.transform = _transform_aug if augment else _transform_base
        self._blank    = np.zeros((image_size, image_size, 3), dtype=np.uint8)

    def __len__(self):
        return len(self.df)

    def _load_jpg(self, path: str) -> np.ndarray:
        img = cv2.imread(path)
        if img is None:
            return self._blank.copy()
        return align_and_crop_face(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), self.size)

    def __getitem__(self, idx: int):
        row   = self.df.iloc[idx]
        label = int(row["emotion_id"])

        onset_img = self._load_jpg(row["onset_path"])
        apex_img  = self._load_jpg(row["apex_path"])

        if self.augment:
            seed = torch.randint(0, 2**32, (1,)).item()
            torch.manual_seed(seed);  onset_t = self.transform(onset_img)
            torch.manual_seed(seed);  apex_t  = self.transform(apex_img)
        else:
            onset_t = self.transform(onset_img)
            apex_t  = self.transform(apex_img)

        return torch.cat([onset_t, apex_t], dim=0), label


# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, num_classes: int, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        self.n_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        confidence = 1.0 - self.smoothing
        smooth_val = self.smoothing / (self.n_classes - 1)
        soft_targets = torch.full_like(logits, smooth_val)
        soft_targets.scatter_(1, targets.unsqueeze(1), confidence)
        return -(soft_targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


class BaselineSwinMER(nn.Module):
    def __init__(self, num_classes: int = 7, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=pretrained,
            num_classes=0,
        )
        self._adapt_patch_embedding()
        feat_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(p=0.3),
            nn.Linear(feat_dim, num_classes),
        )

    def _adapt_patch_embedding(self):
        old = self.backbone.patch_embed.proj
        w   = old.weight.data
        new = nn.Conv2d(6, old.out_channels, old.kernel_size, old.stride,
                        old.padding, bias=(old.bias is not None))
        new.weight.data = torch.cat([w, w], dim=1) * 0.5
        if old.bias is not None:
            new.bias.data = old.bias.data.clone()
        self.backbone.patch_embed.proj = new
        print(f"Patch embedding adapted: 3ch → 6ch  {tuple(new.weight.shape)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(x))


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip) -> Tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in tqdm(loader, desc="  train", leave=False):
        x, y = x.to(device), y.to(device)
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
def evaluate(model, loader, criterion, device) -> Tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in tqdm(loader, desc="  eval ", leave=False):
        x, y   = x.to(device), y.to(device)
        logits = model(x)
        loss   = criterion(logits, y)
        total_loss += loss.item() * y.size(0)
        correct    += (logits.argmax(1) == y).sum().item()
        total      += y.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def train(model, train_loader, val_loader, device, config, save_path) -> List[Dict]:
    epochs    = config["epochs"]
    criterion = LabelSmoothingCrossEntropy(config["num_classes"], config["label_smoothing"])

    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(),   "lr": config["lr"] * 0.1},
        {"params": model.classifier.parameters(), "lr": config["lr"]},
    ], weight_decay=config["weight_decay"])

    warmup = config["warmup_epochs"]
    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, epochs - warmup)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_acc = 0.0
    history      = []

    for epoch in range(1, epochs + 1):
        lr = optimizer.param_groups[1]["lr"]
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, config["grad_clip"]
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        is_best = val_acc > best_val_acc
        print(
            f"Epoch {epoch:03d}/{epochs} | lr={lr:.2e} | "
            f"train loss={train_loss:.4f} acc={train_acc:.3f} | "
            f"val loss={val_loss:.4f} acc={val_acc:.3f}"
            + (" ← best" if is_best else "")
        )

        if is_best:
            best_val_acc = val_acc
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "val_acc": val_acc, "config": config}, save_path)

        history.append({"epoch": epoch, "lr": lr, "train_loss": train_loss,
                        "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc})

    print(f"\nBest val accuracy: {best_val_acc:.4f}  →  {save_path}")
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
    print(f"Starting LOSO: {len(subjects)} subjects, {config['epochs']} epochs each")
    print("=" * 60)

    for i, subj in enumerate(subjects):
        train_df = annotations[annotations["subject"] != subj]
        val_df   = annotations[annotations["subject"] == subj]
        print(f"\n[{i+1}/{len(subjects)}] Held-out subject {subj} | "
              f"train={len(train_df)} val={len(val_df)}")

        if len(val_df) == 0:
            print("  Skipped — no val samples.")
            continue

        train_loader, val_loader = make_loaders(train_df, val_df, config, pin_memory)

        model = BaselineSwinMER(config["num_classes"], config["pretrained"]).to(device)
        save_path = config["save_path"].replace(".pth", f"_subj{subj}.pth")

        history  = train(model, train_loader, val_loader, device, config, save_path)
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
    print(f"\nMean accuracy: {df['best_acc'].mean():.4f} ± {df['best_acc'].std():.4f}")
    df.to_csv("loso_results.csv", index=False)
    print("Results saved → loso_results.csv")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Baseline MER — 6ch Swin Transformer")
    p.add_argument("--data-root",        default=CONFIG["data_root"])
    p.add_argument("--epochs",           type=int,   default=CONFIG["epochs"])
    p.add_argument("--batch-size",       type=int,   default=CONFIG["batch_size"])
    p.add_argument("--sample-per-class", type=int,   default=CONFIG["sample_per_class"])
    p.add_argument("--num-workers",      type=int,   default=CONFIG["num_workers"])
    p.add_argument("--lr",               type=float, default=CONFIG["lr"])
    p.add_argument("--seed",             type=int,   default=CONFIG["seed"])
    p.add_argument("--no-loso",          action="store_true",
                   help="Single fold only (first subject held out), for a quick sanity check")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Merge CLI args into config
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

    # Load & sample data
    all_clips   = load_casme_processed(config["data_root"])
    annotations = sample_clips(all_clips, config["sample_per_class"], config["seed"])
    save_sample_log(annotations, config["sample_log_path"])

    if args.no_loso:
        # Single fold: first subject held out
        subjects    = sorted(annotations["subject"].unique())
        held_out    = subjects[0]
        train_df    = annotations[annotations["subject"] != held_out]
        val_df      = annotations[annotations["subject"] == held_out]
        print(f"\nSingle fold — held-out: {held_out} | train={len(train_df)} val={len(val_df)}")

        train_loader, val_loader = make_loaders(train_df, val_df, config, pin_memory)
        model = BaselineSwinMER(config["num_classes"], config["pretrained"]).to(device)
        train(model, train_loader, val_loader, device, config, config["save_path"])
    else:
        run_loso(annotations, config, device, pin_memory)
