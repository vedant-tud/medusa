import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score, confusion_matrix
import csv
import sys
import os

# Import our common plotter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from medusa_plotter import create_summary_figure, summarize_predictions
from medusa_plotter import create_summary_figure, summarize_predictions

from dataset import CasmeDualSwinDataset
from models import DualStreamModel
from utils import WeightedFocalLoss

# Map raw emotions to positive/negative/surprise
TARGET_CLASSES = {
    'positive': 0, 'happiness': 0,
    'negative': 1, 'disgust': 1, 'repression': 1, 'anger': 1, 'fear': 1, 'sadness': 1,
    'surprise': 2
}

def _parse_info_txt(path):
    info = {}
    with open(path, 'r') as f:
        for line in f:
            if ':' in line:
                key, val = line.strip().split(':', 1)
                info[key.strip().lower()] = val.strip().lower()
    return info

def load_casme_processed(data_root: str) -> pd.DataFrame:
    records = []
    if not os.path.isdir(data_root): raise FileNotFoundError(f"Missing {data_root}")
    for subj_dir in sorted(os.listdir(data_root)):
        subj_path = os.path.join(data_root, subj_dir)
        if not os.path.isdir(subj_path): continue
        for clip_dir in sorted(os.listdir(subj_path)):
            clip_path = os.path.join(subj_path, clip_dir)
            if not os.path.isdir(clip_path): continue
            info_path = os.path.join(clip_path, "info.txt")
            flow_npy = os.path.join(clip_path, "flow.npy")
            flow_flip_npy = os.path.join(clip_path, "flow_flip.npy")
            
            if not os.path.isfile(info_path): continue
            info = _parse_info_txt(info_path)
            raw_emotion = info.get("emotion")
            if raw_emotion not in TARGET_CLASSES: continue
            
            emotion_id = TARGET_CLASSES[raw_emotion]
            records.append({
                "subject": subj_dir,
                "clip_folder": clip_path,
                "emotion_id": emotion_id,
                "flow_npy": flow_npy,
                "flow_flip_npy": flow_flip_npy
            })
    return pd.DataFrame(records)

def evaluate(model, loader, criterion, device, num_classes=3):
    model.eval()
    all_preds, all_labels = [], []
    val_loss = 0.0
    with torch.no_grad():
        for (flow_x, spatial_x), labels in loader:
            flow_x = flow_x.to(device)
            spatial_x = spatial_x.to(device)
            labels = labels.to(device)
            
            logits = model(flow_x, spatial_x)
            loss = criterion(logits, labels)
            val_loss += loss.item()
            
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    val_loss /= len(loader)
    return val_loss, all_labels, all_preds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/scratch/smiyyapuram/medusa/casme_raft_processed10')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--folds', type=int, default=10)
    parser.add_argument('--out_dir', type=str, default='save_models')
    parser.add_argument('--frame_type', type=str, default='apex')
    parser.add_argument('--channels', type=str, default='rgb')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    df = load_casme_processed(args.data_root)
    print(f"Loaded {len(df)} samples")
    
    subjects = df['subject'].unique()
    kf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    
    y_full = df['emotion_id'].values
    
    fold_results = []
    history_by_fold = {}
    fold_confusions = {}
    overall_targets = []
    overall_preds = []
    overall_history = {}
    best_fold = 1
    best_fold_uf1 = -1
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(df, y_full)):
        print(f"Fold {fold+1}/{args.folds}")
        
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()
        
        # Class weights calculation
        num_classes = 3
        class_counts = np.zeros(num_classes)
        for eid, cnt in train_df['emotion_id'].value_counts().items():
            class_counts[int(eid)] = cnt
        class_counts = np.where(class_counts == 0, 1, class_counts)
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum() * num_classes
        class_weights_t = torch.FloatTensor(class_weights).to(device)
        
        # Weighted sampler
        sample_weights = [class_weights[int(y)] for y in train_df['emotion_id'].values]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        
        train_ds = CasmeDualSwinDataset(train_df, augment=True, frame_type=args.frame_type, frame_channels=args.channels)
        val_ds = CasmeDualSwinDataset(val_df, augment=False, frame_type=args.frame_type, frame_channels=args.channels)
        
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=4, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        in_channels = 1 if args.channels == 'gray' else 3
        model = DualStreamModel(num_classes=3, in_channels=in_channels).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6)
        criterion = WeightedFocalLoss(weight=class_weights_t, gamma=2.0)
        
        best_uar = -1.0
        best_uf1 = 0
        best_acc = 0
        best_score = 0
        best_cm = None
        best_preds = []
        best_labels = []
        
        fold_history = []
        
        for epoch in range(args.epochs):
            model.train()
            train_loss = 0
            for (flow_x, spatial_x), labels in train_loader:
                flow_x = flow_x.to(device)
                spatial_x = spatial_x.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                logits = model(flow_x, spatial_x)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
            train_loss /= len(train_loader)
            val_loss, all_labels, all_preds = evaluate(model, val_loader, criterion, device)
            
            summ = summarize_predictions(all_labels, all_preds)
            epoch_history = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": summ["acc"],
                "val_uar": summ["uar"],
                "val_uf1": summ["uf1"],
                "val_score": summ["score"]
            }
            fold_history.append(epoch_history)
            
            is_best = False
            if summ['uf1'] > best_uf1:
                is_best = True
                best_uar = summ['uar']
                best_uf1 = summ['uf1']
                best_acc = summ['acc']
                best_score = summ['score']
                best_cm = summ['confusion_matrix']
                best_preds = all_preds
                best_labels = all_labels
                torch.save(model.state_dict(), os.path.join(args.out_dir, f"best_model_fold{fold+1}.pth"))
                
            best_marker = "(*Best*)" if is_best else ""
            print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {summ['acc']:.4f} | Val UAR: {summ['uar']:.4f} | Val UF1: {summ['uf1']:.4f} {best_marker}")
            
            scheduler.step(summ['uar'])
                
        history_by_fold[fold+1] = pd.DataFrame(fold_history)
        overall_history[fold+1] = fold_history
        fold_confusions[fold+1] = best_cm
        overall_targets.extend(best_labels)
        overall_preds.extend(best_preds)
        
        fold_results.append({
            'fold': fold + 1,
            'best_acc': best_acc,
            'best_uar': best_uar,
            'best_uf1': best_uf1,
            'best_score': best_score
        })
        
        if best_uf1 > best_fold_uf1:
            best_fold_uf1 = best_uf1
            best_fold = fold + 1
            
        print(f"Fold {fold+1} Best UAR: {best_uar:.4f} | Best UF1: {best_uf1:.4f}")
        
    res_df = pd.DataFrame(fold_results)
    res_df.to_csv(os.path.join(args.out_dir, "results.csv"), index=False)
    
    overall_summary = summarize_predictions(overall_targets, overall_preds)
    
    print("-" * 60)
    print(f"Mean Best UAR: {res_df['best_uar'].mean():.4f}")
    print(f"Overall UAR: {overall_summary['uar']:.4f}")
    print("-" * 60)
    
    try:
        exp_name = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
        output_png = os.path.join(args.out_dir, f"{exp_name}_summary.png")
        create_summary_figure(output_png, res_df, history_by_fold, fold_confusions, overall_summary, best_fold, args.folds)
        print(f"Saved plot to {output_png}")
    except Exception as e:
        print(f"Failed to create summary figure: {e}")

if __name__ == '__main__':
    main()
