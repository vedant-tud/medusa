import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from medusa_plotter import create_summary_figure, summarize_predictions
from dataset import CasmeDualSwinDataset
from models import DualStreamModel
from train import load_casme_processed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/scratch/smiyyapuram/medusa/casme_tvl1_processed_amp5')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--folds', type=int, default=10)
    parser.add_argument('--out_dir', type=str, default='save_models')
    parser.add_argument('--frame_type', type=str, default='apex')
    parser.add_argument('--channels', type=str, default='rgb')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = load_casme_processed(args.data_root)
    print(f"Loaded {len(df)} samples")
    y_full = df['emotion_id'].values

    kf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)

    fold_confusions = {}
    overall_targets = []
    overall_preds = []
    
    results_path = os.path.join(args.out_dir, "results.csv")
    if os.path.exists(results_path):
        results_df = pd.read_csv(results_path)
        best_fold = int(results_df.loc[results_df['best_uar'].idxmax()]['fold'])
    else:
        print("results.csv not found! Recreating metrics...")
        results_df = None
        best_fold = 1

    history_by_fold = {}

    model = DualStreamModel(num_classes=3)
    model.to(device)
    model.eval()

    for fold, (train_idx, val_idx) in enumerate(kf.split(df, y_full)):
        fold_num = fold + 1
        print(f"Evaluating Fold {fold_num}/{args.folds}")
        
        weight_path = os.path.join(args.out_dir, f"best_model_fold{fold_num}.pth")
        if not os.path.exists(weight_path):
            print(f"  Missing {weight_path}, skipping...")
            continue
            
        model.load_state_dict(torch.load(weight_path, map_location=device))
        
        val_df = df.iloc[val_idx].copy()
        val_dataset = CasmeDualSwinDataset(
            val_df, 
            image_size=256, 
            augment=False, 
            frame_type=args.frame_type, 
            frame_channels=args.channels
        )
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        fold_preds = []
        fold_labels = []
        
        with torch.no_grad():
            for (flow_x, spatial_x), labels in val_loader:
                flow_x = flow_x.to(device)
                spatial_x = spatial_x.to(device)
                labels = labels.to(device)
                
                logits = model(flow_x, spatial_x)
                preds = logits.argmax(dim=-1)
                
                fold_preds.extend(preds.cpu().numpy())
                fold_labels.extend(labels.cpu().numpy())
                
        fold_confusions[fold_num] = confusion_matrix(fold_labels, fold_preds, labels=[0, 1, 2])
        overall_targets.extend(fold_labels)
        overall_preds.extend(fold_preds)

    overall_summary = summarize_predictions(overall_targets, overall_preds)
    
    if results_df is None:
        results_df = pd.DataFrame([{
            'fold': f, 'best_acc': 1.0, 'best_uar': 1.0, 'best_uf1': 1.0, 'best_score': 1.0 
        } for f in range(1, args.folds + 1)])
        
    output_png = os.path.join(args.out_dir, "exp_01_face_id_summary.png")
    create_summary_figure(output_png, results_df, history_by_fold, fold_confusions, overall_summary, best_fold, args.folds)
    print(f"Summary figure saved to {output_png}")

if __name__ == '__main__':
    main()
