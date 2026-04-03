import os

# --- Set Local Caching & Load .env ---
env_path = os.path.join(os.getcwd(), ".env")
if os.path.exists(env_path):
    with open(env_path, "r") as f:
        for line in f:
            if line.strip() and not line.startswith("#") and "=" in line:
                key, val = line.strip().split("=", 1)
                os.environ[key] = val

os.environ["HF_HOME"] = os.path.join(os.getcwd(), "hf_cache")
os.environ["TORCH_HOME"] = os.path.join(os.getcwd(), "torch_cache")

import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import importlib.util

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str, choices=["cnn", "swin"], required=True)
    parser.add_argument("--frame-type", type=str, choices=["onset", "apex"], required=True)
    parser.add_argument("--frame-channels", type=str, choices=["gray", "rgb"], required=True)
    parser.add_argument("--kfold", type=int, default=10)
    args = parser.parse_args()

    # Load proper script to reuse functions
    script_path = "trainKfold_scl_swin.py" if args.model_type == "swin" else "trainKfold_scl_cnn.py"
    train_mod = load_module("train_mod", script_path)
    
    config = train_mod.CONFIG.copy()
    config["frame_type"] = args.frame_type
    config["frame_channels"] = args.frame_channels
    device = train_mod.get_device()
    pin_memory = (device.type == "cuda")
    k = args.kfold
    
    # Generate identical dataset as training
    all_clips = train_mod.load_casme_processed(config["data_root"])
    annotations = all_clips.copy() # --all-data was used
    
    subjects = np.array(sorted(annotations["subject"].unique()))
    rng = np.random.default_rng(config["seed"])
    folds = np.array_split(rng.permutation(subjects), k)
    
    fold_metrics = {}
    best_fold = -1
    best_uar = -1.0
    best_cm = None
    best_true = []
    best_pred = []
    
    all_true = []
    all_pred = []
    
    all_fold_cms = {}
    
    spatial_chans = 1 if config['frame_channels'] == 'gray' else 3
    
    for fold_idx, val_subjects in enumerate(folds):
        val_df = annotations[annotations["subject"].isin(val_subjects)]
        if len(val_df) == 0: continue
            
        print(f"Loading Fold {fold_idx + 1}/{k} Validation Set ({len(val_df)} clips)...")
        # Reuse their dataset initialization
        if args.model_type == "swin":
            val_ds = train_mod.CasmeDualSwinDataset(val_df, config["image_size"], augment=False, frame_type=config['frame_type'], frame_channels=config['frame_channels'])
            model = train_mod.DualSwinMER(config["num_classes"], config["pretrained"], spatial_chans=spatial_chans).to(device)
            save_dir = "save_dual_swin"
        else:
            val_ds = train_mod.CasmeDualCNNDataset(val_df, config["image_size"], augment=False, frame_type=config['frame_type'], frame_channels=config['frame_channels'])
            model = train_mod.DualCnnMER(config["num_classes"], spatial_chans=spatial_chans).to(device)
            save_dir = "save_dual_cnn"
            
        val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, drop_last=False, num_workers=config["num_workers"], pin_memory=pin_memory)
        
        save_path = f"./{save_dir}/best_scl_dual_{args.model_type}_fold{fold_idx+1}_{config['frame_type']}_{config['frame_channels']}.pth"
        if not os.path.exists(save_path):
            print(f"Warning: Missing saved model for fold {fold_idx+1}: {save_path}")
            continue
            
        try:
            checkpoint = torch.load(save_path, map_location=device, weights_only=False)
            state_dict = checkpoint.get("model", checkpoint)
            model.load_state_dict(state_dict, strict=False)
        except TypeError:
            model.load_state_dict(torch.load(save_path, map_location=device), strict=False)
        
        model.eval()
        fold_true = []
        fold_pred = []
        with torch.no_grad():
            for batch in val_loader:
                inputs, target = batch
                if isinstance(inputs, list) or isinstance(inputs, tuple):
                    flow, spatial = inputs[0], inputs[1]
                else:
                    flow = inputs
                spatial, flow, target = spatial.to(device), flow.to(device), target.to(device)
                outputs = model(spatial, flow) if args.model_type == "swin" else model(flow, spatial)
                preds = torch.argmax(outputs, dim=1)
                
                fold_true.extend(target.cpu().numpy())
                fold_pred.extend(preds.cpu().numpy())
                
        # Calculate UAR for this fold
        from sklearn.metrics import recall_score
        uar = recall_score(fold_true, fold_pred, average='macro', zero_division=0)
        fold_metrics[fold_idx + 1] = uar
        print(f"Fold {fold_idx + 1} UAR: {uar:.4f}")
        
        all_true.extend(fold_true)
        all_pred.extend(fold_pred)

        fold_cm = confusion_matrix(fold_true, fold_pred)
        all_fold_cms[fold_idx + 1] = fold_cm
        print(f"Fold {fold_idx + 1} Text Confusion Matrix:")
        print(fold_cm)

        if uar > best_uar:
            best_uar = uar
            best_fold = fold_idx + 1
            best_true = fold_true
            best_pred = fold_pred
            best_cm = confusion_matrix(fold_true, fold_pred)
                
    if best_fold == -1:
        print("No validation data collected. Aborting cm generation.")
        return
        
    print(f"\nEvaluating Best Model: Fold {best_fold} with UAR {best_uar:.4f}")
    
    mapping = {0: "positive", 1: "negative", 2: "surprise"}
    class_names = [mapping.get(i, f"Class {i}") for i in range(config["num_classes"])]
    
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)
    
    # Calculate overall metrics
    from sklearn.metrics import recall_score
    overall_uar = recall_score(all_true, all_pred, average='macro', zero_division=0) if len(all_true) > 0 else 0.0
    overall_cm = confusion_matrix(all_true, all_pred) if len(all_true) > 0 else None
    
    # Setup subplots array (k folds + overall + best)
    total_plots = k + 2
    cols = 4
    import math
    rows = math.ceil(total_plots / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten()
    
    # Plot each fold
    for i in range(k):
        fold_num = i + 1
        ax = axes[i]
        if fold_num in all_fold_cms:
            sns.heatmap(all_fold_cms[fold_num], annot=True, fmt='d', cmap='Blues', 
                        xticklabels=class_names, yticklabels=class_names, ax=ax, cbar=False)
            ax.set_title(f"Fold {fold_num} (UAR: {fold_metrics.get(fold_num, 0):.4f})")
            ax.set_xlabel('Pred')
            ax.set_ylabel('True')
        else:
            ax.set_visible(False)
            
    # Plot Overall
    ax_overall = axes[k]
    if overall_cm is not None:
        sns.heatmap(overall_cm, annot=True, fmt='d', cmap='Oranges', 
                    xticklabels=class_names, yticklabels=class_names, ax=ax_overall, cbar=False)
        ax_overall.set_title(f"Overall Matrix (UAR: {overall_uar:.4f})")
        ax_overall.set_xlabel('Pred')
        ax_overall.set_ylabel('True')
    else:
        ax_overall.set_visible(False)
        
    # Plot Best Fold
    ax_best = axes[k+1]
    if best_cm is not None:
        sns.heatmap(best_cm, annot=True, fmt='d', cmap='Greens', 
                    xticklabels=class_names, yticklabels=class_names, ax=ax_best, cbar=False)
        ax_best.set_title(f"Best Fold: {best_fold} (UAR: {best_uar:.4f})")
        ax_best.set_xlabel('Pred')
        ax_best.set_ylabel('True')
    else:
        ax_best.set_visible(False)
        
    # Hide any remaining axes
    for i in range(k+2, len(axes)):
        axes[i].set_visible(False)
        
    plt.tight_layout()
    fig.suptitle(f"SCL {args.model_type.upper()} | Frame: {args.frame_type} | Ch: {args.frame_channels}", fontsize=16, y=1.02)
    
    out_img = os.path.join(out_dir, f"cm_combined_{args.model_type}_{args.frame_type}_{args.frame_channels}.png")
    plt.savefig(out_img, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved Combined Confusion Matrix to: {out_img}")
    print(f"Overall UAR: {overall_uar:.4f}")
    if overall_cm is not None:
        print("Overall Text Confusion Matrix:\n", overall_cm)

if __name__ == "__main__":
    main()
