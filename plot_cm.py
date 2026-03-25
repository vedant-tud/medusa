import os
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
    script_path = "trainKfold_dual_swin_flexible.py" if args.model_type == "swin" else "trainKfold_dual_cnn_flexible.py"
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
    
    all_true = []
    all_pred = []
    
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
        
        # Determine exact expected save path from original script logic
        # Original: save_path = f"./{save_dir}/best_focal_loss_dual_{args.model_type}_fold{fold_idx+1}_{config['frame_type']}_{config['frame_channels']}.pth"
        save_path = f"./{save_dir}/best_focal_loss_dual_{args.model_type}_fold{fold_idx+1}_{config['frame_type']}_{config['frame_channels']}.pth"
        
        if not os.path.exists(save_path):
            print(f"Warning: Missing saved model for fold {fold_idx+1}: {save_path}")
            continue
            
        # Add weights_only=False to address warnings if possible or just use default
        try:
            checkpoint = torch.load(save_path, map_location=device, weights_only=False)
            state_dict = checkpoint.get("model", checkpoint)
            model.load_state_dict(state_dict)
        except TypeError:
            model.load_state_dict(torch.load(save_path, map_location=device))
        
        model.eval()
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
                
                all_true.extend(target.cpu().numpy())
                all_pred.extend(preds.cpu().numpy())
                
    if len(all_true) == 0:
        print("No validation data collected. Aborting cm generation.")
        return
        
    cm = confusion_matrix(all_true, all_pred)
    # The classes in CASME are typically: 0: negative, 1: positive, 2: surprise (or similar depending on mapping)
    # the mapping isn't explicitly printed here but let's use integers directly if needed, or get them from train_mod
    mapping = {0: "negative", 1: "positive", 2: "surprise"}
    class_names = [mapping.get(i, f"Class {i}") for i in range(config["num_classes"])]
    
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix (10-Fold CV)\nDual {args.model_type.upper()} | Frame: {args.frame_type} | Ch: {args.frame_channels}")
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    out_img = f"cm_dual_{args.model_type}_{args.frame_type}_{args.frame_channels}.png"
    plt.savefig(out_img, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved Confusion Matrix to: {out_img}")
    
    # Print numerical confusion matrix
    print("\nText Confusion Matrix (Row=True, Col=Pred):")
    print("Classes:", class_names)
    print(cm)
    
if __name__ == "__main__":
    main()
