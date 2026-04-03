import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
from sklearn.metrics import confusion_matrix, f1_score, recall_score

CLASS_NAMES = ["positive", "negative", "surprise"]

def compute_score(acc: float, uar: float, uf1: float) -> float:
    return 0.2 * float(acc) + 0.4 * float(uar) + 0.4 * float(uf1)

def summarize_predictions(targets, preds, num_classes=3):
    targets_array = np.asarray(targets, dtype=np.int64)
    preds_array = np.asarray(preds, dtype=np.int64)
    accuracy = float(np.mean(targets_array == preds_array)) if len(targets_array) else 0.0
    uar = recall_score(targets_array, preds_array, average="macro", zero_division=0)
    uf1 = f1_score(targets_array, preds_array, average="macro", zero_division=0)
    score = compute_score(accuracy, uar, uf1)
    cm = confusion_matrix(targets_array, preds_array, labels=list(range(num_classes)))
    return {"acc": accuracy, "uar": uar, "uf1": uf1, "score": score, "confusion_matrix": cm}

def create_summary_figure(output_path, results_df, history_by_fold, fold_confusions, overall_summary, best_fold, k):
    import warnings
    warnings.filterwarnings("ignore")
    total_plots = 3 + k + 2
    cols = 4
    rows = math.ceil(total_plots / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.2, rows * 4.2))
    axes = np.atleast_1d(axes).flatten()

    ax = axes[0]
    metric_cols = ["best_acc", "best_uar", "best_uf1", "best_score"]
    for metric in metric_cols:
        if metric in results_df.columns:
            ax.plot(results_df["fold"], results_df[metric], marker="o", label=metric.replace("best_", "").upper())
    ax.set_title("Best Metrics Per Fold")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Metric")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    best_history = history_by_fold.get(best_fold, pd.DataFrame())
    ax = axes[1]
    if not best_history.empty and "train_loss" in best_history.columns.values:
        ax.plot(best_history["epoch"], best_history["train_loss"], label="Train Loss", lw=2)
        ax.plot(best_history["epoch"], best_history["val_loss"], label="Val Loss", lw=2)
        ax.set_title(f"Best Fold {best_fold} Loss Curves")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    ax = axes[2]
    if not best_history.empty and "val_uar" in best_history.columns.values:
        ax.plot(best_history["epoch"], best_history.get("val_acc", best_history["val_uar"]), label="ACC")
        ax.plot(best_history["epoch"], best_history["val_uar"], label="UAR", lw=2)
        ax.plot(best_history["epoch"], best_history.get("val_uf1", best_history["val_uar"]), label="UF1")
        if "val_score" in best_history.columns:
            ax.plot(best_history["epoch"], best_history["val_score"], label="SCORE")
        ax.set_title(f"Best Fold {best_fold} Validation Curves")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Metric")
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    plot_offset = 3
    for fold_idx in range(1, k + 1):
        ax = axes[plot_offset + fold_idx - 1]
        if fold_idx in fold_confusions:
            row = results_df[results_df["fold"] == fold_idx].iloc[0]
            cm = fold_confusions[fold_idx]
            cm_norm = cm.astype('float') / np.maximum(cm.sum(axis=1, keepdims=True), 1)
            annot_data = np.empty_like(cm, dtype=object)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    annot_data[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]:.2f})"
            sns.heatmap(cm_norm, annot=annot_data, fmt="", cmap="Blues", 
                       xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cbar=False, ax=ax, vmin=0, vmax=1)
            ax.set_title(f"Fold {fold_idx}\nUAR {row['best_uar']:.4f} | UF1 {row['best_uf1']:.4f}", fontsize=10)
            ax.set_xlabel("Pred")
            ax.set_ylabel("True")

    overall_ax = axes[plot_offset + k]
    if "confusion_matrix" in overall_summary:
        cm = overall_summary["confusion_matrix"]
        cm_norm = cm.astype('float') / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        annot_data = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot_data[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]:.2f})"
        sns.heatmap(cm_norm, annot=annot_data, fmt="", cmap="Oranges",
                   xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cbar=False, ax=overall_ax, vmin=0, vmax=1)
        overall_ax.set_title(f"Overall\nACC {overall_summary.get('acc',0):.4f} | UAR {overall_summary.get('uar',0):.4f}\nUF1 {overall_summary.get('uf1',0):.4f} | SCORE {overall_summary.get('score',0):.4f}", fontsize=10)
        overall_ax.set_xlabel("Pred")
        overall_ax.set_ylabel("True")

    best_ax = axes[plot_offset + k + 1]
    if best_fold in fold_confusions:
        best_row = results_df[results_df["fold"] == best_fold].iloc[0]
        cm = fold_confusions[best_fold]
        cm_norm = cm.astype('float') / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        annot_data = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot_data[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]:.2f})"
        sns.heatmap(cm_norm, annot=annot_data, fmt="", cmap="Greens",
                   xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cbar=False, ax=best_ax, vmin=0, vmax=1)
        best_ax.set_title(f"Best Fold {best_fold}\nACC {best_row.get('best_acc',0):.4f} | UAR {best_row.get('best_uar',0):.4f}\nUF1 {best_row.get('best_uf1',0):.4f} | SCORE {best_row.get('best_score',0):.4f}", fontsize=10)
        best_ax.set_xlabel("Pred")
        best_ax.set_ylabel("True")

    for axis in axes[plot_offset + k + 2 :]:
        axis.axis("off")

    mean_score = results_df['best_score'].mean() if 'best_score' in results_df else 0.0
    mean_uar = results_df['best_uar'].mean() if 'best_uar' in results_df else 0.0
    exp_name = Path(output_path).stem.replace("_results", "")
    
    fig.suptitle(f"{exp_name} MER | Mean SCORE {mean_score:.4f} | Mean UAR {mean_uar:.4f}", fontsize=16, y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

