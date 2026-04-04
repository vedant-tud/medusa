# Share Visual Package (Exp00-Exp06)

This folder contains the final visuals prepared for sharing.

## Open This README
Open this file in VS Code and use Markdown Preview to see the visuals rendered inline.

## Blog Post
Open [blog_post.md](blog_post.md) in this same folder to view the full report with local image links.

## Folder Layout
- `comparisons/`
  - 9 balanced per-subject comparison panels (Exp00-Exp06)
  - `comparison_index.csv`
- `confusion_matrices/`
  - `Exp00/`: best-fold confusion matrices for Dual-CNN RGB and Gray
  - `Exp01` to `Exp06/`: summary figures that include confusion matrix panels
- `flow_comparisons/`
  - RAFT vs TV-L1 visuals
  - TV-L1 variant comparisons
  - AMP comparison visuals
- `blog_post.md`
  - full narrative report, adjusted to render from this folder

## Balanced Subject Comparisons (Exp00-Exp06)

![Comparison sub159 j_1176_1220](comparisons/sub159_sub159_j_1176_1220_comparison.png)
![Comparison sub162 c_303_374](comparisons/sub162_sub162_c_303_374_comparison.png)
![Comparison sub184 j_643_653](comparisons/sub184_sub184_j_643_653_comparison.png)
![Comparison sub186 j_1187_1196](comparisons/sub186_sub186_j_1187_1196_comparison.png)
![Comparison sub187 l_4178_4188](comparisons/sub187_sub187_l_4178_4188_comparison.png)
![Comparison sub202 g_2676_2696](comparisons/sub202_sub202_g_2676_2696_comparison.png)
![Comparison sub207 k_27_42](comparisons/sub207_sub207_k_27_42_comparison.png)
![Comparison sub213 g_5_24](comparisons/sub213_sub213_g_5_24_comparison.png)
![Comparison sub216 h_2189_2199](comparisons/sub216_sub216_h_2189_2199_comparison.png)

## Confusion Matrices by Experiment

### Exp00 Dual-CNN (RGB)
![Exp00 RGB Confusion Matrix](confusion_matrices/Exp00/Exp00_Dual_CNN_RGB_best_fold_cm.png)

### Exp00 Dual-CNN (Gray)
![Exp00 Gray Confusion Matrix](confusion_matrices/Exp00/Exp00_Dual_CNN_Gray_best_fold_cm.png)

### Exp01 ResNet50-Flow + SwinV2-Spatial
![Exp01 Confusion Matrix](confusion_matrices/Exp01/Exp01_summary_with_cm.png)

### Exp02 SwinV2-Flow + ResNet50-Spatial
![Exp02 Confusion Matrix](confusion_matrices/Exp02/Exp02_summary_with_cm.png)

### Exp03 ConvNeXt-Flow + ViT-Spatial
![Exp03 Confusion Matrix](confusion_matrices/Exp03/Exp03_summary_with_cm.png)

### Exp04 VGGFace2-Flow + Swin-Spatial
![Exp04 Confusion Matrix](confusion_matrices/Exp04/Exp04_summary_with_cm.png)

### Exp05 Single Swin Apex
![Exp05 Confusion Matrix](confusion_matrices/Exp05/Exp05_summary_with_cm.png)

### Exp06 Single ResNet Flow
![Exp06 Confusion Matrix](confusion_matrices/Exp06/Exp06_summary_with_cm.png)

## Flow Method Comparisons (Exp00)

### RAFT vs TV-L1
![Exp00 RAFT masked comparison](flow_comparisons/raft_masked_comparison.png)
![Exp00 TV-L1 comparison](flow_comparisons/tvl1_comparison.png)
![Exp00 fixed-flow comparison](flow_comparisons/fixed_flow_comparison.png)

### TV-L1 Variant Comparisons
![Exp00 pure TV-L1 comparison](flow_comparisons/pure_tvl1_comparison.png)
![Exp00 blur TV-L1 comparison](flow_comparisons/blur_tvl1_comparison.png)
![Exp00 TV-L1 no-mask comparison](flow_comparisons/tvl1_no_mask_comparison.png)

### AMP Comparisons
![Exp00 AMP comparison](flow_comparisons/amp_comparison.png)
![Exp00 detailed AMP comparison](flow_comparisons/detailed_amp_comparison.png)
![Exp00 multiple AMP comparison](flow_comparisons/multiple_amps_comparison.png)

## Note
Exp04 source summary filename in its original experiment folder was `exp_01_face_id_summary.png`, and it is copied here as `Exp04_summary_with_cm.png` for consistent naming.

Exp00 RGB/Gray confusion matrix images in this package were regenerated from `exp_00_dual_cnn/save_dual_cnn` via `exp_00_dual_cnn/plot_cm.py` and then copied here.
