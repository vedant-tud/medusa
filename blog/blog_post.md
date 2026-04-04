# Medusa MER Experiments: What Worked, What Broke, and Why

## Introduction
Micro-expression recognition (MER) is a hard setting: tiny datasets, subtle facial motion, and severe class imbalance. This report summarizes your available results across Exp00 (Dual-CNN baseline) and experiments Exp01, Exp02, Exp03, Exp04, Exp05, and Exp06.

## Experimental Setup
- Dataset setup: CASME clips mapped to 3 classes (Positive, Negative, Surprise).
- Inputs: apex spatial frame plus optical flow-derived representation.
- Validation: 10-fold stratified CV for most runs.
- Core metric: UAR (unweighted average recall), which is more reliable than raw accuracy under imbalance.
- Lossing strategy in Exp01/Exp02/Exp03/Exp04/Exp05/Exp06: weighted focal loss with weighted sampling.

Important comparison caveats:
- Checkpoint selection differs across experiments.
- Exp01 saves best epoch by UAR.
- Exp02/03/04/05/06 save best epoch by UF1.
- Exp00 Dual-CNN summary rows are from its own SCL + auxiliary-flow training pipeline.

## Models Tried
- Exp00 Dual-CNN (RGB): dual-stream lightweight CNN with supervised contrastive and auxiliary flow reconstruction.
- Exp00 Dual-CNN (Gray): same architecture with grayscale spatial input.
- Exp01 ResNet50-Flow + SwinV2-Spatial: dual-stream fusion.
- Exp02 SwinV2-Flow + ResNet50-Spatial: dual-stream fusion.
- Exp03 ConvNeXt-Flow + ViT-Spatial: dual-stream fusion.
- Exp04 VGGFace2-Flow + Swin-Spatial: dual-stream fusion.
- Exp05 Single Swin Apex: spatial-only ablation.
- Exp06 Single ResNet Flow: flow-only ablation.

## Results

### Aggregated Metrics (mean of per-fold best rows)

| Experiment | Mean Best UAR | Mean Best UF1 | Mean Best Acc | UAR Std | Best Fold UAR |
|---|---:|---:|---:|---:|---:|
| Exp02 SwinV2-Flow + ResNet50-Spatial | 0.742 | 0.577 | 0.763 | 0.064 | 0.863 |
| Exp01 ResNet50-Flow + SwinV2-Spatial | 0.730 | 0.553 | 0.726 | 0.065 | 0.845 |
| Exp04 VGGFace2-Flow + Swin-Spatial | 0.715 | 0.563 | 0.735 | 0.057 | 0.796 |
| Exp05 Single Swin Apex | 0.692 | 0.667 | 0.883 | 0.081 | 0.864 |
| Exp06 Single ResNet Flow | 0.572 | 0.532 | 0.786 | 0.051 | 0.660 |
| Exp03 ConvNeXt-Flow + ViT-Spatial | 0.524 | 0.424 | 0.565 | 0.113 | 0.773 |
| Exp00 Dual-CNN (RGB) | 0.458 | 0.417 | 0.736 | 0.048 | 0.549 |
| Exp00 Dual-CNN (Gray) | 0.449 | 0.390 | 0.706 | 0.047 | 0.545 |

### Quick ranking by UAR
1. Exp02
2. Exp01
3. Exp04
4. Exp05
5. Exp06
6. Exp03
7. Exp00 Dual-CNN RGB
8. Exp00 Dual-CNN Gray


## Analysis & Insights

### 1) Dual-stream still matters
Exp06 (flow-only) trails badly in UAR versus Exp01/02/04, which supports a familiar MER pattern: motion alone is not enough. You need stable facial structure context for subtle micro-events.

### 2) Spatial-only can look strong on paper, but watch the metric
Exp05 posts very high accuracy and strong UF1, yet lower UAR than Exp02/01. With imbalance, this often means the model is doing well on dominant classes but not balancing recalls equally across all classes.

### 3) Pretraining choice helped, but not equally
Exp02 leads UAR, with Exp01 close behind. Exp04 (VGGFace2 variant) is competitive but slightly lower on UAR. This suggests the backbone pairing and modality alignment matter more than just "using a face-pretrained model".

### 4) Code-path confound likely affected some runs
In Exp01/Exp04 code paths, flow/spatial stream semantics appear partially swapped relative to comments and naming. Since tensor shapes are compatible, training still runs, but the intended inductive bias can shift. Treat cross-experiment deltas as directional, not absolute.

### 5) Why Exp00 Dual-CNN is lower here
The Exp00 Dual-CNN results are from a different training recipe (SCL + auxiliary reconstruction + different optimization behavior). They are useful as a baseline family, but not a strict apples-to-apples replacement for Exp01/Exp02/Exp04/Exp05/Exp06.

### 6) Exp03 status
Exp03 now has complete 10-fold results, with mean UAR 0.524 and high fold variance (UAR std 0.113). Its best fold reaches 0.773 UAR, but overall consistency is weaker than Exp01/02/04.

## Conclusion
If your goal is balanced class recall, Exp02 is currently the best-performing complete run in this workspace. Exp01 and Exp04 are close but slightly behind on UAR. Exp03 is now fully scored, but trails these top dual-stream runs with larger fold-to-fold instability. Single-stream ablations confirm the expected trend: spatial-only is stronger than flow-only, but dual-stream remains the safer path for robust MER.

Practically, the next best move is to standardize selection criteria (UAR vs UF1), re-run a clean apples-to-apples sweep, and then lock conclusions from that controlled comparison.

## Updated Visual Package

The visuals below are now sourced from the dedicated share folder:

- ./ (this folder)

### Balanced Subject Comparisons (Exp00-Exp06)

These are the per-subject cross-experiment comparison panels from the balanced set (3 Positive, 3 Negative, 3 Surprise).

![Comparison sub159 j_1176_1220](comparisons/sub159_sub159_j_1176_1220_comparison.png)
![Comparison sub162 c_303_374](comparisons/sub162_sub162_c_303_374_comparison.png)
![Comparison sub184 j_643_653](comparisons/sub184_sub184_j_643_653_comparison.png)
![Comparison sub186 j_1187_1196](comparisons/sub186_sub186_j_1187_1196_comparison.png)
![Comparison sub187 l_4178_4188](comparisons/sub187_sub187_l_4178_4188_comparison.png)
![Comparison sub202 g_2676_2696](comparisons/sub202_sub202_g_2676_2696_comparison.png)
![Comparison sub207 k_27_42](comparisons/sub207_sub207_k_27_42_comparison.png)
![Comparison sub213 g_5_24](comparisons/sub213_sub213_g_5_24_comparison.png)
![Comparison sub216 h_2189_2199](comparisons/sub216_sub216_h_2189_2199_comparison.png)

### Confusion Matrices By Experiment

#### Exp00 Dual-CNN (RGB)
![Exp00 RGB Confusion Matrix (generated from exp_00_dual_cnn/save_dual_cnn)](confusion_matrices/Exp00/Exp00_Dual_CNN_RGB_best_fold_cm.png)

#### Exp00 Dual-CNN (Gray)
![Exp00 Gray Confusion Matrix (generated from exp_00_dual_cnn/save_dual_cnn)](confusion_matrices/Exp00/Exp00_Dual_CNN_Gray_best_fold_cm.png)

#### Exp01 ResNet50-Flow + SwinV2-Spatial
![Exp01 Confusion Matrix](confusion_matrices/Exp01/Exp01_summary_with_cm.png)

#### Exp02 SwinV2-Flow + ResNet50-Spatial
![Exp02 Confusion Matrix](confusion_matrices/Exp02/Exp02_summary_with_cm.png)

#### Exp03 ConvNeXt-Flow + ViT-Spatial
![Exp03 Confusion Matrix](confusion_matrices/Exp03/Exp03_summary_with_cm.png)

#### Exp04 VGGFace2-Flow + Swin-Spatial
![Exp04 Confusion Matrix](confusion_matrices/Exp04/Exp04_summary_with_cm.png)

#### Exp05 Single Swin Apex
![Exp05 Confusion Matrix](confusion_matrices/Exp05/Exp05_summary_with_cm.png)

#### Exp06 Single ResNet Flow
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