# Experimental Report: Dual-Stream Micro-Expression Recognition (MER)

## Introduction
Building effective models for Micro-Expression Recognition (MER) is notoriously difficult. Micro-expressions are subtle, fleeting, and highly localized facial movements. Compounding the challenge is the extreme class imbalance typically found in MER datasets. In our latest experiments (v2), we aimed to address these challenges by exploring a Dual-Stream architecture. We compared a lightweight Custom Convolutional Neural Network (CNN) against a heavy Swin Transformer to see how different model complexities handle the nuanced, sparse data of micro-expressions. 

## Experimental Setup
Our experiments were conducted using a 10-fold cross-validation strategy on the CASME dataset. We condensed the target emotions into three distinct classes to ensure a manageable evaluation surface:
- **Positive:** Happiness
- **Negative:** Disgust, Anger, Fear, Sadness
- **Surprise:** Surprise

### Architecture: The Dual-Stream Approach
Both models were designed around a Dual-Stream framework to capture both structural geometry and localized temporal motion:
1. **Stream 1 (Temporal/Motion):** Optical Flow. For the CNN, we utilized raw U/V flow channels. For the Swin Transformer, optical flow was converted to RGB via HSV representations.
2. **Stream 2 (Spatial/Structural):** Extracted frame images (e.g., Apex frames in RGB).

### Handling Imbalance 
The dataset features a massive **30:1 class imbalance**, overwhelmingly skewed toward the "negative" class. To mitigate this, we utilized:
1. **Square-root inverse class frequency weighting** to scale losses without over-penalizing the majority class.
2. **Supervised Contrastive Learning (SCL) Loss** coupled with **Focal Loss** to push the models to learn better feature separations between the heavily outnumbered minority classes and the majority.

## Models Tried

### 1. SCL Dual-Stream Custom CNN
Given that traditional deep networks (like ResNet-18) tend to aggressively overfit on the minute CASME datasets, we built a **lightweight, custom CNN**. This network natively processes a smaller spatial footprint (112x112) using raw U/V flow and spatial frames. 

### 2. SCL Dual-Stream Swin Transformer
To see if self-attention mechanisms could better correlate distant facial action units, we tested a **Swin Transformer**. The Swin model processed higher-resolution images (256x256), leveraging a combination of RGB-rendered optical flow and spatial image frames.

---

## Results
We used **Unweighted Average Recall (UAR)** as our primary metric to ensure minority classes carried proper weight in the evaluation. The results reflect the aggregated performance across all 10 folds.

| Model | Overall UAR | Best Fold UAR | Top UAR Fold # |
| :--- | :--- | :--- | :--- |
| **SCL CNN (Apex, RGB)** | **0.4317** | **0.5413** | Fold 6 |
| **SCL SWIN (Apex, RGB)** | 0.3282 | 0.3424 | Fold 7 |

*Note: The CNN dramatically outperformed the Swin Transformer both overall and consistently across individual folds.*

### CNN Confusion Matrix Highlights (Overall)
- True Negatives: 1923
- True Positives: 101 out of 580 (~17% recall)
- True Surprises: 30 out of 390 (~8% recall)

### Swin Transformer Confusion Matrix Highlights (Overall)
- True Negatives: 2017
- True Positives: 46 out of 392 (~12% recall)
- True Surprises: 0 out of 74 (0% recall)

---

## Analysis & Insights

**1. The Triumph of Lightweight Architectures in MER**
The custom CNN's significant lead (UAR of 0.4317 vs 0.3282) highlights a well-known vulnerability in vision applied to MER: **over-parameterization leads to catastrophic overfitting or collapse**. The Swin Transformer, despite its powerful self-attention mechanisms, simply has too much capacity for a small, heavily imbalanced dataset like CASME. It requires vast amounts of data to learn inductive biases that standard spatial convolutions get "for free."

**2. The Impact of Class Imbalance**
Even with Focal Loss and careful inverse weighting, class imbalance stubbornly dictated the outputs. The Swin Transformer essentially collapsed into a majority-class predictor (predicting "Negative" almost universally, yielding a near 0% recall on the "Surprise" class). The lightweight CNN, constrained by its narrower architecture, was better forced to learn the minority class features, capturing significantly more True Positives and True Surprises.

**3. High Cross-Fold Variance**
Looking at the CNN's fold-by-fold UAR (ranging from 0.3447 in Fold 7 to 0.5413 in Fold 6), it's clear the test distributions vary heavily between subjects/splits. A single subject with highly expressive (or highly suppressed) micro-expressions can dramatically skew a fold's evaluation, showcasing why 10-fold CV is crucial in MER literature.

## Conclusion
Our v2 experiments yield a clear takeaway: **Bigger is not better for Micro-Expression Recognition**. 
The lack of massive, balanced datasets makes heavy architectures like the Swin Transformer unviable without extreme pre-training or synthetic data generation. A purpose-built, lightweight CNN that integrates raw optical flow and spatial frames remains the superior approach. Moving forward, shifting focus toward stronger synthetic minority oversampling or refining the Supervised Contrastive feature boundaries will likely yield the best improvements on top of our CNN baseline.
