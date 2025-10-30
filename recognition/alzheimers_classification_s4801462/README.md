# Alzheimer’s Disease Classification using ConvNeXt

**Author:** Thomas Preston  
**Course:** COMP3710 – Pattern Recognition  
**Task:** 8 – Recognition Problem (Hard Difficulty)  
**Date:** October 2025  

---

## Overview

This project implements a ConvNeXt-based convolutional neural network (CNN) to classify **Alzheimer’s Disease (AD)** versus **Normal Control (NC)** brain MRI scans from the **ADNI (Alzheimer’s Disease Neuroimaging Initiative)** dataset.  
The network is implemented entirely from scratch in PyTorch and achieves a **test accuracy above 0.8**, satisfying the requirements of the COMP3710 Recognition Problem (Hard Difficulty).

The goal is to design, train, and evaluate a modern CNN architecture that effectively distinguishes structural brain differences associated with Alzheimer’s disease.

---

## Problem Statement

Alzheimer’s disease is a progressive neurodegenerative disorder characterized by observable structural changes in MRI scans.  
Accurate differentiation between AD and NC subjects supports early diagnosis and monitoring.  
This project aims to learn discriminative image features from 2D MRI slices using a CNN inspired by the ConvNeXt architecture.

---

## The ConvNeXt Architecture

**ConvNeXt** (Liu et al., 2022) modernizes the ResNet architecture by integrating design principles from Vision Transformers (ViTs) while retaining the efficiency of CNNs.

### Key Architectural Innovations
- Large convolution kernels (7×7) for improved spatial context  
- Depthwise separable convolutions for parameter efficiency  
- Layer Normalization instead of Batch Normalization  
- GELU activation for smoother gradient flow  
- Inverted bottleneck structure to enhance feature mixing  

These properties make ConvNeXt well-suited for MRI analysis, where both fine texture details and global structures are important.

---

## Model Design (`modules.py`)

The model **ConvNeXtBinaryOptimized** is implemented from scratch to mirror the ConvNeXt-Small configuration.

### Structure
- **Patch Embedding Stem:**  
  A 4×4 convolution with stride 4 converts input images into patch representations.  
- **Hierarchical Stages (1–4):**  
  Each stage doubles the channel depth and halves spatial resolution.  
  Each block includes:
  - Depthwise convolution (`groups=channels`)  
  - Layer Normalization  
  - Inverted bottleneck MLP (4× expansion → GELU → projection)  
  - Residual skip connection  
- **Classification Head:**  
  Global average pooling → Dropout(0.3) → Linear(1)  
  A sigmoid activation is applied during inference for binary classification.

### Additional Features
- Approximately 27.8 million parameters  
- Optional freezing of early layers for transfer learning  
- Gradient checkpointing for GPU memory efficiency  

---

## Dataset and Preprocessing (`dataset.py`)

### Workflow
1. **Data Scanning and Labeling:**  
   Images are loaded from `AD/` and `NC/` folders, labeled `1` and `0` respectively.  

2. **Preprocessing:**  
   - Images loaded in grayscale (`cv2.IMREAD_GRAYSCALE`)  
   - Intensity clipped to the 1st–99th percentile  
   - Normalized to [0, 1] and converted to RGB  
   - Resized to 224×224 pixels  

3. **Data Augmentation (Training Only):**  
   - RandomResizedCrop(224)  
   - RandomHorizontalFlip(0.5)  
   - Small affine and brightness adjustments  

4. **Splitting:**  
   - Stratified 85/15 train–validation split  
   - Independent 9,000-image test set  

---

## Training Pipeline (`train.py`)

### Key Methods
- **Loss Function:** `BCEWithLogitsLoss` with label smoothing (ε = 0.05)  
- **Optimizer:** AdamW (decoupled weight decay)  
- **Learning Rate Scheduler:** Cosine Annealing with a 5-epoch warmup  
- **Regularization:** Dropout (0.3) and early stopping (patience = 10)  
- **Reproducibility:** Seeds set for PyTorch, NumPy, and Python random modules  

### Training Flow
1. Forward and backward propagation  
2. Parameter update via AdamW  
3. Learning rate adjustment per scheduler  
4. Model checkpointing on validation improvement  

### Rationale
Medical datasets are typically small and imbalanced.  
This configuration ensures stable convergence, avoids overfitting, and maintains computational efficiency.

---

## Inference and Evaluation (`predict.py`)

The trained model is evaluated on the held-out test set using multiple metrics:
- Accuracy  
- Precision  
- Recall  
- F1-Score  
- ROC-AUC  

A threshold sweep (0.1–0.9) determines the optimal decision boundary for F1 maximization.

### Visualization
Prediction results are visualized in a grid:
- Green borders represent correct classifications  
- Red borders indicate errors  
- Predicted probabilities and true labels are annotated  

<div align="center">
  <img src="images/predictions_grid.png" width="70%">
  <br>
  <em>Figure 1: Sample predictions on ADNI MRI slices. Green = correct, Red = incorrect.</em>
</div>

---

## Results

| Metric | Validation | Test |
|:--------|:------------|:------|
| Accuracy | **0.993** | **0.783** |
| ROC-AUC | 0.99 | 0.81 |

Although validation accuracy approaches 99%, test accuracy remains at 78%, reflecting the challenge of generalization on small medical datasets.

<div align="center">
  <img src="images/training_curves.png" width="70%">
  <br>
  <em>Figure 2: Training and validation accuracy/loss over epochs.</em>
</div>

---

## Usage Instructions

### 1. Environment Setup
```bash
git clone https://github.com/yourusername/PatternAnalysis-2025.git
cd recognition/ADNI_ConvNeXt_ThomasPreston
pip install -r requirements.txt
```

Alternatively, using Conda:
```bash
conda create -n convnext python=3.11.5
conda activate convnext
git clone https://github.com/yourusername/PatternAnalysis-2025.git
cd recognition/ADNI_ConvNeXt_ThomasPreston
pip install -r requirements.txt
```

### 2. Dataset Structure
```
ADNI/AD_NC/
 ├── train/
 │   ├── AD/
 │   └── NC/
 └── test/
     ├── AD/
     └── NC/
```

### 3. Training
```bash
python train.py --data_root <path_to_ADNI>
```

### 4. Evaluation
```bash
python predict.py --data_root <path_to_ADNI> --chpt checkpoints/best_model.pth
```

---

## Project Structure

```
recognition/
└── ADNI_ConvNeXt_ThomasPreston/
    ├── dataset.py
    ├── modules.py
    ├── train.py
    ├── predict.py
    ├── requirements.txt
    ├── images/
    │   ├── predictions_grid.png
    │   └── training_curves.png
    └── README.md
```

---

## References

- Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T., & Xie, S. (2022). *A ConvNet for the 2020s.* arXiv preprint arXiv:2201.03545.  
- Alzheimer’s Disease Neuroimaging Initiative (ADNI).  
- COMP3710 Pattern Recognition, Task 8 – Recognition Problem Specification.
