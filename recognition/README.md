# Recognition Tasks
Various recognition tasks solved in deep learning frameworks.

Tasks may include:
* Image Segmentation
* Object detection
* Graph node classification
* Image super resolution
* Disease classification
* Generative modelling with StyleGAN and Stable Diffusion

# 🧠 Alzheimer's Classification using Custom ConvNeXt (COMP3710 Task 8)

**Author:** Thomas Preston  
**Course:** COMP3710 – Pattern Recognition  
**Task:** 8 – Recognition (ConvNeXt Implementation from Scratch)  
**Date:** October 2025  

---

## Overview

This project implements a **ConvNeXt-style convolutional neural network from scratch** to perform binary classification of brain MRI scans from the **ADNI dataset**, distinguishing between **Alzheimer’s Disease (AD)** and **Normal Control (NC)** patients.

The implementation follows the **Task 8 specification**, which requires students to build a ConvNeXt model manually (without using pretrained weights or `torchvision.models.convnext`).  

The model is trained and evaluated on MRI images using modern deep learning optimization techniques to improve stability, speed, and generalization.

---

## Dataset

The dataset used is a curated subset of the **ADNI (Alzheimer’s Disease Neuroimaging Initiative)** collection, consisting of preprocessed 2D MRI slices.

| Split | Samples | Description |
|:------|:---------|:-------------|
| Train | ~21,000 | Used for optimization (85 % of total data) |
| Validation | ~3,200 | Used for model selection (15 %) |
| Test | 9,000 | Unseen evaluation set |

### Directory Structure
ADNI/AD_NC/
    ├── train/
    │ ├── AD/ # Alzheimer's MRI scans
    │ └── NC/ # Normal control MRI scans
    ├── test/
    │ ├── AD/
    │ └── NC/
