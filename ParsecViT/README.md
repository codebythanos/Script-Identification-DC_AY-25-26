# PARSeq-style Vision Transformer for Script Identification

This directory contains multiple implementations of a PARSeq-style Vision Transformer model for Indian language script identification. Each model version explores different configurations and improvements over the baseline.

---

## Overview

The models in this folder are based on Vision Transformer (ViT) architectures and are implemented using PyTorch and the `timm` library. These models are designed to classify text images into one of 12 Indian script classes.

Each version includes:
- Training and evaluation pipeline
- Patch-based ViT configurations
- Performance comparison across different settings

---

## Directory Structure


ParseqViT/
│
├── Model1_parseq.py
├── Model2_parseq.py
├── Model3_parseq.py
├── requirement_Model1_parseq.txt
├── requirement_Model2_parseq.txt
├── requirement_Model3_parseq.txt
├── README.md


---

## Model Variants

- **Model1_parseq.py**  
  Baseline PARSeq-style Vision Transformer model.

- **Model2_parseq.py**  
  Modified version with improvements such as better training strategy or architecture changes.

- **Model3_parseq.py**  
  Advanced implementation including patch-size experiments, attention visualization, and performance comparison.

---

## Installation

Navigate to this directory:

```bash
cd ParseqViT
```

Install dependencies for the required model:

Example for Model 3:
```bash
pip install -r requirement_Model3_parseq.txt
```
Similarly:

Model1 → requirement_Model1_parseq.txt
Model2 → requirement_Model2_parseq.txt
Usage

Run the desired model:

Example:

```bash
python Model3_parseq.py
```
Similarly:

python Model1_parseq.py
python Model2_parseq.py
Dataset

Update dataset paths inside the script before running.

Expected structure:

dataset/
├── train_1800/
├── test_478/

Each folder contains subdirectories corresponding to script classes.

Features
Vision Transformer models using timm
Patch-size experimentation (patch8, patch16, patch32)
GPU support with mixed precision
Confusion matrix and classification report
Attention visualization using heatmaps
Requirements
Python 3.8+
PyTorch
timm
NumPy
Matplotlib
Seaborn
Scikit-learn
Pillow
OpenCV

Refer to the specific requirement file for exact dependencies.

Notes
GPU is strongly recommended for training.
Ensure dataset paths are correctly configured in each script.
Each model version is independent and may require separate installation.
