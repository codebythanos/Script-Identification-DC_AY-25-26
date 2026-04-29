# Vision Transformer (ViT) for Script Identification

This directory contains multiple implementations of Vision Transformer (ViT) models for Indian language script identification. Each model version represents different configurations and improvements for classification performance.

---

## Overview

The models in this folder are based on Vision Transformer architectures implemented using TensorFlow and `keras-hub`. These models process images as sequences of patches and perform classification into 12 Indian script classes.

Each implementation explores variations such as:
- Different training strategies
- Data augmentation and preprocessing techniques
- Hyperparameter tuning
- Backbone fine-tuning

---

## Directory Structure
```bash
ViT/
│
├── vit_Model1.py
├── vit_Model2.py
├── vit_Model3.py
├── requirement_vitmodel1.txt
├── requirement_vitmodel2.txt
├── requirement_vitmodel3.txt
├── README.md
```
---

## Model Variants

- **vit_Model1.py**  
  Baseline Vision Transformer model without extensive augmentation.

- **vit_Model2.py**  
  Improved version with better training pipeline and configuration.

- **vit_Model3.py**  
  Advanced implementation with optimization techniques, tuning, and improved performance.

---

## Installation

Navigate to this directory:

```bash
cd ViT
```
Install dependencies for the required model:

Example for Model 1:
```bash
pip install -r requirement_vitmodel1.txt
```
Similarly:
```bash
Model2 → requirement_vitmodel2.txt
Model3 → requirement_vitmodel3.txt
```
Usage
Run the desired model:

Example:
```bash
python vit_Model1.py
```
Similarly:
```bash
python vit_Model2.py
python vit_Model3.py
```
Dataset

Update dataset paths inside the script before running.

Expected structure:
```bash
dataset/
├── train_1800/
├── test_478/
```

Each folder contains subdirectories corresponding to script classes.

Features
```bash
Vision Transformer models using keras-hub
Patch-based image processing
Fine-tuning support
Evaluation using confusion matrix and classification report
Support for mixed precision and GPU acceleration
Requirements
Python 3.8+
TensorFlow
keras-hub
NumPy
Matplotlib
Seaborn
Scikit-learn
```
Refer to the specific requirement file for exact dependencies.

Notes
GPU is recommended for training, especially for transformer models.
Each model version is independent and may require separate dependencies.
Ensure dataset paths are correctly configured before execution.
