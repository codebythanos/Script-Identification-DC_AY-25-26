# ResNet-based Script Identification

This directory contains multiple implementations of ResNet-based models for Indian language script identification. Each model version represents a different configuration or improvement over the baseline architecture.

---

## Overview

The models in this folder are based on Convolutional Neural Networks using ResNet architectures. These models are designed to classify text images into one of 12 Indian script classes.

Each implementation explores variations such as:
- Architectural modifications
- Data augmentation techniques
- Training and optimization strategies

---

## Directory Structure
```bash
ResNet/
│
├── Model1.py
├── Model2.py
├── Model3.py
├── p1_50k_best.keras
├── requirement_model1.txt
├── requirement_model2.txt
├── requirement_model3.txt
├── README.md
```
---

## Model Variants

- **Model1.py**  
  Baseline ResNet model for script classification.

- **Model2.py**  
  Improved version with modifications in architecture or training pipeline.

- **Model3.py**  
  Advanced implementation including enhancements such as data augmentation or optimized backbone usage.

---
## Installation

Navigate to this directory:

```bash
cd ResNet

Install dependencies for the required model:
```
Example for Model 1:
```bash
pip install -r requirement_model1.txt
```
Similarly:
```bash
Model2 → requirement_model2.txt
Model3 → requirement_model3.txt
```
Usage
Run the desired model:

Example:
```bash
python Model1.py
```
Similarly:
```bash
python Model2.py
python Model3.py
```
Dataset

Update dataset paths inside the script before running.
This is the link for data set :https://drive.google.com/file/d/1S7KUYfB-lQvbu6GtvZxxDPOD5ZZ080M0/view

Change the path for train and test dir in the .py scripts wherever required
Expected structure:
```bash
dataset/
├── train_1800/
├── test_478/
````
Each folder contains subdirectories corresponding to different script classes.
```bash
Features
ResNet-based convolutional models
Multiple training configurations
Support for data augmentation
Evaluation using confusion matrix and classification report
Requirements
Python 3.8+
TensorFlow or PyTorch (depending on implementation)
NumPy
Matplotlib
Seaborn
Scikit-learn
```
Refer to the specific requirement file for exact dependencies.

Notes
Each model version is independent and may use different techniques.
Always install the corresponding requirement file before running a model.
GPU is recommended for faster training.
