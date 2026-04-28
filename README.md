# Script Identification for Indian Language Scene Text

This repository contains implementations of different deep learning models for identifying scripts in Indian language text images. The project explores multiple approaches to solve script classification using modern machine learning techniques.

---

## Overview

This repository provides implementations for three different script identification methods:

**ResNet:** A Convolutional Neural Network-based approach for image classification using residual learning.

**ViT:** A Vision Transformer-based model that processes images as sequences of patches for script identification tasks.

**ParseqViT:** A hybrid transformer-based architecture designed to improve performance for script recognition tasks.

Each method has its own folder containing scripts for training, testing, and inference. All models are compatible with Python environments, and each method has its own dependencies listed in the respective `requirements.txt` file.

---

## Installation

To get started, clone the repository and install the necessary dependencies for the respective method you wish to use.

### Clone the repository

```bash
git clone https://github.com/codebythanos/Script-Identification-DC_AY-25-26.git
cd Script-Identification-DC_AY-25-26
```

## Project Structure

The repository is divided into three main folders:

- `ResNet/`
- `ViT/`
- `ParseqViT/`

Each folder contains multiple implementations of the model.

### Example structure inside a folder

ResNet/  
│  
├── Model1.py  
├── Model2.py  
├── Model3.py  
├── requirement_model1.txt  
├── requirement_model2.txt  
├── requirement_model3.txt  
├── README.md  

---

## How to Use

Each model folder contains its own `README.md` file.

Follow these steps:

1. Go to the desired model folder:

2. Open and follow the instructions in that folder’s `README.md`.

---

## Running Individual Models

Each folder contains multiple versions of the model.

### Step 1: Install dependencies

Each model has its own requirement file.

Example:
```bash
pip install -r requirement_model1.txt
```

### Step 2: Run the model
```bash
python Model1.py
```

Similarly:

- Model2 → use `requirement_model2.txt`
- Model3 → use `requirement_model3.txt`

---

## Important Notes

- Each model version is independent and may use different architectures or hyperparameters.
- Always install the corresponding requirement file before running a model.
- Dataset paths should be correctly set inside the scripts before execution.
- GPU is recommended for training, especially for transformer-based models.

---

## Dataset

The models are trained on a 12-class Indian script classification dataset.

Typical structure:

dataset/  
├── train_1800/  
├── test_478/  

Each class is stored in separate folders.

---

## Evaluation

The models are evaluated using:

- Accuracy  
- Class-wise Accuracy  
- Confusion Matrix  
- Precision, Recall, F1-score  

---

## Requirements

Common dependencies across models:

- Python 3.8+
- TensorFlow
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- keras-hub (for ViT models)

Note: Exact dependencies may vary. Refer to the requirement files inside each model folder.

---

## Acknowledgements

This project is based on established deep learning architectures:

- Vision Transformer (ViT)
- ResNet
- Transformer-based models
- TensorFlow ecosystem

---

