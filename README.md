# Grid-Based UE Localization Using Deep Learning

This repository contains Python implementations of deep learning models for **grid-based User Equipment (UE) localization** using radio fingerprint data. The goal is to predict the most probable spatial grid of a UE based on serving and neighboring cell measurements.

Three different neural network architectures are implemented and evaluated for this task.

---

## Files in This Repository

- **Grid_Estimator_CNN.py**  
  Implements a Convolutional Neural Network (CNN)–based grid estimator.

- **Grid_Estimator_FFNN.py**  
  Implements a Feed-Forward Neural Network (FFNN)–based grid estimator.

- **Grid_Estimator_FFNN_ATTN.py**  
  Implements an FFNN with a self-attention mechanism for improved feature weighting.

---

## Problem Description

- **Task:** Grid-based localization (multi-class classification)
- **Input Features:**
  - Serving cell measurements: RSRP, RSRQ, RSSI, SINR
  - Neighbor cell measurements: RSRP, RSRQ, RSSI
- **Output:** The spatial coordinate of the UE. From this prediction of grid ID (nearest) corresponding to UE location.
- **Dataset Format:** JSON-based radio fingerprint dataset

The localization area is divided into fixed-size grids, and the model predicts which grid the UE belongs to.

---

## Model Architectures

### CNN-Based Grid Estimator
- Uses convolutional layers to extract local patterns from radio fingerprints
- Effective for structured and high-dimensional input features
- Includes batch normalization and dropout for regularization

### FFNN-Based Grid Estimator
- Fully connected neural network
- Lightweight and computationally efficient
- Serves as a baseline model for comparison

### FFNN with Attention
- Extends the FFNN using a self-attention layer
- Learns feature importance dynamically
- Improves performance in dense grid scenarios

---

## Training and Evaluation

Each model:
- Splits data into training and validation sets
- Uses Adam/AdamW optimizer
- Applies learning rate scheduling and early stopping
- Evaluates performance using:
  - Grid classification accuracy
  - Localization error (after grid-to-coordinate mapping)
  - Percentile-based error statistics

---

## Requirements

- Python 3.8 or higher
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

Install dependencies using:
```bash
pip install torch numpy pandas scikit-learn matplotlib
