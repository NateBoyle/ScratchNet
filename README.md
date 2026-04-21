# ScratchNet
# Neural Network from Scratch for PII Detection

A **from-scratch** multi-layer neural network built with only NumPy to classify whether a text contains Personally Identifiable Information (PII).

This project demonstrates a deep understanding of neural networks by implementing forward propagation, backpropagation, dropout, and training from the ground up — without using PyTorch, TensorFlow, or any high-level "black-box" libraries.

### Why This Project?
Most modern ML tutorials use high-level frameworks that hide the math. Here, I built everything manually using only NumPy to truly understand how neural networks work under the hood.

---

## Project Overview

- **Task**: Binary classification — Detect if masked text contains PII (1 = has PII, 0 = no PII)
- **Input**: Pre-trained GloVe word embeddings (300 dimensions)
- **Model**: Custom `ScratchNet` (MLP) implemented from scratch
- **Key Techniques**: He initialization, ReLU activation, Dropout, Weighted BCE loss, L2 regularization

**Final Test Accuracy**: [Insert your best accuracy here, e.g. XX.X%]

---

## Why GloVe Embeddings?

Instead of using simple Bag-of-Words or TF-IDF, I chose **GloVe (Global Vectors for Word Representation)** because:

- It captures **semantic meaning** — words with similar meanings have similar vectors
- It handles context better than basic count-based methods
- Averaging word vectors gives a simple yet powerful fixed-size representation for entire documents
- It's a standard, well-studied embedding method from Stanford, making results more meaningful

This approach is especially useful for PII detection, where understanding word relationships (e.g., names, addresses, IDs) matters.

---

# Neural Network from Scratch for PII Detection

A **from-scratch** multi-layer neural network built with only NumPy to classify whether a text contains Personally Identifiable Information (PII).

This project demonstrates a deep understanding of neural networks by implementing forward propagation, backpropagation, dropout, and training from the ground up — without using PyTorch, TensorFlow, or any high-level "black-box" libraries.

### Why This Project?
Most modern ML tutorials use high-level frameworks that hide the math. Here, I built everything manually using only NumPy to truly understand how neural networks work under the hood.

---

## Project Overview

- **Task**: Binary classification — Detect if masked text contains PII (1 = has PII, 0 = no PII)
- **Input**: Pre-trained GloVe word embeddings (300 dimensions)
- **Model**: Custom `ScratchNet` (MLP) implemented from scratch
- **Key Techniques**: He initialization, ReLU activation, Dropout, Weighted BCE loss, L2 regularization

**Final Test Accuracy**: [Insert your best accuracy here, e.g. XX.X%]

---

## Why GloVe Embeddings?

Instead of using simple Bag-of-Words or TF-IDF, I chose **GloVe (Global Vectors for Word Representation)** because:

- It captures **semantic meaning** — words with similar meanings have similar vectors
- It handles context better than basic count-based methods
- Averaging word vectors gives a simple yet powerful fixed-size representation for entire documents
- It's a standard, well-studied embedding method from Stanford, making results more meaningful

This approach is especially useful for PII detection, where understanding word relationships (e.g., names, addresses, IDs) matters.

---

## How the Model Works (Beginner-Friendly Explanation)

### 1. Data Preparation
- Text is converted into 300-dimensional vectors using **average GloVe embeddings**
- Each document gets one fixed-size vector representing its overall meaning

### 2. The Neural Network (`ScratchNet`)

I implemented the following from scratch:

- **Forward Pass**:
  - Takes 300-dim input
  - Passes through two hidden layers (512 → 256 neurons) with **ReLU** activation
  - Outputs a single logit for binary classification

- **Backward Pass (Backpropagation)**:
  - Calculates gradients using the chain rule
  - Applies ReLU derivative and dropout mask correction
  - Updates weights with mini-batch SGD + L2 regularization

- **Key Features Implemented**:
  - He initialization (better for ReLU)
  - Dropout for regularization
  - Weighted Binary Cross Entropy loss (to handle imbalanced PII data)
  - Stable loss computation to prevent numerical issues

---

## Technologies Used

- **Core**: NumPy (only math library)
- **Embeddings**: GloVe 6B 300d (Stanford)
- **Data**: pandas
- **Visualization/Progress**: matplotlib, tqdm

**No** deep learning frameworks were used.

---

## Results

- Test Accuracy: 93.3%
- Classification Report:
              precision    recall  f1-score   support

           0     0.9167    1.0000    0.9565        22
           1     1.0000    0.7500    0.8571         8

    accuracy                         0.9333        30
   macro avg     0.9583    0.8750    0.9068        30
weighted avg     0.9389    0.9333    0.9300        30


Confusion Matrix:
[[22  0]
 [ 2  6]]

---

