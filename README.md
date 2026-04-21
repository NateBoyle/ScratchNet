# ScratchNet
# Neural Network from Scratch for PII Detection

A **from-scratch** multi-layer neural network built with only NumPy to classify whether a text contains Personally Identifiable Information (PII).

This project demonstrates a deep understanding of neural networks by implementing forward propagation, backpropagation, dropout, and training from the ground up — without using PyTorch, TensorFlow, or any high-level "black-box" libraries.

### Why This Project?
Most modern ML tutorials use high-level frameworks that hide the math. Here, I built everything manually using only NumPy to truly understand how neural networks work under the hood.

---
## Data Source and Transformation

The data was synthetic PII data which you can download in Excel form from [Mendeley Data](https://data.mendeley.com/datasets/tzrjx692jy/1). Once the data was downloaded I used Excel formulas and the PII feature columns to mask the PII in the document text column for roughly 3/4 of the rows for each set.

---

## Project Overview

- **Task**: Binary classification — Detect if masked text contains PII (1 = has PII, 0 = no PII)
- **Input**: Pre-trained GloVe word embeddings (300 dimensions)
- **Model**: Custom `ScratchNet` (MLP) implemented from scratch
- **Key Techniques**: He initialization, ReLU activation, Dropout, Weighted BCE loss, L2 regularization

**Final Test Accuracy**: 93.3%

---

## Why GloVe Embeddings?

Instead of using simple Bag-of-Words or TF-IDF, I chose **GloVe (Global Vectors for Word Representation)** because:

- It captures **semantic meaning** — words with similar meanings have similar vectors
- It handles context better than basic count-based methods
- Averaging word vectors gives a simple yet powerful fixed-size representation for entire documents
- It's a standard, well-studied embedding method from Stanford, making results more meaningful. You can download it [here](https://nlp.stanford.edu/data/glove.6B.zip).

This approach is especially useful for PII detection, where understanding word relationships (e.g., names, addresses, IDs) matters.

---

## How the Model Works 

### 1. Data Preparation
Raw text is converted into fixed-size numerical vectors using **average GloVe embeddings** (300 dimensions).  
This turns each document into one vector that captures the overall meaning of the words.

### 2. The Neural Network (`ScratchNet`)

I built the entire network from scratch using only NumPy. Here’s what each major part does:

- **Forward Pass**  
  Takes the 300-dimensional input vector and passes it through two hidden layers (512 → 256 neurons).    
  Each hidden layer uses **ReLU (Rectified Linear Unit) activation** (`np.maximum(0, Z)`).  
  **What is ReLU?** It is a piecewise linear function that outputs the input directly if it is positive; otherwise, it outputs zero.  
  **Why ReLU?** It is simple, fast, and helps prevent the “vanishing gradient” problem. It also makes the model learn faster and perform better on most classification tasks like this one.

- **He Initialization**  
  Before training, the weights are initialized using He initialization:  
  `W = np.random.randn(prev_size, h) * np.sqrt(2.0 / prev_size)`  
  **What is He Initialization?** A method for setting the initial random weights of a neural network. It was specifically designed to solve the issues that arise when using the ReLU activation function in very deep networks.  
  **Why it works well with ReLU:** ReLU “turns off” negative values, so this method keeps the variance of activations roughly the same across layers. This avoids exploding or vanishing signals early in training — a common problem with random initialization.

- **Backward Pass (Backpropagation)**  
  After calculating the loss, the network computes gradients for every weight and bias using the chain rule.  
  It correctly handles the ReLU derivative and applies the dropout masks saved during the forward pass.

- **Parameter Update (Mini-batch SGD + L2 Regularization)**  
  Weights and biases are updated using **mini-batch Stochastic Gradient Descent** (`batch_size=128`).  
  **Mini-batch SGD** is the industry-standard algorithm for training deep learning models. It sits in the "Goldilocks zone" between processing one data point at a time and processing the entire dataset at once. It is much faster and more stable than full-batch gradient descent.  
  We also add **L2 regularization** (weight decay) with `l2_lambda=0.005`.  
  **L2 regularization** is a technique used to prevent overfitting by penalizing large weights in a neural network. It forces the model to keep the weights small, which results in a "simpler" model that generalizes better to unseen data and prevents the model from becoming too complex and overfitting to the training data.

### 3. Loss Function

We use a **weighted Binary Cross Entropy (BCE)** loss with `pos_weight=3.0`.

**What is Binary Cross Entropy?**  
It measures how well the model’s predicted probability matches the true label (0 or 1). It heavily penalizes confident wrong predictions.

**Why we used a weighted version:**  
The dataset is imbalanced — only about 25% of samples contain unmasked PII. By giving the positive class 3× more weight, the model is forced to pay much more attention to correctly detecting the rare (but important) “has PII” cases.

The implementation also includes a numerically stable version to avoid overflow or log(0) errors.

---

## Technologies Used

- **Core**: NumPy (only math library)
- **Embeddings**: GloVe 6B 300d (Stanford)
- **Data**: pandas
- **Visualization/Progress**: matplotlib, tqdm

**No** deep learning frameworks were used.

---

## Results

**Test Accuracy:** 93.3%

### Classification Report

| Class       | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| 0 (No PII)  | 0.9167    | 1.0000 | 0.9565   | 22      |
| 1 (Has PII) | 1.0000    | 0.7500 | 0.8571   | 8       |
| **Accuracy**    |           |        | **0.9333** | **30**  |
| Macro Avg   | 0.9583    | 0.8750 | 0.9068   | 30      |
| Weighted Avg| 0.9389    | 0.9333 | 0.9300   | 30      |

### Confusion Matrix

|                  | Predicted No PII | Predicted Has PII |
|------------------|------------------|-------------------|
| **Actual No PII** | 22               | 0                 |
| **Actual Has PII**| 2                | 6                 |
---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/NateBoyle/ScratchNet.git
   cd ScratchNet
   
2. Install the required dependencies (if necessary):
- pip install numpy pandas tqdm matplotlib scikit-learn  

3. Download GloVe embeddings:
- Download glove.6B.300d.zip from Stanford GloVe
- Extract it and place glove.6B.300d.txt in the project root folder.  

4. Open and run the notebook:
- Launch Jupyter Notebook, VS Code, or Google Colab
- Open ScratchNet.ipynb


Important Notes:

A small sample training dataset is included for quick testing and demonstration.
The full training set (~45MB) is not included in the repo due to GitHub file size limits.
Pre-trained model weights (scratchnet_2575weights_best.npz) are provided, so you can skip training and directly run the testing/evaluation cells.
   
