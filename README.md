# Statistical Machine Learning (SML) Assignments

Welcome to the SML Assignments repository! This repository contains a collection of Statistical Machine Learning assignments completed as part of the coursework. The projects involve implementing various core machine learning algorithms from scratch, analyzing their mathematical foundations, and evaluating their performance on real-world datasets like MNIST and Fashion-MNIST.

## 📂 Repository Structure

The repository is divided into four main assignments, each located in its respective directory:

### [Assignment 1 (2024011_A1)](./2024011_A1)
**Focus:** Generative Models and Discriminant Analysis
- **Maximum Likelihood Estimation (MLE):** Estimating the mean and covariance matrix assuming Gaussian distributions.
- **Linear Discriminant Analysis (LDA) & Quadratic Discriminant Analysis (QDA):** Classifying a subset of the MNIST dataset (digits 0, 1, and 2).
- **Visualization:** Using t-SNE to project and visualize the high-dimensional MNIST training and testing data in 2D.

### [Assignment 2 (2024011_A2)](./2024011_A2)
**Focus:** Dimensionality Reduction and Feature Extraction
- **Principal Component Analysis (PCA):** Reducing the dimensionality of the MNIST dataset while retaining specific variance thresholds (e.g., 75%, 90%).
- **Fisher’s Discriminant Analysis (FDA):** Finding the optimal low-dimensional projection that maximizes class separation.
- **Classification:** Applying LDA and QDA on the reduced datasets to compare performance and efficiency.

### [Assignment 3 (2024011_A3)](./2024011_A3)
**Focus:** Regularization and Ensemble Methods
- **Linear Models (Ridge & Lasso Regression):** Implementing L1 and L2 regularization from scratch to handle multi-class classification via an indicator schema, and exploring the bias-variance tradeoff.
- **Decision Trees:** Building decision trees from scratch (limited to 3 terminal nodes) using Gini impurity.
- **Ensemble Techniques:** Implementing Bagging (with Out-of-Bag error calculations) and Random Forests.
- **Regression Stumps:** Utilizing regression stumps on the Fashion-MNIST dataset.

### [Assignment 4 (2024011_A4)](./2024011_A4)
**Focus:** Boosting and Perceptrons
- **AdaBoost:** Implementing adaptive boosting with decision stumps on a PCA-reduced MNIST dataset.
- **Gradient Boosting:** Building a gradient boosting model with decision stumps to minimize the sum of squared residues (SSR).
- **Rosenblatt Perceptron:** Implementing a perceptron from scratch with early stopping and decision boundary visualization to classify synthetic Gaussian-distributed datasets.

---

## 🚀 Usage & Setup

Each assignment directory is self-contained and includes the source code (`.py` files or `.ipynb` Jupyter notebooks), generated figures, and a detailed PDF/Markdown report documenting the methodology, observations, and results.

### Prerequisites

To run the code in this repository, you will need Python 3 installed on your system. You will also need standard data science libraries. You can install the required dependencies using `pip`:

```bash
pip install numpy matplotlib scikit-learn tensorflow jupyter
```
*(Note: `tensorflow` is primarily used across these assignments as a convenient utility to load the MNIST/Fashion-MNIST datasets via `tensorflow.keras.datasets`)*

### Running the Code

**1. Running Python Scripts:**
Navigate to the directory of the assignment you wish to run and execute the specific script.
```bash
cd 2024011_A1
python sml_assignment1_2024011_py.py
```

**2. Running Jupyter Notebooks:**
If you prefer an interactive environment, you can launch Jupyter Notebook in the repository root and open the respective `.ipynb` files.
```bash
jupyter notebook
```

**3. Reading the Reports:**
For a comprehensive breakdown of the math, decisions, and graphical plots associated with each script, check out the `report.md` or `report.pdf` files located in the respective assignment folders.
