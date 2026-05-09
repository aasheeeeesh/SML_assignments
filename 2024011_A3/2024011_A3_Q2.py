import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import os

def load_and_preprocess_mnist(p=10):
    path = "mnist.npz"
    if not os.path.exists(path):
        urllib.request.urlretrieve("https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz", path)
    
    with np.load(path) as data:
        train_images = data['x_train']
        train_labels = data['y_train']
        test_images = data['x_test']
        test_labels = data['y_test']

    train_mask = np.isin(train_labels, [0, 1, 2])
    test_mask = np.isin(test_labels, [0, 1, 2])
    
    X_train = train_images[train_mask]
    y_train = train_labels[train_mask]
    X_test = test_images[test_mask]
    y_test = test_labels[test_mask]
    
    X_train = X_train.reshape(X_train.shape[0], -1).astype(float) / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1).astype(float) / 255.0
    
    mu = np.mean(X_train, axis=0, keepdims=True)
    Xc_train = X_train - mu
    Xc_test = X_test - mu
    
    S = (Xc_train.T @ Xc_train) / (Xc_train.shape[0] - 1)
    evals, evecs = np.linalg.eigh(S)
    
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]
    
    W_pca = evecs[:, :p]
    
    X_train_pca = Xc_train @ W_pca
    X_test_pca = Xc_test @ W_pca
    
    return X_train_pca, y_train, X_test_pca, y_test

X_train, y_train, X_test, y_test = load_and_preprocess_mnist(p=10)

def get_gini(y):
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return 1.0 - np.sum(p**2)

class DecisionNode:
    def __init__(self, data_indices):
        self.data_indices = data_indices
        self.feature_idx = None
        self.threshold = None
        self.left = None
        self.right = None
        self.is_leaf = True
        self.label = None

class CustomTree:
    def __init__(self, max_splits=2, max_features=None):
        self.max_splits = max_splits
        self.max_features = max_features
        self.root = None
        self.leaves = []
        
        self.X_train_ref = None
        self.y_train_ref = None

    def fit(self, X, y):
        self.X_train_ref = X
        self.y_train_ref = y
        self.root = DecisionNode(np.arange(len(y)))
        self.leaves = [self.root]
        
        overall_n = len(y)
        
        for _ in range(self.max_splits):
            best_split_info = None
            best_overall_gini = float('inf')
            
            for leaf in self.leaves:
                leaf_X = X[leaf.data_indices]
                leaf_y = y[leaf.data_indices]
                
                if len(np.unique(leaf_y)) <= 1:
                    continue # Pure node, no need to split
                
                if self.max_features is None:
                    features = np.arange(X.shape[1])
                else:
                    features = np.random.choice(X.shape[1], min(self.max_features, X.shape[1]), replace=False)
                    
                for f_idx in features:
                    threshold = np.mean(leaf_X[:, f_idx])
                    
                    left_mask = leaf_X[:, f_idx] <= threshold
                    right_mask = ~left_mask
                    
                    if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                        continue
                    
                    gini_left = get_gini(leaf_y[left_mask])
                    gini_right = get_gini(leaf_y[right_mask])
                    
                    n_left = np.sum(left_mask)
                    n_right = np.sum(right_mask)
                    
                    gini_split = (n_left / overall_n) * gini_left + (n_right / overall_n) * gini_right
                    
                    current_overall = 0.0
                    for other_leaf in self.leaves:
                        if other_leaf != leaf:
                            current_overall += (len(other_leaf.data_indices) / overall_n) * get_gini(y[other_leaf.data_indices])
                    
                    candidate_gini = current_overall + gini_split
                    
                    if candidate_gini < best_overall_gini:
                        best_overall_gini = candidate_gini
                        left_indices = leaf.data_indices[left_mask]
                        right_indices = leaf.data_indices[right_mask]
                        best_split_info = (leaf, f_idx, threshold, left_indices, right_indices)
                        
            if best_split_info is not None:
                split_leaf, best_f, best_thresh, left_idx, right_idx = best_split_info
                
                split_leaf.is_leaf = False
                split_leaf.feature_idx = best_f
                split_leaf.threshold = best_thresh
                
                split_leaf.left = DecisionNode(left_idx)
                split_leaf.right = DecisionNode(right_idx)
                
                self.leaves.remove(split_leaf)
                self.leaves.extend([split_leaf.left, split_leaf.right])
            else:
                break
                
        for leaf in self.leaves:
            vals, counts = np.unique(self.y_train_ref[leaf.data_indices], return_counts=True)
            leaf.label = vals[np.argmax(counts)]
            
    def _predict_single(self, x, node):
        if node.is_leaf:
            return node.label
        if x[node.feature_idx] <= node.threshold:
            return self._predict_single(x, node.left)
        return self._predict_single(x, node.right)
    
    def predict(self, X):
        return np.array([self._predict_single(x, self.root) for x in X])

print("\n--- Training Single Decision Tree (3 Terminal Nodes) ---")
tree = CustomTree(max_splits=2, max_features=None)
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test)

total_acc = np.mean(y_pred == y_test)
print(f"Overall Classification Accuracy (Single Tree): {total_acc * 100:.2f}%")
print("Class-wise Accuracy:")
for c in [0, 1, 2]:
    mask = (y_test == c)
    acc = np.mean(y_pred[mask] == y_test[mask])
    print(f"  Class {c}: {acc * 100:.2f}%")


print("\n--- Bagging and Out-of-Bag (OOB) Error ---")
np.random.seed(42)
n_samples = X_train.shape[0]

bagged_trees = []
oob_errors = []

for i in range(5):
    indices = np.random.choice(n_samples, n_samples, replace=True)
    oob_indices = np.setdiff1d(np.arange(n_samples), indices)
    
    X_boot = X_train[indices]
    y_boot = y_train[indices]
    
    X_oob = X_train[oob_indices]
    y_oob = y_train[oob_indices]
    
    t = CustomTree(max_splits=2, max_features=None)
    t.fit(X_boot, y_boot)
    bagged_trees.append(t)
    
    oob_preds = t.predict(X_oob)
    oob_err = np.mean(oob_preds != y_oob)
    oob_errors.append(oob_err)

print(f"Average OOB Error across 5 Bagged Trees: {np.mean(oob_errors):.4f}")

ensemble_preds = np.zeros((X_test.shape[0], 5), dtype=int)
for i, t in enumerate(bagged_trees):
    ensemble_preds[:, i] = t.predict(X_test)

final_preds = []
for i in range(X_test.shape[0]):
    vals, counts = np.unique(ensemble_preds[i], return_counts=True)
    final_preds.append(vals[np.argmax(counts)])
final_preds = np.array(final_preds)

bag_acc = np.mean(final_preds == y_test)
print(f"Overall Classification Accuracy (Bagging): {bag_acc * 100:.2f}%")
print("Class-wise Accuracy (Bagging):")
for c in [0, 1, 2]:
    mask = (y_test == c)
    acc = np.mean(final_preds[mask] == y_test[mask])
    print(f"  Class {c}: {acc * 100:.2f}%")


print("\n--- Random Forest ---")
k_optimal = 3 

rf_trees = []
rf_oob_errors = []

np.random.seed(42)
for i in range(5):
    indices = np.random.choice(n_samples, n_samples, replace=True)
    oob_indices = np.setdiff1d(np.arange(n_samples), indices)
    
    X_boot = X_train[indices]
    y_boot = y_train[indices]
    
    X_oob = X_train[oob_indices]
    y_oob = y_train[oob_indices]
    
    t = CustomTree(max_splits=2, max_features=k_optimal)
    t.fit(X_boot, y_boot)
    rf_trees.append(t)
    
    oob_preds = t.predict(X_oob)
    oob_err = np.mean(oob_preds != y_oob)
    rf_oob_errors.append(oob_err)

print(f"Optimal Value of k: {k_optimal} (Justification: sqrt(p) approx 3)")
print(f"Average OOB Error across 5 Random Forest Trees: {np.mean(rf_oob_errors):.4f}")

rf_ensemble_preds = np.zeros((X_test.shape[0], 5), dtype=int)
for i, t in enumerate(rf_trees):
    rf_ensemble_preds[:, i] = t.predict(X_test)

rf_final_preds = []
for i in range(X_test.shape[0]):
    vals, counts = np.unique(rf_ensemble_preds[i], return_counts=True)
    rf_final_preds.append(vals[np.argmax(counts)])
rf_final_preds = np.array(rf_final_preds)

rf_acc = np.mean(rf_final_preds == y_test)
print(f"Overall Classification Accuracy (Random Forest): {rf_acc * 100:.2f}%")
print("Class-wise Accuracy (Random Forest):")
for c in [0, 1, 2]:
    mask = (y_test == c)
    acc = np.mean(rf_final_preds[mask] == y_test[mask])
    print(f"  Class {c}: {acc * 100:.2f}%")
