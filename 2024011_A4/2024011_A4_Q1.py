import numpy as np
import matplotlib.pyplot as plt
import os

# Set a random seed for reproducibility
np.random.seed(42)

def load_and_preprocess_data(file_path):
    print("Loading data...")
    with np.load(file_path) as data:
        x_train, y_train = data['x_train'], data['y_train']
        x_test, y_test = data['x_test'], data['y_test']
    
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    
    x_train = x_train.astype(np.float64) / 255.0
    x_test = x_test.astype(np.float64) / 255.0
    
    train_mask = (y_train == 4) | (y_train == 9)
    x_train_filtered = x_train[train_mask]
    y_train_filtered = y_train[train_mask]
    
    test_mask = (y_test == 4) | (y_test == 9)
    x_test_filtered = x_test[test_mask]
    y_test_filtered = y_test[test_mask]
    
    # Relabel 4 -> -1, 9 -> 1
    y_train_binary = np.where(y_train_filtered == 9, 1, -1)
    y_test_binary = np.where(y_test_filtered == 9, 1, -1)
    
    # Validation split: keep aside 1000 from each class
    idx_4 = np.where(y_train_binary == -1)[0]
    idx_9 = np.where(y_train_binary == 1)[0]
    
    # Randomly shuffle indices to ensure unbiased validation set
    np.random.shuffle(idx_4)
    np.random.shuffle(idx_9)
    
    val_idx_4 = idx_4[:1000]
    val_idx_9 = idx_9[:1000]
    
    train_idx_4 = idx_4[1000:]
    train_idx_9 = idx_9[1000:]
    
    val_indices = np.concatenate([val_idx_4, val_idx_9])
    train_indices = np.concatenate([train_idx_4, train_idx_9])
    
    np.random.shuffle(val_indices)
    np.random.shuffle(train_indices)
    
    x_val = x_train_filtered[val_indices]
    y_val = y_train_binary[val_indices]
    
    x_train_final = x_train_filtered[train_indices]
    y_train_final = y_train_binary[train_indices]
    
    print(f"Train set: {x_train_final.shape}")
    print(f"Validation set: {x_val.shape}")
    print(f"Test set: {x_test_filtered.shape}")
    
    return x_train_final, y_train_final, x_val, y_val, x_test_filtered, y_test_binary

def apply_pca(x_train, x_val, x_test, n_components=5):
    print(f"Applying PCA (reducing to {n_components} dimensions)...")
    mean = np.mean(x_train, axis=0)
    x_train_centered = x_train - mean
    
    cov_matrix = np.cov(x_train_centered, rowvar=False)
    
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    sorted_idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_idx]
    
    # Top n components
    pca_matrix = eigenvectors[:, :n_components]
    
    # Transform all sets
    x_train_pca = np.dot(x_train - mean, pca_matrix)
    x_val_pca = np.dot(x_val - mean, pca_matrix)
    x_test_pca = np.dot(x_test - mean, pca_matrix)
    
    return x_train_pca, x_val_pca, x_test_pca

class DecisionStump:
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.polarity = 1 # 1 means < threshold is -1
        self.alpha = None
        
    def fit(self, X, y, weights):
        n_samples, n_features = X.shape
        min_error = float('inf')
        
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            unique_vals = np.unique(feature_values)
            sorted_unique = np.sort(unique_vals)
            
            if len(sorted_unique) > 1:
                midpoints = (sorted_unique[:-1] + sorted_unique[1:]) / 2.0
                if len(midpoints) > 1000:
                    midpoints = np.random.choice(midpoints, 1000, replace=False)
            else:
                midpoints = sorted_unique
                
            if len(midpoints) == 0:
                continue
                
            # Vectorized computation of errors over all candidate midpoints
            less_than = feature_values[:, np.newaxis] < midpoints[np.newaxis, :]
            
            # For polarity = 1, predictions are -1 when less_than is True, else 1
            pred_1 = np.where(less_than, -1, 1)
            
            mismatch_1 = pred_1 != y[:, np.newaxis]
            error_1 = np.sum(weights[:, np.newaxis] * mismatch_1, axis=0)
            
            # For polarity = -1, predictions are inverted
            error_neg_1 = np.sum(weights) - error_1
            
            best_idx_1 = np.argmin(error_1)
            min_err_1 = error_1[best_idx_1]
            
            best_idx_neg_1 = np.argmin(error_neg_1)
            min_err_neg_1 = error_neg_1[best_idx_neg_1]
            
            if min_err_1 < min_error:
                min_error = min_err_1
                self.feature_index = feature_idx
                self.threshold = midpoints[best_idx_1]
                self.polarity = 1
                
            if min_err_neg_1 < min_error:
                min_error = min_err_neg_1
                self.feature_index = feature_idx
                self.threshold = midpoints[best_idx_neg_1]
                self.polarity = -1
                
        return min_error

    def predict(self, X):
        n_samples = X.shape[0]
        predictions = np.ones(n_samples)
        feature_values = X[:, self.feature_index]
        if self.polarity == 1:
            predictions[feature_values < self.threshold] = -1
        else:
            predictions[feature_values >= self.threshold] = -1
        return predictions

def adaboost(X_train, y_train, X_val, y_val, n_estimators=300):
    print(f"Training AdaBoost with {n_estimators} stumps...")
    n_samples = X_train.shape[0]
    weights = np.ones(n_samples) / n_samples
    
    stumps = []
    val_accuracies = []
    
    val_preds = np.zeros(X_val.shape[0])
    
    for i in range(n_estimators):
        stump = DecisionStump()
        error = stump.fit(X_train, y_train, weights)
        
        eps = 1e-10
        alpha = 0.5 * np.log((1.0 - error + eps) / (error + eps))
        stump.alpha = alpha
        
        predictions = stump.predict(X_train)
        weights = weights * np.exp(-alpha * y_train * predictions)
        weights /= np.sum(weights)
        
        stumps.append(stump)
        
        val_preds += alpha * stump.predict(X_val)
        val_acc = np.mean(np.where(val_preds >= 0, 1, -1) == y_val)
        val_accuracies.append(val_acc)
        
        if (i+1) % 30 == 0:
            print(f"Tree {i+1}/{n_estimators}, Validation Accuracy: {val_acc:.4f}")
            
    return stumps, val_accuracies

def main():
    file_path = './mnist.npz'
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return
        
    x_train, y_train, x_val, y_val, x_test, y_test = load_and_preprocess_data(file_path)
    
    x_train_pca, x_val_pca, x_test_pca = apply_pca(x_train, x_val, x_test, n_components=5)
    
    stumps, val_accuracies = adaboost(x_train_pca, y_train, x_val_pca, y_val, n_estimators=300)
    
    best_iter = np.argmax(val_accuracies)
    print(f"\nBest iteration: {best_iter + 1} with validation accuracy: {val_accuracies[best_iter]:.4f}")
    
    # Evaluate on test set using the best iteration
    print("Evaluating on test set...")
    test_preds = np.zeros(x_test_pca.shape[0])
    for i in range(best_iter + 1):
        test_preds += stumps[i].alpha * stumps[i].predict(x_test_pca)
        
    final_test_preds = np.where(test_preds >= 0, 1, -1)
    test_accuracy = np.mean(final_test_preds == y_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 301), val_accuracies, label='Validation Accuracy', color='b')
    plt.axvline(x=best_iter + 1, color='r', linestyle='--', label=f'Best Iteration ({best_iter + 1})')
    plt.xlabel('Number of Trees')
    plt.ylabel('Validation Accuracy')
    plt.title('AdaBoost Validation Accuracy vs Number of Trees')
    plt.legend()
    plt.grid(True)
    plt.savefig('q1_val_accuracy.png')
    print("Plot saved to q1_val_accuracy.png")

if __name__ == '__main__':
    main()
