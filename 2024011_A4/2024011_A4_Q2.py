import numpy as np
import matplotlib.pyplot as plt
import os

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
    
    y_train_binary = np.where(y_train_filtered == 9, 1, -1)
    y_test_binary = np.where(y_test_filtered == 9, 1, -1)
    
    idx_4 = np.where(y_train_binary == -1)[0]
    idx_9 = np.where(y_train_binary == 1)[0]
    
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
    
    return x_train_final, y_train_final, x_val, y_val, x_test_filtered, y_test_binary

def apply_pca(x_train, x_val, x_test, n_components=5):
    print(f"Applying PCA (reducing to {n_components} dimensions)...")
    mean = np.mean(x_train, axis=0)
    x_train_centered = x_train - mean
    
    cov_matrix = np.cov(x_train_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    sorted_idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_idx]
    
    pca_matrix = eigenvectors[:, :n_components]
    
    x_train_pca = np.dot(x_train - mean, pca_matrix)
    x_val_pca = np.dot(x_val - mean, pca_matrix)
    x_test_pca = np.dot(x_test - mean, pca_matrix)
    
    return x_train_pca, x_val_pca, x_test_pca

class RegressionStump:
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.c_left = None
        self.c_right = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        best_score = -float('inf')
        
        S_total = np.sum(y)
        
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
                
            less_than = feature_values[:, np.newaxis] < midpoints[np.newaxis, :]
            
            N_L = np.sum(less_than, axis=0)
            N_R = n_samples - N_L
            
            # Fast matrix multiplication instead of element-wise + sum
            S_L = y @ less_than
            S_R = S_total - S_L
            
            # Minimize SSR is equivalent to maximizing (S_L^2 / N_L) + (S_R^2 / N_R)
            # Avoid division by zero
            score = (S_L**2) / np.maximum(N_L, 1) + (S_R**2) / np.maximum(N_R, 1)
            
            best_idx = np.argmax(score)
            if score[best_idx] > best_score:
                best_score = score[best_idx]
                self.feature_index = feature_idx
                self.threshold = midpoints[best_idx]
                
                # Compute leaf values
                n_l = N_L[best_idx]
                n_r = N_R[best_idx]
                s_l = S_L[best_idx]
                s_r = S_R[best_idx]
                
                self.c_left = s_l / n_l if n_l > 0 else 0
                self.c_right = s_r / n_r if n_r > 0 else 0
                
    def predict(self, X):
        feature_values = X[:, self.feature_index]
        return np.where(feature_values < self.threshold, self.c_left, self.c_right)

def gradient_boosting(X_train, y_train, X_val, y_val, n_estimators=300, learning_rate=0.01):
    stumps = []
    val_mses = []
    
    # Initialize predictions with 0
    train_preds = np.zeros(X_train.shape[0])
    val_preds = np.zeros(X_val.shape[0])
    
    labels = y_train.copy()
    
    for i in range(n_estimators):
        stump = RegressionStump()
        stump.fit(X_train, labels)
        
        train_preds += learning_rate * stump.predict(X_train)
        val_preds += learning_rate * stump.predict(X_val)
        
        # Absolute loss pseudo-residuals = sign(y - F_m(x))
        labels = np.sign(y_train - train_preds)
        
        stumps.append(stump)
        
        # Evaluate MSE on validation set
        val_mse = np.mean((y_val - val_preds)**2)
        val_mses.append(val_mse)
        
    return stumps, val_mses

def run_experiment(x_train, y_train, x_val, y_val, x_test, y_test, learning_rates=[0.1]):
    results = {}
    
    plt.figure(figsize=(10, 6))
    
    for lr in learning_rates:
        print(f"Training Gradient Boosting with learning rate {lr}...")
        stumps, val_mses = gradient_boosting(x_train, y_train, x_val, y_val, n_estimators=300, learning_rate=lr)
        
        best_iter = np.argmin(val_mses)
        print(f"  LR={lr}: Best iteration: {best_iter + 1} with validation MSE: {val_mses[best_iter]:.4f}")
        
        test_preds = np.zeros(x_test.shape[0])
        for i in range(best_iter + 1):
            test_preds += lr * stumps[i].predict(x_test)
            
        test_mse = np.mean((y_test - test_preds)**2)
        print(f"  LR={lr}: Test MSE: {test_mse:.4f}")
        
        results[lr] = {'val_mses': val_mses, 'test_mse': test_mse, 'best_iter': best_iter + 1}
        
        plt.plot(range(1, 301), val_mses, label=f'LR = {lr}')
        
    plt.xlabel('Number of Trees')
    plt.ylabel('Validation MSE')
    plt.title('Validation MSE vs Number of Trees for different Learning Rates')
    plt.legend()
    plt.grid(True)
    plt.savefig('q2_val_mse_all.png')
    print("Plot saved to q2_val_mse_all.png")
    
    return results

def main():
    file_path = './mnist.npz'
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return
        
    x_train, y_train, x_val, y_val, x_test, y_test = load_and_preprocess_data(file_path)
    x_train_pca, x_val_pca, x_test_pca = apply_pca(x_train, x_val, x_test, n_components=5)
    
    # First, just run for eta = 0.01 and plot as requested by the first part of the question
    print("--- Running for eta = 0.01 ---")
    stumps_01, val_mses_01 = gradient_boosting(x_train_pca, y_train, x_val_pca, y_val, n_estimators=300, learning_rate=0.01)
    best_iter_01 = np.argmin(val_mses_01)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 301), val_mses_01, color='b')
    plt.axvline(x=best_iter_01 + 1, color='r', linestyle='--', label=f'Best Iteration ({best_iter_01 + 1})')
    plt.xlabel('Number of Trees')
    plt.ylabel('Validation MSE')
    plt.title('Validation MSE vs Number of Trees (eta = 0.01)')
    plt.legend()
    plt.grid(True)
    plt.savefig('q2_val_mse_0.01.png')
    print("Plot for eta=0.01 saved to q2_val_mse_0.01.png")
    
    test_preds_01 = np.zeros(x_test_pca.shape[0])
    for i in range(best_iter_01 + 1):
        test_preds_01 += 0.01 * stumps_01[i].predict(x_test_pca)
    test_mse_01 = np.mean((y_test - test_preds_01)**2)
    print(f"Test MSE for eta=0.01: {test_mse_01:.4f}\n")
    
    # Now run for all learning rates
    print("--- Running for all learning rates ---")
    run_experiment(x_train_pca, y_train, x_val_pca, y_val, x_test_pca, y_test)

if __name__ == '__main__':
    main()
