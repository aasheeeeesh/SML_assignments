import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.datasets import fashion_mnist


os.makedirs("figures", exist_ok=True)

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

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

p = 10
W_pca = evecs[:, :p]

X_train_pca = Xc_train @ W_pca
X_test_pca = Xc_test @ W_pca

y_train = y_train.astype(float) # Treat as continuous for regression
y_test = y_test.astype(float)

class RegressionStump:
    def __init__(self):
        self.feature_idx = None
        self.threshold = None
        self.left_val = None
        self.right_val = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        best_ssr = float('inf')
        best_split = None
        
        for f_idx in range(n_features):
            sort_indices = np.argsort(X[:, f_idx])
            X_sorted = X[sort_indices, f_idx]
            y_sorted = y[sort_indices]
            
            sum_left = 0.0
            sq_sum_left = 0.0
            
            sum_right = np.sum(y_sorted)
            sq_sum_right = np.sum(y_sorted ** 2)
            
            for i in range(1, n_samples):
                val = y_sorted[i-1]
                sum_left += val
                sq_sum_left += val ** 2
                
                sum_right -= val
                sq_sum_right -= val ** 2
                
                n_left = i
                n_right = n_samples - i
                
                if X_sorted[i-1] < X_sorted[i]:
                    ssr_left = sq_sum_left - (sum_left**2) / n_left
                    ssr_right = sq_sum_right - (sum_right**2) / n_right
                    ssr_total = ssr_left + ssr_right
                    
                    if ssr_total < best_ssr:
                        best_ssr = ssr_total
                        threshold = (X_sorted[i-1] + X_sorted[i]) / 2.0
                        left_mean = sum_left / n_left
                        right_mean = sum_right / n_right
                        best_split = (f_idx, threshold, left_mean, right_mean)
                        
        if best_split is not None:
            self.feature_idx = best_split[0]
            self.threshold = best_split[1]
            self.left_val = best_split[2]
            self.right_val = best_split[3]

    def predict(self, X):
        preds = np.zeros(X.shape[0])
        left_mask = X[:, self.feature_idx] <= self.threshold
        right_mask = ~left_mask
        preds[left_mask] = self.left_val
        preds[right_mask] = self.right_val
        return preds

print("\n--- Training Single Decision Stump ---")
stump = RegressionStump()
stump.fit(X_train_pca, y_train)

preds = stump.predict(X_test_pca)
mse = np.mean((preds - y_test) ** 2)
print(f"Test Set MSE (Single Stump): {mse:.4f}")
print(f"Stump chose feature index {stump.feature_idx} with threshold {stump.threshold:.4f}")

print("\n--- Bagging on Random Samples ---")
np.random.seed(42)
n_samples = X_train_pca.shape[0]

bagged_stumps = []
oob_errors = []

for i in range(5):
    indices = np.random.choice(n_samples, n_samples, replace=True)
    oob_indices = np.setdiff1d(np.arange(n_samples), indices)
    
    X_boot = X_train_pca[indices]
    y_boot = y_train[indices]
    
    X_oob = X_train_pca[oob_indices]
    y_oob = y_train[oob_indices]
    
    s = RegressionStump()
    s.fit(X_boot, y_boot)
    bagged_stumps.append(s)
    
    oob_preds = s.predict(X_oob)
    oob_err = np.mean((oob_preds - y_oob) ** 2)
    oob_errors.append(oob_err)

print(f"Average OOB Error (MSE) across 5 Bagged Stumps: {np.mean(oob_errors):.4f}")

ensemble_test_preds = np.zeros_like(y_test)
for s in bagged_stumps:
    ensemble_test_preds += s.predict(X_test_pca)
ensemble_test_preds /= 5.0

bagging_mse = np.mean((ensemble_test_preds - y_test) ** 2)
print(f"Test Set MSE (Bagging): {bagging_mse:.4f}")


best_f_idx = stump.feature_idx

np.random.seed(42)
plot_indices = np.random.choice(X_test_pca.shape[0], 200, replace=False)
X_plot = X_test_pca[plot_indices]
y_plot = y_test[plot_indices]

sort_args = np.argsort(X_plot[:, best_f_idx])
X_plot_sorted = X_plot[sort_args, best_f_idx]
y_plot_sorted = y_plot[sort_args]

single_preds_plot = stump.predict(X_plot)[sort_args]
bagged_preds_plot = ensemble_test_preds[plot_indices][sort_args]

plt.figure(figsize=(10, 6))
plt.scatter(X_plot_sorted, y_plot_sorted, alpha=0.5, label='True Function (Actual Labels)', color='gray', s=15)
plt.plot(X_plot_sorted, single_preds_plot, label='Single Decision Stump', color='blue', linewidth=2, linestyle='dashed')
plt.plot(X_plot_sorted, bagged_preds_plot, label='Bagged Stumps (Ensemble)', color='red', linewidth=2)

plt.axvline(x=stump.threshold, color='blue', linestyle='dotted', alpha=0.5, label=f'Stump Threshold')
plt.title('Performance Comparison: Single Stump vs Bagged Model')
plt.xlabel(f'PCA Feature Dimension {best_f_idx} (Selected by Single Stump)')
plt.ylabel('Response (y)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('figures/stump_vs_bagging.png', dpi=300)
plt.show()
