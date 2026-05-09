import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
import urllib.request
import os

os.makedirs("figures", exist_ok=True)

def load_and_preprocess_mnist(p=10):
    path = "mnist.npz"
    if not os.path.exists(path):
        print("Downloading MNIST dataset...")
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
    
    return X_train_pca, y_train, X_test_pca, y_test, X_train, X_test, mu, evecs

X_train, y_train, X_test, y_test, X_train_raw, X_test_raw, mu_raw, evecs_raw = load_and_preprocess_mnist(p=10)

print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

X_train_aug = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
X_test_aug = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

lambdas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]

def ridge_fit(X, y, lam):
    D = X.shape[1]
    I_mod = np.eye(D)
    I_mod[0, 0] = 0.0 # Do not penalize intercept
    W = np.linalg.inv(X.T @ X + lam * I_mod) @ X.T @ y
    return W

def get_targets(y, k):
    return (y == k).astype(float)

ridge_train_mse, ridge_test_mse = [], []
lasso_train_mse, lasso_test_mse = [], []

ridge_paths_class1 = []
lasso_paths_class1 = []
lasso_nonzeros = []

for lam in lambdas:
    r_tr_err = 0; r_te_err = 0
    l_tr_err = 0; l_te_err = 0
    
    ridge_weights_c1 = None
    lasso_weights_c1 = None
    nonzero_count = 0
    
    for k in [0, 1, 2]:
        yk_train = get_targets(y_train, k)
        yk_test = get_targets(y_test, k)
        
        W_r = ridge_fit(X_train_aug, yk_train, lam)
        pred_r_tr = X_train_aug @ W_r
        pred_r_te = X_test_aug @ W_r
        
        r_tr_err += np.mean((pred_r_tr - yk_train)**2)
        r_te_err += np.mean((pred_r_te - yk_test)**2)
        
        if k == 1:
            ridge_weights_c1 = W_r[1:] # Exclude intercept for plot
            
        lasso = Lasso(alpha=lam, fit_intercept=True, max_iter=10000, tol=1e-4) # fit_intercept=True handles intercept without aug
        lasso.fit(X_train, yk_train)
        
        pred_l_tr = lasso.predict(X_train)
        pred_l_te = lasso.predict(X_test)
        
        l_tr_err += np.mean((pred_l_tr - yk_train)**2)
        l_te_err += np.mean((pred_l_te - yk_test)**2)
        
        nonzero_count += np.sum(lasso.coef_ != 0)
        
        if k == 1:
            lasso_weights_c1 = lasso.coef_
            
    ridge_train_mse.append(r_tr_err / 3)
    ridge_test_mse.append(r_te_err / 3)
    ridge_paths_class1.append(ridge_weights_c1)
    
    lasso_train_mse.append(l_tr_err / 3)
    lasso_test_mse.append(l_te_err / 3)
    lasso_paths_class1.append(lasso_weights_c1)
    lasso_nonzeros.append(nonzero_count / 3) # Average non-zero components per class

ridge_paths_class1 = np.array(ridge_paths_class1)
lasso_paths_class1 = np.array(lasso_paths_class1)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(lambdas, ridge_train_mse, marker='o', label='Train MSE')
plt.plot(lambdas, ridge_test_mse, marker='s', label='Test MSE')
plt.xscale('log')
plt.xlabel('Lambda ($\lambda$)')
plt.ylabel('Mean Squared Error')
plt.title('Ridge Regression MSE (p=10)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.subplot(1, 2, 2)
plt.plot(lambdas, lasso_train_mse, marker='o', label='Train MSE')
plt.plot(lambdas, lasso_test_mse, marker='s', label='Test MSE')
plt.xscale('log')
plt.xlabel('Lambda ($\lambda$)')
plt.ylabel('Mean Squared Error')
plt.title('Lasso Regression MSE (p=10)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('figures/mse_vs_lambda.png', dpi=300)
plt.show()

plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
plt.plot(lambdas, lasso_nonzeros, marker='o', color='purple')
plt.xscale('log')
plt.xlabel('Lambda ($\lambda$)')
plt.ylabel('Average Non-Zero Coefficients')
plt.title('Lasso: Non-Zero Coefficients vs $\lambda$')
plt.grid(True, linestyle='--', alpha=0.6)

plt.subplot(1, 3, 2)
for i in range(ridge_paths_class1.shape[1]):
    plt.plot(lambdas, ridge_paths_class1[:, i])
plt.xscale('log')
plt.xlabel('Lambda ($\lambda$)')
plt.ylabel('Coefficient Value')
plt.title('Ridge Regularization Path (Class 1)')
plt.grid(True, linestyle='--', alpha=0.6)

plt.subplot(1, 3, 3)
for i in range(lasso_paths_class1.shape[1]):
    plt.plot(lambdas, lasso_paths_class1[:, i])
plt.xscale('log')
plt.xlabel('Lambda ($\lambda$)')
plt.ylabel('Coefficient Value')
plt.title('Lasso Regularization Path (Class 1)')
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('figures/regularization_paths.png', dpi=300)
plt.show()

best_lambda_ridge_idx = np.argmin(ridge_test_mse)
best_lam = lambdas[best_lambda_ridge_idx]
print(f"Best lambda for Ridge based on Test MSE: {best_lam}")

complexities = [2, 5, 10, 20, 30]
comp_train_mse, comp_test_mse = [], []

for p in complexities:
    W_pca_p = evecs_raw[:, :p]
    
    Xc_train_p = X_train_raw - mu_raw
    Xc_test_p = X_test_raw - mu_raw
    
    X_train_p = Xc_train_p @ W_pca_p
    X_test_p = Xc_test_p @ W_pca_p
    
    X_train_p_aug = np.hstack([np.ones((X_train_p.shape[0], 1)), X_train_p])
    X_test_p_aug = np.hstack([np.ones((X_test_p.shape[0], 1)), X_test_p])
    
    r_tr_err = 0; r_te_err = 0
    for k in [0, 1, 2]:
        yk_train = get_targets(y_train, k)
        yk_test = get_targets(y_test, k)
        
        W_r = ridge_fit(X_train_p_aug, yk_train, best_lam)
        
        r_tr_err += np.mean((X_train_p_aug @ W_r - yk_train)**2)
        r_te_err += np.mean((X_test_p_aug @ W_r - yk_test)**2)
        
    comp_train_mse.append(r_tr_err / 3)
    comp_test_mse.append(r_te_err / 3)

plt.figure(figsize=(7, 5))
plt.plot(complexities, comp_train_mse, marker='o', label='Train MSE')
plt.plot(complexities, comp_test_mse, marker='s', label='Test MSE')
plt.xlabel('PCA Dimensions (p)')
plt.ylabel('Mean Squared Error')
plt.title(f'Ridge Regression MSE vs Complexity ($\lambda$={best_lam})')
plt.xticks(complexities)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('figures/mse_vs_complexity.png', dpi=300)
plt.show()

def evaluate_accuracy(X_tr, y_tr, X_te, y_te, method='ridge', lam=0.1):
    preds_te = []
    
    if method == 'ridge':
        X_tr_aug = np.hstack([np.ones((X_tr.shape[0], 1)), X_tr])
        X_te_aug = np.hstack([np.ones((X_te.shape[0], 1)), X_te])
        
        scores = np.zeros((X_te.shape[0], 3))
        for k in [0, 1, 2]:
            yk_train = get_targets(y_tr, k)
            W_r = ridge_fit(X_tr_aug, yk_train, lam)
            scores[:, k] = X_te_aug @ W_r
            
        preds_te = np.argmax(scores, axis=1)
        
    elif method == 'lasso':
        scores = np.zeros((X_te.shape[0], 3))
        for k in [0, 1, 2]:
            yk_train = get_targets(y_tr, k)
            lasso = Lasso(alpha=lam, fit_intercept=True, max_iter=10000, tol=1e-4)
            lasso.fit(X_tr, yk_train)
            scores[:, k] = lasso.predict(X_te)
            
        preds_te = np.argmax(scores, axis=1)
        
    return np.mean(preds_te == y_te)

best_lam_lasso_idx = np.argmin(lasso_test_mse)
best_lam_lasso = lambdas[best_lam_lasso_idx]

ridge_acc = evaluate_accuracy(X_train, y_train, X_test, y_test, method='ridge', lam=best_lam)
lasso_acc = evaluate_accuracy(X_train, y_train, X_test, y_test, method='lasso', lam=best_lam_lasso)

print("\n--- Final Classification Accuracies (p=10) ---")
print(f"Ridge Regression Accuracy (lambda={best_lam}): {ridge_acc * 100:.2f}%")
print(f"Lasso Regression Accuracy (lambda={best_lam_lasso}): {lasso_acc * 100:.2f}%")
