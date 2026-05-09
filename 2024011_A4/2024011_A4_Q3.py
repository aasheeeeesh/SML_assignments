import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def generate_dataset(mean1, cov1, mean2, cov2, n_samples=200):
    X1 = np.random.multivariate_normal(mean1, cov1, n_samples)
    y1 = -np.ones(n_samples)
    
    X2 = np.random.multivariate_normal(mean2, cov2, n_samples)
    y2 = np.ones(n_samples)
    
    X = np.vstack((X1, X2))
    y = np.concatenate((y1, y2))
    
    # Shuffle
    indices = np.arange(2 * n_samples)
    np.random.shuffle(indices)
    
    return X[indices], y[indices]

class Perceptron:
    def __init__(self, learning_rate=0.01, max_epochs=300):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.weights = None
        self.bias = None
        self.misclassifications = []
        self.converged_epoch = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.misclassifications = []
        
        for epoch in range(self.max_epochs):
            misclassified = 0
            for i in range(n_samples):
                linear_output = np.dot(X[i], self.weights) + self.bias
                y_pred = 1 if linear_output >= 0 else -1
                
                if y[i] * linear_output <= 0: # Misclassified or exactly 0
                    self.weights += self.learning_rate * y[i] * X[i]
                    self.bias += self.learning_rate * y[i]
                    misclassified += 1
                    
            self.misclassifications.append(misclassified)
            
            if misclassified == 0:
                self.converged_epoch = epoch + 1
                break
                
        if self.converged_epoch is None:
            self.converged_epoch = self.max_epochs
            
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0, 1, -1)

def plot_decision_boundary(X_train, y_train, X_test, y_test, model, dataset_name):
    plt.figure(figsize=(10, 8))
    
    # Create a meshgrid to plot the decision boundary
    x_min, x_max = min(X_train[:, 0].min(), X_test[:, 0].min()) - 1, max(X_train[:, 0].max(), X_test[:, 0].max()) + 1
    y_min, y_max = min(X_train[:, 1].min(), X_test[:, 1].min()) - 1, max(X_train[:, 1].max(), X_test[:, 1].max()) + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
                         
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    
    # Plot training data
    plt.scatter(X_train[y_train == -1][:, 0], X_train[y_train == -1][:, 1], color='blue', marker='o', label='Train Class -1', alpha=0.6)
    plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], color='red', marker='o', label='Train Class 1', alpha=0.6)
    
    # Plot test data
    plt.scatter(X_test[y_test == -1][:, 0], X_test[y_test == -1][:, 1], color='blue', marker='x', label='Test Class -1', s=80)
    plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], color='red', marker='x', label='Test Class 1', s=80)
    
    plt.title(f'Decision Boundary for Dataset {dataset_name}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'q3_decision_boundary_{dataset_name}.png')
    plt.close()

def plot_misclassifications(model, dataset_name):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(model.misclassifications) + 1), model.misclassifications, marker='o')
    plt.title(f'Misclassified Samples per Epoch - Dataset {dataset_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Number of Misclassifications')
    plt.grid(True)
    plt.savefig(f'q3_misclassifications_{dataset_name}.png')
    plt.close()

def process_dataset(X, y, dataset_name):
    print(f"\n--- Processing Dataset {dataset_name} ---")
    n_samples = len(X)
    train_size = int(0.7 * n_samples)
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]
    
    model = Perceptron(learning_rate=0.01, max_epochs=300)
    model.fit(X_train, y_train)
    
    print(f"Convergence Epoch: {model.converged_epoch}")
    
    y_test_pred = model.predict(X_test)
    test_accuracy = np.mean(y_test_pred == y_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    plot_misclassifications(model, dataset_name)
    plot_decision_boundary(X_train, y_train, X_test, y_test, model, dataset_name)

def main():
    # Dataset A
    mean1 = [-3, -3]
    cov1 = np.eye(2)
    mean2 = [3, 3]
    cov2 = np.eye(2)
    X_A, y_A = generate_dataset(mean1, cov1, mean2, cov2, 200)
    process_dataset(X_A, y_A, 'A')
    
    # Dataset B
    cov1_B = 3 * np.eye(2)
    cov2_B = 3 * np.eye(2)
    X_B, y_B = generate_dataset(mean1, cov1_B, mean2, cov2_B, 200)
    process_dataset(X_B, y_B, 'B')

if __name__ == '__main__':
    main()
