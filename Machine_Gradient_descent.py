import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


def compute_loss(X, y, weights):
    """Compute MSE and MAE loss."""
    pred = np.dot(X, weights)
    error = y - pred
    n = X.shape[0]
    MSE = np.sum(error ** 2) / (2 * n)  # MSE with 1/2 for gradient descent convenience
    MAE = np.sum(np.abs(error)) / n     # Standard MAE (without 1/2)
    return MSE, MAE


def gradient(X, y, weights):
    """Compute gradient for MSE loss."""
    pred = np.dot(X, weights)
    error = y - pred
    return -X.T.dot(error) / X.shape[0]  # Negative gradient for descent


def gradient_descent(X_train, y_train, iterations=1000, lr=0.01, verbose=True):
    """Perform gradient descent optimization."""
    weights = np.random.randn(X_train.shape[1])  # Random initialization

    for i in range(iterations):
        grad = gradient(X_train, y_train, weights)
        weights -= lr * grad

        if verbose and i % 100 == 0:
            MSE, MAE = compute_loss(X_train, y_train, weights)
            print(f"Iteration {i:4d} - MSE: {MSE:.4f}, MAE: {MAE:.4f}")

    return weights


def find_best_lr(X, y, lr_candidates=[0.1, 0.01, 0.001, 0.05], iterations=500):
    """Find the best learning rate from candidates."""
    best_lr = None
    best_loss = float('inf')

    for lr in lr_candidates:
        weights = gradient_descent(X, y, iterations=iterations, lr=lr, verbose=False)
        current_loss, _ = compute_loss(X, y, weights)

        if current_loss < best_loss:
            best_loss = current_loss
            best_lr = lr

    return best_lr


if __name__ == '__main__':
    # Generate and split data
    X, y = make_regression(n_samples=1000, n_features=3, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Option 1: Use fixed learning rate
    print("Training with fixed learning rate...")
    weights = gradient_descent(X_train, y_train, lr=0.01)

    # Option 2: Find best learning rate automatically
    # print("Finding best learning rate...")
    # best_lr = find_best_lr(X_train, y_train)
    # print(f"Best learning rate: {best_lr}")
    # weights = gradient_descent(X_train, y_train, lr=best_lr)

    # Evaluate
    train_mse, train_mae = compute_loss(X_train, y_train, weights)
    test_mse, test_mae = compute_loss(X_test, y_test, weights)

    print("\nFinal Results:")
    print(f"Optimized weights: {weights}")
    print(f"Train MSE: {train_mse:.4f}, MAE: {train_mae:.4f}")
    print(f"Test MSE:  {test_mse:.4f}, MAE: {test_mae:.4f}")
