import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


def compute_loss(X, y, weights):
    """Calculates both Mean Squared Error (MSE) and Mean Absolute Error (MAE)."""
    n = len(y)
    predictions = np.dot(X, weights)
    error = predictions - y

    mse = np.sum(error ** 2) / (2 * n)
    mae = np.sum(np.abs(error)) / n
    return mse, mae


def gradient_descent(X, y, iterations=100, lr=0.01):
    """Performs gradient descent to optimize the weights."""
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)  # Stable zero initialization

    for i in range(iterations):
        predictions = np.dot(X, weights)
        error = predictions - y
        gradient = np.dot(X.T, error) / n_samples

        # Update weights moving against the gradient
        weights -= lr * gradient

        if i % 10 == 0:
            mse, mae = compute_loss(X, y, weights)
            print(f"Iteration {i:3d} | MSE: {mse:.4f} | MAE: {mae:.4f}")

    return weights


def predict(X, weights):
    """Vectorized prediction using matrix multiplication."""
    return np.dot(X, weights)


if __name__ == '__main__':
    # Generate regression dataset
    X, y = make_regression(n_samples=1000, n_features=3, noise=10, random_state=42)

    # Train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model (Adjusted learning rate slightly for smoother convergence)
    optimized_weights = gradient_descent(X_train, y_train, iterations=100, lr=0.05)

    # Evaluate model performance on unseen test data
    test_mse, test_mae = compute_loss(X_test, y_test, optimized_weights)

    print("\n" + "=" * 30)
    print("FINAL TEST SET EVALUATION")
    print("=" * 30)
    print(f"Test MSE: {test_mse:.4f}")
    print(f"Test MAE: {test_mae:.4f}")