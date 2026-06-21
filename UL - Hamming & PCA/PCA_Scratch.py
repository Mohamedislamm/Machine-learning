import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class PCA_Scratch:
    def __init__(self, n_comp):
        self.n_comp = n_comp
        self.mean = None
        self.comp = None
        self.explained_variance = None
        self.explained_variance_ratio = None
        self.cumulative_explained_variance_ratio = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        cov = self._compute_covariance_matrix(X_centered)

        eigenval, eigenvec = self._eigen_decomposition(cov, self.n_comp)

        idx = np.argsort(eigenval)[::-1]
        eigenval_sorted = eigenval[idx]
        eigenvec_sorted = eigenvec[idx]

        self.comp = eigenvec_sorted[:self.n_comp]
        self.explained_variance = eigenval_sorted[:self.n_comp]
        total_var = np.sum(eigenval_sorted)
        self.explained_variance_ratio = self.explained_variance / total_var
        self.cumulative_explained_variance_ratio = np.cumsum(self.explained_variance_ratio)

    def _compute_covariance_matrix(self, X_centered):
        """
        Compute covariance matrix from scratch.
        Cov = (1/(n-1)) * X^T * X
        """
        n_samples = X_centered.shape[0]
        cov = (X_centered.T @ X_centered) / (n_samples - 1)
        return cov

    def _eigen_decomposition(self, A, n_comp, n_iter=1000, tol=1e-6):
        """
        Compute eigenvalues and eigenvectors using Power Iteration with Deflation.
        """
        n_features = A.shape[0]
        eigenvalues = []
        eigenvectors = []

        A_copy = A.copy()

        for _ in range(n_comp):
            v = np.random.rand(n_features)
            v = v / np.linalg.norm(v)

            for _ in range(n_iter):
                Av = A_copy @ v
                Av_norm = np.linalg.norm(Av)
                if Av_norm == 0:
                    break

                v_next = Av / Av_norm

                if np.linalg.norm(v_next - v) < tol:
                    v = v_next
                    break

                v = v_next

            eigenvalue = v.T @ (A_copy @ v)

            eigenvalues.append(eigenvalue)
            eigenvectors.append(v)
            A_copy = A_copy - eigenvalue * np.outer(v, v)

        return np.array(eigenvalues), np.array(eigenvectors)

    def transform(self, X):
        return (X - self.mean) @ self.comp.T

    def inverse_transform(self, X_compressed):
        """Reconstruct data from compressed representation."""
        return (X_compressed @ self.comp) + self.mean