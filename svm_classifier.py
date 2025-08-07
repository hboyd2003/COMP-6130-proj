"""Support Vector Machine classifier implementation."""
from __future__ import annotations

from typing import Dict
from numpy.typing import NDArray

import numpy as np
from cvxopt import matrix, solvers

from kernels import Kernel

class SVM:
    """Support Vector Machine classifier using quadratic programming."""
    support_vectors: NDArray
    support_labels: NDArray
    support_alphas: NDArray
    inverse_label_map: Dict | None
    label_map: Dict | None
    b: np.floating

    def __init__(self, kernel: Kernel, C: float, max_iter: int, tolerance: float):
        """
        Creates a new unfit SVM classifier. You should not call this directly.
        Use the fit class method instead.

        :param C: Regularization parameter
        :param kernel: Kernel function
        :param max_iter: Maximum number of iterations
        :param tolerance: Tolerance for stopping criterion
        """
        self.C = C
        self.kernel: Kernel = kernel
        self.tolerance: float = tolerance
        self.max_iter: int = max_iter

        self.inverse_label_map = None
        self.label_map = None

    @classmethod
    def fit(cls, kernel: Kernel, C, max_iter: int, tolerance: float, X: NDArray, y: NDArray):
        """
        Fits a new SVM classifier.

        :param kernel: Kernel function
        :param C: Regularization parameter
        :param max_iter: Maximum number of iterations
        :param tolerance: Tolerance for stopping criterion
        :param X:
        :param y:
        :return: Fit SVM classifier
        """

        svm = cls(kernel, C, max_iter, tolerance)
        svm._fit(X, y)
        return svm

    def _fit(self, X: NDArray, y: NDArray) -> None:
        """
        Fits a new SVM classifier.

        :param X:
        :param y:
        :return: Trained SVM classifier
        """
        # Convert labels to -1, +1 if needed
        unique_labels = np.unique(y)
        if len(unique_labels) != 2:
            raise ValueError("SVM requires exactly 2 classes")

        if not np.array_equal(np.sort(unique_labels), [-1, 1]):
            # Map labels to -1, +1
            self.label_map = {unique_labels[0]: -1, unique_labels[1]: 1}
            self.inverse_label_map = {-1: unique_labels[0], 1: unique_labels[1]}
            y = np.array([self.label_map[label] for label in y])
        # If labels are already -1, +1, no mapping needed

        n_samples = X.shape[0]

        # Compute kernel matrix
        K = self.kernel(X, X)
        Q = np.outer(y, y) * K
        P = matrix(Q)
        q = matrix(-np.ones(n_samples))
        G = matrix(np.vstack((-np.eye(n_samples), np.eye(n_samples))))
        h = matrix(np.hstack((np.zeros(n_samples), np.full(n_samples, self.C))))
        A = matrix(y, (1, n_samples), 'd')
        b = matrix(0.0)

        solvers.options['show_progress'] = False
        solvers.options['maxiters'] = self.max_iter
        solvers.options['abstol'] = self.tolerance

        result = solvers.qp(P, q, G, h, A, b)

        alphas = np.ravel(result['x'])

        # Find support vectors (alpha > tolerance)
        sv_indices = alphas > self.tolerance
        self.support_vectors = X[sv_indices]
        self.support_labels = y[sv_indices]
        self.support_alphas = alphas[sv_indices]

        # Compute bias term b
        # Use support vectors with 0 < alpha < C (unbounded support vectors)
        unbounded_sv = (alphas > self.tolerance) & (alphas < self.C - self.tolerance)

        # Compute b using unbounded support vectors
        b_values = []
        for i in np.where(unbounded_sv)[0]:
            decision_value = np.sum(alphas * y * K[i, :])
            b_values.append(y[i] - decision_value)
        self.b = np.mean(b_values)

    def decision_function(self, X: NDArray) -> NDArray:
        """
        Compute the decision function for samples in X.

        :param X: Data to compute for (n_samples, n_features)
        :return: Computed decision function
        """
        X = np.array(X)
        K = self.kernel(X, self.support_vectors)

        # Ensure K has the right shape (n_test_samples, n_support_vectors)
        if K.ndim == 1:
            K = K.reshape(1, -1)

        # Compute decision function: sum over support vectors
        assert self.support_alphas is not None
        decision = np.dot(K, self.support_alphas * self.support_labels) + self.b
        return decision

    def predict(self, X: NDArray) -> NDArray:
        """
        Predict labels for X.

        :param X: Data to predict labels for
        :return: Predicted labels
        """
        decision = self.decision_function(X)
        # Convert decision values to -1/+1, handling the case where decision = 0
        predictions = np.where(decision >= 0, 1, -1)

        # Map back to original labels if needed
        if self.inverse_label_map is not None:
            predictions = np.array([self.inverse_label_map[pred] for pred in predictions])

        return predictions

    def predict_proba(self, X: NDArray) -> NDArray:
        """
        Predict class probabilities using Platt scaling approximation.

        :param X: Test data (n_samples, n_features)
        :return:
        """
        decision = self.decision_function(X)
        # Simple sigmoid approximation for probabilities
        proba_pos = 1 / (1 + np.exp(-decision))
        proba_neg = 1 - proba_pos
        return np.column_stack([proba_neg, proba_pos])
