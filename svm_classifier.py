from __future__ import annotations

from typing import Dict, Any, Tuple
from numpy.typing import NDArray
from random import randrange
import torch
import numpy as np

from kernels import Kernel


class SVM:
    """Support Vector Machine classifier using Sequential Minimal Optimization (SMO)"""
    support_vectors: NDArray
    support_labels: NDArray
    support_alphas: NDArray
    inverse_label_map: Dict
    label_map: Dict
    X_tensor: torch.Tensor
    y_tensor: torch.Tensor
    alphas_tensor: torch.Tensor
    errors_tensor: torch.Tensor


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

        # SMO algorithm state
        self.b: float = 0.0
        self.eps: float = 1e-3  # epsilon for SMO
        
        # Device selection
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            print("Using CPU (GPU not available)")

        self.m: int = 0
        self.n: int = 0

    @classmethod
    def fit(cls, kernel: Kernel, C: float, max_iter: int, tolerance: float, X: NDArray, y: NDArray):
        """
        Fits a new SVM classifier

        :param kernel: Instance of Kernel class to use
        :param C: Regularization parameter
        :param max_iter: Maximum number of iterations
        :param tolerance: Tolerance for stopping criterion
        :param X: Training data
        :param y: Training labels
        :return: Fit SVM classifier
        """
        svm = cls(kernel, C, max_iter, tolerance)
        svm._fit(X, y)
        return svm

    def _fit(self, X: NDArray, y: NDArray) -> None:
        """
        Fits the SVM classifier using SMO algorithm with GPU acceleration.

        :param X: Training data
        :param y: Training labels
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

        # Store original numpy data for compatibility
        self.X = X
        self.y = y
        self.m, self.n = X.shape

        # Convert to PyTorch tensors and move to device
        self.X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        self.y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device)
        
        # Initialize SMO algorithm state on device
        self.alphas_tensor = torch.zeros(self.m, dtype=torch.float32, device=self.device)
        self.errors_tensor = torch.zeros(self.m, dtype=torch.float32, device=self.device)
        self.b = 0.0
        
        # Pre-compute initial errors
        self._initialize_errors()

        # Run SMO algorithm
        self._smo_main_routine()

        # Convert results back to numpy and extract support vectors
        self.alphas = self.alphas_tensor.cpu().numpy()
        sv_indices = self.alphas > self.tolerance
        self.support_vectors = self.X[sv_indices]
        self.support_labels = self.y[sv_indices]
        self.support_alphas = self.alphas[sv_indices]

    def _get_kernel_value(self, i: int, j: int) -> float:
        """Get kernel value K(x_i, x_j) efficiently."""
        k_val = self.kernel(
            self.X_tensor[i:i+1],
            self.X_tensor[j:j+1]
        )
        if isinstance(k_val, torch.Tensor):
            return float(k_val[0, 0].item())
        else:
            return float(k_val[0, 0])

    def _initialize_errors(self) -> None:
        """Initialize error cache"""
        # When alphas are all 0, output is -b (which is 0 initially), so error = -b - y[i] = -y[i]
        self.errors_tensor = -self.y_tensor.clone()

    def _smo_output(self, i: int) -> float:
        """Compute the SVM output for example i"""
        # Find non-zero alphas for efficiency
        non_zero_mask = self.alphas_tensor != 0
        
        if not non_zero_mask.any():
            return -self.b
        
        # Get kernel values
        k_row = self.kernel(self.X_tensor[i:i+1], self.X_tensor).squeeze(0)
        
        # Ensure k_row is a tensor
        if not isinstance(k_row, torch.Tensor):
            k_row = torch.tensor(k_row, dtype=torch.float32, device=self.device)
        
        output = torch.sum(self.alphas_tensor[non_zero_mask] * 
                          self.y_tensor[non_zero_mask] * 
                          k_row[non_zero_mask])
        
        return float(output.item() - self.b)

    def _smo_get_error(self, i1: int) -> float:
        """Get error for example i1."""
        alpha_i1 = float(self.alphas_tensor[i1].item())
        
        if 0 < alpha_i1 < self.C:
            return float(self.errors_tensor[i1].item())
        
        return self._smo_output(i1) - float(self.y_tensor[i1].item())

    def _smo_get_non_bound_indexes(self) -> list:
        """Get indices of non-bound support vectors."""
        mask = (self.alphas_tensor > 0) & (self.alphas_tensor < self.C)
        return torch.where(mask)[0].cpu().tolist()

    def _smo_take_step(self, i1: int, i2: int) -> bool:
        """Take an optimization step for the pair (i1, i2)."""
        if i1 == i2:
            return False

        a1 = float(self.alphas_tensor[i1].item())
        y1 = float(self.y_tensor[i1].item())
        E1 = self._smo_get_error(i1)

        a2 = float(self.alphas_tensor[i2].item())
        y2 = float(self.y_tensor[i2].item())
        E2 = self._smo_get_error(i2)

        s = y1 * y2

        # Compute the bounds of the new alpha2
        if y1 != y2:
            L = max(0, a2 - a1)
            H = min(self.C, self.C + a2 - a1)
        else:
            L = max(0, a2 + a1 - self.C)
            H = min(self.C, a2 + a1)

        if L == H:
            return False

        k11 = self._get_kernel_value(i1, i1)
        k12 = self._get_kernel_value(i1, i2)
        k22 = self._get_kernel_value(i2, i2)

        # Compute the second derivative of the objective function
        eta = k11 + k22 - 2 * k12

        if eta > 0:
            a2_new = a2 + y2 * (E1 - E2) / eta
            
            if a2_new < L:
                a2_new = L
            elif a2_new > H:
                a2_new = H
        else:
            # Handle unusual circumstances
            f1 = y1 * (E1 + self.b) - a1 * k11 - s * a2 * k12
            f2 = y2 * (E2 + self.b) - s * a1 * k12 - a2 * k22
            L1 = a1 + s * (a2 - L)
            H1 = a1 + s * (a2 - H)
            Lobj = L1 * f1 + L * f2 + 0.5 * (L1 ** 2) * k11 + 0.5 * (L ** 2) * k22 + s * L * L1 * k12
            Hobj = H1 * f1 + H * f2 + 0.5 * (H1 ** 2) * k11 + 0.5 * (H ** 2) * k22 + s * H * H1 * k12

            if Lobj < Hobj - self.eps:
                a2_new = L
            elif Lobj > Hobj + self.eps:
                a2_new = H
            else:
                a2_new = a2

        # Check if alpha2 changed enough
        if abs(a2_new - a2) < self.eps * (a2_new + a2 + self.eps):
            return False

        a1_new = a1 + s * (a2 - a2_new)

        # Compute new threshold b
        b1 = E1 + y1 * (a1_new - a1) * k11 + y2 * (a2_new - a2) * k12 + self.b
        b2 = E2 + y1 * (a1_new - a1) * k12 + y2 * (a2_new - a2) * k22 + self.b

        if 0 < a1_new < self.C:
            new_b = b1
        elif 0 < a2_new < self.C:
            new_b = b2
        else:
            new_b = (b1 + b2) / 2.0

        delta_b = new_b - self.b
        self.b = new_b

        # Update error cache using GPU operations
        delta1 = y1 * (a1_new - a1)
        delta2 = y2 * (a2_new - a2)

        # Get non-bound indices
        non_bound_mask = (self.alphas_tensor > 0) & (self.alphas_tensor < self.C)
        
        if non_bound_mask.any():
            # Get indices as a list to avoid tensor indexing issues
            non_bound_indices = torch.where(non_bound_mask)[0]
            
            if len(non_bound_indices) > 0:
                # Compute kernel values
                k1_result = self.kernel(
                    self.X_tensor[i1:i1+1], 
                    self.X_tensor[non_bound_indices]
                )
                k2_result = self.kernel(
                    self.X_tensor[i2:i2+1], 
                    self.X_tensor[non_bound_indices]
                )
                
                # Ensure results are tensors
                if isinstance(k1_result, torch.Tensor):
                    k1_values = k1_result.squeeze(0).to(self.device)
                else:
                    k1_values = torch.tensor(k1_result, dtype=torch.float32, device=self.device).squeeze(0)
                    
                if isinstance(k2_result, torch.Tensor):
                    k2_values = k2_result.squeeze(0).to(self.device)
                else:
                    k2_values = torch.tensor(k2_result, dtype=torch.float32, device=self.device).squeeze(0)
                
                # Compute error updates
                error_update = delta1 * k1_values + delta2 * k2_values - delta_b
                
                # Update errors using a loop to avoid indexing issues
                for idx, nb_idx in enumerate(non_bound_indices):
                    self.errors_tensor[nb_idx] = self.errors_tensor[nb_idx] + error_update[idx]

        # Set errors to 0 for the updated alphas
        self.errors_tensor[i1] = 0.0
        self.errors_tensor[i2] = 0.0

        # Update alphas
        self.alphas_tensor[i1] = a1_new
        self.alphas_tensor[i2] = a2_new

        return True

    def _smo_examine_example(self, i2: int) -> int:
        """Examine example i2 and try to find a suitable pair for optimization."""
        y2 = float(self.y_tensor[i2].item())
        a2 = float(self.alphas_tensor[i2].item())
        E2 = self._smo_get_error(i2)

        r2 = E2 * y2

        if not ((r2 < -self.tolerance and a2 < self.C) or (r2 > self.tolerance and a2 > 0)):
            return 0

        # Choose the Lagrange multiplier that maximizes the absolute error
        i1 = -1
        non_bound_indices = self._smo_get_non_bound_indexes()
        
        if len(non_bound_indices) > 1:
            # Compute errors for non-bound examples
            errors_nb = self.errors_tensor[non_bound_indices]
            steps = torch.abs(errors_nb - E2)
            max_idx = torch.argmax(steps)
            i1 = non_bound_indices[max_idx]

        if i1 >= 0 and self._smo_take_step(i1, i2):
            return 1

        # Try non-bound examples
        if len(non_bound_indices) > 0:
            rand_i = randrange(len(non_bound_indices))
            for i1 in non_bound_indices[rand_i:] + non_bound_indices[:rand_i]:
                if self._smo_take_step(i1, i2):
                    return 1

        # Try all examples
        rand_i = randrange(self.m)
        all_indices = list(range(self.m))
        for i1 in all_indices[rand_i:] + all_indices[:rand_i]:
            if self._smo_take_step(i1, i2):
                return 1

        return 0

    def _smo_main_routine(self) -> None:
        """Main SMO routine with GPU optimization."""
        num_changed = 0
        examine_all = True
        iterations = 0

        while (num_changed > 0 or examine_all) and iterations < self.max_iter:
            num_changed = 0

            if examine_all:
                for i in range(self.m):
                    num_changed += self._smo_examine_example(i)
            else:
                non_bound_idx = self._smo_get_non_bound_indexes()
                for i in non_bound_idx:
                    num_changed += self._smo_examine_example(i)

            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True

            iterations += 1
            
            if iterations % 10 == 0:
                print(f"SMO iteration {iterations}, num_changed: {num_changed}")

    def decision_function(self, X: NDArray) -> NDArray:
        """
        Compute the decision function for samples in X.

        :param X: Data to compute for (n_samples, n_features)
        :return: Computed decision function
        """
        if self.support_vectors is None or self.support_alphas is None or self.support_labels is None:
            raise ValueError("SVM has not been fitted yet")

        # Convert to tensor for GPU computation
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        sv_tensor = torch.tensor(self.support_vectors, dtype=torch.float32, device=self.device)
        sa_tensor = torch.tensor(self.support_alphas, dtype=torch.float32, device=self.device)
        sl_tensor = torch.tensor(self.support_labels, dtype=torch.float32, device=self.device)
        
        K = self.kernel(X_tensor, sv_tensor)
    
        if K.ndim == 1:
            K = K.reshape(1, -1)
        
        # Compute decision function on GPU
        decision = torch.matmul(K, sa_tensor * sl_tensor) + self.b
        
        # Convert back to numpy
        return decision.cpu().numpy()

    def predict(self, X: NDArray) -> NDArray:
        """
        Predict labels for X.

        :param X: Data to predict labels for
        :return: Predicted labels
        """
        decision = self.decision_function(X)
        predictions = np.where(decision >= 0, 1, -1)

        # Map back to original labels if needed
        if self.inverse_label_map is not None:
            predictions = np.array([self.inverse_label_map[pred] for pred in predictions])

        return predictions

    def predict_proba(self, X: NDArray) -> NDArray:
        """
        Predict class probabilities using Platt scaling approximation.

        :param X: Test data (n_samples, n_features)
        :return: Class probabilities
        """
        decision = self.decision_function(X)
        # Simple sigmoid approximation for probabilities
        proba_pos = 1 / (1 + np.exp(-decision))
        proba_neg = 1 - proba_pos
        return np.column_stack([proba_neg, proba_pos])
