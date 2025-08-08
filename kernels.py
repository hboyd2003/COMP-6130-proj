import torch
import numpy as np
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel, polynomial_kernel


class Kernel:
    """Base kernel class."""

    def __call__(self, x1, x2):
        raise NotImplementedError


class LinearKernel(Kernel):
    """Linear kernel"""

    def __call__(self, x1, x2):
        # Check if inputs are PyTorch tensors
        if isinstance(x1, torch.Tensor) and isinstance(x2, torch.Tensor):
            # Ensure tensors are on the same device
            if x1.device != x2.device:
                x2 = x2.to(x1.device)
            # Compute linear kernel: K(x, y) = x @ y^T
            return torch.mm(x1, x2.t())
        else:
            # Fall back to sklearn for numpy arrays
            return linear_kernel(x1, x2)


class RBFKernel(Kernel):
    """Radial Basis Function (Gaussian) kernel"""

    def __init__(self, gamma: float = 1.0):
        self.gamma = gamma

    def __call__(self, x1, x2):
        if isinstance(x1, torch.Tensor) and isinstance(x2, torch.Tensor):
            # Ensure tensors are on the same device
            if x1.device != x2.device:
                x2 = x2.to(x1.device)
            
            # Compute RBF kernel: K(x, y) = exp(-gamma * ||x - y||^2)
            # Efficient computation using the expansion: ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x@y^T
            x1_norm = (x1 ** 2).sum(dim=1, keepdim=True)
            x2_norm = (x2 ** 2).sum(dim=1, keepdim=True)
            
            # Compute pairwise squared distances
            distances_sq = x1_norm + x2_norm.t() - 2.0 * torch.mm(x1, x2.t())
            
            # Ensure non-negative distances (numerical stability)
            distances_sq = torch.clamp(distances_sq, min=0.0)
            
            # Apply RBF kernel
            return torch.exp(-self.gamma * distances_sq)
        else:
            # Fall back to sklearn for numpy arrays
            return rbf_kernel(x1, x2, gamma=self.gamma)


class PolynomialKernel(Kernel):
    """Polynomial kernel"""

    def __init__(self, degree: int = 3, gamma: float = 1.0, coef0: float = 0.0):
        self.degree: int = degree
        self.gamma: float = gamma
        self.coef0: float = coef0

    def __call__(self, x1, x2):
        if isinstance(x1, torch.Tensor) and isinstance(x2, torch.Tensor):
            # Ensure tensors are on the same device
            if x1.device != x2.device:
                x2 = x2.to(x1.device)
            
            # Compute polynomial kernel: K(x, y) = (gamma * x @ y^T + coef0)^degree
            linear_term = torch.mm(x1, x2.t())
            return (self.gamma * linear_term + self.coef0) ** self.degree
        else:
            # Fall back to sklearn for numpy arrays
            return polynomial_kernel(x1, x2, degree=self.degree, gamma=self.gamma, coef0=self.coef0)  # pyright: ignore [reportArgumentType]
