from sklearn.metrics.pairwise import linear_kernel, rbf_kernel, polynomial_kernel

class Kernel:
    """Base kernel class."""

    def __call__(self, x1, x2):
        raise NotImplementedError


class LinearKernel(Kernel):
    """Linear kernel"""

    def __call__(self, x1, x2):
        return linear_kernel(x1, x2)


class RBFKernel(Kernel):
    """Radial Basis Function (Gaussian) kernel"""

    def __init__(self, gamma: float = 1.0):
        self.gamma = gamma

    def __call__(self, x1, x2):
        return rbf_kernel(x1, x2, gamma=self.gamma)


class PolynomialKernel(Kernel):
    """Polynomial kernel"""

    def __init__(self, degree: int = 3, gamma: float = 1.0, coef0: float = 0.0):
        self.degree: int = degree
        self.gamma: float = gamma
        self.coef0: float = coef0

    def __call__(self, x1, x2):
        return polynomial_kernel(x1, x2, degree=self.degree, gamma=self.gamma, coef0=self.coef0)  # pyright: ignore [reportArgumentType]
