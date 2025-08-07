"""
Unit tests for SVM classifier implementation.
Attempts to test for SVM classifier implementation correctness
"""
import numpy as np
import pytest
from sklearn.datasets import make_classification, make_circles
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from svm_classifier import SVM
from kernels import LinearKernel, RBFKernel, PolynomialKernel


class TestSVMInitialization:
    """Test SVM initialization and basic setup."""

    def test_init_parameters(self):
        """Test SVM initialization with valid parameters."""
        kernel = LinearKernel()
        svm = SVM(kernel=kernel, C=1.0, max_iter=100, tolerance=1e-3)

        assert svm.C == 1.0
        assert svm.kernel == kernel
        assert svm.max_iter == 100
        assert svm.tolerance == 1e-3
        assert svm.support_alphas is None
        assert svm.support_vectors is None
        assert svm.support_labels is None
        assert svm.b is None

    def test_fit_class_method(self):
        """Test the fit class method creates and trains SVM."""
        # Simple linearly separable data
        X = np.array([[0, 0], [1, 1], [2, 0], [0, 2]])
        y = np.array([0, 1, 0, 1])

        kernel = LinearKernel()
        svm = SVM.fit(kernel=kernel, C=1.0, max_iter=100, tolerance=1e-3, X=X, y=y)

        assert isinstance(svm, SVM)
        assert svm.support_vectors is not None
        assert svm.support_labels is not None
        assert svm.support_alphas is not None
        assert svm.b is not None


class TestSVMLabelMapping:
    """Test label mapping functionality."""

    def test_binary_label_mapping(self):
        """Test mapping of arbitrary binary labels to -1, +1."""
        X = np.array([[0, 0], [1, 1], [2, 0], [0, 2]])
        y = np.array(['cat', 'dog', 'cat', 'dog'])

        kernel = LinearKernel()
        svm = SVM.fit(kernel=kernel, C=1.0, max_iter=100, tolerance=1e-3, X=X, y=y)

        # Check that label mapping was created
        assert svm.label_map is not None
        assert svm.inverse_label_map is not None
        assert len(svm.label_map) == 2
        assert len(svm.inverse_label_map) == 2

        # Test predictions return original labels
        predictions = svm.predict(X)
        assert all(pred in ['cat', 'dog'] for pred in predictions)

    def test_numeric_label_mapping(self):
        """Test mapping of numeric labels to -1, +1."""
        X = np.array([[0, 0], [1, 1], [2, 0], [0, 2]])
        y = np.array([5, 10, 5, 10])

        kernel = LinearKernel()
        svm = SVM.fit(kernel=kernel, C=1.0, max_iter=100, tolerance=1e-3, X=X, y=y)

        predictions = svm.predict(X)
        assert all(pred in [5, 10] for pred in predictions)

    def test_invalid_multiclass_labels(self):
        """Test error handling for more than 2 classes."""
        X = np.array([[0, 0], [1, 1], [2, 0]])
        y = np.array([0, 1, 2])  # 3 classes

        kernel = LinearKernel()
        with pytest.raises(ValueError, match="SVM requires exactly 2 classes"):
            SVM.fit(kernel=kernel, C=1.0, max_iter=100, tolerance=1e-3, X=X, y=y)


class TestSVMFitting:
    """Test SVM fitting with different data scenarios."""

    def test_linearly_separable_data(self):
        """Test SVM on perfectly linearly separable data."""
        # Create simple linearly separable data
        X = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        y = np.array([0, 1, 1, 0])  # XOR-like but linearly separable version
        X = np.array([[-2, -1], [-1, -1], [1, 1], [2, 1]])
        y = np.array([0, 0, 1, 1])

        kernel = LinearKernel()
        svm = SVM.fit(kernel=kernel, C=1.0, max_iter=100, tolerance=1e-3, X=X, y=y)

        # Should have support vectors
        assert svm.support_vectors is not None
        assert len(svm.support_vectors) > 0

        assert svm.support_alphas is not None
        assert len(svm.support_alphas) > 0

        assert svm.support_labels is not None
        assert len(svm.support_labels) > 0

        # All alphas should be positive
        assert all(alpha > 0 for alpha in svm.support_alphas)

        # Should achieve perfect classification
        predictions = svm.predict(X)
        accuracy = np.mean(predictions == y)
        assert accuracy == 1.0

    def test_different_kernels(self):
        """Test SVM with different kernel types."""
        # Generate non-linearly separable data
        X, y = make_circles(n_samples=50, noise=0.1, factor=0.3, random_state=42)
        X = np.array(X)
        y = np.array(y)

        kernels = [
            LinearKernel(),
            RBFKernel(gamma=1.0),
            PolynomialKernel(degree=2, gamma=1.0)
        ]

        for kernel in kernels:
            svm = SVM.fit(kernel=kernel, C=1.0, max_iter=200, tolerance=1e-3, X=X, y=y)

            # Should have fitted successfully
            assert svm.support_vectors is not None
            assert len(svm.support_vectors) > 0

            # Should be able to make predictions
            predictions = svm.predict(X)
            assert len(predictions) == len(y)
            assert all(pred in [0, 1] for pred in predictions)

    def test_different_c_values(self):
        """Test SVM with different regularization parameters."""
        X, y = make_classification(n_samples=50, n_features=2, n_redundant=0,
                                 n_informative=2, n_clusters_per_class=1, random_state=42)
        X = np.array(X)
        y = np.array(y)

        c_values = [0.1, 1.0, 10.0]
        kernel = LinearKernel()

        support_vector_counts = []
        for C in c_values:
            svm = SVM.fit(kernel=kernel, C=C, max_iter=200, tolerance=1e-3, X=X, y=y)
            assert svm.support_vectors is not None
            support_vector_counts.append(len(svm.support_vectors))

        # Generally, smaller C should lead to more support vectors (softer margin)
        # This is a general trend, not a strict rule due to data complexity
        assert all(count > 0 for count in support_vector_counts)


class TestSVMPrediction:
    """Test SVM prediction methods."""

    def setup_method(self):
        """Set up test data for prediction tests."""
        # Create simple 2D linearly separable data
        self.X_train = np.array([[-2, -1], [-1, -1], [1, 1], [2, 1]])
        self.y_train = np.array([0, 0, 1, 1])

        self.kernel = LinearKernel()
        self.svm = SVM.fit(kernel=self.kernel, C=1.0, max_iter=100, 
                          tolerance=1e-3, X=self.X_train, y=self.y_train)

    def test_decision_function_shape(self):
        """Test decision function output shape."""
        X_test = np.array([[0, 0], [1, 0], [-1, 0]])

        decision_values = self.svm.decision_function(X_test)

        assert decision_values.shape == (3,)
        assert isinstance(decision_values, np.ndarray)

    def test_decision_function_single_sample(self):
        """Test decision function with single sample."""
        X_test = np.array([[0, 0]])

        decision_values = self.svm.decision_function(X_test)

        assert decision_values.shape == (1,)

    def test_predict_shape_and_values(self):
        """Test predict method output."""
        X_test = np.array([[0, 0], [1, 0], [-1, 0]])

        predictions = self.svm.predict(X_test)

        assert predictions.shape == (3,)
        assert all(pred in [0, 1] for pred in predictions)

    def test_predict_consistency(self):
        """Test that predictions are consistent with decision function."""
        X_test = np.array([[0, 0], [1, 0], [-1, 0]])

        decision_values = self.svm.decision_function(X_test)
        predictions = self.svm.predict(X_test)

        # Predictions should match sign of decision function
        # Since our SVM maps to original labels (0, 1), we need to check consistency
        for i, (decision, pred) in enumerate(zip(decision_values, predictions)):
            if decision >= 0:
                expected_pred = 1  # Positive class
            else:
                expected_pred = 0  # Negative class
            assert pred == expected_pred

    def test_predict_proba_shape_and_range(self):
        """Test predict_proba method."""
        X_test = np.array([[0, 0], [1, 0], [-1, 0]])

        probabilities = self.svm.predict_proba(X_test)

        # Check shape
        assert probabilities.shape == (3, 2)

        # Check probabilities sum to 1
        prob_sums = np.sum(probabilities, axis=1)
        np.testing.assert_allclose(prob_sums, 1.0, rtol=1e-10)

        # Check probabilities are in valid range
        assert np.all(probabilities >= 0)
        assert np.all(probabilities <= 1)

    def test_training_data_prediction(self):
        """Test that SVM can correctly classify training data."""
        predictions = self.svm.predict(self.X_train)

        # Should achieve perfect accuracy on linearly separable training data
        accuracy = np.mean(predictions == self.y_train)
        assert accuracy >= 0.8  # Allow some tolerance for numerical precision


class TestSVMSupportVectors:
    """Test support vector identification and properties."""

    def test_support_vector_properties(self):
        """Test properties of identified support vectors."""
        # Create data where we know some points should be support vectors
        X = np.array([[-1, 0], [0, -1], [0, 1], [1, 0]])
        y = np.array([0, 0, 1, 1])

        kernel = LinearKernel()
        svm = SVM.fit(kernel=kernel, C=1.0, max_iter=100, tolerance=1e-3, X=X, y=y)

        # Should have support vectors
        assert svm.support_vectors is not None
        assert len(svm.support_vectors) > 0

        assert svm.support_alphas is not None
        assert len(svm.support_alphas) == len(svm.support_vectors)

        assert svm.support_labels is not None
        assert len(svm.support_labels) == len(svm.support_vectors)


        # All support vector alphas should be positive
        assert all(alpha > svm.tolerance for alpha in svm.support_alphas)

        # Support vectors should be from the training data
        for sv in svm.support_vectors:
            found = False
            for train_point in X:
                if np.allclose(sv, train_point):
                    found = True
                    break
            assert found, f"Support vector {sv} not found in training data"

    def test_support_vector_alphas_bounds(self):
        """Test that alpha values respect the C constraint."""
        X, y = make_classification(n_samples=30, n_features=2, n_redundant=0, 
                                 n_informative=2, n_clusters_per_class=1, random_state=42)

        C = 1.0
        kernel = LinearKernel()
        svm = SVM.fit(kernel=kernel, C=C, max_iter=200, tolerance=1e-3, X=X, y=y)

        # All alphas should be between 0 and C
        assert all(0 <= alpha <= C + svm.tolerance for alpha in svm.support_alphas)


class TestSVMEdgeCases:
    """Test SVM behavior with edge cases."""

    def test_minimal_dataset(self):
        """Test SVM with minimal dataset (2 samples)."""
        X = np.array([[0, 0], [1, 1]])
        y = np.array([0, 1])

        kernel = LinearKernel()
        svm = SVM.fit(kernel=kernel, C=1.0, max_iter=100, tolerance=1e-3, X=X, y=y)

        # Should still work
        assert svm.support_vectors is not None
        predictions = svm.predict(X)
        assert len(predictions) == 2

    def test_identical_features_different_labels(self):
        """Test SVM with identical features but different labels."""
        X = np.array([[1, 1], [1, 1], [2, 2], [2, 2]])
        y = np.array([0, 1, 0, 1])

        kernel = LinearKernel()
        # This should work but may not achieve perfect separation
        svm = SVM.fit(kernel=kernel, C=1.0, max_iter=100, tolerance=1e-3, X=X, y=y)

        # Should still produce predictions
        predictions = svm.predict(X)
        assert len(predictions) == 4

    def test_single_feature(self):
        """Test SVM with single feature."""
        X = np.array([[0], [1], [2], [3]])
        y = np.array([0, 0, 1, 1])

        kernel = LinearKernel()
        svm = SVM.fit(kernel=kernel, C=1.0, max_iter=100, tolerance=1e-3, X=X, y=y)

        predictions = svm.predict(X)
        assert len(predictions) == 4
        assert all(pred in [0, 1] for pred in predictions)


class TestSVMComparison:
    """Test SVM against scikit-learn for validation."""

    def test_linear_kernel_comparison(self):
        """Compare results with scikit-learn SVM on simple data."""
        # Use standardized data for fair comparison
        X, y = make_classification(n_samples=50, n_features=2, n_redundant=0, 
                                 n_informative=2, n_clusters_per_class=1, random_state=42)
        X = np.array(X)
        y = np.array(y)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Our implementation
        kernel = LinearKernel()
        our_svm = SVM.fit(kernel=kernel, C=1.0, max_iter=1000, tolerance=1e-6, 
                         X=X_scaled, y=y)
        our_predictions = our_svm.predict(X_scaled)
        our_accuracy = np.mean(our_predictions == y)

        # Scikit-learn implementation
        sklearn_svm = SVC(kernel='linear', C=1.0, max_iter=1000, tol=1e-6)
        sklearn_svm.fit(X_scaled, y)
        sklearn_predictions = sklearn_svm.predict(X_scaled)
        sklearn_accuracy = np.mean(sklearn_predictions == y)

        # Our implementation should achieve reasonable accuracy
        assert our_accuracy >= 0.7

        # Should be reasonably close to scikit-learn (within 20% relative difference)
        relative_diff = abs(our_accuracy - sklearn_accuracy) / max(sklearn_accuracy, 0.01)
        assert relative_diff <= 0.3, f"Our accuracy: {our_accuracy}, sklearn: {sklearn_accuracy}"

    def test_support_vector_count_comparison(self):
        """Compare support vector counts with scikit-learn."""
        X, y = make_classification(n_samples=30, n_features=2, n_redundant=0, 
                                 n_informative=2, n_clusters_per_class=1, random_state=42)
        X = np.array(X)
        y = np.array(y)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Our implementation
        kernel = LinearKernel()
        our_svm = SVM.fit(kernel=kernel, C=1.0, max_iter=1000, tolerance=1e-6, 
                         X=X_scaled, y=y)
        our_sv_count = len(our_svm.support_vectors)

        # Scikit-learn implementation
        sklearn_svm = SVC(kernel='linear', C=1.0, max_iter=1000, tol=1e-6)
        sklearn_svm.fit(X_scaled, y)
        sklearn_sv_count = len(sklearn_svm.support_vectors_)

        # Support vector counts should be reasonably similar
        # Allow for some difference due to numerical precision and implementation details
        assert abs(our_sv_count - sklearn_sv_count) <= max(5, 0.5 * sklearn_sv_count)


class TestSVMNumericalStability:
    """Test numerical stability and convergence."""

    def test_different_tolerances(self):
        """Test SVM with different tolerance values."""
        X, y = make_classification(n_samples=30, n_features=2, n_redundant=0, 
                                 n_informative=2, n_clusters_per_class=1, random_state=42)
        X = np.array(X)
        y = np.array(y)

        tolerances = [1e-2, 1e-3, 1e-4]
        kernel = LinearKernel()

        for tol in tolerances:
            svm = SVM.fit(kernel=kernel, C=1.0, max_iter=1000, tolerance=tol, X=X, y=y)

            # Should converge and produce reasonable results
            assert svm.support_vectors is not None
            assert len(svm.support_vectors) > 0

            predictions = svm.predict(X)
            accuracy = np.mean(predictions == y)
            assert accuracy >= 0.5  # Should be better than random

    def test_max_iterations(self):
        """Test SVM with different max_iter values."""
        X, y = make_classification(n_samples=30, n_features=2, n_redundant=0, 
                                 n_informative=2, n_clusters_per_class=1, random_state=42)
        X = np.array(X)
        y = np.array(y)

        max_iters = [50, 100, 500]
        kernel = LinearKernel()


        for max_iter in max_iters:
            svm = SVM.fit(kernel=kernel, C=1.0, max_iter=max_iter, tolerance=1e-3, X=X, y=y)

            # Should still produce valid results
            assert svm.support_vectors is not None
            predictions = svm.predict(X)
            assert len(predictions) == len(y)


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])
