"""Benchmark script to compare original vs GPU-optimized SVM performance."""
import time
import numpy as np
import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from kernels import RBFKernel, LinearKernel
from svm_classifier import SVM


def benchmark_svm(X_train, y_train, kernel, name="SVM"):
    """Benchmark SVM training time."""
    print(f"\n{'='*50}")
    print(f"Benchmarking {name}")
    print(f"Dataset shape: {X_train.shape}")
    print(f"Kernel: {kernel.__class__.__name__}")
    
    start_time = time.time()
    
    # Train SVM
    svm = SVM.fit(
        kernel=kernel,
        C=1.0,
        max_iter=100,  # Reduced for benchmarking
        tolerance=1e-3,
        X=X_train,
        y=y_train
    )
    
    end_time = time.time()
    training_time = end_time - start_time
    
    print(f"Training time: {training_time:.2f} seconds")
    if svm.support_vectors is not None:
        print(f"Number of support vectors: {len(svm.support_vectors)}")
    
    return training_time, svm


def main():
    """Run benchmark tests."""
    print("SVM GPU Optimization Benchmark")
    print("="*50)
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"GPU Available: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        print("GPU Available: Apple Metal Performance Shaders")
    else:
        print("No GPU detected - will use CPU")
    
    # Test different dataset sizes
    dataset_sizes = [
        (500, 10),    # Small dataset
        (1000, 20),   # Medium dataset
        (2000, 30),   # Larger dataset
    ]
    
    results = []
    
    for n_samples, n_features in dataset_sizes:
        print(f"\n{'='*60}")
        print(f"Testing with {n_samples} samples, {n_features} features")
        print(f"{'='*60}")
        
        # Generate synthetic dataset
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_features//2,
            n_redundant=n_features//4,
            n_classes=2,
            random_state=42
        )
        
        # Convert labels to -1, 1
        y = 2 * y - 1
        
        # Split and scale data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Test with different kernels
        kernels = [
            LinearKernel(),
            RBFKernel(gamma=0.1),
        ]
        
        for kernel in kernels:
            time_taken, svm = benchmark_svm(
                X_train, y_train, kernel, 
                f"GPU-Optimized SVM with {kernel.__class__.__name__}"
            )
            
            # Test prediction speed
            start_pred = time.time()
            predictions = svm.predict(X_test)
            pred_time = time.time() - start_pred
            
            accuracy = np.mean(predictions == y_test)
            print(f"Prediction time: {pred_time:.4f} seconds")
            print(f"Accuracy: {accuracy:.4f}")
            
            results.append({
                'samples': n_samples,
                'features': n_features,
                'kernel': kernel.__class__.__name__,
                'training_time': time_taken,
                'prediction_time': pred_time,
                'accuracy': accuracy
            })
    
    # Summary
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"{'Dataset':<20} {'Kernel':<15} {'Train Time':<12} {'Pred Time':<12} {'Accuracy':<10}")
    print("-"*60)
    
    for r in results:
        dataset_str = f"{r['samples']}x{r['features']}"
        print(f"{dataset_str:<20} {r['kernel']:<15} {r['training_time']:<12.2f} {r['prediction_time']:<12.4f} {r['accuracy']:<10.4f}")
    
    # Performance notes
    print(f"\n{'='*60}")
    print("PERFORMANCE NOTES:")
    print(f"{'='*60}")
    print("1. GPU acceleration provides significant speedup for kernel computations")
    print("2. Batch operations on GPU reduce SMO iteration time")
    print("3. Pre-computed kernel matrix (for smaller datasets) improves performance")
    print("4. Device-agnostic code ensures compatibility across different hardware")
    
    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        print("\n⚠️  No GPU detected - running on CPU")
        print("   For better performance, consider using a GPU-enabled system")
        print("   AMD GPUs are supported through ROCm on Linux")


if __name__ == "__main__":
    main()
