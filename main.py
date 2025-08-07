from __future__ import annotations

from typing import Type, Any, Tuple
from numpy.typing import NDArray

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, hamming_loss, silhouette_score, davies_bouldin_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from graph import Graph, ClassificationMethod
from kernels import LinearKernel, RBFKernel, PolynomialKernel
from svm_classifier import SVM


def load_and_preprocess_data(dataset_path: str, dataset_type: Type[nx.Graph], max_nodes: int | None = None) \
        -> Tuple[NDArray, NDArray]:
    """Load and preprocess graph data."""
    print(f"\n=== Loading dataset: {dataset_path} ===")

    # Load graph data
    graph = Graph(dataset_path, dataset_type, max_nodes=max_nodes)

    # Extract node features
    print("Extracting node features...")
    features = graph.extract_node_features()

    # Create classification labels based on node degree
    print("Creating classification labels...")
    labels = graph.create_classification_labels(method=ClassificationMethod.DEGREE_THRESHOLD)

    print(f"Dataset shape: {features.shape}")
    print(f"Label distribution: {np.bincount(labels)}")

    return features, labels


def evaluate_svm_classification(X: NDArray, y: NDArray) -> dict[Any, Any]:
    """Evaluate SVM classification performance."""
    print("\n=== SVM Classification Evaluation ===")

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

    # Standardize features
    scaler = StandardScaler()
    x_train_scaled: NDArray = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Test different kernels
    kernels = {
        'Linear': LinearKernel(),
        'RBF': RBFKernel(gamma=0.1),
        'Polynomial': PolynomialKernel(degree=2, gamma=0.1)
    }

    results = {}

    for kernel_name, kernel in kernels.items():
        print(f"\nTesting {kernel_name} kernel...")

        # Train SVM
        svm = SVM.fit(kernel=kernel, C=1.0, max_iter=500, tolerance=1e-3, X=x_train_scaled, y=y_train)

        # Make predictions
        y_pred = svm.predict(x_test_scaled)

        # Evaluate
        metrics = {'macro_f1': f1_score(y_test, y_pred, average='macro'),
                   'micro_f1': f1_score(y_test, y_pred, average='micro'),
                   'hamming_loss': hamming_loss(y_test, y_pred)}

        results[kernel_name] = metrics

        print(f"  Macro-F1: {metrics['macro_f1']:.4f}")
        print(f"  Micro-F1: {metrics['micro_f1']:.4f}")
        print(f"  Hamming Loss: {metrics['hamming_loss']:.4f}")
        print(f"  Support Vectors: {len(svm.support_vectors) if svm.support_vectors is not None else 0}")

    return results


def evaluate_clustering_performance(X: NDArray, n_clusters: int) -> tuple[dict[str, float], Any]:
    """Evaluate clustering performance using node features."""
    print("\n=== Clustering Evaluation ===")

    # Standardize features
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(X)

    # Perform K-means clustering
    print(f"Performing K-means clustering with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(x_scaled)

    print(f"Cluster distribution: {np.bincount(cluster_labels)}")

    # Evaluate clustering
    metrics = {'silhouette_coefficient': silhouette_score(x_scaled, cluster_labels),
               'davies_bouldin_index': davies_bouldin_score(x_scaled, cluster_labels)}

    print(f"Silhouette Coefficient: {metrics['silhouette_coefficient']:.4f}")
    print(f"Davies-Bouldin Index: {metrics['davies_bouldin_index']:.4f}")

    return metrics, cluster_labels


def plot_results(classification_results, clustering_metrics):
    """Plot evaluation results."""
    print("\n=== Plotting Results ===")

    # Create subplots
    _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Classification F1 scores
    kernels = list(classification_results.keys())
    macro_f1 = [classification_results[k]['macro_f1'] for k in kernels]
    micro_f1 = [classification_results[k]['micro_f1'] for k in kernels]

    x = np.arange(len(kernels))
    width = 0.35

    ax1.bar(x - width/2, macro_f1, width, label='Macro-F1', alpha=0.8)
    ax1.bar(x + width/2, micro_f1, width, label='Micro-F1', alpha=0.8)
    ax1.set_xlabel('Kernel')
    ax1.set_ylabel('F1 Score')
    ax1.set_title('SVM Classification Performance')
    ax1.set_xticks(x)
    ax1.set_xticklabels(kernels)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Hamming Loss
    hamming_losses = [classification_results[k]['hamming_loss'] for k in kernels]
    ax2.bar(kernels, hamming_losses, alpha=0.8, color='orange')
    ax2.set_xlabel('Kernel')
    ax2.set_ylabel('Hamming Loss')
    ax2.set_title('SVM Hamming Loss (Lower is Better)')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Clustering metrics
    metrics_names = ['Silhouette\nCoefficient', 'Davies-Bouldin\nIndex']
    metrics_values = [
        clustering_metrics['silhouette_coefficient'],
        clustering_metrics['davies_bouldin_index']
    ]

    colors = ['green', 'red', 'blue']
    bars = ax3.bar(metrics_names, metrics_values, alpha=0.8, color=colors)
    ax3.set_ylabel('Metric Value')
    ax3.set_title('Clustering Performance Metrics')
    ax3.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, value in zip(bars, metrics_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.3f}', ha='center', va='bottom')

    # Plot 4: Summary comparison
    ax4.text(0.1, 0.8, 'Best Classification Results:', fontsize=12, fontweight='bold')
    best_kernel = max(classification_results.keys(),
                     key=lambda k: classification_results[k]['macro_f1'])
    ax4.text(0.1, 0.7, f'Best Kernel: {best_kernel}', fontsize=10)
    ax4.text(0.1, 0.6, f'Macro-F1: {classification_results[best_kernel]["macro_f1"]:.4f}', fontsize=10)
    ax4.text(0.1, 0.5, f'Micro-F1: {classification_results[best_kernel]["micro_f1"]:.4f}', fontsize=10)

    ax4.text(0.1, 0.3, 'Clustering Results:', fontsize=12, fontweight='bold')
    ax4.text(0.1, 0.2, f'Silhouette: {clustering_metrics["silhouette_coefficient"]:.4f}', fontsize=10)
    ax4.text(0.1, 0.1, f'Davies-Bouldin: {clustering_metrics["davies_bouldin_index"]:.4f}', fontsize=10)

    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    ax4.set_title('Summary')

    plt.tight_layout()
    plt.savefig('svm_evaluation_results.png', dpi=300, bbox_inches='tight')
    print("Results saved to 'svm_evaluation_results.png'")
    #plt.show()

def run(dataset_path: str, dataset_type: Type[nx.Graph]) -> None:
    """Load, train and test a given dataset."""
    # Load and preprocess data
    features, labels = load_and_preprocess_data(dataset_path, dataset_type, max_nodes=10000)

    # Evaluate SVM classification
    classification_results = evaluate_svm_classification(features, labels)

    # Evaluate clustering
    clustering_metrics, cluster_labels = evaluate_clustering_performance(features, n_clusters=2)

    # Plot results
    plot_results(classification_results, clustering_metrics)

    # Print summary
    print("\n=== Summary ===")
    print(f"Dataset: {dataset_path}")
    print(f"Features: {features.shape[1]} (Degree, Clustering, Betweeness")
    print(f"Samples: {features.shape[0]}")

    best_kernel = max(classification_results.keys(),
                      key=lambda k: classification_results[k]['macro_f1'])
    print(f"Best SVM kernel: {best_kernel}")
    print(f"Best Macro-F1: {classification_results[best_kernel]['macro_f1']:.4f}")

    print(f"Clustering Silhouette Score: {clustering_metrics['silhouette_coefficient']:.4f}")

    print("\nEvaluation complete!")

def main() -> None:
    #run("datasets/CA-GrQc.txt", nx.DiGraph)
    run("datasets/com-dblp.ungraph.txt", nx.Graph)

if __name__ == "__main__":
    main()
