from __future__ import annotations

import enum
from typing import Type, Any, Dict

import networkx as nx
import numpy as np
from numpy.typing import ArrayLike, NDArray


class ClassificationMethod(enum.StrEnum):
    """Types of classification methods"""

    DEGREE_THRESHOLD = 'degree_threshold'
    CLUSTERING_THRESHOLD = 'clustering_threshold'


class Graph:
    """Represents a Graph"""

    def __init__(self,
                 filepath: str,
                 graph_type: Type[nx.Graph],
                 max_nodes: int | None) -> None:
        self.graph = self.load(filepath, graph_type, max_nodes)

    @staticmethod
    def load(filepath: str,
             graph_type: Type[nx.Graph],
             max_nodes: int | None = None) -> nx.Graph:
        """
        Load file into NetworkX graph object with graph_type.

        :param filepath: Path to the file to load.
        :param graph_type: NetworkX graph class type to load file as
        :param max_nodes: Max number of nodes to load.
        :return: NetworkX graph object.
        """
        graph: nx.Graph = nx.read_edgelist(filepath, nodetype=int, data=True, create_using=graph_type())
        if max_nodes is not None:
            nodes_to_keep = set(list(graph.nodes)[:max_nodes])
            graph = graph.subgraph(nodes_to_keep)

        return graph

    def get_adjacency_matrix(self) -> ArrayLike:
        """
        Gets the adjacency matrix representation of the graph.

        :return: Adjacency matrix.
        """
        if self.graph is None:
            raise ValueError("Graph must be loaded first")

        # Get adjacency matrix with consistent node ordering
        adj_matrix = nx.adjacency_matrix(self.graph, nodelist=sorted(self.graph.nodes()))

        return adj_matrix.toarray()

    def extract_node_features(self) -> NDArray:
        """
        Extract Degree, Clustering and Betweeness node features from the graph.

        :return: Array of Degree, Clustering and Betweeness node features
        """
        if self.graph is None:
            raise ValueError("Graph must be loaded first")

        features = []
        nodes = sorted(self.graph.nodes())

        # Degree
        degrees = [self.graph.degree(node) for node in nodes]
        features.append(degrees)

        # Clustering
        clustering: float | int | Dict[nx.Graph, float | int] = nx.clustering(self.graph)
        assert isinstance(clustering, Dict)
        clustering_values = [clustering[node] for node in nodes]
        features.append(clustering_values)

        # Betweeness
        # Use sampling for large graphs
        if self.graph.number_of_nodes() > 1000:
            betweenness = nx.betweenness_centrality(self.graph, k=min(100, self.graph.number_of_nodes()))
        else:
            betweenness = nx.betweenness_centrality(self.graph)
        betweenness_values = [betweenness.get(node, 0) for node in nodes]
        features.append(betweenness_values)

        return np.column_stack(features)

    def create_classification_labels(self,
                                     method: ClassificationMethod = ClassificationMethod.DEGREE_THRESHOLD,
                                     threshold: np.floating[Any] | None = None) -> NDArray:
        """
        Create binary classification labels based on graph properties.

        :param method: Classification method to use.
        :param threshold: Threshold for classification (defaults to median degree if `None`).
        :return: Array of binary classification labels.
        """
        if self.graph is None:
            raise ValueError("Graph must be loaded first")

        nodes = sorted(self.graph.nodes())

        if method == 'degree_threshold':
            degrees: ArrayLike = np.array([self.graph.degree(node) for node in nodes])
            if threshold is None:
                threshold = np.median(degrees)
            labels = [1 if deg > threshold else 0 for deg in degrees]
            print(f"Created degree-based labels with threshold {threshold}")

        elif method == 'clustering_threshold':
            clustering: float | int | Dict[nx.Graph, float | int] = nx.clustering(self.graph)
            assert isinstance(clustering, Dict)
            clustering_values = [clustering[node] for node in nodes]
            if threshold is None:
                threshold = np.median(clustering_values)
            labels = [1 if clust > threshold else 0 for clust in clustering_values]
            print(f"Created clustering-based labels with threshold {threshold}")

        else:
            raise ValueError(f"Unknown method: {method}")

        return np.array(labels)
