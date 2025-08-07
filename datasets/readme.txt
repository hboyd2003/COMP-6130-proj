Two graph datsets for experiment evaluation: Collaboration network of Arxiv (small scale) and Collaboration network of DBLP (large scale)

In these two graphs, a node/vertex represents an author and an edge denotes an coauthorship between two authors. Thus, each row corresponds to an edge with a value of 1. During the experiment evaluation, these two datasets could be used to model data matrices of non-graph data or adjacency matrices of graph data. For example, the first row of "3466	937" in CA-GrQc.txt could be used to represent the entry between row 3466 and column 937 in the matrix with the value of 1.

Notice that CA-GrQc is a directed graph. i.e., its matrix representation is a symmetric matrix. But the undirected graph com-dblp.ungraph has an asymmetric matrix representation. You should first transform it into a symmetric matrix since most of data mining algorithms only handle symmetric data matrix.

In addition, these two datasets are optional. You could use the public datasets used in the papers that you choose for the experiments.

Snippet from `CA-GrQc.txt`:
# Directed graph (each unordered pair of nodes is saved once): CA-GrQc.txt 
# Collaboration network of Arxiv General Relativity category (there is an edge if authors coauthored at least one paper)
# Nodes: 5242 Edges: 28980
# FromNodeId	ToNodeId
3466	937
3466	5233
3466	8579
3466	10310
3466	15931
3466	17038
3466	18720
3466	19607
10310	1854


Snippet from `com-dblp.ungraph.txt`:
# Undirected graph: ../../data/output/dblp.ungraph.txt
# DBLP
# Nodes: 317080 Edges: 1049866
# FromNodeId	ToNodeId
0	1
0	2
0	4519
0	23073
0	33043
0	33971
0	75503
0	101215
0	120044
0	123880
0	124002
0	206567
0	274042
0	369692
0	411025
0	413808
1	2
1	5915
1	7741
1	7852
1	7979
1	8085
1	8086
1	9335
1	10971
1	12238
1	13090