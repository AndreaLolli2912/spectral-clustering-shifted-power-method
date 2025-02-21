import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

def count_connected_components_with_scipy(X:sp.csr_array) -> int:
    """
    Determine the number of connected components in a graph represented by its Laplacian matrix.

    PARAMETERS
    ----------
    X : scipy.sparse.csr_array
        Laplacian matrix of the graph. Must be a square symmetric matrix representing the graph structure.

    RETURNS
    -------
    n_components : int
        Number of connected components in the graph.

    NOTES
    -----
    - This function utilizes SciPy's `connected_components` from the `scipy.sparse.csgraph` module
      to compute the connected components of the graph.
    - The Laplacian matrix should be properly constructed, where non-zero off-diagonal entries
      represent edges between nodes, and diagonal entries represent the degree of each node.
    """
    n_components, labels = connected_components(X)
    return n_components


def count_connected_components_with_traversal(X: sp.csr_array, method: str = 'BFS') -> int:
    """
    Count the number of connected components in an undirected graph using BFS or DFS on a sparse adjacency matrix.

    PARAMETERS
    ----------
    X : scipy.sparse.csr_array
        Sparse adjacency matrix of the graph in Compressed Sparse Row (CSR) format.
        Each non-zero entry `W[i, j]` indicates an undirected edge between nodes `i` and `j`.
        The matrix must be square and symmetric to represent an undirected graph correctly.

    method : str, optional
        The graph traversal method to use: 'BFS' for Breadth-First Search or 'DFS' for Depth-First Search.
        Default is 'BFS'.

    RETURNS
    -------
    components: int
        The total number of connected components in the graph.
    """

    n = X.shape[0]  # Number of nodes in the graph
    visited = [False] * n  # Track visited nodes
    components = 0  # Initialize connected components count

    def bfs(start_node: int):
        """Perform Breadth-First Search starting from the given node."""
        queue = [start_node]  # Initialize BFS queue with the start node
        visited[start_node] = True  # Mark start node as visited
        while queue:
            current = queue.pop(0)  # Dequeue the next node to visit
            neighbors = X[current].indices  # Retrieve adjacent nodes efficiently
            for neighbor in neighbors:
                if not visited[neighbor]:
                    visited[neighbor] = True  # Mark neighbor as visited
                    queue.append(neighbor)  # Enqueue neighbor for exploration

    def dfs(node: int):
        """Perform Depth-First Search starting from the given node."""
        stack = [node]  # Initialize DFS stack with the start node
        visited[node] = True  # Mark the node as visited
        while stack:
            current = stack.pop()  # Pop the top node from the stack
            neighbors = X[current].indices  # Retrieve adjacent nodes efficiently
            for neighbor in neighbors:
                if not visited[neighbor]:
                    visited[neighbor] = True  # Mark neighbor as visited
                    stack.append(neighbor)  # Push neighbor onto the stack for further exploration

    # Iterate through each node in the graph
    for node in range(n):
        if not visited[node]:
            components += 1  # Found a new connected component
            if method.upper() == 'BFS':
                bfs(node)  # Explore using BFS
            elif method.upper() == 'DFS':
                dfs(node)  # Explore using DFS
            else:
                raise ValueError("Method must be either 'BFS' or 'DFS'.")

    return components


