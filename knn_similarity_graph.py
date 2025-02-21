import numpy as np
from scipy.spatial.distance import cdist

def weight_adjacency_matrix(X: np.ndarray, sigma: float = 1) -> np.ndarray:
    """
    Constructs the weight adjacency matrix using the Gaussian similarity function.

    PARAMETERS:
    ----------
    X : np.ndarray
        Data matrix where each row is a sample.
    sigma : float, optional
        Standard deviation of the Gaussian kernel.

    RETURNS:
    -------
    np.ndarray
        Weight adjacency matrix.
    """
    # Compute pairwise distances and apply the Gaussian kernel
    distances = cdist(X, X, metric="sqeuclidean")  # Squared Euclidean distances
    W = np.exp(-distances / (2 * sigma ** 2))
    np.fill_diagonal(W, 0)  # Set diagonal to zero
    return W


def knn_similarity_graph(W: np.ndarray, k: int = 5, mutuality: bool = False) -> np.ndarray:
    """
    Constructs a k-nearest neighbors similarity graph from a weight adjacency matrix.

    PARAMETERS:
    ----------
    W : np.ndarray
        Weight adjacency matrix.
    k : int, optional
        Number of nearest neighbors.
    mutuality : bool, optional
        If True, uses mutual k-NN.

    RETURNS:
    -------
    S: np.ndarray
        k-nearest neighbors similarity graph.
    """
    n = W.shape[0]
    S = np.zeros_like(W)

    for i in range(n):
        # Find k-nearest neighbors
        neighbors_i = np.argsort(W[i, :])[-k:]

        for j in neighbors_i:
            if mutuality:
                # Mutual k-NN check
                neighbors_j = np.argsort(W[j, :])[-k:]
                if i in neighbors_j:
                    S[i, j] = W[i, j]
                    S[j, i] = W[j, i]
            else:
                # Regular k-NN
                S[i, j] = W[i, j]
                S[j, i] = W[j, i]

    return S


def get_knn_similarity_graph(X: np.ndarray, k: int = 5, mutuality: bool = False, sigma: float = 1) -> np.ndarray:
    """
    Constructs a k-nearest neighbors similarity graph from a given data matrix.

    PARAMETERS:
    ----------
    X : np.ndarray
        Data matrix where each row represents a sample.
    k : int, optional
        Number of nearest neighbors.
    mutuality : bool, optional
        If True, uses mutual k-NN.
    sigma : float, optional
        Standard deviation of the Gaussian kernel.

    RETURNS:
    -------
    np.ndarray
        k-nearest neighbors similarity graph.
    """
    W = weight_adjacency_matrix(X, sigma)
    return knn_similarity_graph(W, k, mutuality)
