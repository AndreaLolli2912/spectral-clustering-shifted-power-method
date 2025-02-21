import numpy as np


def similarity_function(X_i: np.ndarray, X_j: np.ndarray, sigma: float=1) -> float:
    """
    PARAMETERS
    ----------
    X_i : numpy array
        first vector
    X_j : numpy array
        second vector

    RETURNS
    ----------
    float : similarity between X_i and X_j
    """
    return np.exp( - (np.linalg.norm(X_i - X_j) ** 2) / 2 * sigma ** 2)

def weight_adjacency_matrix(X: np.ndarray) -> np.ndarray:
    """
    Constructs the weight adjacency matrix from a given data matrix.
    PARAMETERS
    ----------
    X : numpy ndarray
        data matrix

    RETURNS
    ----------
    W: numpy ndarray
        weight adjacency matrix
    """
    W = np.zeros((X.shape[0], X.shape[0]))
    for i in range(X.shape[0]):
        for j in range(i+1, X.shape[0]): # upper triangular matrix, we can exploit the simmetry of the matrix
            W[i, j] = similarity_function(X[i], X[j])
            W[j, i] = W[i, j]
    return W

def knn_similarity_graph(W:np.ndarray, k:int=5, mutuality:bool=False) -> np.ndarray:
    """
    Constructs a k-nearest neighbors similarity graph from a given weight adjacency matrix.
    The graph representation is an adjacency matrix where S[i, j] represents the weight of the edge between vertices i and j.
    Parameters
    ----------
    W : numpy.ndarray
        Weight adjacency matrix where W[i, j] represents the weight of the edge between vertices i and j.
    k : int, optional
        Number of nearest neighbors to consider.
    mutuality : bool, optional
        - If False, connect vi and vj with an undirected edge if vi is among the k-nearest neighbors of vj or if vj is among the k-nearest neighbors of vi.
        - If True, connect vi and vj if both vi is among the k-nearest neighbors of vj and vj is among the k-nearest neighbors of vi. 
        The resulting graph is called the mutual k-nearest neighbor graph.

    Returns
    -------
    S: numpy.ndarray
        k-nearest neighbors similarity graph
    """
    n = W.shape[0]
    S = np.zeros((n, n))
    for i in range(n): # iterate for the rows of the matrix
        k_nearest_neighbours = np.argsort(W[i, :])[-np.min(k, n):] # get the k-nearest neighbours
        for j in k_nearest_neighbours:
            if mutuality:
                if i in np.argsort(W[j, :])[-k:]:
                    S[i, j] = W[i, j]
                    S[j, i] = W[j, i]
            else:
                S[i, j] = W[i, j]
                S[j, i] = W[j, i]
    return S

def get_knn_similarity_graph(X:np.ndarray, k:int=5, mutuality:bool=False, sigma:float=1) -> np.ndarray:
    """
    Constructs the k-nearest neighbors similarity graph from a given data matrix.
    The graph representation is an adjacency matrix where S[i, j] represents the weight of the edge between vertices i and j.
    Parameters
    ----------
    X : numpy.ndarray
        Data matrix where each row represents a sample and each column a feature.
    k : int, optional
        Number of nearest neighbors to consider.
    mutuality : bool, optional
        - If False, connect vi and vj with an undirected edge if vi is among the k-nearest neighbors of vj or if vj is among the k-nearest neighbors of vi.
        - If True, connect vi and vj if both vi is among the k-nearest neighbors of vj and vj is among the k-nearest neighbors of vi. 
        The resulting graph is called the mutual k-nearest neighbor graph.
    sigma : float, optional
        Standard deviation of the Gaussian kernel used to compute similarities between points.

    Returns
    -------
    S: numpy.ndarray
        k-nearest neighbors similarity graph
    """
    W = weight_adjacency_matrix(X)
    S = knn_similarity_graph(W, k, mutuality)
    return S