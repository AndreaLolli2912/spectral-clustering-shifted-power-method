import numpy as np
import scipy.sparse as sp
from typing import Tuple, Union
from knn_similarity_graph import get_knn_similarity_graph

def degree_matrix(W: Union[np.ndarray, sp.csr_array]) -> Union[np.ndarray, sp.dia_array]:
    """
    Constructs the degree matrix from a given weight adjacency matrix.
    The degree matrix is a diagonal matrix where D[i, i] represents the degree of vertex i.

    PARAMETERS:
    ----------
    W : np.ndarray or sp.csr_array
        Weight adjacency matrix (dense or sparse).

    RETURNS:
    -------
    np.ndarray or sp.dia_array
        Degree matrix (dense or sparse depending on input).
    """
    degrees = np.array(W.sum(axis=1)).flatten()  # Ensure degrees is a 1D array
    
    return sp.dia_matrix((degrees, [0]), shape=(W.shape[0], W.shape[0]))



def get_laplacian_matrix(X: np.ndarray, 
                         k: int = 5, 
                         mutuality: bool = False, 
                         sigma: float = 1, 
                         sparse: bool = False) -> Tuple[Union[np.ndarray, sp.csr_array], 
                                                        Union[np.ndarray, sp.csr_array], 
                                                        Union[np.ndarray, sp.dia_array]]:
    """
    Constructs the Laplacian matrix from a given data matrix considering k nearest neighbours.
    The Laplacian matrix is defined as L = D - W.

    PARAMETERS:
    ----------
    X : np.ndarray
        Data matrix where each row represents a sample and each column a feature.
    k : int, optional
        Number of nearest neighbors to consider.
    mutuality : bool, optional
        Use mutual k-NN if True, else regular k-NN.
        - If False, connect vi and vj with an undirected edge if vi is among the k-nearest neighbors of vj or if vj is among the k-nearest neighbors of vi.
        - If True, connect vi and vj if both vi is among the k-nearest neighbors of vj and vj is among the k-nearest neighbors of vi. 
    sigma : float, optional
        Standard deviation of the Gaussian kernel.
    sparse : bool, optional
        If True, the matrices are returned in sparse format.

    RETURNS:
    -------
    Tuple[
        Union[np.ndarray, sp.csr_matrix], 
        Union[np.ndarray, sp.csr_matrix], 
        Union[np.ndarray, sp.dia_matrix]
    ]
        Laplacian matrix (L), weight matrix (W), and degree matrix (D).
    """
    # Get weight adjacency matrix
    W = get_knn_similarity_graph(X, k, mutuality, sigma)

    # Convert to sparse if needed
    if sparse:
        W = sp.csr_matrix(W)

    # Compute degree matrix
    D = degree_matrix(W)

    # Compute Laplacian
    L = D - W

    return L, W, D
