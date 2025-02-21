import scipy.sparse as sp
import scipy.sparse.linalg as spla


def get_smallest_eigenpairs(X: sp.csr_array, M: int):
    """
    Compute the M smallest eigenvalues and their corresponding eigenvectors of a symmetric sparse matrix X.

    PARAMETERS
    ----------
    X : scipy.sparse.csr_array
        Symmetric sparse matrix for which to compute the smallest eigenvalues and eigenvectors.
        Must be a square matrix.
    M : int
        The number of smallest eigenvalues and eigenvectors to compute.

    RETURNS
    -------
    eigenvalues : numpy.ndarray
        Array containing the M smallest eigenvalues of matrix X in ascending order.
    eigenvectors : numpy.ndarray
        Array where each column is the eigenvector corresponding to the eigenvalue at the same index in `eigenvalues`.
    """
    # check if the matrix X is symmetric
    if (X != X.transpose()).nnz != 0:
        raise ValueError("Input matrix X must be symmetric.")

    # Compute the M smallest eigenvalues and their corresponding eigenvectors using eigsh
    eigenvalues, eigenvectors = spla.eigsh(X, which="SM", k=M)

    # Sort the eigenvalues and corresponding eigenvectors in ascending order
    sorted_indices = eigenvalues.argsort()
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    return eigenvalues[:M], eigenvectors[:, :M]