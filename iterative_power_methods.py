import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def inverse_power_method(X: sp.csr_array,
                         v_init: sp.csr_array = None,
                         shift: float = 1e-6,
                         max_iter: int = 1000,
                         tolerance: float = 1e-8):
    """
    Perform the inverse power method to find the smallest eigenvalue and 
    corresponding eigenvector of a symmetric sparse matrix X.

    PARAMETERS
    ----------
    X: sp.csr_array
        Sparse matrix for which to find the smallest eigenvalue.
    v_init: sp.csr_array
        Sparse vector (n x 1) for initial guess of the eigenvector.
    shift: float
        Shift value to apply to the matrix (X - shift*I). 
        Must be chosen so that (X - shift*I) is invertible.
    max_iter: int
        Maximum number of iterations.
    tolerance: float
        Convergence tolerance.

    RETURNS
    -------
    eigenvalue: float
        The smallest eigenvalue found (approximately).
    eigenvector: sp.csr_array
        The corresponding eigenvector (as a sparse column vector).
    """

    # Basic check
    assert X.shape[0] == X.shape[1], "Matrix is not square"
    
    n = X.shape[0]
    
    # If no initial vector is provided, create a random sparse vector (n x 1).
    if v_init is None:
        v_init = sp.rand(n, 1, density=1.0, format='csr')

    init_norm = np.sqrt(v_init.multiply(v_init).sum())
    if init_norm == 0:
        raise ValueError("Initial vector has zero norm. Please provide a nonzero v_init.")

    # Normalize initial vector
    v_curr = v_init / init_norm

    # Construct the shifted matrix
    I_n = sp.eye(n, format='csr')
    X_shifted = X - shift * I_n

    # Precompute LU factorization of the shifted matrix
    lu = spla.splu(X_shifted.tocsc())

    mu_curr = shift  # Initial guess for the eigenvalue shift

    for _ in range(max_iter):
        v_prev = v_curr

        # Solve (X - shift*I) * v_curr_bar = v_prev using LU factorization
        rhs_dense = v_prev.toarray().ravel()
        v_curr_bar_dense = lu.solve(rhs_dense)

        # Convert the dense solution back to a sparse (n x 1) vector
        v_curr_bar = sp.csr_matrix(v_curr_bar_dense).reshape(-1, 1)

        # Rayleigh-like estimate for 1/(lambda - shift): dot(v_curr_bar, v_prev)
        mu_prev = mu_curr
        mu_curr = v_curr_bar.multiply(v_prev).sum()

        # Normalize v_curr_bar
        bar_norm = np.sqrt(v_curr_bar.multiply(v_curr_bar).sum())
        if bar_norm == 0:
            break
        v_curr = v_curr_bar / bar_norm

        # Convergence check
        if abs(mu_curr - mu_prev) < tolerance * abs(mu_curr):
            break

    # The corresponding eigenvalue is shift + 1/mu
    eigenvalue = shift + 1.0 / mu_curr

    # Fixing negative eigenvalues due to numerical imprecision
    #eigenvalue = abs(eigenvalue) # TODO: CHECK

    eigenvector = v_curr.tocsr()

    return eigenvalue, eigenvector


def deflation_step(X: sp.csr_array, eigenvector: sp.csr_array):
    """
    Perform a deflation step on matrix X by removing the influence of the given eigenvector.

    Parameters
    ----------
    X : scipy.sparse.csr_array
        Symmetric sparse matrix to be deflated. Must be square.
    eigenvector : scipy.sparse.csr_array
        The eigenvector corresponding to the eigenvalue to be deflated.
        Should be a normalized column vector with shape (n, 1).

    Returns
    -------
    B : scipy.sparse.csr_array
        The deflated matrix, where the influence of the specified eigenvalue and eigenvector has been removed.
    P : scipy.sparse.csr_array
        The projection matrix used for deflation.
    """
    n = X.shape[0]

    I_n = sp.eye(n, format='csr')

    # initialize the first standard basis vector e_1 as a sparse column vector
    e_1 = sp.csr_matrix(([1], ([0], [0])), shape=(n, 1))

    # compute the projection matrix P
    nominator = (eigenvector + e_1) @ (eigenvector + e_1).transpose()
    denominator = 1 + eigenvector[0, 0]
    P = I_n - nominator / denominator

    # apply the projection to matrix X from both sides to obtain the deflated matrix B
    B = P @ X @ P

    return B, P

def slice_matrix(X: sp.csr_array):
    """
    Slice a sparse matrix by removing the first row and first column (i.e., obtain B[1:, 1:]).

    Parameters
    ----------
    X : scipy.sparse.csr_array
        The input sparse matrix to slice. Must be square and have at least 2 rows and columns.

    Returns
    -------
    X_sliced_csr : scipy.sparse.csr_array
        The sliced sparse matrix in CSR format, excluding the first row and first column.
    """
    # ensure the matrix is square
    if X.shape[0] != X.shape[1]:
        raise ValueError("Input matrix must be square (same number of rows and columns).")
    
    n = X.shape[0]
    
    # ensure the matrix has at least 2 rows and columns to perform B[1:, 1:]
    if n < 2:
        raise ValueError("Input matrix must have at least 2 rows and 2 columns to perform slicing B[1:, 1:].")
    
    # slice rows: remove the first row (index 0)
    X_sliced_rows = X[1:, :]
    
    # convert to CSC for efficient column slicing
    X_sliced_rows_csc = X_sliced_rows.tocsc()
    
    # slice columns: remove the first column (index 0)
    X_sliced_csc = X_sliced_rows_csc[:, 1:]
    
    # convert back to CSR format for consistency
    X_sliced_csr = X_sliced_csc.tocsr()
    
    return X_sliced_csr


def get_smallest_eigenvalues(X: sp.csr_array, M: int):
    """
    Compute the M smallest eigenvalues of a symmetric sparse matrix X using the inverse power method with deflation.

    Parameters
    ----------
    X : scipy.sparse.csr_array
        Symmetric sparse matrix for which to compute the smallest eigenvalues.
        Must be a square matrix.
    M : int
        The number of smallest eigenvalues to compute.

    Returns
    -------
    eigenvalues : list of float
        A list containing the M smallest eigenvalues of matrix X in ascending order.
    """
    print(f"----> Computing {M} smallest eigenvalues of the matrix...")
    # empty list to store the computed eigenvalues
    eigenvalues = []

    # compute the smallest eigenvalue and its corresponding eigenvector using the inverse power method
    eigenvalue, eigenvector = inverse_power_method(X=X)

    # perform a deflation step to remove the influence of the found eigenvalue and eigenvector
    B, P = deflation_step(X=X, eigenvector=eigenvector)

    # slice the deflated matrix to prepare for the next eigenvalue computation
    L_sub = slice_matrix(B)

    # append the first computed eigenvalue to the eigenvalues list
    eigenvalues.append(eigenvalue)

    # Iterate to compute the remaining M-1 smallest eigenvalues
    for i in range(1, M):
        # Adjust the shift for small eigenvalues
        if abs(eigenvalue) < 1e-6:
            shift = 1e-6  # Keep the shift small for near-zero eigenvalues
            print(f"        Eigenvalue {i} is near zero; maintaining shift at {shift:.1e}")
        else:
            shift = eigenvalue + 1e-6  # Regularize shift slightly
        eigenvalue, eigenvector_bar = inverse_power_method(X=L_sub, shift=shift) # compute smallest eigenvalue
        B, P = deflation_step(X=L_sub, eigenvector=eigenvector_bar) # deflation step
        L_sub = slice_matrix(B) # slice the deflated matrix
        eigenvalues.append(eigenvalue)

    print("----> Eigenvalue computation complete.")

    return eigenvalues

