import numpy as np


def lu_basic(A):
    """
    Calculate the LU decomposition of a size n square matrix A, without pivoting.
    Must assume every leading principal matrix of A is non-singular. L and U are stored in A.
    """
    n = A.shape[0]
    for j in range(n):
        A[j + 1:, j] /= A[j, j]
        A[j + 1:, j + 1:] -= np.outer(A[j + 1:, j], A[j, j + 1:])
    return A

def lu_partial_pivoting(A):
    """
    Calculate the LU decomposition of a size n square matrix A, with partial pivoting(reorder rows).
    Return a tuple of matrix (A, P), in which L and U are stored in A, and P is the size n permutation matrix.
    """
    # todo