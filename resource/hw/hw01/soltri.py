import numpy as np

def solve_tril_by_col(L, b):
    """
    Solve the linear equation Lx = b in which L is a size n lower triangular matrix
    using column-oriented forward substitution. The solution x is stored in b.
    """
    n = b.shape[0]
    for j in range(n):
        b[j] = b[j] / L[j, j]
        b[j + 1:] = b[j + 1:] - b[j] * L[j + 1:, j]
    return b


def solve_tril_by_row(L, b):
    """
    Solve the linear equation Lx = b in which L is a size n lower triangular matrix
    using row-oriented forward substitution. The solution x is stored in b.
    """
    n = b.shape[0]
    for j in range(n):
        b[j] = (b[j] - np.inner(L[j, :j], b[:j])) / L[j, j]
    return b

def solve_triu_by_row(U, b):
    """
    Solve the linear equation Ux = b in which U is a size n upper triangular matrix
    using row-oriented forward substitution. The solution x is stored in b.
    """
    n = b.shape[0]
    for j in range(n - 1, -1, -1):
        b[j] = (b[j] - np.inner(U[j, j + 1:], b[j + 1:])) / U[j, j]
    return b