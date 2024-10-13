import numpy as np
import timeit
import matplotlib.pyplot as plt
from soltri import *


def solve_linear_basic(A, b):
    """
    Solve the linear system Ax=b, without pivoting.
    Must assume every leading principal matrix of A is non-singular.
    A and b will be mutated.
    """
    n = b.shape[0]
    A = np.concatenate((A, b.reshape((n, 1))), axis=1)
    for j in range(n - 1):
        A[j + 1:, j:] -= np.outer(A[j + 1:, j] / A[j, j], A[j, j:])
    return solve_triu_by_row(A[:, :n], A[:, n].reshape((n,)))


def solve_linear_col_pivoting(A, b):
    """
    Solve the linear system Ax=b, with column pivoting.
    A and b will be mutated.
    """
    n = b.shape[0]
    A = np.concatenate((A, b.reshape((n, 1))), axis=1)
    for j in range(n - 1):
        t = np.argmax(np.abs(A[j:, j])) + j
        tmp = A[j, j:].copy()
        A[j, j:] = A[t, j:]
        A[t, j:] = tmp
        A[j + 1:, j:] -= np.outer(A[j + 1:, j] / A[j, j], A[j, j:])
    return solve_triu_by_row(A[:, :n], A[:, n].reshape((n,)))


def solve_linear_full_pivoting(A, b):
    """
    Solve the linear system Ax=b, with full pivoting.
    A and b will be mutated.
    """
    n = b.shape[0]
    A = np.concatenate((A, b.reshape((n, 1))), axis=1)
    p = np.arange(n)
    for j in range(n - 1):
        t1, t2 = np.unravel_index(np.argmax(np.abs(A[j:, j:n]), axis=None), (n - j, n - j))
        t1 += j
        t2 += j
        tmp = A[j, j:].copy()
        A[j, j:] = A[t1, j:]
        A[t1, j:] = tmp
        tmp = A[:, j].copy()
        A[:, j] = A[:, t2]
        A[:, t2] = tmp
        p[t2] = j
        p[j] = t2
        assert A[j, j] != 0
        A[j + 1:, j:] -= np.outer(A[j + 1:, j] / A[j, j], A[j, j:])
    y = solve_triu_by_row(A[:, :n], A[:, n].reshape((n,)))
    for i in range(n):
        tmp = y[i]
        y[i] = y[p[i]]
        y[p[i]] = tmp
        p[p[i]] = p[i]
    return y


if __name__ == "__main__":
    x = np.array([2 ** x for x in range(3, 12)])
    y = np.zeros(x.shape[0])
    z = np.zeros(x.shape[0])
    w = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        A = np.random.random((x[i], x[i])) + 1
        b = np.random.randn(x[i])
        y[i] = timeit.timeit("solve_linear_full_pivoting(A.copy(), b.copy())", number=2, globals=globals())
        z[i] = timeit.timeit("solve_linear_col_pivoting(A.copy(), b.copy())", number=2, globals=globals())
        w[i] = timeit.timeit("np.linalg.solve(A.copy(), b.copy())", number=2, globals=globals())
    fig, ax = plt.subplots()
    ax.loglog(x, y, label="with full pivoting")
    ax.loglog(x, z, label="with partial pivoting")
    ax.loglog(x, w, label="with np.linalg.solve")
    ax.set(xlabel='size of A', ylabel='time (usec)', title='Execution time of solving Ax=b')
    ax.legend()
    ax.grid()
    fig.savefig("hw02pic1.png")
    plt.show()