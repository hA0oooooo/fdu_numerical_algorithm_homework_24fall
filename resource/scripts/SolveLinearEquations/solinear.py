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


if __name__ == "__main__":
    x = np.array([2 ** x for x in range(3, 13)])
    y = np.zeros(x.shape[0])
    z = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        print(i)
        A = np.random.random((x[i], x[i])) + 1
        b = np.random.randn(x[i])
        y[i] = timeit.timeit("solve_linear_basic(A.copy(), b.copy())", number=2, globals=globals())
        z[i] = timeit.timeit("solve_linear_col_pivoting(A.copy(), b.copy())", number=2, globals=globals())

    fig, ax = plt.subplots()
    ax.loglog(x, y, label="without pivoting")
    ax.loglog(x, z, label="with column pivoting")
    ax.set(xlabel='size of A', ylabel='time (usec)', title='Execution time of solving Ax=b')
    ax.legend()
    ax.grid()
    fig.savefig("hw01pic1.png")
    plt.show()