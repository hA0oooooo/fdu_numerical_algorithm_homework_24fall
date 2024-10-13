import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import time
from numba import jit
from scipy.linalg import solve_triangular, solve
from scipy.sparse.linalg import gmres as scipy_gmres
import scipy.sparse

norm = np.linalg.norm


@jit(nopython=True)
def givens_rotation(x):
    """
    :param x: a 2d array-like obj
    :return: an 2 * 2 matrix G that Gx_0 = ||x||, Gx_1 = 0.
    """
    if x[1] == 0:
        return 1., 0.
    if abs(x[1]) > abs(x[0]):
        t = - x[0] / x[1]
        s = 1 / np.sqrt(1 + t ** 2)
        c = s * t
    else:
        t = - x[1] / x[0]
        c = 1 / np.sqrt(1 + t ** 2)
        s = c * t
    return c, s


def gmres_restart(*args, **kwargs):
    try:
        restart = kwargs.pop("restart") + 1
    except KeyError:
        restart = 1

    num_starts = 0
    while num_starts < restart:

        x, exit_code, data = gmres_iter(*args, **kwargs)
        if exit_code:
            return x, data
        num_starts += 1
        print(f"restart: {num_starts}")
        kwargs["x0"] = x
    print(f"warning: max restarts exceeded, still not converged")
    return x, data


def gmres_iter(A, b, max_m=30, x0=None, m_interval=10, eps=1e-5, record_residuals=False):
    n = A.shape[0]
    if x0 is None:
        r = b
        x0 = 0
    else:
        r = b - A @ x0
    beta = norm(r)
    V = np.empty((n, max_m + 1))
    H = np.zeros((max_m + 1, max_m))
    givens_rotations = {}

    if record_residuals:
        data = pd.DataFrame(columns=["m", "residual"])
    else:
        data = None

    def lstsq_helper(H, beta):
        Htri = H.copy()
        m = H.shape[1]
        b = np.empty(m + 1)
        b[0] = beta
        for i in range(m):
            try:
                c, s = givens_rotations[i]
            except:
                c, s = givens_rotation((Htri[i, i], Htri[i + 1, i]))
                givens_rotations[i] = (c, s)
            Htri[i:i + 2, :] = np.array([[c, -s], [s, c]]) @ Htri[i:i + 2, :]
            b[i], b[i + 1] = c * b[i], s * b[i]
        y = solve_triangular(Htri[:m, :], b[:m], check_finite=True, overwrite_b=True)
        rv = H @ y
        rv[0] -= beta
        return y, norm(rv)

    V[:, 0] = r
    try:
        V[:, 0] /= beta
    except ZeroDivisionError:
        print("GMRES: x0 is the exact solution. ")
        return x0
    residual = beta
    m = 0
    while residual > eps and m < max_m:
        prev_m = m
        m += min(m_interval, max_m - m)
        for j in range(prev_m, m):
            V[:, j + 1] = A @ V[:, j]
            for i in range(j + 1):
                H[i, j] = np.inner(V[:, i].conj(), V[:, j + 1])
                V[:, j + 1] -= H[i, j] * V[:, i]
            H[j + 1][j] = norm(V[:, j + 1])

            if H[j + 1, j] < beta * eps:
                print(f"GMRES: lucky breakdown at index {j} of m: {m}, beta * eps: {beta * eps}")
                H = H[:j + 1, :j + 1]
                V = V[:, :j + 1]
                b = np.zeros(j + 1)
                b[0] = beta
                return x0 + V @ solve(H, b, overwrite_b=True), True, data

            V[:, j + 1] /= H[j + 1][j]
        y, residual = lstsq_helper(H[:m + 1, :m], beta)
        if record_residuals:
            data.loc[len(data.index)] = [m, residual]
        # print(f"m : {m}, residual: {residual}")
        # print(f"ortholoss if V: {norm(V[:, :m].T @ V[:, :m] - np.eye(m))}")
    exit_code = residual < eps
    return x0 + V[:, :m] @ y, exit_code, data


if __name__ == '__main__':
    np.set_printoptions(suppress=False, precision=3)
    sns.set()

    n = 4000
    N = n
    shape = (n, n)
    tol = 1e-6
    m_interval = 5

    # Create random sparse (n, n) matrix with N non-zero entries
    coords = np.random.choice(n * n, size=N, replace=False)
    coords = np.unravel_index(coords, shape)
    values = np.random.normal(size=N)
    A_sparse = scipy.sparse.coo_matrix((values, coords), shape=shape)
    A_sparse = A_sparse.tocsr()
    A_sparse += scipy.sparse.eye(n)

    b = np.random.normal(size=n)
    b = A_sparse @ b

    print("===GMRES implementation===")
    start = time.perf_counter()
    x, residuals = gmres_restart(A_sparse, b, max_m=55, m_interval=m_interval, restart=0, eps=tol, record_residuals=True)
    end = time.perf_counter()
    print(f"res: {norm(b - A_sparse @ x)}")
    print(f"time: {end - start}")

    # visualize the residuals
    residuals.plot(x="m", y="residual", logy=True, legend=False, ylabel="2-norm of the residual vector",
                  title=f"GMRES, n={n}, tol={tol}, compute residual for every {m_interval} steps,no restarts")
    plt.savefig("prob02.jpg")
    plt.show()


    print("===scipy GMRES===")
    start = time.perf_counter()
    xs, exit_code = scipy_gmres(A_sparse, b, maxiter=1000, tol=tol)
    end = time.perf_counter()
    print(f"res: {norm(b - A_sparse @ xs)}")
    print(f"time: {end - start}")
    print(f"exit_code: {exit_code}")
