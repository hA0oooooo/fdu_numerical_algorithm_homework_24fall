import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse
import pandas as pd
from scipy.linalg import eigvalsh_tridiagonal

norm = np.linalg.norm


def lanczos(A, m, q0=None, eps=1e-8):
    n = A.shape[0]
    if q0 is None:
        q0 = np.random.random(n)
    q0 /= norm(q0)
    V = np.empty((n, m + 1), dtype=A.dtype)
    V[:, 0] = q0
    mdiag = np.empty(m, dtype=A.dtype)
    sdiag = np.empty(m, dtype=A.dtype)

    sdiag[-1] = 0
    for j in range(m):
        V[:, j + 1] = A @ V[:, j] - sdiag[j - 1] * V[:, j - 1]
        mdiag[j] = np.inner(V[:, j + 1], V[:, j])
        V[:, j + 1] -= mdiag[j] * V[:, j]

        sdiag[j] = norm(V[:, j + 1])
        if sdiag[j] < eps:
            return V[:, :j + 1], mdiag[:j + 1], sdiag[:j]
        V[:, j + 1] /= sdiag[j]
    return V, mdiag, sdiag


def vis_ortho(V):
    n = V.shape[0]
    m = V.shape[1]
    data = pd.DataFrame(columns=["first __ Lanczos vectors", "ortholoss"])
    for i in range(2, m):
        v = V[:, :i]
        data.loc[len(data.index)] = [i, norm(v.T @ v - np.eye(i))]
    data.plot(title="orthogonality of Lanczos vectors", x="first __ Lanczos vectors", y="ortholoss")
    plt.savefig("hw13pic3.jpg")
    plt.show()


def vis_ritz(A):
    mls = list(range(1, 13))
    # mls = [10, 20, 30]
    data = pd.DataFrame(columns=["m", "ritz"])
    for m in mls:
        V, md, sd = lanczos(A_sparse, m)
        try:
            eigs = eigvalsh_tridiagonal(md, sd[:m - 1])
        except:
            continue
        for i in range(m):
            data.loc[len(data.index)] = [m, eigs[i]]
    sns.scatterplot(data, x="m", y="ritz",
                    sizes=3, alpha=.5, palette="muted")
    plt.title(f"convergence of ritz pairs, n={A.shape[0]}")
    plt.savefig("hw13pic4.jpg")
    plt.show()


if __name__ == "__main__":
    np.set_printoptions(suppress=False, precision=3)
    sns.set()

    n = 4000
    N = n
    shape = (n, n)
    m = 50

    # Create random sparse (n, n) matrix with N non-zero entries
    coords = np.random.choice(n * n, size=N, replace=False)
    coords = np.unravel_index(coords, shape)
    values = np.random.random(size=N)
    A_sparse = scipy.sparse.coo_matrix((values, coords), shape=shape)
    A_sparse = A_sparse.tocsr()
    A_sparse =A_sparse + A_sparse.T

    V, md, sd = lanczos(A_sparse, m)
    vis_ortho(V)
    vis_ritz(A_sparse)
