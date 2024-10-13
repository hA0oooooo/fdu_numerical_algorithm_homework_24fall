import numpy as np
import math
from numba import jit


@jit(nopython=True)
def Cholesky(A):
    """
    Compute the Cholesky factorization of hermitian positive-definite matrix A.
    Return the upper triangular Cholesky factor of A. Will mutate A.
    """
    n = A.shape[0]
    for j in range(n):
        A[j, j] = math.sqrt(A[j, j].real)
        A[j, j + 1:] /= A[j, j]
        for k in range(j + 1, n):
            A[k, k:] -= A[j, k] * A[j, k:]
    return np.triu(A)


@jit(nopython=True)
def CholeskyQR(A):
    """
    Use Cholesky factorization of A^*A to compute the QR factorization of A.
    :param A: general complex matrix
    :return Q, R: the QR factorization of A
    """
    R = Cholesky(A.conj().T @ A)
    if np.isinf(R).any():
        raise CholeskyQRError("R contains infs.")
    if np.isnan(R).any():
        raise CholeskyQRError("R contains NaNs.")
    Q = A @ np.linalg.inv(R)
    return Q, R


class CholeskyQRError(Exception):
    pass


# @jit(nopython=True)
def house_vec(x):
    """

    :param x: the vector in a matrix
    :return b: the b value in the Householder transformation $I-bvv^*$.
    """
    sq_length = np.dot(x[1:].conj(), x[1:]).real
    if sq_length == 0:
        if x[0].real > 0:
            b = 0
        elif x[0].real < 0:
            x[0] = - x[0]
            b = -2
        else:
            raise Exception("Householder broken: no Householder vector for a zero vector")
        return b
    angle = x[0] / abs(x[0]) if abs(x[0]) != 0 else 1
    length = math.sqrt((sq_length + x[0] * x[0].conjugate()).real)
    if angle.real > 0:
        x[0] = - angle * (sq_length / (length + abs(x[0])))
        b = 2 * (abs(x[0]) ** 2) / (sq_length + abs(x[0]) ** 2)
        x[1:] /= x[0]
        x[0] = length * angle  # part of R
    else:
        x[0] = angle * (abs(x[0]) + length)
        b = 2 * (abs(x[0]) ** 2) / (sq_length + abs(x[0]) ** 2)
        x[1:] /= x[0]
        x[0] = length * (-angle)  # part of R
    return b


def house_left(A, h, b, inplace=False):
    if len(A.shape) == 2:
        if inplace:
            tmp = h.conj() @ A * b
            for i in range(h.shape[0]):
                for j in range(tmp.shape[0]):
                    A[i, j] -= h[i] * tmp[j]
        else:
            A -= b * np.outer(h, h.conj() @ A)
    else:
        if inplace:
            tmp = h.conj() @ A * b
            for i in range(h.shape[0]):
                    A[i] -= h[i] * tmp
        else:
            A -= b * np.outer(h, h.conj() @ A)
    return A


def house_right(A, h, b):
    A -= b * np.outer(A @ h, h.conj())
    return A


def house_tri(A):
    """
    Use Householder reflection to unitary triangular complex matrix A. Will mutate A.
    :param A: general complex matrix
    :return A: R is stored in the upper triangular part of A,
    and the essential part of Householder vectors in the lower part of A.
    """
    n = A.shape[1]
    betas = np.zeros((n,), dtype=A.dtype)
    for i in range(n):
        betas[i] = house_vec(A[i:, i])
        house_left(A[i:, i + 1:], np.concatenate([[1], A[i + 1:, i]]), betas[i], True)
    return A, betas


def house_Q(A, betas):
    """
    Use backward accumulation to explicitly construct Q
    :param A: a complex matrix AFTER house_tri()
    :return Q: the unitary matrix.
    """
    n = A.shape[1]
    Q = np.eye(A.shape[0], dtype=A.dtype)
    for j in range(len(betas) - 1, -1, -1):
        h = np.concatenate([[1], A[j + 1:, j]])
        Q[j:, j:] -= betas[j] * np.outer(h, h.conj() @ Q[j:, j:])
    return Q


def GSQR(A, modified=True, reortho=True):
    n = A.shape[1]
    R = np.zeros((n, n), dtype=A.dtype)
    if not modified:
        for j in range(n):
            R[:j, j] = A[:, :j].T.conj() @ A[:, j]
            A[:, j] -= A[:, :j] @ R[:j, j]
            if reortho:
                r = A[:, :j].T.conj() @ A[:, j]
                A[:, j] -= A[:, :j] @ r
                R[:j, j] += r
            R[j, j] = np.linalg.norm(A[:, j])
            A[:, j] /= R[j, j]
    else:
        for j in range(0, n):
            for i in range(0, j):
                R[i, j] = np.dot(A[:, j].conj(), A[:, i])
                A[:, j] -= R[i, j] * A[:, i]
            if reortho:
                for i in range(0, j):
                    r = np.dot(A[:, j].conj(), A[:, i])
                    A[:, j] -= r * A[:, i]
                    R[i, j] += r
            R[j, j] = np.linalg.norm(A[:, j])
            A[:, j] /= R[j, j]
    return A, R


def rand_matrix_by_kappa(m, n, kappa):
    singulars = np.random.random(n) * (1 - 1. / kappa) + 1. / kappa
    singulars.sort()
    singulars[0] = 1. / kappa
    singulars[-1] = 1.
    D = np.zeros((n, n))
    for i in range(n):
        D[i, i] = singulars[i]
    return np.linalg.qr(np.random.random((m, n)) - 0.5)[0] @ D @ np.linalg.qr(np.random.random((n, n)) - 0.5)[0]


def vis_ortholoss():
    # visualize the loss of orthogonality of CGS, MGS
    sns.set_theme(palette='YlGn')
    m = 200
    n = 50
    A = rand_matrix_by_kappa(m, n, 10 ** 10)
    ortho_loss = []
    fro_norms = []
    for reortho in [False, True]:
        for modified in [False, True]:
            Q, _ = GSQR(A.copy(), modified, reortho)
            ls = np.abs(Q.T.conj() @ Q - np.eye(n, dtype=A.dtype))
            ortho_loss.append(ls)
            fro_norms.append(np.linalg.norm(ls))

    titles = ["CGS", "MGS", "CGS(2)", "MGS(2)"]
    f, axes = plt.subplots(2, 2, figsize=(9, 9), sharex=True, sharey=True)
    cbar_ax = f.add_axes([.85, .045, .03, .8])
    i = 0
    f.suptitle("Orthogonal Loss of Gram-Schmidt QR (size = $200 \cdot 50$, $\kappa = 10^{10} $)", y=0.95)
    for ax, s in zip(axes.flat, np.linspace(0, 2, 5)):
        fig = sns.heatmap(ortho_loss[i], vmin=0, vmax=1e-15, cmap="YlGnBu",
                          cbar=(i == 0), ax=ax, cbar_ax = None if i else cbar_ax)
        fig.set(title=titles[i])
        ax.text(10, 54, f"$||Q^*Q-I||_F=${fro_norms[i]:.2e}")
        i += 1
        ax.set_axis_off()
    f.subplots_adjust(.05, .05, .80, .85, .13, .23)
    plt.savefig("hw05pic1.png")
    plt.show()


def vis_loss_and_res():
    sns.set_theme()
    norm = np.linalg.norm
    data = pd.DataFrame(columns=['$\kappa(A)$', 'algorithms', 'loss of orthogonality', 'residual'])
    kappas = [10 ** x for x in range(16)]
    m , n = 300, 30
    for k in kappas:
        A = rand_matrix_by_kappa(m, n, k)
        # Householder
        hA, betas = house_tri(A.copy())
        R = np.triu(hA)
        Q = house_Q(hA, betas)
        data.loc[len(data.index)] = [k, "Householder", norm(Q.T.conj() @ Q - np.eye(m)), norm(A-Q @ R)]
        # Cholesky
        try:
            Q, R = CholeskyQR(A.copy())
        except CholeskyQRError:
            data.loc[len(data.index)] = [k, "Cholesky", None, None]
        else:
            data.loc[len(data.index)] = [k, "Cholesky", norm(Q.T.conj() @ Q - np.eye(n)), norm(A - Q @ R)]
        # CGS and MGS
        Q, R = GSQR(A.copy(), False, False)
        data.loc[len(data.index)] = [k, "CGS", norm(Q.T.conj() @ Q - np.eye(n)), norm(A - Q @ R)]
        Q, R = GSQR(A.copy(), True, False)
        data.loc[len(data.index)] = [k, "MGS", norm(Q.T.conj() @ Q - np.eye(n)), norm(A - Q @ R)]
        Q, R = GSQR(A.copy(), False, True)
        data.loc[len(data.index)] = [k, "CGS(2)", norm(Q.T.conj() @ Q - np.eye(n)), norm(A - Q @ R)]
        Q, R = GSQR(A.copy(), True, True)
        data.loc[len(data.index)] = [k, "MGS(2)", norm(Q.T.conj() @ Q - np.eye(n)), norm(A - Q @ R)]

    fig, ax = plt.subplots(nrows=2, figsize=[6.4, 8])
    sns.scatterplot(data, x='$\kappa(A)$', y='loss of orthogonality', legend=False,
                hue='algorithms', style="algorithms", ax=ax[0]).set(xscale="log", yscale="log")
    sns.lineplot(data, x='$\kappa(A)$', y='loss of orthogonality', legend='brief',
                hue='algorithms', style="algorithms", ax=ax[0]).set(xscale="log", yscale="log")
    sns.scatterplot(data, x='$\kappa(A)$', y='residual', legend=False,
                    hue='algorithms', style="algorithms", ax=ax[1]).set(xscale="log", yscale="log")
    sns.lineplot(data, x='$\kappa(A)$', y='residual', legend=False,
                 hue='algorithms', style="algorithms", ax=ax[1]).set(xscale="log", yscale="log")
    fig.savefig("hw05pic2.png")
    plt.show()


if __name__ == "__main__":
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    vis_ortholoss()
    vis_loss_and_res()

