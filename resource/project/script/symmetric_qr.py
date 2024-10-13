"""
The symmetric QR algorithm for computing all eigenvalues and eigenvectors of a real symmetric matrix.
"""
import numpy as np
from numba import jit
from sys import float_info



@jit(nopython=True)
def householder_vector(x):
    lens = np.dot(x[1:], x[1:])
    if lens == 0:
        return None, False
    v = x.copy()
    sign = 1 if x[0] >= 0 else -1
    v[0] = x[0] + sign * np.sqrt(x[0] ** 2 + lens)
    v /= np.linalg.norm(v)
    return v, True


def left_house(A, v):
    A -= 2 * np.outer(v, v @ A)
    return


def right_house(A, v):
    A -= 2 * np.outer(A @ v, v)
    return


def tri_diagonal(A, save_q=True):
    n = A.shape[0]
    if save_q:
        Q = np.eye(n)
        houses = []
        for i in range(n - 2):
            v, valid = householder_vector(A[i + 1:, i])
            if valid:
                houses.append(v)
                left_house(A[i + 1:, i:], v)
                right_house(A[i:, i + 1:], v)
        # use backward accumulation to reconstruct Q, faster and more stable.
        for j in range(len(houses) - 1, -1, -1):
            v = houses[j]
            i = n - 1 - v.shape[0]
            left_house(Q[i + 1:, i:], v)
        diag, sdiag = np.array(np.diag(A)), np.array(np.diag(A, k=-1))
        return diag, sdiag, Q
    else:
        for i in range(n - 2):
            v, valid = householder_vector(A[i + 1:, i])
            if valid:
                left_house(A[i + 1:, i:], v)
                right_house(A[i:, i + 1:], v)
        diag, sdiag = np.array(np.diag(A)), np.array(np.diag(A, k=-1))
        return diag, sdiag, None


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


@jit(nopython=True)
def wilkinson_shift(a, b, c):
    """
    Compute the value of Wilkinson shift, which is the eigenvalue of
    T[n-2:, n-2:] = | a    b |
                    | b    c |
    that closer to
    """
    d = (a - c) / 2
    sign = 1 if d >= 0 else -1
    return c - b * b / (d + sign * np.sqrt(d * d + b * b))


@jit(nopython=True)
def normal_shift(a, b, c):
    return c


@jit(nopython=True)
def deflation(diag, sdiag, p, q, eps):
    """
    # :param A: the whole n * n tridiagonal matrix
    :param p: start
    :param q: end
    :param eps:
    :return: p', q': new start and end
    """
    for i in range(p, q - 1):
        if abs(sdiag[i]) < eps * (abs(diag[i]) + abs(diag[i + 1])):
            sdiag[i] = 0.

    for i in range(q - 2, -1, -1):
        if sdiag[i] != 0.:
            q = i + 2
            break
    else:
        return 0, 0, sdiag
    for i in range(q - 2, -1, -1):
        if sdiag[i] == 0.:
            p = i + 1
            break
    else:
        p = 0
    return p, q, sdiag


@jit(nopython=True)
def apply_givens_to_2x2(c, s, d0, s0, d1):
    return c * c * d0 - 2 * c * s * s0 + s * s * d1, \
           s * c * (d0 - d1) + s0 * (c * c - s * s), c * c * d1 + 2 * c * s * s0 + s * s * d0


def implicit_shift(diag: np.array, sdiag: np.array, Q, p, q, shift=normal_shift, need_q=True):
    if need_q:
        mu = shift(diag[q - 2], sdiag[q - 2], diag[q - 1])
        c, s = givens_rotation((diag[p] - mu, sdiag[p]))
        Q[:, p:p + 2] = Q[:, p:p + 2] @ np.array([[c, -s], [s, c]]).T
        diag[p], sdiag[p], diag[p + 1] = apply_givens_to_2x2(c, s, diag[p], sdiag[p], diag[p + 1])
        if q - p != 2:
            bulge, sdiag[p + 1] = -s * sdiag[p + 1], c * sdiag[p + 1]
            v = (sdiag[p], bulge)
        for i in range(p + 1, q - 1):
            c, s = givens_rotation(v)
            diag[i], sdiag[i], diag[i + 1] = apply_givens_to_2x2(c, s, diag[i], sdiag[i], diag[i + 1])
            sdiag[i - 1] = c * sdiag[i - 1] - s * bulge
            Q[:, i:i + 2] = Q[:, i:i + 2] @ np.array([[c, -s], [s, c]]).T
            if i != q - 2:
                bulge, sdiag[i + 1] = -s * sdiag[i + 1], c * sdiag[i + 1]
                v = (sdiag[i], bulge)
    else:
        mu = shift(diag[q - 2], sdiag[q - 2], diag[q - 1])
        c, s = givens_rotation((diag[p] - mu, sdiag[p]))
        diag[p], sdiag[p], diag[p + 1] = apply_givens_to_2x2(c, s, diag[p], sdiag[p], diag[p + 1])
        if q - p != 2:
            bulge, sdiag[p + 1] = -s * sdiag[p + 1], c * sdiag[p + 1]
            v = (sdiag[p], bulge)
        for i in range(p + 1, q - 1):
            c, s = givens_rotation(v)
            diag[i], sdiag[i], diag[i + 1] = apply_givens_to_2x2(c, s, diag[i], sdiag[i], diag[i + 1])
            sdiag[i - 1] = c * sdiag[i - 1] - s * bulge
            if i != q - 2:
                bulge, sdiag[i + 1] = -s * sdiag[i + 1], c * sdiag[i + 1]
                v = (sdiag[i], bulge)
    return diag, sdiag


def symmetric_qr(A, max_iter=None, copy=True, record=False, need_q=True):
    n = A.shape[0]
    p, q = 0, n
    if copy: A = A.copy()
    diag, sdiag, Q = tri_diagonal(A, save_q=need_q)
    eps = float_info.epsilon
    if record:
        starts, ends = np.empty(max_iter), np.empty(max_iter)
    if max_iter is None:
        max_iter = max(n * 5, 10000)
    for i in range(max_iter):
        if q - p <= 1:
            print(f"Converged after {i + 1} iter")
            break
        diag, sdiag = implicit_shift(diag, sdiag, Q, p, q, shift=wilkinson_shift, need_q=need_q)

        p, q, sdiag = deflation(diag, sdiag, p, q, eps)

        if record:
            starts[i] = p
            ends[i] = q
    else:
        print(f"Warning: max_iter {max_iter} exceeded but failed to converge")
    if not record:
        if need_q:
            return diag, Q, i + 1
        else:
            return diag, i + 1
    else:
        if need_q:
            return diag, Q, i + 1, starts[:i + 1], ends[:i + 1]
        else:
            return diag, i + 1, starts[:i + 1], ends[:i + 1]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    np.set_printoptions(precision=3)
    n = 500

    A = np.random.random((n, n))
    A = (A + A.T) / 0.1

    diag, Q, iters, starts, ends = symmetric_qr(A, max_iter=30000, record=True)
    D = np.diag(diag)
    print(f"matrix size: {n} * {n}")
    print(D)
    print(f"ortholoss of Q: {np.linalg.norm(Q.T @ Q - np.eye(n)):.3e}")
    print(f"error of D: {np.linalg.norm(D - Q.T @ A @ Q):.3e}")
    fig, ax = plt.subplots()
    ax.plot(range(iters ), starts, label="starts")
    ax.plot(range(iters ), ends, label="ends")
    ax.set(title=f"n = {n}, iters = {iters }, 1 eigen for every {iters / n:.2f} steps", xlabel="iters",
           ylabel="index")
    plt.legend()
    plt.savefig(f"fig{n}.jpg")
    plt.show()
