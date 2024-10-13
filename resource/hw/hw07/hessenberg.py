import numpy as np
from qrdecomp import house_vec, house_left, house_right, house_Q


def house_hsb(A):
    n = A.shape[0]
    betas = np.zeros((n - 2,), dtype=A.dtype)
    for i in range(n - 2):
        betas[i] = house_vec(A[i + 1:, i])
        house_left(A[i + 1:, i + 1:], np.concatenate([[1], A[i + 2:, i]]), betas[i], True)
        house_right(A[:, i + 1:], np.concatenate([[1], A[i + 2:, i]]), betas[i])
    return A, betas


def arnoldi_hsb(A):
    n = A.shape[0]
    Q = np.zeros((n, n), dtype=float)
    H = np.zeros((n, n), dtype=float)
    Q[:, 0] = np.random.randn(n)
    Q[:, 0] /= np.linalg.norm(Q[:, 0])
    for i in range(n - 1):
        H[:i + 1, i] = Q[:, :i + 1].T.conj() @ A @ Q[:, i]
        Q[:, i + 1] = A @ Q[:, i] - Q[:, :i + 1] @ H[:i + 1, i]
        H[i + 1, i] = np.linalg.norm(Q[:, i + 1])
        Q[:, i + 1] /= H[i + 1, i]
    H[:, n - 1] = Q[:, :].T.conj() @ A @ Q[:, n - 1]
    return H, Q




if __name__ == "__main__":
    n = 100
    A = np.random.random((n, n))
    raw_A, betas = house_hsb(A.copy())
    smallQ = house_Q(raw_A[1:, :n - 1], betas)
    Q = np.zeros((n, n))
    Q[0, 0] = 1
    Q[1:, 1:] = smallQ
    H = np.triu(raw_A)
    for i in range(n - 1):
        H[i + 1, i] = A[i + 1, i]
    print(f"size of A: {n} * {n}.")
    print("Upper Hessenberg via Householder: ")
    print(f"norm of difference between H and Q^TAQ: {np.linalg.norm(Q.T @ A @ Q - H):.3e}")
    print(f"orthogonal loss of Q: {np.linalg.norm(Q.T @ Q - np.eye(n)):.3e}")
    H, Q = arnoldi_hsb(A.copy())
    print("Upper Hessenberg via Arnoldi process: ")
    print(f"norm of difference between H and Q^TAQ: {np.linalg.norm(Q.T @ A @ Q - H):.3e}")
    print(f"orthogonal loss of Q: {np.linalg.norm(Q.T @ Q - np.eye(n)):.3e}")