import numpy as np
import matplotlib.pyplot as plt


def house(x):
    """
    householder transformation to let x become a unit vector
    :param x: a vector
    :return: a matrix H such that Hx = ||x||e_1, and H has the corresponding size of x
    """
    n = len(x)
    e1 = np.zeros(n)
    e1[0] = 1
    v = x + np.sign(x[0]) * np.linalg.norm(x) * e1

    v = v / np.linalg.norm(v)
    return np.eye(n) - 2 * np.outer(v, v)


def hessenberg(origin_A):
    """
    Use Householder transformation to transform a matrix to Hessenberg form
    :param A: a square matrix
    :return: Hessenberg matrix A, and the corresponding orthogonal matrix Q, Q A Q.T = Hessenberg matrix
    """
    n1, n2 = origin_A.shape
    if n1 != n2:
        print('Input error')
        return None
    A = np.copy(origin_A)
    n = n1
    Q = np.eye(n)
    for k in range(n-2):
        x = A[k+1:n, k]
        H = house(x)
        A[k+1:n, k:n] = H @ A[k+1:n, k:n]
        A[0:n, k+1:n] = A[0:n, k+1:n] @ H
        Q[k+1:n, :] = H @ Q[k+1:n, :]

    return A, Q


def francis_one(H, Q_accum):
    """
    Fransic's double shift QR algorithm for Hessenberg matrix, iterate once
    :param H: a Hessenberg matrix
    :return: H hat, a Hessenberg matrix with the same eigenvalues as H
    """
    n1, n2 = H.shape
    if n1 != n2:
        print('Input error')
        return None
    n = n1
    m = n - 1
    # double shift
    s = H[m-1, m-1] + H[n-1, n-1]
    t = H[m-1, m-1] * H[n-1, n-1] - H[m-1, n-1] * H[n-1, m-1]
    x = H[0, 0] * H[0, 0] + H[0, 1] * H[1, 0] - s * H[0, 0] + t
    y = H[1, 0] * (H[0, 0] + H[1, 1] - s)
    z = H[1, 0] * H[2, 1]

    # implicit QR
    for k in range(n-2):
        Q = house(np.array([x, y, z]))
        q = max(0, k-1)
        H[k:k+3, q:n] = Q @ H[k:k+3, q:n]
        r = min(k+4, n)
        H[0:r, k:k+3] = H[0:r, k:k+3] @ Q
        x = H[k+1, k]
        y = H[k+2, k]
        if k < n-3:
            z = H[k+3, k]
        # record the transformation matrix
        Q_accum[k:k+3, :] = Q @ Q_accum[k:k+3, :]

    Q = house(np.array([x, y]))
    H[n-2:n, n-3:n] = Q @ H[n-2:n, n-3:n]
    H[0:n, n-2:n] = H[0:n, n-2:n] @ Q
    # record the transformation matrix
    Q_accum[n-2:n, :] = Q @ Q_accum[n-2:n, :]

    return H, Q_accum


def hessenberg_to_schur_form(H, max_iter=np.inf, tol=1e-16):
    """
    turn a Hessenberg matrix to Schur form, from the right-bottom to the left-top
    """

    n = H.shape[0]
    Q_accum = np.eye(H.shape[0])
    if n <= 2:
        return H, Q_accum, 0
    m = n
    iteration_time = 0

    while m > 2 and iteration_time < max_iter:
        
        found_block = False
        for i in range(m-1, -1, -1):
            if abs(H[i, i-1]) < tol * (abs(H[i-1, i-1]) + abs(H[i, i])):
                H[i, i-1] = 0
                m = i
                found_block = True
                break

        if not found_block:
            H[0:m, 0:m], Q_accum = francis_one(H[0:m, 0:m], Q_accum)
        
        iteration_time += 1

    return H, Q_accum, iteration_time


def zeros_hessenberg(H):

    n = H.shape[0]
    for i in range(n-2):
        H[i+2:n, i] = 0

    return H


def generate_test_matrix(n):

    np.random.seed(0)
    A = np.random.randn(n, n) * 10
    return A


# main
dimension = 20
A = generate_test_matrix(dimension)

# Q1 A Q1^T = H
origin_H, Q1 = hessenberg(A)

print(np.linalg.norm(A - Q1.T @ origin_H @ Q1))
print(np.linalg.norm(Q1 @ Q1.T - np.eye(dimension)))

# Q2 H Q2^T = T
H = np.copy(origin_H)
T, Q2, iteration_time = hessenberg_to_schur_form(H)
T = zeros_hessenberg(T)
print(np.linalg.norm(origin_H - Q2.T @ T @ Q2))
print(np.linalg.norm(Q2 @ Q2.T - np.eye(dimension)))

# A = Q1.T H Q1 = Q1.T Q2.T T Q2 Q1
Q = Q1.T @ Q2.T
final_A = Q @ T @ Q.T
print(np.linalg.norm(A - final_A))
print(np.linalg.norm(Q @ Q.T - np.eye(dimension)))

# 检查拟上三角阵和原矩阵的特征值是否相同
eigenvalues = np.linalg.eigvals(A)
print(np.sort(np.linalg.eigvals(T)) - np.sort(np.linalg.eigvals(A)))


# 输出到文本文件中方便观察
with open("output.txt", "w") as f:
    for val in eigenvalues:
        f.write(f"{val:.3e}\n")
    for row in T:
        f.write(" ".join([f"{elem:.3e}" for elem in row]) + "\n")
