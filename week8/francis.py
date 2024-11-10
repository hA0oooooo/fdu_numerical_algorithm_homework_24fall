import numpy as np

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

def francis(oringin_H):
    """
    Fransic's double shift QR algorithm for Hessenberg matrix, iterate once
    :param H: a Hessenberg matrix
    :return: H hat, a Hessenberg matrix with the same eigenvalues as H
    """
    H = np.copy(oringin_H)
    n1, n2 = H.shape
    if n1 != n2:
        print('Input error')
        return None
    n = n1
    m = n - 1
    s = H[m-1, m-1] + H[n-1, n-1]
    t = H[m-1, m-1] * H[n-1, n-1] - H[m-1, n-1] * H[n-1, m-1]
    x = H[0, 0] * H[0, 0] + H[0, 1] * H[1, 0] - s * H[0, 0] + t
    y = H[1, 0] * (H[0, 0] + H[1, 1] - s)
    z = H[1, 0] * H[2, 1]

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

    Q = house(np.array([x, y]))
    H[n-2:n, n-3:n] = Q @ H[n-2:n, n-3:n]
    H[0:n, n-2:n] = H[0:n, n-2:n] @ Q
    return H

def generate(n):
    """
    Generate a special hessenberg matrix A
    :param n: the size of the matrix
    :return: a special matrix
    """
    A = np.zeros((n, n))
    A[0, n-1] = 1
    for i in range(1, n):
        A[i, i-1] = 1
    return A

n = 6
A = generate(n)
print(A)
A1 = francis(A)
print(A1)
A2 = francis(A1)
print(A2)