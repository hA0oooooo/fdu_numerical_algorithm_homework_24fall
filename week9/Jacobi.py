import numpy as np
import matplotlib.pyplot as plt

def generate_sym_matrix(n):
    """
    genereate a random real symmetric matrix with size n x n
    """
    A = np.random.rand(n, n)
    return A + A.T

def rotate(a, b, c, d):
    """
    matrix rotation for ((a, b),(c, d))
    :return: cos(\theta), sin(\theta), theta is below pi/4
    """
    if a == d:
        return 1/np.sqrt(2), 1/np.sqrt(2)

    alpha = (d-a)/(b)
    t = (-2)/(alpha + np.sign(alpha) * np.sqrt(4 + alpha**2))
    cos0 = 1 / np.sqrt(1 + t**2)
    sin0 = t * cos0

    return cos0, sin0

def wrong_rotate(a, b, c, d):
    """
    matrix rotation for ((a, b),(c, d))
    :return: cos(\theta), sin(\theta), theta is above pi/4
    """
    if a == d:
        return 1/np.sqrt(2), 1/np.sqrt(2)

    alpha = (d-a)/(b)
    t = (-2)/(alpha - np.sign(alpha) * np.sqrt(4+alpha**2))
    cos0 = 1 / np.sqrt(1 + t**2)
    sin0 = t * cos0
    return cos0, sin0

def off(A):
    """
    calculate the off-diagonal sum of A
    """
    n = A.shape[0]
    off_sum = 0
    for i in range(n):
        for j in range(i+1, n):
            off_sum += A[i, j]**2
    return off_sum


def jacobi_pivoting(origin_A, tol=1e-10):   
    """
    Jacobi method with pivoting
    :return: diagonalized matrix D
    :return: orthogonal matrix Q, QAQ* = D
    """
    A = np.copy(origin_A)
    n = A.shape[0]
    Q = np.eye(n)
    max_array = []
    off_sum = [off(A)]

    while True:
        max = 0
        p, q = -1, -1
        for i in range(n):
            for j in range(i):
                if abs(A[i, j]) > max:
                    max = abs(A[i, j])
                    p, q= i, j
        if max < tol:
            break

        cos0, sin0 = rotate(A[p, p], A[p, q], A[q, p], A[q, q])
        rot_matrix = np.eye(n)
        rot_matrix[p, p] = cos0
        rot_matrix[p, q] = sin0
        rot_matrix[q, p] = -sin0
        rot_matrix[q, q] = cos0
        A[p, :], A[q, :] = cos0 * A[p, :] + sin0 * A[q, :], -sin0 * A[p, :] + cos0 * A[q, :]
        A[:, p], A[:, q] = cos0 * A[:, p] + sin0 * A[:, q], -sin0 * A[:, p] + cos0 * A[:, q]            
        # A = rot_matrix @ A @ rot_matrix.T
        Q = rot_matrix @ Q
        max_array.append(max)
        off_sum.append(off(A))

    return A, Q, max_array, off_sum


def jacobi_cyclic(origin_A, tol=1e-10):
    """
    Jacobi method, cyclic version
    """
    A = np.copy(origin_A)
    n = A.shape[0]
    Q = np.eye(n)
    time = 4
    off_sum = [off(A)]

    while off(A) > tol:
        for p in range(n):
            for q in range(p+1, n):
                cos0, sin0 = rotate(A[p, p], A[p, q], A[q, p], A[q, q])
                rot_matrix = np.eye(n)
                rot_matrix[p, p] = cos0
                rot_matrix[p, q] = sin0
                rot_matrix[q, p] = -sin0
                rot_matrix[q, q] = cos0
                A[p, :], A[q, :] = cos0 * A[p, :] + sin0 * A[q, :], -sin0 * A[p, :] + cos0 * A[q, :]
                A[:, p], A[:, q] = cos0 * A[:, p] + sin0 * A[:, q], -sin0 * A[:, p] + cos0 * A[:, q]            
                # A = rot_matrix @ A @ rot_matrix.T
                off_sum.append(off(A))

    return A, Q, off_sum


def jacobi_wrong_cyclic(origin_A, tol=1e-10):
    """
    Jacobi method, wrong cyclic version
    """
    A = np.copy(origin_A)
    n = A.shape[0]
    Q = np.eye(n)
    time = 10
    off_sum = [off(A)]

    for i in range(time):
        for p in range(n):
            for q in range(p+1, n):
                cos0, sin0 = wrong_rotate(A[p, p], A[p, q], A[q, p], A[q, q])
                rot_matrix = np.eye(n)
                rot_matrix[p, p] = cos0
                rot_matrix[p, q] = sin0
                rot_matrix[q, p] = -sin0
                rot_matrix[q, q] = cos0
                A[p, :], A[q, :] = cos0 * A[p, :] + sin0 * A[q, :], -sin0 * A[p, :] + cos0 * A[q, :]
                A[:, p], A[:, q] = cos0 * A[:, p] + sin0 * A[:, q], -sin0 * A[:, p] + cos0 * A[:, q]            
                # A = rot_matrix @ A @ rot_matrix.T
                off_sum.append(off(A))

    return A, Q, off_sum

n = 50
A = generate_sym_matrix(n)
D1, Q1, max_array, off_sum1 = jacobi_pivoting(A)
D2, Q2, off_sum2 = jacobi_cyclic(A)
D3, Q3, off_sum3 = jacobi_wrong_cyclic(A)

plt.figure(figsize=(12, 8))
plt.plot(max_array)
plt.xlabel('iteration')
plt.ylabel('max off-diagonal element')
plt.title(f'max off-diagonal of {n}x{n} matrix, Jacobi method, pivoting version, tol=1e-10') 
plt.grid()
plt.show()


plt.figure(figsize=(12, 8))
plt.plot(off_sum1)
plt.xlabel('iteration')
plt.ylabel('max off-diagonal element')
plt.title(f'max off-diagonal of {n}x{n} matrix, Jacobi method, pivoting version, tol=1e-10') 
plt.grid()
plt.show()

plt.figure(figsize=(12, 8))
plt.plot(off_sum2)
plt.xlabel('iteration')
plt.ylabel('off-diagonal sum')
plt.title(f'off-diagonal sum of {n}x{n} matrix, Jacobi method, cyclic version, tol=1e-10')
plt.grid()
plt.show()

plt.figure(figsize=(12, 8))
plt.plot(off_sum3)
plt.xlabel('iteration')
plt.ylabel('off-diagonal sum')
plt.title(f'off-diagonal sum of {n}x{n} matrix, Jacobi method, wrong cyclic version')
plt.grid()
plt.show()

print(len(off_sum3))