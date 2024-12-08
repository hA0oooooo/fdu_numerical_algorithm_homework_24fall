import numpy as np


def house(x):
    """
    Householder transformation to let x become a unit vector
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
    Transform a square matrix into Hessenberg form using Householder transformations
    """
    n1, n2 = origin_A.shape
    if n1 != n2:
        raise ValueError("Input matrix must be square")
    A = np.copy(origin_A)
    n = n1
    Q = np.eye(n)
    for k in range(n - 2):
        x = A[k + 1:n, k]
        H = house(x)
        A[k + 1:n, k:n] = H @ A[k + 1:n, k:n]
        A[0:n, k + 1:n] = A[0:n, k + 1:n] @ H
        Q[k + 1:n, :] = H @ Q[k + 1:n, :]
    return A, Q


def simpleshift(H, m, n):
    """
    Compute simple shift using trace and determinant
    """
    a, b, c, d = H[m-1, m-1], H[m-1, n-1], H[n-1, m-1], H[n-1, n-1]
    trace = a + d
    determinant = a * d - b * c
    delta = np.sqrt((trace / 2)**2 - determinant)
    lambda1 = trace / 2 + delta
    lambda2 = trace / 2 - delta
    return lambda1 if abs(lambda1 - d) < abs(lambda2 - d) else lambda2


def wilkinsonshift(H, m, n):
    """
    Compute Wilkinson shift
    """
    a, b, c, d = H[m-1, m-1], H[m-1, n-1], H[n-1, m-1], H[n-1, n-1]
    delta = (a - d) / 2
    return d - b**2 / (delta + np.sign(delta) * np.sqrt(delta**2 + b**2))


def is_converged(H, tol=1e-10):
    """
    Check if the Hessenberg matrix has converged by examining the subdiagonal elements
    """
    n = H.shape[0]
    for i in range(1, n):
        if abs(H[i, i-1]) > tol * (abs(H[i-1, i-1]) + abs(H[i, i])):
            return False
    return True


def francis_qr(H, shift_method="simpleshift", tol=1e-10):
    """
    Perform Francis QR iterations with specified shift method
    """
    n = H.shape[0]
    Q_accum = np.eye(n)  # To accumulate the orthogonal transformations

    while not is_converged(H, tol):
        # Select shift
        m, n = H.shape[0] - 2, H.shape[0] - 1
        if shift_method == "simpleshift":
            mu = simpleshift(H, m, n)
        elif shift_method == "wilkinsonshift":
            mu = wilkinsonshift(H, m, n)
        else:
            raise ValueError("Unknown shift method")

        # Begin implicit QR with shift
        x = H[0, 0] - mu
        y = H[1, 0]
        for k in range(H.shape[0] - 1):
            if k < H.shape[0] - 2:
                z = H[k + 2, k]
                v = np.array([x, y, z])
            else:
                v = np.array([x, y])
            Q = house(v)
            q_start = max(0, k - 1)
            r_end = min(k + 3, H.shape[0])

            # Apply transformation to H
            H[k:k+len(v), q_start:] = Q @ H[k:k+len(v), q_start:]
            H[:r_end, k:k+len(v)] = H[:r_end, k:k+len(v)] @ Q.T

            # Update x and y
            if k < H.shape[0] - 2:
                x, y = H[k + 1, k], H[k + 2, k]
            else:
                x, y = H[k + 1, k], 0.0

            # Accumulate Q
            Q_expanded = np.eye(H.shape[0])
            Q_expanded[k:k+len(v), k:k+len(v)] = Q
            Q_accum = Q_accum @ Q_expanded

        # Zero out small subdiagonal elements
        for i in range(1, H.shape[0]):
            if abs(H[i, i-1]) < tol * (abs(H[i-1, i-1]) + abs(H[i, i])):
                H[i, i-1] = 0.0

    return H, Q_accum


def generate_test_matrix(n):
    """
    Generate a random square matrix
    """
    return np.random.randn(n, n) * 10


# Test
dimension = 5
A = generate_test_matrix(dimension)
H, Q = hessenberg(A)  # Transform to Hessenberg form
T, Q_final = francis_qr(H, shift_method="wilkinsonshift")

# Output results
print("Original Hessenberg Matrix (H):")
print(np.array2string(H, formatter={'float_kind': lambda x: f"{x:.2e}"}))

print("\nSchur Form Matrix (T):")
print(np.array2string(T, formatter={'float_kind': lambda x: f"{x:.2e}"}))

print("\nOrthogonal Transformation Matrix (Q):")
print(np.array2string(Q_final, formatter={'float_kind': lambda x: f"{x:.2e}"}))
