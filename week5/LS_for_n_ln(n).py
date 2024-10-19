import numpy as np
import cmath
import matplotlib.pyplot as plt


def householder_matric(x, n):
    """
    get the householder vector of x
    input: vector x belongs to complex field ^ k, n
    Output: householder matrix H, phase of x[0] p which makes x can be householdered
    """
    # rotate x[0] to make it real, x[0] maybe positive
    length_x = len(x)
    phase_x0 = np.angle(x[0]).item()
    x = cmath.exp(-phase_x0 * 1j) * x

    # Compute the Householder vector
    x[0] += np.linalg.norm(x)
    new_length = np.linalg.norm(x)
    x /= new_length

    # Create the full n x n Householder matrix
    zero_vec = np.zeros((n - length_x, 1), dtype=np.complex128)
    full_x = np.vstack((zero_vec, x))
    H = np.eye(n, dtype=np.complex128) - 2.0 * np.outer(full_x, full_x.conj().T)
    
    return H, phase_x0

def householder_QR(A):
    """ 
    Use householder to get the QR factorization of A
    Input: A
    Output: Q, R from mutated A
    """
    m, n = A.shape
    Q = np.eye(m, dtype=np.complex128)
    R = A.copy()
    for i in range(n):
        x = R[i:, i]
        x_col = x.reshape(-1, 1)
        H_i, p = householder_matric(x_col, m)
        R = H_i @ R 
        Q = Q @ H_i
    return Q, R

number = 6
x_coordinates = np.arange(2, 2 + number).reshape(-1, 1) 
y_coordinates = np.log(x_coordinates).reshape(-1, 1)
ones_column = np.ones((number, 1))
A = np.hstack((x_coordinates, ones_column)) 
m, n = A.shape

# complete QR
Q, R = householder_QR(A)

# incomplete QR
inQ = np.zeros((m, n))
inQ = Q[:,:2]
inR = np.zeros((n, n))
inR = R[:2, :2]

# Ax = b -> QRx = b -> Rx = Q*b
righthand = inQ.conj().T @ y_coordinates
inrighthand = np.zeros((n, n))
inrighthand = righthand[:2, :2]

# solving inR x = inrighthand, 2x2 @ 2x1 = 2x1
a = [0] * n
for i in range(n-1, -1, -1):
    a[i] = inrighthand[i, 0] / inR[i, i]
    inrighthand[:i, 0] = inrighthand[:i, 0] - inR[:i, i] * a[i]

plt.figure(figsize=(12, 8))
x = np.arange(2, 2 + number)
y = np.log(x)
plt.scatter(x, y)
plt.plot(x, a[0] * x + a[1], label=f'y = {a[0].real:.5f}x + {a[1].real:.5f}')
plt.xlabel('n')
plt.ylabel('ln(n)')
plt.legend()
plt.grid()
plt.show()