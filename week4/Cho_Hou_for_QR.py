import cmath
import matplotlib.pyplot as plt
import numpy as np

def cholesky(origin_A):
    """
    Compute the cholesky factorication of hermittian positive definitive matrix: A = LL*
    Input: A
    Output: L, the lower triangular part of A
    """
    A = np.copy(origin_A).astype(np.complex128)
    n = A.shape[0]
    for i in range(n):
        A[i, i] = cmath.sqrt(A[i, i])
        A[i:, i] /= A[i, i]
        for j in range(i+1, n):
            A[j:, j] -= A[j:, i] * (A[j, i])
    return np.tril(A)
        
def cholesky_QR(origin_A):
    """
    Use cholesky factorization of (A^*)A=L(L^*) to get R=L* and let Q=AR^-1
    Input: A
    Output: Q, R
    """
    A = np.copy(origin_A).astype(np.complex128)
    R = cholesky(A.conj().T @ A).conj().T
    Q = A @ np.linalg.inv(R)
    return Q, R

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
    n = A.shape[0]
    Q = np.eye(n, dtype=np.complex128)
    R = A.copy()
    for i in range(n-1):
        x = R[i:, i]
        x_col = x.reshape(-1, 1)
        H_i, p = householder_matric(x_col, n)
        R = H_i @ R 
        Q = Q @ H_i
    return Q, R

def rand_matrix_by_kappa(n, kappa):
    diag = np.random.random(n) * 1. /kappa + (1 - 1. /kappa)
    diag.sort()
    diag[0] = 1. / kappa 
    diag[-1] = 1.
    D = np.diag(diag).astype(np.complex128)
    
    A = np.random.randn(n, n) + 1j * np.random.randn(n, n)

    U, S, Vh = np.linalg.svd(A)

    return U @ D @ Vh

n, times = 25, 10
kappa_power = 10
kappas = [10 ** i for i in range(kappa_power)]

average_loss_cho = [0]*kappa_power
average_loss_hou = [0]*kappa_power
for i in range(times):
    loss_cho, loss_hou = [], []
    for kappa in kappas:
        A = rand_matrix_by_kappa(n, kappa)
        Q1, R1 = cholesky_QR(A)
        Q2, R2 = householder_QR(A)
        loss_cho.append(np.linalg.norm(Q1.conj().T @ Q1 - np.eye(n).astype(np.complex128), 'fro'))
        loss_hou.append(np.linalg.norm(Q2.conj().T @ Q2 - np.eye(n).astype(np.complex128), 'fro'))

    for j in range(kappa_power):
        average_loss_cho[j] += loss_cho[j]
        average_loss_hou[j] += loss_hou[j]

average_loss_cho = [x/times for x in average_loss_cho]
average_loss_hou = [x/times for x in average_loss_hou]

plt.figure(figsize=(12,8))
plt.plot(kappas, average_loss_cho, label='Cholesky QR', linestyle='-', color = 'b')
plt.plot(kappas, average_loss_hou, label='Householder QR', linestyle='--', color = 'b')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('kappa')
plt.ylabel('Frobenies norm of |Q*Q-I|')
plt.legend()
plt.grid()
plt.title(f'Orthogonality Loss vs Condition Number for n = {n}')
plt.show()
    



