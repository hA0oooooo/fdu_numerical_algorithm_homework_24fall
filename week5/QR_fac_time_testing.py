import numpy as np
import matplotlib.pyplot as plt
import cmath

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

def CGS_QR(origin_A):
    """
    get the QR factorization of A through Classic Gram Schmidt without reorthogonalization
    Input: A
    Output: Q belongs to C m x n, R belongs to C n x n, Q*Q = I, R is upper triangular matrix
    """
    A = np.copy(origin_A)
    m = A.shape[0]
    n = A.shape[1]
    Q, R = np.zeros((m, n), dtype=np.complex128), np.zeros((n, n), dtype=np.complex128)

    for i in range(n):
        Q[:, i] = A[:, i]
        for j in range(i):
            R[j, i] = np.dot(Q[:, j].conj(), Q[:, i])
        # get all the projection length and then correct Q[:, i] 
        for j in range(i):
            Q[:, i] = Q[:, i] - R[j, i] * Q[:, j]
        
        # normalization
        R[i, i] = np.linalg.norm(Q[:, i], ord = 2)
        Q[:, i] = Q[:, i] / R[i, i]
    return Q, R

def MGS_QR(origin_A):
    """
    get the QR factorization of A through Modified Gram Schmidt without reorthogonalization
    Input: A, wont mutate A
    Output: Q belongs to C m x n, R belongs to C n x n, Q*Q = I, R is upper triangular matrix
    """
    A = np.copy(origin_A)
    m = A.shape[0]
    n = A.shape[1]
    Q, R = np.zeros((m, n), dtype=np.complex128), np.zeros((n, n), dtype=np.complex128)

    for i in range(n):
        Q[:, i] = A[:, i]
        for j in range(i):
            R[j, i] = np.dot(Q[:, j].conj(), Q[:, i])
        # get one projection length and correct Q[:, i] immediately
            Q[:, i] = Q[:, i] - R[j, i] * Q[:, j]
        
        # normalization
        R[i, i] = np.linalg.norm(Q[:, i], ord = 2)
        Q[:, i] = Q[:, i] / R[i, i]

    return Q, R

def CGS_reortho_QR(origin_A):
    """
    get the QR factorization of A through Classic Gram Schmidt with reorthogonalization
    Input: A, wont mutate A
    Output: Q belongs to C m x n, R belongs to C n x n, Q*Q = I, R is upper triangular matrix
    """
    A = np.copy(origin_A)
    m = A.shape[0]
    n = A.shape[1]
    Q, R = np.zeros((m, n), dtype=np.complex128), np.zeros((n, n), dtype=np.complex128)
    correction = np.zeros((n, n), dtype=np.complex128)

    for i in range(n):
        Q[:, i] = A[:, i]

        # first reorthogonalization
        for j in range(i):
            R[j, i] = np.dot(Q[:, j].conj(), Q[:, i])
        # get all the projection length and then correct Q[:, i] 
        for j in range(i):
            Q[:, i] = Q[:, i] - R[j, i] * Q[:, j]

        # second reorthohonalization
        for j in range(i):
            correction[j, i] = np.dot(Q[:, j].conj(), Q[:, i])
        for j in range(i):
            R[j, i] = R[j, i] + correction[j, i]
            Q[:, i] = Q[:, i] - correction[j, i] * Q[:, j]
        
        # normalization
        R[i, i] = np.linalg.norm(Q[:, i], ord = 2)
        Q[:, i] = Q[:, i] / R[i, i]

    return Q, R

def MGS_reortho_QR(origin_A):
    """
    get the QR factorization of A through Modified Gram Schmidt with reorthogonalization
    Input: A, wont mutate A
    Output: Q belongs to C m x n, R belongs to C n x n, Q*Q = I, R is upper triangular matrix
    """
    A = np.copy(origin_A)
    m = A.shape[0]
    n = A.shape[1]
    Q, R = np.zeros((m, n), dtype=np.complex128), np.zeros((n, n), dtype=np.complex128)
    correction = 0 + 0j

    for i in range(n):
        Q[:, i] = A[:, i]

        # first reorthogonalization
        for j in range(i):
            R[j, i] = np.dot(Q[:, j].conj(), Q[:, i])
        # get one projection length and correct Q[:, i] immediately
            Q[:, i] = Q[:, i] - R[j, i] * Q[:, j]

        # second reorthogonalization
        for j in range(i):
            correction = np.dot(Q[:, j].conj(), Q[:, i])
            R[j, i] = R[j, i] + correction
            Q[:, i] = Q[:, i] - correction * Q[:, j]
        
        # normalization
        R[i, i] = np.linalg.norm(Q[:, i], ord = 2)
        Q[:, i] = Q[:, i] / R[i, i]
    return Q, R

def rand_matrix_by_kappa(m, n, kappa):
    diag = np.random.random(n) * 1. /kappa + (1 - 1. /kappa)
    diag.sort()
    diag[0] = 1. / kappa 
    diag[-1] = 1.
    D = np.diag(diag).astype(np.complex128)
    D_final = np.zeros((m, n)).astype(np.complex128)
    D_final[:n, :n] = D

    A = np.random.randn(m, n) + 1j *np.random.randn(m, n)

    U, S, Vh = np.linalg.svd(A)

    return U @ D_final @ Vh


# generate a few tall-skinny matrices
m, n = 500, 50
if m < n:
    print("warning: m should be above n !")
    exit()

kappa_power = 15
kappas = [10 ** i for i in range(1, kappa_power)]

residual_Cho = []
residual_Hou = []
residual_CGS = []
residual_MGS = []
residual_CGS_reortho = []
residual_MGS_reortho = []

loss_Hou = []
loss_Cho = []
loss_CGS = []
loss_MGS = []
loss_CGS_reortho = []
loss_MGS_reortho = []

for kappa in kappas:
    A = rand_matrix_by_kappa(m, n, kappa)
    Q1, R1 = CGS_QR(A)
    Q2, R2 = MGS_QR(A)
    Q3, R3 = CGS_reortho_QR(A)
    Q4, R4 = MGS_reortho_QR(A)
    Q5, R5 = cholesky_QR(A)
    Q6, R6 = householder_QR(A)

    loss_CGS.append(np.linalg.norm(Q1.conj().T @ Q1 - np.eye(n).astype(np.complex128), 'fro'))
    loss_MGS.append(np.linalg.norm(Q2.conj().T @ Q2 - np.eye(n).astype(np.complex128), 'fro'))
    loss_CGS_reortho.append(np.linalg.norm(Q3.conj().T @ Q3 - np.eye(n).astype(np.complex128), 'fro'))
    loss_MGS_reortho.append(np.linalg.norm(Q4.conj().T @ Q4 - np.eye(n).astype(np.complex128), 'fro'))
    loss_Cho.append(np.linalg.norm(Q5.conj().T @ Q5 - np.eye(n).astype(np.complex128), 'fro'))
    loss_Hou.append(np.linalg.norm(Q6.conj().T @ Q6 - np.eye(m).astype(np.complex128), 'fro'))

    residual_CGS.append(np.linalg.norm(Q1 @ R1 -A, 'fro'))
    residual_MGS.append(np.linalg.norm(Q2 @ R2 - A, 'fro'))
    residual_CGS_reortho.append(np.linalg.norm(Q3 @ R3 - A, 'fro'))
    residual_MGS_reortho.append(np.linalg.norm(Q4 @ R4 - A, 'fro'))
    residual_Cho.append(np.linalg.norm(Q5 @ R5 - A, 'fro'))
    residual_Hou.append(np.linalg.norm(Q6 @ R6 - A, 'fro'))

# the orthogonality loss for QR
plt.figure(figsize=(12,8))
plt.plot(kappas, loss_CGS, label='CGS', linestyle='-', color = 'b')
plt.plot(kappas, loss_MGS, label='MGS', linestyle='-', color = 'g')
plt.plot(kappas, loss_CGS_reortho, label='CGS2', linestyle='--', color = 'b')
plt.plot(kappas, loss_MGS_reortho, label='MGS2', linestyle='--', color = 'g')
plt.plot(kappas, loss_Cho, label='Cholesky_QR', linestyle='-', color = 'r')
plt.plot(kappas, loss_Hou, label='Householder_QR', linestyle='-', color = 'y')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('kappa')
plt.ylabel('Frobenies norm of |Q*Q-I|')
plt.legend()
plt.grid()
plt.title(f'Orthogonality Loss for QR of tall-skinny {m}x{n} matrices')
plt.show(block = False)

# the residual norm for QRs
plt.figure(figsize=(12,8))
plt.plot(kappas, residual_CGS, label='CGS', linestyle='-', color = 'b')
plt.plot(kappas, residual_MGS, label='MGS', linestyle='-', color = 'g')
plt.plot(kappas, residual_CGS_reortho, label='CGS2', linestyle='--', color = 'b')
plt.plot(kappas, residual_MGS_reortho, label='MGS2', linestyle='--', color = 'g')
plt.plot(kappas, residual_Cho, label='Cholesky_QR', linestyle='-', color = 'r')
plt.plot(kappas, residual_Hou, label='Householder_QR', linestyle='-', color = 'y')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('kappa')
plt.ylabel('the residual norm of |QR-A|')
plt.legend()
plt.grid()
plt.title(f'the residual norm for QR of tall-skinny {m}x{n} matrices')
plt.show()


