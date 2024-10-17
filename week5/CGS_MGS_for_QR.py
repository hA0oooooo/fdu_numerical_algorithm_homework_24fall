import numpy as np
import matplotlib.pyplot as plt

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

m, n = 5, 3
if m < n:
    print("warning: m should be above n !")
    exit()

kappa_power = 17
kappas = [10 ** i for i in range(1, kappa_power)]

loss_CGS = []
loss_MGS = []
loss_CGS_reortho = []
loss_MGS_reortho = []
# loss_qr = []

for kappa in kappas:
    A = rand_matrix_by_kappa(m, n, kappa)
    Q1, R1 = CGS_QR(A)
    Q2, R2 = MGS_QR(A)
    Q3, R3 = CGS_reortho_QR(A)
    Q4, R4 = MGS_reortho_QR(A)
    # Q, R = np.linalg.qr(A)

    loss_CGS.append(np.linalg.norm(Q1.conj().T @ Q1 - np.eye(n).astype(np.complex128), 'fro'))
    loss_MGS.append(np.linalg.norm(Q2.conj().T @ Q2 - np.eye(n).astype(np.complex128), 'fro'))
    loss_CGS_reortho.append(np.linalg.norm(Q3.conj().T @ Q3 - np.eye(n).astype(np.complex128), 'fro'))
    loss_MGS_reortho.append(np.linalg.norm(Q4.conj().T @ Q4 - np.eye(n).astype(np.complex128), 'fro'))
    # loss_qr.append(np.linalg.norm(Q.conj().T @ Q - np.eye(n).astype(np.complex128), 'fro'))
    
plt.figure(figsize=(12,8))
plt.plot(kappas, loss_CGS, label='CGS', linestyle='-', color = 'b')
plt.plot(kappas, loss_MGS, label='MGS', linestyle='-', color = 'g')
plt.plot(kappas, loss_CGS_reortho, label='CGS2', linestyle='--', color = 'b')
plt.plot(kappas, loss_MGS_reortho, label='MGS2', linestyle='--', color = 'g')

# plt.plot(kappas, loss_qr, label='np.linalg.qr', linestyle='-', color = 'r')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('kappa')
plt.ylabel('Frobenies norm of |Q*Q-I|')
plt.legend()
plt.grid()
plt.title(f'Orthogonality Loss, using CGS/MGS with/without reorthogonalization')
plt.show()