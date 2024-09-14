import numpy as np
from math import sqrt

# Method One: get Cholesky factorization through LU factorization

def cholesky_1(A):
    n = A.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            factor = A[j, i] / A[i, i]
            A[j, i] = factor
            A[j, i+1:] = A[j, i+1:] - factor * A[i, i+1:]        
    diag_elements = np.diag(A)
    diag_matrix = np.diag(np.sqrt(diag_elements))

    L_tilde = np.tril(A, k = -1)
    np.fill_diagonal(L_tilde, 1)
    L = np.dot(L_tilde, diag_matrix)
    return L

# Method Two: get Cholesky factorization through undetermined coefficient
def cholesky_2(A):
    n = A.shape[0]
    for i in range(n):
        A[i, i] = sqrt(A[i, i])
        A[i+1:, i] = A[i+1:, i] / A[i, i]
        for j in range(i+1 , n):
            A[j:, j] = A[j:, j] - A[j, i] * A[j:, i]
    L = np.tril(A)
    return L        

def check_factorization(A, fac_fun):
    L = fac_fun(A)
    A_result = np.dot(L, L.T)
    print(f"{fac_fun.__name__} gets \n{A_result}")
    return A_result

# test
A = np.array([[1,2,3],
              [2,7,6],
              [3,6,10]], dtype = np.float64)

A_copy = np.copy(A)

check_factorization(A, cholesky_1)
check_factorization(A_copy, cholesky_2)