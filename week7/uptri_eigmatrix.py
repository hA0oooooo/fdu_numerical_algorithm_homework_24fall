import numpy as np


def generate_uptri_with_distinct_diagentries(n):
    """
    generate an upper triangular matrix with distinct diagonal entries
    :param n: the dimension of the square matrix
    """
    # np.random.seed(0)
    off_diag = np.triu(np.random.rand(n, n), 1) * 10
    while True:
        diag_entries = np.random.rand(n) * 10 
        if len(np.unique(diag_entries)) == n:
            break  
    U = np.diag(diag_entries) + off_diag
    return U


def one_eigenvector_solver(origin_A, lamb):
    """
    given one of the eigenvalue lamb, solve the corresponding eigenvector
    :param originA: the origin upper triangular matrix with distinct diagonal entries
    :param lamb: the given eigenvalue
    :return: the correspongding eigenvector, 1 x n
    """
    n1, n2 = origin_A.shape
    if n1 != n2:
        print("input error")
        return None
    A = np.copy(origin_A) - lamb * np.eye(n1)
    eigenvector = np.zeros(n1)
    diag = np.diag(A)

    for idx, value in enumerate(diag):
        if value == 0:
            break

    eigenvector[idx] = 1
    eigenvector[idx+1:] = 0

    if idx == 0:
        return eigenvector

    reduced_left = A[:idx, :idx]
    reduced_right = np.zeros(idx) - A[:idx, idx]
    part_eigv = np.zeros(idx)

    for i in range(idx-1, 0, -1):
        part_eigv[i] = reduced_right[i] / reduced_left[i][i]
        reduced_right[:i] = reduced_right[:i] - part_eigv[i] * reduced_left[:i, i]
    part_eigv[0] = reduced_right[0] / reduced_left[0][0]

    eigenvector[:idx] = part_eigv

    return eigenvector

def all_eigenvector_solver(origin_A):
    """
    use one_eigenvector_solver to solve all the eigenvectors
    :param originA: the origin upper triangular matrix with distinct diagonal entries
    :return: the matrix with all the eigenvectors, n x n
    """
    n1, n2 = origin_A.shape
    if n1 != n2:
        print("input error")
        return None
    A = np.copy(origin_A)
    diag_A = np.diag(A)
    eigenmatrix = np.zeros((n1, n1))

    for idx, lamb in enumerate(diag_A):
        eigenmatrix[:, idx] = one_eigenvector_solver(A, lamb)

    return eigenmatrix


n = 100
A = generate_uptri_with_distinct_diagentries(n)
eigenmatrix = all_eigenvector_solver(A)
# print(A)
# print(eigenmatrix)

np.savetxt('T5_A.txt', A, fmt='%.6e')

np.savetxt('T5_my_eigenmatrix.txt', eigenmatrix, fmt='%.6e')