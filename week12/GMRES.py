import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp

def generate_sparse_matrix(n, nonzero_num):
    A = np.zeros((n, n))
    for _ in range(nonzero_num):
        row = np.random.randint(0, n)
        col = np.random.randint(0, n)
        A[row, col] = np.random.rand()*10
    return A

def generate_symmetric_sparse_matrix(n, nonzero_num):
    A = generate_sparse_matrix(n, nonzero_num)
    return (A + A.T)/2

def givens_rotation(a, b):
    if b == 0:
        c = 1.0; s = 0.0; r = a
    else:
        r = np.sqrt(a**2 + b**2)
        c = a/r; s = b/r
    return c, s, r

def arnoldi_iteration(A, V, H, j):
    w = A @ V[:, j]
    for i in range(j+1):
        H[i, j] = np.dot(V[:, i], w)
        w = w - H[i, j]*V[:, i]
    H[j+1, j] = np.linalg.norm(w)
    if H[j+1, j] != 0:
        w = w / H[j+1, j]
    return w

def apply_givens_rotations(H, cos, sin, ls_b, k):
    for i in range(k):
        temp = cos[i]*H[i,k] + sin[i]*H[i+1,k]
        H[i+1,k] = -sin[i]*H[i,k] + cos[i]*H[i+1,k]
        H[i,k] = temp

    c, s_, r = givens_rotation(H[k,k], H[k+1,k])
    cos[k] = c
    sin[k] = s_

    H[k,k] = r
    H[k+1,k] = 0.0

    temp = cos[k] * ls_b[k] + sin[k] * ls_b[k+1]
    ls_b[k+1] = -sin[k] * ls_b[k] + cos[k] * ls_b[k+1]
    ls_b[k] = temp

def back_substitution(H, ls_b, m):
    y = np.zeros(m)
    for i in reversed(range(m)):
        y[i] = ls_b[i]
        for k in range(i+1, m):
            y[i] -= H[i, k] * y[k]
        y[i] = y[i] / H[i, i]
    return y

def gmres(A, b, x0, tol, max_iter):
    n = A.shape[0]
    r0 = b - A @ x0
    beta = np.linalg.norm(r0)
    if beta < tol:
        return x0, []

    V = np.zeros((n, max_iter+1))
    V[:, 0] = r0 / beta

    H = np.zeros((max_iter+1, max_iter))
    cos = np.zeros(max_iter)
    sin = np.zeros(max_iter)
    ls_b = np.zeros(max_iter+1)
    ls_b[0] = beta

    residuals = [beta]

    for j in range(max_iter):
        w = arnoldi_iteration(A, V, H, j)
        if H[j+1, j] == 0: # maybe < tol
            # Lucky breakdown
            m = j+1
            y = back_substitution(H[0:m, 0:m], ls_b[0:m], m)
            x = x0 + V[:, 0:m] @ y
            return x, residuals

        V[:, j+1] = w
        apply_givens_rotations(H, cos, sin, ls_b, j)
        relres = abs(ls_b[j+1])
        residuals.append(relres)
        if relres < tol:
            m = j+1
            y = back_substitution(H[0:m ,0:m], ls_b[0:m], m)
            x = x0 + V[:,0:m] @ y
            return x, residuals

    m = max_iter
    y = back_substitution(H[0:m, 0:m], ls_b[0:m], m)
    x = x0 + V[:, 0:m] @ y
    return x, residuals

# main
n = 500
nonzero_num = 2000
max_iter = n*10
tol = 1e-50
x0 = np.zeros(n)

A_sym = sp.csr_matrix(generate_symmetric_sparse_matrix(n, nonzero_num))
b_sym = np.random.randn(n)
x_sym, residuals_sym = gmres(A_sym, b_sym, x0, tol, max_iter)

A_nonsym = sp.csr_matrix(generate_sparse_matrix(n, nonzero_num))
b_nonsym = np.random.randn(n)
x_nonsym, residuals_nonsym = gmres(A_nonsym, b_nonsym, x0, tol, max_iter)

plt.figure(figsize=(12, 8))
plt.semilogy(residuals_sym)
plt.xlabel('Iteration')
plt.ylabel('Residual Norm')
plt.yscale('log')
plt.title(f'GMRES Residual History for {n}x{n} sparse matrix with {nonzero_num} nonzeros (Symmetric), tol={tol}')
plt.grid(True)
plt.show(block = False)

plt.figure(figsize=(12, 8))
plt.semilogy(residuals_nonsym)
plt.xlabel('Iteration')
plt.ylabel('Residual Norm')
plt.yscale('log')
plt.title(f'GMRES Residual History for {n}x{n} sparse matrix with {nonzero_num} nonzeros (Nonsymmetric), tol={tol}')
plt.grid(True)

plt.tight_layout()
plt.show()
