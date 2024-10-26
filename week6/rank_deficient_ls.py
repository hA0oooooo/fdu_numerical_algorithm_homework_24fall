import numpy as np
import matplotlib.pyplot as plt

def qr_decomposition_with_pivoting(origin_A, tol=1e-10):
    """
    matrix A is a rank deficient matrix, return QR decomposition with column pivoting
    :param origin_A: matrix A, m x n, m >= n
    :return: Q (m x m, Q*Q = I), R, P (record column exchange)
    """
    A = np.copy(origin_A).astype(float)
    m, n = A.shape
    if m < n:
        print(f'warning, m={m} < n={n}')
        return None, None, None
    exchange = np.arange(n)
    Q = np.zeros((m, m))
    R = np.zeros((m, n))
    col_norms = np.sum(A**2, axis=0)
    rank = 0

    for i in range(n):
        pivot = np.argmax(col_norms[i:]) + i
        if col_norms[pivot] < tol:
            break
        if pivot != i:
            A[:, [i, pivot]] = A[:, [pivot, i]]
            exchange[i], exchange[pivot] = exchange[pivot], exchange[i]
            col_norms[i], col_norms[pivot] = col_norms[pivot], col_norms[i]
        
        R[i, i] = np.linalg.norm(A[:, [i]], ord=2)
        Q[:, [i]] = A[:, [i]] / R[i, i]
        R[i, i+1:] = Q[:, [i]].T @ A[:, i+1:]
        
        A[:, i+1:] = A[:, i+1:] - np.outer(Q[:, [i]], R[i, i+1:])
        col_norms[i+1:] = col_norms[i+1:] - R[i, i+1:]**2
        col_norms[col_norms < tol] = 0

        rank = rank + 1

    # make Q orthogonalized square matrix

    for j in range(m):
        if rank == m:
            break  
        e = np.zeros(m, dtype=float)
        e[j] = 1.0

        for k in range(rank):
            projection = np.dot(e, Q[:, k])
            e = e - projection * Q[:, k]
        norm_e = np.linalg.norm(e, ord=2)
        if norm_e > tol:
            Q[:, rank] = e / norm_e
            rank += 1

    P = np.zeros((n, n), dtype=int)
    for i in range(n):
        P[i, exchange[i]] = 1

    return Q, R, P

# now min|Ax - b| -> min|QRPx - b| -> min|RPx - Q^T b| -> min|[R1 0]^T y - [c1 c2]^T|, in which y = Px, R1 is full-ranked
# so min|Ax - b| = min|R1^T y - c1| + |c2| = |c2|, thanks to R1's rank <= n

def ls_for_rank_deficient_matrix(A, b, tol=1e-10):
    Q, R, P = qr_decomposition_with_pivoting(A, tol)
    c = Q.T @ b
    rank_R = np.linalg.matrix_rank(R)
    c2 = c[rank_R:]
    norm_c2 = np.linalg.norm(c2, ord=2)
    return norm_c2 ** 2

def generate_rank_deficient_matrix_method(m, n, r):
    U = np.random.randn(m, r)
    V = np.random.randn(n, r)
    singular_values = np.linspace(1, 10, r)
    Sigma = np.diag(singular_values)
    C = U @ Sigma @ V.T
    return C

# test
m = 1000
n = 500
r = 300
A = generate_rank_deficient_matrix_method(m, n, r)
b = np.random.randn(m, 1)
Q, R, P = qr_decomposition_with_pivoting(A)

x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
residual = b - A @ x
residual_sum = np.sum(residual**2)

error = np.abs(ls_for_rank_deficient_matrix(A, b) - residual_sum).astype(np.float64)

print(f'fun *np.linalg.lstsq(A, b, rcond=None)*: {residual_sum}')
print(f'fun *ls_for_rank_deficient_matrix(A, b)*: {ls_for_rank_deficient_matrix(A, b)}')
print(f'error: {error}')

plt.figure(figsize=(8, 6))
plt.title(f'rank deficient least squares problems with two solver \n A is {m} × {n} matrix with rank {r}')
x = ['my solver'  , 'np.linalg.lstsq' , 'error']
y = [ls_for_rank_deficient_matrix(A, b), residual_sum, error]
bars = plt.bar(x, y, color='skyblue', width=0.4)
plt.ylabel('min|Ax - b|')

for i, bar in enumerate(bars):
    height = bar.get_height()
    if i == 2: 
        label = f'{height:.2e}'
    else:
        label = f'{height:.6f}'
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        label,
        ha='center',
        va='bottom',
        fontsize=12,
        color='black'
    )

plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()