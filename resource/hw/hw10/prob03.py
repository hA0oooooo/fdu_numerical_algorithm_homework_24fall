import numpy as np


def gen_diagonalizable_matrices(D):
    n = D.shape[0]
    S = np.random.random((n, n))
    diag_sum = np.sum(np.abs(S), axis=1)
    np.fill_diagonal(S, diag_sum)
    return np.linalg.inv(S) @ D @ S


def direct_taylor_exp(A, optorder=float("inf")):
    n = A.shape[0]
    tol = 1e-15
    term = A @ A / 2
    a1 = np.eye(n)
    a2 = a1 + A
    iters = 3
    while np.linalg.norm(a2 - a1) > tol and iters < optorder:
        a1 = a2
        a2 += term
        term = term @ A / iters
        iters += 1
    return a2


def scaling_squaring(A):
    norm = np.linalg.norm(A, ord=2)
    optimum_choices = {-2: (5, 1), -1: (5, 4), 0: (7, 5), 1: (9, 7),
                       2: (10, 10), 3: (8, 14)}
    if norm < 0.5e-2:
        return direct_taylor_exp(A)
    elif norm > 0.8e4:
        norm_after_scaling = norm / 16384
        j = 14
        while norm_after_scaling > 0.5:
            j += 1
            norm_after_scaling /= 2
        optorder = float("inf")
    else:
        optorder, j = optimum_choices[int(np.log10(norm))]
    result = direct_taylor_exp(A / (2 ** j), optorder)
    for _ in range(j):
        result = result @ result
    return result


if __name__ == "__main__":
    from scipy.linalg import expm
    np.set_printoptions(suppress=True, precision=2)
    n = 5
    D = np.diag(np.diag(np.random.random((n, n))))
    A = gen_diagonalizable_matrices(D)
    result1 = scaling_squaring(A)
    baseline = expm(A)
    print(f"2-norm of A: {np.linalg.norm(A)}")
    print(f"difference between e^A and expm(A) from scipy: {np.linalg.norm(result1 - baseline)}")




