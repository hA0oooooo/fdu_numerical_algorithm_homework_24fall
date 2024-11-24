import numpy as np
from scipy.linalg import expm

def gen_diagonalizable_matrices(D):
    n1, n2 = D.shape
    if n1 != n2:
        print("Matrix D must be square")
        return None
    n = n1
    S = np.random.random((n, n))
    diag_sum = np.sum(np.abs(S), axis=1)
    np.fill_diagonal(S, diag_sum)
    return np.linalg.inv(S) @ D @ S


def truncated_Taylor_exp(origin_A, numterms = np.inf):
    A = origin_A.copy()
    n1, n2 = A.shape
    if n1 != n2:
        print("Matrix A must be square")
        return None
    n = n1
    tol = 1e-16
    sum0 = np.eye(n)
    sum1 = sum0 + A
    term = A @ A / 2
    times = 2
    while np.linalg.norm(sum1 - sum0) > tol and times < numterms:
        sum0 = sum1
        sum1 += term
        term = term @ A / (times)
        times += 1

    return sum1

def pade_approximant(origin_A):

    n1, n2 = origin_A.shape
    if n1 != n2:
        raise ValueError("Matrix A must be square.")
    A = origin_A.copy()
    n = n1

    I = np.eye(n)
    num = I + 1/2 * A
    denom = I - 1/2 * A
    exp_A = np.linalg.inv(denom) @ num
    
    return exp_A


def scaling_squaring(A, fun):
    """
    MV2003 - Nineteen dubious ways to compute the exponential of a matrix, twenty-five years later
    P13 Table 1 Optimum scaling and squaring parameters with diagonal Pad´e and Taylor series approximation.
    Dict returns (nums of terms, scaling factor) according to int(log(2-norm of A))
    """
    norm = np.linalg.norm(A, ord=2)
    close_to_zero = 2
    if fun == "truncated_Taylor_exp":
        # test result: for 5x5 matrices, when A's 2-norm is less than 1e-2, the relative error of direct truncated_Taylor_exp is about 1e-6
        j = int(np.floor(np.log2(10 ** close_to_zero * norm)) + 1)
        numterms = np.inf
        result = truncated_Taylor_exp(A / (2 ** j), numterms)
    if fun == "pade_approximant":
        j = int(np.floor(np.log2(10 ** close_to_zero * norm)) + 1)
        result = pade_approximant(A / (2 ** j))
    
    for _ in range(j):
        result = result @ result

    return result
    
if __name__ == "__main__":
    
    n = 5
    D = np.diag(np.random.randn(n))
    A = gen_diagonalizable_matrices(D)
    norm = np.linalg.norm(A)
    result_taylor = scaling_squaring(A, "truncated_Taylor_exp")
    result_pade = scaling_squaring(A, "pade_approximant")
    standard =expm(A)
    print(f"2-norm of A: {norm}")
    print("\n")
    print("* scaling-and-squaring algorithm (combined with truncated Taylor series)")
    print(f"relative difference between my e^A and expm()'s e^A: {np.linalg.norm(result_taylor - standard)/ np.linalg.norm(standard)}")
    print("\n")
    print("* scaling-and-squaring algorithm (combined with Pad´e approximants)")
    print(f"relative difference between my e^A and expm()'s e^A: {np.linalg.norm(result_pade - standard)/ np.linalg.norm(standard)}")


    """
    // test the accruacy of truncated_Taylor_exp, no scaling and squaring
    n = 5
    A = np.random.randn(n, n) / 10 ** 2
    result = truncated_Taylor_exp(A)
    print(np.linalg.norm(A))
    print(np.linalg.norm(result-expm(A), ord = 'fro')/np.linalg.norm(expm(A), ord = 'fro'))
    """