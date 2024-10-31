import matplotlib.pyplot as plt
import numpy as np
import time

def generate_hilbert(n):
    """
    Generate a Hilbert curve of order n.
    :param n: the dimension of the Hilbert matrix
    """
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            H[i, j] = 1 / (i + j + 1)  # 注意索引从0开始
    return H

def inverse_rayleigh_iter(origin_A, x, nu, tol=10e-16):
    """
    Compute the eigenvalue of A that is closest to nu using inverse iteration and Rayleigh quotient iteration. 
    :param origin_A: matrix A
    :param x: initial guess of eigenvector, an array
    :param nu: the eigenvalue that is closest to nu
    :return: max closest eigenvalue to nu and corresponding eigenvector
    :return: iteration history
    """
    target = nu
    n1, n2, n3 = origin_A.shape[0], origin_A.shape[1], len(x)
    if n1 != n2 or n3 != n1:
        print("input error")
        return None  
    
    A = np.copy(origin_A)
    A_inverse = lambda nu: np.linalg.inv(A - nu * np.eye(n1))

    A_norm = np.linalg.norm(A, ord=np.inf)
    x = x / np.linalg.norm(x, ord=np.inf)
    x = x.T
    times = 0
    times_record = []
    res_record = []
    nu_record = []

    while True:
        y = A_inverse(nu) @ x
        y = y / np.linalg.norm(y, ord=np.inf)
        nu = y.T @ A @ y / (y.T @ y)
        res =  A @ y - nu * y
        times = times + 1
        res_norm = np.linalg.norm(res, ord=np.inf)

        print("iter: ", times, "residual: ", res_norm, "closest eigenvalue: ", nu)
        times_record.append(times)
        res_record.append(res_norm)
        nu_record.append(nu)

        if res_norm < tol:
            break

        x = y

    return y, nu, times_record, res_record, nu_record

n = 200
target_eigenvalue = 1.0
A = generate_hilbert(n)
x = np.random.rand(n)

start = time.time()
y, nu, times_record, res_record, nu_record = inverse_rayleigh_iter(A, x, target_eigenvalue)
delta = time.time() - start

print("total time: ", delta)
print("target eigenvalue: ", target_eigenvalue)
print("closest eigenvalue: ", nu)
# print("corresponding eigenvector: ", y)

plt.figure(figsize=(12, 8))
plt.title("the convergence of residual using inverse iteration and Rayleigh quotient iteration")
plt.plot(times_record, res_record, label="residual")
plt.xlabel("iteration times")
plt.ylabel("residual of ||(A-(nu)*I)^(-1)y-(nu)y||")
plt.xticks(np.arange(0, times_record[-1]+1, 1))
plt.axhline(y=0, color='blue', linestyle='--', label=f'Asymptote y = {0}')
plt.legend()
plt.show(block = False)

plt.figure(figsize=(12, 8))
plt.title("the convergence of max eigenvalue using inverse iteration and Rayleigh quotient iteration")
plt.plot(times_record, nu_record, label="closest eigenvalue")
plt.xlabel("iteration times")
plt.ylabel("closest eigenvalue")
plt.xticks(np.arange(0, times_record[-1]+1, 1))
plt.axhline(y=target_eigenvalue, color='blue', linestyle='--', label=f'intial = {target_eigenvalue}')
plt.axhline(y=nu, color='green', linestyle='--', label=f'closest eigenvector = {nu:.6f}')
plt.legend()
plt.show()