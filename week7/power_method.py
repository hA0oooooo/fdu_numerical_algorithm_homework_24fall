import numpy as np
import matplotlib.pyplot as plt

def power_method(origin_A, x, tol=10e-16):
    """
    use power method to find max eigenvalue and corresponding eigenvector
    :param origin_A: matrix A
    :param x: initial guess of eigenvector, an array
    :param tol: tolerance
    :return: max eigenvalue and corresponding eigenvector
    :return: iteration history
    """
    A = np.copy(origin_A)
    A_norm = np.linalg.norm(A, ord=np.inf)
    x = x / np.linalg.norm(x, ord=np.inf)
    x = x.T
    times = 0
    times_record = []
    res_record = []
    u_record = []
    while True:
        y = A @ x
        u = np.linalg.norm(y, ord=np.inf)
        y = y / u
        res = A @ y - u * y 
        times = times + 1
        res_norm = np.linalg.norm(res, ord=np.inf)

        if times % 10 == 0:
            if times % 1000 == 0:
                print("iter: ", times, "residual: ", res_norm, "max eigenvalue: ", u)
            if res_norm > 1e-4:
                times_record.append(times)
                res_record.append(res_norm)
                u_record.append(u)

        if res_norm < tol: # maybe tol*(A_norm + abs(u)) (?)
            break
        x = y
    return y, u, times_record, res_record, u_record

def generate_A_with_max_eigenvalue(n, max_lambda):
    """
    generate a matrix A with max eigenvalue
    :param n: size of A
    :param max_lambda: max eigenvalue
    :return: matrix A
    """
    D = np.random.rand(n)
    max_D = np.max(D) 
    D = D / max_D * max_lambda
    D_diag = np.diag(D)
    Q = np.random.rand(n, n)
    Q, _ = np.linalg.qr(Q)
    A = Q @ D_diag @ Q.T
    return A

n = 100
max_eigenvalue = np.random.rand() * 10
A = generate_A_with_max_eigenvalue(n, max_eigenvalue)
x = np.random.rand(n)
y, u, times_record, res_record, u_record = power_method(A, x)
print("theoretically max eigenvalue: ", max_eigenvalue)
print("max eigenvalue: ", u)
# print("corresponding eigenvector: ", y)

plt.figure(figsize=(12, 8))
plt.title("the convergence of residual using power method")
plt.plot(times_record, res_record, label="residual")
plt.xlabel("iteration times")
plt.ylabel("residual of ||Ay-uy||")
plt.axhline(y=0, color='blue', linestyle='--', label=f'Asymptote y = {0}')
plt.legend()
plt.show(block = False)

plt.figure(figsize=(12, 8))
plt.title("the convergence of max eigenvalue using power method")
plt.plot(times_record, u_record, label="max eigenvalue")
plt.xlabel("iteration times")
plt.ylabel("max eigenvalue")
plt.axhline(y=max_eigenvalue, color='blue', linestyle='--', label=f'theoretically max eigenvalue = {max_eigenvalue:.6f}')
plt.legend()
plt.show()