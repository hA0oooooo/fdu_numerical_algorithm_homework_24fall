import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse as sparse

norm = np.linalg.norm


def CG(b, A, x0=None, eps=1e-5, miter=5000, his=True):
    n = b.shape[0]
    if x0 is None:
        x = np.zeros((n,))
        r = b
    else:
        r = b - A(x0)
        x = x0
    k = 0
    beta = 0
    p = 0
    res = [norm(r)]
    while res[-1] > eps and k < miter:
        p = r + beta * p
        w = A(p)
        alpha = np.inner(r, r) / np.inner(p, w)
        x += alpha * p
        lr = r.copy()
        r -= alpha * w
        beta = np.inner(r, r) / np.inner(lr, lr)
        k += 1
        print(k)
        res.append(norm(r))
    if k == miter:
        exit = miter
        print(f"CG warning: maxiter {miter} exceeded and still not converged")
        print(f"norm of residual: {res[-1]}")
    else:
        exit = 0
    if his:
        return exit, x, res
    return exit, x


def init_grid(n):
    grid = np.random.random((n, n))
    # grid = np.zeros((n, n))
    return grid


def left_mulA(v):
    w = np.zeros(v.shape, dtype=v.dtype)
    n = int(np.sqrt(v.shape[0]))
    for i in range(n):
        for j in range(n):
            k = i * n + j
            w[k] = 4 * v[k]
            if i != 0:
                w[k] -= v[k - n]
            else:
                w[k] -= np.sin(np.pi * j / (n - 1))
            if i != n - 1:
                w[k] -= v[k + n]
            if j != 0:
                w[k] -= v[k - 1]
            if j != n - 1:
                w[k] -= v[k + 1]
            w[k] /= 4
    return w


def vis_solution(n, x):
    plt.imshow(x.reshape((n, n), order="F"), interpolation='none', origin='lower')
    plt.title(f"solution")
    plt.savefig("hw13pic2.png")
    plt.show()


def solve_laplace2d(n):
    grid = init_grid(n)
    x0 = grid.reshape(n * n, order="F")
    exit, x, res = CG(b=np.zeros(n * n), A=left_mulA, x0=x0, eps=1e-5, miter=5000)
    return x, res


def vis_conv_his(n, res):
    fig, ax = plt.subplots()
    ax.semilogy(range(len(res)), res, label="CG")
    ax.set(title=f"convergence history, n={n}", xlabel="iterations", ylabel="2-norm of the residual")
    ax.legend()
    plt.savefig("hw13pic1.png")
    plt.show()


if __name__ == "__main__":
    np.set_printoptions(suppress=False, precision=1)
    sns.set()

    n = 1000
    x, res = solve_laplace2d(n)
    vis_conv_his(n, res)
    vis_solution(n, x)
