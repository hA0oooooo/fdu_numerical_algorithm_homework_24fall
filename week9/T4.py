import numpy as np
import matplotlib.pyplot as plt

def diag_add_rank1(n, d, z):
    A = np.zeros((n, n))
    np.fill_diagonal(A, d)
    A += np.outer(z, z)
    return A

def generate_func(n, d, z):
    def f(t):
        result = 1
        for i in range(n):
            result -= z[i] ** 2 / (t - d[i])
        return result
    return f

if __name__ == "__main__":
    n = 6
    d = np.linspace(1, 9, n)
    z = 1 * np.random.randn(n)
    A = diag_add_rank1(n, d, z)

    eigs = np.linalg.eigvalsh(A)

    f = generate_func(n, d, z)
    x = np.arange(min(eigs) - 1, max(eigs) + 1, 0.00001)
    y = f(x)

    plt.figure(figsize = (12, 8))
    y = np.ma.masked_where(y > 15, y)
    y = np.ma.masked_where(y < -15, y)
    plt.plot(x, y, color='navy')

    for i in d:
        plt.axvline(x=i, linestyle='--', color='black', alpha=0.7)
    xlim = plt.gca().get_xlim()
    plt.plot([xlim[0], d[0]], [1, 1], linestyle='--', color='black', alpha=0.7)
    plt.plot([d[n-1], xlim[1]], [1, 1], linestyle='--', color='black', alpha=0.7)
    plt.axhline(y=0, color='black', label=f"y=0")
    plt.plot(eigs, [0] * len(eigs), ".", color="red", label="true eigs")
    plt.legend()
    plt.title(f"n={n}")
    plt.show()
