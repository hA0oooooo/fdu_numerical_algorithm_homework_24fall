import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def diag_pls_rank1(n, d, z):
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
    d = np.arange(1, n + 1)
    z = np.random.randn(n)
    A = diag_pls_rank1(n, d, z)
    eigs = np.linalg.eigvalsh(A)
    f = generate_func(n, d, z)

    sns.set_theme()
    x = np.arange(min(eigs) - 0.5, max(eigs) + 0.5, 0.00001)
    y = f(x)
    y = np.ma.masked_where(y > 20, y)
    y = np.ma.masked_where(y < -20, y)
    fig, ax = plt.subplots()
    ax.plot(x, y)
    plt.axhline(y=0, c='red')
    ax.plot(eigs, [0] * len(eigs), ".", c="black", label="true eigs")
    plt.legend()
    plt.title(f"n={n}")
    plt.savefig("hw09pic1.png")
    plt.show()