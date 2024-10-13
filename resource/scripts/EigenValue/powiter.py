from random import random

import numpy as np
from scipy.linalg import lu_factor, lu_solve, solve


def power_iter(A, initial=None, show_history=True, eps=None):
    if initial:
        x = initial
    else:
        x = np.random.random(A.shape[0]) + np.random.random(A.shape[0]) * 1j
    if not eps:
        eps = np.linalg.norm(A) * 1e-16
    x /= np.linalg.norm(x)
    eigen, eigen_ = np.nan, np.nan
    history = []
    while not abs(eigen - eigen_) < eps:
        Ax = A @ x
        eigen, eigen_ = np.inner(x.conj(), Ax), eigen
        x = Ax / np.linalg.norm(Ax)
        history.append(eigen)
    if not show_history:
        return eigen
    else:
        return eigen, history


def inv_iter(A, eig, initial=None, show_history=True):
    A -= np.diagflat(np.full(A.shape[0], eig))
    if initial:
        x = initial
    else:
        x = np.random.random(A.shape[0]) + np.random.random(A.shape[0]) * 1j
    x /= np.linalg.norm(x)
    eps = np.linalg.norm(A) * 10e-16
    eigen, eigen_ = np.nan, np.nan
    history = []
    lu, piv = lu_factor(A, check_finite=False, overwrite_a=True)
    while not abs(eigen - eigen_) < eps:
        Ax = lu_solve((lu, piv), x, check_finite=False)
        eigen, eigen_ = np.inner(x.conj(), Ax), eigen
        x = Ax / np.linalg.norm(Ax)
        history.append(1./eigen + eig)
    if not show_history:
        return 1./eigen + eig
    else:
        return 1./eigen + eig, history


def rayleigh_iter(A, eig, initial=None, show_history=True):
    if initial:
        x = initial
    else:
        x = np.random.random(A.shape[0]) + np.random.random(A.shape[0]) * 1j
    eps = np.linalg.norm(A) * 2e-16
    eigen, eigen_ = eig - 1, eig
    x /= np.linalg.norm(x)
    history = []
    while abs(eigen - eigen_) > eps:
        eigen = eigen_
        Al = A - np.diagflat(np.full(A.shape[0], eigen))
        Ax = solve(Al, x, overwrite_a=True, check_finite=False)
        eigen_ = eigen + 1. / np.inner(x.conj(), Ax)
        x = Ax / np.linalg.norm(Ax)
        history.append(eigen)
    if not show_history:
        return eigen_
    else:
        return eigen_, history


def prob4():
    A = np.random.random((1000, 1000)) + np.eye(1000) * 0.5
    eigen, history = power_iter(A)
    diff = list(map(lambda x: abs(x - eigen), history))
    data = pd.DataFrame()
    data = pd.DataFrame()
    data["times"] = list(range(1, len(diff) + 1))
    data["difference"]= diff
    sns.relplot(data, x="times", y="difference", kind="line").set(yscale="log")
    plt.savefig("hw06pic1.png")
    plt.show()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    sns.set_theme()
    # problem 5
    from timeit import timeit
    from scipy.stats import unitary_group
    from random import choice
    def random_A(n):
        specs = np.random.random(200) + np.random.random(200) * 1j
        Q = unitary_group.rvs(200)
        A = Q @ np.diagflat(specs) @ Q.T.conj()
        spec = choice(specs)
        return A, specs, spec

    def perturb(a:complex):
        return a + (random() + random() * 1j) * (10 ** -2)

    # specs.pop(specs.index(spec))

    # visualize eigenvalue distribution
    # A, specs, spec = random_A(200)
    # result, history = inv_iter(A, perturb(spec))
    # fig, ax = plt.subplots()
    # ax.scatter(specs.real, specs.imag, s=12, c="lightblue", label="true eigenvalues")
    # plt.xlabel("real")
    # plt.ylabel("imag")
    # ax.scatter([spec.real], [spec.imag], s=12, c="b", marker="o", label="chosen eigenvalue")
    # ax.scatter([result.real], [result.imag], s=8, c="r", marker="x", label="computed eigenvalue")
    # print(result)
    # ax.legend()
    # plt.title("use inverse iter to compute eigenvalues")
    # plt.savefig("hw06pic2.png")
    # plt.show()

    # visualize convergence history of different methods
    data = pd.DataFrame(columns=["iterations", "algorithms", "residual"])
    for t in range(10):
        A, specs, spec = random_A(200)
        perturbed_spec = perturb(spec)
        # inv_iter
        eig, history = inv_iter(A.copy(), perturbed_spec)
        diff = list(map(lambda x: abs(x - eig), history))
        for i in range(1, len(diff) + 1):
            data.loc[len(data.index)] = [i, "Inverse Iteration", diff[i - 1]]
        # rayleigh_iter
        eig, history = rayleigh_iter(A.copy(), perturbed_spec)
        diff = list(map(lambda x: abs(x - eig), history))
        for i in range(1, len(diff) + 1):
            data.loc[len(data.index)] = [i, "Rayleigh Quotient Iteration", diff[i - 1]]
    p1 = sns.relplot(data, x="iterations", y="residual", hue="algorithms", kind="line").set(yscale="log", xscale="log")
    # visualize execution time
    A, specs, spec = random_A(200)
    perturbed_spec = perturb(spec)
    t1 = timeit("inv_iter(A.copy(), perturbed_spec)", number=10, globals=globals())
    t2 = timeit("rayleigh_iter(A.copy(), perturbed_spec)", number=10, globals=globals())
    p1.ax.text(100, 1e-14, f"Inverse iteration: {t1/ 10:.3e} sec")
    p1.ax.text(100, 1e-15, f"Rayleigh iteration: {t2 / 10:.3e} sec")
    plt.savefig("hw06pic03.png")
    plt.show()

