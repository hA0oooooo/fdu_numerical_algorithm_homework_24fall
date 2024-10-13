import numpy as np


def initialize(n):
    grid = np.random.random((n + 1, n + 1))
    grid[0, :] = 0
    grid[-1, :] = 0
    grid[:, -1] = 0
    grid[:, 0] = np.array([np.sin(np.pi * i / n) for i in range(n + 1)])
    return grid


def jacobi_iter(grid):
    new_grid = np.empty(shape=grid.shape, dtype=grid.dtype)

    new_grid[1:-1, 1:-1] = 0.25 * (grid[1:-1, :-2] + grid[1:-1, 2:] + grid[:-2, 1:-1] + grid[2:, 1:-1])

    new_grid[0, :] = grid[0, :]
    new_grid[-1, :] = grid[-1, :]
    new_grid[:, 0] = grid[:, 0]
    new_grid[:, -1] = grid[:, -1]

    return new_grid


def gauss_seidel_iter(grid):
    newgrid = grid.copy()

    for i in range(1, newgrid.shape[0] - 1):
        for j in range(1, newgrid.shape[1] - 1):
            newgrid[i, j] = 0.25 * (newgrid[i, j + 1] + newgrid[i, j - 1] +
                                    newgrid[i + 1, j] + newgrid[i - 1, j])

    return newgrid


def convergence_his(n, func, iters=1000, sol=None):
    eps = 2e-14
    errs = []
    if sol is None:
        grid = initialize(n)
        for _ in range(iters):
            new_grid = func(grid)
            grid = new_grid
        sol = grid
    grid = initialize(n)
    for _ in range(iters):
        new_grid = func(grid)
        errs.append(np.linalg.norm(grid - sol))
        if errs[-1] < eps:
            break
        grid = new_grid
    return np.array(errs)


def vis_grid(n, func):
    grid = initialize(n)

    plt.figure(figsize=(10, 10))
    if func == jacobi_iter:
        func_name = "Jacobi"
    else:
        func_name = "Gauss-Seidel"
    for i in range(200 + 1):
        if i % 20 == 0:
            plt.subplot(4, 3, int(i / 20 + 1))
            plt.imshow(grid.T, interpolation='none', origin='lower')
            plt.title(func_name + f', iter = {i}')

        grid = func(grid)

    plt.tight_layout()
    plt.savefig("hw11pic1" + func_name + ".png")
    plt.show()


def vis_conv_his(n):
    errs_j = convergence_his(n, jacobi_iter, 10000)
    errs_gs = convergence_his(n, gauss_seidel_iter, 10000)
    fig, ax = plt.subplots()
    ax.semilogy(range(len(errs_j)), errs_j, label="Jacobi")
    ax.semilogy(range(len(errs_gs)), errs_gs, label="Gauss-Seidel")
    ax.set(title=f"Error of the iteration, n={n}", xlabel="iterations", ylabel="error")
    ax.legend()
    plt.savefig("hw11pic2.png")
    plt.show()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    np.set_printoptions(suppress=True, precision=3)
    vis_grid(25, jacobi_iter)
    vis_grid(25, gauss_seidel_iter)
    vis_conv_his(25)
