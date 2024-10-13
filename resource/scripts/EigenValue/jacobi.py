import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns


def sym_schur2(A, p, q):
    if A[p, q] != 0:
        tau = (A[q, q] - A[p, p]) / (2 * A[p, q])
        if tau >= 0:
            t = 1 / (tau + np.sqrt(1 + tau ** 2))
        else:
            t = 1 / (tau - np.sqrt(1 + tau ** 2))
        if t == t + 1:
            c, s = 1., 0.
        else:
            c = 1 / np.sqrt(1 + t ** 2)
            s = t * c
    else:
        c, s = 1., 0.
    return np.array([[c, s], [-s, c]])


def cyc_jacobi(A, history=False):
    """Jacobi diagonalization algorithm for real symmetric matrices."""
    n = A.shape[0]
    sweeps = 0
    delta = 2 * 1e-15 * np.linalg.norm(A)
    if not history:
        while sum([abs(A[i, j]) ** 2 for i in range(n) for j in range(n) if i != j]) > delta:
            sweeps += 1
            for p in range(n - 1):
                for q in range(p, n):
                    J = sym_schur2(A, p, q)
                    A[(p, q), :] = J.T @ A[(p, q), :]
                    A[:, (p, q)] = A[:, (p, q)] @ J
        return A, sweeps
    else:
        records = [A.copy()]
        while sum([abs(A[i, j]) ** 2 for i in range(n) for j in range(n) if i != j]) > delta:
            sweeps += 1
            for p in range(n - 1):
                for q in range(p, n):
                    J = sym_schur2(A, p, q)
                    A[(p, q), :] = J.T @ A[(p, q), :]
                    A[:, (p, q)] = A[:, (p, q)] @ J
            records.append(A.copy())
        return A, sweeps, records


def vis_performance():
    import pandas as pd
    data = pd.DataFrame(columns=["length of side", "sweeps", "number of Jacobi rotations"])
    for n in [2 ** n for n in range(2, 10)]:
        A = np.random.random((n, n))
        A = A + A.T
        A, sweeps = cyc_jacobi(A)
        data.loc[len(data.index)] = [n, sweeps, sweeps * n * (n - 1) / 2]
        print(f"length {n} finished!")

    plt.figure()
    ax = data.plot(x="length of side", y=["sweeps", "number of Jacobi rotations"],
                   secondary_y=["number of Jacobi rotations"])
    ax.set_ylabel("sweeps")
    ax.right_ax.set_ylabel("number of Jacobi rotations")
    ax.right_ax.set_yscale("log")
    ax.set_xscale("log")
    plt.title("Performance of Jacobi diagonalization")
    plt.savefig("Jacobi_performances.png")

    plt.show()


def vis_convergence():
    n = 500
    A = np.random.random((n, n))
    A = A + A.T
    A, sweeps, records = cyc_jacobi(A, True)
    print(sweeps)

    def update(i):
        mat.set_array(records[i])

    fig, ax = plt.subplots()
    mat = ax.matshow(records[0])
    plt.colorbar(mat)
    plt.title(f"n = {n}, {sweeps} sweeps, {int(sweeps * n * (n - 1)/2)} Jacobi rotations")
    ani = animation.FuncAnimation(fig, update, frames=len(records), interval=500)
    ani.save("convergence_history.gif")


if __name__ == "__main__":
    sns.set_theme()
    np.set_printoptions(suppress=True, precision=2)
    vis_performance()
