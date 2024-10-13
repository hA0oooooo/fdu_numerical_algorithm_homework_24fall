import numpy as np
import pandas as pd


def gen_RD(r, m):
    P = np.random.random((r, m))
    for i in range(r):
        P[i, :] /= np.linalg.norm(P[i, :])
        for j in range(i + 1, r):
            P[j, :] -= np.inner(P[i, :].conj(), P[j, :]) * P[i, :]
    return P


def rank_factor_RD(A, r):
    m, n = A.shape
    P = gen_RD(r, m)
    Q = P @ A
    return P.T, Q


def gen_testing_mat(m, n, r):
    diag = np.diag(np.diag(np.random.random((r, r)))) + 1
    return gen_RD(r, m).T @ diag @ gen_RD(r, n)


if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=3)
    data = pd.DataFrame(columns=["m", "n", "r", "cond of A", "residual", "residual/cond"])
    for m in range(20, 200):
        n = np.random.randint(10, m - 1)
        r = np.random.randint(2, n // 2)
        A = gen_testing_mat(m, n, r)
        cond = np.linalg.norm(A, ord=2) * np.linalg.norm(np.linalg.pinv(A), ord=2)
        P, Q = rank_factor_RD(A, r)
        residual = np.linalg.norm(A - P @ Q)
        data.loc[len(data.index)] = [m, n, r, cond, residual, residual / cond]
    print(data)
    import openpyxl # to export Dataframe into excel
    data.to_excel("prob05output.xlsx", engine="openpyxl")
