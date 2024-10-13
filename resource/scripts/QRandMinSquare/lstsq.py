import numpy as np
import qrdecomp as qr
import soltri as st


def lsCholesky(A, b):
    R = qr.Cholesky(A.T.conj() @ A)
    y = st.solve_tril_by_row(R.conj().T, A.conj().T @ b)
    return st.solve_triu_by_row(R, y)


def aug_solve(A, b):
    m, n = A.shape
    aA = np.zeros((m+n, m+n))
    aA[:m, :m] = np.eye(m)
    aA[:m, m:] = A.copy()
    aA[m:, :m] = A.conj().T
    ab = np.concatenate((b, np.zeros(n)))
    x = np.linalg.solve(aA, ab)[m:]
    return x


def HouseQR_solve(A, b):
    n = A.shape[1]
    hA, betas = qr.house_tri(A)
    R = np.triu(hA)
    for i in range(hA.shape[1]):
        b[i:] = qr.house_left(b[i:], np.concatenate([[1], hA[i + 1:, i]]), betas[i], True)
    return st.solve_triu_by_row(R[:n, :], b[:n])


def MGS_solve(A, b, reortho=False):
    Q, R = qr.GSQR(A, True, reortho)
    return st.solve_triu_by_row(R, Q.T.conj() @ b)


def test():
    from math import sqrt
    m, n = 200, 50
    times = 20
    sns.set_theme()
    norm = np.linalg.norm
    data = pd.DataFrame(columns=['$\kappa(A)$', 'algorithms', 'relative difference'])
    kappas = [10 ** x for x in range(16)]
    for i in range(times):
        for k in kappas:
            A = qr.rand_matrix_by_kappa(m, n, k)
            b = np.random.random((m, ))
            standard_res = sqrt(np.linalg.lstsq(A, b, rcond=-1)[1])
            # Householder
            x = HouseQR_solve(A.copy(), b.copy())
            res = norm(b - A @ x)
            data.loc[len(data.index)] = [k, "Householder", (res - standard_res) / standard_res]
            # Cholesky
            x = lsCholesky(A.copy(), b.copy())
            res = norm(b - A @ x)
            data.loc[len(data.index)] = [k, "Cholesky", (res - standard_res) / standard_res]
            # augmented system
            x = aug_solve(A.copy(), b.copy())
            res = norm(b - A @ x)
            data.loc[len(data.index)] = [k, "solve augmented system", (res - standard_res) / standard_res]
            # MGS
            x = MGS_solve(A.copy(), b.copy())
            res = norm(b - A @ x)
            data.loc[len(data.index)] = [k, "MGS", (res - standard_res) / standard_res]
            # MGS(2)
            x = MGS_solve(A.copy(), b.copy(), True)
            res = norm(b - A @ x)
            data.loc[len(data.index)] = [k, "MGS(2)", (res - standard_res) / standard_res]
        print(f"---test{i} finished---")
    sns.relplot(data, x='$\kappa(A)$', y='relative difference', legend='brief',
                 hue='algorithms', style="algorithms", kind="line").set(xscale="log")
    plt.savefig("hw05pic3.png")
    plt.show()




def Problem5():
    data = pd.DataFrame({"n": list(range(2, 8)), '$ln\ n$': [np.log(x) for x in range(2, 8)]})
    b = data['$ln\ n$'].to_numpy()
    A = np.concatenate((data["n"].to_numpy(), [1] * 6)).reshape((2, 6)).T
    x = np.linalg.lstsq(A ,b, rcond=-1)
    a1, a2 = x[0][0], x[0][1]
    data["lsq"] = data["n"] * a1 + a2
    fig, ax = plt.subplots()
    sns.scatterplot(data, x='n', y='$ln\ n$', legend=False, ax=ax)
    sns.lineplot(data, x='n', y='lsq', legend=False, ax=ax)
    ax.text(2, 2, f"y = {a1:.3f}x + {a2:.3f}")
    plt.savefig("hw05pic4.png")
    plt.show()

if __name__ == "__main__":
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt
    Problem5()








