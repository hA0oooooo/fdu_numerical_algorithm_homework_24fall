import numpy as np
from scipy.sparse import random as sparse_random
from scipy.linalg import eigvalsh_tridiagonal
import matplotlib.pyplot as plt

def lanczos_algorithm(A, iteration_time, v_init=None, tol=1e-16, reorth=True):
    n = A.shape[0]
    if v_init is None:
        v_init = np.random.randn(n)
    v_init = v_init / np.linalg.norm(v_init)

    Q = np.zeros((n, iteration_time+1))
    alpha = np.zeros(iteration_time)
    beta = np.zeros(iteration_time)

    Q[:, 0] = v_init
    w = A @ Q[:, 0]
    alpha[0] = np.vdot(Q[:, 0], w)
    w = w - alpha[0]*Q[:, 0]
    beta[0] = np.linalg.norm(w)

    if beta[0] < tol:
        Q = Q[:, :1]
        alpha = alpha[:1]
        beta = beta[:0]
        return alpha, beta, Q

    Q[:, 1] = w / beta[0]

    for j in range(1, iteration_time):
        w = A @ Q[:, j]
        w = w - beta[j-1] * Q[:, j-1]
        alpha[j] = np.vdot(Q[:, j], w)
        w = w - alpha[j]*Q[:, j]
        beta[j] = np.linalg.norm(w)

        if beta[j] < tol:
            Q = Q[:, :j+1]
            alpha = alpha[:j+1]
            beta = beta[:j]
            break

        Q[:, j+1] = w / beta[j]

        if reorth:
            
            for k in range(j+1):
                c = np.vdot(Q[:, k], Q[:, j+1])
                Q[:, j+1] -= c * Q[:, k]
            for k in range(j+1):
                c = np.vdot(Q[:, k], Q[:, j+1])
                Q[:, j+1] -= c * Q[:, k]

            norm_q = np.linalg.norm(Q[:, j+1])

            if norm_q < tol:
                Q = Q[:, :j+1]
                alpha = alpha[:j+1]
                beta = beta[:j]
                break
            Q[:, j+1] /= norm_q

    if Q.shape[1] > len(alpha)+1:
        Q = Q[:, :len(alpha)+1]

    return alpha, beta[:len(alpha)-1], Q


def build_tridiag(alpha, beta):
    dimension = len(alpha)
    T = np.zeros((dimension, dimension))
    T[np.arange(dimension), np.arange(dimension)] = alpha
    if dimension > 1:
        T[np.arange(dimension-1), np.arange(1,dimension)] = beta
        T[np.arange(1,dimension), np.arange(dimension-1)] = beta
    return T


def generate_sparse_symmetric_matrix(n, density, seed=None):

    if seed is not None:
        np.random.seed(seed)
    A = np.zeros((n, n))
    nonzero_num = int(n * (n-1) * density / 2)  
    for _ in range(nonzero_num):
        row = np.random.randint(0, n)
        col = np.random.randint(0, n)
        A[row, col] = np.random.rand()
        A = (A + A.T) / 2

    return A


def test_lanczos(A, iteration_time, plot_ritz_convergence=True, plot_orthogonality=True):

    dimension = A.shape[0]
    alpha, beta, Q = lanczos_algorithm(A, iteration_time, reorth=True)
    ritz_vals = eigvalsh_tridiagonal(alpha, beta) if len(beta) > 0 else alpha

    if plot_ritz_convergence:
        exact_eigvals = np.linalg.eigvalsh(A)
        exact_eigvals.sort()

        ritz_values_by_iteration = []
        for i in range(1, len(alpha) + 1):
            Ti = build_tridiag(alpha[:i], beta[:i-1]) if i > 1 else np.array([[alpha[0]]])
            current_ritz_vals = np.linalg.eigvalsh(Ti)
            ritz_values_by_iteration.append(current_ritz_vals)

        plt.figure(figsize=(8, 6))

        max_iter = len(ritz_values_by_iteration)
        for k in range(max_iter):
            x_vals = np.arange(k+1, max_iter+1)
            y_vals = [ritz_values_by_iteration[i][k] for i in range(k, max_iter)]
            plt.plot(x_vals, y_vals, 'o', color='b',  markersize=1.5)

        plt.plot([iteration_time+1]*len(exact_eigvals), exact_eigvals, 'or', label='Exact eigenvalues (A)', markersize=0.3)

        plt.title(f'Ritz Pair Convergence, {dimension}x{dimension} sparse matrix')
        plt.xlabel('Iteration')
        plt.ylabel('Eigenvalues')
        plt.xlim([0, iteration_time+2])  
        plt.legend()
        plt.grid(True)
        plt.show(block = False)

    if plot_orthogonality:
        Q_t_Q = Q.T @ Q
        orth_loss = np.linalg.norm(Q_t_Q - np.eye(Q.shape[1]), ord='fro')
        print(f"Orthogonality loss ||Q^T Q - I|| = {orth_loss:.5e}")

        orth_losses = []
        for i in range(1, Q.shape[1]+1):
            Qi = Q[:,:i]
            orth_losses.append(np.linalg.norm(Qi.T @ Qi - np.eye(i), ord='fro'))

        plt.figure(figsize=(8,6))
        plt.plot(range(1, len(orth_losses)+1), orth_losses, '.-b', label='Orthogonality Loss')
        plt.title(f'Orthogonality Loss vs Iterations, {dimension}x{dimension} sparse matrix')
        plt.xlabel('Iteration')
        plt.ylabel('||Q_i^T Q_i - I||_F')
        plt.yscale('log')
        plt.legend()
        plt.show(block = False)

    print(f"Number of steps performed: {len(alpha)}")
    print("Approximate final Ritz values:")
    print(ritz_vals)
    print("Exact eigenvalues:")
    print(np.linalg.eigvalsh(A))


if __name__ == "__main__":

    dimension = 500
    iteration_time = 100
    A = generate_sparse_symmetric_matrix(dimension, density=0.1)
    test_lanczos(A, iteration_time)
    plt.show()