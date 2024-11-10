import numpy as np
import matplotlib.pyplot as plt

def generate(n):
    A = np.random.rand(n, n)
    return A

def house(x):
    """
    householder transformation to let x become a unit vector
    :param x: a vector
    :return: a matrix H such that Hx = ||x||e_1, and H has the corresponding size of x
    """
    n = len(x)
    e1 = np.zeros(n)
    e1[0] = 1
    v = x + np.sign(x[0]) * np.linalg.norm(x) * e1
    v = v / np.linalg.norm(v)
    H = np.eye(n) - 2 * np.outer(v, v)
    return H


def hessenberg_reduction_hou(origin_A):
    """
    Reduce a matrix A to Hessenberg form with householder reflection
    :param A: a matrix
    :return: A, a Hessenberg matrix
    :return: Q, a orthogonal matrix, and origin_A = Q*AQ, QAQ* = H
    """
    n1, n2 = origin_A.shape
    if n1 != n2:
        print("input error")
        return None
    A = np.copy(origin_A)
    n = n1
    Q = np.eye(n)

    for k in range(n-2):
        x = A[k+1:, k]
        H = house(x)
        A[k+1:, k:] = H @ A[k+1:, k:]
        A[:, k+1:] = A[:, k+1:] @ H
        full_H = np.eye(n)
        full_H[k+1:, k+1:] = H
        Q = full_H @ Q

    return A, Q


def hessenberg_reduction_arnoldi(origin_A):
    """
    Reduce a matrix A to Hessenberg form with Arnoldi process based on modified Gram Schmidt orthogonalization.
    :param A: a matrix
    :return: H, a Hessenberg matrix
    :return: Q, a orthogonal matrix, and A = QHQ*

    """
    n1, n2 = origin_A.shape
    if n1 != n2:
        print("input error")
        return None
    
    A = np.copy(origin_A)
    n = n1
    Q = np.zeros((n, n))
    b = generate_vector(n)
    Q[:, 0] = b / np.linalg.norm(b, ord=2)
    H = np.zeros((n, n))

    for i in range(n-1):
        cur = A @ Q[:, [i]]
        for j in range(i+1):
            H[j, i] = np.dot(cur.T, Q[:, [j]]).item()
            cur = cur - H[j, i] * Q[:, [j]]
        H[i+1, i] = np.linalg.norm(cur, ord=2)
        if H[i + 1, i] == 0:
            print(f'H[{i+1}, {i}] = 0')
            return None
        cur = cur / H[i+1, i]
        Q[:, [i+1]] = cur
    
    H[:, [n-1]] = Q.T @ A @ Q[:, [n-1]]

    return H, Q


def generate_vector(n):
    """
    Generate a random vector x
    :param n: the size of the vector
    :return: x, a random vector
    """
    x = np.random.rand(n)
    return x


n = 400
number = 50
accuracy_hou = []
accuracy_arn = []
ortho_loss_hou = []
ortho_loss_arn = []

for i in range(number):
    A = generate(n)
    H_hou, Q_hou = hessenberg_reduction_hou(A)
    H_arn, Q_arn = hessenberg_reduction_arnoldi(A)
    accuracy_hou.append(np.linalg.norm(H_hou - Q_hou @ A @ Q_hou.T, ord="fro"))
    accuracy_arn.append(np.linalg.norm(H_arn - Q_arn.T @ A @ Q_arn, ord="fro"))
    ortho_loss_hou.append(np.linalg.norm(Q_hou.T @ Q_hou - np.eye(n), ord="fro"))
    ortho_loss_arn.append(np.linalg.norm(Q_arn.T @ Q_arn - np.eye(n), ord="fro"))

fig, axs = plt.subplots(2, 2, figsize=(15, 10))

axs[0, 0].bar(range(number), accuracy_hou, color='b', width=0.2)
axs[0, 0].set_title('the frobenius norm of (Q*AQ - H) using Householder reflections')
axs[0, 0].set_xlabel('a few random matrices')
axs[0, 0].set_ylabel('||Q*AQ - H||')


axs[0, 1].bar(range(number), accuracy_arn, color='b', width=0.2)
axs[0, 1].set_title('the frobenius norm of (Q*AQ - H) using Arnoldi process based on MGS')
axs[0, 1].set_xlabel('a few random matrices')
axs[0, 1].set_ylabel('||Q*AQ - H||')



axs[1, 0].bar(range(number), ortho_loss_hou, color='b', width=0.2)
axs[1, 0].set_title('the orthogonality loss of Q using Householder reflections')
axs[1, 0].set_xlabel('a few random matrices')
axs[1, 0].set_ylabel('||Q*Q - I||')


axs[1, 1].bar(range(number), ortho_loss_arn, color='b', width=0.2)
axs[1, 1].set_title('the orthogonality loss of Q using Arnoldi process based on MGS')
axs[1, 1].set_xlabel('a few random matrices')
axs[1, 1].set_ylabel('||Q*Q - I||')



fig.suptitle(f'a few random {n}x{n} matrices', fontsize=14)

plt.tight_layout()

plt.show()