import matplotlib.pyplot as plt
import numpy as np

def Arnoldi_precess(A, b, k, modified, reortho):
    """
    Arnoldi process for matrix A and vector v, using CGS/CGS2/MGS/MGS2 when orthogonalization
    A[q_1, q_2, ... ,q_k] = [q_1, q_2, ..., q_(k+1)] H
    :param A: matrix A, n x n 
    :param b: vector b, iterative initial vector
    :param k: number of iterations
    :param modified: use CGS/CGS2 if modified is not True else use MGS/MGS2
    :param reortho: use CGS/MGS if reortho is not True else use CGS2/MGS2 
    :return: Q (n x (k+1) orthogonal matrix), H ((k+1) x k upper hessenberg matrix) 
    """
    n = len(b)  
    n1, n2 = A.shape
    if n1 != n or n2 != n:
        print(f'input matrix A and vector b have different size, A: {n1} x {n2}, b: {n} x 1')
        return None, None
    
    Q = np.zeros((n, (k+1)))
    H = np.zeros(((k+1), k))
    Q[:, [0]] = b / np.linalg.norm(b, ord=2)

    if modified is not True:
    # Use BLAS2 may be faster, but here use BLAS1 for simplicity
        for i in range(k):
            cur = A @ Q[:, [i]]
            for j in range(i+1):
                H[i, j] = np.dot(cur.T, Q[:, [j]]).item()
            for j in range(i+1):
                cur = cur - H[i, j] * Q[:, [j]]
            if reortho is True:
                correct = [0] * (i+1)
                for j in range(i+1):
                    correct[j] = np.dot(cur.T, Q[:, [j]]).item()
                    H[i, j] = H[i, j] + correct[j]
                for j in range(i+1):
                    cur = cur - correct[j] * Q[:, [j]]
            H[i+1, i] = np.linalg.norm(cur, ord=2)

            if H[i+1, i] == 0:
                print(f'cannot continue iteration when generating q_{i+1}, H[{i+1}, {i}] = 0')
                return Q, H
            
            Q[:, [i+1]] = cur / H[i+1, i]
    
    if modified is True:
        for i in range(k):
            cur = A @ Q[:, [i]]
            for j in range(i+1):
                H[i, j] = np.dot(cur.T, Q[:, [j]]).item()
                cur = cur - H[i, j] * Q[:, [j]]
                if reortho is True:
                    correct = np.dot(cur.T, Q[:, [j]]).item()
                    H[i, j] = H[i, j] + correct
                    cur = cur - correct * Q[:, [j]]
            H[i+1, i] = np.linalg.norm(cur, ord=2)

            if H[i+1, i] == 0:
                print(f'cannot continue iteration when generating q_{i+1}, H[{i+1}, {i}] = 0')
                return Q, H
            
            Q[:, [i+1]] = cur / H[i+1, i]

    return Q, H

def generate_random_matrix(n):
    """
    Generate a random matrix A with size n x n
    :param n: size of the matrix
    :return: random matrix A
    """
    A = np.random.rand(n, n)
    return A

def generate_random_vector(n):
    """
    Generate a random vector b with size n x 1
    :param n: size of the vector
    :return: random vector b
    """
    b = np.random.rand(n, 1)
    return b

n = 1000
col_num_Q = 30
times = 10

iteration_time = col_num_Q - 1
A = generate_random_matrix(n)
b = generate_random_vector(n)

average_loss = [0, 0, 0, 0]

for i in range(times):
    Q4, H4 = Arnoldi_precess(A, b, iteration_time, modified=True, reortho=True) # MGS2
    Q2, H2 = Arnoldi_precess(A, b, iteration_time, modified=True, reortho=False) # MGS
    Q3, H3 = Arnoldi_precess(A, b, iteration_time, modified=False, reortho=True) # CGS2
    Q1, H1 = Arnoldi_precess(A, b, iteration_time, modified=False, reortho=False) # CGS
    y = [np.linalg.norm(np.eye(col_num_Q) - Q1.T @ Q1, ord='fro'), np.linalg.norm(np.eye(col_num_Q) - Q2.T @ Q2, ord='fro'), np.linalg.norm(np.eye(col_num_Q) - Q3.T @ Q3, ord='fro'), np.linalg.norm(np.eye(col_num_Q) - Q4.T @ Q4, ord='fro')]
    average_loss = [average_loss[i] + y[i] for i in range(4)]

average_loss = [average_loss[i] / times for i in range(4)]

plt.figure(figsize=(8, 6))
plt.title(f'Orthogonality Loss of Arnoldi process \n when generating {col_num_Q}-dimensional Krylov subspace with {n} × {n} matrix')
x = ['CGS'  , 'MGS'  , 'CGS2' , 'MGS2']
plt.bar(x, average_loss, color='skyblue', width=0.4)
plt.xlabel('Orthogonalization method')
plt.ylabel('Orthogonality loss, Frobenies norm of |Q*Q-I|')
plt.show()