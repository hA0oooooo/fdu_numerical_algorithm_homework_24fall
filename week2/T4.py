import numpy as np
import time
import matplotlib.pyplot as plt


def partial_pivoting_gaussion_elimination(A):
    copy_A = np.copy(A)
    n = copy_A.shape[0]

    for i in range(n):
        max_row_i =  np.argmax(np.abs(A[i:, i])) + i
        if max_row_i != i:
            A[[i, max_row_i], :] = A[[max_row_i, i], :]

        for j in range(i+1, n):
            factor = A[j, i] / A[i, i]
            A[j, i:] = A[j, i:] - factor * A[i, i]
    
    return copy_A


def complete_pivoting_gaussion_elimination(A):
    copy_A = np.copy(A)
    n = copy_A.shape[0]

    for i in range(n):

        max_row_i = np.argmax(np.abs(A[i:, i])) + i
        if max_row_i != i:
            A[[i, max_row_i], :] = A[[max_row_i, i], :]

        max_col_i = np.argmax(np.abs(A[i, i:])) + i
        if max_col_i != i:
            A[:, [max_col_i, i]] = A[:, [i, max_col_i]]

        for j in range(i+1, n):
            factor = A[j, i] / A[i, i]
            A[j, i:] = A[j, i:] - factor * A[i, i:]
    
    return copy_A


def generate_test_matrix(n):
    A = np.random.randn(n,n)
    return A


def measure_execution_time(n, strategy):
    A = generate_test_matrix(n)
    start_time = time.time()
    strategy(A)
    end_time = time.time()
    return end_time - start_time
    

def plot_log_log(strategy):
    dimention = np.arange(1000, 2001, 10)
    times = []

    for n in dimention:
        exec_time = measure_execution_time(n, strategy)
        times.append(exec_time)

    log_dimentions = np.log10(dimention)
    log_times = np.log10(times)

    plt.figure(figsize=(8, 6))
    plt.plot(log_dimentions, log_times, marker='o', linestyle='-')
    plt.xlabel('log(matrix dimension)', fontsize=12)
    plt.ylabel('log(execution time)', fontsize=12)
    plt.title(f'Log(matrix dimension)-log(execution time) for {strategy.__name__}')
    plt.grid(True, which='both', ls='--')
    plt.show(block=False)


plot_log_log(partial_pivoting_gaussion_elimination)
plot_log_log(complete_pivoting_gaussion_elimination)
plt.show()